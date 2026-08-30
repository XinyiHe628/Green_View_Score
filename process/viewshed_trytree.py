import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import rasterize, geometry_mask
from skimage.draw import line
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import Point


def get_observer_point(building_geom, offset_m: float = 0.5):
    centroid = building_geom.centroid
    boundary = building_geom.exterior
    nearest_pt = boundary.interpolate(boundary.project(centroid))

    dx = nearest_pt.x - centroid.x
    dy = nearest_pt.y - centroid.y
    dist = np.sqrt(dx ** 2 + dy ** 2)

    if dist == 0:
        return centroid

    ux, uy = dx / dist, dy / dist
    obs_x = nearest_pt.x + ux * offset_m
    obs_y = nearest_pt.y + uy * offset_m

    return Point(obs_x, obs_y)


def estimate_building_floors(building_geom, ndsm_array, transform, floor_height_m: float = 3.0,
                              percentile: float = 90):
    """
    用building footprint范围内的nDSM算这栋楼的高度(取percentile,不用max/mean),
    再换算成层数(向下取整,至少1层)。
    """
    # 用building的几何范围, 在nDSM上生成一个mask, 标出footprint覆盖的像素
    mask = geometry_mask(
        [building_geom],
        out_shape=ndsm_array.shape,
        transform=transform,
        invert=True  # True = footprint内的像素标记为True
    )

    footprint_ndsm_values = ndsm_array[mask]
    # 排除掉nodata/异常值(负数或者极端值), 只保留合理范围
    footprint_ndsm_values = footprint_ndsm_values[
        (footprint_ndsm_values > 0) & (footprint_ndsm_values < 200)
        ]

    if len(footprint_ndsm_values) == 0:
        return 0.0, 1  # 没有有效数据, 保守假设1层

    building_height = np.percentile(footprint_ndsm_values, percentile)
    num_floors = max(1, int(building_height // floor_height_m))

    return building_height, num_floors


def calculate_3d_green_view_score(
        dsm_path: Path,
        dem_path: Path,
        tree_crown_shp: Path,
        building_shp: Path,
        output_shp: Path,
        viewshed_range_m: float = 100.0,
        floor_height_m: float = 3.0,       # 每层假设高度
        floor_eye_offset: float = 1.5,      # 每层观察点距地板的眼高偏移
        trunk_clear_height: float = 2.5,
        wall_offset_m: float = 0.5,
        height_percentile: float = 90
):
    print("1. 正在加载高程矩阵并计算 nDSM...")
    with rasterio.open(dsm_path) as src_dsm, rasterio.open(dem_path) as src_dem:
        transform = src_dsm.transform
        res = transform.a

        dsm_array = src_dsm.read(1)
        dem_array = src_dem.read(1)
        ndsm_array = np.clip(dsm_array - dem_array, 0, None)  # 现在有用了

        print("2. 正在加载矢量要素并生成带ID的树木mask...")
        trees = gpd.read_file(tree_crown_shp)
        buildings = gpd.read_file(building_shp)

        tree_shapes = ((geom, idx + 1) for idx, geom in zip(trees.index, trees.geometry))
        tree_id_mask = rasterize(
            tree_shapes,
            out_shape=dsm_array.shape,
            transform=transform,
            fill=0,
            dtype=np.int32
        )

    print("3. 正在估算每栋建筑的层数...")
    buildings["bldg_h_m"] = 0.0
    buildings["num_floors"] = 1
    for idx, row in tqdm(buildings.iterrows(), total=len(buildings), desc="Estimating floors"):
        h, n = estimate_building_floors(row.geometry, ndsm_array, transform,
                                         floor_height_m=floor_height_m,
                                         percentile=height_percentile)
        buildings.loc[idx, "bldg_h_m"] = h
        buildings.loc[idx, "num_floors"] = n

    max_floors_in_data = int(buildings["num_floors"].max())
    # 动态生成楼层高度列表, 比如每层3米, 眼高在楼板上方1.5米
    # 1楼眼高=1.5, 2楼眼高=3+1.5=4.5, 3楼=6+1.5=7.5, 以此类推
    floor_heights = [floor_eye_offset + i * floor_height_m for i in range(max_floors_in_data)]
    print(f"   数据集中最高建筑有 {max_floors_in_data} 层, 将计算的眼高: {floor_heights}")

    print("4. 开始执行 3D 射线追踪 (Ray-casting)...")
    for i, h in enumerate(floor_heights):
        buildings[f"GVS_F{i+1}"] = np.nan
        buildings[f"VTC_F{i+1}"] = np.nan

    for idx, row in tqdm(buildings.iterrows(), total=len(buildings), desc="Building Progress"):
        this_building_floors = int(row["num_floors"])

        observer_point = get_observer_point(row.geometry, offset_m=wall_offset_m)
        x0, y0 = observer_point.x, observer_point.y

        c0, r0 = ~transform * (x0, y0)
        c0, r0 = int(c0), int(r0)

        if not (0 <= r0 < dsm_array.shape[0] and 0 <= c0 < dsm_array.shape[1]):
            continue

        base_elevation = dem_array[r0, c0]
        if np.isnan(base_elevation) or base_elevation == -9999.0:
            continue

        pixel_range = int(viewshed_range_m / res)
        r_min, r_max = max(0, r0 - pixel_range), min(dsm_array.shape[0], r0 + pixel_range + 1)
        c_min, c_max = max(0, c0 - pixel_range), min(dsm_array.shape[1], c0 + pixel_range + 1)

        lr0, lc0 = r0 - r_min, c0 - c_min

        loc_dsm = dsm_array[r_min:r_max, c_min:c_max]
        loc_dem = dem_array[r_min:r_max, c_min:c_max]
        loc_tree_id = tree_id_mask[r_min:r_max, c_min:c_max]

        tree_pixel_coords = np.argwhere(loc_tree_id > 0)

        # ===== 关键改动: 只算这栋楼真实存在的楼层, 超出的留NaN =====
        for floor_idx in range(this_building_floors):
            h = floor_heights[floor_idx]
            z_observer = base_elevation + h
            score = 0.0
            visible_tree_ids = set()

            for tr, tc in tree_pixel_coords:
                current_tree_id = loc_tree_id[tr, tc]

                dist_pixels = np.sqrt((tr - lr0) ** 2 + (tc - lc0) ** 2)
                dist_m = dist_pixels * res

                if dist_m > viewshed_range_m or dist_m == 0:
                    continue

                z_target = loc_dsm[tr, tc]
                rr, cc = line(lr0, lc0, tr, tc)

                if len(rr) > 2:
                    rr, cc = rr[1:-1], cc[1:-1]
                    dists_along_ray = np.sqrt((rr - lr0) ** 2 + (cc - lc0) ** 2) * res
                    z_ray = z_observer + (z_target - z_observer) * (dists_along_ray / dist_m)

                    ray_dsm = loc_dsm[rr, cc]
                    ray_dem = loc_dem[rr, cc]
                    ray_tree_id = loc_tree_id[rr, cc]

                    blocked = False
                    for i in range(len(z_ray)):
                        zr = z_ray[i]
                        dsm_val = ray_dsm[i]
                        dem_val = ray_dem[i]

                        if ray_tree_id[i] > 0:
                            z_trunk_top = dem_val + trunk_clear_height
                            if z_trunk_top <= zr <= dsm_val:
                                blocked = True
                                break
                        else:
                            if zr <= dsm_val:
                                blocked = True
                                break

                    if blocked:
                        continue

                score += (1.0 / (dist_m ** 2))
                visible_tree_ids.add(current_tree_id)

            buildings.loc[idx, f"GVS_F{floor_idx+1}"] = score
            buildings.loc[idx, f"VTC_F{floor_idx+1}"] = len(visible_tree_ids)
            # 超出this_building_floors的楼层, 保持初始化时的NaN, 不做任何计算

    print(f"5. 正在导出分析结果至: {output_shp}")
    buildings.to_file(output_shp)
    print("🎉 完成！")


if __name__ == "__main__":
    calculate_3d_green_view_score(
        dsm_path=Path(r"C:\Users\xhe40\Thesis_Data\Campus\test_dsm_Afterfill.tif"),
        dem_path=Path(r"C:\Users\xhe40\Thesis_Data\Campus\test_dem_1m.tif"),
        tree_crown_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\individual_trees.shp"),
        building_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\Building_Campus.shp"),
        output_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\building_result_3D_v4.shp"),
        viewshed_range_m=100.0,
        floor_height_m=3.0,
        floor_eye_offset=1.5,
        trunk_clear_height=2.5,
        wall_offset_m=0.5,
        height_percentile=90
    )