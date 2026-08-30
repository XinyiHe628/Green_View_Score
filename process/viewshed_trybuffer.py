import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import rasterize
from skimage.draw import line
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import Point


def get_observer_point(building_geom, offset_m: float = 0.5):
    """
    找到building外墙上离centroid最近的点，再沿着centroid->该点的方向
    继续往外推offset_m米，避免observer落在自己的屋顶footprint范围内。
    """
    centroid = building_geom.centroid
    boundary = building_geom.exterior

    # 在边界上找到离centroid最近的点(沿边界插值)
    nearest_pt = boundary.interpolate(boundary.project(centroid))

    # 计算从centroid指向nearest_pt的方向向量，归一化
    dx = nearest_pt.x - centroid.x
    dy = nearest_pt.y - centroid.y
    dist = np.sqrt(dx ** 2 + dy ** 2)

    if dist == 0:
        # 极少数异常geometry, 退化处理: 直接用centroid
        return centroid

    ux, uy = dx / dist, dy / dist

    # 在墙面点的基础上, 沿同方向再往外推 offset_m
    obs_x = nearest_pt.x + ux * offset_m
    obs_y = nearest_pt.y + uy * offset_m

    return Point(obs_x, obs_y)


def calculate_3d_green_view_score(
        dsm_path: Path,
        dem_path: Path,
        tree_crown_shp: Path,
        building_shp: Path,
        output_shp: Path,
        viewshed_range_m: float = 100.0,
        floor_heights: list = [1.5, 4.5, 7.5],
        trunk_clear_height: float = 2.5,
        wall_offset_m: float = 0.5  # 新增: observer往外推的距离
):
    print("1. 正在加载高程矩阵...")
    with rasterio.open(dsm_path) as src_dsm, rasterio.open(dem_path) as src_dem:
        transform = src_dsm.transform
        res = transform.a

        dsm_array = src_dsm.read(1)
        dem_array = src_dem.read(1)

        print("2. 正在加载矢量要素并生成三维植被掩膜...")
        trees = gpd.read_file(tree_crown_shp)
        buildings = gpd.read_file(building_shp)

        tree_shapes = ((geom, 1) for geom in trees.geometry)
        is_tree_mask = rasterize(
            tree_shapes,
            out_shape=dsm_array.shape,
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

    print("3. 开始执行 3D 射线追踪 (Ray-casting) 计算各楼层绿视率...")
    for h in floor_heights:
        buildings[f"GVS_H{h}"] = 0.0

    for idx, row in tqdm(buildings.iterrows(), total=len(buildings), desc="Building Progress"):
        # ===== 关键改动: observer从centroid换成外墙+偏移点 =====
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
        loc_tree = is_tree_mask[r_min:r_max, c_min:c_max]

        tree_pixels = np.argwhere(loc_tree == 1)

        for h in floor_heights:
            z_observer = base_elevation + h
            score = 0.0

            for tr, tc in tree_pixels:
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
                    ray_is_tree = loc_tree[rr, cc]

                    blocked = False
                    for i in range(len(z_ray)):
                        zr = z_ray[i]
                        dsm_val = ray_dsm[i]
                        dem_val = ray_dem[i]

                        if ray_is_tree[i] == 1:
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

            buildings.loc[idx, f"GVS_H{h}"] = score

    print(f"4. 正在导出分析结果至: {output_shp}")
    buildings.to_file(output_shp)
    print("🎉 三维动态楼层绿视率计算完成！")


if __name__ == "__main__":
    calculate_3d_green_view_score(
        dsm_path=Path(r"C:\Users\xhe40\Thesis_Data\Campus\test_dsm_Afterfill.tif"),
        dem_path=Path(r"C:\Users\xhe40\Thesis_Data\Campus\test_dem_1m.tif"),
        tree_crown_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\individual_trees.shp"),
        building_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\Building_Campus.shp"),
        output_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\building_result_3D_v2.shp"),
        viewshed_range_m=100.0,
        floor_heights=[1.5, 4.5, 7.5],
        trunk_clear_height=2.5,
        wall_offset_m=0.5
    )