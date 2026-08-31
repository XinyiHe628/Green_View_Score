import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize, geometry_mask
from skimage.draw import line
from scipy.spatial import cKDTree
from shapely.geometry import Point
from pathlib import Path
from tqdm import tqdm


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
    return Point(nearest_pt.x + ux * offset_m, nearest_pt.y + uy * offset_m)


def estimate_building_floors(building_geom, ndsm_array, transform,
                              floor_height_m: float = 3.0, percentile: float = 90):
    mask = geometry_mask([building_geom], out_shape=ndsm_array.shape,
                          transform=transform, invert=True)
    vals = ndsm_array[mask]
    vals = vals[(vals > 0) & (vals < 200)]
    if len(vals) == 0:
        return 0.0, 1
    h = np.percentile(vals, percentile)
    return h, max(1, int(h // floor_height_m))


def calculate_green_view_tree_centric(
        dsm_path: Path,
        dem_path: Path,
        tree_crown_shp: Path,
        building_shp: Path,
        output_shp: Path,
        viewshed_range_m: float = 100.0,
        floor_height_m: float = 3.0,
        floor_eye_offset: float = 1.5,
        trunk_clear_height: float = 2.5,
        wall_offset_m: float = 0.5,
        height_percentile: float = 90,
        tree_id_field: str = "Tree_ID",
):
    print("1. 加载DSM/DEM, 计算nDSM...")
    with rasterio.open(dsm_path) as src_dsm, rasterio.open(dem_path) as src_dem:
        transform = src_dsm.transform
        res = transform.a
        dsm_array = src_dsm.read(1)
        dem_array = src_dem.read(1)
        ndsm_array = np.clip(dsm_array - dem_array, 0, None)

    print("2. 加载building和树木矢量...")
    trees = gpd.read_file(tree_crown_shp)
    buildings = gpd.read_file(building_shp)

    if tree_id_field not in trees.columns:
        raise ValueError(f"树木数据里没有 {tree_id_field} 字段, 检查一下individual_trees.shp的列名")

    # 用真实Tree_ID烧录树冠栅格, 不再用行号代替
    tree_shapes = ((geom, tid) for geom, tid in zip(trees.geometry, trees[tree_id_field]))
    tree_id_mask = rasterize(tree_shapes, out_shape=dsm_array.shape, transform=transform,
                              fill=0, dtype=np.int32)

    print("3. 估算每栋建筑层数, 生成所有observer点(building x floor)...")
    observer_records = []  # 每条记录代表一个"某栋楼第几层"的观察点
    buildings["bldg_h_m"] = 0.0
    buildings["num_floors"] = 1

    for b_idx, row in tqdm(buildings.iterrows(), total=len(buildings), desc="Building setup"):
        h, n_floors = estimate_building_floors(row.geometry, ndsm_array, transform,
                                                 floor_height_m=floor_height_m,
                                                 percentile=height_percentile)
        buildings.loc[b_idx, "bldg_h_m"] = h
        buildings.loc[b_idx, "num_floors"] = n_floors

        obs_pt = get_observer_point(row.geometry, offset_m=wall_offset_m)
        c0, r0 = ~transform * (obs_pt.x, obs_pt.y)
        c0, r0 = int(c0), int(r0)
        if not (0 <= r0 < dsm_array.shape[0] and 0 <= c0 < dsm_array.shape[1]):
            continue
        base_elev = dem_array[r0, c0]
        if np.isnan(base_elev) or base_elev == -9999.0:
            continue

        for floor_idx in range(n_floors):
            eye_h = floor_eye_offset + floor_idx * floor_height_m
            observer_records.append({
                "building_idx": b_idx, "floor_idx": floor_idx,
                "row": r0, "col": c0, "x": obs_pt.x, "y": obs_pt.y,
                "z_observer": base_elev + eye_h,
            })

    if not observer_records:
        raise ValueError("没有生成任何有效observer点, 检查building数据和DEM是否对齐/同一坐标系")

    # 建KDTree, 方便后面按每棵树的位置快速查"附近有哪些observer点", 不用每棵树都扫一遍全部observer
    obs_xy = np.array([[r["x"], r["y"]] for r in observer_records])
    obs_kdtree = cKDTree(obs_xy)

    n_obs = len(observer_records)
    scores = np.zeros(n_obs, dtype=np.float64)
    visible_sets = [set() for _ in range(n_obs)]

    print("4. 从每棵树出发, 反向发射视线判断遮挡 (Reverse Viewshed)...")
    for t_idx, t_row in tqdm(trees.iterrows(), total=len(trees), desc="Tree Progress"):
        tree_id = t_row[tree_id_field]
        geom = t_row.geometry

        # 只在这棵树自己的bounding box范围内找它的像素, 不用扫全图
        minx, miny, maxx, maxy = geom.bounds
        col_a, row_a = ~transform * (minx, maxy)
        col_b, row_b = ~transform * (maxx, miny)
        row_min = max(0, int(row_a) - 1)
        row_max = min(dsm_array.shape[0], int(row_b) + 2)
        col_min = max(0, int(col_a) - 1)
        col_max = min(dsm_array.shape[1], int(col_b) + 2)

        local_mask = tree_id_mask[row_min:row_max, col_min:col_max] == tree_id
        tp_rows, tp_cols = np.where(local_mask)
        if len(tp_rows) == 0:
            continue
        tp_rows = tp_rows + row_min
        tp_cols = tp_cols + col_min

        # 查这棵树附近viewshed_range_m内, 有哪些observer点(building-floor)
        tc = geom.centroid
        nearby_idx = obs_kdtree.query_ball_point([tc.x, tc.y], r=viewshed_range_m + 5.0)
        if not nearby_idx:
            continue

        for obs_i in nearby_idx:
            rec = observer_records[obs_i]
            lr0, lc0, z_observer = rec["row"], rec["col"], rec["z_observer"]

            for tr, tcx in zip(tp_rows, tp_cols):
                dist_m = np.sqrt((tr - lr0) ** 2 + (tcx - lc0) ** 2) * res
                if dist_m > viewshed_range_m or dist_m == 0:
                    continue

                z_target = dsm_array[tr, tcx]
                rr, cc = line(lr0, lc0, tr, tcx)

                if len(rr) <= 2:
                    scores[obs_i] += 1.0 / (dist_m ** 2)
                    visible_sets[obs_i].add(tree_id)
                    continue

                rr, cc = rr[1:-1], cc[1:-1]
                dists_along = np.sqrt((rr - lr0) ** 2 + (cc - lc0) ** 2) * res
                z_ray = z_observer + (z_target - z_observer) * (dists_along / dist_m)

                ray_dsm = dsm_array[rr, cc]
                ray_dem = dem_array[rr, cc]
                ray_tid = tree_id_mask[rr, cc]

                blocked = False
                for i in range(len(z_ray)):
                    zr, dsm_val, dem_val = z_ray[i], ray_dsm[i], ray_dem[i]
                    if ray_tid[i] > 0:
                        if (dem_val + trunk_clear_height) <= zr <= dsm_val:
                            blocked = True
                            break
                    else:
                        if zr <= dsm_val:
                            blocked = True
                            break

                if not blocked:
                    scores[obs_i] += 1.0 / (dist_m ** 2)
                    visible_sets[obs_i].add(tree_id)

    print("5. 汇总结果到building属性表...")
    max_floors = int(buildings["num_floors"].max())
    for i in range(max_floors):
        buildings[f"GVS_F{i+1}"] = np.nan
        buildings[f"VTC_F{i+1}"] = np.nan

    for obs_i, rec in enumerate(observer_records):
        b_idx, floor_idx = rec["building_idx"], rec["floor_idx"]
        buildings.loc[b_idx, f"GVS_F{floor_idx+1}"] = scores[obs_i]
        buildings.loc[b_idx, f"VTC_F{floor_idx+1}"] = len(visible_sets[obs_i])

    output_shp.parent.mkdir(parents=True, exist_ok=True)
    buildings.to_file(output_shp)
    print(f"🎉 完成! 写入 {output_shp}")


if __name__ == "__main__":
    calculate_green_view_tree_centric(
        dsm_path=Path(r"C:\Users\xhe40\Thesis_Data\Campus\test_dsm_Afterfill.tif"),
        dem_path=Path(r"C:\Users\xhe40\Thesis_Data\Campus\test_dem_1m.tif"),
        tree_crown_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\individual_trees.shp"),
        building_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\Building_Campus.shp"),
        output_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\building_result_3D_treecentric.shp"),
    )