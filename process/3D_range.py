"""
3D Reverse Viewshed (Bresenham射线追踪版本) —— 沿墙多点版

跟原版的核心区别:
  原版每栋building每层只有1个observer点。
  这版先用buffer法简化building轮廓(buffer_m=5), 再沿简化后的墙面每隔interval_m米
  生成一个点(每栋楼每层因此有好几个observer), 最终每栋楼每层输出GVS的min/max/mean。
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize, geometry_mask
from skimage.draw import line
from scipy.spatial import cKDTree
from shapely.geometry import Point
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


# ---------------------------------------------------------------------------
# 1. Building简化(buffer法) + 沿墙撒点
# ---------------------------------------------------------------------------

def simplify_building_buffer(building_geom, buffer_m: float = 5.0):
    geom = building_geom
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)

    smoothed = geom.buffer(buffer_m, join_style=2).buffer(-buffer_m, join_style=2)

    if smoothed.is_empty:
        return geom
    if smoothed.geom_type == "MultiPolygon":
        smoothed = max(smoothed.geoms, key=lambda g: g.area)

    return smoothed


def get_observer_points_along_wall(
        building_geom,
        buffer_m: float = 5.0,
        interval_m: float = 5.0,
        wall_offset_m: float = 0.5,
):
    simplified = simplify_building_buffer(building_geom, buffer_m=buffer_m)
    boundary = simplified.exterior
    length = boundary.length
    centroid = simplified.centroid

    if length == 0:
        return [centroid]

    n_steps = max(1, int(length // interval_m))
    eps = min(0.5, interval_m / 10.0)

    points = []
    skipped = 0
    for i in range(n_steps):
        d = (i * interval_m) % length
        p = boundary.interpolate(d)

        d_next = (d + eps) % length
        d_prev = (d - eps) % length
        p_next = boundary.interpolate(d_next)
        p_prev = boundary.interpolate(d_prev)

        tangent = np.array([p_next.x - p_prev.x, p_next.y - p_prev.y])
        norm = np.linalg.norm(tangent)
        if norm == 0:
            continue
        tangent = tangent / norm

        normal = np.array([-tangent[1], tangent[0]])
        to_p = np.array([p.x - centroid.x, p.y - centroid.y])
        if np.dot(normal, to_p) < 0:
            normal = -normal

        offset_pt = Point(
            p.x + normal[0] * wall_offset_m,
            p.y + normal[1] * wall_offset_m,
        )

        # ===== 新增: 合理性检查 =====
        # 如果算出来的点掉进了building轮廓内部(说明法线方向在这个转角附近算错了),
        # 先尝试翻转方向重算一次；还是在内部的话，直接跳过这个点，不勉强放一个错误位置。
        if simplified.contains(offset_pt):
            normal_flipped = -normal
            offset_pt_flipped = Point(
                p.x + normal_flipped[0] * wall_offset_m,
                p.y + normal_flipped[1] * wall_offset_m,
            )
            if not simplified.contains(offset_pt_flipped):
                offset_pt = offset_pt_flipped
            else:
                skipped += 1
                continue

        points.append(offset_pt)

    if skipped > 0:
        print(f"   [提示] 有 {skipped} 个转角附近的点因位置异常被跳过")

    if not points:
        # 极端情况下全部被跳过，兜底返回centroid，避免这栋building完全没有观察点
        return [centroid]

    return points

# ---------------------------------------------------------------------------
# 2. 楼层估算 (跟原版一样)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 3. 主流程
# ---------------------------------------------------------------------------

def calculate_green_view_tree_centric(
        dsm_path: Path,
        dem_path: Path,
        tree_crown_shp: Path,
        building_shp: Path,
        output_shp: Path,
        output_observer_points_shp: Path = None,
        viewshed_range_m: float = 100.0,
        floor_height_m: float = 4.0,
        floor_eye_offset: float = 1.5,
        trunk_clear_height: float = 2.5,
        wall_offset_m: float = 0.5,
        height_percentile: float = 90,
        tree_id_field: str = "Tree_ID",
        buffer_m: float = 5.0,
        interval_m: float = 5.0,
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
    crs = buildings.crs

    if tree_id_field not in trees.columns:
        raise ValueError(f"树木数据里没有 {tree_id_field} 字段, 检查一下individual_trees.shp的列名")

    tree_shapes = ((geom, tid) for geom, tid in zip(trees.geometry, trees[tree_id_field]))
    tree_id_mask = rasterize(tree_shapes, out_shape=dsm_array.shape, transform=transform,
                              fill=0, dtype=np.int32)

    print("3. 简化building轮廓 + 沿墙撒点 + 估算层数...")
    observer_records = []
    buildings["bldg_h_m"] = 0.0
    buildings["num_floors"] = 1
    buildings["n_wall_pts"] = 0

    building_id_field = "building_o" if "building_o" in buildings.columns else None

    for b_idx, row in tqdm(buildings.iterrows(), total=len(buildings), desc="Building setup"):
        h, n_floors = estimate_building_floors(row.geometry, ndsm_array, transform,
                                                 floor_height_m=floor_height_m,
                                                 percentile=height_percentile)
        buildings.loc[b_idx, "bldg_h_m"] = h
        buildings.loc[b_idx, "num_floors"] = n_floors

        wall_points = get_observer_points_along_wall(
            row.geometry, buffer_m=buffer_m, interval_m=interval_m, wall_offset_m=wall_offset_m
        )
        buildings.loc[b_idx, "n_wall_pts"] = len(wall_points)

        for wp_idx, obs_pt in enumerate(wall_points):
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
                    "building_idx": b_idx,
                    "building_o": row[building_id_field] if building_id_field else b_idx,
                    "wall_pt_idx": wp_idx,
                    "floor_idx": floor_idx,
                    "row": r0, "col": c0, "x": obs_pt.x, "y": obs_pt.y,
                    "z_observer": base_elev + eye_h,
                })

    if not observer_records:
        raise ValueError("没有生成任何有效observer点, 检查building数据和DEM是否对齐/同一坐标系")

    # ===== 新增: 跑之前先给出规模提示, 心里有数再等 =====
    n_buildings = len(buildings)
    n_obs = len(observer_records)
    n_trees = len(trees)
    print(f"\n   >>> 规模预估 <<<")
    print(f"   building数: {n_buildings}, 平均每栋 {n_obs/n_buildings:.1f} 个observer点")
    print(f"   observer点总数: {n_obs} (原来单点版本大约是 {n_buildings} 个, 现在是它的 {n_obs/n_buildings:.1f} 倍)")
    print(f"   树木数: {n_trees}")
    print(f"   Bresenham遮挡判断量级 ≈ observer点数 × 附近树木数, 请留意运行时间\n")

    obs_xy = np.array([[r["x"], r["y"]] for r in observer_records])
    obs_kdtree = cKDTree(obs_xy)

    n_obs = len(observer_records)
    scores = np.zeros(n_obs, dtype=np.float64)
    visible_sets = [set() for _ in range(n_obs)]

    print("4. 从每棵树出发, 反向发射视线判断遮挡 (Reverse Viewshed)...")
    for t_idx, t_row in tqdm(trees.iterrows(), total=len(trees), desc="Tree Progress"):
        tree_id = t_row[tree_id_field]
        geom = t_row.geometry

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

    print("5. 汇总结果到building属性表 (每栋楼每层, 沿墙多点取min/max/mean)...")
    max_floors = int(buildings["num_floors"].max())
    for i in range(max_floors):
        buildings[f"GVS_F{i+1}_min"] = np.nan
        buildings[f"GVS_F{i+1}_max"] = np.nan
        buildings[f"GVS_F{i+1}_mean"] = np.nan
        buildings[f"VTC_F{i+1}_min"] = np.nan
        buildings[f"VTC_F{i+1}_max"] = np.nan
        buildings[f"VTC_F{i+1}_mean"] = np.nan

    groups = defaultdict(list)
    for obs_i, rec in enumerate(observer_records):
        key = (rec["building_idx"], rec["floor_idx"])
        groups[key].append(obs_i)

    for (b_idx, floor_idx), obs_indices in groups.items():
        gvs_vals = [scores[i] for i in obs_indices]
        vtc_vals = [len(visible_sets[i]) for i in obs_indices]

        buildings.loc[b_idx, f"GVS_F{floor_idx+1}_min"] = min(gvs_vals)
        buildings.loc[b_idx, f"GVS_F{floor_idx+1}_max"] = max(gvs_vals)
        buildings.loc[b_idx, f"GVS_F{floor_idx+1}_mean"] = sum(gvs_vals) / len(gvs_vals)
        buildings.loc[b_idx, f"VTC_F{floor_idx+1}_min"] = min(vtc_vals)
        buildings.loc[b_idx, f"VTC_F{floor_idx+1}_max"] = max(vtc_vals)
        buildings.loc[b_idx, f"VTC_F{floor_idx+1}_mean"] = sum(vtc_vals) / len(vtc_vals)

    output_shp.parent.mkdir(parents=True, exist_ok=True)
    buildings.to_file(output_shp)
    print(f"🎉 完成! 写入 {output_shp}")

    if output_observer_points_shp is not None:
        print("6. 生成observer点shapefile, 供fieldwork和QC使用...")
        point_records = []
        for obs_i, rec in enumerate(observer_records):
            point_records.append({
                "building_o": rec["building_o"],
                "wall_pt": rec["wall_pt_idx"],
                "floor": rec["floor_idx"] + 1,
                "z_observ_m": round(rec["z_observer"], 2),
                "VTC": len(visible_sets[obs_i]),
                "GVS": round(scores[obs_i], 3),
                "geometry": Point(rec["x"], rec["y"]),
            })

        observer_points_gdf = gpd.GeoDataFrame(point_records, crs=crs)
        output_observer_points_shp.parent.mkdir(parents=True, exist_ok=True)
        observer_points_gdf.to_file(output_observer_points_shp)
        print(f"🎉 observer点shapefile已写入: {output_observer_points_shp}")
        print(f"   共 {len(observer_points_gdf)} 个观察点(building x floor x wall_point)")


if __name__ == "__main__":
    calculate_green_view_tree_centric(
        dsm_path=Path(r"C:\Users\xhe40\Thesis_Data\Campus\test_dsm_Afterfill.tif"),
        dem_path=Path(r"C:\Users\xhe40\Thesis_Data\Campus\test_dem_1m.tif"),
        tree_crown_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\individual_trees_high.shp"),
        building_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\Building_Campus_with_floors_4.shp"),
        output_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\building_result_3D_wallpoints_plusangle_high.shp"),
        output_observer_points_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\observer_points_wallpoints_plusangle_high.shp"),
        floor_height_m=4.0,
        buffer_m=5.0,
        interval_m=5.0,
    )