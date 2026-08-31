import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from skimage.draw import line
from pathlib import Path


def get_observer_point(building_geom, offset_m: float = 0.5):
    from shapely.geometry import Point
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


def bresenham_check(loc_dsm, loc_dem, loc_tree_id, lr0, lc0, z_observer, tr, tc, dist_m, res, trunk_clear_height=2.5):
    """逐点画线版本的判断逻辑, 跟viewshed_3d_adaptive.py完全一致, 用作ground truth对照。"""
    z_target = loc_dsm[tr, tc]
    rr, cc = line(lr0, lc0, tr, tc)
    if len(rr) <= 2:
        return True, []

    rr, cc = rr[1:-1], cc[1:-1]
    dists_along = np.sqrt((rr - lr0) ** 2 + (cc - lc0) ** 2) * res
    z_ray = z_observer + (z_target - z_observer) * (dists_along / dist_m)

    ray_dsm = loc_dsm[rr, cc]
    ray_dem = loc_dem[rr, cc]
    ray_tid = loc_tree_id[rr, cc]

    trace = []
    for i in range(len(z_ray)):
        zr, dsm_val, dem_val, tid = z_ray[i], ray_dsm[i], ray_dem[i], ray_tid[i]
        blocked_here = False
        if tid > 0:
            if (dem_val + trunk_clear_height) <= zr <= dsm_val:
                blocked_here = True
        else:
            if zr <= dsm_val:
                blocked_here = True
        trace.append((rr[i], cc[i], round(float(zr), 2), round(float(dsm_val), 2), blocked_here))
        if blocked_here:
            return False, trace
    return True, trace


def sweep_check_single_target(loc_dsm, lr0, lc0, z_observer, res, target_r, target_c, viewshed_range_m):
    """
    对着sweep_from_tree里的逻辑, 但反过来: 以observer为origin, 只关心一个target点(某棵树),
    打印这个target所在bin里, 排在它前面的所有像素的角度信息。
    """
    n_rows, n_cols = loc_dsm.shape
    rr, cc = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")

    dr = rr - lr0
    dc = cc - lc0
    dist_m_grid = np.sqrt(dr ** 2 + dc ** 2) * res
    in_range = (dist_m_grid > 0) & (dist_m_grid <= viewshed_range_m)

    elevation_angle = np.arctan2(loc_dsm - z_observer, np.maximum(dist_m_grid, 1e-9))

    radius_px = viewshed_range_m / res
    n_bins = max(360, int(2 * np.pi * radius_px))
    angle = np.arctan2(dc, dr)
    bin_idx = np.floor((angle + np.pi) / (2 * np.pi) * n_bins).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    target_bin = bin_idx[target_r, target_c]
    target_dist = dist_m_grid[target_r, target_c]
    target_elev = elevation_angle[target_r, target_c]

    valid_r = rr[in_range]
    valid_c = cc[in_range]
    valid_dist = dist_m_grid[in_range]
    valid_elev = elevation_angle[in_range]
    valid_bin = bin_idx[in_range]

    # 只挑出跟target同一个bin、且距离比target更近的像素(sweep里会先于target被处理的)
    same_bin_mask = (valid_bin == target_bin) & (valid_dist < target_dist)
    closer_r = valid_r[same_bin_mask]
    closer_c = valid_c[same_bin_mask]
    closer_dist = valid_dist[same_bin_mask]
    closer_elev = valid_elev[same_bin_mask]

    order = np.argsort(closer_dist)

    print(f"    target所在bin: {target_bin}, target距离: {target_dist:.2f}m, target仰角: {target_elev:.4f}")
    print(f"    同一bin内、比target更近的像素数: {len(order)}")

    max_angle_so_far = -np.inf
    blocked_by = None
    for i in order[:30]:  # 只打印前30个, 避免刷屏
        r_, c_, d_, e_ = closer_r[i], closer_c[i], closer_dist[i], closer_elev[i]
        is_new_max = e_ > max_angle_so_far
        print(f"      -> 像素({r_},{c_}) 距离={d_:.2f}m 仰角={e_:.4f} DSM高={loc_dsm[r_,c_]:.2f} {'<-刷新max' if is_new_max else ''}")
        if is_new_max:
            max_angle_so_far = e_
            if blocked_by is None or True:
                blocked_by = (r_, c_, e_)

    is_visible = target_elev > max_angle_so_far
    print(f"    最终 max_angle_so_far = {max_angle_so_far:.4f}, target仰角 = {target_elev:.4f}")
    print(f"    sweep判定: {'可见' if is_visible else '被挡住'}")
    return is_visible


def diagnose(
        dsm_path: Path,
        dem_path: Path,
        tree_crown_shp: Path,
        building_shp: Path,
        target_building_o: int,
        floor_eye_offset: float = 1.5,
        wall_offset_m: float = 0.5,
        viewshed_range_m: float = 100.0,
        tree_id_field: str = "Tree_ID",
        max_trees_to_check: int = 15,
):
    with rasterio.open(dsm_path) as src_dsm, rasterio.open(dem_path) as src_dem:
        transform = src_dsm.transform
        res = transform.a
        dsm_array = src_dsm.read(1)
        dem_array = src_dem.read(1)

    trees = gpd.read_file(tree_crown_shp)
    buildings = gpd.read_file(building_shp)

    tree_shapes = ((geom, tid) for geom, tid in zip(trees.geometry, trees[tree_id_field]))
    tree_id_mask = rasterize(tree_shapes, out_shape=dsm_array.shape, transform=transform, fill=0, dtype=np.int32)

    target_row = buildings[buildings["building_o"] == target_building_o]
    if len(target_row) == 0:
        raise ValueError(f"找不到 building_o={target_building_o}")
    building_geom = target_row.iloc[0].geometry

    obs_pt = get_observer_point(building_geom, offset_m=wall_offset_m)
    c0, r0 = ~transform * (obs_pt.x, obs_pt.y)
    c0, r0 = int(c0), int(r0)
    base_elev = dem_array[r0, c0]
    z_observer = base_elev + floor_eye_offset

    print(f"=== 诊断 building_o={target_building_o}, floor1, observer=({r0},{c0}), z_observer={z_observer:.2f} ===\n")

    pixel_range = int(viewshed_range_m / res)
    r_min, r_max = max(0, r0 - pixel_range), min(dsm_array.shape[0], r0 + pixel_range + 1)
    c_min, c_max = max(0, c0 - pixel_range), min(dsm_array.shape[1], c0 + pixel_range + 1)
    lr0, lc0 = r0 - r_min, c0 - c_min

    loc_dsm = dsm_array[r_min:r_max, c_min:c_max]
    loc_dem = dem_array[r_min:r_max, c_min:c_max]
    loc_tree_id = tree_id_mask[r_min:r_max, c_min:c_max]

    tree_r_all, tree_c_all = np.where(loc_tree_id > 0)
    dists = np.sqrt((tree_r_all - lr0) ** 2 + (tree_c_all - lc0) ** 2) * res
    order = np.argsort(dists)

    checked_tree_ids = set()
    n_checked = 0
    n_mismatch = 0

    for idx in order:
        if n_checked >= max_trees_to_check:
            break
        tr, tc_ = tree_r_all[idx], tree_c_all[idx]
        tid = loc_tree_id[tr, tc_]
        if tid in checked_tree_ids:
            continue
        checked_tree_ids.add(tid)
        n_checked += 1
        dist_m = dists[idx]

        bres_visible, trace = bresenham_check(loc_dsm, loc_dem, loc_tree_id, lr0, lc0, z_observer, tr, tc_, dist_m, res)

        print(f"--- 树 Tree_ID={tid} 像素({tr},{tc_}) 距离={dist_m:.2f}m ---")
        print(f"  逐点画线判定: {'可见' if bres_visible else '被挡住'}")

        sweep_visible = sweep_check_single_target(loc_dsm, lr0, lc0, z_observer, res, tr, tc_, viewshed_range_m)

        if bres_visible != sweep_visible:
            n_mismatch += 1
            print(f"  !!! 两种方法判定不一致 !!!")
        print()

    print(f"\n=== 总结: 检查了{n_checked}棵树, {n_mismatch}棵判定不一致 ===")


if __name__ == "__main__":
    diagnose(
        dsm_path=Path(r"C:\Users\xhe40\Thesis_Data\Campus\test_dsm_Afterfill.tif"),
        dem_path=Path(r"C:\Users\xhe40\Thesis_Data\Campus\test_dem_1m.tif"),
        tree_crown_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\individual_trees.shp"),
        building_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\Building_Campus.shp"),
        target_building_o=6470432,  # 你确认差异最大的那栋矮建筑
        max_trees_to_check=15,
    )