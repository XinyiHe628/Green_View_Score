import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize, geometry_mask
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


def sweep_from_origin(loc_dsm, origin_r, origin_c, origin_z, res,
                       query_rows, query_cols, query_zs, viewshed_range_m):
    """
    以(origin_r, origin_c, origin_z)为观察起点做sweep。
    loc_dsm里每个像素都是"遮挡物"。
    query点只负责被查询、不参与遮挡刷新。
    返回跟query点等长的bool数组, True=这个query点从origin看过去是可见的。
    """
    n_rows, n_cols = loc_dsm.shape
    rr, cc = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")

    dr = rr - origin_r
    dc = cc - origin_c
    dist_m_grid = np.sqrt(dr ** 2 + dc ** 2) * res
    in_range = (dist_m_grid > 0) & (dist_m_grid <= viewshed_range_m)

    blocker_elev = np.arctan2(loc_dsm - origin_z, np.maximum(dist_m_grid, 1e-9))

    n_bins = max(360, int(2 * np.pi * (viewshed_range_m / res)))
    angle_grid = np.arctan2(dc, dr)
    bin_grid = np.floor((angle_grid + np.pi) / (2 * np.pi) * n_bins).astype(int)
    bin_grid = np.clip(bin_grid, 0, n_bins - 1)

    b_dist = dist_m_grid[in_range]
    b_elev = blocker_elev[in_range]
    b_bin = bin_grid[in_range]

    qdr = query_rows - origin_r
    qdc = query_cols - origin_c
    q_dist = np.sqrt(qdr ** 2 + qdc ** 2) * res
    q_valid = (q_dist > 0) & (q_dist <= viewshed_range_m)

    q_elev = np.arctan2(query_zs - origin_z, np.maximum(q_dist, 1e-9))
    q_angle = np.arctan2(qdc, qdr)
    q_bin = np.floor((q_angle + np.pi) / (2 * np.pi) * n_bins).astype(int)
    q_bin = np.clip(q_bin, 0, n_bins - 1)

    q_dist_v = q_dist[q_valid]
    q_elev_v = q_elev[q_valid]
    q_bin_v = q_bin[q_valid]
    q_orig_idx = np.where(q_valid)[0]

    all_dist = np.concatenate([b_dist, q_dist_v])
    all_elev = np.concatenate([b_elev, q_elev_v])
    all_bin = np.concatenate([b_bin, q_bin_v])
    all_is_query = np.concatenate([np.zeros(len(b_dist), dtype=bool), np.ones(len(q_dist_v), dtype=bool)])
    all_query_orig_idx = np.concatenate([np.full(len(b_dist), -1, dtype=int), q_orig_idx])

    order = np.lexsort((all_dist, all_bin))

    n_query_total = len(query_rows)
    visible_result = np.zeros(n_query_total, dtype=bool)

    current_bin = -1
    max_angle_so_far = -np.inf

    for i in order:
        if all_bin[i] != current_bin:
            current_bin = all_bin[i]
            max_angle_so_far = -np.inf

        if all_is_query[i]:
            if all_elev[i] > max_angle_so_far:
                visible_result[all_query_orig_idx[i]] = True
        else:
            if all_elev[i] > max_angle_so_far:
                max_angle_so_far = all_elev[i]

    return visible_result


def calculate_green_view_tree_sweep_sampled(
        dsm_path: Path,
        dem_path: Path,
        tree_crown_shp: Path,
        building_shp: Path,
        output_shp: Path,
        viewshed_range_m: float = 100.0,
        floor_height_m: float = 3.0,
        floor_eye_offset: float = 1.5,
        wall_offset_m: float = 0.5,
        height_percentile: float = 90,
        tree_id_field: str = "Tree_ID",
        sampling_ratio: float = 1.0,   # 新增: 树冠采样比例, 1.0=全部像素
        random_seed: int = 42,
):
    print("1. 加载DSM/DEM, 计算nDSM...")
    with rasterio.open(dsm_path) as src_dsm, rasterio.open(dem_path) as src_dem:
        transform = src_dsm.transform
        res = transform.a
        dsm_array = src_dsm.read(1)
        dem_array = src_dem.read(1)
        ndsm_array = np.clip(dsm_array - dem_array, 0, None)

    print("2. 加载building/树木, 用真实Tree_ID烧录栅格...")
    trees = gpd.read_file(tree_crown_shp)
    buildings = gpd.read_file(building_shp)

    if tree_id_field not in trees.columns:
        raise ValueError(f"树木数据里没有 {tree_id_field} 字段")

    tree_shapes = ((geom, tid) for geom, tid in zip(trees.geometry, trees[tree_id_field]))
    tree_id_mask = rasterize(tree_shapes, out_shape=dsm_array.shape, transform=transform,
                              fill=0, dtype=np.int32)

    print("3. 估算层数, 生成所有building-floor观察点...")
    observer_records = []
    buildings["bldg_h_m"] = 0.0
    buildings["num_floors"] = 1

    for b_idx, row in tqdm(buildings.iterrows(), total=len(buildings), desc="Building setup"):
        h, n_floors = estimate_building_floors(row.geometry, ndsm_array, transform,
                                                 floor_height_m=floor_height_m, percentile=height_percentile)
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
        raise ValueError("没有生成任何有效observer点")

    obs_rows = np.array([r["row"] for r in observer_records])
    obs_cols = np.array([r["col"] for r in observer_records])
    obs_zs = np.array([r["z_observer"] for r in observer_records])
    obs_xy = np.array([[r["x"], r["y"]] for r in observer_records])
    obs_kdtree = cKDTree(obs_xy)

    n_obs = len(observer_records)
    scores = np.zeros(n_obs, dtype=np.float64)
    visible_sets = [set() for _ in range(n_obs)]

    rng = np.random.default_rng(random_seed)

    print(f"4. 从每棵树的树冠采样点出发(sampling_ratio={sampling_ratio}), 用sweep算法判断可见性...")
    pixel_range = int(viewshed_range_m / res)

    for t_idx, t_row in tqdm(trees.iterrows(), total=len(trees), desc="Tree Progress"):
        tree_id = t_row[tree_id_field]
        geom = t_row.geometry
        tc = geom.centroid

        col_t, row_t = ~transform * (tc.x, tc.y)
        row_t, col_t = int(row_t), int(col_t)
        if not (0 <= row_t < dsm_array.shape[0] and 0 <= col_t < dsm_array.shape[1]):
            continue

        r_min = max(0, row_t - pixel_range)
        r_max = min(dsm_array.shape[0], row_t + pixel_range + 1)
        c_min = max(0, col_t - pixel_range)
        c_max = min(dsm_array.shape[1], col_t + pixel_range + 1)

        loc_dsm = dsm_array[r_min:r_max, c_min:c_max]
        loc_tree_id = tree_id_mask[r_min:r_max, c_min:c_max]

        # ===== 关键改动: 从树冠所有像素里, 按sampling_ratio采样一批origin点 =====
        crown_r, crown_c = np.where(loc_tree_id == tree_id)
        n_crown_px = len(crown_r)
        if n_crown_px == 0:
            continue

        n_sample = max(1, int(round(n_crown_px * sampling_ratio)))
        if n_sample >= n_crown_px:
            sample_idx = np.arange(n_crown_px)  # sampling_ratio=1.0时, 全部像素都用
        else:
            sample_idx = rng.choice(n_crown_px, size=n_sample, replace=False)

        sample_r = crown_r[sample_idx]
        sample_c = crown_c[sample_idx]
        sample_z = loc_dsm[sample_r, sample_c]

        # 找这棵树附近viewshed_range_m内, 有哪些building-floor观察点
        nearby_idx = obs_kdtree.query_ball_point([tc.x, tc.y], r=viewshed_range_m + 5.0)
        if not nearby_idx:
            continue
        nearby_idx = np.array(nearby_idx)

        local_q_rows = obs_rows[nearby_idx] - r_min
        local_q_cols = obs_cols[nearby_idx] - c_min
        local_q_zs = obs_zs[nearby_idx]

        dists_to_obs = np.sqrt((obs_rows[nearby_idx] - row_t) ** 2 +
                                (obs_cols[nearby_idx] - col_t) ** 2) * res

        # 每个采样点各自做一次sweep, 只要任意一个采样点可见, 这棵树对该观察点就算可见
        already_visible = np.zeros(len(nearby_idx), dtype=bool)

        for s_r, s_c, s_z in zip(sample_r, sample_c, sample_z):
            still_pending = ~already_visible
            if not still_pending.any():
                break  # 这棵树对所有附近观察点都已判定可见, 不用再采样了

            visible_flags = sweep_from_origin(
                loc_dsm, s_r, s_c, s_z, res,
                local_q_rows[still_pending], local_q_cols[still_pending], local_q_zs[still_pending],
                viewshed_range_m
            )
            pending_idx = np.where(still_pending)[0]
            already_visible[pending_idx[visible_flags]] = True

        for k, obs_i in enumerate(nearby_idx):
            if already_visible[k]:
                d = max(dists_to_obs[k], 1e-9)
                scores[obs_i] += 1.0 / (d ** 2)
                visible_sets[obs_i].add(tree_id)

    print("5. 汇总结果...")
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
    calculate_green_view_tree_sweep_sampled(
        dsm_path=Path(r"C:\Users\xhe40\Thesis_Data\Campus\test_dsm_Afterfill.tif"),
        dem_path=Path(r"C:\Users\xhe40\Thesis_Data\Campus\test_dem_1m.tif"),
        tree_crown_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\individual_trees.shp"),
        building_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\Building_Campus.shp"),
        output_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\building_result_tree_sweep_sampled.shp"),
        sampling_ratio=1.0,   # 先跑100%, 后面想测采样比例直接改这个数字
    )