"""
用你自己的 Bresenham 逻辑, 给某栋建筑某一层生成连续的 GVS / VTC 场,
效果类似 Cimburova et al. (2023) Figure 1 那种放射状图, 只是"被看的东西"
从一棵树换成了一整层楼看到的所有附近树木的叠加。

和 single_tree_viewshed.py 的核心区别:
  - single_tree_viewshed.py: 观察点铺满一片区域, 目标固定是一棵树,
    观察高度固定 1.5m (地面行人)。
  - 这个脚本: 观察点铺满建筑周围一片区域, 目标是这栋楼附近所有的树 (会叠加),
    观察高度是"这一层楼的高度" (floor_eye_offset + floor_idx*floor_height_m),
    加在每个候选像元自己的地面高程 (DEM) 上。

重要提醒 (跟 GRASS 那版一样的道理, 务必记住):
  观察高度是套在整片候选区域上的, 不是只套在这栋楼的墙面上。也就是说,
  输出的这张连续栅格, 只有落在这栋楼墙面轮廓上的那些像元, 才对应"这一层
  真实存在的观察点"; 图上其他地方的颜色, 回答的是一个假设性的问题
  ("如果这里悬浮着一个这个高度的观察点"), 不代表任何真实存在的人。
  写进 RP 时, 只应该截取墙面轮廓覆盖到的像元来解读。

性能警告 (务必读):
  这是纯 Python 三重循环 (候选像元 x 附近树 x 树冠像元), 没有做任何加速。
  跟今天在 GRASS 里跑一栋楼、225棵树、200x200米区域是同一个量级的计算量,
  不会因为换回 Python 就变快。强烈建议第一次先用很小的 viewshed_range_m
  (比如 30-50m) 和树冠不大的建筑测试, 确认能跑通、图对了, 再考虑加大范围。
  这份脚本只用于给 RP 出一张示范图, 不是用来覆盖全部 57 栋楼的正式流程。
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, transform as window_transform
from skimage.draw import line
from pathlib import Path


def _is_blocked(rr, cc, z_ray, dsm_array, dem_array, tree_id_mask,
                 target_tree_id, trunk_clear_height):
    """
    沿着一条视线判断是否被挡住。
    挡路的像元如果属于"目标树自己", 不算挡住; 属于别的树, 允许从树干净空
    以下穿过; 非树 (建筑/地面), 视线不能低于地表。
    """
    ray_dsm = dsm_array[rr, cc]
    ray_dem = dem_array[rr, cc]
    ray_tid = tree_id_mask[rr, cc]

    for i in range(len(z_ray)):
        zr, dsm_val, dem_val, tid = z_ray[i], ray_dsm[i], ray_dem[i], ray_tid[i]

        if tid == target_tree_id:
            continue
        elif tid > 0:
            if (dem_val + trunk_clear_height) <= zr <= dsm_val:
                return True
        else:
            if zr <= dsm_val:
                return True

    return False


def compute_building_floor_continuous_field(
        dsm_path: Path,
        dem_path: Path,
        tree_crown_shp: Path,
        building_shp: Path,
        target_building_id,
        floor_idx: int,
        output_gvs_tif: Path,
        output_vtc_tif: Path,
        building_id_field: str = "building_o",
        tree_id_field: str = "Tree_ID",
        floor_height_m: float = 4.0,
        floor_eye_offset: float = 1.5,
        viewshed_range_m: float = 100.0,
        trunk_clear_height: float = 2.5,
):
    """
    对某栋楼某一层, 生成连续的 GVS 场和 VTC 场。

    floor_idx 从 0 开始计数 (floor_idx=0 是 1 楼, floor_idx=1 是 2 楼, 以此类推),
    跟你 pilot 主脚本 calculate_green_view_tree_centric 里的用法一致。
    """
    print("1. 加载 DSM/DEM...")
    with rasterio.open(dsm_path) as src_dsm, rasterio.open(dem_path) as src_dem:
        transform = src_dsm.transform
        res = transform.a
        dsm_array = src_dsm.read(1)
        dem_array = src_dem.read(1)
        n_rows_full, n_cols_full = dsm_array.shape

    print(f"2. 加载建筑矢量, 定位目标建筑 ({building_id_field}={target_building_id})...")
    buildings = gpd.read_file(building_shp)
    target_bldg = buildings[buildings[building_id_field] == target_building_id]
    if len(target_bldg) == 0:
        raise ValueError(f"找不到 {building_id_field}={target_building_id} 这栋楼")
    bldg_geom = target_bldg.geometry.iloc[0]
    bminx, bminy, bmaxx, bmaxy = bldg_geom.bounds
    print(f"   建筑范围: x [{bminx:.1f}, {bmaxx:.1f}], y [{bminy:.1f}, {bmaxy:.1f}]")

    print(f"3. 加载树木矢量, 筛选建筑 {viewshed_range_m}m 范围内的树...")
    trees = gpd.read_file(tree_crown_shp)
    if tree_id_field not in trees.columns:
        raise ValueError(f"树木数据里没有 {tree_id_field} 字段")

    bldg_buffer = bldg_geom.buffer(viewshed_range_m)
    nearby_trees = trees[trees.geometry.intersects(bldg_buffer)]
    nearby_tree_ids = list(nearby_trees[tree_id_field].values)
    print(f"   找到 {len(nearby_tree_ids)} 棵附近的树")
    if len(nearby_tree_ids) == 0:
        raise ValueError("建筑周围没有找到树, 检查 viewshed_range_m 或者建筑/树木数据是否对齐")

    print("4. 栅格化全部树木 (用于遮挡判断) 并预取每棵附近树的树冠像元...")
    tree_shapes = ((geom, tid) for geom, tid in zip(trees.geometry, trees[tree_id_field]))
    tree_id_mask = rasterize(tree_shapes, out_shape=dsm_array.shape, transform=transform,
                              fill=0, dtype=np.int32)

    tree_crowns = {}
    for tid in nearby_tree_ids:
        tp_rows, tp_cols = np.where(tree_id_mask == tid)
        if len(tp_rows) > 0:
            tree_crowns[tid] = (tp_rows, tp_cols)
    print(f"   其中 {len(tree_crowns)} 棵树在栅格里有对应的树冠像元")

    print(f"5. 圈定建筑周围 {viewshed_range_m}m 的候选观察像元范围...")
    col_min_f, row_max_f = ~transform * (bminx - viewshed_range_m, bminy - viewshed_range_m)
    col_max_f, row_min_f = ~transform * (bmaxx + viewshed_range_m, bmaxy + viewshed_range_m)
    row_min = max(0, int(row_min_f))
    row_max = min(n_rows_full, int(row_max_f) + 1)
    col_min = max(0, int(col_min_f))
    col_max = min(n_cols_full, int(col_max_f) + 1)

    sub_h = row_max - row_min
    sub_w = col_max - col_min
    gvs_field = np.full((sub_h, sub_w), np.nan, dtype=np.float64)
    vtc_field = np.full((sub_h, sub_w), np.nan, dtype=np.float64)

    observer_elev_offset = floor_eye_offset + floor_idx * floor_height_m
    n_candidates = sub_h * sub_w
    print(f"   候选区域: {sub_h} x {sub_w} = {n_candidates} 像元, "
          f"floor_idx={floor_idx}, 观察高度偏移={observer_elev_offset}m")
    print(f"6. 逐像元计算 (候选像元 x {len(tree_crowns)} 棵树, 可能较慢)...")

    for local_r in range(sub_h):
        if local_r % 10 == 0:
            print(f"   进度: {local_r}/{sub_h} 行")
        for local_c in range(sub_w):
            obs_r, obs_c = row_min + local_r, col_min + local_c

            obs_ground = dem_array[obs_r, obs_c]
            if np.isnan(obs_ground) or obs_ground == -9999.0:
                continue
            z_observer = obs_ground + observer_elev_offset

            gvs_val = 0.0
            vtc_val = 0

            for tid, (tp_rows, tp_cols) in tree_crowns.items():
                tree_visible = False
                for tr, tcx in zip(tp_rows, tp_cols):
                    d = np.sqrt((tr - obs_r) ** 2 + (tcx - obs_c) ** 2) * res
                    if d > viewshed_range_m or d == 0:
                        continue

                    z_target = dsm_array[tr, tcx]
                    rr, cc = line(obs_r, obs_c, tr, tcx)

                    if len(rr) <= 2:
                        gvs_val += 1.0 / (d ** 2)
                        tree_visible = True
                        continue

                    rr, cc = rr[1:-1], cc[1:-1]
                    dists_along = np.sqrt((rr - obs_r) ** 2 + (cc - obs_c) ** 2) * res
                    z_ray = z_observer + (z_target - z_observer) * (dists_along / d)

                    blocked = _is_blocked(rr, cc, z_ray, dsm_array, dem_array,
                                           tree_id_mask, tid, trunk_clear_height)

                    if not blocked:
                        gvs_val += 1.0 / (d ** 2)
                        tree_visible = True

                if tree_visible:
                    vtc_val += 1

            gvs_field[local_r, local_c] = gvs_val
            vtc_field[local_r, local_c] = vtc_val

    print("7. 写出 GeoTIFF (用 QGIS 打开查看, 把 -1 设为 nodata/透明)...")
    window = Window(col_off=col_min, row_off=row_min, width=sub_w, height=sub_h)
    sub_transform = window_transform(window, transform)

    with rasterio.open(dsm_path) as src_dsm:
        crs = src_dsm.crs

    def _write(array, out_path):
        out_array = np.where(np.isnan(array), -1.0, array).astype(np.float32)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            out_path, "w", driver="GTiff",
            height=sub_h, width=sub_w, count=1, dtype=np.float32,
            crs=crs, transform=sub_transform, nodata=-1.0,
        ) as dst:
            dst.write(out_array, 1)

    _write(gvs_field, output_gvs_tif)
    _write(vtc_field, output_vtc_tif)

    print(f"完成。写入:\n  {output_gvs_tif}\n  {output_vtc_tif}")
    print("提醒: 只有落在建筑墙面轮廓上的像元才对应这一层真实的观察点, "
          "其他地方的颜色是假设性的 (\"如果这里悬浮一个这个高度的观察点\")。")

    return gvs_field, vtc_field


if __name__ == "__main__":
    compute_building_floor_continuous_field(
        dsm_path=Path(r"C:\Users\xhe40\Thesis_Data\Campus\test_dsm_Afterfill.tif"),
        dem_path=Path(r"C:\Users\xhe40\Thesis_Data\Campus\test_dem_1m.tif"),
        tree_crown_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\individual_trees.shp"),
        building_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\Building_Campus_with_floors_4.shp"),
        target_building_id=6577346,
        floor_idx=1,  # 0 = 1楼, 1 = 2楼
        output_gvs_tif=Path(r"C:\Users\xhe40\Thesis_Data\Campus\b6577346_f2_GVS_continuous.tif"),
        output_vtc_tif=Path(r"C:\Users\xhe40\Thesis_Data\Campus\b6577346_f2_VTC_continuous.tif"),
        viewshed_range_m=50.0,  # 先用小范围测试, 确认跑通再考虑加大
    )