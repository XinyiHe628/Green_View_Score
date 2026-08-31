"""
Process 核心模块 - 2D Distance-Decay版本 (对照组):
模拟 Cimburova 的反向视域与距离衰减逻辑，不做遮挡判断，
纯粹按距离衰减累加每棵树对周边building的影响。

与3D版本(viewshed_3d_treecentric)的核心区别:
  - 不做任何视线遮挡检查（没有Bresenham射线，没有DSM高度判断）
  - 不区分楼层（所有building只有一个observer点，不分floor）
  - observer点同样使用墙面+0.5m偏移（与3D版本保持一致，确保对比公平）
  - 输出: 每栋building一个green_view_score（连续加权值），不含VTC

作为对照组，目的是展示"如果不考虑3D遮挡，结果会差多少"。
"""

import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
import numpy as np
from pathlib import Path
from shapely.geometry import Point


def get_observer_point(building_geom, offset_m: float = 0.5):
    """
    跟3D版本完全相同的observer点计算逻辑:
    找building外墙上离centroid最近的点, 再往外推offset_m米。
    保持一致才能确保2D vs 3D的对比是公平的。
    """
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


def filter_current_buildings(buildings, end_lifespan_field="end_lifesp"):
    """过滤LINZ building数据里的历史版本，只保留当前有效记录。"""
    if end_lifespan_field not in buildings.columns:
        print(f"   警告: 找不到字段 '{end_lifespan_field}', 跳过过滤")
        return buildings

    n_before = len(buildings)
    current = buildings[buildings[end_lifespan_field].isna()].copy()
    print(f"   过滤历史版本: {n_before} → {len(current)} 栋")
    return current.reset_index(drop=True)


def run_viewshed_aggregate(
        dsm_path: Path,
        tree_crown_shp: Path,
        building_shp: Path,
        output_shp: Path,
        grass_workdir: Path = None,   # 兼容参数，不使用
        viewshed_range_m: float = 100.0,
        observer_height_m: float = 1.5,   # 保留但不影响计算（2D无高度维度）
        wall_offset_m: float = 0.5,        # 新增：与3D版本一致的墙面偏移
        end_lifespan_field: str = "end_lifesp",
        epsg: str = "2193",
) -> Path:
    print(">>> 启动2D Distance-Decay视域累加计算（对照组）...")

    # 1. 读取DSM元数据（只需要坐标系和transform，不需要高度值）
    with rasterio.open(str(dsm_path)) as src:
        transform = src.transform
        height_px, width_px = src.height, src.width
        resolution = transform.a

    # 2. 读取树冠数据
    trees_gdf = gpd.read_file(tree_crown_shp)
    print(f"   树冠数据: {len(trees_gdf)} 棵")

    # 3. 初始化全局累加矩阵
    total_exposure = np.zeros((height_px, width_px), dtype=np.float32)
    max_radius_px = int(viewshed_range_m / resolution)

    # 4. 遍历每一棵树，累加distance-decay影响
    for idx, row in trees_gdf.iterrows():
        centroid = row['geometry'].centroid
        r_center, c_center = rowcol(transform, centroid.x, centroid.y)

        r_min = max(0, r_center - max_radius_px)
        r_max = min(height_px, r_center + max_radius_px + 1)
        c_min = max(0, c_center - max_radius_px)
        c_max = min(width_px, c_center + max_radius_px + 1)

        if r_min >= r_max or c_min >= c_max:
            continue

        rr, cc = np.ogrid[r_min:r_max, c_min:c_max]
        dist_px = np.sqrt((rr - r_center) ** 2 + (cc - c_center) ** 2)
        dist_m = dist_px * resolution

        in_range = (dist_m > 0) & (dist_m <= viewshed_range_m)
        local_exposure = np.zeros_like(dist_m, dtype=np.float32)
        local_exposure[in_range] = 1.0 / (dist_m[in_range] ** 2)

        total_exposure[r_min:r_max, c_min:c_max] += local_exposure

        if (idx + 1) % 200 == 0 or (idx + 1) == len(trees_gdf):
            print(f"   已处理: {idx + 1} / {len(trees_gdf)} 棵")

    print("   累加完成，开始提取建筑分数...")

    # 5. 读取building数据，先过滤历史版本
    buildings_gdf = gpd.read_file(building_shp)
    buildings_gdf = filter_current_buildings(buildings_gdf, end_lifespan_field)

    # 6. 对每栋building，用墙面偏移observer点采样（与3D版本保持一致）
    scores = []
    for geom in buildings_gdf.geometry:
        obs_pt = get_observer_point(geom, offset_m=wall_offset_m)
        try:
            br, bc = rowcol(transform, obs_pt.x, obs_pt.y)
            if 0 <= br < height_px and 0 <= bc < width_px:
                val = float(total_exposure[br, bc])
            else:
                val = 0.0
        except Exception:
            val = 0.0
        scores.append(val)

    buildings_gdf['GVS_2D'] = scores   # 字段名改成GVS_2D, 跟3D版本的GVS_F1区分开

    # 7. 导出结果
    output_shp.parent.mkdir(parents=True, exist_ok=True)
    buildings_gdf.to_file(output_shp)
    print(f"🎉 完成! 结果写入: {output_shp}")
    print(f"   共 {len(buildings_gdf)} 栋building, GVS_2D范围: {min(scores):.4f} - {max(scores):.4f}")

    return output_shp


if __name__ == "__main__":
    run_viewshed_aggregate(
        dsm_path=Path(r"C:\Users\xhe40\Thesis_Data\Campus\test_dsm_Afterfill.tif"),
        tree_crown_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\individual_trees.shp"),
        building_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\Building_Campus.shp"),
        output_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\building_result_2D_aggregate.shp"),
        viewshed_range_m=100.0,
        wall_offset_m=0.5,
    )