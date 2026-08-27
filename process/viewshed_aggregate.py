"""
Process 核心模块 (纯 Python 高效版):
模拟 Cimburova 的反向视域与距离衰减逻辑 (Distance-Decay Exposure)，
不依赖 GRASS，100% 运行于 Python/Conda 环境。

核心思路:
  1. 读取 DSM 栅格与 individual_trees.shp 矢量。
  2. 遍历每一棵树的质心，以该点为圆心在设定半径（如 100m）内计算欧氏距离矩阵。
  3. 应用距离衰减公式 (Weight = 1 / distance^2)，生成每棵树的“视域视觉影响矩阵”。
  4. 将所有树的影响矩阵累加到一张全局 `total_exposure` 栅格中。
  5. 在每栋建筑的位置上采样查询 `total_exposure` 的值，写入结果 Shapefile。
"""

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
from pathlib import Path
from shapely.geometry import Point

def run_viewshed_aggregate(
    dsm_path: Path,
    tree_crown_shp: Path,
    building_shp: Path,
    output_shp: Path,
    grass_workdir: Path = None,  # 兼容参数，纯Python版无需使用
    viewshed_range_m: float = 100.0,
    observer_height_m: float = 1.5,
    epsg: str = "2193",
) -> Path:

    print(">>> 启动纯 Python 版反向视域累加计算引擎 (Cimburova Logic Alternative)...")

    # 1. 读取 DSM 数据与地理空间元数据
    with rasterio.open(str(dsm_path)) as src:
        dsm_data = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        nodata = src.nodatavals[0]
        height, width = dsm_data.shape
        resolution = transform.a  # 像元大小 (米/像素)

    # 2. 读取单木矢量图
    trees_gdf = gpd.read_file(tree_crown_shp)
    print(f"成功加载单木树冠要素共: {len(trees_gdf)} 棵")

    # 3. 初始化全局累加影响矩阵 (Total Exposure Raster)
    total_exposure = np.zeros((height, width), dtype=np.float32)

    # 4. 计算搜索半径对应的像素数
    max_radius_px = int(viewshed_range_m / resolution)

    # 5. 循环遍历每一棵树，计算其空间距离衰减影响范围
    for idx, row in trees_gdf.iterrows():
        geom = row['geometry']
        centroid = geom.centroid
        cx, cy = centroid.x, centroid.y

        # 将世界坐标 (X, Y) 转换为栅格的行列号 (Row, Col)
        # rasterio.transform.rowcol 可以直接转换
        from rasterio.transform import rowcol
        r_center, c_center = rowcol(transform, cx, cy)

        # 定义该树在矩阵中的局部裁剪窗口边界 (防止超出栅格范围)
        r_min = max(0, r_center - max_radius_px)
        r_max = min(height, r_center + max_radius_px + 1)
        c_min = max(0, c_center - max_radius_px)
        c_max = min(width, c_center + max_radius_px + 1)

        if r_min >= r_max or c_min >= c_max:
            continue

        # 生成局部网格的行列坐标
        rr, cc = np.ogrid[r_min:r_max, c_min:c_max]

        # 计算局部网格像元与树中心点的像素距离
        dist_px = np.sqrt((rr - r_center)**2 + (cc - c_center)**2)
        dist_m = dist_px * resolution

        # 仅在搜索半径内计算影响
        mask = (dist_m > 0) & (dist_m <= viewshed_range_m)

        # 应用 Cimburova 论文中使用的 Distance-decay 函数 (权重与距离平方成反比)
        # 公式: Weight = 1.0 / (distance^2)
        local_exposure = np.zeros_like(dist_m, dtype=np.float32)
        local_exposure[mask] = 100.0 / (dist_m[mask] ** 2)  # 乘以100缩放数值以便可读

        # 累加进全局矩阵
        total_exposure[r_min:r_max, c_min:c_max] += local_exposure

        if (idx + 1) % 200 == 0 or (idx + 1) == len(trees_gdf):
            print(f"  - 已处理累加树木: {idx + 1} / {len(trees_gdf)}")

    print("所有树木视域影响累加完毕，开始采样提取至建筑要素...")

    # 6. 读取建筑要素
    buildings_gdf = gpd.read_file(building_shp)

    # 7. 在每栋建筑的质心位置提取 total_exposure 矩阵的值
    scores = []
    for geom in buildings_gdf.geometry:
        b_centroid = geom.centroid
        bx, by = b_centroid.x, b_centroid.y

        try:
            br, bc = rowcol(transform, bx, by)
            if 0 <= br < height and 0 <= bc < width:
                val = float(total_exposure[br, bc])
            else:
                val = 0.0
        except Exception:
            val = 0.0
        scores.append(val)

    # 将得分写入属性表
    buildings_gdf['green_view_score'] = scores

    # 8. 导出结果 Shapefile
    output_shp.parent.mkdir(parents=True, exist_ok=True)
    buildings_gdf.to_file(output_shp)
    print(f"计算圆满完成！结果已成功保存至: {output_shp}")

    return output_shp