"""
Preprocess 步骤4: 单木分割 (Individual Tree Segmentation)

思路:
  - 基于上一步生成的纯净 Canopy Mask 和高程模型 DSM。
  - 对 DSM 进行轻微的高斯平滑（消除同一棵树上的多个细小树枝高点）。
  - 使用 Local Maxima 寻找局部最高点作为树顶 (Tree Tops)。
  - 运用 Watershed (分水岭算法) 划定每棵树的树冠边界。
  - 将每个树冠栅格块转换为独立的矢量 Polygon，输出 shapefile。
"""

import rasterio
from rasterio.features import shapes
import numpy as np
import geopandas as gpd
from shapely.geometry import shape
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from pathlib import Path


def generate_individual_trees(
        dsm_path: Path,
        canopy_path: Path,
        output_shp: Path,
        min_distance_m: float = 3.0
) -> Path:
    """
    参数:
        dsm_path: 步骤2生成的 DSM (包含高度信息)
        canopy_path: 步骤3生成的 纯净 Canopy 二值掩膜
        output_shp: 输出的单木树冠矢量文件路径
        min_distance_m: 两棵树顶之间的最小距离(米)，防止一棵大树被过度分割
    """
    print(">>> 开始进行单木树冠分割 (Watershed Algorithm) ...")

    # 1. 读取数据
    with rasterio.open(dsm_path) as dsm_src:
        dsm = dsm_src.read(1).astype(np.float32)
        transform = dsm_src.transform
        crs = dsm_src.crs
        resolution = transform.a

    with rasterio.open(canopy_path) as can_src:
        canopy_mask = can_src.read(1) > 0  # 转为布尔掩膜

    # 仅在树冠掩膜内保留 DSM 高度，其余设为 0
    tree_dsm = np.where(canopy_mask, dsm, 0)

    # 2. 高斯平滑 (Gaussian Smoothing)
    # 平滑处理非常关键，否则一棵树的多个小树干会被识别成多棵树
    sigma = 1.0 / resolution  # 约 1 米的平滑核
    smoothed_dsm = gaussian_filter(tree_dsm, sigma=sigma)
    smoothed_dsm = np.where(canopy_mask, smoothed_dsm, 0)

    # 3. 寻找树顶 (Local Maxima)
    min_distance_px = max(1, int(min_distance_m / resolution))

    # 找到局部极值点的坐标
    local_max_coords = peak_local_max(
        smoothed_dsm,
        min_distance=min_distance_px,
        labels=canopy_mask
    )

    # 为每个树顶生成唯一的 ID 标记 (Marker)
    markers = np.zeros_like(smoothed_dsm, dtype=np.int32)
    for i, (row, col) in enumerate(local_max_coords):
        markers[row, col] = i + 1  # 树木 ID 从 1 开始

    # 4. 分水岭算法 (Watershed)
    # 分水岭算法找的是汇水盆地(最小值)，所以我们要把 DSM 倒置 (-smoothed_dsm)
    labels = watershed(-smoothed_dsm, markers, mask=canopy_mask)

    # 5. 栅格转矢量 (Raster to Vector Polygons)
    # labels 矩阵中，每个非 0 像素组代表一棵独立的树
    polygons = []
    tree_ids = []

    # 遍历生成的每个树冠区块
    for geom, value in shapes(labels, mask=canopy_mask, transform=transform):
        if value > 0:
            polygons.append(shape(geom))
            tree_ids.append(int(value))

    # 6. 生成 GeoDataFrame 并保存为 Shapefile
    gdf = gpd.GeoDataFrame(
        {'Tree_ID': tree_ids},
        geometry=polygons,
        crs=crs
    )

    # 输出结果
    output_shp.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_shp)

    print(f"单木分割完成！共提取出 {len(gdf)} 棵独立树木。")
    print(f"树冠矢量已保存至: {output_shp}")

    return output_shp