import laspy
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt


def generate_dem_from_laz(laz_path: str, output_tif_path: str, resolution: float = 1.0) -> None:
    """
    从已分类的 LAZ 点云文件中提取地面点并生成 DEM 栅格（基于极速 Binning 填充法）。
    """
    print(f"读取点云: {laz_path}")
    las = laspy.read(laz_path)

    # 提取分类号为 2 的地面点 (Ground)
    ground_mask = las.classification == 2
    ground_points = las.points[ground_mask]

    x = np.array(ground_points.x)
    y = np.array(ground_points.y)
    z = np.array(ground_points.z)
    num_points = len(x)

    if num_points == 0:
        raise ValueError("未在 LAZ 文件中找到地面点 (Class 2)！")

    print(f"成功提取 {num_points} 个地面点，开始映射至 2D 网格...")

    # 1. 确定边界与网格维度
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    cols = int(np.ceil((x_max - x_min) / resolution))
    rows = int(np.ceil((y_max - y_min) / resolution))

    # 2. 将 3D 点映射到矩阵的行列索引 (Row, Col)
    col_idx = np.clip(((x - x_min) / resolution).astype(int), 0, cols - 1)
    row_idx = np.clip(((y_max - y) / resolution).astype(int), 0, rows - 1)

    # 3. 构造网格并累加点的高程 (取每个网格内的平均高度)
    dem_sum = np.zeros((rows, cols), dtype=np.float64)
    dem_count = np.zeros((rows, cols), dtype=np.int32)

    np.add.at(dem_sum, (row_idx, col_idx), z)
    np.add.at(dem_count, (row_idx, col_idx), 1)

    # 计算均值网格
    with np.errstate(divide='ignore', invalid='ignore'):
        dem_array = dem_sum / dem_count
        dem_array[dem_count == 0] = np.nan

    # 4. 极速填补建筑底下/未扫描到的无数据空洞 (Distance Transform 插值)
        # 4. 受限的极速填补：只填内部空洞，不拉伸外部边缘
        nan_mask = np.isnan(dem_array)
        if np.any(nan_mask):
            print("正在填补建筑底下的无数据空洞...")
            # 同时返回距离矩阵和索引矩阵
            distances, indices = distance_transform_edt(nan_mask, return_distances=True, return_indices=True)

            # 设定最大填补距离（假设 resolution=1，50 代表 50 米）
            # 超过 50 米的空白（通常是矩形边界外的区域）将保持为 NaN
            max_fill_distance = 50.0
            fill_mask = nan_mask & (distances <= max_fill_distance)

            # 只在允许的距离范围内进行最近邻填补
            dem_array[fill_mask] = dem_array[tuple(indices)][fill_mask]

    # 5. 导出 TIF
    print("正在导出为 TIF...")
    transform = from_origin(x_min, y_max, resolution, resolution)
    with rasterio.open(
            output_tif_path,
            'w',
            driver='GTiff',
            height=rows,
            width=cols,
            count=1,
            dtype=np.float32,
            crs="EPSG:2193",
            transform=transform,
            nodata=-9999.0
    ) as dst:
        dst.write(dem_array.astype(np.float32), 1)

    print(f"🎉 成功！DEM 已生成: {output_tif_path}")