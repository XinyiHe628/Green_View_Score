"""
Preprocess 步骤3: 生成 Canopy (树冠) 栅格。

重构思路:
  - 放弃“DSM - DTM 然后剪矢量”的传统方案，直接利用标准 LiDAR 点云的自带分类 (Class 4/5 为植被)。
  - 将 Class 4/5 植被点直接栅格化到与 DSM 完全对齐的网格中。
  - 计算 CHM (DSM - DTM) 仅用于根据 height_threshold 过滤低矮植被 (如 < 2m 的灌木/草地)。
  - 使用形态学开运算 (Morphological Opening) 瞬间抹除孤立噪点与微小伪影，无需依赖 Building Footprint 矢量！
"""

import laspy
import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt, binary_opening
from pathlib import Path
from rasterio.crs import CRS

def _fill_gaps_nearest(array: np.ndarray, nodata_mask: np.ndarray) -> np.ndarray:
    """用最近邻填补 DTM 的空洞"""
    if not nodata_mask.any():
        return array
    indices = distance_transform_edt(nodata_mask, return_distances=False, return_indices=True)
    return array[tuple(indices)]


def generate_canopy(
    las_path: Path,
    dsm_path: Path,
    output_tif: Path,
    height_threshold: float = 2.0,
    ground_class: int = 2,
    veg_classes: list = [4, 5],  # Standard LAS: 4=Medium Veg, 5=High Veg
) -> Path:
    """
    参数:
        las_path: 合并裁剪好的点云文件 (.las / .laz)
        dsm_path: 上一步生成的 DSM，用来精细对齐栅格网格
        output_tif: 输出的 Canopy 二值栅格 (1=树冠, 0=非树冠)
        height_threshold: 树冠高度阈值 (米)，低于此高度的植被判定为草地/低矮灌木
        ground_class: LAS 地面点分类码 (标准=2)
        veg_classes: LAS 植被点分类码列表 (标准=[4, 5])
    """
    # 1. 读取 DSM 尺寸与 Transform 信息 (保证输出栅格 100% 对齐)
    with rasterio.open(dsm_path) as dsm_src:
        dsm = dsm_src.read(1)
        transform = dsm_src.transform
        dsm_nodata = dsm_src.nodata
        crs = dsm_src.crs
        n_rows, n_cols = dsm.shape

    # 2. 读取 LiDAR 点云
    with laspy.open(las_path) as las_file:
        las = las_file.read()

    las_cls = np.array(las.classification)

    # 3. 计算 DTM (用 Class 2 裸地点) 算出相对真实高度 CHM
    is_ground = (las_cls == ground_class)
    if not is_ground.any():
        raise ValueError(f"点云中未找到分类码={ground_class}的地面点！")

    gx, gy, gz = np.array(las.x)[is_ground], np.array(las.y)[is_ground], np.array(las.z)[is_ground]

    minx, maxy = transform.c, transform.f
    resolution = transform.a

    col_idx_g = np.clip(np.floor((gx - minx) / resolution).astype(int), 0, n_cols - 1)
    row_idx_g = np.clip(np.floor((maxy - gy) / resolution).astype(int), 0, n_rows - 1)

    dtm = np.full((n_rows, n_cols), np.inf, dtype=np.float32)
    flat_idx_g = row_idx_g * n_cols + col_idx_g
    np.minimum.at(dtm.ravel(), flat_idx_g, gz.astype(np.float32))
    dtm = _fill_gaps_nearest(dtm, np.isinf(dtm))

    # 4. 核心逻辑修改：直接提取 Class 4 和 Class 5 的植被点
    is_veg = np.isin(las_cls, veg_classes)
    if not is_veg.any():
        print("警告: 未在 LAS 中提取到 Class 4/5 植被点，请检查点云分类！")

    vx, vy, vz = np.array(las.x)[is_veg], np.array(las.y)[is_veg], np.array(las.z)[is_veg]

    col_idx_v = np.clip(np.floor((vx - minx) / resolution).astype(int), 0, n_cols - 1)
    row_idx_v = np.clip(np.floor((maxy - vy) / resolution).astype(int), 0, n_rows - 1)

    # 生成初步的植被高度网格 (每个像素记录落入该像素的最大植被点高程)
    veg_z_grid = np.full((n_rows, n_cols), -np.inf, dtype=np.float32)
    flat_idx_v = row_idx_v * n_cols + col_idx_v
    np.maximum.at(veg_z_grid.ravel(), flat_idx_v, vz.astype(np.float32))

    # 5. 计算植被相对地面高度 CHM_veg = Veg_Z - DTM
    veg_chm = np.where(veg_z_grid != -np.inf, veg_z_grid - dtm, 0.0)

    # 6. 施加高度阈值过滤 (CHM > height_threshold 判定为树冠)
    canopy_binary = (veg_chm >= height_threshold)

    # 7. 后处理：形态学开运算 (Opening) 抹除微小噪点和边缘粗糙点
    # structure 定义 3x3 的连通域
    cleaned_canopy = binary_opening(canopy_binary, structure=np.ones((3, 3))).astype(np.uint8)

    # 如果 DSM 有 NODATA 区域，同步掩膜
    if dsm_nodata is not None:
        cleaned_canopy[dsm == dsm_nodata] = 0

    # 8. 写入结果 GeoTIFF
    output_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_tif, "w",
        driver="GTiff",
        height=n_rows, width=n_cols,
        count=1, dtype=np.uint8,
        crs=crs if crs else CRS.from_epsg(2193),
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(cleaned_canopy, 1)

    n_canopy_pixels = cleaned_canopy.sum()
    print(f"Canopy 生成成功 (已消除建筑伪影): {n_canopy_pixels} 个像素被判定为树冠 (占比 {n_canopy_pixels / cleaned_canopy.size:.2%}), 写入 {output_tif}")
    return output_tif