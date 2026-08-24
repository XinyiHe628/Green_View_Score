"""
Preprocess 步骤3:生成Canopy(树冠)栅格。

思路: Canopy Height Model (CHM) = DSM - DTM
  - DSM(上一步已经生成): 表面最高点,包含建筑物屋顶、树冠顶部
  - DTM(这一步内部生成): 只用LiDAR点云里"地面分类"的点(class=2)算出来的裸地高度
  - CHM: 两者相减,得到"每个像素比地面高出多少" —— 树冠、建筑物都会比地面高
  - 再用高度阈值筛掉太矮的东西(草地、灌木),剩下的当作树冠

注意: 这一步区分不了"树冠"和"高层建筑物"(两者CHM都可能很高),
如果study area里有高层建筑物紧挨着树,后续可能需要额外规则
(比如结合building footprint,把落在building范围内的高值排除掉)。
先跑通pilot,这个问题在验证阶段(afterprocess)重点检查。
"""

import laspy
import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt
from pathlib import Path


def _fill_gaps_nearest(array: np.ndarray, nodata_mask: np.ndarray) -> np.ndarray:
    """用最近邻填补array里nodata_mask标记的空洞(地面点比较稀疏,DTM栅格化后容易有空洞)。"""
    if not nodata_mask.any():
        return array
    indices = distance_transform_edt(nodata_mask, return_distances=False, return_indices=True)
    return array[tuple(indices)]


def generate_canopy(
    las_path: Path,
    dsm_path: Path,
    output_tif: Path,
    height_threshold: float,
    ground_class: int = 2,
) -> Path:
    """
    参数:
        las_path: 合并裁剪好的点云文件(跟生成DSM用的是同一个)
        dsm_path: 上一步生成的DSM,用来对齐网格(grid必须完全一致才能相减)
        output_tif: 输出的canopy二值栅格(1=树冠, 0=不是)
        height_threshold: 高于地面多少米才算树冠,单位米
        ground_class: LAS分类码里代表"地面点"的数值,标准是2

    返回:
        output_tif
    """
    with rasterio.open(dsm_path) as dsm_src:
        dsm = dsm_src.read(1)
        transform = dsm_src.transform
        dsm_nodata = dsm_src.nodata
        n_rows, n_cols = dsm.shape

    with laspy.open(las_path) as las_file:
        las = las_file.read()

    is_ground = np.array(las.classification) == ground_class
    if is_ground.sum() == 0:
        raise ValueError(
            f"点云里没有找到分类码={ground_class}的地面点。"
            f"检查一下这份LiDAR数据是否自带标准LAS分类,如果没有分类,需要先做ground classification才能生成DTM。"
        )

    gx, gy, gz = np.array(las.x)[is_ground], np.array(las.y)[is_ground], np.array(las.z)[is_ground]

    # 用DSM同样的transform,把地面点栅格化到完全对齐的网格上(这样才能跟DSM逐像素相减)
    minx, maxy = transform.c, transform.f
    resolution = transform.a

    col_idx = np.clip(np.floor((gx - minx) / resolution).astype(int), 0, n_cols - 1)
    row_idx = np.clip(np.floor((maxy - gy) / resolution).astype(int), 0, n_rows - 1)

    # DTM每个像素取地面点的最小值(裸地高度,不要混入任何高出来的东西)
    dtm = np.full((n_rows, n_cols), np.inf, dtype=np.float32)
    flat_idx = row_idx * n_cols + col_idx
    np.minimum.at(dtm.ravel(), flat_idx, gz.astype(np.float32))

    dtm_gap_mask = np.isinf(dtm)
    dtm = _fill_gaps_nearest(dtm, dtm_gap_mask)

    # CHM = DSM - DTM,DSM本身的nodata像素也标记为无效
    dsm_valid_mask = dsm != dsm_nodata if dsm_nodata is not None else np.ones_like(dsm, dtype=bool)
    chm = np.where(dsm_valid_mask, dsm - dtm, np.nan)

    canopy = (chm > height_threshold).astype(np.uint8)

    output_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_tif, "w",
        driver="GTiff",
        height=n_rows, width=n_cols,
        count=1, dtype=canopy.dtype,
        crs=None,  # TODO: 跟generate_dsm.py保持一致,确认实际坐标系
        transform=transform,
        nodata=255,
    ) as dst:
        dst.write(canopy, 1)

    n_canopy_pixels = canopy.sum()
    print(f"Canopy生成完成: {n_canopy_pixels} 个像素被判定为树冠 (占比 {n_canopy_pixels / canopy.size:.2%}), 写入 {output_tif}")
    return output_tif