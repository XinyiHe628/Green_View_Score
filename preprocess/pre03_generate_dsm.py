"""
Preprocess 步骤2:把点云转成DSM(数字表面模型)。

DSM取的是每个栅格像素范围内,所有点里"最高"的那个值——
这样建筑物屋顶、树冠顶部都会体现在表面高度上,
后面viewshed计算需要的正是这张"会被建筑物和树木挡住视线"的表面。

不用GRASS,用numpy自己做binning(每个像素取最大值),rasterio负责写出GeoTIFF。
"""

import laspy
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pathlib import Path
from rasterio.crs import CRS
from scipy.ndimage import distance_transform_edt
import geopandas as gpd
from rasterio.features import rasterize

def _fill_gaps_nearest(array: np.ndarray, nodata_mask: np.ndarray) -> np.ndarray:
    """用最近邻填补array里nodata_mask标记的空洞。"""
    if not nodata_mask.any():
        return array
    indices = distance_transform_edt(nodata_mask, return_distances=False, return_indices=True)
    return array[tuple(indices)]


def generate_dsm(las_path: Path, resolution: float, output_tif: Path, boundary_shp: Path, buffer_m: float) -> Path:
    """
    读取点云,按resolution做栅格化,每个像素取落入其中所有点的最大高度值。

    参数:
        las_path: 输入的.laz/.las文件路径(应该是已经合并裁剪好的那个文件)
        resolution: 输出栅格的分辨率,单位米(论文里用1m)
        output_tif: 输出的DSM GeoTIFF路径

    返回:
        output_tif
    """
    with laspy.open(las_path) as las_file:
        las = las_file.read()

    x, y, z = np.array(las.x), np.array(las.y), np.array(las.z)

    minx, miny, maxx, maxy = x.min(), y.min(), x.max(), y.max()
    n_cols = int(np.ceil((maxx - minx) / resolution))
    n_rows = int(np.ceil((maxy - miny) / resolution))

    # 每个点落在哪一列、哪一行(栅格第0行对应最北边,所以y方向要反过来)
    col_idx = np.floor((x - minx) / resolution).astype(int)
    row_idx = np.floor((maxy - y) / resolution).astype(int)
    col_idx = np.clip(col_idx, 0, n_cols - 1)
    row_idx = np.clip(row_idx, 0, n_rows - 1)

    # 用一个"负无穷"初始化的数组,每个像素取落入其中所有点的最大值
    dsm = np.full((n_rows, n_cols), -np.inf, dtype=np.float32)
    flat_idx = row_idx * n_cols + col_idx
    np.maximum.at(dsm.ravel(), flat_idx, z.astype(np.float32))

    # 没有任何点落入的像素(还是-inf),标记成nodata
    no_data_mask = np.isinf(dsm)
    dsm = _fill_gaps_nearest(dsm, no_data_mask)
    nodata_value = -9999.0

    transform = from_origin(minx, maxy, resolution, resolution)

    # 新增:用meshblock真实形状,把study area之外的像素重新盖回nodata
    boundary_gdf = gpd.read_file(boundary_shp)
    study_area = boundary_gdf.union_all()
    study_area_buffered = study_area.buffer(buffer_m)

    inside_mask = rasterize(
        [(study_area_buffered, 1)],
        out_shape=(n_rows, n_cols),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    ).astype(bool)

    dsm[~inside_mask] = nodata_value   # 不在真实形状范围内的,强制改回nodata
    output_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_tif, "w",
        driver="GTiff",
        height=n_rows, width=n_cols,
        count=1, dtype=dsm.dtype,
        crs=CRS.from_epsg(2193),  # TODO: 确认坐标系,通常是EPSG:2193 (NZTM2000),从las.header.vlrs里读或者手动指定
        transform=transform,
        nodata=nodata_value,
    ) as dst:
        dst.write(dsm, 1)

    n_empty = no_data_mask.sum()
    if n_empty > 0:
        print(f"警告: DSM里有 {n_empty} 个像素没有任何点落入(空洞),占比 {n_empty / dsm.size:.2%}")
    print(f"DSM生成完成: {n_rows}行 x {n_cols}列, 写入 {output_tif}")
    return output_tif