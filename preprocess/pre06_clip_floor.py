"""
Preprocess步骤: 估算每栋building的高度和楼层数。

输入: DSM, DEM, building footprint矢量
输出: (1) 带bldg_h_m和num_floors两个新字段的building shapefile
      (2) nDSM栅格文件(相对地面高度), 供后续步骤复用

跟process阶段的viewshed计算完全解耦, 只跑一次, 后面不管用哪个版本的
viewshed脚本(逐点画线/sweep/树出发/building出发), 都直接读这一步产出的
building shapefile里的num_floors字段, 不用重复算。
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from pathlib import Path
from tqdm import tqdm


def estimate_building_floors(building_geom, ndsm_array, transform,
                              floor_height_m: float = 3.0, percentile: float = 90):
    """
    用building footprint范围内的nDSM像素, 取percentile当作建筑高度估算值,
    再除以floor_height_m向下取整, 得到估算层数(至少1层)。
    """
    mask = geometry_mask([building_geom], out_shape=ndsm_array.shape,
                          transform=transform, invert=True)
    vals = ndsm_array[mask]
    vals = vals[(vals > 0) & (vals < 200)]  # 排除nodata/异常值

    if len(vals) == 0:
        return 0.0, 1

    h = np.percentile(vals, percentile)
    n_floors = max(1, int(h // floor_height_m))
    return h, n_floors


def generate_ndsm_and_floors(
        dsm_path: Path,
        dem_path: Path,
        building_shp: Path,
        output_building_shp: Path,
        output_ndsm_path: Path = None,
        floor_height_m: float = 3.0,
        height_percentile: float = 90,
):
    """
    参数:
        dsm_path: 输入DSM栅格路径
        dem_path: 输入DEM栅格路径
        building_shp: 输入building footprint矢量路径
        output_building_shp: 输出路径, 带bldg_h_m和num_floors字段的building shapefile
        output_ndsm_path: 可选, nDSM栅格的输出路径。传None则不保存。
        floor_height_m: 假设的每层楼高度(米), 默认3米
        height_percentile: 计算建筑高度时用的百分位数, 默认90
    """
    print("1. 加载DSM/DEM, 计算nDSM...")
    with rasterio.open(dsm_path) as src_dsm, rasterio.open(dem_path) as src_dem:
        transform = src_dsm.transform
        profile = src_dsm.profile
        dsm_array = src_dsm.read(1)
        dem_array = src_dem.read(1)
        ndsm_array = np.clip(dsm_array - dem_array, 0, None)

    if output_ndsm_path is not None:
        print(f"   保存nDSM栅格至: {output_ndsm_path}")
        profile.update(dtype=ndsm_array.dtype)
        output_ndsm_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_ndsm_path, "w", **profile) as dst:
            dst.write(ndsm_array, 1)

    print("2. 加载building矢量...")
    buildings = gpd.read_file(building_shp)

    print(f"3. 估算每栋建筑的高度(percentile={height_percentile})和层数(floor_height_m={floor_height_m})...")
    buildings["bldg_h_m"] = 0.0
    buildings["num_floors"] = 1

    for idx, row in tqdm(buildings.iterrows(), total=len(buildings), desc="Estimating floors"):
        h, n = estimate_building_floors(
            row.geometry, ndsm_array, transform,
            floor_height_m=floor_height_m, percentile=height_percentile
        )
        buildings.loc[idx, "bldg_h_m"] = h
        buildings.loc[idx, "num_floors"] = n

    print(f"4. 写入结果至: {output_building_shp}")
    output_building_shp.parent.mkdir(parents=True, exist_ok=True)
    buildings.to_file(output_building_shp)

    print(f"🎉 完成! 共处理 {len(buildings)} 栋建筑, "
          f"层数范围 {buildings['num_floors'].min()}-{buildings['num_floors'].max()}, "
          f"平均 {buildings['num_floors'].mean():.1f} 层")

    return buildings


if __name__ == "__main__":
    generate_ndsm_and_floors(
        dsm_path=Path(r"C:\Users\xhe40\Thesis_Data\Campus\test_dsm_Afterfill.tif"),
        dem_path=Path(r"C:\Users\xhe40\Thesis_Data\Campus\test_dem_1m.tif"),
        building_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\Building_Campus.shp"),
        output_building_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\Building_Campus_with_floors.shp"),
        output_ndsm_path=Path(r"C:\Users\xhe40\Thesis_Data\Campus\ndsm_1m.tif"),
        floor_height_m=3.0,
        height_percentile=90,
    )