"""
Preprocess步骤: 过滤building重复历史版本 + 估算每栋building的高度和楼层数。

背景: building_shp来自LINZ building outlines数据集, 该数据集带时间版本,
同一栋楼可能有多条历史记录(end_lifesp字段有值=已失效)和一条当前记录
(end_lifesp为空=当前有效), 需要先过滤掉历史版本, 否则楼层估算会把同一栋
物理建筑重复计算。

输入: DSM, DEM, building footprint矢量(可能含重复历史版本)
输出: (1) 过滤重复后, 带bldg_h_m和num_floors两个新字段的building shapefile
      (2) nDSM栅格文件, 供后续步骤复用
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from pathlib import Path
from tqdm import tqdm


def filter_current_buildings(buildings: gpd.GeoDataFrame,
                              end_lifespan_field: str = "end_lifesp") -> gpd.GeoDataFrame:
    """
    过滤building shapefile里的历史版本, 只保留当前有效的记录。

    LINZ building outlines数据集里, end_lifespan字段有值代表这条记录
    已经被更新的捕获记录取代(历史版本), 为空(NaT/None)代表当前仍然有效。
    """
    if end_lifespan_field not in buildings.columns:
        print(f"   警告: 找不到字段 '{end_lifespan_field}', 跳过过滤, 直接使用全部记录")
        return buildings

    n_before = len(buildings)
    current = buildings[buildings[end_lifespan_field].isna()].copy()
    n_after = len(current)
    n_removed = n_before - n_after

    print(f"   过滤前: {n_before} 条记录")
    print(f"   移除历史版本(end_lifesp有值): {n_removed} 条")
    print(f"   过滤后(当前有效): {n_after} 条")

    current = current.reset_index(drop=True)
    return current


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
        floor_height_m: float = 4.0,
        height_percentile: float = 90,
        end_lifespan_field: str = "end_lifesp",
):
    """
    参数:
        dsm_path: 输入DSM栅格路径
        dem_path: 输入DEM栅格路径
        building_shp: 输入building footprint矢量路径(可能含重复历史版本)
        output_building_shp: 输出路径, 过滤+带bldg_h_m和num_floors字段的building shapefile
        output_ndsm_path: 可选, nDSM栅格的输出路径。传None则不保存。
        floor_height_m: 假设的每层楼高度(米), 默认3米
        height_percentile: 计算建筑高度时用的百分位数, 默认90
        end_lifespan_field: LINZ数据集里标记历史版本的字段名, 默认"end_lifesp"
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

    print("2. 加载building矢量, 过滤历史版本...")
    buildings_raw = gpd.read_file(building_shp)
    buildings = filter_current_buildings(buildings_raw, end_lifespan_field=end_lifespan_field)

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

    print(f"🎉 完成! 共处理 {len(buildings)} 栋建筑(已过滤重复历史版本), "
          f"层数范围 {buildings['num_floors'].min()}-{buildings['num_floors'].max()}, "
          f"平均 {buildings['num_floors'].mean():.1f} 层")

    return buildings


if __name__ == "__main__":
    generate_ndsm_and_floors(
        dsm_path=Path(r"C:\Users\xhe40\Thesis_Data\Campus\test_dsm_Afterfill.tif"),
        dem_path=Path(r"C:\Users\xhe40\Thesis_Data\Campus\test_dem_1m.tif"),
        building_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\Building_Campus.shp"),
        output_building_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\Building_Campus_with_floors_4.shp"),
        output_ndsm_path=Path(r"C:\Users\xhe40\Thesis_Data\Campus\ndsm_1m.tif"),
        floor_height_m=4.0,
        height_percentile=90,
    )