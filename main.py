"""
主控脚本 —— 按顺序跑完整个pipeline:
  Preprocess (合并裁剪LiDAR -> 生成DSM -> 生成Canopy)
  -> Process (viewshed计算 + 聚合到building)
  -> Afterprocess (验证结果)

具体每一步的实现,都在对应的模块文件里,这里只负责按顺序调用、传参数。
路径和参数统一从 config.py 读,不要在这里写死。
"""

import config
from preprocess.pre01_merge_clip_Lidar import merge_and_clip_lidar
from preprocess.pre03_generate_dsm import generate_dsm
from preprocess.pre04_generate_canopy import generate_canopy
from process.viewshed_aggregate import run_viewshed_aggregate
from postprocess.validation import validate_results


def main():
    # ---------- Preprocess ----------
    print("=== Preprocess 1/3: 合并 + 裁剪 LiDAR ===")
    merged_las = merge_and_clip_lidar(
        tile_paths=config.LIDAR_TILES,
        boundary_shp=config.MESHBLOCK_SHP,
        output_path=config.MERGED_CLIPPED_LAS,
        buffer_m=config.STUDY_AREA_BUFFER_M,
    )

    print("=== Preprocess 2/3: 生成 DSM ===")
    dsm_path = generate_dsm(
        las_path=merged_las,
        resolution=config.RASTER_RESOLUTION_M,
        output_tif=config.DSM_PATH,
    )

    print("=== Preprocess 3/3: 生成 Canopy ===")
    canopy_path = generate_canopy(
        las_path=merged_las,
        dsm_path=dsm_path,
        output_tif=config.CANOPY_PATH,
        height_threshold=config.CANOPY_HEIGHT_THRESHOLD_M,
        ground_class=config.LAS_CLASS_GROUND,
    )

    # ---------- Process ----------
    print("=== Process: viewshed 计算 + 聚合到 building ===")
    result_shp = run_viewshed_aggregate(
        dsm_path=dsm_path,
        canopy_path=canopy_path,
        building_shp=config.BUILDING_SHP,
        output_shp=config.OUTPUT_DIR / "building_green_view_score.shp",
    )

    # ---------- Afterprocess ----------
    #print("=== Afterprocess: 验证结果 ===")
    #validate_results(result_shp)

    print("=== 全部完成 ===")


if __name__ == "__main__":
    main()