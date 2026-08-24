"""
项目配置文件 —— 所有路径和参数集中写在这里。

以后要换pilot区域、换分辨率、换阈值,只改这个文件,
不要去preprocess/process/postprocess里面改硬编码的数字。
"""
import os
os.environ["GDAL_DATA"] = r"C:\ProgramData\anaconda3\envs\myenv_campus\Library\share\gdal"
os.environ["PROJ_LIB"] = r"C:\ProgramData\anaconda3\envs\myenv_campus\Library\share\proj"

from pathlib import Path

# ---------- 项目根目录 ----------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = Path(r"C:\Users\xhe40\Thesis_Data\Campus")   # 原始下载下来的数据,不要手动改这里面的文件
INTERIM_DIR = DATA_DIR / "interim"  # 中间产物(合并裁剪后的点云、DSM、Canopy)
OUTPUT_DIR = DATA_DIR / "output"    # 最终结果

# ---------- 输入数据 ----------
# TODO: 换成你两块LiDAR Tile的实际文件名
LIDAR_TILES = [
    RAW_DIR / "Tile_Left.laz",
    RAW_DIR / "Tile_Right.laz",
]

# TODO: 换成你的building footprint shapefile实际路径
BUILDING_SHP = RAW_DIR / "Building_Campus.shp"

# TODO: 换成你的meshblock shapefile实际路径
MESHBLOCK_SHP = RAW_DIR / "Meshblock_Campus.shp"

# ---------- 中间产物路径(一般不用改,自动生成到interim目录) ----------
MERGED_CLIPPED_LAS = INTERIM_DIR / "merged_clipped.laz"
DSM_PATH = INTERIM_DIR / "dsm.tif"
CANOPY_PATH = INTERIM_DIR / "canopy.tif"

# ---------- 参数 ----------
STUDY_AREA_BUFFER_M = 100      # 裁剪时,study area边界往外扩多少米,避免边缘效应(参照Cimburova论文)
RASTER_RESOLUTION_M = 1.0      # DSM/Canopy的栅格分辨率,论文里用的是1m
CANOPY_HEIGHT_THRESHOLD_M = 2.0  # 高于地面多少米算作"树冠",不是地面/建筑物

# LAS标准分类代码(ASPRS),用于区分地面点和其他点
# 大部分LINZ/OpenTopography的LiDAR数据自带这个分类,不需要自己重新分类
LAS_CLASS_GROUND = 2