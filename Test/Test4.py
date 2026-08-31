from pathlib import Path
from preprocess.pre05_generate_individual_trees import generate_individual_trees

RAW_DIR = Path(r"C:\Users\xhe40\Thesis_Data\Campus")

las_file = RAW_DIR / "test_merged.laz"
dsm_file = RAW_DIR / "test_dsm_Afterfill.tif"
canopy_file = RAW_DIR / " test_canopy_clean.tif"
out_shp = RAW_DIR / "individual_trees.shp"  # 修正：去掉了空格

print(f"正在读取 Canopy: {canopy_file} | 是否存在: {canopy_file.exists()}")

generate_individual_trees(
    dsm_path=dsm_file,
    canopy_path=canopy_file,
    output_shp=out_shp,
    min_distance_m=3.0
)