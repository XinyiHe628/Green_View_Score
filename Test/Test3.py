from pathlib import Path
from preprocess.generate_canopy import generate_canopy


RAW_DIR = Path(r"C:\Users\xhe40\Thesis_Data\Campus")
# 替换成你本地的实际路径
las_file = RAW_DIR / "test_merged.laz"    # 或 data/interim 下剪好的 laz
dsm_file = RAW_DIR / "test_dsm_Afterfill.tif"   # 步骤2生成的 DSM
out_file = RAW_DIR / " test_canopy_clean.tif"

# 执行生成
generate_canopy(
    las_path=las_file,
    dsm_path=dsm_file,
    output_tif=out_file,
    height_threshold=2.0,  # 高于 2 米算树冠
    ground_class=2,       # 地面点
    veg_classes=[4, 5]     # 4=中等植被, 5=高大树木
)