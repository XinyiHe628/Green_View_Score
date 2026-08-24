from pathlib import Path
from preprocess.generate_dsm import generate_dsm

generate_dsm(
    las_path=Path(r"C:\Users\xhe40\Thesis_Data\Campus\test_merged.laz"),
    resolution=1.0,
    output_tif=Path(r"C:\Users\xhe40\Thesis_Data\Campus\test_dsm.tif"),
)