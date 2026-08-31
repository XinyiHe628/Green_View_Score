from pathlib import Path
from preprocess.pre02_generate_dem import generate_dem_from_laz

generate_dem_from_laz(
    laz_path=Path(r"C:\Users\xhe40\Thesis_Data\Campus\test_merged.laz"),
    output_tif_path=Path(r"C:\Users\xhe40\Thesis_Data\Campus\test_dem_1m.tif"),
    resolution=1.0,
)