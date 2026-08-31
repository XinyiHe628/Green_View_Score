from pathlib import Path
from preprocess.pre03_generate_dsm import generate_dsm
import rasterio
import numpy as np


generate_dsm(
    las_path=Path(r"C:\Users\xhe40\Thesis_Data\Campus\test_merged.laz"),
    resolution=1.0,
    output_tif=Path(r"C:\Users\xhe40\Thesis_Data\Campus\test_dsm_Afterfill.tif"),
    boundary_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\Meshblock_Campus.shp"),
    buffer_m=100.0,
)


