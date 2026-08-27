from pathlib import Path
from process.viewshed_aggregate import run_viewshed_aggregate

run_viewshed_aggregate(
    dsm_path=Path(r"C:\Users\xhe40\Thesis_Data\Campus\test_dsm_Afterfill.tif"),
    tree_crown_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\individual_trees.shp"),
    building_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\Building_Campus.shp"),
    output_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\building_result.shp"),
    viewshed_range_m=100.0,
)