from pathlib import Path
from preprocess.merge_clip_Lidar import merge_and_clip_lidar

merge_and_clip_lidar(
    tile_paths=[
        Path(r"C:\Users\xhe40\Thesis_Data\Campus\Tile_Left.laz"),
        Path(r"C:\Users\xhe40\Thesis_Data\Campus\Tile_Right.laz"),
    ],
    boundary_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\Meshblock_Campus.shp"),
    output_path=Path(r"C:\Users\xhe40\Thesis_Data\Campus\test_merged.laz"),
    buffer_m=100.0,
)