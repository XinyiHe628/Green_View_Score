"""
Preprocess 步骤1:合并多块LiDAR Tile,裁剪到study area范围。

输入:多个.laz/.las文件(可能互不重叠,也可能有重叠区域) + 一个study area边界(shapefile)
输出:一个合并、裁剪好的.laz文件,后面DSM/Canopy生成都基于这一个文件

这一步不依赖GRASS,只用laspy(读写点云)+ geopandas/shapely(处理边界多边形)。
"""

import laspy
import numpy as np
import geopandas as gpd
from shapely import vectorized as shapely_vectorized
from pathlib import Path


def merge_and_clip_lidar(
    tile_paths: list[Path],
    boundary_shp: Path,
    output_path: Path,
    buffer_m: float = 100.0,
) -> Path:
    """
    合并多个LiDAR tile,裁剪到boundary_shp的范围(外扩buffer_m米)。

    参数:
        tile_paths: 多个.laz/.las文件路径的列表
        boundary_shp: 定义study area的shapefile(比如你的meshblock或者building footprint)
        output_path: 合并裁剪后的输出.laz路径
        buffer_m: study area边界往外扩多少米,避免viewshed计算在边缘失真

    返回:
        output_path(方便串起来用)
    """
    # 1. 读取边界,外扩buffer,拿到一个bounding box用于快速预筛选
    boundary_gdf = gpd.read_file(boundary_shp)
    study_area = boundary_gdf.union_all() if hasattr(boundary_gdf, "union_all") else boundary_gdf.unary_union
    study_area_buffered = study_area.buffer(buffer_m)
    minx, miny, maxx, maxy = study_area_buffered.bounds

    all_points = []  # 收集所有tile里落在范围内的点

    for tile_path in tile_paths:
        with laspy.open(tile_path) as las_file:
            las = las_file.read()

            # 先用bounding box做一次快速筛选(比逐点判断多边形快很多)
            mask_bbox = (
                (las.x >= minx) & (las.x <= maxx) &
                (las.y >= miny) & (las.y <= maxy)
            )
            if not mask_bbox.any():
                # 这块tile完全不在study area范围内,跳过
                continue

            # bbox预筛选之后,再用真正的多边形做精确裁剪(bbox是矩形,study_area_buffered可能不是)
            candidate_x = las.x[mask_bbox]
            candidate_y = las.y[mask_bbox]
            mask_polygon = shapely_vectorized.contains(study_area_buffered, candidate_x, candidate_y)

            filtered = las.points[mask_bbox][mask_polygon]
            if len(filtered) == 0:
                continue
            all_points.append((las, filtered))

    if not all_points:
        raise ValueError(
            "没有任何LiDAR点落在study area范围内,检查一下boundary_shp和tile_paths是不是同一个坐标系、同一个地方"
        )

    # 2. 以第一个tile的header为模板,创建输出文件
    template_las = all_points[0][0]
    output_las = laspy.LasData(template_las.header)

    # 3. 把所有筛选后的点合并起来
    merged_points = laspy.PackedPointRecord.empty(template_las.points.point_format)
    for _, filtered in all_points:
        merged_points = laspy.PackedPointRecord(
            np.concatenate([merged_points.array, filtered.array]),
            filtered.point_format,
        )
    output_las.points = merged_points

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_las.write(output_path)

    print(f"合并裁剪完成: {len(merged_points)} 个点, 写入 {output_path}")
    return output_path


