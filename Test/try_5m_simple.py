"""
用一串沿墙面等距分布的observer点代替原来"一栋building一个点"的做法。
两步: 1. simplify去掉琐碎节点(保留L形/U形整体形状)  2. 沿简化后的轮廓每隔interval_m米取一点, 往外偏移wall_offset_m米
"""

import numpy as np
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point


def get_observer_points_along_wall(
        building_geom,
        tolerance_m: float = 1.0,
        interval_m: float = 5.0,
        wall_offset_m: float = 0.5,
):
    geom = building_geom
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)

    simplified = geom.simplify(tolerance_m, preserve_topology=True)
    boundary = simplified.exterior
    length = boundary.length
    centroid = simplified.centroid

    if length == 0:
        return [centroid]

    n_steps = max(1, int(length // interval_m))
    eps = min(0.5, interval_m / 10.0)

    points = []
    for i in range(n_steps):
        d = (i * interval_m) % length
        p = boundary.interpolate(d)

        d_next = (d + eps) % length
        d_prev = (d - eps) % length
        p_next = boundary.interpolate(d_next)
        p_prev = boundary.interpolate(d_prev)

        tangent = np.array([p_next.x - p_prev.x, p_next.y - p_prev.y])
        norm = np.linalg.norm(tangent)
        if norm == 0:
            continue
        tangent = tangent / norm

        normal = np.array([-tangent[1], tangent[0]])
        to_p = np.array([p.x - centroid.x, p.y - centroid.y])
        if np.dot(normal, to_p) < 0:
            normal = -normal

        offset_pt = Point(
            p.x + normal[0] * wall_offset_m,
            p.y + normal[1] * wall_offset_m,
        )
        points.append(offset_pt)

    return points


def generate_all_observer_points(
        building_shp,
        output_points_shp,
        tolerance_m: float = 1.0,
        interval_m: float = 5.0,
        wall_offset_m: float = 0.5,
        building_id_field: str = None,
):
    buildings = gpd.read_file(building_shp)
    crs = buildings.crs

    records = []
    for b_idx, row in buildings.iterrows():
        bid = row[building_id_field] if building_id_field and building_id_field in buildings.columns else b_idx
        pts = get_observer_points_along_wall(
            row.geometry,
            tolerance_m=tolerance_m,
            interval_m=interval_m,
            wall_offset_m=wall_offset_m,
        )
        for pt_idx, pt in enumerate(pts):
            records.append({
                "building_idx": b_idx,
                "building_id": bid,
                "point_idx": pt_idx,
                "n_points": len(pts),
                "geometry": pt,
            })

    points_gdf = gpd.GeoDataFrame(records, crs=crs)
    output_points_shp.parent.mkdir(parents=True, exist_ok=True)
    points_gdf.to_file(output_points_shp)

    print(f"共 {len(buildings)} 栋building, 生成 {len(points_gdf)} 个observer点")
    print(f"平均每栋 {len(points_gdf) / len(buildings):.1f} 个点")
    print(f"结果写入: {output_points_shp}")

    return output_points_shp


if __name__ == "__main__":
    generate_all_observer_points(
        building_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\Building_Campus_with_floors_4.shp"),
        output_points_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\observer_points_along_wall_test_7.shp"),
        tolerance_m=7.0,
        interval_m=5.0,
        wall_offset_m=0.5,
    )