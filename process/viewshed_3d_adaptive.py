import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import rasterize
from skimage.draw import line
from pathlib import Path
from tqdm import tqdm  # 用于显示进度条


def calculate_3d_green_view_score(
        dsm_path: Path,
        dem_path: Path,
        tree_crown_shp: Path,
        building_shp: Path,
        output_shp: Path,
        viewshed_range_m: float = 100.0,
        floor_heights: list = [1.5, 4.5, 7.5],  # 1楼, 2楼, 3楼的视线高度
        trunk_clear_height: float = 2.5  # 树干底部的透空高度(米)
):
    print("1. 正在加载高程矩阵并动态计算 nDSM...")
    with rasterio.open(dsm_path) as src_dsm, rasterio.open(dem_path) as src_dem:
        # 确保基础元数据一致
        transform = src_dsm.transform
        res = transform.a

        # 提取全局矩阵
        dsm_array = src_dsm.read(1)
        dem_array = src_dem.read(1)

        # 动态生成 nDSM (相对高程)，并用 clip 抹平插值造成的微小负数误差
        ndsm_array = np.clip(dsm_array - dem_array, 0, None)

        print("2. 正在加载矢量要素并生成三维植被掩膜...")
        trees = gpd.read_file(tree_crown_shp)
        buildings = gpd.read_file(building_shp)

        # 将树冠矢量烧录成栅格掩膜，1代表树木，0代表建筑/其他
        tree_shapes = ((geom, 1) for geom in trees.geometry)
        is_tree_mask = rasterize(
            tree_shapes,
            out_shape=dsm_array.shape,
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

    print("3. 开始执行 3D 射线追踪 (Ray-casting) 计算各楼层绿视率...")
    # 为每一层初始化得分字段
    for h in floor_heights:
        buildings[f"GVS_H{h}"] = 0.0

    # 遍历每一个建筑物（起点）
    for idx, row in tqdm(buildings.iterrows(), total=len(buildings), desc="Building Progress"):
        centroid = row.geometry.centroid
        x0, y0 = centroid.x, centroid.y

        # 获取起点的行列号
        c0, r0 = ~transform * (x0, y0)
        c0, r0 = int(c0), int(r0)

        # 检查起点是否在矩阵内
        if not (0 <= r0 < dsm_array.shape[0] and 0 <= c0 < dsm_array.shape[1]):
            continue

        base_elevation = dem_array[r0, c0]
        if np.isnan(base_elevation) or base_elevation == -9999.0:
            continue

        # 确定 100 米缓冲区的局部矩阵边界，减少全局搜索以大幅提升速度
        pixel_range = int(viewshed_range_m / res)
        r_min, r_max = max(0, r0 - pixel_range), min(dsm_array.shape[0], r0 + pixel_range + 1)
        c_min, c_max = max(0, c0 - pixel_range), min(dsm_array.shape[1], c0 + pixel_range + 1)

        # 局部矩阵起点在全图的相对坐标
        lr0, lc0 = r0 - r_min, c0 - c_min

        # 切片提取局部数据
        loc_dsm = dsm_array[r_min:r_max, c_min:c_max]
        loc_dem = dem_array[r_min:r_max, c_min:c_max]
        loc_tree = is_tree_mask[r_min:r_max, c_min:c_max]

        # 寻找缓冲区内所有的“树冠像素”作为视线目标
        tree_pixels = np.argwhere(loc_tree == 1)

        # 遍历每一层楼计算独立暴露度
        for h in floor_heights:
            z_observer = base_elevation + h
            score = 0.0

            # 向局部范围内的每一棵树发射 3D 射线
            for tr, tc in tree_pixels:
                dist_pixels = np.sqrt((tr - lr0) ** 2 + (tc - lc0) ** 2)
                dist_m = dist_pixels * res

                # 排除超出 100 米圆圈的像素和观察点本身
                if dist_m > viewshed_range_m or dist_m == 0:
                    continue

                z_target = loc_dsm[tr, tc]

                # 提取射线经过的所有像素坐标 (Bresenham 画线算法)
                rr, cc = line(lr0, lc0, tr, tc)

                # 剔除起点(人)和终点(树顶)，只判定中间途经的像素是否遮挡
                if len(rr) > 2:
                    rr, cc = rr[1:-1], cc[1:-1]

                    # 相似三角形计算射线在每一个像素点上的绝对高度
                    dists_along_ray = np.sqrt((rr - lr0) ** 2 + (cc - lc0) ** 2) * res
                    z_ray = z_observer + (z_target - z_observer) * (dists_along_ray / dist_m)

                    # 提取沿线地形和地物
                    ray_dsm = loc_dsm[rr, cc]
                    ray_dem = loc_dem[rr, cc]
                    ray_is_tree = loc_tree[rr, cc]

                    # ====== 核心伪 3D 碰撞检测逻辑 ======
                    blocked = False
                    for i in range(len(z_ray)):
                        zr = z_ray[i]
                        dsm_val = ray_dsm[i]
                        dem_val = ray_dem[i]

                        if ray_is_tree[i] == 1:
                            # 途径树木：存在树干透空区 (Trunk Pass-through)
                            z_trunk_top = dem_val + trunk_clear_height
                            if z_trunk_top <= zr <= dsm_val:
                                blocked = True
                                break
                        else:
                            # 途径建筑：实心阻挡 (Solid Block)
                            if zr <= dsm_val:
                                blocked = True
                                break

                    if blocked:
                        continue  # 视线被挡死，该树不计分

                # 如果代码走到这里，说明光线成功穿越所有障碍物到达树冠
                score += (1.0 / (dist_m ** 2))

            # 将该楼层得分写入矢量属性表
            buildings.loc[idx, f"GVS_H{h}"] = score

    print(f"4. 正在导出分析结果至: {output_shp}")
    buildings.to_file(output_shp)
    print("🎉 三维动态楼层绿视率计算完成！")


# ==================== 测试入口 ====================
if __name__ == "__main__":
    calculate_3d_green_view_score(
        dsm_path=Path(r"C:\Users\xhe40\Thesis_Data\Campus\test_dsm_Afterfill.tif"),
        dem_path=Path(r"C:\Users\xhe40\Thesis_Data\Campus\test_dem_1m.tif"),
        tree_crown_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\individual_trees.shp"),
        building_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\Building_Campus.shp"),
        output_shp=Path(r"C:\Users\xhe40\Thesis_Data\Campus\building_result_3D.shp"),
        viewshed_range_m=100.0,
        floor_heights=[1.5, 4.5, 7.5],  # 自定义你的楼层高度
        trunk_clear_height=2.5  # 低于 2.5 米的视线可穿透树干
    )