"""
Process 步骤:核心算法 —— 调用Cimburova方法(r.viewshed.exposure),
逐棵树累加viewshed贡献,最后聚合到每栋building。

设计:
  1. 维护一张"累加图" total_exposure,初始值全部是0
  2. 对每一棵树(individual_trees.shp里的每个要素):
     - 以这棵树为起点,用DSM算它的viewshed(用Cimburova验证过的r.viewshed.exposure)
     - 排除树自身像素
     - 把这张图加进 total_exposure 累加图里
  3. 全部树算完之后,在每栋building的位置上,查一次 total_exposure 的值
     —— 这个值就是"够得着这栋楼的所有树,贡献值加总"

不需要给每棵树单独存一张图(那样要存1505张,浪费空间也慢),
边算边加总,只需要维护一张累加图,效率高很多。

这一步依赖GRASS(通过grass.script调用,不需要打开GRASS图形界面)。
"""

import os
from pathlib import Path

# ==================== 环境变量硬核注入 (必须在最前面) ====
CONDA_PREFIX = r"C:\ProgramData\anaconda3\envs\myenv_campus"

os.environ["GDAL_DATA"] = os.path.join(CONDA_PREFIX, r"Library\share\gdal")
os.environ["PROJ_LIB"] = os.path.join(CONDA_PREFIX, r"Library\share\proj")

# 核心：将 conda 环境的 Library bin 目录加入系统 PATH，彻底治愈 DLL 找不到的 3221225781 崩溃
conda_bins = [
    os.path.join(CONDA_PREFIX, r"Library\bin"),
    os.path.join(CONDA_PREFIX, r"Library\lib"),
    os.path.join(CONDA_PREFIX, r"bin"),
]
current_path = os.environ.get("PATH", "")
os.environ["PATH"] = os.pathsep.join(conda_bins) + os.pathsep + current_path
# ========================================================

import sys
GRASS_PYTHON_PATH = os.path.join(CONDA_PREFIX, r"Library\lib\grass85\etc\python")
sys.path.append(GRASS_PYTHON_PATH)

import grass.script as gs
import grass.script.setup as gsetup
from grass_session import Session

import os
os.environ["GDAL_DATA"] = r"C:\ProgramData\anaconda3\envs\myenv_campus\Library\share\gdal"
os.environ["PROJ_LIB"] = r"C:\ProgramData\anaconda3\envs\myenv_campus\Library\share\proj"

import sys
from pathlib import Path

# TODO: 确认这个路径跟你环境里 `grass --config python_path` 输出的一致
GRASS_PYTHON_PATH = r"C:\ProgramData\anaconda3\envs\myenv_campus\Library\lib\grass85\etc\python"
sys.path.append(GRASS_PYTHON_PATH)

import grass.script as gs
from grass_session import Session

def run_viewshed_aggregate(
    dsm_path: Path,
    tree_crown_shp: Path,
    building_shp: Path,
    output_shp: Path,
    grass_workdir: Path,
    viewshed_range_m: float = 100.0,
    observer_height_m: float = 1.5,
    epsg: str = "2193",
) -> Path:
    """
    参数:
        dsm_path: preprocess生成的DSM(已经补洞、裁剪成meshblock形状那一版)
        tree_crown_shp: 个体树冠矢量图(每个要素一棵独立的树)
        building_shp: building footprint,最终结果写回这个矢量图的属性表
        output_shp: 输出的building shapefile,带一列 green_view_score
        grass_workdir: GRASS需要一个工作目录来建立"project"(存放临时的GRASS数据库),
                       随便指定一个空文件夹路径即可,比如 data/interim/grass_workdir
        viewshed_range_m: 每棵树的viewshed最大搜索半径,单位米
        observer_height_m: 观察者(树)的默认高度参数,这里其实是viewshed里对"高度"的技术参数,
                            不是人的身高 —— 因为起点是树,这个参数影响不大,先用默认值
        epsg: 坐标系代码,NZTM2000是2193

    返回:
        output_shp
    """
    grass_workdir.mkdir(parents=True, exist_ok=True)
    project_path = grass_workdir / "pilot"
    permanent_path = project_path / "PERMANENT"

    # 核心修复：在 Windows 下手动创建 GRASS Location 所需的基础目录结构
    permanent_path.mkdir(parents=True, exist_ok=True)

    # 如果是全新创建，写入一个最小合法的 DEFAULT_WIND 文件（GRASS 的地图范围配置文件）
    default_wind = permanent_path / "DEFAULT_WIND"
    if not default_wind.exists():
        with open(default_wind, "w") as f:
            f.write(
                "proj:       1\nzone:       0\nnorth:      1\nsouth:      0\neast:       1\nwest:       0\ncols:       1\nrows:       1\ne-w resol:  1\nn-s resol:  1\ntop:        1\nbottom:     0\nd-b resol:  1\n")

    session = Session()
    # 改为直接打开已存在的路径，避开会报错的 grass.BAT -c 自动创建指令
    session.open(gisdb=str(grass_workdir), location="pilot")
    try:
        # 2. 导入DSM,设置计算区域跟DSM完全对齐
        gs.run_command("r.in.gdal", input=str(dsm_path), output="dsm", overwrite=True, quiet=True)
        gs.run_command("g.region", raster="dsm")

        # 3. 导入树冠矢量图
        gs.run_command("v.in.ogr", input=str(tree_crown_shp), output="trees", overwrite=True, quiet=True)

        # 4. 初始化累加图,全部像素先设为0
        gs.mapcalc("total_exposure = 0", overwrite=True)

        # 5. 拿到所有树的cat(唯一ID)列表
        cats_raw = gs.read_command("v.category", input="trees", option="print").strip()
        tree_cats = sorted(set(cats_raw.split("\n"))) if cats_raw else []

        print(f"开始处理 {len(tree_cats)} 棵树...")

        for i, cat in enumerate(tree_cats, start=1):
            _accumulate_one_tree(cat, viewshed_range_m, observer_height_m)
            if i % 100 == 0:
                print(f"  已处理 {i}/{len(tree_cats)} 棵树")

        print("所有树处理完成,开始聚合到building...")

        # 6. 导入building footprint
        gs.run_command("v.in.ogr", input=str(building_shp), output="buildings", overwrite=True, quiet=True)

        # 7. 在每栋building的位置上,查询累加图的值,写入新的一列
        gs.run_command(
            "v.db.addcolumn", map="buildings", columns="green_view_score double precision"
        )
        gs.run_command(
            "v.what.rast", map="buildings", raster="total_exposure", column="green_view_score"
        )

        # 8. 导出成shapefile
        output_shp.parent.mkdir(parents=True, exist_ok=True)
        gs.run_command(
            "v.out.ogr",
            input="buildings",
            output=str(output_shp),
            format="ESRI_Shapefile",
            overwrite=True,
            quiet=True,
        )

        print(f"完成,写入 {output_shp}")

    finally:
        session.close()

    return output_shp


def _accumulate_one_tree(cat: str, viewshed_range_m: float, observer_height_m: float) -> None:
    """处理单独一棵树:算它的viewshed,排除自身像素,加进累加图。"""
    tag = f"t{cat}"

    # 只取出这一棵树
    gs.run_command("v.extract", input="trees", cats=cat, output=f"tree_{tag}", overwrite=True, quiet=True)

    # 栅格化这棵树(用于后面排除自身像素)
    gs.run_command(
        "v.to.rast", input=f"tree_{tag}", output=f"rast_{tag}",
        use="val", value=1, overwrite=True, quiet=True,
    )

    # 撒采样点近似树冠形状(树是polygon,需要多个点代表整个树冠)
    gs.run_command(
        "r.random", input=f"rast_{tag}", vector=f"pts_{tag}",
        npoints="25%", overwrite=True, quiet=True,
    )

    # 核心:调用Cimburova验证过的viewshed引擎
    gs.run_command(
        "r.viewshed.exposure",
        input="dsm", output=f"exp_{tag}",
        sampling_points=f"pts_{tag}",
        observer_elevation=observer_height_m,
        range=viewshed_range_m,
        function="Distance_decay",
        overwrite=True, quiet=True,
    )

    # 排除树自身像素,再加进累加图(用temp变量避免自我引用的mapcalc问题)
    gs.mapcalc(
        f"clean_{tag} = if(isnull(rast_{tag}), if(isnull(exp_{tag}), 0, exp_{tag}), 0)",
        overwrite=True, quiet=True,
    )
    gs.mapcalc(
        f"total_exposure_new = total_exposure + clean_{tag}",
        overwrite=True, quiet=True,
    )
    gs.run_command("g.rename", raster="total_exposure_new,total_exposure", overwrite=True, quiet=True)

    # 清理这棵树的临时图层,不要让1505棵树的中间产物把硬盘占满
    gs.run_command(
        "g.remove", type="raster", pattern=f"*_{tag}", flags="f", quiet=True,
    )
    gs.run_command(
        "g.remove", type="vector", pattern=f"*_{tag}", flags="f", quiet=True,
    )