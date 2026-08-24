"""
Process 步骤:核心算法 —— 树的viewshed计算 + 聚合到building。

这一步具体用什么引擎(GRASS/WhiteboxTools/纯numpy自己写)还没最终定,
先把接口定清楚,main.py先调用这个函数,内部实现之后再填。

设计思路(跟之前讨论过的一致):
1. 以每棵树为起点,算它的viewshed范围(距离衰减/binary,具体先用哪种还没定)
2. 每棵树的viewshed结果,在每栋building的位置上取值
3. 对每栋building,把所有"够得着它"的树的值加总,得到这栋building的绿视分数
"""

from pathlib import Path
import geopandas as gpd


def run_viewshed_aggregate(
    dsm_path: Path,
    canopy_path: Path,
    building_shp: Path,
    output_shp: Path,
) -> Path:
    """
    参数:
        dsm_path: preprocess生成的DSM,用作视线遮挡面
        canopy_path: preprocess生成的canopy栅格,需要先转成独立的树冠polygon(每棵树一个要素)
                     才能"以每棵树为起点"迭代 —— 这一步的树冠分割逻辑还没写
        building_shp: building footprint,聚合结果最终写回这个矢量图的属性表
        output_shp: 输出的building shapefile,带一列绿视分数

    返回:
        output_shp

    TODO:
        1. canopy栅格 -> 独立树冠polygon的分割逻辑(连通域分析,比如scipy.ndimage.label
           或者skimage的watershed,如果树冠连成一片需要更细致的分割)
        2. viewshed引擎选型(GRASS的r.viewshed.exposure / WhiteboxTools / 自己写)
        3. 取值+聚合逻辑(在每栋building位置,查询每棵树viewshed图里的值并求和)
    """
    raise NotImplementedError(
        "process步骤的具体算法还没确定,先跑通preprocess这三步,"
        "等DSM和canopy质量确认没问题,再回来实现这里。"
    )