"""
Afterprocess 步骤:验证process算出来的结果靠不靠谱。

具体验证方式还没定,先占位。可能的方向(参考你之前workflow图里Step 8):
- 挑几个building,人工去实地/用街景照片确认绿视水平跟算出来的分数是不是趋势一致
- 跟Cimburova论文里的做法类似,对比某几个点位算出来的数值和肉眼判断是否吻合
"""

from pathlib import Path


def validate_results(result_shp: Path) -> None:
    """
    参数:
        result_shp: process步骤输出的、带绿视分数的building shapefile

    TODO: 具体验证方法待定(实地/街景照片抽样对照,或者别的方式)
    """
    raise NotImplementedError("验证方法还没定,等process步骤跑通、有实际结果之后再设计这一步。")