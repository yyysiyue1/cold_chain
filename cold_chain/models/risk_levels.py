# models/risk_levels.py
from __future__ import annotations
import pandas as pd
from typing import Dict, Callable, Optional, Tuple, Union
# ==============================================================================
# ---标志物 风险等级判断---
# ==============================================================================
# flag -> 表名（同时兼容中文/英文入参）
_TABLE_MAP = {
    "chem": "chemicalmarker_risk_level",
    "化学": "chemicalmarker_risk_level",
    "bio":  "biomarker_risk_level",
    "生物":  "biomarker_risk_level",
}
_NEEDED_COLS = ["FoodClassificationCode", "LowRiskLine", "MiddleRiskLine", "HighRiskLine"]
# 惰性缓存：key=规范化flag('chem'/'bio') -> DataFrame(index=FoodClassificationCode)
_threshold_cache: Dict[str, pd.DataFrame] = {}

def _norm_flag(flag: str) -> str:
    if flag in ("chem", "化学"): return "chem"
    if flag in ("bio", "生物"):  return "bio"
    raise ValueError(f"flag 只能是 'chem'/'化学' 或 'bio'/'生物'，收到：{flag}")
#加载阈值缓存
def load_thresholds(
    engine,
    flag: str,
    refresh: bool = False,
    read_sql: Callable[[str, Union[object, None]], pd.DataFrame] = pd.read_sql,
) -> pd.DataFrame:
    """
     从数据库把对应表的阈值读入缓存；返回 DataFrame（index=FoodClassificationCode）
     """

    nflag = _norm_flag(flag)
    if (not refresh) and (nflag in _threshold_cache):
        return _threshold_cache[nflag]
    table = _TABLE_MAP[nflag]
    sql = f"SELECT {', '.join(_NEEDED_COLS)} FROM {table}"
    df = read_sql(sql, engine).copy()
    df = df[_NEEDED_COLS].drop_duplicates(subset=["FoodClassificationCode"])
    df.set_index("FoodClassificationCode", inplace=True)
    for c in ("LowRiskLine","MiddleRiskLine","HighRiskLine"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    _threshold_cache[nflag] = df
    return df

def get_thresholds(food_code: str, flag: str, engine, refresh: bool = False
) -> Optional[Tuple[float, float, float]]:
    """
    返回 (low, mid, high)；没有或缺失则返回 None
    """
    df = load_thresholds(engine, flag, refresh=refresh)
    if food_code not in df.index: return None
    row = df.loc[food_code]
    low, mid, high = row["LowRiskLine"], row["MiddleRiskLine"], row["HighRiskLine"]
    if pd.isna(low) or pd.isna(mid) or pd.isna(high): return None
    return float(low), float(mid), float(high)

def determine_risk_level(
    food_code: str, value: float | None, flag: str, engine, *, return_detail: bool = False
):
    """
    通用风险等级判断（适用于化学/生物的任意标志物）：
    规则：value < low → '无'；[low, mid) → '低'；[mid, high) → '中'；>= high → '高'
    value 为 None 返回 '无'；找不到阈值返回 '未知'
    """
    if value is None: return ("无", None) if return_detail else "无"
    th = get_thresholds(food_code, flag, engine)
    if th is None: return ("未知", None) if return_detail else "未知"
    low, mid, high = th
    lvl = "无" if value < low else ("低" if value < mid else ("中" if value < high else "高"))
    return (lvl, th) if return_detail else lvl
