# app/features/shelf_life.py
from __future__ import annotations
import pandas as pd
#保质期单位转换
def _to_days(value, unit):
    """把保质期按单位统一换算成天；未知单位时原样返回整数天。"""
    if value is None:
        return None
    unit = (unit or "").strip().lower()
    v = float(value)
    if unit in ("d", "day", "days", "天", "日"):
        return int(round(v))
    if unit in ("m", "mon", "month", "months", "月"):
        return int(round(v * 30))
    if unit in ("y", "yr", "year", "years", "年"):
        return int(round(v * 365))
    return int(round(v))  # 单位缺失/未知，按“天”处理
# 过期异常判断
def is_shelf_life_abnormal(pro_date, rec_time, shelf_life) -> bool:
    try:
        if pd.isna(pro_date) or pd.isna(rec_time) or pd.isna(shelf_life):
            return False
        pro = pd.to_datetime(pro_date, errors='coerce')
        rec = pd.to_datetime(rec_time, errors='coerce')
        days = int(shelf_life)
        if pd.isna(pro) or pd.isna(rec): return False
        return (rec - pro).days - days > 0
    except Exception:
        return False
# 过期异常时长计算
def calculate_shelf_life_abnormal_duration(pro_date, rec_time, shelf_life):
    try:
        if pd.isna(pro_date) or pd.isna(rec_time): return None
        pro = pd.to_datetime(pro_date, errors='coerce')
        rec = pd.to_datetime(rec_time, errors='coerce')
        days = None
        if not pd.isna(shelf_life):
            try: days = int(shelf_life)
            except (ValueError, TypeError): pass
        if pd.isna(pro) or pd.isna(rec) or days is None: return None
        return (rec - pro).days - days
    except Exception:
        return None
