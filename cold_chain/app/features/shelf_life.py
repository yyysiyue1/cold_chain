# app/features/shelf_life.py
from __future__ import annotations
import pandas as pd
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
