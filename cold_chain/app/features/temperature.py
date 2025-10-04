# app/features/temperature.py
from __future__ import annotations
import pandas as pd
from datetime import datetime
from sqlalchemy import text
from app.repos.orders import find_previous_monitor_time, find_previous_abnormal_value
# 温度异常判断函数
def is_temp_abnormal(temp, lower, upper) -> bool:
    try:
        if pd.isna(temp): return False
        t = float(temp)
        lo = float(lower) if pd.notna(lower) else None
        hi = float(upper) if pd.notna(upper) else None
    except Exception:
        return False
    if lo is not None and t < lo: return True
    if hi is not None and t > hi: return True
    return False
# 计算温度异常时长
def calculate_temp_predict_value(temp_abnormal, order_number, rec_time, tra_code, order_tra_chain, engine) -> float:
    if not isinstance(rec_time, datetime):
        rec_time = pd.to_datetime(rec_time, errors='coerce')
        if pd.isna(rec_time): return 0.0
    last_time = find_previous_monitor_time(order_number, rec_time, tra_code, order_tra_chain, engine)
    last_val  = find_previous_abnormal_value(order_number, rec_time, tra_code, order_tra_chain, engine, flag="温度")
    if not temp_abnormal:
        return float(last_val) if pd.notna(last_val) else 0.0
    if last_time is None or pd.isna(last_time):
        return float(last_val) if pd.notna(last_val) else 0.0
    try:
        dt_sec = (rec_time - last_time).total_seconds()
        inc_min = 0.0 if dt_sec < 0 else dt_sec / 60.0
        return float(inc_min + (last_val or 0.0))
    except Exception:
        return float(last_val) if pd.notna(last_val) else 0.0
# 温度风险等级判断
def is_exceeding_temp_threshold(food_class_code, temp_predict_value, engine, *, temp_abnormal=True) -> bool:
    if pd.isna(food_class_code) or pd.isna(temp_predict_value) or not temp_abnormal or temp_predict_value <= 0:
        return False
    try:
        sql = """
        SELECT TempAnonalyDuration
        FROM food_knowledge
        WHERE FoodClassificationCode = :code
        LIMIT 1
        """
        with engine.connect() as conn:
            hours = conn.execute(text(sql), {"code": food_class_code}).scalar()
        if hours is None: return False
        return temp_predict_value > float(hours) * 60.0
    except Exception:
        return False
