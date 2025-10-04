# cold_chain/models/pork_gbr/logic.py
from __future__ import annotations
import pandas as pd
from typing import Dict, Any, List, Optional
from .PM_gbr_predictor import PMAPCPredictor

from app.repos.orders import (
    get_order_tra_chain, find_previous_abnormal_value,
    find_previous_monitor_time, get_storage_time
)
from models.risk_levels import RiskClassifier

RISK_FLAG = "生物"
RISK_NAME = "菌落总数"
UNIT = "CFU/mL"
SUPPORTED_FOOD_CLASSES = {"B08005"}

def _compute_gbr_value(
    row: Dict[str, Any],
    food_info: Dict[str, Any],
    engine,
    predictor: PMAPCPredictor
) -> float:
    rec_time = pd.to_datetime(row.get("RecTime"), errors="coerce")
    tra_code = row.get("TraCode")
    order_number = row.get("OrderNumber")
    curr_temp = row.get("Temp")

    # 订单链上下文
    order_tra_chain = get_order_tra_chain(order_number, tra_code, engine)
    last_time = find_previous_monitor_time(order_number, rec_time, tra_code, order_tra_chain, engine)
    last_val = find_previous_abnormal_value(
        order_number, rec_time, tra_code, order_tra_chain, engine,
        flag=RISK_FLAG, risk_name=RISK_NAME
    )

    # 初值兜底（log10 CFU/g）
    try:
        if last_val is None or pd.isna(last_val):
            last_val = 0.5
        else:
            last_val = float(last_val)
    except Exception:
        last_val = 0.5

    # 起点时间：用于把时间转成“小时数”
    start_time = get_storage_time(tra_code, rec_time, engine) or food_info.get("ProDate")
    start_time = pd.to_datetime(start_time, errors="coerce")
    if pd.isna(start_time):
        # 兜底：没有起点就以当前时刻减 1 小时作为起点
        start_time = rec_time - pd.Timedelta(hours=1)

    # 上一次时间兜底
    last_time = pd.to_datetime(last_time, errors="coerce")
    if pd.isna(last_time):
        last_time = start_time

    # 将时间统一为“从起点计的小时数（float）”
    t_prev = (last_time - start_time).total_seconds() / 3600.0
    t_curr = (rec_time  - start_time).total_seconds() / 3600.0

    # 温度兜底 & 转 float
    try:
        temp_c = float(curr_temp) if curr_temp is not None else 4.0
    except Exception:
        temp_c = 4.0

    # 调用单步预测（只接受 float 小时）
    if t_curr > t_prev:
        return float(
            predictor.predict_single_step(
                prev_time=t_prev,
                prev_conc_log10=last_val,
                current_time=t_curr,
                current_temp=temp_c
            )
        )
    else:
        # 非正时间步长时，直接返回上一次值
        return float(last_val)


def _make_gbr_record(
    row: Dict[str, Any],
    food_info: Dict[str, Any],
    value: float,
    risk_level: str
) -> Optional[Dict[str, Any]]:
    monitor_num = row.get("MonitorNum")
    if monitor_num is None:
        return None
    return {
        "PredictResultID": f"{monitor_num}51",
        "MonitorNum": monitor_num,
        "OrderNumber": row.get("OrderNumber"),
        "TraCode": row.get("TraCode"),
        "RecTime": row.get("RecTime"),
        "FoodClassificationCode": food_info.get("FoodClassificationCode"),
        "Temp": row.get("Temp"),
        "Humid": row.get("Humid"),
        "PredictFlag": RISK_FLAG,
        "RiskName": RISK_NAME,
        "PredictValue": float(value),
        "Unit": UNIT,
        "RiskLevel": risk_level,
    }

# --------- 对外入口 ---------
def execute(
    row: Dict[str, Any],
    food_info: Dict[str, Any],
    engine,
    predictor_cache: Dict[str, Any]
) -> List[Dict[str, Any]]:

    if str(food_info.get("FoodClassificationCode")) not in SUPPORTED_FOOD_CLASSES:
        return []

    # 1) 懒加载/复用模型
    predictor: PMAPCPredictor = predictor_cache.get("pork_gbr")
    if predictor is None:
        predictor = PMAPCPredictor()
        predictor_cache["pork_gbr"] = predictor

    # 2) 计算 gbr 数值
    value = _compute_gbr_value(row, food_info, engine, predictor)

    # 3) 判级（优先用 prediction_logic 注入的 RiskClassifier）
    low  = RiskClassifier.pm_gbr_low
    mid  = RiskClassifier.pm_gbr_mid
    high = RiskClassifier.pm_gbr_high
    if value is None:
        risk_level = "无"
    elif (low is None) or (mid is None) or (high is None):
        risk_level = "未知"
    else:
        v = float(value)
        if v < float(low):
            risk_level = "无"
        elif v < float(mid):
            risk_level = "低"
        elif v < float(high):
            risk_level = "中"
        else:
            risk_level = "高"
    # 4) 生成记录
    rec = _make_gbr_record(row, food_info, value, risk_level)
    return [rec] if rec else []



