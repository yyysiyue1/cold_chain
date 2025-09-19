# models/crayfish_tvbn/logic.py
from __future__ import annotations
import pandas as pd
from typing import Dict, Any, List, Optional
from .crayfish_tvbn_predictor import Crayfish_TVBNPredictor
from models.risk_levels import determine_risk_level
from models.crayfish_tvbn.advanced_prediction_models import calculate_dynamic_tvbn_value

# 直接复用你现有的这些函数（仍从 app.prediction_logic import 也行，
# 但更推荐把它们逐步迁到 app/repos/orders.py；这里先复用，保证最小改动）
# ✅ 新的独立来源，不会再循环
from app.repos.orders import (
    get_order_tra_chain, find_previous_abnormal_value,
    find_previous_monitor_time, get_storage_time
)


RISK_FLAG = "化学"
RISK_NAME = "挥发性盐基氮"
UNIT = "mg/100g"
SUPPORTED_FOOD_CLASSES = {"C09006"}  # 小龙虾

def _compute_tvbn_value(row: Dict[str, Any], food_info: Dict[str, Any], engine, predictor: Crayfish_TVBNPredictor) -> float:
    rec_time = row.get("RecTime")
    tra_code = row.get("TraCode")
    order_number = row.get("OrderNumber")

    chain = get_order_tra_chain(order_number, tra_code, engine)
    last_val = find_previous_abnormal_value(order_number, rec_time, tra_code, chain, engine, flag=RISK_FLAG)
    last_time = find_previous_monitor_time(order_number, rec_time, tra_code, chain, engine)

    if last_val is None or pd.isna(last_val) or float(last_val) == 0.0:
        last_val = 8.0

    if not last_time:
        last_time = get_storage_time(tra_code, rec_time, engine) or food_info.get("ProDate")
        if not last_time:
            last_time = pd.to_datetime(rec_time) - pd.Timedelta(hours=1)
    last_time = pd.to_datetime(last_time) if last_time is not None else None

    if pd.notna(last_time) and pd.notna(rec_time):
        return float(calculate_dynamic_tvbn_value(
            last_abnormal_value=last_val,
            last_monitor_time=last_time,
            rec_time=rec_time,
            current_temp=row.get("Temp"),
            predictor=predictor
        ))
    return float(last_val)

def _make_tvbn_record(row: Dict[str, Any], food_info: Dict[str, Any], value: float, engine) -> Optional[Dict[str, Any]]:
    monitor_num = row.get("MonitorNum")
    if monitor_num is None: return None
    food_code = food_info.get("FoodClassificationCode")
    risk_level = determine_risk_level(food_code, value, RISK_FLAG, engine)
    return {
        "PredictResultID": f"{monitor_num}11",
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
        "RiskLevel": risk_level
    }

def execute(row: Dict[str, Any], food_info: Dict[str, Any], engine, predictor_cache: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    入口：由外层逻辑调用。
    仅在食品分类匹配小龙虾(C09006)时生成记录。
    """
    if food_info.get("FoodClassificationCode") not in SUPPORTED_FOOD_CLASSES:
        return []

    # 懒加载/重用模型实例
    predictor: Crayfish_TVBNPredictor = predictor_cache.get("crayfish_tvbn")  # 约定 key
    if predictor is None:
        predictor = Crayfish_TVBNPredictor()
        predictor_cache["crayfish_tvbn"] = predictor

    value = _compute_tvbn_value(row, food_info, engine, predictor)
    rec = _make_tvbn_record(row, food_info, value, engine)
    return [rec] if rec else []
