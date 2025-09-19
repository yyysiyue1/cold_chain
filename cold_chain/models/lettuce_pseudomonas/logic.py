# models/lettuce_pseudomonas/logic.py
from __future__ import annotations
import pandas as pd
from typing import Dict, Any, List, Optional
from models.risk_levels import determine_risk_level
from .lettuce_pseudomonas_predictor import LettucePseudomonasPredictor

from app.repos.orders import (
    get_order_tra_chain, find_previous_abnormal_value,
    find_previous_monitor_time, get_storage_time
)


RISK_FLAG = "生物"
RISK_NAME = "假单胞菌"
UNIT = "log10 CFU/g"
SUPPORTED_FOOD_CLASSES = {"B04072"}

def _compute_pseudomonas_value(row: Dict[str, Any], food_info: Dict[str, Any], engine, predictor: LettucePseudomonasPredictor) -> float:
    rec_time = row.get("RecTime")
    tra_code = row.get("TraCode")
    order_number = row.get("OrderNumber")
    # 获取运输链上下文
    chain = get_order_tra_chain(order_number, tra_code, engine)
    # 获取上一次的浓度值和监测时间
    last_val = find_previous_abnormal_value(order_number, rec_time, tra_code, chain, engine, flag=RISK_FLAG)
    last_time = find_previous_monitor_time(order_number, rec_time, tra_code, chain, engine)
    # 初始值兜底
    if last_val is None or pd.isna(last_val) or float(last_val) == 0.0:
        last_val = 0.5      # 生菜的初始浓度
    # 兜底监测时间
    if not last_time:
        last_time = get_storage_time(tra_code, rec_time, engine) or food_info.get("ProDate")
        if not last_time:
            last_time = pd.to_datetime(rec_time) - pd.Timedelta(hours=1)
    # 转换时间
    last_time = pd.to_datetime(last_time) if last_time is not None else None
    # 调用单步预测
    if pd.notna(last_time) and pd.notna(rec_time):
        time_diff_hours = (rec_time - last_time).total_seconds() / 3600
        return float(LettucePseudomonasPredictor.predict_single_step(
                    prev_time=0,  # 相对时间起点
                    prev_conc_log10=last_val,
                    current_time=time_diff_hours,
                    current_temp=row.get("Temp")
        ))
    return float(last_val)

def _make_pseudomonas_record(row: Dict[str, Any], food_info: Dict[str, Any], value: float, engine) -> Optional[Dict[str, Any]]:
    monitor_num = row.get("MonitorNum")
    if monitor_num is None: return None
    food_code = food_info.get("FoodClassificationCode")
    risk_level = determine_risk_level(food_code, value, RISK_FLAG, engine)
    return {
        "PredictResultID": f"{monitor_num}12", # 使用12作为标识
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
    predictor: LettucePseudomonasPredictor = predictor_cache.get("crayfish_tvbn")  # 约定 key
    if predictor is None:
        predictor = LettucePseudomonasPredictor()
        predictor_cache["crayfish_tvbn"] = predictor

    value = _compute_pseudomonas_value(row, food_info, engine, predictor)
    rec = _make_pseudomonas_record(row, food_info, value, engine)
    return [rec] if rec else []
