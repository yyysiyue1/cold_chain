# cold_chain/models/pork_Methylquinoline/logic.py
from __future__ import annotations
import pandas as pd
from typing import Dict, Any, List, Optional
from .pork_Methylquinoline_predictor import Pork_Methylquinoline_Predictor

from app.repos.orders import (
    get_order_tra_chain, find_previous_abnormal_value,
    find_previous_monitor_time, get_storage_time
)
from models.risk_levels import RiskClassifier

RISK_FLAG = "化学"
RISK_NAME = "Methylquinoline"
UNIT = "mg/100g"
SUPPORTED_FOOD_CLASSES = {"B01001"}

def _compute_Methylquinoline_value(row: Dict[str, Any], food_info: Dict[str, Any], engine, predictor: Pork_Methylquinoline_Predictor) -> float:
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
        last_val = 0.04      # 初始浓度
    # 兜底监测时间
    if not last_time:
        last_time = get_storage_time(tra_code, rec_time, engine) or food_info.get("ProDate")
        if not last_time:
            last_time = pd.to_datetime(rec_time) - pd.Timedelta(hours=1)
    # 转换时间
    last_time = pd.to_datetime(last_time) if last_time is not None else None
    # 调用单步预测
    if pd.notna(last_time) and pd.notna(rec_time):
        return float(predictor.predict_step(
            C_prev=last_val,
            T=row.get("Temp"),
            t_prev=last_time,
            t=rec_time
        ))
    return float(last_val)

def _make_Methylquinoline_record(
    row: Dict[str, Any],
    food_info: Dict[str, Any],
    value: float,
    risk_level: str
) -> Optional[Dict[str, Any]]:
    monitor_num = row.get("MonitorNum")
    if monitor_num is None:
        return None
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
    predictor: Pork_Methylquinoline_Predictor = predictor_cache.get("pork_Methylquinoline")
    if predictor is None:
        predictor = Pork_Methylquinoline_Predictor()
        predictor_cache["pork_Methylquinoline"] = predictor

    # 2) 计算 Methylquinoline 数值
    value = _compute_Methylquinoline_value(row, food_info, engine, predictor)

    # 3) 判级（优先用 prediction_logic 注入的 RiskClassifier）
    # food_code = food_info.get("FoodClassificationCode")
    # risk_level: str
    # rc = predictor_cache.get("risk_classifier")
    # if rc is not None:
    #     risk_level = rc.classify(food_code, value, RISK_FLAG)
    # else:
    #     # 兜底：仅加载当前分类阈值，避免全表 IO
    #     tmp = RiskClassifier(engine)
    #     tmp.preload(flags=(RISK_FLAG,), food_codes=[food_code])
    #     risk_level = tmp.classify(food_code, value, RISK_FLAG)
    risk_level = '无'
    # 4) 生成记录
    rec = _make_Methylquinoline_record(row, food_info, value, risk_level)
    return [rec] if rec else []



