# cold_chain/models/salmon_gbr/logic.py
from __future__ import annotations
import pandas as pd
from typing import Dict, Any, List, Optional
from .salmon_gbr_predictor import SalmonGBRPredictor

from app.repos.orders import (
    get_order_tra_chain, find_previous_abnormal_value,
    find_previous_monitor_time, get_storage_time
)
from models.risk_levels import RiskClassifier

RISK_FLAG = "生物"
RISK_NAME = "三文鱼"
UNIT = "log10 CFU/g"
SUPPORTED_FOOD_CLASSES = {"B06031"}
# 初始浓度默认值 (log10 CFU/g)
INITIAL_LOG10_CONCENTRATION = 2.3
def _compute_gbr_value(row: Dict[str, Any], food_info: Dict[str, Any], engine, predictor: SalmonGBRPredictor) -> float:
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
        last_val = INITIAL_LOG10_CONCENTRATION       # 生菜的初始浓度
    # 兜底监测时间
    if not last_time:
        last_time = get_storage_time(tra_code, rec_time, engine) or food_info.get("ProDate")
        if not last_time:
            last_time = pd.to_datetime(rec_time) - pd.Timedelta(hours=1)
    # 转换时间
    last_time = pd.to_datetime(last_time) if last_time is not None else None
    # 调用单步预测
    if pd.notna(last_time) and pd.notna(rec_time):
        time_diff_hours = (rec_time - last_time).total_seconds() / 3600  #时间步长
        return float(SalmonGBRPredictor.predict_single_step(
            time_h=time_diff_hours,
            temp_c=row.get("Temp"),
            prev_log10_conc=last_val
        ))
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
    predictor: SalmonGBRPredictor = predictor_cache.get("crayfish_tvbn")
    if predictor is None:
        predictor = SalmonGBRPredictor()
        predictor_cache["crayfish_tvbn"] = predictor

    # 2) 计算 TVBN 数值
    value = _compute_gbr_value(row, food_info, engine, predictor)

    # 3) 判级（优先用 prediction_logic 注入的 RiskClassifier）
    food_code = food_info.get("FoodClassificationCode")
    risk_level: str
    rc = predictor_cache.get("risk_classifier")
    if rc is not None:
        risk_level = rc.classify(food_code, value, RISK_FLAG)
    else:
        # 兜底：仅加载当前分类阈值，避免全表 IO
        tmp = RiskClassifier(engine)
        tmp.preload(flags=(RISK_FLAG,), food_codes=[food_code])
        risk_level = tmp.classify(food_code, value, RISK_FLAG)

    # 4) 生成记录
    rec = _make_gbr_record(row, food_info, value, risk_level)
    return [rec] if rec else []



