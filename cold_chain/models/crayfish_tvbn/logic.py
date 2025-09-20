# models/crayfish_tvbn/logic.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
import pandas as pd

from .crayfish_tvbn_predictor import Crayfish_TVBN_Predictor
from models.crayfish_tvbn.advanced_prediction_models import calculate_dynamic_tvbn_value

from app.repos.orders import (
    get_order_tra_chain, find_previous_abnormal_value,
    find_previous_monitor_time, get_storage_time
)

# 兜底判级时用（只有在未注入 RiskClassifier 的情况下才会触发）
from models.risk_levels import RiskClassifier

RISK_FLAG  = "化学"
RISK_NAME  = "挥发性盐基氮"
UNIT       = "mg/100g"
SUPPORTED_FOOD_CLASSES = {"C09006"}  # 小龙虾

# --------- 内部：计算 TVBN 数值 ---------
def _compute_tvbn_value(
    row: Dict[str, Any],
    food_info: Dict[str, Any],
    engine,
    predictor: Crayfish_TVBN_Predictor
) -> float:
    rec_time = pd.to_datetime(row.get("RecTime"), errors="coerce")
    tra_code = row.get("TraCode")
    order_number = row.get("OrderNumber")

    chain = get_order_tra_chain(order_number, tra_code, engine)
    last_val = find_previous_abnormal_value(order_number, rec_time, tra_code, chain, engine, flag=RISK_FLAG)
    last_time = find_previous_monitor_time(order_number, rec_time, tra_code, chain, engine)

    # 初值兜底
    try:
        if last_val is None or pd.isna(last_val) or float(last_val) == 0.0:
            last_val = 8.0
        else:
            last_val = float(last_val)
    except Exception:
        last_val = 8.0

    # 上次时间兜底
    if not last_time:
        last_time = get_storage_time(tra_code, rec_time, engine) or food_info.get("ProDate")
        if not last_time:
            last_time = (rec_time if pd.notna(rec_time) else pd.Timestamp.utcnow()) - pd.Timedelta(hours=1)
    last_time = pd.to_datetime(last_time, errors="coerce")

    # 计算动态值
    if pd.notna(last_time) and pd.notna(rec_time):
        return float(calculate_dynamic_tvbn_value(
            last_abnormal_value=last_val,
            last_monitor_time=last_time,
            rec_time=rec_time,
            current_temp=row.get("Temp"),
            predictor=predictor
        ))
    return float(last_val)

# --------- 内部：拼装记录（已外部传入风险等级）---------
def _make_tvbn_record(
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
    """
    仅在食品分类匹配小龙虾(C09006)时生成记录。
    依赖外部注入的 risk_classifier 做风险判级；若未注入则临时构造一个兜底判级器。
    """
    if str(food_info.get("FoodClassificationCode")) not in SUPPORTED_FOOD_CLASSES:
        return []

    # 1) 懒加载/复用模型
    predictor: Crayfish_TVBN_Predictor = predictor_cache.get("crayfish_tvbn")
    if predictor is None:
        predictor = Crayfish_TVBN_Predictor()
        predictor_cache["crayfish_tvbn"] = predictor

    # 2) 计算 TVBN 数值
    value = _compute_tvbn_value(row, food_info, engine, predictor)

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
    rec = _make_tvbn_record(row, food_info, value, risk_level)
    return [rec] if rec else []
