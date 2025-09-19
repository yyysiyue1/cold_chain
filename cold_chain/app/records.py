# app/records.py
import math

def replace_nan_with_none(d: dict):
    import numpy as np
    return {k: (None if isinstance(v, float) and (math.isnan(v) if isinstance(v, float) else False) or v is None else v)
            for k, v in d.items()}
# 生成温度预测记录
def generate_temperature_abnormal_record(row, food_info, value, risk):
    m = row.get('MonitorNum')
    if m is None: return None
    lo, hi = food_info.get("StorTempLower"), food_info.get("StorTempUpper")
    lo_s = "    " if (lo is None or (isinstance(lo, float) and math.isnan(lo))) else str(lo)
    hi_s = "    " if (hi is None or (isinstance(hi, float) and math.isnan(hi))) else str(hi)
    return {
        "PredictResultID": f"{m}02","MonitorNum": m,
        "OrderNumber": row.get("OrderNumber"),
        "TraCode": row.get("TraCode"),
        "RecTime": row.get("RecTime"),
        "FoodClassificationCode": food_info.get("FoodClassificationCode"),
        "Temp": row.get("Temp"),
        "Humid": row.get("Humid"),
        "PredictFlag": "温度",
        "RiskName": f"{lo_s}℃~{hi_s}℃",
        "PredictValue": value,
        "Unit": "min",
        "RiskLevel": ("低" if risk else "无")
    }
# 生成湿度预测记录
def generate_humidity_abnormal_record(row, food_info, value, risk):
    m = row.get('MonitorNum')
    if m is None: return None
    lo, hi = food_info.get("StorHumidLower"), food_info.get("StorHumidUpper")
    lo_s = "    " if (lo is None or (isinstance(lo, float) and math.isnan(lo))) else str(lo)
    hi_s = "    " if (hi is None or (isinstance(hi, float) and math.isnan(hi))) else str(hi)
    return {
        "PredictResultID": f"{m}03","MonitorNum": m,
        "OrderNumber": row.get("OrderNumber"),"TraCode": row.get("TraCode"),
        "RecTime": row.get("RecTime"),"FoodClassificationCode": food_info.get("FoodClassificationCode"),
        "Temp": row.get("Temp"),"Humid": row.get("Humid"),
        "PredictFlag": "湿度","RiskName": f"{lo_s}%RH~{hi_s}%RH",
        "PredictValue": value,"Unit": "min","RiskLevel": ("低" if risk else "无")
    }
# 生成过期预测记录
def generate_shelf_life_abnormal_record(row, food_info, value, is_abnormal):
    m = row.get('MonitorNum')
    if m is None: return None
    return {
        "PredictResultID": f"{m}01","MonitorNum": m,
        "OrderNumber": row.get("OrderNumber"),"TraCode": row.get("TraCode"),
        "RecTime": row.get("RecTime"),"FoodClassificationCode": food_info.get("FoodClassificationCode"),
        "Temp": row.get("Temp"),"Humid": row.get("Humid"),
        "PredictFlag": "过期","RiskName": food_info.get("ShelfLife"),
        "PredictValue": value,"Unit": "天","RiskLevel": ("高" if is_abnormal else "无")
    }
