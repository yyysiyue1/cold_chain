# app/repos/orders.py
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
from sqlalchemy import text
from db.database_setup import prediction_table_name
# =========================
# 食品知识/信息读取
# =========================
# 从食品知识表中获得食品信息（保质期、温度上下限、食品分类代码）
# ******** 开始:预测逻辑函数 ********
import pandas as pd

#从食品知识表里获取映射数据
def get_food_info(engine):
    """
    从数据库查询冷链食品信息,包括温度/湿度上下限、保质期、生产日期等。
    返回: DataFrame
    """
    query = """
    SELECT
        ccfo.TraCode,
        fsti.StorTempUpper,
        fsti.StorTempLower,
        fsti.StorHumidUpper,
        fsti.StorHumidLower,
        fsti.ShelfLife,
        fsti.FoodClassificationCode,
        fsti.SecondaryClassificationName,
        fsti.TempAnonalyDuration,
        fsti.HumidAnonalyDuration  ,
        ccfo.ProDate,
        ccfo.FoodName
    FROM cold_chain_food_origin ccfo
    JOIN food_knowledge fsti 
        ON ccfo.FoodClassificationCode = fsti.FoodClassificationCode
    JOIN (
        -- 每个 TraCode 只取最新的 ProDate
        SELECT TraCode, MAX(ProDate) AS MaxProDate
        FROM cold_chain_food_origin
        GROUP BY TraCode
    ) latest 
        ON ccfo.TraCode = latest.TraCode AND ccfo.ProDate = latest.MaxProDate
    WHERE EXISTS (
        SELECT 1 FROM monitoring_informationtable mit WHERE mit.TraCode = ccfo.TraCode
    );
    """
    try:
        food_df = pd.read_sql(query, engine)
        food_df["ProDate"] = pd.to_datetime(food_df["ProDate"], errors="coerce")
        numeric_cols = [
            "StorTempUpper","StorTempLower","StorHumidUpper","StorHumidLower",
            "ShelfLife","TempAnonalyDuration","HumidAnonalyDuration"
        ]
        for c in numeric_cols:
            food_df[c] = pd.to_numeric(food_df[c], errors="coerce")
        return food_df
    except Exception as e:
        print(f"❌ 获取或处理食品信息时出错:{e}")
        return pd.DataFrame()

#从食品信息表中获取保质期
def get_shelf_life_by_tracode(engine, tra_code: str):
    """
    返回: dict 示例:
      {"TraCode": "...", "ShelfLife": 7, "ShelfLifeUnit": "天", "MatchedBy": "FoodNum"}
    命中顺序: 先按 FoodNum 找到 cold_chain_food 最新记录; 若没有，再按 FoodClassificationCode 找最新。
    “最新”比较顺序: UpdateTime DESC > CreateTime DESC > id DESC
    """
    sql = text("""
        SELECT
          o.TraCode,
          COALESCE(n.ShelfLife, c.ShelfLife)         AS ShelfLife,
          COALESCE(n.ShelfLifeUnit, c.ShelfLifeUnit) AS ShelfLifeUnit,
          CASE WHEN n.FoodNum IS NOT NULL THEN 'FoodNum' ELSE 'FoodClassificationCode' END AS MatchedBy
        FROM (
          SELECT TraCode, FoodNum, FoodClassificationCode
          FROM cold_chain_food_origin
          WHERE TraCode = :tra_code
          LIMIT 1
        ) o
        LEFT JOIN cold_chain_food n
          ON n.FoodNum = o.FoodNum
         AND n.id = (
              SELECT n2.id
              FROM cold_chain_food n2
              WHERE n2.FoodNum = o.FoodNum
              ORDER BY
                  (n2.UpdateTime IS NULL) ASC, n2.UpdateTime DESC,
                  (n2.CreateTime IS NULL) ASC, n2.CreateTime DESC,
                  n2.id DESC
              LIMIT 1
          )
        LEFT JOIN cold_chain_food c
          ON c.FoodClassificationCode = o.FoodClassificationCode
         AND c.id = (
              SELECT c2.id
              FROM cold_chain_food c2
              WHERE c2.FoodClassificationCode = o.FoodClassificationCode
              ORDER BY
                  (c2.UpdateTime IS NULL) ASC, c2.UpdateTime DESC,
                  (c2.CreateTime IS NULL) ASC, c2.CreateTime DESC,
                  c2.id DESC
              LIMIT 1
          );
    """)
    try:
        df = pd.read_sql(sql, engine, params={"tra_code": tra_code})
        if df.empty:
            return None
        return df.iloc[0].to_dict()
    except Exception as e:
        # 这里用 print/logger 均可，按你项目风格来
        print(f"❌ get_shelf_life_by_tracode 执行失败: {e}")
        return None

# =========================
# 订单/监测回溯工具
# =========================
# 获取订单路径信息
def get_order_tra_chain(order_number, tra_code, engine):
    if pd.isna(order_number) or pd.isna(tra_code):
        return None
    sql = text("""
        SELECT OrderTraChain
        FROM order_info
        WHERE OrderNumber = :order_number AND TraCode = :tra_code
        LIMIT 1
    """)
    try:
        df = pd.read_sql(sql, engine, params={"order_number": str(order_number), "tra_code": str(tra_code)})
        return df.iloc[0]['OrderTraChain'] if not df.empty else None
    except Exception as e:
        print(f"❌ 获取订单路径链时出错 (Order:{order_number}, TraCode:{tra_code}):{e}")
        return None

# 查找路径中的上一条订单
def find_previous_order(order_number, order_tra_chain):
    """在路径链中找上一个订单号；第一个则返回 False；找不到返回 None。"""
    if not order_tra_chain or pd.isna(order_number):
        return None
    order_list = order_tra_chain.split(",")
    if order_number not in order_list:
        return None
    try:
        index = order_list.index(order_number)
        if index == 0:
            return False
        previous_order = order_list[index - 1]
        return previous_order if previous_order.startswith("O") else False
    except ValueError:
        return None
    except Exception as e:
        print(f"❌ 查找上一个订单时出错 ({order_number} in {order_tra_chain}):{e}")
        return None


# 判断数据是否是该订单上第一条数据
def is_first_record_simple(order_number, tra_code, rec_time, engine):
    if pd.isna(order_number) or pd.isna(tra_code) or pd.isna(rec_time):
        return False
    rec_time = pd.to_datetime(rec_time, errors='coerce')
    if pd.isna(rec_time):
        return False
    rec_time_str = rec_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    sql = text("""
        SELECT 1
        FROM monitoring_informationtable
        WHERE OrderNumber = :order_number
          AND TraCode = :tra_code
          AND RecTime < :rec_time
        LIMIT 1
    """)
    try:
        df = pd.read_sql(sql, engine, params={
            "order_number": str(order_number),
            "tra_code": str(tra_code),
            "rec_time": rec_time_str
        })
        return df.empty
    except Exception as e:
        print(f"❌ 检查首条记录时出错 (Order:{order_number}, TraCode:{tra_code}):{e}")
        return False


#查找追溯码对应的储运时间函数
def get_storage_time(tra_code, rec_time, engine):
    """从 cold_chain_food_origin 取 StorageTime（需 <= rec_time）。"""
    if pd.isna(tra_code):
        return None
    try:
        storage_sql = text("""
            SELECT StorageTime
            FROM cold_chain_food_origin
            WHERE TraCode = :tra_code
            LIMIT 1
        """)
        storage_df = pd.read_sql(storage_sql, engine, params={"tra_code": tra_code})
        if not storage_df.empty and pd.notna(storage_df.iloc[0]['StorageTime']):
            storage_time = pd.to_datetime(storage_df.iloc[0]['StorageTime'], errors='coerce')
            if pd.notna(storage_time) and storage_time <= rec_time:
                return storage_time
    except Exception as e:
        print(f"❌ 查询 StorageTime 时出错 (TraCode: {tra_code}): {e}")
    return None

# 首站查相邻监测数据
def get_previous_first_station_data(tra_code, rec_time, engine):
    """首站：取该 TraCode 在无 OrderNumber 的上一条 RecTime；失败则回退 StorageTime。"""
    if pd.isna(tra_code) or pd.isna(rec_time):
        return None
    if not isinstance(rec_time, datetime):
        rec_time = pd.to_datetime(rec_time, errors='coerce')
        if pd.isna(rec_time):
            return None
    rec_time_str = rec_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    sql = text("""
        SELECT RecTime
        FROM monitoring_informationtable
        WHERE TraCode = :tra_code
          AND (OrderNumber IS NULL OR OrderNumber = '')
          AND RecTime < :rec_time
        ORDER BY RecTime DESC
        LIMIT 1
    """)
    try:
        df = pd.read_sql(sql, engine, params={"tra_code": tra_code, "rec_time": rec_time_str})
        if not df.empty and pd.notna(df.iloc[0]['RecTime']):
            return pd.to_datetime(df.iloc[0]['RecTime'], errors='coerce')
        return get_storage_time(tra_code, rec_time, engine)
    except Exception as e:
        print(f"❌ 查询首站监控数据时出错 (TraCode: {tra_code}): {e}")
        return get_storage_time(tra_code, rec_time, engine)


# 同一订单中查找当前监测记录再路径上的上一条监测记录
def get_previous_monitor_record(order_number, tra_code, rec_time, engine):
    """同一订单中，取当前记录之前的上一条 RecTime。"""
    if pd.isna(order_number) or pd.isna(tra_code) or pd.isna(rec_time):
        return None
    rec_time = pd.to_datetime(rec_time, errors='coerce')
    if pd.isna(rec_time):
        return None
    rec_time_str = rec_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    sql = text("""
        SELECT RecTime
        FROM monitoring_informationtable
        WHERE OrderNumber = :order_number
          AND TraCode = :tra_code
          AND RecTime < :rec_time
        ORDER BY RecTime DESC
        LIMIT 1
    """)
    try:
        df = pd.read_sql(sql, engine, params={
            "order_number": str(order_number),
            "tra_code": str(tra_code),
            "rec_time": rec_time_str
        })
        return pd.to_datetime(df.iloc[0]['RecTime'], errors='coerce') if not df.empty else None
    except Exception as e:
        print(f"❌ 获取同一订单上一条记录时出错 (Order:{order_number}, TraCode:{tra_code}):{e}")
        return None

# =========================
# 预测值回溯（支持可选 RiskName）
# =========================

def _should_use_riskname(flag: Optional[str], risk_name: Optional[str]) -> bool:
    """
    仅当 flag ∈ {'生物','化学'} 且 risk_name 非空时，使用 RiskName 精确过滤。
    温度/湿度/过期等，即使 RiskName 有文本，也不做过滤。
    """
    if not flag or not risk_name:
        return False
    f = str(flag).strip()
    return f in {"生物", "化学"}

# 查找首站预测温度/湿度异常值
def get_previous_first_station_predict_value(
    tra_code, rec_time, engine, flag: str = "温度", risk_name: Optional[str] = None
):
    """首站上一条预测值（可选 RiskName 过滤）。"""
    if pd.isna(tra_code) or pd.isna(rec_time):
        return 0.0
    rec_time = pd.to_datetime(rec_time, errors='coerce')
    if pd.isna(rec_time):
        return 0.0
    rec_time_str = rec_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    use_risk = _should_use_riskname(flag, risk_name)
    risk_sql = " AND RiskName = :risk_name " if use_risk else ""

    sql = text(f"""
        SELECT PredictValue
        FROM {prediction_table_name}
        WHERE TraCode = :tra_code
          AND (OrderNumber IS NULL OR OrderNumber = '')
          AND RecTime < :rec_time
          AND PredictFlag = :flag
          {risk_sql}
        ORDER BY RecTime DESC
        LIMIT 1
    """)
    params = {"tra_code": str(tra_code), "rec_time": rec_time_str, "flag": flag}
    if use_risk:
        params["risk_name"] = risk_name

    try:
        df = pd.read_sql(sql, engine, params=params)
        val = df.iloc[0]['PredictValue'] if not df.empty else 0.0
        try:
            return float(val) if pd.notna(val) else 0.0
        except (ValueError, TypeError):
            return 0.0
    except Exception as e:
        print(f"❌ 获取首站先前预测值时出错 (TraCode:{tra_code}, Flag:{flag}, RiskName:{risk_name}):{e}")
        return 0.0


# 同一订单中查找当前监测记录再路径上的上一条监测记录的异常值
def get_previous_monitor_predict_value(
    order_number, tra_code, rec_time, engine, flag: str = "温度", risk_name: Optional[str] = None
):
    """订单内上一条预测值（可选 RiskName 过滤）。"""
    if pd.isna(order_number) or pd.isna(tra_code) or pd.isna(rec_time):
        return 0.0
    rec_time = pd.to_datetime(rec_time, errors='coerce')
    if pd.isna(rec_time):
        return 0.0
    rec_time_str = rec_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    use_risk = _should_use_riskname(flag, risk_name)
    risk_sql = " AND RiskName = :risk_name " if use_risk else ""

    sql = text(f"""
        SELECT PredictValue
        FROM {prediction_table_name}
        WHERE OrderNumber = :order_number
          AND TraCode = :tra_code
          AND RecTime < :rec_time
          AND PredictFlag = :flag
          {risk_sql}
        ORDER BY RecTime DESC
        LIMIT 1
    """)
    params = {
        "order_number": str(order_number),
        "tra_code": str(tra_code),
        "rec_time": rec_time_str,
        "flag": flag
    }
    if use_risk:
        params["risk_name"] = risk_name

    try:
        df = pd.read_sql(sql, engine, params=params)
        val = df.iloc[0]['PredictValue'] if not df.empty else 0.0
        try:
            return float(val) if pd.notna(val) else 0.0
        except (ValueError, TypeError):
            return 0.0
    except Exception as e:
        print(f"❌ 获取同一订单先前预测值时出错 (Order:{order_number}, TraCode:{tra_code}, Flag:{flag}, RiskName:{risk_name}):{e}")
        return 0.0



# 查找上一条监测数据的时间
def find_previous_monitor_time(order_number, rec_time, tra_code, order_tra_chain, engine):
    """
    查找上一条监测数据的时间。
    """
    # 处理 NaN 或 None 的 order_number (保留优化)
    current_order = None if pd.isna(order_number) else order_number
    if current_order is None:
        # 首站数据
        return get_previous_first_station_data(tra_code, rec_time, engine)
    # 确保 rec_time 是 datetime (保留优化)
    if not isinstance(rec_time, datetime):
        rec_time = pd.to_datetime(rec_time, errors='coerce')
        if pd.isna(rec_time): return None  # 无效时间无法判断
    if is_first_record_simple(current_order, tra_code, rec_time, engine):
        # 是订单上的第一条
        previous_order = find_previous_order(current_order, order_tra_chain)
        if previous_order:
            # 往前递归找 (代码的递归调用)
            # 注意：递归可能未正确处理 order_tra_chain 的传递,这里保持原样
            return find_previous_monitor_time(previous_order, rec_time, tra_code, order_tra_chain, engine)
        else:  # previous_order is False or None
            # 已经到路径最前面了,去首站找
            return get_previous_first_station_data(tra_code, rec_time, engine)
    else:
        # 正常情况,直接在当前订单上查上一条
        return get_previous_monitor_record(current_order, tra_code, rec_time, engine)


# 查找上一条监测记录的异常时长
def find_previous_abnormal_value(
    order_number, rec_time, tra_code, order_tra_chain, engine,
    flag: str = "温度", risk_name: Optional[str] = None
):
    """
    查找上一条预测值：
      - 温度/湿度/过期：忽略 risk_name
      - 生物/化学：按 risk_name 精确匹配同一标志物
    返回 float（未找到则 0.0）
    """
    current_order = None if pd.isna(order_number) else order_number
    #如果是首站信息
    if current_order is None:
        return get_previous_first_station_predict_value(tra_code, rec_time, engine, flag, risk_name)

    if not isinstance(rec_time, datetime):
        rec_time = pd.to_datetime(rec_time, errors='coerce')
        if pd.isna(rec_time):
            return 0.0
    #如果是当前订单上第一条记录
    if is_first_record_simple(current_order, tra_code, rec_time, engine):
        previous_order = find_previous_order(current_order, order_tra_chain)
        #上一订单存在
        if previous_order:
            return find_previous_abnormal_value(previous_order, rec_time, tra_code, order_tra_chain, engine, flag, risk_name)
        else:
            #上一订单不存在 即上一订单为首站
            return get_previous_first_station_predict_value(tra_code, rec_time, engine, flag, risk_name)
    else:
        return get_previous_monitor_predict_value(current_order, tra_code, rec_time, engine, flag, risk_name)