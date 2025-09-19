# app/repos/orders.py
import numpy as np
from sqlalchemy import text
import pandas as pd
from datetime import datetime
from db.database_setup import prediction_table_name
# ==============================================================================
# --- 预测逻辑函数  ---
# ==============================================================================
# 从食品知识表中获得食品信息（保质期、温度上下限、食品分类代码）
# ******** 开始:预测逻辑函数 ********
def get_food_info(engine):
    """
    从数据库查询冷链食品信息,包括温度/湿度上下限、保质期、生产日期等。
    返回: DataFrame
    """
    query = """
    SELECT
        mit.TraCode,
        fsti.StorTempUpper,
        fsti.StorTempLower,
        fsti.StorHumidUpper,
        fsti.StorHumidLower,
        fsti.ShelfLife,
        fsti.FoodClassificationCode,
        fsti.SecondaryClassificationName,
        fsti.TempAnonalyDuration,
        fsti.HumidAnonalyDuration,
        ccfo.ProDate,
        ccfo.FoodName
    FROM monitoring_informationtable mit
    JOIN cold_chain_food_origin ccfo ON mit.TraCode = ccfo.TraCode
    JOIN food_knowledge fsti ON ccfo.FoodClassificationCode = fsti.FoodClassificationCode
    GROUP BY mit.TraCode, fsti.StorTempUpper, fsti.StorTempLower, fsti.FoodClassificationCode,
             fsti.SecondaryClassificationName, fsti.StorHumidUpper, fsti.StorHumidLower,
             fsti.ShelfLife, ccfo.ProDate, fsti.TempAnonalyDuration, fsti.HumidAnonalyDuration, ccfo.FoodName;
    """
    try:
        food_df = pd.read_sql(query, engine)
        food_df["ProDate"] = pd.to_datetime(food_df["ProDate"], errors="coerce")
        numeric_cols = ["StorTempUpper","StorTempLower","StorHumidUpper","StorHumidLower",
                        "ShelfLife","TempAnonalyDuration","HumidAnonalyDuration"]
        for c in numeric_cols:
            food_df[c] = pd.to_numeric(food_df[c], errors="coerce")
        return food_df
    except Exception as e:
        print(f"❌ 获取或处理食品信息时出错:{e}")
        return pd.DataFrame()
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
    """查找路径链中的上一个订单号"""
    if not order_tra_chain or pd.isna(order_number):  # 增加 pd.isna 检查
        return None

    order_list = order_tra_chain.split(",")
    # 检查 order_number 是否在列表中 (代码没有此检查,但建议保留)
    if order_number not in order_list:
        # print(f"⚠️ 订单 {order_number} 不在路径链 {order_tra_chain} 中。")
        return None

    try:  # 保留错误处理
        index = order_list.index(order_number)
        if index == 0:
            return False  # 是第一个订单

        previous_order = order_list[index - 1]
        # print(f"上一个元素：{previous_order}")
        return previous_order if previous_order.startswith("O") else False  # 上一个是订单号就返回,否则返回 False
    except ValueError:  # 如果 index() 失败
        return None
    except Exception as e:
        print(f"❌ 查找上一个订单时出错 ({order_number} in {order_tra_chain}):{e}")
        return None


# 判断数据是否是该订单上第一条数据
def is_first_record_simple(order_number, tra_code, rec_time, engine):
    if pd.isna(order_number) or pd.isna(tra_code) or pd.isna(rec_time):
        return False
    rec_time = pd.to_datetime(rec_time, errors='coerce')
    if pd.isna(rec_time): return False
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
    """
    查找指定追溯码在 cold_chain_food_origin 表中的储运时间 (StorageTime)。

    参数:
        tra_code (str): 追溯码。
        rec_time (datetime): 当前记录时间，用于验证储运时间不是未来的时间。
        engine: SQLAlchemy 数据库连接引擎。

    返回:
        datetime or None: 找到且有效的储运时间，否则返回 None。
    """
    if pd.isna(tra_code):
        return None

    try:
        # 使用参数化查询以防止SQL注入
        storage_sql = text("""
            SELECT StorageTime
            FROM cold_chain_food_origin
            WHERE TraCode = :tra_code
            LIMIT 1
        """)
        storage_df = pd.read_sql(storage_sql, engine, params={"tra_code": tra_code})

        if not storage_df.empty and pd.notna(storage_df.iloc[0]['StorageTime']):
            storage_time = pd.to_datetime(storage_df.iloc[0]['StorageTime'], errors='coerce')
            # 确保 StorageTime 早于或等于当前记录时间
            if pd.notna(storage_time) and storage_time <= rec_time:
                return storage_time
    except Exception as e:
        print(f"❌ 查询 StorageTime 时出错 (TraCode: {tra_code}): {e}")

    return None # 其他所有情况都返回 None



# 首站查相邻监测数据
def get_previous_first_station_data(tra_code, rec_time, engine):
    """
    获取某追溯码下、无订单号的监测数据中,时间早于当前时间的最近一条（首站数据）的监测时间。
    如果找不到,则回退查询 cold_chain_food_origin 表的 StorageTime。
    (注意：代码未包含 CreateTime 或 ProDate 回退)

    返回：
    - datetime,如果找到
    - None,如果都没找到
    """
    # 保留对 None 或 NaN 的检查
    if pd.isna(tra_code) or pd.isna(rec_time):
        return None

    # 确保 rec_time 是 datetime 对象 (保留优化)
    if not isinstance(rec_time, datetime):
        rec_time = pd.to_datetime(rec_time, errors='coerce')
        if pd.isna(rec_time):return None

    rec_time_str = rec_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    sql = text(f"""
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
        else:
            # 如果在监控表中找不到，则调用新函数作为回退
            return get_storage_time(tra_code, rec_time, engine)
    except Exception as e:
        print(f"❌ 查询首站监控数据时出错 (TraCode: {tra_code}): {e}")
        # 如果查询监控表出错，仍然尝试获取储运时间
        return get_storage_time(tra_code, rec_time, engine)


# 同一订单中查找当前监测记录再路径上的上一条监测记录
def get_previous_monitor_record(order_number, tra_code, rec_time, engine):
    if pd.isna(order_number) or pd.isna(tra_code) or pd.isna(rec_time):
        return None
    rec_time = pd.to_datetime(rec_time, errors='coerce')
    if pd.isna(rec_time): return None
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



# 查找首站预测温度/湿度异常值
def get_previous_first_station_predict_value(tra_code, rec_time, engine, flag="温度"):
    if pd.isna(tra_code) or pd.isna(rec_time):
        return 0.0
    rec_time = pd.to_datetime(rec_time, errors='coerce')
    if pd.isna(rec_time): return 0.0
    rec_time_str = rec_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    sql = text(f"""
        SELECT PredictValue
        FROM {prediction_table_name}
        WHERE TraCode = :tra_code
          AND (OrderNumber IS NULL OR OrderNumber = '')
          AND RecTime < :rec_time
          AND PredictFlag = :flag
        ORDER BY RecTime DESC
        LIMIT 1
    """)
    try:
        df = pd.read_sql(sql, engine, params={
            "tra_code": str(tra_code),
            "rec_time": rec_time_str,
            "flag": flag
        })
        val = df.iloc[0]['PredictValue'] if not df.empty else 0.0
        try:
            return float(val) if pd.notna(val) else 0.0
        except (ValueError, TypeError):
            return 0.0
    except Exception as e:
        print(f"❌ 获取首站先前预测值时出错 (TraCode:{tra_code}, Flag:{flag}):{e}")
        return 0.0



# 同一订单中查找当前监测记录再路径上的上一条监测记录的异常值
def get_previous_monitor_predict_value(order_number, tra_code, rec_time, engine, flag="温度"):
    if pd.isna(order_number) or pd.isna(tra_code) or pd.isna(rec_time):
        return 0.0
    rec_time = pd.to_datetime(rec_time, errors='coerce')
    if pd.isna(rec_time): return 0.0
    rec_time_str = rec_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    sql = text(f"""
        SELECT PredictValue
        FROM {prediction_table_name}
        WHERE OrderNumber = :order_number
          AND TraCode = :tra_code
          AND RecTime < :rec_time
          AND PredictFlag = :flag
        ORDER BY RecTime DESC
        LIMIT 1
    """)
    try:
        df = pd.read_sql(sql, engine, params={
            "order_number": str(order_number),
            "tra_code": str(tra_code),
            "rec_time": rec_time_str,
            "flag": flag
        })
        val = df.iloc[0]['PredictValue'] if not df.empty else 0.0
        try:
            return float(val) if pd.notna(val) else 0.0
        except (ValueError, TypeError):
            return 0.0
    except Exception as e:
        print(f"❌ 获取同一订单先前预测值时出错 (Order:{order_number}, TraCode:{tra_code}, Flag:{flag}):{e}")
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
def find_previous_abnormal_value(order_number, rec_time, tra_code, order_tra_chain, engine, flag="温度"):
    """
    查找上一条预测值（支持温度/湿度）。
    """
    # 处理 NaN 或 None 的 order_number (保留优化)
    current_order = None if pd.isna(order_number) else order_number
    if current_order is None:
        return get_previous_first_station_predict_value(tra_code, rec_time, engine, flag)
    # 确保 rec_time 是 datetime (保留优化)
    if not isinstance(rec_time, datetime):
        rec_time = pd.to_datetime(rec_time, errors='coerce')
        if pd.isna(rec_time): return 0.0
    if is_first_record_simple(current_order, tra_code, rec_time, engine):
        previous_order = find_previous_order(current_order, order_tra_chain)
        if previous_order:
            # 往前递归找 (代码的递归调用)
            return find_previous_abnormal_value(previous_order, rec_time, tra_code, order_tra_chain, engine, flag)
        else:  # previous_order is False or None
            return get_previous_first_station_predict_value(tra_code, rec_time, engine, flag)
    else:
        return get_previous_monitor_predict_value(current_order, tra_code, rec_time, engine, flag)

