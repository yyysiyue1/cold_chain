import pandas as pd
from models.crayfish_tvbn.crayfish_tvbn_predictor import Crayfish_TVBNPredictor
from models.crayfish_tvbn.logic import execute as exec_crayfish_tvbn
from app.repos.orders import get_order_tra_chain, get_storage_time
from app.repos.predictions import insert_results
from app.features.temperature import (
    is_temp_abnormal, calculate_temp_predict_value, is_exceeding_temp_threshold
)
from app.features.humidity   import (
    is_humidity_abnormal, calculate_humidity_predict_value, is_exceeding_humid_threshold
)
from app.features.shelf_life import (
    is_shelf_life_abnormal, calculate_shelf_life_abnormal_duration
)
from app.records import (
    generate_temperature_abnormal_record, generate_humidity_abnormal_record,
    generate_shelf_life_abnormal_record
)

# ==============================================================================
# --- 温度   ---
# ==============================================================================
def process_temperature(row, food, *, engine, order_chain):
    temp_abnormal = is_temp_abnormal(row["Temp"], food["StorTempLower"], food["StorTempUpper"])
    temp_val = calculate_temp_predict_value(
        temp_abnormal, row["OrderNumber"], row["RecTime"], row["TraCode"], order_chain, engine
    )
    over = is_exceeding_temp_threshold(food["FoodClassificationCode"], temp_val, engine,
                                       temp_abnormal=temp_abnormal)
    # —— 用位置参数传入风险布尔值
    return generate_temperature_abnormal_record(row, food, temp_val, over)


# ==============================================================================
# --- 湿度  ---
# ==============================================================================
def process_humidity(row, food, *, engine, order_chain):
    humid_abnormal = is_humidity_abnormal(row["Humid"], food["StorHumidLower"], food["StorHumidUpper"])
    humid_val = calculate_humidity_predict_value(
        humid_abnormal, row["OrderNumber"], row["RecTime"], row["TraCode"], order_chain, engine
    )
    over = is_exceeding_humid_threshold(food["FoodClassificationCode"], humid_val, engine,
                                        humid_abnormal=humid_abnormal)
    # —— 用位置参数
    return generate_humidity_abnormal_record(row, food, humid_val, over)


# ==============================================================================
# --- 过期   ---
# ==============================================================================
def process_shelf_life(row, food, *, engine):
    shelf_over = is_shelf_life_abnormal(food["ProDate"], row["RecTime"], food["ShelfLife"])
    shelf_val  = calculate_shelf_life_abnormal_duration(food["ProDate"], row["RecTime"], food["ShelfLife"])
    # —— 用位置参数
    return generate_shelf_life_abnormal_record(row, food, shelf_val, shelf_over)

# ==============================================================================
# --- 统一调度预测模型  ---
# ==============================================================================
def execute_prediction_unit(row, food_info, engine, predictor_cache):
    """
    统一预测调度单元：根据食品分类码路由到具体标志物模型。
    predictor_cache: 例如 {'crayfish_tvbn': TVBNPredictor(), ...}
    返回：list[dict]
    """
    food_class_code = food_info.get("FoodClassificationCode")
    records = []

    # —— 路由映射：食品分类码 -> 对应模型的 execute 函数
    ROUTES = {
        'C09006': exec_crayfish_tvbn,   # 小龙虾-TVBN
        # 'P00001': exec_pork_tvbn,     # 举例：猪肉-TVBN（添加时解除注释）
        # 其他食品分类继续在这里加
    }

    handler = ROUTES.get(str(food_class_code))
    if handler:
        try:
            out = handler(row, food_info, engine, predictor_cache)  # 约定返回 list[dict]
            if out:
                records.extend(out)
        except Exception as e:
            print(f"❌ 标志物执行失败（{food_class_code}）: {e}")
    else:
        # 没有匹配的食品分类码就不做标志物预测
        pass

    return records

# 温湿度过期判断并写入预测表 (逻辑,恢复循环内插入)
# ---------- 精简后的主函数 ----------

def handle_prediction_results(monitor_df, food_df, engine):
    """处理监控数据, 计算风险并写入预测表"""
    if monitor_df.empty:
        print("ℹ️ 无监控数据需要处理。"); return []
    if food_df.empty:
        print("⚠️ 缺少食品基础信息,无法进行预测处理。"); return []

    # 预处理一次时间和必要字段
    df = monitor_df.copy()
    df["RecTime"] = pd.to_datetime(df["RecTime"], errors="coerce")
    df = df.dropna(subset=["RecTime", "TraCode"])
    if df.empty:
        print("ℹ️ 预处理后无有效的监控数据。"); return []

    # TraCode -> food_info 一次映射（把常用字段挑出来，后面用下标访问更快）
    food_idx = (
        food_df.drop_duplicates(subset=["TraCode"], keep="first")
               .set_index("TraCode")
               .loc[:, ["FoodClassificationCode","StorTempLower","StorTempUpper",
                        "StorHumidLower","StorHumidUpper","ShelfLife","ProDate"]]
               .to_dict("index")
    )

    # 模型缓存（需要就放这里；也可以懒加载）
    predictor_cache = {"crayfish_tvbn": Crayfish_TVBNPredictor()}

    results = []
    for _, r in df.sort_values(by=["TraCode", "RecTime"]).iterrows():
        # 统一把常用字段提取成 row dict，用[]保证缺就报错（便于尽早发现数据问题）
        try:
            row = {
                "TraCode": r["TraCode"],
                "RecTime": r["RecTime"],
                "OrderNumber": r.get("OrderNumber"),
                "Temp": r.get("Temp"),
                "Humid": r.get("Humid"),
                "MonitorNum": r.get("MonitorNum")
            }
        except KeyError as e:
            print(f"⚠️ 缺少必要字段：{e}; 跳过该行。"); continue

        food = food_idx.get(row["TraCode"])
        if not food:
            # 找不到对应食品信息，跳过
            continue

        # 过滤入库时间（只查一次）
        stime = get_storage_time(row["TraCode"], row["RecTime"], engine)
        if stime and row["RecTime"] < stime:
            continue

        # 订单路径链（温/湿/标志物都会用）
        order_chain = get_order_tra_chain(row["OrderNumber"], row["TraCode"], engine)

        # ① 温度
        rec_temp = process_temperature(row, food, engine=engine, order_chain=order_chain)
        # ② 湿度
        rec_humid = process_humidity(row, food, engine=engine, order_chain=order_chain)
        # ③ 过期
        rec_shelf = process_shelf_life(row, food, engine=engine)
        # ④ 标志物（TVB-N 等）
        marker_records = execute_prediction_unit(row, food, engine, predictor_cache)

        to_insert = [x for x in (rec_temp, rec_humid, rec_shelf) if x] + (marker_records or [])
        if to_insert:
            insert_results(to_insert, engine)
            results.extend(to_insert)

    print(f"✅ 预测结果处理完成。尝试生成 {len(results)} 条预测记录。")
    return results

