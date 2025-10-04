import pandas as pd
from models.lettuce_pseudomonas.logic import execute as exec_lettuce_pseudomonas
from models.crayfish_tvbn.logic import execute as exec_crayfish_tvbn
from models.pork_Methylquinoline.logic import execute as exec_pork_Methylquinoline
from models.pork_gbr.logic import execute as exec_pork_gbr
from models.salmon_gbr.logic import execute as exec_salmon_gbr
from models.pork_tvbn.logic import execute as exec_pork_tvbn
from models.pm_gbr.logic import execute as exec_pm_gbr
# 未来还有同分类的其它模型就继续导入：
from models.crayfish_tvbn.crayfish_tvbn_predictor import Crayfish_TVBN_Predictor
from models.lettuce_pseudomonas.lettuce_pseudomonas_predictor import LettucePseudomonasPredictor
from models.pork_tvbn.pork_tvbn_predictor import Pork_TVBN_Predictor
from models.pork_gbr.pork_gbr_predictor import Pork_gbr_Predictor
from models.pork_Methylquinoline.pork_Methylquinoline_predictor import Pork_Methylquinoline_Predictor
from models.salmon_gbr.salmon_gbr_predictor import SalmonGBRPredictor
from models.pm_gbr.PM_gbr_predictor import PMAPCPredictor
from models.risk_levels import RiskClassifier

from app.repos.orders import (
    get_order_tra_chain, get_storage_time,get_shelf_life_by_tracode
)
from app.repos.predictions import insert_results
from app.features.temperature import (
    is_temp_abnormal, calculate_temp_predict_value, is_exceeding_temp_threshold
)
from app.features.humidity   import (
    is_humidity_abnormal, calculate_humidity_predict_value, is_exceeding_humid_threshold
)
from app.features.shelf_life import (
    is_shelf_life_abnormal, calculate_shelf_life_abnormal_duration, _to_days
)
from app.records import (
    generate_temperature_abnormal_record, generate_humidity_abnormal_record,
    generate_shelf_life_abnormal_record
)

# 跨批次复用的全局预测器缓存（只创建一次，进程常驻）
_GLOBAL_PREDICTOR_CACHE = {  # 风险判级器可复用，也可每批刷新阈值
}
# 每个食品分类码可能需要 1 个或多个模型实例（键名要与各模型 execute 内取用的 key 一致）
# 每个食品分类码可能需要 1 个或多个模型实例（键名要与各模型 execute 内取用的 key 一致）
PREDICTOR_FACTORIES = {
    "C09006": [  # 小龙虾
        ("crayfish_tvbn", Crayfish_TVBN_Predictor),
    ],
    "B04072": [  # 生菜 假单胞菌
        ("lettuce_pseudomonas", LettucePseudomonasPredictor),
    ],
    "B06031": [  # 三文鱼GBR
        ("salmon_gbr", SalmonGBRPredictor),
    ],
    "B01001": [  # 猪肉：三个模型
        ("pork_tvbn", Pork_TVBN_Predictor),
        ("pork_gbr",  Pork_gbr_Predictor),
        ("pork_Methylquinoline", Pork_Methylquinoline_Predictor),  # ← 修正这里
    ],
    "B08005": [  # 巴氏杀菌乳
        ("pm_gbr", PMAPCPredictor),
    ],
}


# 路由映射：食品分类码 -> [一个或多个 execute 函数]
ROUTES = {
    'C09006': [exec_crayfish_tvbn],            # 小龙虾：目前只有 TVBN
    'B04072': [exec_lettuce_pseudomonas],  # 将来同分类两个模型
    'B06031': [exec_salmon_gbr],
    'B01001': [exec_pork_tvbn,exec_pork_gbr,exec_pork_Methylquinoline],
    'B08005': [exec_pm_gbr],
    # 'P00001': [exec_pork_tvbn, exec_pork_tma],          # 举例：猪肉两个模型
}

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
    # row: 监测表一行；food: 你从映射里拿到的食品知识表信息（含 ProDate, ShelfLife 等）
    res = get_shelf_life_by_tracode(engine, row.get("TraCode"))  # 可能返回 None

    if res and res.get("ShelfLife") is not None:
        # 优先用食品信息表 cold_chain_food 的保质期（带单位）
        shelf_days = _to_days(res["ShelfLife"], res.get("ShelfLifeUnit"))
    else:
        # 没查到/为空时，回退到食品知识表 food_knowledge 的保质期（通常就是“天”）
        shelf_days = food.get("ShelfLife")

    # 若仍为空，给个安全兜底（可按业务改）
    if shelf_days is None:
        shelf_days = 0

    # 判定是否超保质期
    shelf_over = is_shelf_life_abnormal(food.get("ProDate"), row.get("RecTime"), shelf_days)
    shelf_val  = calculate_shelf_life_abnormal_duration(food["ProDate"], row["RecTime"], shelf_days)
    # —— 用位置参数
    return generate_shelf_life_abnormal_record(row, food, shelf_val, shelf_over)

# ==============================================================================
# --- 统一调度预测模型  ---
# ==============================================================================
def execute_prediction_unit(row, food_info, engine, predictor_cache):
    """
    按食品分类码调度一个或多个标志物模型，汇总结果。
    predictor_cache: {'crayfish_tvbn': TVBNPredictor(), ...}
    返回：list[dict]
    """
    code = str(food_info.get("FoodClassificationCode"))
    handlers = ROUTES.get(code)
    if not handlers:
        return []
    # 兼容老配置：值若是单个函数而非列表，转成列表
    if callable(handlers):
        handlers = [handlers]
    out = []
    for handler in handlers:
        try:
            recs = handler(row, food_info, engine, predictor_cache)  # 约定返回 list[dict]
            if recs:
                out.extend(recs)
        except Exception as e:
            hname = getattr(handler, "__name__", str(handler))
            print(f"❌ 标志物执行失败（{code}/{hname}）: {e}")
    return out

# 温湿度过期判断并写入预测表 (逻辑,恢复循环内插入)
# ---------- 精简后的主函数 ----------

def handle_prediction_results(monitor_df, food_df, engine):
    """处理监控数据, 计算风险并写入预测表（跨批次复用模型实例）"""
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

    # TraCode -> food_info 一次映射
    cols = ["FoodClassificationCode", "StorTempLower", "StorTempUpper",
            "StorHumidLower", "StorHumidUpper", "ShelfLife", "ProDate"]
    #如果后续需要食品只是表里的其他字段则加入下面代码
    # cols = ["FoodClassificationCode", "SecondaryClassificationName",
    #         "StorTempLower", "StorTempUpper", "StorHumidLower", "StorHumidUpper",
    #         "ShelfLife", "TempAnonalyDuration", "HumidAnonalyDuration", "ProDate"]

    cols = [c for c in cols if c in food_df.columns]
    food_idx = (food_df.drop_duplicates(subset=["TraCode"], keep="first")
                .set_index("TraCode")[cols]
                .to_dict("index"))

    # ======== 1) 计算本批将会用到的食品分类码 ========
    batch_codes = set()
    for tra in df["TraCode"].unique():
        fi = food_idx.get(tra)
        if fi and fi.get("FoodClassificationCode"):
            batch_codes.add(str(fi["FoodClassificationCode"]))

    # ======== 2) 取全局缓存并准备风险判级器 ========
    RiskClassifier.refresh_from_db(engine)
    predictor_cache = _GLOBAL_PREDICTOR_CACHE  # 引用同一份全局字典
    #
    # # 新版 RiskClassifier：静态三元组 + 从数据库一次性全量写入成员变量
    # if predictor_cache.get("risk_classifier") is None:
    #     predictor_cache["risk_classifier"] = RiskClassifier()
    #
    # risk = predictor_cache["risk_classifier"]
    #
    # # 关键：全量加载（不传 food_codes，内部会把化学/生物两张表全部读出并写入 *low/*mid/*high 成员）
    # risk.load_from_db(engine)  # <- 全量

    # ======== 3) 按需补齐模型实例（已存在的直接复用） ========
    for code in batch_codes:
        for key, factory in PREDICTOR_FACTORIES.get(code, []):
            if key not in predictor_cache:      # 只在不存在时创建
                predictor_cache[key] = factory()

    # ======== 4) 主循环 ========
    results = []
    for _, r in df.sort_values(by=["TraCode", "RecTime"]).iterrows():
        try:
            row = {
                "TraCode": r["TraCode"],
                "RecTime": r["RecTime"],
                "OrderNumber": r.get("OrderNumber"),
                "Temp": r.get("Temp"),
                "Humid": r.get("Humid"),
                "MonitorNum": r.get("MonitorNum"),
            }
        except KeyError as e:
            print(f"⚠️ 缺少必要字段：{e}; 跳过该行。"); continue

        food = food_idx.get(row.get("TraCode"), {})  # 不存在时给空 dict，避免 KeyError
        if not food:
            continue

        # 过滤入库时间（只查一次）
        stime = get_storage_time(row["TraCode"], row["RecTime"], engine)
        if stime and row["RecTime"] < stime:
            continue

        order_chain = get_order_tra_chain(row["OrderNumber"], row["TraCode"], engine)

        # ① 温度
        rec_temp  = process_temperature(row, food, engine=engine, order_chain=order_chain)
        # ② 湿度
        rec_humid = process_humidity(row, food, engine=engine, order_chain=order_chain)
        # ③ 过期
        rec_shelf = process_shelf_life(row, food, engine=engine)
        # ④ 标志物（按路由执行；模型实例来自 predictor_cache，全局复用）
        marker_records = execute_prediction_unit(row, food, engine, predictor_cache)

        to_insert = [x for x in (rec_temp, rec_humid, rec_shelf) if x] + (marker_records or [])
        if to_insert:
            insert_results(to_insert, engine)
            results.extend(to_insert)

    print(f"✅ 预测结果处理完成。尝试生成 {len(results)} 条预测记录。")
    return results
