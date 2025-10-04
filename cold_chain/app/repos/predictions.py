# app/repos/predictions.py
import pandas as pd
from sqlalchemy import text
from db.database_setup import prediction_table_name
from app.records import replace_nan_with_none

# 将预测记录插入预测结果表
def insert_results(results, engine):
    """将预测结果列表插入或更新到 risk_prediction_results 表 """
    if not results:
        # print("⚠️ 当前批次无数据可入库") # 打印信息
        return []  # 返回空列表

    # 过滤掉 None 值 (如果 generate 函数返回 None) (保留优化)
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        # print("⚠️ 当前批次无有效数据可入库")
        return []

    if not isinstance(valid_results, list) or not all(isinstance(item, dict) for item in valid_results):
        # 代码会直接抛出 ValueError
        raise ValueError("❌ insert_results 接收的是一个由 dict 构成的列表,例如：[record1, record2]")

    result_df = pd.DataFrame(valid_results)

    # 代码没有预处理和检查 DataFrame

    insert_sql = f"""
    INSERT INTO {prediction_table_name} (
        PredictResultID, MonitorNum, OrderNumber, TraCode, RecTime,
        FoodClassificationCode, Temp, Humid, PredictFlag, RiskName,
        PredictValue, Unit, RiskLevel
    ) VALUES (
        :PredictResultID, :MonitorNum, :OrderNumber, :TraCode, :RecTime,
        :FoodClassificationCode, :Temp, :Humid, :PredictFlag, :RiskName,
        :PredictValue, :Unit, :RiskLevel
    )
    ON DUPLICATE KEY UPDATE
        PredictValue = VALUES(PredictValue),
        -- RiskLevel = VALUES(RiskLevel), # 代码没有更新 RiskLevel
        RecTime = VALUES(RecTime);
        -- 代码只更新 PredictValue 和 RecTime
    """

    # 代码使用 replace_nan_with_none
    insert_data_list = [replace_nan_with_none(d) for d in result_df.to_dict(orient='records')]

    try:
        with engine.begin() as conn:
            for data in insert_data_list:
                # 代码没有检查 data 有效性
                conn.execute(text(insert_sql), data)
        # print(f"✅ 插入记录数：{len(result_df)}") # 打印信息 (可能不准确)
    except Exception as e:
        print("❌ 插入失败：", e)  # 打印信息
        # 代码在失败时不返回任何东西,这里保持返回列表
        return results  # 或者可以返回 [] 表示失败

    return results  # 代码返回传入的 results

