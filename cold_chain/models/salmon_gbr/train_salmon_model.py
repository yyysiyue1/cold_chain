# cold_chain/models/salmon_gbr/train_salmon_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
import os
import joblib # 用于保存模型


# =====================================================================================
# ✅ 步骤 1: 参数配置与数据加载
# =====================================================================================

# --- 文件与列名适配 ---
# 请确保这个数据文件与本脚本在同一个目录下
data_file = 'Baranyi_all_plot_data.csv'
temp_col = 'temperature'
time_col = 'time'
concentration_col = 'concentration'
target_col = 'log10_concentration'

try:
    original_data = pd.read_csv(data_file)
    # 筛选出真实测量数据进行训练
    original_data = original_data[original_data['type'] == '实测'].copy()
    # 计算目标列：浓度的常用对数
    original_data[target_col] = np.log10(original_data[concentration_col] + 1e-9)
    print(f"成功加载并适配数据 '{data_file}'.")
except FileNotFoundError:
    print(f"错误：找不到文件 '{data_file}'。请确保它和脚本在同一目录。")
    exit()


# 用于存储模型和结果的字典
delta_models = {}
static_models = {}
static_results = []
pseudo_dynamic_results = []

FEATURE_NAMES = [time_col, 'prev_target']
# =====================================================================================
# ✅ 步骤 2: 为每个温度独立训练模型
# =====================================================================================

temps = sorted(original_data[temp_col].unique())

print("\n--- 开始为每个温度独立训练模型 ---")
for temp in temps:
    print(f"\n--- 正在处理温度: {temp}°C ---")
    temp_data = original_data[original_data[temp_col] == temp].copy()

    if len(temp_data) < 5:
        print("   数据点过少，跳过。")
        continue

    # --- 训练“增量预测”模型 (Delta Model) ---
    # 这是预测系统的核心，预测的是下一个时间点的浓度 *增量*
    temp_data['prev_target'] = temp_data[target_col].shift(1)
    train_df = temp_data.dropna()

    if len(train_df) < 2:
        print(f"   为增量模型准备的数据不足，跳过温度 {temp}°C。")
        continue

    X_delta = train_df[['time', 'prev_target']]
    y_delta = train_df[target_col] - train_df['prev_target']

    delta_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    delta_model.fit(X_delta, y_delta)
    delta_models[temp] = delta_model
    print(f"   ✅ 增量预测模型 (delta model) for {temp}°C 训练完成。")

    # --- 训练“静态拟合”模型 (用于线下验证对比) ---
    # 这个模型直接根据时间预测浓度，用于评估和绘图
    X_static = temp_data[[time_col]]
    y_static = temp_data[target_col]

    static_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    static_model.fit(X_static, y_static)
    static_models[temp] = static_model
    print(f"   ✅ 静态拟合模型 for {temp}°C 训练完成。")


# =====================================================================================
# ✅ 步骤 4: 保存用于生产环境的模型
# =====================================================================================
if delta_models:
    model_filename = 'salmon_gbr_models.pkl'

    # 统一温度键为 float，避免后续比较时类型不一致
    delta_models = {float(k): v for k, v in delta_models.items()}
    training_temps_sorted = sorted(delta_models.keys())

    model_payload = {
        'delta_models': delta_models,                 # {温度: 已拟合的 GBDT 增量模型}
        'training_temps': training_temps_sorted,      # 训练过的温度点（已排序）
        'metadata': {
            'description': '三文鱼菌落总数增量预测模型 (梯度提升)',
            'model_type': 'delta_prediction',
            'train_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            # ✅ 关键：推理端会优先读取这个来构造带列名的 DataFrame，消除 sklearn 告警
            'feature_names': FEATURE_NAMES,           # 例如 ['time', 'prev_target']
            # （可选）记录环境信息，方便回溯
            'library_versions': {
                'sklearn': __import__('sklearn').__version__,
                'numpy':   np.__version__,
                'pandas':  pd.__version__,
            }
        }
    }

    try:
        joblib.dump(model_payload, model_filename)
        print(f"\n✅ [核心] 生产模型已成功保存到: {model_filename}")
        print(f"   训练温度点: {training_temps_sorted}")
        print(f"   特征列名: {FEATURE_NAMES}")
    except Exception as e:
        print(f"\n❌ 保存生产模型失败: {e}")
else:
    print("\n⚠️ 没有任何模型被训练成功，无法保存生产模型文件。")


print("\n--- 全部流程完成！ ---")