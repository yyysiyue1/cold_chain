# cold_chain/models/salmon_gbr/train_salmon_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
import os
import joblib # 用于保存模型

# --- 初始设置 ---
warnings.filterwarnings('ignore')
# 设置中文字体，确保图表能正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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

# --- 结果存储设置 ---
# 训练过程中的图表和指标会保存在这个文件夹
output_dir = 'salmon_training_results'
os.makedirs(output_dir, exist_ok=True)

# 用于存储模型和结果的字典
delta_models = {}
static_models = {}
static_results = []
pseudo_dynamic_results = []


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
# ✅ 步骤 3: (线下验证) 统一进行静态和伪动态验证
# =====================================================================================
print("\n--- 开始统一验证所有已训练的模型 ---")

for temp in temps:
    if temp not in static_models or temp not in delta_models:
        continue # 如果某个温度的模型没训练成功，则跳过

    print(f"\n--- 正在验证温度: {temp}°C ---")
    temp_data = original_data[original_data[temp_col] == temp].copy()
    t_real = temp_data[time_col].values
    y_real = temp_data[target_col].values

    # --- 1. 静态验证 ---
    static_model = static_models[temp]
    y_pred_static = static_model.predict(temp_data[[time_col]])
    mae_s = mean_absolute_error(y_real, y_pred_static)
    mse_s = mean_squared_error(y_real, y_pred_static)
    r2_s = r2_score(y_real, y_pred_static)
    static_results.append({'温度': temp, 'MAE': mae_s, 'MSE': mse_s, 'R2': r2_s})
    print(f"   📊 静态验证指标: MAE={mae_s:.4f}, MSE={mse_s:.4f}, R²={r2_s:.4f}")

    # --- 2. 伪动态验证 ---
    # 模拟真实预测场景：用上一个点的预测结果来推算下一个点
    delta_model = delta_models[temp]
    pseudo_dynamic_preds = [y_real[0]] # 初始值使用真实值
    current_target = y_real[0]

    for i in range(len(t_real) - 1):
        t_current = t_real[i]
        X_pred_delta = np.array([[t_current, current_target]])
        predicted_delta = delta_model.predict(X_pred_delta)[0]
        next_target = current_target + predicted_delta
        pseudo_dynamic_preds.append(next_target)
        current_target = next_target # 更新当前浓度为预测值

    pseudo_dynamic_preds = np.array(pseudo_dynamic_preds)
    mae_pd = mean_absolute_error(y_real, pseudo_dynamic_preds)
    mse_pd = mean_squared_error(y_real, pseudo_dynamic_preds)
    r2_pd = r2_score(y_real, pseudo_dynamic_preds)
    pseudo_dynamic_results.append({'温度': temp, 'MAE': mae_pd, 'MSE': mse_pd, 'R2': r2_pd})
    print(f"   📈 伪动态验证指标: MAE={mae_pd:.4f}, MSE={mse_pd:.4f}, R²={r2_pd:.4f}")

    # --- 3. 统一绘图 ---
    plt.figure(figsize=(10, 6))
    plt.scatter(t_real, y_real, color='blue', label='实测数据', alpha=0.7, zorder=5)
    plt.plot(t_real, y_pred_static, color='red', linestyle='--', label=f'静态拟合 (R²={r2_s:.2f})')
    plt.plot(t_real, pseudo_dynamic_preds, color='green', linestyle=':', label=f'伪动态预测 (R²={r2_pd:.2f})')
    plt.title(f'模型验证对比 @ {temp}°C')
    plt.xlabel('时间 (小时)')
    plt.ylabel('常用对数浓度 (log10 CFU/g)')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.savefig(os.path.join(output_dir, f'validation_plot_{temp}C.png'))
    plt.close()

# --- 保存指标汇总 ---
static_results_df = pd.DataFrame(static_results)
pseudo_dynamic_results_df = pd.DataFrame(pseudo_dynamic_results)
static_results_df.to_csv(os.path.join(output_dir, 'static_validation_metrics.csv'), index=False, encoding='utf_8_sig')
pseudo_dynamic_results_df.to_csv(os.path.join(output_dir, 'pseudo_dynamic_validation_metrics.csv'), index=False, encoding='utf_8_sig')
print("\n--- 验证指标和对比图已保存至文件夹:", output_dir)


# =====================================================================================
# ✅ 步骤 4: 保存用于生产环境的模型
# =====================================================================================
# 这是最关键的一步：将训练好的、用于预测的 delta_models 保存成一个 pkl 文件。
# 主系统将加载这个文件来执行预测。

if delta_models:
    # 模型将保存在当前目录下
    model_filename = 'salmon_gbr_models.pkl'

    # 创建一个包含模型和元数据的字典，方便未来追溯
    model_payload = {
        'delta_models': delta_models,
        'training_temps': list(delta_models.keys()),
        'metadata': {
            'description': '三文鱼菌落总数增量预测模型 (梯度提升)',
            'model_type': 'delta_prediction',
            'train_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }

    try:
        joblib.dump(model_payload, model_filename)
        print(f"\n✅ [核心] 生产模型已成功保存到: {model_filename}")
    except Exception as e:
        print(f"\n❌ 保存生产模型失败: {e}")
else:
    print("\n⚠️ 没有任何模型被训练成功，无法保存生产模型文件。")

print("\n--- 全部流程完成！ ---")