# train_pm_gbr_interp.py
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib  # 也可用 pickle

warnings.filterwarnings('ignore')

# --------------------
# 1) 配置
# --------------------
from pathlib import Path
BASE_DIR   = Path(__file__).resolve().parent           # 当前文件所在目录
data_file  = BASE_DIR / '巴氏杀菌乳_data_all_temps_monotonic - 副本.csv'
temp_col   = 'temp'
time_col   = 'time_hours_interp'
target_col = 'hanliang_log10'

# 输出 pkl（注意：这是“插值网格”版本，不含 sklearn 模型）
output_dir = BASE_DIR                                  # 也可以改成 BASE_DIR / 'models_out'
output_pkl = output_dir / 'pm_gbr_model.pkl'

os.makedirs(output_dir, exist_ok=True)

# --------------------
# 2) 加载数据
# --------------------
df = pd.read_csv(data_file)
if df.empty:
    raise RuntimeError("输入数据为空。")

# 参与训练的温度集合
temps = sorted(df[temp_col].unique())
low_temps = [0, 1, 2]          # 与 4℃ 共用曲线
models_sklearn = {}            # 仅在训练阶段使用
results_rows = []              # 评估指标
interp_models = {}             # 要导出的纯 numpy 插值网格

# --------------------
# 3) 先训练 4℃ 模型
# --------------------
if 4 in temps:
    d4 = df[df[temp_col] == 4].dropna(subset=[time_col, target_col])
    X4 = d4[[time_col]].values
    y4 = d4[target_col].values

    m4 = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=3,
        min_samples_split=5, min_samples_leaf=2,
        loss='squared_error', random_state=42
    )
    m4.fit(X4, y4)
    models_sklearn[4] = m4

    # 为 4℃ 构建插值网格
    tmin, tmax = float(X4.min()), float(X4.max())
    t_grid = np.linspace(tmin, tmax, 201).reshape(-1, 1)
    y_grid = m4.predict(t_grid).astype(float)
    interp_models[4] = {
        "t_min": tmin, "t_max": tmax,
        "time_grid": t_grid.ravel().astype(float),
        "value_grid": y_grid
    }

    # 4℃ 的训练评估
    y_pred4 = m4.predict(X4)
    results_rows.append({
        "温度": 4,
        "MAE": mean_absolute_error(y4, y_pred4),
        "RMSE": mean_squared_error(y4, y_pred4, squared=False),
        "R2": r2_score(y4, y_pred4),
        "模型来源": "独立模型"
    })
else:
    raise RuntimeError("数据中缺少 4℃，无法完成 0–2℃ 的复用逻辑。")

# --------------------
# 4) 0–2℃ 完全复用 4℃ 的网格
# --------------------
for t in low_temps:
    # 直接复用 4℃ 的插值网格（拷贝一份，避免后续修改时相互影响）
    base = interp_models[4]
    interp_models[t] = {
        "t_min": base["t_min"], "t_max": base["t_max"],
        "time_grid": base["time_grid"].copy(),
        "value_grid": base["value_grid"].copy(),
    }
    # 评估指标用 4℃ 数据评估即可（只是记录用途）
    results_rows.append({
        "温度": t,
        "MAE": results_rows[0]["MAE"],
        "RMSE": results_rows[0]["RMSE"],
        "R2": results_rows[0]["R2"],
        "模型来源": "4℃复用"
    })

# --------------------
# 5) 训练其它温度，并生成插值网格
# --------------------
for t in temps:
    if t in low_temps or t == 4:
        continue
    sub = df[df[temp_col] == t].dropna(subset=[time_col, target_col])
    if len(sub) < 3:
        # 数据太少就跳过；也可以考虑复用最接近温度的模型
        continue

    X = sub[[time_col]].values
    y = sub[target_col].values

    m = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=3,
        min_samples_split=5, min_samples_leaf=2,
        loss='squared_error', random_state=42
    )
    m.fit(X, y)
    models_sklearn[t] = m

    # 构建插值网格
    tmin, tmax = float(X.min()), float(X.max())
    t_grid = np.linspace(tmin, tmax, 201).reshape(-1, 1)
    y_grid = m.predict(t_grid).astype(float)
    interp_models[t] = {
        "t_min": tmin, "t_max": tmax,
        "time_grid": t_grid.ravel().astype(float),
        "value_grid": y_grid
    }

    # 记录指标
    y_pred = m.predict(X)
    results_rows.append({
        "温度": t,
        "MAE": mean_absolute_error(y, y_pred),
        "RMSE": mean_squared_error(y, y_pred, squared=False),
        "R2": r2_score(y, y_pred),
        "模型来源": "独立模型"
    })

# --------------------
# 6) 导出轻量 pkl（仅插值网格 + 元数据）
# --------------------
results_df = pd.DataFrame(results_rows).sort_values(by="温度")
bundle = {
    "interp_models": interp_models,      # {temp: {"time_grid": ndarray, "value_grid": ndarray, "t_min": float, "t_max": float}}
    "metadata": {
        "description": "PM-APC per-temperature interpolation grids (no sklearn objects).",
        "train_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "time_col": time_col,
        "target_col": target_col,
        "temps": sorted(list(interp_models.keys()))
    },
    "metrics": results_df
}

joblib.dump(bundle, output_pkl)
print(f"✅ 轻量模型已保存：{output_pkl}")
print("覆盖温度：", bundle["metadata"]["temps"])
print("\n拟合指标：\n", results_df.to_string(index=False))
