# models/pm_gbr/PM_gbr_predictor.py
from __future__ import annotations
import os
import joblib
import numpy as np

class PMAPCPredictor:
    """
    极简版 PM-APC 预测器：
    - 仅加载训练脚本导出的 interp_models（每个温度一条 time->value 曲线）
    - 单步预测：next = prev_conc + [y_T(t_curr) - y_T(t_prev)]
    """

    def __init__(self, model_path: str = "pm_gbr_model.pkl"):
        self.models: dict[int, dict] = {}   # {temp: {"time_grid": ndarray, "value_grid": ndarray}}
        self._load_interp_models(model_path)

    # ---------------- 加载 ----------------
    def _load_interp_models(self, model_path: str) -> None:
        """读取 pkl 中的 interp_models -> self.models"""
        # 兼容相对路径：以当前文件夹为基准
        abs_path = os.path.join(os.path.dirname(__file__), model_path)
        if not os.path.exists(abs_path):
            abs_path = model_path
        bundle = joblib.load(abs_path)

        interp = bundle.get("interp_models", None)
        if not isinstance(interp, dict):
            # 兼容：若顶层就是温度->网格字典
            interp = {k: v for k, v in bundle.items()
                      if isinstance(v, dict) and "time_grid" in v and "value_grid" in v}

        # 规范化为 int 温度键，并且只留必要字段
        models = {}
        for k, v in (interp or {}).items():
            try:
                tkey = int(round(float(k)))
                tg = np.asarray(v["time_grid"], dtype=float).ravel()
                vg = np.asarray(v["value_grid"], dtype=float).ravel()
                if tg.size and tg.size == vg.size:
                    # 确保 time_grid 单调递增
                    order = np.argsort(tg)
                    models[tkey] = {"time_grid": tg[order], "value_grid": vg[order]}
            except Exception:
                continue
        self.models = models

    # ---------------- 工具：在一条曲线上插值 ----------------
    @staticmethod
    def _interp_on_curve(curve: dict, t: float) -> float:
        tg = curve["time_grid"]; vg = curve["value_grid"]
        # 超出范围时做截断后插值
        t_clip = float(np.clip(t, tg.min(), tg.max()))
        return float(np.interp(t_clip, tg, vg))

    # ---------------- 单步预测 ----------------
    def predict_single_step(
        self,
        prev_time: float,
        prev_conc_log10: float,
        current_time: float,
        current_temp: float,
    ) -> float:
        """
        prev_time / current_time: 小时(float)
        prev_conc_log10: 上一步浓度（log10 CFU/g）
        current_temp: 当前温度(°C)
        """
        if not self.models:
            return float(prev_conc_log10)

        # 非正时间步长直接返回
        if current_time <= prev_time:
            return float(prev_conc_log10)

        # 选择最接近的温度曲线
        tkey = int(round(current_temp))
        if tkey not in self.models:
            tkey = min(self.models.keys(), key=lambda x: abs(x - current_temp))
        curve = self.models[tkey]

        # 绝对值曲线的差分作为“增量”
        y_prev = self._interp_on_curve(curve, float(prev_time))
        y_curr = self._interp_on_curve(curve, float(current_time))
        delta = y_curr - y_prev

        return float(prev_conc_log10) + float(delta)
