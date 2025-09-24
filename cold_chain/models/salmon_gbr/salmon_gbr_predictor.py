# cold_chain/models/salmon_gbr/salmon_gbr_predictor.py
from __future__ import annotations
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

# 默认的模型文件名
DEFAULT_MODEL_FILENAME = "salmon_gbr_models.pkl"

class SalmonGBRPredictor:
    """
    三文鱼菌落总数预测器。
    该预测器使用基于梯度提升的增量模型 (delta model)。
    它会根据当前温度，选择最接近的预训练模型来进行预测。
    """
    def __init__(self, model_path="salmon_gbr_models.pkl") :
        """
        初始化预测器并加载模型。
        :param model_path: 模型文件路径。如果为 None，则使用默认路径。
        """
        base_dir = Path(__file__).resolve().parent
        p = Path(model_path) if model_path else Path(DEFAULT_MODEL_FILENAME)
        abs_path = p if p.is_absolute() else (base_dir / p)

        if not abs_path.is_file():
            raise FileNotFoundError(f"❌ 三文鱼模型文件未找到: {abs_path}")

        try:
            model_payload = joblib.load(str(abs_path))
            self.delta_models: Dict[float, Any] = model_payload['delta_models']
            self.training_temps = sorted(self.delta_models.keys())
            self.metadata = model_payload.get('metadata', {})
            print(f"✅ 三文鱼模型已成功加载: {abs_path}")
            print(f"   包含 {len(self.training_temps)} 个温度的模型: {self.training_temps}")
        except Exception as e:
            raise RuntimeError(f"❌ 加载三文鱼模型失败: {abs_path}\n原因: {e}") from e

    def _find_closest_model(self, current_temp: float):
        """
        根据当前温度，找到最接近的已训练模型。
        """
        model_temp_key = min(self.training_temps, key=lambda x: abs(x - current_temp))
        return self.delta_models[model_temp_key]

    def predict_single_step(self, time_h: float, temp_c: float, prev_log10_conc: float) -> float:
        """
        单步预测：输入时刻 t、小环境温度 T、上一时刻菌落数 log10 值，输出当前时刻 log10 值
        """
        model = self._find_closest_model(float(temp_c))

        # —— 统一用 DataFrame 并带上训练时的列名，避免 sklearn 的特征名告警
        # 先从 metadata 里拿（如果你训练阶段把列名存进了 metadata['feature_names']）
        feature_names = None
        if isinstance(getattr(self, "metadata", None), dict):
            feature_names = self.metadata.get("feature_names")

        # 再尝试从模型对象上拿（sklearn >=1.0 会有）
        if not feature_names and hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)

        # 最后兜底成你训练时用的两个列名（按你的注释）
        if not feature_names:
            feature_names = ["time", "prev_target"]

        X = pd.DataFrame(
            [[float(time_h), float(prev_log10_conc)]],
            columns=feature_names
        )

        predicted_delta = float(np.asarray(model.predict(X)).ravel()[0])
        return float(prev_log10_conc) + predicted_delta

    def __repr__(self) -> str:
        return f"SalmonGBRPredictor(trained_temps={self.training_temps})"