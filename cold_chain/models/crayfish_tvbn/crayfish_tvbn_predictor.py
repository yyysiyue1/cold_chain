# -*- coding: utf-8 -*-
"""
crayfish_tvbn_predictor.py
封装 TVBN 模型的加载、单点预测、多项式曲线与动态预测（小龙虾）。

用法示例：
    pred = Crayfish_TVBN_Predictor()  # 默认加载同目录 pkl
    y = pred.predict(time_h=12, temp_c=4)
    y2 = pred.calculate_dynamic_value(last_value=8.0, last_time=t1, rec_time=t2, current_temp=4)
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import joblib

DEFAULT_MODEL_FILENAME = "crayfish_tvbn_predictor_model.pkl"


class Crayfish_TVBN_Predictor:
    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        初始化预测器并加载模型。
        :param model_path: 模型文件路径。None 或未传 -> 使用本文件同目录的默认 pkl。
                           传入相对路径时，将相对于本文件目录解析，而不是 CWD。
        """
        base_dir = Path(__file__).resolve().parent  # .../models/crayfish_tvbn
        p = Path(model_path) if model_path else Path(DEFAULT_MODEL_FILENAME)
        abs_path = p if p.is_absolute() else (base_dir / p)

        if not abs_path.is_file():
            raise FileNotFoundError(f"❌ 模型文件未找到: {abs_path}")

        try:
            self.model = joblib.load(str(abs_path))
        except Exception as e:
            raise RuntimeError(f"❌ 加载模型失败: {abs_path}\n原因: {e}") from e

        # 缓存：key=(temps_tuple, time_max, poly_order)
        self._poly_models_cache: Dict[Tuple[Tuple[float, ...], int, int], Dict[float, np.poly1d]] = {}
        print(f"✅ 模型已成功加载: {abs_path}")

    # --------------------------
    # 特征工程
    # --------------------------
    @staticmethod
    def make_features(time_h: float, temp_c: float) -> np.ndarray:
        """根据时间(h)与温度(°C)构建特征。"""
        t = float(time_h)
        c = float(temp_c)
        feats = np.array(
            [[t, np.log1p(t), c, c ** 2, t * c]],
            dtype=float
        )
        return feats

    # --------------------------
    # 单点推理
    # --------------------------
    def predict(self, time_h: float, temp_c: float) -> float:
        """单点预测 TVBN 含量 (mg/100g)。"""
        features = self.make_features(time_h, temp_c)
        y = self.model.predict(features)
        return float(np.asarray(y).ravel()[0])

    # --------------------------
    # 多项式近似（用于绘曲线/搜索）
    # --------------------------
    @staticmethod
    def _build_poly_models_core(model, temps: Iterable[float], time_max: int = 2000, poly_order: int = 3) -> Dict[float, np.poly1d]:
        """
        为每个温度拟合多项式 (time -> TVB-N)，返回 {temp: poly_model}
        """
        poly_models: Dict[float, np.poly1d] = {}
        time_grid = np.arange(0, int(time_max) + 1)

        temps = [float(x) for x in temps]
        for temp in temps:
            # 构造输入特征
            features = np.array([[t, np.log1p(t), temp, temp ** 2, t * temp] for t in time_grid], dtype=float)
            amines = np.asarray(model.predict(features)).ravel()

            # 多项式拟合
            coeffs = np.polyfit(time_grid, amines, int(poly_order))
            poly_models[temp] = np.poly1d(coeffs)

        return poly_models

    def get_poly_models(self, temps: Iterable[float], time_max: int = 2000, poly_order: int = 3) -> Dict[float, np.poly1d]:
        """
        获取或构建多项式拟合曲线，避免重复计算。
        """
        temps_tuple = tuple(sorted(float(x) for x in temps))
        key = (temps_tuple, int(time_max), int(poly_order))
        if key not in self._poly_models_cache:
            print(f"⚡ 缓存未命中，重建多项式：temps={temps_tuple}, time_max={time_max}, order={poly_order}")
            self._poly_models_cache[key] = self._build_poly_models_core(self.model, temps_tuple, time_max, poly_order)
        else:
            print(f"⚡ 缓存命中：temps={temps_tuple}, time_max={time_max}, order={poly_order}")
        return self._poly_models_cache[key]

    # --------------------------
    # 动态预测（多项式导数近似）
    # --------------------------
    def calculate_dynamic_value(
        self,
        last_value: float,
        last_time: pd.Timestamp,
        rec_time: pd.Timestamp,
        current_temp: float,
        *,
        poly_models: Optional[Dict[float, np.poly1d]] = None,
        poly_order: int = 3,
        max_rate: float = 0.3,
        time_max: int = 2000,
    ) -> float:
        """
        TVB-N 动态含量计算（基于多项式导数的增量近似）

        参数
        ----
        last_value : 上一次的 TVB-N 含量
        last_time  : 上一次监测时间
        rec_time   : 当前记录时间
        current_temp : 当前温度
        poly_models : 预构建的 {温度: 多项式}；若 None 则按常用范围临时构建
        poly_order  : 多项式阶数
        max_rate    : 速率上限 (mg/100g/h)
        time_max    : 拟合时间上限（小时）
        """
        # 1) 时间差
        if pd.isna(last_time) or pd.isna(rec_time):
            return float(last_value)
        dt_h = (pd.to_datetime(rec_time) - pd.to_datetime(last_time)).total_seconds() / 3600.0
        if dt_h <= 0:
            return float(last_value)

        # 2) 多项式集合
        if poly_models is None:
            temps = np.linspace(-20, 10, 31)  # 依据你的温区调整
            poly_models = self.get_poly_models(temps, time_max=time_max, poly_order=poly_order)

        # 3) 选择最接近温度的多项式
        temp_keys = list(poly_models.keys())
        closest_temp = min(temp_keys, key=lambda x: abs(x - float(current_temp)))
        poly = poly_models[closest_temp]

        # 4) 用导数估算增速，并截断到 [0, max_rate]
        t_eff = min(int(time_max), int(max(dt_h, 1)))  # 简单取有效 t
        rate = float(np.clip(poly.deriv()(t_eff), 0.0, float(max_rate)))

        # 5) 累积
        return float(max(last_value + rate * dt_h, last_value))

    # --------------------------
    # 友好显示
    # --------------------------
    def __repr__(self) -> str:
        return f"TVBNPredictor(model=<{self.model.__class__.__name__}>)"
