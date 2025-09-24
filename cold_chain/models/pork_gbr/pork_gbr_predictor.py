# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import  Optional
_EPS = 1e-9  # 防止除零/取对数等数值问题

def _to_ts(x):
    """任何时间输入 -> pandas.Timestamp（若已是 Timedelta 则返回 None 交给外面处理）"""
    if isinstance(x, pd.Timedelta):
        return None
    return pd.to_datetime(x, errors="coerce")

def _hours_between(t_now, t_prev, *, eps=_EPS) -> float:
    """
    两个“时间点”之间的小时差：负数→0；0→eps；缺失→0
    支持 str / datetime / pd.Timestamp / numpy.datetime64
    """
    t_now = _to_ts(t_now)
    t_prev = _to_ts(t_prev)
    if t_now is None or t_prev is None or pd.isna(t_now) or pd.isna(t_prev):
        return 0.0
    dh = (t_now - t_prev).total_seconds() / 3600.0
    if dh < 0:
        return 0.0
    if dh == 0:
        return float(eps)
    return float(dh)

DEFAULT_MODEL_FILENAME = "pork_gbr_model.pkl"
class Pork_gbr_Predictor:
    def __init__(self, model_pkl: Optional[str] = None) -> None:
        """
        如果传相对路径，则相对于本文件所在目录解析；
        若不传，则默认加载同目录下 DEFAULT_MODEL_FILENAME。
        """
        base_dir = Path(__file__).resolve().parent  # .../models/pork_tvbn
        p = Path(model_pkl) if model_pkl else Path(DEFAULT_MODEL_FILENAME)
        abs_path = p if p.is_absolute() else (base_dir / p)

        if not abs_path.is_file():
            raise FileNotFoundError(f"❌ 模型文件未找到: {abs_path}")

        bundle = joblib.load(str(abs_path))
        self.model_r  = bundle["model_r"]
        self.poly     = bundle["poly"]
        self.r_at_4   = float(bundle["r_at_4"])
        self.r_floor  = float(bundle["r_floor"])
        self.K_fixed  = float(bundle["K_fixed"])
        self.C0       = float(bundle["C0_default"])

    def predict_step(self, C_prev: float, T: float, t_prev, t) -> float:
        """
        单步预测（步长以“小时”计）：
          C(t) = K / (1 + (K/C_prev - 1) * exp(-r * dt_hours))

        参数：
          C_prev : 上一步 TVBN 浓度
          T      : 当前温度(°C)
          t_prev : 上一步时间（任意可解析成时间点的类型）
          t      : 当前时间（任意可解析成时间点的类型）

        备注：
          - 假设 r 的单位是 “每小时”，若你的训练里是“每天”，请把 r/24.
        """
        # 1) 步长（小时）
        dt_h = _hours_between(t, t_prev)  # -> float hours, 已处理 0/负数
        # 调试输出（可注释）
        print(f"gbr-----dt_h={dt_h:.6f}h, T={T}, C_prev={C_prev}, t={t}, t_prev={t_prev}")

        # 2) 保护 C_prev
        if C_prev is None or not np.isfinite(C_prev) or C_prev <= 0:
            C_prev = max(self.C0, _EPS)

        # 3) 增长率 r（单位：/h）
        T = float(T) if T is not None and np.isfinite(T) else 4.0
        if T < 4.0:
            r = self.r_at_4
        else:
            r = float(self.model_r.predict(self.poly.transform([[T]]))[0])
            r = max(r, self.r_floor)

        # 4) 承载量
        K = max(float(self.K_fixed), C_prev + _EPS)  # K 至少要大于当前值一点点

        # 5) Logistic 单步更新（使用小时步长）
        #    当 dt_h 很小时，exp(-r*dt_h) ~ 1 - r*dt_h，不会除零
        Ct = K / (1.0 + (K / C_prev - 1.0) * np.exp(-r * dt_h))

        # 调试输出（可注释）
        print(f"K={K}, Ct={Ct}, r={r}")
        return float(Ct)
