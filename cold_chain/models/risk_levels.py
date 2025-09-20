# models/risk_levels.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Iterable, Union, Callable
import pandas as pd

# flag -> 表名（兼容中英文）
_TABLE_MAP = {
    "chem": "chemicalmarker_risk_level",
    "化学": "chemicalmarker_risk_level",
    "bio":  "biomarker_risk_level",
    "生物":  "biomarker_risk_level",
}
_NEED_COLS = ["FoodClassificationCode", "LowRiskLine", "MiddleRiskLine", "HighRiskLine"]

def _norm_flag(flag: str) -> str:
    if flag in ("chem", "化学"): return "chem"
    if flag in ("bio", "生物"):  return "bio"
    raise ValueError(f"flag 必须是 chem/化学 或 bio/生物，收到：{flag}")

@dataclass
class RiskClassifier:
    """
    风险判级服务（新版）：
    - preload(): 一次性把阈值加载到内存缓存（动态）
    - get_thresholds(): 返回 (low, mid, high) 或 None
    - classify(): 直接返回 '无'/'低'/'中'/'高'/'未知'
    - set_static(): 为缺失分类设置静态兜底阈值
    """
    engine: any
    # 静态兜底阈值：{flag_norm: {food_code: (low, mid, high)}}
    static_overrides: Dict[str, Dict[str, Tuple[float, float, float]]] = field(default_factory=dict)
    # 内存缓存：{flag_norm: DataFrame(index=FoodClassificationCode, cols=Low/Middle/High)}
    _cache: Dict[str, pd.DataFrame] = field(default_factory=dict, init=False)

    # ---------- 预载 ----------
    def preload(
        self,
        *,
        flags: Iterable[str] = ("chem", "bio"),
        food_codes: Optional[Iterable[Union[str, int]]] = None,
        refresh: bool = False,
        read_sql: Callable[[str, Union[object, None]], pd.DataFrame] = pd.read_sql,
    ) -> None:
        """将需要的阈值一次性读入内存缓存。可指定仅加载部分 food_codes。"""
        food_set = None if food_codes is None else {str(x) for x in food_codes}
        for f in flags:
            nflag = _norm_flag(f)
            if (not refresh) and (nflag in self._cache):
                continue
            table = _TABLE_MAP[nflag]
            sql = f"SELECT {', '.join(_NEED_COLS)} FROM {table}"
            df = read_sql(sql, self.engine).copy()
            df = df[_NEED_COLS].drop_duplicates(subset=["FoodClassificationCode"]).set_index("FoodClassificationCode")
            for c in ("LowRiskLine", "MiddleRiskLine", "HighRiskLine"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            if food_set is not None:
                df = df.loc[df.index.intersection(food_set)]
            self._cache[nflag] = df

    # ---------- 查询 ----------
    def get_thresholds(self, food_code: Union[str, int], flag: str) -> Optional[Tuple[float, float, float]]:
        """优先从内存缓存取；没有则看静态兜底；都没有返回 None。"""
        nflag = _norm_flag(flag)
        code = str(food_code)

        df = self._cache.get(nflag)
        if df is not None and code in df.index:
            row = df.loc[code]
            low, mid, high = row["LowRiskLine"], row["MiddleRiskLine"], row["HighRiskLine"]
            if not (pd.isna(low) or pd.isna(mid) or pd.isna(high)):
                return float(low), float(mid), float(high)

        if nflag in self.static_overrides and code in self.static_overrides[nflag]:
            return self.static_overrides[nflag][code]
        return None

    # ---------- 判级 ----------
    def classify(self, food_code: Union[str, int], value: Optional[float], flag: str) -> str:
        """
        规则：value < low → '无'；[low, mid) → '低'；[mid, high) → '中'；>= high → '高'
        特殊：value is None → '无'；阈值缺失 → '未知'
        """
        if value is None:
            return "无"
        th = self.get_thresholds(food_code, flag)
        if th is None:
            return "未知"
        low, mid, high = th
        v = float(value)
        if v < low:  return "无"
        if v < mid:  return "低"
        if v < high: return "中"
        return "高"

    # ---------- 静态兜底 ----------
    def set_static(self, flag: str, food_code: Union[str, int], low: float, mid: float, high: float) -> None:
        """为缺失分类设置静态阈值；运行中也可更新。"""
        nflag = _norm_flag(flag)
        self.static_overrides.setdefault(nflag, {})[str(food_code)] = (float(low), float(mid), float(high))
