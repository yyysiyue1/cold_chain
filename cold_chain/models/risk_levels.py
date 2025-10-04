# models/risk_levels.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, ClassVar, Dict, Tuple
import threading
import pandas as pd

@dataclass
class RiskClassifier:
    """
    - 类属性保存各食品/标志物阈值（Java 风格：RiskClassifier.crayfish_tvbn_low）
    - 每批在 prediction_logic 里调用 refresh_from_db(engine)，把 DB 阈值写入这些类属性
    - 各模型里既可用 rc.classify(...)，也可以直接读类属性做 if 判级
    """

    # ====== 静态成员（按需增/改）======
    # C09006 小龙虾 TVBN（化学）
    crayfish_tvbn_low:  ClassVar[Optional[float]] = 15.0
    crayfish_tvbn_mid:  ClassVar[Optional[float]] = 25.0
    crayfish_tvbn_high: ClassVar[Optional[float]] = 30.0

    # B04072 生菜 假单胞菌（生物）
    lettuce_pseudomonas_low:  ClassVar[Optional[float]] = 10000.0
    lettuce_pseudomonas_mid:  ClassVar[Optional[float]] = 100000.0
    lettuce_pseudomonas_high: ClassVar[Optional[float]] = 1000000.0

    # B06031 三文鱼 菌落总数（生物）
    salmon_gbr_low:   ClassVar[Optional[float]] = 10000.0
    salmon_gbr_mid:   ClassVar[Optional[float]] = 50000.0
    salmon_gbr_high:  ClassVar[Optional[float]] = 1000000.0

    # B01001 猪肉多个标志物
    pork_tvbn_low:  ClassVar[Optional[float]] = 10.0
    pork_tvbn_mid:  ClassVar[Optional[float]] = 12.0
    pork_tvbn_high: ClassVar[Optional[float]] = 15.0

    pork_gbr_low:   ClassVar[Optional[float]] = 10000.0
    pork_gbr_mid:   ClassVar[Optional[float]] = 100000.0
    pork_gbr_high:  ClassVar[Optional[float]] = 100000.0

    pork_Methylquinoline_low:   ClassVar[Optional[float]] = 0
    pork_Methylquinoline_mid:   ClassVar[Optional[float]] = 0
    pork_Methylquinoline_high:  ClassVar[Optional[float]] = 0

    # B08005 巴氏杀菌乳 菌落总数（生物）
    pm_gbr_low:  ClassVar[Optional[float]] = 10000.0
    pm_gbr_mid:  ClassVar[Optional[float]] = 50000.0
    pm_gbr_high: ClassVar[Optional[float]] = 100000.0


    # 并发锁（防止并发刷新冲突）
    _lock: ClassVar[threading.Lock] = threading.Lock()

    # ------------ 规范化 flag ------------
    @staticmethod
    def _norm_flag(flag: str) -> str:
        if flag in ("chem", "化学"):
            return "chem"
        if flag in ("bio", "生物"):
            return "bio"
        raise ValueError(f"flag 必须是 chem/化学 或 bio/生物，收到：{flag}")

    # ------------ (flag, FoodCode, MarkerName) -> 类属性前缀 ------------
    # 这里的 MarkerName 必须与库里的显示一致：
    # - 生物表：BioMakerCN，如 “假单胞菌”、“菌落总数”
    # - 化学表：ChemicalMakerCN，如 “挥发性盐基氮”、“6-Methylquinoline”
    _ATTR_MAP: ClassVar[Dict[Tuple[str, str, str], str]] = {
        ("chem", "C09006", "挥发性盐基氮"): "crayfish_tvbn",

        ("bio",  "B04072", "假单胞菌"):   "lettuce_pseudomonas",

        ("bio",  "B06031", "菌落总数"):   "salmon_gbr",

        ("chem", "B01001", "挥发性盐基氮"): "pork_tvbn",            # 若化学表里也存了猪肉TVBN
        ("bio",  "B01001", "菌落总数"):     "pork_gbr",
        ("chem", "B01001", "6-Methylquinoline"): "pork_Methylquinoline",

        ("bio",  "B08005", "菌落总数"):     "pm_gbr",
    }

    # 允许实例化；engine 只是兼容旧签名
    engine: Optional[object] = None

    # ========== 每批调用：把两张表的阈值写入类属性 ==========
    @classmethod
    def refresh_from_db(cls, engine) -> None:
        """
        一次性把两张表全部读出，并写入类属性。表结构：
          - chemicalmarker_risk_level: ChemicalMakerCN, FoodClassificationCode, LowRiskLine, MiddleRiskLine, HighRiskLine
          - biomarker_risk_level     : BioMakerCN,      FoodClassificationCode, LowRiskLine, MiddleRiskLine, HighRiskLine
        """
        with cls._lock:
            chem_sql = """
                SELECT
                    FoodClassificationCode,
                    ChemicalMakerCN AS RiskName,
                    LowRiskLine, MiddleRiskLine, HighRiskLine
                FROM chemicalmarker_risk_level
            """
            bio_sql = """
                SELECT
                    FoodClassificationCode,
                    BioMakerCN AS RiskName,
                    LowRiskLine, MiddleRiskLine, HighRiskLine
                FROM biomarker_risk_level
            """

            chem_df = pd.read_sql(chem_sql, engine)
            bio_df  = pd.read_sql(bio_sql,  engine)

            chem_df["flag"] = "chem"
            bio_df["flag"]  = "bio"
            df = pd.concat([chem_df, bio_df], ignore_index=True)

            # 转数值
            for c in ("LowRiskLine", "MiddleRiskLine", "HighRiskLine"):
                df[c] = pd.to_numeric(df[c], errors="coerce")

            # 写回类属性
            updated = 0
            for _, row in df.iterrows():
                flag   = cls._norm_flag(row["flag"])
                fcode  = str(row["FoodClassificationCode"]).strip()
                rname  = (str(row["RiskName"]).strip()
                          if pd.notna(row["RiskName"]) else "")
                key    = (flag, fcode, rname)
                prefix = cls._ATTR_MAP.get(key)
                if not prefix:
                    continue
                low, mid, high = row["LowRiskLine"], row["MiddleRiskLine"], row["HighRiskLine"]
                if pd.isna(low) or pd.isna(mid) or pd.isna(high):
                    continue
                setattr(cls, f"{prefix}_low",  float(low))
                setattr(cls, f"{prefix}_mid",  float(mid))
                setattr(cls, f"{prefix}_high", float(high))
                updated += 1
            print(f"✅ RiskClassifier: 阈值刷新完成，写入 {updated} 条。")





