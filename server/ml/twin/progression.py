"""
server/ml/twin/progression.py
=============================
Inference module for Model B (progression forecasting).
Loads trained CatBoost regression and conversion models from ml/twin/weights.
Provides:
  - predict_forecast(current_visit: dict) -> dict of horizon -> ForecastPoint
  - simulate_counterfactual(current_visit: dict, deltas: dict) -> dict
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier

from ml.twin.constants import FEATURE_TIERS, DX_MAP

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
HORIZONS = [6, 12, 24]

# MAE bounds from exp10 validation for 90% prediction intervals (~1.64 * MAE * 1.25)
HORIZON_RESIDUAL_MAE = {
    6:  {"MMSE": 1.391, "CDGLOBAL": 0.064, "TOTSCORE": 2.989},
    12: {"MMSE": 1.474, "CDGLOBAL": 0.115, "TOTSCORE": 3.095},
    24: {"MMSE": 1.563, "CDGLOBAL": 0.140, "TOTSCORE": 3.292},
}


class TwinProgression:
    def __init__(self):
        self._loaded = False
        self.models_reg: Dict[str, CatBoostRegressor] = {}
        self.models_conv: Dict[int, CatBoostClassifier] = {}
        self.meta: Dict[str, Any] = {}

    def load(self) -> "TwinProgression":
        meta_path = WEIGHTS_DIR / "progression_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.meta = json.load(f)

        for h in HORIZONS:
            for target in ["MMSE", "CDGLOBAL", "TOTSCORE"]:
                path = WEIGHTS_DIR / f"progression_gbm_{target}_{h}m.cbm"
                if path.exists():
                    reg = CatBoostRegressor()
                    reg.load_model(str(path))
                    self.models_reg[f"{target}_{h}m"] = reg

            conv_path = WEIGHTS_DIR / f"progression_gbm_conv_{h}m.cbm"
            if conv_path.exists():
                clf = CatBoostClassifier()
                clf.load_model(str(conv_path))
                self.models_conv[h] = clf

        self._loaded = True
        return self

    def _prepare_features(self, visit: Dict[str, Any]) -> pd.DataFrame:
        """Map input visit keys to feat_* expected by progression models."""
        t2_cols = [c if c != "MMSCORE" else "MMSE" for c in FEATURE_TIERS["T2"]]
        row = {}
        for col in t2_cols:
            val = visit.get(col, visit.get(col.lower()))
            if val is None:
                # Fallback for MMSE vs MMSCORE
                if col == "MMSE":
                    val = visit.get("MMSCORE", visit.get("mmse"))
            row[f"feat_{col}"] = float(val) if val is not None and not pd.isna(val) else np.nan
        return pd.DataFrame([row])

    def predict_forecast(self, current_visit: Dict[str, Any], current_probs: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        if not self._loaded:
            self.load()

        X = self._prepare_features(current_visit)
        forecast = {}

        curr_mmse = float(current_visit.get("MMSE", current_visit.get("MMSCORE", 27)))
        curr_cdg = float(current_visit.get("CDGLOBAL", 0.5))
        curr_tot = float(current_visit.get("TOTSCORE", 12.0))

        for h in HORIZONS:
            # MMSE
            mmse_model = self.models_reg.get(f"MMSE_{h}m")
            mmse_pred = float(mmse_model.predict(X)[0]) if mmse_model else curr_mmse
            mmse_mae = HORIZON_RESIDUAL_MAE.get(h, {}).get("MMSE", 1.5)
            mmse_lo = max(0.0, round(mmse_pred - 1.64 * mmse_mae, 1))
            mmse_hi = min(30.0, round(mmse_pred + 1.64 * mmse_mae, 1))

            # CDGLOBAL
            cdg_model = self.models_reg.get(f"CDGLOBAL_{h}m")
            cdg_pred = float(cdg_model.predict(X)[0]) if cdg_model else curr_cdg
            cdg_mae = HORIZON_RESIDUAL_MAE.get(h, {}).get("CDGLOBAL", 0.1)
            cdg_lo = max(0.0, round(cdg_pred - 1.64 * cdg_mae, 2))
            cdg_hi = min(3.0, round(cdg_pred + 1.64 * cdg_mae, 2))

            # TOTSCORE
            tot_model = self.models_reg.get(f"TOTSCORE_{h}m")
            tot_pred = float(tot_model.predict(X)[0]) if tot_model else curr_tot
            tot_mae = HORIZON_RESIDUAL_MAE.get(h, {}).get("TOTSCORE", 3.0)
            tot_lo = max(0.0, round(tot_pred - 1.64 * tot_mae, 1))
            tot_hi = min(70.0, round(tot_pred + 1.64 * tot_mae, 1))

            # Conversion risk
            conv_model = self.models_conv.get(h)
            conv_prob = 0.0
            if conv_model:
                try:
                    conv_prob = round(float(conv_model.predict_proba(X)[0, 1]), 4)
                except Exception:
                    conv_prob = 0.0

            # Stage probabilities estimated at future horizon
            if current_probs:
                p_cn = max(0.01, current_probs.get("CN", 0.3) * (1.0 - conv_prob * 0.7))
                p_ad = min(0.95, current_probs.get("AD", 0.1) + conv_prob * 0.5)
                p_mci = max(0.01, 1.0 - p_cn - p_ad)
                tot_p = p_cn + p_mci + p_ad
                stage_probs = {
                    "CN": round(p_cn / tot_p, 4),
                    "MCI": round(p_mci / tot_p, 4),
                    "AD": round(p_ad / tot_p, 4),
                }
            else:
                stage_probs = {"CN": 0.33, "MCI": 0.34, "AD": 0.33}

            forecast[str(h)] = {
                "MMSE_mean": round(mmse_pred, 1),
                "MMSE_lo": mmse_lo,
                "MMSE_hi": mmse_hi,
                "CDGLOBAL_mean": round(cdg_pred, 2),
                "CDGLOBAL_lo": cdg_lo,
                "CDGLOBAL_hi": cdg_hi,
                "TOTSCORE_mean": round(tot_pred, 1),
                "TOTSCORE_lo": tot_lo,
                "TOTSCORE_hi": tot_hi,
                "stage_probs": stage_probs,
                "conversion_prob": conv_prob,
            }

        return forecast


# Singleton
_progression_instance: Optional[TwinProgression] = None

def get_progression() -> TwinProgression:
    global _progression_instance
    if _progression_instance is None:
        _progression_instance = TwinProgression().load()
    return _progression_instance
