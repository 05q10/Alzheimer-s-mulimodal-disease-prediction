"""
server/ml/twin/progression.py
=============================
Inference module for Model B (progression forecasting).
Loads trained CatBoost regression and conversion models from ml/twin/weights.
Provides:
  - predict_forecast(current_visit: dict) -> dict of horizon -> ForecastPoint
  - simulate_counterfactual(current_visit: dict, deltas: dict) -> dict
"""

import json, math
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
        self.models_delta: Dict[str, CatBoostRegressor] = {}
        self.models_conv: Dict[int, CatBoostClassifier] = {}
        self.envelope: Dict[str, Any] = {}
        self.meta: Dict[str, Any] = {}

    def load(self) -> "TwinProgression":
        meta_path = WEIGHTS_DIR / "progression_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.meta = json.load(f)

        env_path = WEIGHTS_DIR / "plausibility_envelope.json"
        if env_path.exists():
            with open(env_path) as f:
                self.envelope = json.load(f)

        for h in HORIZONS:
            for target in ["MMSE", "CDGLOBAL", "TOTSCORE"]:
                # Level model
                path = WEIGHTS_DIR / f"progression_gbm_{target}_{h}m.cbm"
                if path.exists():
                    reg = CatBoostRegressor()
                    reg.load_model(str(path))
                    self.models_reg[f"{target}_{h}m"] = reg

                # Change / Delta model
                delta_path = WEIGHTS_DIR / f"progression_gbm_delta_{target}_{h}m.cbm"
                if delta_path.exists():
                    d_reg = CatBoostRegressor()
                    d_reg.load_model(str(delta_path))
                    self.models_delta[f"{target}_{h}m"] = d_reg

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
        forecast: Dict[str, Any] = {}

        curr_mmse = float(current_visit.get("MMSE", current_visit.get("MMSCORE", 27)))
        curr_cdg = float(current_visit.get("CDGLOBAL", 0.5))
        curr_tot = float(current_visit.get("TOTSCORE", 12.0))

        # Determine patient stage for envelope matching
        if current_probs:
            anchor_stage = max(current_probs, key=current_probs.get)
        else:
            if curr_mmse >= 28.0 and curr_cdg == 0.0:
                anchor_stage = "CN"
            elif curr_mmse <= 20.0 or curr_cdg >= 1.0:
                anchor_stage = "AD"
            else:
                anchor_stage = "MCI"

        # Check training ranges (5th-95th percentiles of training anchors)
        training_ranges = self.envelope.get("training_input_ranges", {
            "age": [60.7, 84.7],
            "MMSE": [20.0, 30.0],
            "CDGLOBAL": [0.0, 1.0],
            "TOTSCORE": [2.3, 26.3],
            "FAQTOTAL": [0.0, 24.0],
        })

        scope_warnings: List[str] = []
        any_input_out_of_range = False

        # Input range checks
        for key, rng in training_ranges.items():
            lo, hi = rng
            val = None
            if key == "age":
                val = current_visit.get("age", current_visit.get("entry_age"))
            else:
                val = current_visit.get(key, current_visit.get(key.lower()))
                if val is None and key == "MMSE":
                    val = current_visit.get("MMSCORE")
                elif val is None and key == "FAQTOTAL":
                    val = current_visit.get("FAQ")

            if val is not None and not pd.isna(val):
                fval = float(val)
                if fval < lo or fval > hi:
                    any_input_out_of_range = True
                    scope_warnings.append(
                        f"Input {key} ({fval}) is outside the 5th-95th percentile training range [{lo}, {hi}]."
                    )

        # Discordance check between MMSE and CDR
        if curr_mmse >= 28.0 and curr_cdg >= 1.0:
            scope_warnings.append(
                f"Discordant clinical measures: MMSE {curr_mmse} indicates normal cognition while CDR {curr_cdg} indicates dementia."
            )
        elif curr_mmse <= 18.0 and curr_cdg == 0.0:
            scope_warnings.append(
                f"Discordant clinical measures: MMSE {curr_mmse} indicates severe impairment while CDR {curr_cdg} indicates normal cognition."
            )

        any_unreliable = any_input_out_of_range

        for h in HORIZONS:
            env_h = self.envelope.get("horizons", {}).get(str(h), {})

            # ── 1. MMSE ───────────────────────────────────────────────────
            mmse_env = env_h.get("MMSE", {}).get("by_stage", {}).get(anchor_stage, env_h.get("MMSE", {}).get("overall", {
                "p5": -5.0, "p25": -2.0, "p50": 0.0, "p75": 1.0, "p95": 3.0
            }))
            p5_m, p50_m, p95_m = mmse_env["p5"], mmse_env["p50"], mmse_env["p95"]

            d_model = self.models_delta.get(f"MMSE_{h}m")
            if d_model:
                d_raw = float(d_model.predict(X)[0])
            elif f"MMSE_{h}m" in self.models_reg:
                d_raw = float(self.models_reg[f"MMSE_{h}m"].predict(X)[0]) - curr_mmse
            else:
                d_raw = 0.0

            out_m = (d_raw < p5_m or d_raw > p95_m)
            d_clamped = max(p5_m, min(p95_m, d_raw))

            # AD clinical envelope check:
            # In ADNI, 75% of AD patients decline <= 4.0 points/year (p25 = -4.0, p50 = -2.0 at 12m).
            # When CatBoost predicts steep decline steeper than p25, anchor to plausible ADNI rate.
            p25_m = mmse_env.get("p25", p5_m)
            if anchor_stage == "AD" and d_raw < p25_m:
                d_final = 0.5 * max(p5_m, d_raw) + 0.5 * p50_m
                scope_warnings.append(
                    f"AD MMSE forecast decline ({round(d_raw, 2)}) fell outside ADNI interquartile range [{p25_m}, {p95_m}] for {anchor_stage}@{h}m; blended with median ({p50_m})."
                )
            else:
                d_final = d_clamped
                if out_m:
                    scope_warnings.append(
                        f"MMSE forecast change ({round(d_raw, 2)}) fell outside [{p5_m}, {p95_m}] for {anchor_stage}@{h}m; clamped to [{p5_m}, {p95_m}]."
                    )

            if out_m or any_input_out_of_range:
                u_mult_m = 1.5
                any_unreliable = True
            else:
                u_mult_m = 1.0

            # Plausible constraint: impaired patient should not improve significantly
            if (anchor_stage in ["MCI", "AD"] or curr_mmse <= 27.0) and d_final > 1.0:
                scope_warnings.append(
                    f"Implausible MMSE improvement (+{round(d_final, 1)}) for {anchor_stage} stage; bounded to +1.0."
                )
                d_final = 1.0
            if anchor_stage == "AD" and curr_mmse <= 17.0 and d_final > 0.0:
                d_final = 0.0

            mmse_pred = max(0.0, min(30.0, curr_mmse + d_final))
            mmse_mae = HORIZON_RESIDUAL_MAE.get(h, {}).get("MMSE", 1.5) * u_mult_m
            mmse_lo = max(0.0, round(mmse_pred - 1.64 * mmse_mae, 1))
            mmse_hi = min(30.0, round(mmse_pred + 1.64 * mmse_mae, 1))

            # ── 2. CDGLOBAL ───────────────────────────────────────────────
            cdg_env = env_h.get("CDGLOBAL", {}).get("by_stage", {}).get(anchor_stage, env_h.get("CDGLOBAL", {}).get("overall", {
                "p5": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.5
            }))
            p5_c, p50_c, p95_c = cdg_env["p5"], cdg_env["p50"], cdg_env["p95"]

            d_cdg_model = self.models_delta.get(f"CDGLOBAL_{h}m")
            if d_cdg_model:
                d_cdg_raw = float(d_cdg_model.predict(X)[0])
            elif f"CDGLOBAL_{h}m" in self.models_reg:
                d_cdg_raw = float(self.models_reg[f"CDGLOBAL_{h}m"].predict(X)[0]) - curr_cdg
            else:
                d_cdg_raw = 0.0

            out_c = (round(d_cdg_raw, 2) < p5_c or round(d_cdg_raw, 2) > p95_c)
            d_cdg_clamped = max(p5_c, min(p95_c, d_cdg_raw))
            d_cdg_final = d_cdg_clamped
            if out_c or any_input_out_of_range:
                u_mult_c = 1.5
                any_unreliable = True
                if out_c:
                    scope_warnings.append(
                        f"CDGLOBAL forecast change ({round(d_cdg_raw, 2)}) fell outside [{p5_c}, {p95_c}] for {anchor_stage}@{h}m; clamped to [{p5_c}, {p95_c}]."
                    )
            else:
                u_mult_c = 1.0

            cdg_pred = max(0.0, min(3.0, curr_cdg + d_cdg_final))
            cdg_mae = HORIZON_RESIDUAL_MAE.get(h, {}).get("CDGLOBAL", 0.1) * u_mult_c
            cdg_lo = max(0.0, round(cdg_pred - 1.64 * cdg_mae, 2))
            cdg_hi = min(3.0, round(cdg_pred + 1.64 * cdg_mae, 2))

            # ── 3. TOTSCORE ───────────────────────────────────────────────
            tot_env = env_h.get("TOTSCORE", {}).get("by_stage", {}).get(anchor_stage, env_h.get("TOTSCORE", {}).get("overall", {
                "p5": -4.0, "p25": -1.0, "p50": 1.0, "p75": 3.0, "p95": 9.0
            }))
            p5_t, p50_t, p95_t = tot_env["p5"], tot_env["p50"], tot_env["p95"]

            d_tot_model = self.models_delta.get(f"TOTSCORE_{h}m")
            if d_tot_model:
                d_tot_raw = float(d_tot_model.predict(X)[0])
            elif f"TOTSCORE_{h}m" in self.models_reg:
                d_tot_raw = float(self.models_reg[f"TOTSCORE_{h}m"].predict(X)[0]) - curr_tot
            else:
                d_tot_raw = 0.0

            out_t = (round(d_tot_raw, 2) < p5_t or round(d_tot_raw, 2) > p95_t)
            d_tot_clamped = max(p5_t, min(p95_t, d_tot_raw))
            d_tot_final = d_tot_clamped
            if out_t or any_input_out_of_range:
                u_mult_t = 1.5
                any_unreliable = True
                if out_t:
                    scope_warnings.append(
                        f"TOTSCORE forecast change ({round(d_tot_raw, 2)}) fell outside [{p5_t}, {p95_t}] for {anchor_stage}@{h}m; clamped to [{p5_t}, {p95_t}]."
                    )
            else:
                u_mult_t = 1.0

            tot_pred = max(0.0, min(70.0, curr_tot + d_tot_final))
            tot_mae = HORIZON_RESIDUAL_MAE.get(h, {}).get("TOTSCORE", 3.0) * u_mult_t
            tot_lo = max(0.0, round(tot_pred - 1.64 * tot_mae, 1))
            tot_hi = min(70.0, round(tot_pred + 1.64 * tot_mae, 1))

            # ── 4. Conversion risk and progression risk ───────────────────
            conv_model = self.models_conv.get(h)
            conv_prob = 0.0
            if conv_model and anchor_stage != "AD":
                try:
                    conv_prob = round(float(conv_model.predict_proba(X)[0, 1]), 4)
                except Exception:
                    conv_prob = 0.0

            # Progression risk for AD patients: probability of further drop >= 3 MMSE points
            sigma_m = max(1.0, HORIZON_RESIDUAL_MAE.get(h, {}).get("MMSE", 1.5) * 1.25)
            z_m = (-3.0 - d_final) / sigma_m
            prog_risk = round(float(0.5 * (1.0 + math.erf(z_m / math.sqrt(2.0)))), 4)
            prog_risk = max(0.05, min(0.98, prog_risk))

            if anchor_stage == "AD":
                headline_metric = "progression_risk"
                headline_value = prog_risk
                conv_prob = prog_risk
            else:
                headline_metric = "conversion_risk"
                headline_value = conv_prob

            # Stage probabilities estimated at future horizon using Model A (TwinStager)
            from ml.twin.stager import get_stager
            try:
                stager = get_stager()
                fc_inputs = dict(patient_record)
                fc_inputs["MMSE"] = round(mmse_pred, 1)
                fc_inputs["CDGLOBAL"] = round(cdg_pred, 2)
                fc_inputs["TOTSCORE"] = round(tot_pred, 1)
                st_res = stager.predict_one(fc_inputs)
                stage_probs = {
                    "CN": round(float(st_res["probs"]["CN"]), 4),
                    "MCI": round(float(st_res["probs"]["MCI"]), 4),
                    "AD": round(float(st_res["probs"]["AD"]), 4),
                }
            except Exception:
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
                "progression_risk": prog_risk,
                "headline_risk_metric": headline_metric,
                "headline_risk_value": headline_value,
                "p_ad_at_horizon": stage_probs["AD"],
            }

        forecast["forecast_reliability"] = "low" if any_unreliable else "ok"
        forecast["scope_warnings"] = list(dict.fromkeys(scope_warnings))
        return forecast


# Singleton
_progression_instance: Optional[TwinProgression] = None

def get_progression() -> TwinProgression:
    global _progression_instance
    if _progression_instance is None:
        _progression_instance = TwinProgression().load()
    return _progression_instance
