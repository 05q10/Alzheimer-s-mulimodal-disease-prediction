"""
server/ml/twin/risk_modifier.py
===============================
Literature-informed lifestyle risk modifier for non-ADNI modifiable risk factors.
Based on published pooled Relative Risks from the Lancet Commission on Dementia
Prevention, Intervention, and Care (Livingston et al., 2020 and 2024 updates).

Transparently modifies:
  (a) predicted rate of cognitive DECLINE in the progression forecast
  (b) conversion hazard probability over time
Crucial clinical guard:
  Never modifies current clinical stage (CN/MCI/AD) or current baseline probabilities.
Hard-capped to [0.85, 1.35].
"""

import json, math
from pathlib import Path
from typing import Any, Dict, List, Optional

CONFIG_PATH = Path(__file__).resolve().parent / "risk_modifier_config.json"

_CONFIG: Optional[Dict[str, Any]] = None

def get_risk_config() -> Dict[str, Any]:
    global _CONFIG
    if _CONFIG is None:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                _CONFIG = json.load(f)
        else:
            _CONFIG = {
                "hard_cap": [0.85, 1.35],
                "shrinkage_power": 0.5,
                "provenance_note": "literature-informed prior, not learned from ADNI",
                "factors": {},
            }
    return _CONFIG


def compute_lifestyle_modifier(lifestyle: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute transparent literature-grounded decline and hazard multiplier from lifestyle inputs.
    Shrunk via square root (power 0.5) and hard-capped to [0.85, 1.35].
    """
    config = get_risk_config()
    cap_lo, cap_hi = config.get("hard_cap", [0.85, 1.35])
    power = config.get("shrinkage_power", 0.5)
    factors_cfg = config.get("factors", {})

    if not lifestyle:
        return {
            "decline_multiplier": 1.0,
            "hazard_multiplier": 1.0,
            "unclamped_decline": 1.0,
            "hard_cap": [cap_lo, cap_hi],
            "contributions": [],
            "factors_applied": {},
            "provenance": config.get("provenance_note", "literature-informed prior, not learned from ADNI"),
        }

    # Normalize input keys (case-insensitive, underscore-insensitive)
    norm_inputs: Dict[str, Any] = {}
    for k, v in lifestyle.items():
        if v is not None:
            clean_k = k.lower().replace("-", "_").replace(" ", "_")
            norm_inputs[clean_k] = v

    log_sum_decline = 0.0
    contributions: List[str] = []
    factors_applied: Dict[str, Any] = {}

    for factor_name, cfg in factors_cfg.items():
        val = None
        # Match potential aliases
        aliases = [factor_name]
        if factor_name == "smoking":
            aliases += ["smoker", "is_smoker", "tobacco"]
        elif factor_name == "physical_activity":
            aliases += ["activity", "exercise", "physicalactivity"]
        elif factor_name == "alcohol":
            aliases += ["alcohol_units", "alcohol_consumption", "alcoholunits", "alcoholconsumption", "drinks_per_week"]
        elif factor_name == "diet_quality":
            aliases += ["diet", "dietquality", "diet_score"]
        elif factor_name == "sleep_quality":
            aliases += ["sleep", "sleepquality", "sleep_score"]
        elif factor_name == "hypertension":
            aliases += ["high_blood_pressure", "htn"]
        elif factor_name == "diabetes":
            aliases += ["dm", "type2_diabetes"]
        elif factor_name == "depression":
            aliases += ["depressive_symptoms"]
        elif factor_name == "head_injury":
            aliases += ["tbi", "concussion", "headinjury", "head_trauma"]
        elif factor_name == "cardiovascular_disease":
            aliases += ["cvd", "heart_disease", "cardiovascular", "cardiovasculardisease"]

        for alias in aliases:
            if alias in norm_inputs:
                val = norm_inputs[alias]
                break

        if val is None:
            continue

        rr = 1.0
        desc = ""

        # Factor type 1: numeric alcohol threshold
        if factor_name == "alcohol" and "thresholds" in cfg:
            try:
                units = float(val)
                for t in cfg["thresholds"]:
                    if units <= t["max_units"]:
                        rr = t["rr"]
                        desc = f"{t['desc']} ({units} units/wk)"
                        break
            except (ValueError, TypeError):
                rr = 1.0

        # Factor type 2: numeric scales for physical_activity, diet_quality, sleep_quality
        elif factor_name in ["physical_activity", "diet_quality", "sleep_quality"]:
            is_num = False
            num_v = 0.0
            try:
                num_v = float(val)
                is_num = True
            except (ValueError, TypeError):
                is_num = False

            if is_num:
                if factor_name == "physical_activity":
                    # Supports 0-10 hrs/wk or 1-5 scale
                    if num_v <= 1.5:
                        rr = 1.39
                    elif num_v <= 2.5:
                        rr = 1.15
                    elif num_v <= 4.0:
                        rr = 1.00
                    elif num_v <= 6.0:
                        rr = 0.92
                    else:
                        rr = 0.85
                elif factor_name == "diet_quality":
                    # Supports 0-10 score or 1-5 scale
                    if num_v <= 2.5:
                        rr = 1.20
                    elif num_v <= 3.5:
                        rr = 1.10
                    elif num_v <= 6.0:
                        rr = 1.00
                    elif num_v <= 7.5:
                        rr = 0.94
                    else:
                        rr = 0.88
                elif factor_name == "sleep_quality":
                    # Supports 0-10 score or 1-5 scale
                    if num_v <= 2.5:
                        rr = 1.25
                    elif num_v <= 4.0:
                        rr = 1.12
                    elif num_v <= 6.0:
                        rr = 1.00
                    elif num_v <= 7.5:
                        rr = 0.98
                    else:
                        rr = 0.95
                desc = f"{cfg.get('label', factor_name)}={num_v}"
            elif "mapping" in cfg:
                str_val = str(val).lower().strip()
                rr = cfg["mapping"].get(str_val, 1.0)
                desc = f"{cfg.get('label', factor_name)}={val}"

        # Factor type 3: mapping table (binary medical conditions, smoking)
        elif "mapping" in cfg:
            str_val = str(val).lower().strip()
            # If boolean converted
            if str_val in ["true", "1", "1.0", "yes"]:
                str_val = "1"
            elif str_val in ["false", "0", "0.0", "no"]:
                str_val = "0"
            rr = cfg["mapping"].get(str_val, 1.0)
            desc = f"{cfg.get('label', factor_name)}={val}"

        if rr != 1.0:
            shrunk_mult = round(math.pow(rr, power), 3)
            log_sum_decline += math.log(shrunk_mult)
            direction_str = "x" if shrunk_mult >= 1.0 else "x"
            contributions.append(f"{desc} {direction_str}{shrunk_mult:.2f}")
            factors_applied[factor_name] = {
                "input_value": val,
                "literature_rr": rr,
                "shrunk_multiplier": shrunk_mult,
                "citation": cfg.get("citation", ""),
            }

    raw_decline_mult = math.exp(log_sum_decline)
    clamped_decline_mult = round(max(cap_lo, min(cap_hi, raw_decline_mult)), 3)
    clamped_hazard_mult = clamped_decline_mult

    return {
        "decline_multiplier": clamped_decline_mult,
        "hazard_multiplier": clamped_hazard_mult,
        "unclamped_decline": round(raw_decline_mult, 3),
        "hard_cap": [cap_lo, cap_hi],
        "contributions": contributions,
        "factors_applied": factors_applied,
        "provenance": config.get("provenance_note", "literature-informed prior, not learned from ADNI"),
    }
