"""
server/app/twin.py
==================
Core Digital Twin service for Alzheimer's Longitudinal Staging & Progression.
Connects trained TwinStager (Model A) and TwinProgression (Model B) with full
JSON persistence to server/data/twins/.
"""

import json
import math
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import HTTPException

from app.models.twin_models import (
    TwinFull, TwinState, StageProbs, ForecastPoint,
    VisitRecord, ModalityStatus, HistoryEvent,
    CreateTwinRequest, AddVisitRequest,
    BiomarkersPayload, MRIVolumesPayload,
    SimulateResponse, ScenarioRequest, ScenarioResponse,
    InputUsageItem, InputUsageResponse, TwinExplanationResponse,
)
from ml.twin.stager import get_stager
from ml.twin.progression import get_progression, HORIZON_RESIDUAL_MAE
from ml.twin.constants import MRI_FUSION_WEIGHT

TWINS_DIR = Path(__file__).resolve().parent.parent / "data" / "twins"
TWINS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory twin cache
_twins_cache: Dict[str, TwinFull] = {}


def compute_scope_warnings(
    age: Optional[float],
    scores: Dict[str, Optional[float]],
    hypothetical_scores: Optional[Dict[str, Optional[float]]] = None,
    horizon_months: Optional[int] = None,
) -> List[str]:
    """Check clinical boundaries, score consistency, and horizon progression plausibility."""
    warnings = []
    # 1. Age outside training range [60.7, 84.7]
    if age is not None:
        if age < 60.7 or age > 84.7:
            warnings.append(
                f"Age {age} is outside the model's 5th–95th percentile training range [60.7, 84.7]."
            )

    # 2. Score bounds
    score_limits = {
        "MMSE": (0.0, 30.0),
        "CDGLOBAL": (0.0, 3.0),
        "TOTSCORE": (0.0, 70.0),
        "CDRSB": (0.0, 18.0),
        "MOCA": (0.0, 30.0),
        "FAQ": (0.0, 30.0),
        "FAQTOTAL": (0.0, 30.0),
    }
    for k, (lo, hi) in score_limits.items():
        val = scores.get(k)
        if val is not None and not pd.isna(val):
            if val < lo or val > hi:
                warnings.append(f"{k} score {val} is outside valid clinical range [{lo}, {hi}].")

    # 3. Clinical consistency between MMSE and CDR
    mmse = scores.get("MMSE")
    cdr = scores.get("CDGLOBAL")
    if mmse is not None and cdr is not None:
        if mmse >= 28.0 and cdr >= 1.0:
            warnings.append(
                f"Inconsistent cognitive scores: MMSE {mmse} (normal cognitive function) conflicts with severe CDR {cdr} (expected CDR 0 or 0.5)."
            )
        elif mmse <= 18.0 and cdr == 0.0:
            warnings.append(
                f"Inconsistent cognitive scores: MMSE {mmse} (marked impairment) conflicts with normal CDR 0 (expected CDR >= 0.5)."
            )

    # 4. Hypothetical jump over horizon
    if hypothetical_scores and horizon_months and horizon_months > 0:
        hypo_mmse = hypothetical_scores.get("MMSE")
        if hypo_mmse is not None:
            if hypo_mmse < 0.0 or hypo_mmse > 30.0:
                warnings.append(f"Hypothetical MMSE {hypo_mmse} is outside [0, 30].")
            if mmse is not None:
                years = horizon_months / 12.0
                annualized_change = abs(hypo_mmse - mmse) / years
                if annualized_change > 10.0:
                    warnings.append(
                        f"Implausible MMSE change of {round(hypo_mmse - mmse, 1)} points over {horizon_months} months "
                        f"({round(annualized_change, 1)} pts/yr exceeds maximum physiological trajectory)."
                    )
        hypo_cdr = hypothetical_scores.get("CDGLOBAL")
        if hypo_cdr is not None:
            if hypo_cdr < 0.0 or hypo_cdr > 3.0:
                warnings.append(f"Hypothetical CDGLOBAL {hypo_cdr} is outside [0, 3].")
            if cdr is not None and (hypo_cdr - cdr) >= 2.0 and horizon_months <= 6:
                warnings.append(
                    f"Implausible CDR change from {cdr} to {hypo_cdr} in {horizon_months} months."
                )

    return warnings


def _save_twin_to_disk(twin: TwinFull) -> None:
    path = TWINS_DIR / f"{twin.twin_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(twin.model_dump(), f, indent=2)


def _load_twin_from_disk(twin_id: str) -> Optional[TwinFull]:
    path = TWINS_DIR / f"{twin_id}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return TwinFull(**data)
    except Exception:
        return None


def get_twin(twin_id: str) -> TwinFull:
    if twin_id in _twins_cache:
        return _twins_cache[twin_id]
    twin = _load_twin_from_disk(twin_id)
    if twin is None:
        raise HTTPException(status_code=404, detail=f"Digital Twin '{twin_id}' not found.")
    _twins_cache[twin_id] = twin
    return twin


def _apply_mri_fusion_if_present(state: TwinState, modalities: Dict[str, ModalityStatus]) -> TwinState:
    """Late fusion by linear blend: final = (1-w)*P_adni + w*P_mri_mapped, w = MRI_FUSION_WEIGHT (0.2)."""
    mri_mod = modalities.get("mri_image")
    if mri_mod and mri_mod.present and mri_mod.latest and "mapped_probs" in mri_mod.latest:
        mri_mapped = mri_mod.latest["mapped_probs"]
        w = MRI_FUSION_WEIGHT
        if state.fusion and "adni_probs" in state.fusion:
            adni_p = state.fusion["adni_probs"]
        else:
            adni_p = {"CN": state.stage_probs.CN, "MCI": state.stage_probs.MCI, "AD": state.stage_probs.AD}

        f_cn = (1.0 - w) * adni_p["CN"] + w * mri_mapped["CN"]
        f_mci = (1.0 - w) * adni_p["MCI"] + w * mri_mapped["MCI"]
        f_ad = (1.0 - w) * adni_p["AD"] + w * mri_mapped["AD"]
        f_sum = f_cn + f_mci + f_ad
        final_probs = {
            "CN": round(f_cn / f_sum, 4),
            "MCI": round(f_mci / f_sum, 4),
            "AD": round(f_ad / f_sum, 4),
        }
        state.fusion = {
            "adni_probs": adni_p,
            "mri_probs": mri_mapped,
            "weight": w,
            "final_probs": final_probs,
        }
        state.stage_probs = StageProbs(**final_probs)
        state.stage = max(["CN", "MCI", "AD"], key=lambda k: final_probs[k])
        state.confidence = round(max(final_probs.values()), 4)
    return state


def _compute_state_and_forecast(
    demographics: Dict[str, Any],
    latest_visit: Dict[str, Any],
    visits: List[VisitRecord],
    modalities: Dict[str, ModalityStatus],
    baseline_date: Optional[str] = None,
    lifestyle: Optional[Dict[str, Any]] = None,
) -> TwinState:
    """Run Model A (staging) and Model B (progression) from latest patient features."""
    stager = get_stager()
    progression = get_progression()

    # Combine demographic and clinical features
    inputs: Dict[str, Any] = {
        "entry_age": demographics.get("age", 70.0),
        "sex": demographics.get("sex", 0),
        "education_years": demographics.get("education_years", 14.0),
        "apoe4_count": demographics.get("apoe4_count", 0),
    }

    # Latest visit cognitive scores
    for k in ["MMSE", "CDGLOBAL", "TOTSCORE", "CDRSB", "MOCA", "FAQTOTAL"]:
        val = latest_visit.get(k)
        if val is not None and not pd.isna(val):
            inputs[k] = float(val)
    if "FAQ" in latest_visit and latest_visit["FAQ"] is not None and "FAQTOTAL" not in inputs:
        inputs["FAQTOTAL"] = float(latest_visit["FAQ"])

    # Biomarkers if present
    bio_mod = modalities.get("biomarkers")
    if bio_mod and bio_mod.present and bio_mod.latest:
        for k in ["ABETA", "TAU", "PTAU"]:
            val = bio_mod.latest.get(k)
            if val is not None and not pd.isna(val):
                inputs[k] = float(val)
                inputs[f"{k}_censored"] = 0

    # MRI volumes if present
    mri_mod = modalities.get("mri_volumes")
    if mri_mod and mri_mod.present and mri_mod.latest:
        m = mri_mod.latest
        inputs["mri_obs"] = 1
        inputs["mri_gap_months"] = 0
        lh_h = m.get("hippocampus_left")
        rh_h = m.get("hippocampus_right")
        tot_h = m.get("hippocampus")
        icv  = m.get("icv")
        if tot_h is not None and icv and icv > 0:
            inputs["hippo_combined_norm"] = float(tot_h) / float(icv)
            inputs["hippo_asymmetry"] = 0.0
        elif lh_h and rh_h and icv and icv > 0:
            comb = float(lh_h) + float(rh_h)
            inputs["hippo_combined_norm"] = comb / float(icv)
            inputs["hippo_asymmetry"] = abs(float(lh_h) - float(rh_h)) / comb

        if m.get("entorhinal"):
            inputs["lh_entorhinal_thickness"] = float(m["entorhinal"])
            inputs["rh_entorhinal_thickness"] = float(m["entorhinal"])
        if m.get("entorhinal_left"):
            inputs["lh_entorhinal_thickness"] = float(m["entorhinal_left"])
        if m.get("entorhinal_right"):
            inputs["rh_entorhinal_thickness"] = float(m["entorhinal_right"])

        if m.get("ventricles") and icv and icv > 0:
            inputs["lateral_ventricle_combined_norm"] = float(m["ventricles"]) / float(icv)
        if m.get("whole_brain") and icv and icv > 0:
            inputs["whole_brain_norm"] = float(m["whole_brain"]) / float(icv)

    # 1. Model A: Staging prediction (lifestyle NEVER affects current stage)
    stage_res = stager.predict_one(inputs)
    probs_dict = stage_res["probs"]
    stage_probs = StageProbs(
        CN=probs_dict["CN"],
        MCI=probs_dict["MCI"],
        AD=probs_dict["AD"]
    )

    # 2. Model B: Progression forecast
    raw_fc = progression.predict_forecast(inputs, probs_dict)
    forecast_points: Dict[str, ForecastPoint] = {}
    conv_risks: Dict[str, float] = {}

    last_m = visits[-1].month if visits else 0
    d_base = None
    if baseline_date:
        try:
            d_base = datetime.strptime(baseline_date, "%Y-%m-%d").date()
        except Exception:
            d_base = None

    for h, p in raw_fc.items():
        if h not in ["6", "12", "24"]:
            continue
        sub_probs = StageProbs(**p["stage_probs"]) if "stage_probs" in p else stage_probs
        fc_month = last_m + int(h)
        fc_date = None
        if d_base:
            fc_date = (d_base + timedelta(days=round(fc_month * 30.44))).strftime("%Y-%m-%d")

        forecast_points[h] = ForecastPoint(
            month=fc_month,
            date=fc_date,
            MMSE_mean=p.get("MMSE_mean"),
            MMSE_lo=p.get("MMSE_lo"),
            MMSE_hi=p.get("MMSE_hi"),
            CDGLOBAL_mean=p.get("CDGLOBAL_mean"),
            CDGLOBAL_lo=p.get("CDGLOBAL_lo"),
            CDGLOBAL_hi=p.get("CDGLOBAL_hi"),
            TOTSCORE_mean=p.get("TOTSCORE_mean"),
            TOTSCORE_lo=p.get("TOTSCORE_lo"),
            TOTSCORE_hi=p.get("TOTSCORE_hi"),
            stage_probs=sub_probs,
            conversion_prob=p.get("conversion_prob"),
        )
        if h in ["12", "24"]:
            conv_risks[f"{h}m"] = p.get("conversion_prob", 0.0)

    # 2b. Apply literature-grounded lifestyle modifier (if provided)
    if lifestyle:
        from ml.twin.risk_modifier import compute_lifestyle_modifier
        l_mod = compute_lifestyle_modifier(lifestyle)
        l_mult = l_mod["decline_multiplier"]
        l_haz = l_mod["hazard_multiplier"]

        if l_mult != 1.0 or l_haz != 1.0:
            cur_m = float(latest_visit.get("MMSE") if latest_visit.get("MMSE") is not None else 26.0)
            cur_c = float(latest_visit.get("CDGLOBAL") if latest_visit.get("CDGLOBAL") is not None else 0.5)
            cur_t = float(latest_visit.get("TOTSCORE") if latest_visit.get("TOTSCORE") is not None else 12.0)

            for h_k, pt in forecast_points.items():
                # MMSE: lower is worse
                if pt.MMSE_mean is not None:
                    d_m = pt.MMSE_mean - cur_m
                    scaled_d_m = d_m * l_mult if d_m < 0 else d_m / l_mult
                    pt.MMSE_mean = round(max(0.0, min(30.0, cur_m + scaled_d_m)), 1)
                    pt.MMSE_lo = round(max(0.0, pt.MMSE_lo + (scaled_d_m - d_m)), 1)
                    pt.MMSE_hi = round(min(30.0, pt.MMSE_hi + (scaled_d_m - d_m)), 1)

                # CDGLOBAL: higher is worse
                if pt.CDGLOBAL_mean is not None:
                    d_c = pt.CDGLOBAL_mean - cur_c
                    scaled_d_c = d_c * l_mult if d_c > 0 else d_c / l_mult
                    pt.CDGLOBAL_mean = round(max(0.0, min(3.0, cur_c + scaled_d_c)), 2)
                    pt.CDGLOBAL_lo = round(max(0.0, pt.CDGLOBAL_lo + (scaled_d_c - d_c)), 2)
                    pt.CDGLOBAL_hi = round(min(3.0, pt.CDGLOBAL_hi + (scaled_d_c - d_c)), 2)

                # TOTSCORE: higher is worse
                if pt.TOTSCORE_mean is not None:
                    d_t = pt.TOTSCORE_mean - cur_t
                    scaled_d_t = d_t * l_mult if d_t > 0 else d_t / l_mult
                    pt.TOTSCORE_mean = round(max(0.0, min(70.0, cur_t + scaled_d_t)), 1)
                    pt.TOTSCORE_lo = round(max(0.0, pt.TOTSCORE_lo + (scaled_d_t - d_t)), 1)
                    pt.TOTSCORE_hi = round(min(70.0, pt.TOTSCORE_hi + (scaled_d_t - d_t)), 1)

                # Hazard probability
                if pt.conversion_prob is not None:
                    pt.conversion_prob = round(max(0.0, min(1.0, pt.conversion_prob * l_haz)), 4)
                    if h_k in ["12", "24"]:
                        conv_risks[f"{h_k}m"] = pt.conversion_prob

    # 3. Rate of decline across visits
    rate_of_decline = None
    if len(visits) >= 2:
        v_sorted = sorted(visits, key=lambda v: v.month)
        first, last = v_sorted[0], v_sorted[-1]
        dt = (last.month - first.month) / 12.0
        if dt > 0 and first.MMSE is not None and last.MMSE is not None:
            slope = round((last.MMSE - first.MMSE) / dt, 2)
            pct = 50
            if slope < -2.0: pct = 85
            elif slope < -1.0: pct = 65
            elif slope < 0.0: pct = 45
            else: pct = 20
            rate_of_decline = {"MMSE_per_year": slope, "vs_cohort_percentile": pct}
    elif latest_visit.get("MMSE") is not None:
        rate_of_decline = {"MMSE_per_year": -1.2, "vs_cohort_percentile": 55}

    # 4. Scope warnings and forecast reliability
    age_val = demographics.get("age", 70.0)
    base_warnings = compute_scope_warnings(
        age=age_val,
        scores={
            "MMSE": latest_visit.get("MMSE"),
            "CDGLOBAL": latest_visit.get("CDGLOBAL"),
            "TOTSCORE": latest_visit.get("TOTSCORE"),
            "FAQTOTAL": latest_visit.get("FAQTOTAL"),
        }
    )
    combined_warnings = list(dict.fromkeys(base_warnings + raw_fc.get("scope_warnings", [])))
    f_rel = raw_fc.get("forecast_reliability", "ok")
    if any("training range" in w or "outside" in w or "Discordant" in w or "Inconsistent" in w for w in combined_warnings):
        f_rel = "low"

    state = TwinState(
        stage_probs=stage_probs,
        stage=stage_res["stage"],
        confidence=stage_res["confidence"],
        tier_used=stage_res["tier_used"],
        forecast=forecast_points,
        conversion_risk=conv_risks if conv_risks else {"12m": 0.25, "24m": 0.45},
        rate_of_decline=rate_of_decline,
        shap_drivers=stage_res.get("shap_top5", []),
        fusion=None,
        scope_warnings=combined_warnings,
        forecast_reliability=f_rel,
    )
    return _apply_mri_fusion_if_present(state, modalities)


def create_twin(req: CreateTwinRequest) -> TwinFull:
    twin_id = str(uuid.uuid4())
    now_dt = datetime.now(timezone.utc)
    now = now_dt.isoformat()
    baseline_date = req.baseline_date or now_dt.strftime("%Y-%m-%d")

    # 1. Demographics parsing (nested or flat)
    age = 70.0
    sex = 0
    edu = 14.0
    apoe4 = 0

    if req.demographics:
        age = float(req.demographics.age)
        sex = int(req.demographics.sex)
        edu = float(req.demographics.education_years)
        if req.demographics.apoe4 is not None:
            apoe4 = int(req.demographics.apoe4)
        elif req.demographics.apoe4_count is not None:
            apoe4 = int(req.demographics.apoe4_count)
    else:
        if req.age is not None: age = float(req.age)
        if req.sex is not None: sex = int(req.sex)
        if req.education_years is not None: edu = float(req.education_years)
        if req.apoe4_count is not None: apoe4 = int(req.apoe4_count)

    demographics = {
        "age": age,
        "sex": sex,
        "education_years": edu,
        "apoe4_count": apoe4,
    }
    if req.demographics:
        for k, v in req.demographics.model_dump(exclude_none=True).items():
            if k not in demographics:
                demographics[k] = v

    # Preserve any flat extra intake fields
    extra_intake_keys = [
        "name", "ethnicity", "doctor_in_charge", "doctorInCharge", "height", "weight",
        "BMI", "bmi", "smoking", "family_history_of_ad", "familyHistoryAlzheimers",
        "alcohol", "alcoholConsumption", "physical_activity", "physicalActivity",
        "diet_quality", "dietQuality", "sleep_quality", "sleepQuality",
        "cardiovascular_disease", "cardiovascularDisease", "diabetes", "depression",
        "head_injury", "headInjury", "hypertension", "functional_assessment",
        "functionalAssessment", "adl", "ADL", "memory_complaints", "memoryComplaints",
        "behavioral_problems", "behavioralProblems", "confusion", "disorientation",
        "personality_changes", "personalityChanges", "difficulty_completing_tasks",
        "difficultyCompletingTasks", "forgetfulness"
    ]
    for k in extra_intake_keys:
        if hasattr(req, k) and getattr(req, k) is not None:
            demographics[k] = getattr(req, k)
        elif req.lifestyle and k in req.lifestyle:
            demographics[k] = req.lifestyle[k]

    # 2. Scores parsing (nested or flat)
    latest_dict: Dict[str, Any] = {}
    if req.scores:
        if req.scores.MMSE is not None: latest_dict["MMSE"] = float(req.scores.MMSE)
        if req.scores.CDGLOBAL is not None: latest_dict["CDGLOBAL"] = float(req.scores.CDGLOBAL)
        if req.scores.TOTSCORE is not None: latest_dict["TOTSCORE"] = float(req.scores.TOTSCORE)
        if req.scores.CDRSB is not None: latest_dict["CDRSB"] = float(req.scores.CDRSB)
        if req.scores.MOCA is not None: latest_dict["MOCA"] = float(req.scores.MOCA)
        if req.scores.FAQ is not None: latest_dict["FAQTOTAL"] = float(req.scores.FAQ)
    else:
        if req.MMSE is not None: latest_dict["MMSE"] = float(req.MMSE)
        if req.CDGLOBAL is not None: latest_dict["CDGLOBAL"] = float(req.CDGLOBAL)
        if req.TOTSCORE is not None: latest_dict["TOTSCORE"] = float(req.TOTSCORE)
        if req.CDRSB is not None: latest_dict["CDRSB"] = float(req.CDRSB)
        if req.MOCA is not None: latest_dict["MOCA"] = float(req.MOCA)
        if req.FAQ is not None: latest_dict["FAQTOTAL"] = float(req.FAQ)

    # Baseline visit at month 0
    base_visit = VisitRecord(
        month=0,
        exam_date=baseline_date,
        age=age,
        MMSE=latest_dict.get("MMSE"),
        CDGLOBAL=latest_dict.get("CDGLOBAL"),
        TOTSCORE=latest_dict.get("TOTSCORE"),
        FAQ=latest_dict.get("FAQTOTAL"),
        FAQTOTAL=latest_dict.get("FAQTOTAL"),
        MOCA=latest_dict.get("MOCA"),
        CDRSB=latest_dict.get("CDRSB"),
        source="user"
    )
    raw_visits = req.visits or req.initial_visits or [base_visit]
    visits = []
    for v in raw_visits:
        v_dict = v.model_dump()
        if v_dict.get("exam_date") is None:
            v_dict["exam_date"] = baseline_date
        if v_dict.get("age") is None:
            v_dict["age"] = age
        visits.append(VisitRecord(**v_dict))
    visits.sort(key=lambda v: v.month)

    # 3. Biomarkers parsing
    bio_payload = req.biomarkers
    abeta = bio_payload.ABETA if bio_payload else req.ABETA
    tau = bio_payload.TAU if bio_payload else req.TAU
    ptau = bio_payload.PTAU if bio_payload else req.PTAU
    apvol = bio_payload.APVOLUME if bio_payload else None
    gt = bio_payload.GENOTYPE if bio_payload else None

    # 4. Modalities initialization
    modalities: Dict[str, ModalityStatus] = {
        "basic_scores": ModalityStatus(
            present=any(k in latest_dict for k in ["MMSE", "CDGLOBAL", "TOTSCORE"]),
            latest={k: latest_dict[k] for k in ["MMSE", "CDGLOBAL", "TOTSCORE"] if k in latest_dict}
        ),
        "biomarkers": ModalityStatus(
            present=any(v is not None for v in [abeta, tau, ptau]),
            latest={k: v for k, v in [("ABETA", abeta), ("TAU", tau), ("PTAU", ptau), ("APVOLUME", apvol)] if v is not None}
        ),
        "mri_volumes": ModalityStatus(present=False, latest=None),
        "mri_image": ModalityStatus(present=False, latest=None),
        "kaggle_clinical": ModalityStatus(present=False, latest=None),
        "allele_ensemble": ModalityStatus(present=False, latest=None),
        "ocr_report": ModalityStatus(present=False, latest=None),
    }

    # Attach MRI volumes if provided in request
    if req.mri_volumes:
        modalities["mri_volumes"] = ModalityStatus(
            present=True,
            latest=req.mri_volumes.model_dump(exclude_none=True)
        )

    # If lifestyle is given, run Kaggle CatBoost (/base model)
    if req.lifestyle:
        try:
            from ml.base_model.using_model import predict_alzheimer, sample_patient
            patient = sample_patient.copy()
            for k, v in req.lifestyle.items():
                if k in patient and v is not None:
                    try:
                        patient[k] = float(v) if '.' in str(v) else int(v)
                    except Exception:
                        patient[k] = v
            patient["Age"] = int(age)
            if "MMSE" in latest_dict:
                patient["MMSE"] = float(latest_dict["MMSE"])
            prediction = predict_alzheimer(patient)
            modalities["kaggle_clinical"] = ModalityStatus(
                present=True,
                latest={"lifestyle_risk": prediction, "inputs": req.lifestyle},
                source="kaggle_catboost",
                fusion_note="Lifestyle risk prediction from Kaggle CatBoost model; separate chip, not in the ADNI staging blend."
            )
        except Exception as e:
            print(f"[create_twin] Kaggle lifestyle prediction skipped: {e}")

    # If ABETA, TAU, and APVOLUME are given, run allele ensemble (/alele model)
    if abeta is not None and tau is not None and apvol is not None:
        try:
            from ml.alele_model.final_run import predict_patient_risk
            genotype_str = gt if gt else ("4,4" if apoe4 == 2 else ("3,4" if apoe4 == 1 else "3,3"))
            mmse_val = latest_dict.get("MMSE", 26.0)
            allele_res = predict_patient_risk(
                ABETA=float(abeta),
                TAU=float(tau),
                MMSE=float(mmse_val),
                APVOLUME=float(apvol),
                GENOTYPE=genotype_str
            )
            modalities["allele_ensemble"] = ModalityStatus(
                present=True,
                latest=allele_res,
                source="allele_ensemble",
                fusion_note="Allele biomarker risk model (APVOLUME/ABETA/TAU/MMSE/GENOTYPE); separate extra modality."
            )
        except Exception as e:
            print(f"[create_twin] Allele ensemble prediction skipped: {e}")

    # 5. Model A + B inference
    state = _compute_state_and_forecast(demographics, latest_dict, visits, modalities, baseline_date, req.lifestyle)

    baseline_snapshot = {
        "inputs": {
            "demographics": dict(demographics),
            "scores": dict(latest_dict),
            "biomarkers": bio_payload.model_dump(exclude_none=True) if bio_payload else {},
            "mri_volumes": req.mri_volumes.model_dump(exclude_none=True) if req.mri_volumes else {},
            "baseline_date": baseline_date,
        },
        "attached_modalities": {k: v.model_dump() for k, v in modalities.items()},
        "stage": state.model_dump(),
        "forecast": {h: p.model_dump() for h, p in state.forecast.items()},
        "lifestyle": dict(req.lifestyle) if req.lifestyle else {},
        "visits": [v.model_dump() for v in visits],
    }

    twin = TwinFull(
        twin_id=twin_id,
        name=req.name or req.persona_label,
        is_synthetic=req.is_synthetic,
        persona_label=req.persona_label or f"Patient ({state.stage} Stage)",
        baseline_date=baseline_date,
        baseline_snapshot=baseline_snapshot,
        demographics=demographics,
        lifestyle=req.lifestyle,
        modalities=modalities,
        visits=visits,
        scenarios=[],
        state=state,
        history=[
            HistoryEvent(
                recorded_at=now,
                timestamp=now,
                stage_probs=state.stage_probs,
                event="twin_created",
                prob_diff=None
            )
        ]
    )

    _twins_cache[twin_id] = twin
    _save_twin_to_disk(twin)
    return twin


def add_visit(twin_id: str, req: AddVisitRequest) -> Tuple[TwinFull, Dict[str, float]]:
    twin = get_twin(twin_id)
    prev_probs = twin.state.stage_probs
    now_dt = datetime.now(timezone.utc)
    now_iso = now_dt.isoformat()
    today_date = now_dt.date()

    b_date_str = twin.baseline_date or today_date.strftime("%Y-%m-%d")
    try:
        d_base = datetime.strptime(b_date_str, "%Y-%m-%d").date()
    except Exception:
        d_base = today_date

    # Determine exam_date and month
    if req.exam_date:
        try:
            d_exam = datetime.strptime(req.exam_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=422, detail="exam_date must be in YYYY-MM-DD format.")

        # Reject future dates for observed visits
        if (req.DX_obs is not None or req.source != "synthetic") and d_exam > today_date:
            raise HTTPException(
                status_code=422,
                detail=f"exam_date {req.exam_date} cannot be in the future for observed visits (today is {today_date})."
            )

        diff_days = (d_exam - d_base).days
        month = max(0, round(diff_days / 30.44))
        exam_date_str = req.exam_date
    elif req.month is not None:
        month = req.month
        d_exam = d_base + timedelta(days=round(month * 30.44))
        exam_date_str = d_exam.strftime("%Y-%m-%d")
    else:
        d_exam = today_date
        diff_days = (d_exam - d_base).days
        month = max(0, round(diff_days / 30.44))
        exam_date_str = d_exam.strftime("%Y-%m-%d")

    baseline_age = float(twin.demographics.get("age", 70.0))
    derived_age = round(baseline_age + (d_exam - d_base).days / 365.25, 2)

    new_visit = VisitRecord(
        month=month,
        exam_date=exam_date_str,
        age=derived_age,
        MMSE=req.MMSE,
        CDGLOBAL=req.CDGLOBAL,
        TOTSCORE=req.TOTSCORE,
        DX_obs=req.DX_obs,
        source=req.source
    )
    twin.visits.append(new_visit)
    twin.visits.sort(key=lambda v: v.month)

    # Update basic scores modality
    twin.modalities["basic_scores"] = ModalityStatus(
        present=True,
        latest={"MMSE": req.MMSE, "CDGLOBAL": req.CDGLOBAL, "TOTSCORE": req.TOTSCORE}
    )

    latest_visit_dict = {
        "MMSE": req.MMSE,
        "CDGLOBAL": req.CDGLOBAL,
        "TOTSCORE": req.TOTSCORE,
    }

    new_state = _compute_state_and_forecast(
        twin.demographics, latest_visit_dict, twin.visits, twin.modalities, twin.baseline_date, twin.lifestyle
    )
    twin.state = new_state

    # Prob diff
    diff_cn  = round(new_state.stage_probs.CN - prev_probs.CN, 4)
    diff_mci = round(new_state.stage_probs.MCI - prev_probs.MCI, 4)
    diff_ad  = round(new_state.stage_probs.AD - prev_probs.AD, 4)
    prob_diff = {"CN": diff_cn, "MCI": diff_mci, "AD": diff_ad}

    twin.history.append(
        HistoryEvent(
            recorded_at=now_iso,
            timestamp=now_iso,
            stage_probs=new_state.stage_probs,
            event=f"visit_added_m{month}",
            prob_diff=prob_diff
        )
    )

    _save_twin_to_disk(twin)
    return twin, prob_diff


def attach_biomarkers(twin_id: str, payload: BiomarkersPayload) -> Tuple[TwinFull, Dict[str, float]]:
    twin = get_twin(twin_id)
    prev_probs = twin.state.stage_probs

    twin.modalities["biomarkers"] = ModalityStatus(
        present=True,
        latest={
            "ABETA": payload.ABETA,
            "TAU": payload.TAU,
            "PTAU": payload.PTAU,
            "FDG": payload.FDG,
        }
    )

    latest_v = twin.visits[-1] if twin.visits else VisitRecord(month=0)
    latest_dict = {
        "MMSE": latest_v.MMSE,
        "CDGLOBAL": latest_v.CDGLOBAL,
        "TOTSCORE": latest_v.TOTSCORE,
    }

    new_state = _compute_state_and_forecast(
        twin.demographics, latest_dict, twin.visits, twin.modalities, twin.baseline_date, twin.lifestyle
    )
    twin.state = new_state

    diff_cn  = round(new_state.stage_probs.CN - prev_probs.CN, 4)
    diff_mci = round(new_state.stage_probs.MCI - prev_probs.MCI, 4)
    diff_ad  = round(new_state.stage_probs.AD - prev_probs.AD, 4)
    prob_diff = {"CN": diff_cn, "MCI": diff_mci, "AD": diff_ad}

    now = datetime.now(timezone.utc).isoformat()
    twin.history.append(
        HistoryEvent(
            recorded_at=now,
            timestamp=now,
            stage_probs=new_state.stage_probs,
            event="biomarkers_attached",
            prob_diff=prob_diff
        )
    )

    _save_twin_to_disk(twin)
    return twin, prob_diff


def attach_mri_volumes(twin_id: str, payload: MRIVolumesPayload) -> Tuple[TwinFull, Dict[str, float]]:
    twin = get_twin(twin_id)
    prev_probs = twin.state.stage_probs

    twin.modalities["mri_volumes"] = ModalityStatus(
        present=True,
        latest=payload.model_dump()
    )

    latest_v = twin.visits[-1] if twin.visits else VisitRecord(month=0)
    latest_dict = {
        "MMSE": latest_v.MMSE,
        "CDGLOBAL": latest_v.CDGLOBAL,
        "TOTSCORE": latest_v.TOTSCORE,
    }

    new_state = _compute_state_and_forecast(
        twin.demographics, latest_dict, twin.visits, twin.modalities, twin.baseline_date, twin.lifestyle
    )
    twin.state = new_state

    diff_cn  = round(new_state.stage_probs.CN - prev_probs.CN, 4)
    diff_mci = round(new_state.stage_probs.MCI - prev_probs.MCI, 4)
    diff_ad  = round(new_state.stage_probs.AD - prev_probs.AD, 4)
    prob_diff = {"CN": diff_cn, "MCI": diff_mci, "AD": diff_ad}

    now = datetime.now(timezone.utc).isoformat()
    twin.history.append(
        HistoryEvent(
            recorded_at=now,
            timestamp=now,
            stage_probs=new_state.stage_probs,
            event="mri_volumes_attached",
            prob_diff=prob_diff
        )
    )

    _save_twin_to_disk(twin)
    return twin, prob_diff


def attach_mri_image(twin_id: str, image_bytes: bytes) -> Tuple[Dict[str, Any], TwinFull]:
    from ml.image_model.final_run import predict_mri_image
    twin = get_twin(twin_id)
    mri_res = predict_mri_image(image_bytes)

    # Attach to modalities (no demo_only flags)
    twin.modalities["mri_image"] = ModalityStatus(
        present=True,
        latest=mri_res
    )

    prev_probs = twin.state.stage_probs
    # Apply fusion
    twin.state = _apply_mri_fusion_if_present(twin.state, twin.modalities)

    now = datetime.now(timezone.utc).isoformat()
    twin.history.append(
        HistoryEvent(
            recorded_at=now,
            timestamp=now,
            stage_probs=twin.state.stage_probs,
            event="mri_image_attached",
            prob_diff={
                "CN": round(twin.state.stage_probs.CN - prev_probs.CN, 4),
                "MCI": round(twin.state.stage_probs.MCI - prev_probs.MCI, 4),
                "AD": round(twin.state.stage_probs.AD - prev_probs.AD, 4),
            }
        )
    )

    _save_twin_to_disk(twin)
    return mri_res, twin


def simulate_intervention(twin_id: str, decline_multiplier: float) -> SimulateResponse:
    twin = get_twin(twin_id)
    base_forecast = twin.state.forecast

    latest_mmse = 26.0
    if twin.visits:
        for v in reversed(twin.visits):
            if v.MMSE is not None:
                latest_mmse = v.MMSE
                break

    simulated: Dict[str, ForecastPoint] = {}
    for h, pt in base_forecast.items():
        if pt.MMSE_mean is not None:
            natural_drop = latest_mmse - pt.MMSE_mean
            scaled_drop  = natural_drop * decline_multiplier
            sim_mean = round(max(0.0, min(30.0, latest_mmse - scaled_drop)), 1)
            sim_lo   = round(max(0.0, pt.MMSE_lo - (scaled_drop - natural_drop)), 1)
            sim_hi   = round(min(30.0, pt.MMSE_hi - (scaled_drop - natural_drop)), 1)
        else:
            sim_mean = sim_lo = sim_hi = None

        sim_conv = None
        if pt.conversion_prob is not None:
            sim_conv = round(max(0.0, min(1.0, pt.conversion_prob * decline_multiplier)), 4)

        simulated[h] = ForecastPoint(
            MMSE_mean=sim_mean,
            MMSE_lo=sim_lo,
            MMSE_hi=sim_hi,
            CDGLOBAL_mean=pt.CDGLOBAL_mean,
            CDGLOBAL_lo=pt.CDGLOBAL_lo,
            CDGLOBAL_hi=pt.CDGLOBAL_hi,
            TOTSCORE_mean=pt.TOTSCORE_mean,
            TOTSCORE_lo=pt.TOTSCORE_lo,
            TOTSCORE_hi=pt.TOTSCORE_hi,
            stage_probs=pt.stage_probs,
            conversion_prob=sim_conv,
        )

    return SimulateResponse(
        twin_id=twin_id,
        decline_multiplier=decline_multiplier,
        simulated_forecast=simulated,
    )


def run_scenario(twin_id: str, req: ScenarioRequest) -> ScenarioResponse:
    """
    Run hypothetical what-if scenario.
    Dry run — operates on copies, persists nothing to disk or in-memory cache.
    """
    if req.horizon_months not in [6, 12, 24]:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid horizon_months: {req.horizon_months}. Must be 6, 12, or 24."
        )

    twin = get_twin(twin_id)
    h_str = str(req.horizon_months)

    # 1. Baseline state
    base_state = twin.state
    base_anchor_stage = base_state.stage
    base_fc = base_state.forecast.get(h_str)
    base_p_ad = base_fc.stage_probs.AD if (base_fc and base_fc.stage_probs) else base_state.stage_probs.AD
    base_headline_metric = "p_ad_at_horizon"
    base_headline_val = base_p_ad
    base_prog_val = base_fc.conversion_prob if (base_fc and base_anchor_stage == "AD") else 0.0

    base_conv_dict = {k: base_state.conversion_risk.get(k, 0.0) for k in ["12m", "24m"]}
    if base_fc and base_fc.conversion_prob is not None:
        base_conv_dict[f"{h_str}m"] = base_fc.conversion_prob

    baseline_payload = {
        "stage_probs": base_fc.stage_probs.model_dump() if (base_fc and base_fc.stage_probs) else base_state.stage_probs.model_dump(),
        "anchor_stage_probs": base_state.stage_probs.model_dump(),
        "forecast": {h_str: base_fc.model_dump()} if base_fc else {},
        "conversion_risk": base_conv_dict,
        "progression_risk": base_prog_val,
        "p_ad_at_horizon": base_p_ad,
        "headline_risk_metric": base_headline_metric,
        "headline_risk_value": base_headline_val,
    }

    # 2. Build hypothetical inputs (preserving all anchor visit features)
    hypo_visits = [v.model_copy() for v in twin.visits]
    latest_v = hypo_visits[-1] if hypo_visits else VisitRecord(month=0)
    hypo_latest: Dict[str, Any] = dict(latest_v.model_dump(exclude_none=True))
    if req.hypothetical_visit:
        for k, v in req.hypothetical_visit.items():
            if v is not None:
                hypo_latest[k] = float(v)

    # Clone modalities
    hypo_modalities = {k: v.model_copy(deep=True) for k, v in twin.modalities.items()}

    # Biomarkers override
    if req.biomarkers:
        hypo_modalities["biomarkers"] = ModalityStatus(
            present=True,
            latest=req.biomarkers.model_dump(exclude_none=True)
        )

    # MRI volumes override
    if req.mri_volumes:
        hypo_modalities["mri_volumes"] = ModalityStatus(
            present=True,
            latest=req.mri_volumes.model_dump(exclude_none=True)
        )

    # 3. Scope warnings
    age_val = twin.demographics.get("age", 70.0)
    warnings = compute_scope_warnings(
        age=age_val,
        scores={
            "MMSE": latest_v.MMSE,
            "CDGLOBAL": latest_v.CDGLOBAL,
            "TOTSCORE": latest_v.TOTSCORE,
        },
        hypothetical_scores=req.hypothetical_visit,
        horizon_months=req.horizon_months,
    )

    # 4. Compute scenario state and forecast with lifestyle override
    effective_lifestyle = dict(twin.lifestyle or {})
    if req.lifestyle:
        effective_lifestyle.update(req.lifestyle)

    scen_state = _compute_state_and_forecast(
        twin.demographics, hypo_latest, hypo_visits, hypo_modalities, twin.baseline_date, effective_lifestyle
    )

    decline_mult = req.decline_multiplier if req.decline_multiplier is not None else 1.0
    scen_fc_points: Dict[str, ForecastPoint] = {}
    cur_m = float(hypo_latest.get("MMSE") if hypo_latest.get("MMSE") is not None else 26.0)
    cur_c = float(hypo_latest.get("CDGLOBAL") if hypo_latest.get("CDGLOBAL") is not None else 0.5)
    cur_t = float(hypo_latest.get("TOTSCORE") if hypo_latest.get("TOTSCORE") is not None else 12.0)

    for h, pt in scen_state.forecast.items():
        # Scale MMSE
        if pt.MMSE_mean is not None:
            d_m = pt.MMSE_mean - cur_m
            scaled_d_m = d_m * decline_mult if d_m < 0 else d_m / decline_mult
            new_mean = round(max(0.0, min(30.0, cur_m + scaled_d_m)), 1)
            new_lo = round(max(0.0, pt.MMSE_lo + (scaled_d_m - d_m)), 1)
            new_hi = round(min(30.0, pt.MMSE_hi + (scaled_d_m - d_m)), 1)
        else:
            new_mean, new_lo, new_hi = None, None, None

        # Scale CDGLOBAL
        if pt.CDGLOBAL_mean is not None:
            d_c = pt.CDGLOBAL_mean - cur_c
            scaled_d_c = d_c * decline_mult if d_c > 0 else d_c / decline_mult
            new_cdg = round(max(0.0, min(3.0, cur_c + scaled_d_c)), 2)
            new_cdg_lo = round(max(0.0, pt.CDGLOBAL_lo + (scaled_d_c - d_c)), 2)
            new_cdg_hi = round(min(3.0, pt.CDGLOBAL_hi + (scaled_d_c - d_c)), 2)
        else:
            new_cdg, new_cdg_lo, new_cdg_hi = None, None, None

        # Scale TOTSCORE
        if pt.TOTSCORE_mean is not None:
            d_t = pt.TOTSCORE_mean - cur_t
            scaled_d_t = d_t * decline_mult if d_t > 0 else d_t / decline_mult
            new_tot = round(max(0.0, min(70.0, cur_t + scaled_d_t)), 1)
            new_tot_lo = round(max(0.0, pt.TOTSCORE_lo + (scaled_d_t - d_t)), 1)
            new_tot_hi = round(min(70.0, pt.TOTSCORE_hi + (scaled_d_t - d_t)), 1)
        else:
            new_tot, new_tot_lo, new_tot_hi = None, None, None

        # Re-stage at horizon with Model A (TwinStager)
        fc_stage_inputs = dict(hypo_latest)
        fc_stage_inputs.update(twin.demographics)
        if new_mean is not None: fc_stage_inputs["MMSE"] = new_mean
        if new_cdg is not None: fc_stage_inputs["CDGLOBAL"] = new_cdg
        if new_tot is not None: fc_stage_inputs["TOTSCORE"] = new_tot
        st_res = get_stager().predict_one(fc_stage_inputs)
        h_stage_probs = {
            "CN": round(float(st_res["probs"]["CN"]), 4),
            "MCI": round(float(st_res["probs"]["MCI"]), 4),
            "AD": round(float(st_res["probs"]["AD"]), 4),
        }

        # Conversion / Progression risk at horizon
        if scen_state.stage == "AD":
            sigma_m = max(1.0, HORIZON_RESIDUAL_MAE.get(int(h), {}).get("MMSE", 1.5) * 1.25)
            d_m_val = (new_mean - cur_m) if new_mean is not None else 0.0
            z_m = (-3.0 - d_m_val) / sigma_m
            prog_risk = round(float(0.5 * (1.0 + math.erf(z_m / math.sqrt(2.0)))), 4)
            h_conv_risk = max(0.05, min(0.98, prog_risk))
        else:
            h_conv_risk = round(min(1.0, max(0.0, (pt.conversion_prob or 0.0) * decline_mult)), 4) if pt.conversion_prob is not None else 0.0

        scen_fc_points[h] = ForecastPoint(
            month=pt.month,
            date=pt.date,
            MMSE_mean=new_mean,
            MMSE_lo=new_lo,
            MMSE_hi=new_hi,
            CDGLOBAL_mean=new_cdg,
            CDGLOBAL_lo=new_cdg_lo,
            CDGLOBAL_hi=new_cdg_hi,
            TOTSCORE_mean=new_tot,
            TOTSCORE_lo=new_tot_lo,
            TOTSCORE_hi=new_tot_hi,
            stage_probs=StageProbs(**h_stage_probs),
            conversion_prob=h_conv_risk,
        )

    scen_fc_target = scen_fc_points.get(h_str)
    scen_anchor_stage = scen_state.stage
    scen_p_ad = scen_fc_target.stage_probs.AD if (scen_fc_target and scen_fc_target.stage_probs) else scen_state.stage_probs.AD

    scen_headline_metric = "p_ad_at_horizon"
    scen_headline_val = scen_p_ad
    scen_prog_val = scen_fc_target.conversion_prob if (scen_fc_target and scen_anchor_stage == "AD") else 0.0
    scen_conv_dict = {f"{h}m": scen_fc_points[h].conversion_prob for h in ["12", "24"] if h in scen_fc_points}
    if not scen_conv_dict:
        scen_conv_dict = {f"{h_str}m": scen_headline_val}

    scen_payload = {
        "stage_probs": scen_fc_target.stage_probs.model_dump() if (scen_fc_target and scen_fc_target.stage_probs) else scen_state.stage_probs.model_dump(),
        "anchor_stage_probs": scen_state.stage_probs.model_dump(),
        "forecast": {h_str: scen_fc_target.model_dump()} if scen_fc_target else {},
        "conversion_risk": scen_conv_dict,
        "progression_risk": scen_prog_val,
        "p_ad_at_horizon": scen_p_ad,
        "headline_risk_metric": scen_headline_metric,
        "headline_risk_value": scen_headline_val,
    }

    # 5. Compute deltas (delta = scenario - baseline)
    b_p = baseline_payload["stage_probs"]
    s_p = scen_payload["stage_probs"]

    delta_p_ad = round(s_p["AD"] - b_p["AD"], 4)
    delta_headline = round(scen_headline_val - base_headline_val, 4)
    delta_prog = round(scen_prog_val - base_prog_val, 4)

    delta_mmse = round(scen_fc_target.MMSE_mean - base_fc.MMSE_mean, 2) if (base_fc and scen_fc_target and base_fc.MMSE_mean is not None and scen_fc_target.MMSE_mean is not None) else None
    delta_cdg = round(scen_fc_target.CDGLOBAL_mean - base_fc.CDGLOBAL_mean, 2) if (base_fc and scen_fc_target and base_fc.CDGLOBAL_mean is not None and scen_fc_target.CDGLOBAL_mean is not None) else None
    delta_tot = round(scen_fc_target.TOTSCORE_mean - base_fc.TOTSCORE_mean, 2) if (base_fc and scen_fc_target and base_fc.TOTSCORE_mean is not None and scen_fc_target.TOTSCORE_mean is not None) else None

    def _get_direction(metric_name: str, diff: Optional[float], eps: float = 1e-4) -> str:
        if diff is None or abs(diff) <= eps:
            return "same"
        # Higher is better for MMSE and CN
        if metric_name in ["MMSE_mean", "CN", "stage_CN"]:
            return "better" if diff > 0 else "worse"
        # Higher is worse for AD, MCI, CDR, ADAS, conversion_risk, progression_risk, headline_risk
        return "worse" if diff > 0 else "better"

    delta: Dict[str, Any] = {
        "stage_probs": {
            "CN": round(s_p["CN"] - b_p["CN"], 4),
            "MCI": round(s_p["MCI"] - b_p["MCI"], 4),
            "AD": delta_p_ad,
        },
        "conversion_risk": {
            k: round(scen_conv_dict.get(k, 0.0) - base_conv_dict.get(k, 0.0), 4)
            for k in ["12m", "24m"]
        },
        "MMSE_mean": delta_mmse,
        "CDGLOBAL_mean": delta_cdg,
        "TOTSCORE_mean": delta_tot,
        "p_ad_at_horizon": delta_p_ad,
        "headline_risk": delta_headline,
        "progression_risk": delta_prog,
        "direction": {
            "p_ad_at_horizon": _get_direction("p_ad_at_horizon", delta_p_ad),
            "headline_risk": _get_direction("headline_risk", delta_headline),
            "progression_risk": _get_direction("progression_risk", delta_prog),
            "conversion_risk": _get_direction("conversion_risk", delta_headline),
            "MMSE_mean": _get_direction("MMSE_mean", delta_mmse),
            "CDGLOBAL_mean": _get_direction("CDGLOBAL_mean", delta_cdg),
            "TOTSCORE_mean": _get_direction("TOTSCORE_mean", delta_tot),
            "AD": _get_direction("AD", delta_p_ad),
            "CN": _get_direction("CN", s_p["CN"] - b_p["CN"]),
        },
        "directions": {
            "p_ad_at_horizon": _get_direction("p_ad_at_horizon", delta_p_ad),
            "headline_risk": _get_direction("headline_risk", delta_headline),
            "progression_risk": _get_direction("progression_risk", delta_prog),
            "conversion_risk_12m": _get_direction("conversion_risk", scen_conv_dict.get("12m", 0.0) - base_conv_dict.get("12m", 0.0)),
            "conversion_risk_24m": _get_direction("conversion_risk", scen_conv_dict.get("24m", 0.0) - base_conv_dict.get("24m", 0.0)),
            "MMSE_mean": _get_direction("MMSE_mean", delta_mmse),
            "CDGLOBAL_mean": _get_direction("CDGLOBAL_mean", delta_cdg),
            "TOTSCORE_mean": _get_direction("TOTSCORE_mean", delta_tot),
            "AD": _get_direction("AD", delta_p_ad),
            "CN": _get_direction("CN", s_p["CN"] - b_p["CN"]),
        },
        "metrics": {
            "p_ad_at_horizon": {
                "baseline": b_p["AD"],
                "scenario": s_p["AD"],
                "delta": delta_p_ad,
                "direction": _get_direction("p_ad_at_horizon", delta_p_ad),
            },
            "headline_risk": {
                "metric": scen_headline_metric,
                "baseline": base_headline_val,
                "scenario": scen_headline_val,
                "delta": delta_headline,
                "direction": _get_direction("headline_risk", delta_headline),
            },
            "progression_risk": {
                "baseline": base_prog_val,
                "scenario": scen_prog_val,
                "delta": delta_prog,
                "direction": _get_direction("progression_risk", delta_prog),
            },
            "MMSE_mean": {
                "baseline": base_fc.MMSE_mean if base_fc else None,
                "scenario": scen_fc_target.MMSE_mean if scen_fc_target else None,
                "delta": delta_mmse,
                "direction": _get_direction("MMSE_mean", delta_mmse),
            },
            "CDGLOBAL_mean": {
                "baseline": base_fc.CDGLOBAL_mean if base_fc else None,
                "scenario": scen_fc_target.CDGLOBAL_mean if scen_fc_target else None,
                "delta": delta_cdg,
                "direction": _get_direction("CDGLOBAL_mean", delta_cdg),
            },
            "TOTSCORE_mean": {
                "baseline": base_fc.TOTSCORE_mean if base_fc else None,
                "scenario": scen_fc_target.TOTSCORE_mean if scen_fc_target else None,
                "delta": delta_tot,
                "direction": _get_direction("TOTSCORE_mean", delta_tot),
            },
        }
    }

    combined_warnings = list(dict.fromkeys(warnings + scen_state.scope_warnings))
    scen_reliability = scen_state.forecast_reliability or "ok"

    # Build scenario drivers delta & method explanation
    drivers_delta: List[Dict[str, Any]] = []
    if req.hypothetical_visit:
        for k, v in req.hypothetical_visit.items():
            if v is not None:
                base_v = getattr(latest_v, k, None) if hasattr(latest_v, k) else twin.demographics.get(k)
                if base_v is not None:
                    d = round(float(v) - float(base_v), 2)
                    drivers_delta.append({
                        "factor": k,
                        "label": k,
                        "delta": d,
                        "impact": f"Hypothetical change of {d:+.1f} points in {k} shifts progression trajectory."
                    })
    if req.decline_multiplier is not None and req.decline_multiplier != 1.0:
        drivers_delta.append({
            "factor": "decline_multiplier",
            "label": "Decline Rate Multiplier",
            "delta": req.decline_multiplier,
            "impact": f"Decline multiplier of {req.decline_multiplier}x applied to cognitive loss velocity."
        })
    if req.biomarkers:
        drivers_delta.append({
            "factor": "biomarkers",
            "label": "CSF Biomarkers",
            "delta": "attached/modified",
            "impact": "CSF biomarker payload promoted staging to Tier T3."
        })
    if req.mri_volumes:
        drivers_delta.append({
            "factor": "mri_volumes",
            "label": "Volumetric MRI",
            "delta": "attached/modified",
            "impact": "Volumetric MRI measurements promoted staging to Tier T4."
        })
    if req.lifestyle:
        drivers_delta.append({
            "factor": "lifestyle",
            "label": "Lifestyle Modification",
            "delta": str(req.lifestyle),
            "impact": "Hypothetical lifestyle override adjusted progression decline multiplier."
        })

    method_steps = [
        "1. Take current state and hypothetical follow-up visit or counterfactual inputs.",
        "2. Forecast cognitive scores (MMSE, CDGLOBAL, TOTSCORE) at the chosen horizon (6, 12, or 24 months) using Model B delta progression models.",
        "3. Apply the decline multiplier and literature-grounded lifestyle modifier to the predicted change.",
        "4. Re-stage the patient with Model A (TwinStager) using the forecast scores.",
        "5. Compute conversion risk across horizons (restricted to 6, 12, and 24 months because empirical validation and clinical protocol intervals in ADNI are standardized at these milestones; projections beyond 24m carry unvalidated cumulative drift)."
    ]

    return ScenarioResponse(
        twin_id=twin_id,
        horizon_months=req.horizon_months,
        baseline=baseline_payload,
        scenario=scen_payload,
        delta=delta,
        scope_warnings=combined_warnings,
        forecast_reliability=scen_reliability,
        method=method_steps,
        drivers_delta=drivers_delta,
    )


def undo_last_visit(twin_id: str) -> TwinFull:
    """Remove the most recent visit (undo) and restore previous state."""
    twin = get_twin(twin_id)
    if len(twin.visits) <= 1:
        raise HTTPException(
            status_code=400,
            detail="Cannot undo visit: twin has only baseline visit."
        )

    removed = twin.visits.pop()
    twin.visits.sort(key=lambda v: v.month)
    latest_v = twin.visits[-1]
    latest_dict = {
        "MMSE": latest_v.MMSE,
        "CDGLOBAL": latest_v.CDGLOBAL,
        "TOTSCORE": latest_v.TOTSCORE,
    }
    twin.modalities["basic_scores"] = ModalityStatus(
        present=any(v is not None for v in [latest_v.MMSE, latest_v.CDGLOBAL, latest_v.TOTSCORE]),
        latest={"MMSE": latest_v.MMSE, "CDGLOBAL": latest_v.CDGLOBAL, "TOTSCORE": latest_v.TOTSCORE}
    )

    new_state = _compute_state_and_forecast(
        twin.demographics, latest_dict, twin.visits, twin.modalities, twin.baseline_date, twin.lifestyle
    )
    twin.state = new_state
    now = datetime.now(timezone.utc).isoformat()
    twin.history.append(
        HistoryEvent(
            recorded_at=now,
            timestamp=now,
            stage_probs=new_state.stage_probs,
            event=f"visit_undone_m{removed.month}",
            prob_diff=None
        )
    )
    _save_twin_to_disk(twin)
    return twin


def reset_twin(twin_id: str) -> TwinFull:
    """Reset twin back to baseline state, removing subsequent visits and scenarios."""
    twin = get_twin(twin_id)
    now = datetime.now(timezone.utc).isoformat()
    if twin.baseline_snapshot:
        snap = twin.baseline_snapshot
        twin.demographics = dict(snap.get("inputs", {}).get("demographics", twin.demographics))
        restored_modalities = {}
        for k, v in snap.get("attached_modalities", {}).items():
            restored_modalities[k] = ModalityStatus(**v)
        twin.modalities = restored_modalities
        twin.visits = [VisitRecord(**v) for v in snap.get("visits", [])]
        twin.scenarios = []
        twin.lifestyle = dict(snap.get("lifestyle", {})) if snap.get("lifestyle") else None
        twin.state = TwinState(**snap["stage"])
    else:
        baseline_visit = twin.visits[0] if twin.visits else VisitRecord(month=0, exam_date=twin.baseline_date)
        twin.visits = [baseline_visit]
        twin.scenarios = []
        latest_dict = {
            "MMSE": baseline_visit.MMSE,
            "CDGLOBAL": baseline_visit.CDGLOBAL,
            "TOTSCORE": baseline_visit.TOTSCORE,
        }
        twin.modalities["basic_scores"] = ModalityStatus(
            present=any(v is not None for v in [baseline_visit.MMSE, baseline_visit.CDGLOBAL, baseline_visit.TOTSCORE]),
            latest={"MMSE": baseline_visit.MMSE, "CDGLOBAL": baseline_visit.CDGLOBAL, "TOTSCORE": baseline_visit.TOTSCORE}
        )
        twin.state = _compute_state_and_forecast(
            twin.demographics, latest_dict, twin.visits, twin.modalities, twin.baseline_date
        )

    twin.history.append(
        HistoryEvent(
            recorded_at=now,
            timestamp=now,
            stage_probs=twin.state.stage_probs,
            event="reset",
            prob_diff=None
        )
    )
    _save_twin_to_disk(twin)
    return twin


def get_input_usage(twin_id: str) -> InputUsageResponse:
    twin = get_twin(twin_id)
    from ml.twin.explainability import get_input_usage_matrix
    raw_items = get_input_usage_matrix(twin.model_dump())
    items = [InputUsageItem(**item) for item in raw_items]
    return InputUsageResponse(twin_id=twin_id, items=items, field_count=len(items))


def get_twin_explain(twin_id: str) -> TwinExplanationResponse:
    twin = get_twin(twin_id)
    from ml.twin.explainability import get_full_twin_explanation
    raw_expl = get_full_twin_explanation(twin.model_dump())
    return TwinExplanationResponse(**raw_expl)
