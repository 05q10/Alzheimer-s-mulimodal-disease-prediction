# Digital Twin API Contract — Live Specification

> **STATUS: LIVE BACKEND WITH ML INFERENCE**
> Powered by calibrated CatBoost TwinStager (T1a/T1b/T2a/T2b/T3/T4) and TwinProgression (6m, 12m, 24m horizons).
> The URL, HTTP method, request schema, and response schema are strictly synchronized with `twin_router.py`.

---

## Base URL (local dev)
```
http://localhost:8000
```

---

## Routes

### 1. `POST /twin` — Create Twin

Supports both **Patient-Builder flow** (nested objects) and legacy flat shape. Missing values mean missing (NaN), never 0. Runs stager at highest applicable tier.

**Patient-Builder Request body** (`application/json`):
```json
{
  "name": "Jane Doe",
  "demographics": {
    "age": 72.0,
    "sex": 1,
    "education_years": 16.0,
    "apoe4": 1
  },
  "scores": {
    "MMSE": 26.0,
    "CDGLOBAL": 0.5,
    "TOTSCORE": 16.0,
    "CDRSB": 1.5,
    "MOCA": 24.0,
    "FAQ": 3.0
  },
  "biomarkers": {
    "ABETA": 720.0,
    "TAU": 390.0,
    "PTAU": 38.0,
    "APVOLUME": 3200.0
  },
  "mri_volumes": {
    "hippocampus": 6400.0,
    "entorhinal": 3200.0,
    "ventricles": 38000.0,
    "whole_brain": 1020000.0,
    "icv": 1500000.0
  },
  "lifestyle": {
    "BMI": 26.5,
    "PhysicalActivity": 3,
    "DietQuality": 4,
    "SleepQuality": 3
  },
  "visits": []
}
```

*Notes*:
- If `lifestyle` is provided, runs Kaggle CatBoost and fills `kaggle_clinical` modality chip (separate lifestyle risk, not blended).
- If `ABETA`, `TAU`, and `APVOLUME` are provided, runs allele ensemble and fills `allele_ensemble` modality. `APVOLUME` is never mapped to FreeSurfer hippocampus.

**Flat Legacy Request body**: Also accepted with flat `age`, `sex`, `education_years`, `apoe4_count`, `MMSE`, etc.

**Response**: `TwinFull` object.

---

### 2. `GET /twin/{twin_id}` — Get Twin State

Returns the full `TwinFull` object for the given `twin_id`.

---

### 3. `POST /twin/{twin_id}/visit` — Add Visit

**Request body**:
```json
{
  "month": 12,
  "MMSE": 24.5,
  "CDGLOBAL": 0.5,
  "TOTSCORE": 14.0,
  "DX_obs": null,
  "source": "user"
}
```

**Response**: `AddVisitResponse`
```json
{
  "twin_id": "...",
  "state": { ... },
  "prob_diff": { "CN": -0.04, "MCI": +0.01, "AD": +0.03 }
}
```

---

### 4. Dedicated Modality Attachment Routes

#### 4a. `POST /twin/{twin_id}/biomarkers` — Attach CSF Biomarkers
```json
{ "ABETA": 850.0, "TAU": 290.0, "PTAU": 28.0, "APVOLUME": 3100.0 }
```

#### 4b. `POST /twin/{twin_id}/mri-volumes` — Attach FreeSurfer MRI Volumes
```json
{ "hippocampus": 6800.0, "entorhinal": 3400.0, "ventricles": 35000.0, "whole_brain": 1050000.0, "icv": 1500000.0 }
```

#### 4c. `POST /twin/{twin_id}/mri-image` — Upload MRI Slice & Late Fusion
Accepts multipart form-data with field `"file"` (JPEG/PNG MRI slice).
Runs ResNet34 with temperature scaling ($T=2.0$), maps predictions (`NonDemented` $\rightarrow$ `CN`, `VeryMild` $\rightarrow$ `MCI`, `Mild`+`Moderate` $\rightarrow$ `AD`), and fuses linearly with ADNI stager:
$\text{final} = 0.8 \times P_{\text{adni}} + 0.2 \times P_{\text{mri\_mapped}}$

**Response**:
```json
{
  "mri": {
    "predicted_class": "Mild",
    "probs": { "NonDemented": 0.0001, "VeryMild": 0.0008, "Mild": 0.9989, "Moderate": 0.0001 },
    "mapped_probs": { "CN": 0.0001, "MCI": 0.0008, "AD": 0.999 }
  },
  "state": {
    "stage": "AD",
    "stage_probs": { "CN": 0.04, "MCI": 0.21, "AD": 0.75 },
    "fusion": {
      "adni_probs": { "CN": 0.05, "MCI": 0.26, "AD": 0.69 },
      "mri_probs": { "CN": 0.0001, "MCI": 0.0008, "AD": 0.999 },
      "weight": 0.2,
      "final_probs": { "CN": 0.04, "MCI": 0.21, "AD": 0.75 }
    }
  }
}
```

#### 4d. `POST /twin/{twin_id}/modality/{modality_name}` — Generic Modality Router
Supports `biomarkers`, `mri_volumes`, `mri_image`, `ocr_report`.

---

### 5. `GET /twin/{twin_id}/forecast?horizon=24` — Get Progression Forecast

Query param: `horizon` = 6 | 12 | 24 (months, default 24).
**Response**:
```json
{
  "twin_id": "...",
  "horizon_months": 24,
  "forecast": {
    "MMSE_mean": 19.4,
    "MMSE_lo": 14.2,
    "MMSE_hi": 24.0,
    "CDGLOBAL_mean": 1.1,
    "stage_probs": { "CN": 0.03, "MCI": 0.37, "AD": 0.60 },
    "conversion_prob": 0.60
  }
}
```

---

### 6. `POST /twin/{twin_id}/scenario` — Counterfactual What-If Scenario

**Dry run: persists nothing.** Validates `horizon_months` (strictly 6, 12, or 24 months; HTTP 422 for others). Projections beyond 24m carry unvalidated cumulative drift, and ADNI clinical protocols standardize visits at 6, 12, and 24 month intervals.

**Method Execution Steps**:
1. Take the current baseline state and the hypothetical follow-up visit or counterfactual inputs.
2. Forecast cognitive scores (MMSE, CDGLOBAL, TOTSCORE) at the chosen horizon (6, 12, or 24 months) using Model B delta progression models.
3. Apply the decline multiplier and literature-grounded lifestyle modifier to the predicted change:
   - `decline_multiplier` scales the predicted change $\Delta$ (e.g. 1.5x accelerates decline by 50%; 0.5x slows decline by 50%).
   - `lifestyle` applies Lancet Commission pooled relative risks shrunk via square root ($\text{RR}^{0.5}$) and hard-capped to $[0.85, 1.35]$ on predicted change and conversion hazard.
   - Crucial clinical guard: Lifestyle and decline multipliers NEVER modify current clinical stage or baseline probabilities.
4. Re-stage the patient with Model A (TwinStager) using the forecast scores.
5. Compute conversion risk across horizons.

**Request body**:
```json
{
  "horizon_months": 12,
  "hypothetical_visit": {
    "MMSE": 22.0,
    "CDGLOBAL": 1.0,
    "TOTSCORE": 20.0
  },
  "biomarkers": { "ABETA": 600.0, "TAU": 450.0, "PTAU": 45.0 },
  "mri_volumes": { "hippocampus": 5900.0, "icv": 1500000.0 },
  "lifestyle": { "alcohol": 2.0, "physical_activity": 4.0 },
  "decline_multiplier": 1.2
}
```

**Response**: `ScenarioResponse`
```json
{
  "twin_id": "...",
  "horizon_months": 12,
  "baseline": {
    "stage_probs": { "CN": 0.15, "MCI": 0.70, "AD": 0.15 },
    "forecast": { "12": { "MMSE_mean": 24.1, "conversion_prob": 0.28 } },
    "conversion_risk": { "12m": 0.28, "24m": 0.45 }
  },
  "scenario": {
    "stage_probs": { "CN": 0.02, "MCI": 0.48, "AD": 0.50 },
    "forecast": { "12": { "MMSE_mean": 20.8, "conversion_prob": 0.58 } },
    "conversion_risk": { "12m": 0.58, "24m": 0.72 }
  },
  "delta": {
    "stage_probs": { "CN": -0.13, "MCI": -0.22, "AD": +0.35 },
    "conversion_risk": { "12m": +0.30, "24m": +0.27 },
    "MMSE_mean": -3.3
  },
  "scope_warnings": [
    "Implausible MMSE change of -4.0 points over 12 months..."
  ],
  "forecast_reliability": "ok",
  "disclaimer": "Hypothetical what-if scenario. Projections represent counterfactual simulations...",
  "method": [
    "1. Take current state and hypothetical follow-up visit or counterfactual inputs.",
    "2. Forecast cognitive scores (MMSE, CDGLOBAL, TOTSCORE) at the chosen horizon (6, 12, or 24 months) using Model B delta progression models.",
    "3. Apply the decline multiplier and literature-grounded lifestyle modifier to the predicted change.",
    "4. Re-stage the patient with Model A (TwinStager) using the forecast scores.",
    "5. Compute conversion risk across horizons (restricted to 6, 12, and 24 months because empirical validation and clinical protocol intervals in ADNI are standardized at these milestones; projections beyond 24m carry unvalidated cumulative drift)."
  ],
  "drivers_delta": [
    { "factor": "MMSE", "label": "MMSE", "delta": -4.0, "impact": "Hypothetical change of -4.0 points in MMSE shifts progression trajectory." },
    { "factor": "decline_multiplier", "label": "Decline Rate Multiplier", "delta": 1.2, "impact": "Decline multiplier of 1.2x applied to cognitive loss velocity." }
  ]
}
```

---

### 7. `DELETE /twin/{twin_id}/visit/last` — Undo Last Visit
Removes the most recent follow-up visit and recalculates the prior clinical stage and trajectory from remaining visits.

---

### 8. `POST /twin/{twin_id}/reset` — Reset to Baseline
Restores EXACT baseline snapshot from creation time, clearing visits and scenarios, and appends a "reset" history event.

---

### 9. `POST /twin/{twin_id}/simulate` — What-If Simulation
Decline scaling (`decline_multiplier`: 0.5–1.5) across all cognitive targets (MMSE, CDGLOBAL, TOTSCORE). Carries mandatory illustrative disclaimer.

---

### 10. `GET /twin/{twin_id}/input-usage` — Input Usage Matrix & Data Lineage
Returns one row for every one of the 43 intake fields collected by the application.
Status values: `used_in_stage`, `used_in_forecast`, `used_in_lifestyle_modifier`, `used_in_symptom_screen`, `used_in_biomarker_ensemble`, `used_in_mri_fusion`, `recorded_only`, `missing`, `assumed_default`.

**Response**:
```json
{
  "twin_id": "...",
  "field_count": 43,
  "items": [
    {
      "field": "age",
      "label": "Patient Age",
      "category": "Demographics",
      "value": 72.0,
      "status": "used_in_stage",
      "display_status": "Used in stage",
      "used_by": ["stager", "progression"],
      "reason": "Primary demographic covariate across ADNI staging and longitudinal progression curves."
    },
    {
      "field": "alcohol",
      "label": "Alcohol Consumption",
      "category": "Lifestyle",
      "value": 2.4,
      "status": "used_in_lifestyle_modifier",
      "display_status": "Lifestyle adjustment",
      "used_by": ["lifestyle_modifier", "symptom_screen"],
      "reason": "Alcohol: no alcohol field exists in ADNI, so it adjusts the forecast through a literature-based multiplier, not the stage."
    },
    {
      "field": "ethnicity",
      "label": "Ethnicity",
      "category": "Demographics",
      "value": 0,
      "status": "recorded_only",
      "display_status": "Recorded only",
      "used_by": [],
      "reason": "Ethnicity: stored only; ADNI model does not use it to avoid demographic bias."
    }
  ]
}
```

---

### 11. `GET /twin/{twin_id}/explain` — Comprehensive Model Explainability
Returns TreeSHAP stage drivers, modality leave-one-out marginal effects, progression forecast drivers, lifestyle modifier breakdown, Kaggle symptom screener, and deterministic 4-to-8 sentence clinical narrative.

**Response**:
```json
{
  "twin_id": "...",
  "stage": {
    "primary": "MCI",
    "stage": "MCI",
    "confidence": 0.86,
    "probs": { "CN": 0.05, "MCI": 0.86, "AD": 0.09 },
    "drivers": [
      {
        "feature": "CDGLOBAL",
        "label": "Clinical Dementia Rating (CDR)",
        "group": "cognitive",
        "value": 0.5,
        "contribution": 0.9125,
        "direction": "toward_MCI"
      }
    ],
    "leave_one_out": [
      {
        "modality": "advanced_scores",
        "label": "Advanced Neuropsych Instruments",
        "without_stage": "CN",
        "without_probs": { "CN": 0.65, "MCI": 0.23, "AD": 0.12 },
        "delta": { "CN": -0.60, "MCI": +0.63, "AD": -0.03 },
        "summary": "Without advanced neuropsych scores: P(MCI) would be 23% instead of 86%."
      }
    ],
    "by_group": {
      "cognitive": 1.15,
      "functional": 0.04,
      "biomarker": 0.0,
      "mri_volumes": 0.03,
      "demographic": 0.02,
      "genetics": 0.0,
      "risk_factor": 0.0
    }
  },
  "forecast": {
    "drivers": [
      {
        "feature": "TOTSCORE",
        "label": "ADAS11-Cog Total",
        "target": "MMSE_12m",
        "contribution": -0.32,
        "direction": "toward faster decline",
        "effect": "accelerating",
        "impact": "ADAS11-Cog Total pushes toward faster decline (-0.32 pts)."
      }
    ],
    "reliability": "high",
    "forecast_reliability": "ok",
    "scope_warnings": []
  },
  "lifestyle_modifier": {
    "total_multiplier": 1.04,
    "hazard_multiplier": 1.04,
    "multipliers": [
      {
        "factor": "alcohol",
        "label": "Alcohol Consumption",
        "value": 3.5,
        "multiplier": 1.02,
        "description": "Alcohol Consumption x1.02 (Lancet 2024)"
      }
    ],
    "caption": "Literature-informed estimate, applied to the forecast and conversion risk only. It does not change the current stage and was not learned from ADNI."
  },
  "symptom_screen": {
    "risk_level": "Moderate Risk",
    "risk_score": 0.42,
    "drivers": [],
    "caption": "A separate screening model from a different population; shown for reference and not part of the ADNI stage"
  },
  "reliability": "high",
  "forecast_reliability": "ok",
  "scope_warnings": [],
  "narrative": "Stage MCI (86% confidence). The biggest push toward MCI is the Clinical Dementia Rating (CDR) (0.5) and Mini-Mental State Exam (MMSE) (26.0). Without advanced neuropsych scores: P(MCI) would be 23% instead of 86%. Progression trajectory is primarily shaped by ADAS11-Cog Total, which acts as an accelerating factor. All clinical measurements fall within the validated 5th–95th percentile training envelope."
}
```

---

### 12. `GET /demo/patients` — List Demo Personas
Returns 6 pre-built demo patient entries (real ADNI + synthetic flagged).

---

### 13. `GET /model-card` — Model Metrics, Scope & Disclaimers
All numbers synchronized with `server/final_metrics.json`.

Returns held-out test metrics, valid clinical bounds and scopes, limitations list, and data provenance.
```json
{
  "stager": { "T1_accuracy": 0.81 },
  "progression": { "MMSE_MAE_12m": 1.465 },
  "scope": {
    "age_range": [60.7, 84.7],
    "horizons_months": [6, 12, 24],
    "score_ranges": {
      "MMSE": [0, 30],
      "CDGLOBAL": [0, 3],
      "TOTSCORE": [0, 70]
    },
    "input_ranges": {
      "age": [60.7, 84.7],
      "MMSE": [20.0, 30.0],
      "CDGLOBAL": [0.0, 1.0],
      "TOTSCORE": [2.3, 26.3],
      "FAQTOTAL": [0.0, 24.0]
    },
    "training_cohort": "ADNI, 2123 subjects"
  },
  "limitations": [ ... ],
  "resnet_disclaimer": "Image model trained on a Kaggle MRI dataset (not ADNI); fused at weight 0.2.",
  "fusion_disclaimer": "Image model trained on a Kaggle MRI dataset (not ADNI); fused at weight 0.2.",
  "lifestyle_modifier": {
    "status": "literature-informed prior, not learned from ADNI",
    "hard_cap": [0.85, 1.35],
    "shrinkage_power": 0.5,
    "source": "Lancet Commission on Dementia Prevention, Intervention, and Care (Livingston et al., 2020, 2024 updates)",
    "note": "Transparently modifies predicted decline and conversion hazard; never modifies current clinical stage."
  }
}
```

### 10. `GET /demo/cases` — List Held-Out Replay Cases

Returns all transparently selected, held-out ADNI test patient replay cases (`case_a_stable_cn`, `case_b_cn_to_mci`, `case_c_mci_to_ad`, `case_d_stable_mci`, `case_hard_cognitive_volatility`).

**Response** (`200 OK`):
```json
[
  {
    "case_id": "case_a_stable_cn",
    "alias": "Eleanor Vance",
    "trajectory_type": "Stable Cognitively Normal (CN)",
    "baseline_stage": "CN",
    "brief": "Patient is a 76-year-old female with 16 years education and zero APOE e4 alleles...",
    "demographics": { "age": 76.0, "sex": 1, "education_years": 16.0, "apoe4_count": 0 },
    "num_visits": 4,
    "illustrative_mri_available": true
  }
]
```

---

### 11. `POST /demo/replay/{case}/start` — Start Replay from Visit 0

Initializes digital twin strictly from baseline Visit 0 (month 0). Precomputes and freezes SHA-256 pre-visit prediction for Visit 1 before any subsequent data is revealed.

**Query Parameters / Body**:
- `include_mri`: `boolean` (default: `true`, attach illustrative Kaggle MRI slice)

**Response** (`200 OK`):
```json
{
  "twin": { /* TwinFull state at baseline */ },
  "replay": {
    "twin_id": "uuid",
    "case_id": "case_a_stable_cn",
    "alias": "Eleanor Vance",
    "current_visit_index": 0,
    "total_visits": 4,
    "is_complete": false,
    "pending_pre_visit_prediction": {
      "visit_index": 1,
      "target_month": 6,
      "anchor_month": 0,
      "horizon_months": 6,
      "predicted_MMSE": 29.9,
      "predicted_MMSE_band": { "lo": 27.6, "hi": 30.0 },
      "predicted_CDGLOBAL": 0.0,
      "predicted_CDGLOBAL_band": { "lo": 0.0, "hi": 0.1 },
      "predicted_TOTSCORE": 4.1,
      "predicted_TOTSCORE_band": { "lo": 0.0, "hi": 9.0 },
      "predicted_stage": "CN",
      "prediction_hash": "sha256_hex_digest"
    }
  }
}
```

---

### 12. `POST /demo/replay/{case}/next` — Advance Replay (Predict Before Reveal)

Advances replay to visit $k$. Enforces strict pre-visit prediction hash integrity, reveals true ADNI scores, computes score errors and 90% band containment, records stage agreement, and freezes next forecast.

**Query Parameters / Body**:
- `twin_id`: `string` (optional, defaults to active twin for case)

**Response** (`200 OK`):
```json
{
  "twin": { /* Updated TwinFull */ },
  "evaluation": {
    "visit_index": 1,
    "month": 6,
    "exam_date": "2025-03-01",
    "actual_scores": { "MMSE": 29.0, "CDGLOBAL": 0.0, "TOTSCORE": 6.0 },
    "predicted_scores": {
      "MMSE": 29.9,
      "MMSE_band": { "lo": 27.6, "hi": 30.0 },
      "CDGLOBAL": 0.0,
      "CDGLOBAL_band": { "lo": 0.0, "hi": 0.1 },
      "TOTSCORE": 4.1,
      "TOTSCORE_band": { "lo": 0.0, "hi": 9.0 }
    },
    "errors": {
      "MMSE": -0.9,
      "abs_MMSE": 0.9,
      "in_band_MMSE": true,
      "CDGLOBAL": 0.0,
      "abs_CDGLOBAL": 0.0,
      "in_band_CDGLOBAL": true,
      "TOTSCORE": 1.9,
      "abs_TOTSCORE": 1.9,
      "in_band_TOTSCORE": true
    },
    "stage_comparison": {
      "true_stage": "CN",
      "predicted_stage": "CN",
      "stage_agrees": true
    },
    "prediction_hash": "sha256_hex_digest",
    "pre_visit_timestamp": "2026-10-30T10:00:00Z",
    "reveal_timestamp": "2026-10-30T10:00:01Z"
  },
  "replay": { /* Updated run state */ }
}
```

---

### 13. `GET /demo/replay/{case}/report` — Replay Audit Report

Returns visit-by-visit audit table and summary metrics (MAE per score, stage agreement rate, 90% predictive interval band coverage).

**Response** (`200 OK`):
```json
{
  "case_id": "case_a_stable_cn",
  "twin_id": "uuid",
  "alias": "Eleanor Vance",
  "trajectory_type": "Stable Cognitively Normal (CN)",
  "baseline_stage": "CN",
  "is_complete": true,
  "summary_metrics": {
    "num_evaluated_visits": 3,
    "mae_MMSE": 0.60,
    "mae_CDGLOBAL": 0.000,
    "mae_TOTSCORE": 2.08,
    "stage_agreement_rate": 1.0,
    "band_coverage_rate": 1.0,
    "empirical_percentiles": {
      "MMSE": { "6m": 42.0, "12m": 44.6, "24m": 0.0 }
    }
  },
  "visits": [ /* All 4 visits with evaluations */ ]
}
```

---

### 14. `GET /demo/replay/summary` — Population Evidence vs Replay Summary

Returns population test set metrics ($N = 456$, $3,022$ visits) alongside the 4 replay patient summaries and anchor error percentiles.

**Response** (`200 OK`):
```json
{
  "evidence_statement": "Four patients are an illustration and the test-set metrics are the evidence.",
  "population_test_metrics": {
    "cohort_evidence": "456 subjects, 3,022 visits strictly partitioned in held-out test split (server/data/splits.json)",
    "stager_model_a": { ... },
    "progression_model_b": { ... }
  },
  "anchor_error_distribution": { ... },
  "replay_patients": [
    {
      "case_id": "case_a_stable_cn",
      "alias": "Eleanor Vance",
      "mae_MMSE": 0.60,
      "mae_CDGLOBAL": 0.000,
      "mae_TOTSCORE": 2.08,
      "stage_agreement_rate": 1.0,
      "band_coverage_rate": 1.0
    }
  ]
}
```

---

### 15. `POST /demo/replay/{case}/reset` — Reset Replay

Resets digital twin and replay state strictly back to baseline Visit 0, recomputing the pre-visit prediction for Visit 1.

---

## Key Schemas

### `TwinFull`
```typescript
{
  twin_id: string
  is_synthetic: boolean              // MUST be shown as "Demo data" badge in UI if true
  persona_label?: string
  demographics: { age, sex, education_years, apoe4_count }
  modalities: {
    basic_scores: { present, latest }
    biomarkers:   { present, latest }
    mri_volumes:  { present, latest }
    mri_image:    { present, latest }
    ocr_report:   { present, latest }
  }
  visits: VisitRecord[]
  state: TwinState
  history: HistoryEvent[]
}
```

### `TwinState`
```typescript
{
  stage_probs: { CN: float, MCI: float, AD: float }  // sum=1, calibrated
  stage: "CN" | "MCI" | "AD"
  confidence: float                                   // max(stage_probs)
  tier_used: "T1a" | "T1b" | "T2a" | "T2b" | "T3" | "T4" // which feature tier fired
  forecast: { "6": ForecastPoint, "12": ForecastPoint, "24": ForecastPoint }
  conversion_risk: { "12m": float, "24m": float }
  rate_of_decline: { MMSE_per_year: float, vs_cohort_percentile: int }
  shap_drivers: [{ feature, value, shap }, ...]       // top-5 SHAP drivers
}
```

### `StageProbs`
```typescript
{ CN: float, MCI: float, AD: float }  // always sum to 1.0, range [0, 1]
```

---

## Design Constraints for Frontend

1. **`is_synthetic: true`** → always show a visible "Demo data" badge. Never mix synthetic metrics with real-subject metrics.
2. **`mri_image` modality** → always show the `fusion_note: "demo-only, not validated on ADNI"` label near the stage estimate.
3. **`simulate` response** → always show the `disclaimer` field verbatim below the chart.
4. **`tier_used`** → show which tier was used in the confidence chip (e.g. "T1b — Demographics + MMSE", "T2b — Comprehensive Neuropsych", "T3 — CSF Biomarkers", "T4 — FreeSurfer MRI").
5. **Forecast bands** → `MMSE_lo`/`MMSE_hi` are 5th–95th percentile bands from cluster-bootstrap.
6. **ResNet accuracy** → never display "99.7%" in the UI for any non-Kaggle context.
