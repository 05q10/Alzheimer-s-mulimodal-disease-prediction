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

#### 4c. `POST /twin/{twin_id}/modality/{modality_name}` — Generic Modality Router
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

**Dry run: persists nothing.** Validates `horizon_months` (rejects anything other than 6, 12, 24 with HTTP 422).

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
  "disclaimer": "Hypothetical what-if scenario. Projections represent counterfactual simulations..."
}
```

---

### 7. `DELETE /twin/{twin_id}/visit/last` — Undo Last Visit
Removes the most recent follow-up visit and recalculates the prior clinical stage and trajectory.

---

### 8. `POST /twin/{twin_id}/reset` — Reset to Baseline
Resets the twin to month-0 baseline state, removing all subsequent visits.

---

### 9. `POST /twin/{twin_id}/simulate` — What-If Simulation
Decline scaling (`decline_multiplier`: 0.5–1.5). Carries mandatory illustrative disclaimer.

---

### 10. `GET /demo/patients` — List Demo Personas
Returns 6 pre-built demo patient entries (real ADNI + synthetic flagged).

---

### 11. `GET /model-card` — Model Metrics, Scope & Disclaimers

Returns held-out test metrics, valid clinical bounds and scopes, limitations list, and data provenance.
```json
{
  "stager": { "T1_accuracy": 0.81, ... },
  "progression": { "MMSE_MAE_12m": 1.465, ... },
  "scope": {
    "age_range": [60.7, 84.7],
    "horizons_months": [6, 12, 24],
    "score_ranges": {
      "MMSE": [0, 30],
      "CDGLOBAL": [0, 3],
      "TOTSCORE": [0, 70]
    },
    "training_cohort": "ADNI, 2123 subjects"
  },
  "limitations": [ ... ],
  "resnet_disclaimer": "...",
  "fusion_disclaimer": "..."
}
```

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
