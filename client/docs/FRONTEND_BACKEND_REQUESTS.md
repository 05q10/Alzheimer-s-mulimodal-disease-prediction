# Frontend <-> Backend API Integration Status

> Documenting live endpoints vs mocks for the Alzheimer's Digital Twin project.
> Frontend communicates via `NEXT_PUBLIC_API_URL` (default: `http://127.0.0.1:8000` / `http://localhost:8001`).

---

## 1. Live vs Mock Status

| Route & Method | Status in Frontend | Backend Status | Schema / Details |
| :--- | :--- | :--- | :--- |
| `POST /twin` | **Live with dual schema compatibility** | Live (stager tier T1-T4) | Accepts both flat fields (`age`, `sex`, `MMSE`, etc.) and new nested payload (`demographics`, `scores`, `biomarkers`, `mri_volumes`, `lifestyle`, `visits`, `baseline_date`). Returns `TwinFull`. |
| `GET /twin/{twin_id}` | **Live** | Live | Returns full `TwinFull` object. Fallback generator provided if ID is not yet in memory. |
| `POST /twin/{twin_id}/visit` | **Live** | Live | Takes `{ month, exam_date, MMSE, CDGLOBAL, TOTSCORE, FAQ?, MOCA? }`. Appends visit, updates trajectory and returns `{ twin_id, state, prob_diff }`. |
| `POST /twin/{twin_id}/visit/full` | **Mocked in frontend with live fallback** | In parallel development | Accepts `multipart/form-data` containing all cognitive scores (MMSE, CDR, ADAS, FAQ, MOCA, symptoms), biomarkers, MRI volumes, lifestyle, medical conditions, and optional `mri_file` and `blood_report_file`. Returns updated `TwinFull` with new visit appended. |
| `GET /twin/{id}/visit/{n}/mri` | **Mocked in frontend with live fallback** | In parallel development | Returns MRI slice image/thumbnail for visit index `n`. Frontend falls back to cached slice or gradient placeholder. |
| `GET /twin/{id}/export` | **Mocked in frontend with live fallback** | In parallel development | Returns a `.zip` archive containing the twin's complete JSON record, visit histories, MRI scans, and model explanations. Frontend generates a complete JSON download bundle if route returns 404. |
| `GET /demo/cases` | **Mocked in frontend with live fallback** | In parallel development | Returns the 3 sequential benchmark trajectories (Case 1: MCI->AD, Case 2: CN->MCI, Case 3: AD lifestyle improvement) with expected checklists and multi-visit clinical records. |
| `POST /twin/{twin_id}/mri-image` | **Mocked in frontend with live fallback** | In parallel development | Accepts multipart file. Returns `{ mri: { predicted_class, probs, mapped_probs }, state: { ...state, fusion: { adni_probs, mri_probs, weight, final_probs } } }`. |
| `GET /twin/{id}/explain` | **Mocked in frontend with live fallback** | In parallel development | Returns `{ stage, forecast, lifestyle_modifier, symptom_screen, reliability, narrative }`. |
| `GET /twin/{id}/input-usage` | **Mocked in frontend with live fallback** | In parallel development | Returns comprehensive table of all entered fields with status and clinical reasoning. |
| `POST /twin/{id}/scenario` | **Mocked in frontend with live fallback** | In parallel development | Accepts `horizon_months`, `hypothetical_visit`, `biomarkers`, `mri_volumes`, `lifestyle`, `decline_multiplier`. Returns `{ method: string[], drivers_delta, scope_warnings, forecast_reliability, baseline, scenario, delta, headline_metrics, disclaimer }`. Deltas include explicit `direction: "worse" \| "better" \| "same"`. Dry run, saves nothing. |
| `DELETE /twin/{id}/visit/last` | **Live / Fallback** | Live / In parallel | Undoes the most recently recorded visit and recalibrates state. |
| `POST /twin/{id}/reset` | **Live / Fallback** | Live / In parallel | Reverts twin to baseline visit 0 state, logs a reset event with `recorded_at`, and recalibrates. |
| `GET /model-card` | **Live** | Live | Returns model stager/progression metrics, data provenance, and scope envelope. |
| `POST /lstm-predict` | **Mocked in frontend with live fallback** | Currently 503 (model unavailable) | Takes `{ visits: [{ visit_month, entry_age, CDGLOBAL, MMSCORE, TOTSCORE }] }`. Returns `{ predicted_class, predicted_index, probabilities: { CN, MCI, AD }, risk_level }`. Frontend falls back to clinical trajectory rule-based prediction if offline. |
| `GET /demo/patients` | **Live** | Live | Returns prebuilt ADNI demo personas. |

---

## 2. Agreed Schemas for New Routes

### `POST /twin/{id}/visit/full` (Multipart)
Form fields:
- `month`: integer (e.g. 12)
- `exam_date`: string (YYYY-MM-DD)
- `MMSE`, `CDGLOBAL`, `TOTSCORE`, `FAQ`, `MOCA`, `functionalAssessment`, `adl`: numbers
- `symptoms`: JSON string of `{ memoryComplaints: boolean, ... }`
- `biomarkers`: JSON string of `{ ABETA, TAU, PTAU, APVOLUME }`
- `mri_volumes`: JSON string of `{ hippocampus, entorhinal, ventricles, whole_brain, icv }`
- `lifestyle`: JSON string of `{ alcohol, sleepQuality, physicalActivity, dietQuality, smoking }`
- `medical`: JSON string of `{ hypertension, diabetes, depression, headInjury, cardiovascularDisease }`
- `mri_file`: file (optional)
- `blood_report_file`: file (optional)

Returns: `TwinFull` object with the visit appended and new state computed.

### `POST /twin/{id}/scenario`
Request body:
```json
{
  "horizon_months": 12,
  "hypothetical_visit": {
    "MMSE": 22,
    "CDGLOBAL": 1.0,
    "TOTSCORE": 24
  },
  "lifestyle": {
    "alcohol": 4,
    "physicalActivity": 2,
    "sleepQuality": 5,
    "dietQuality": 4,
    "smoking": 1
  },
  "decline_multiplier": 1.2
}
```

Response body:
```json
{
  "twin_id": "twin-123",
  "horizon_months": 12,
  "method": ["ADNI Progression Stager", "Multi-modal Risk Estimator"],
  "drivers_delta": [
    {
      "factor": "MMSE",
      "label": "MMSE Drop (-4.0 pts)",
      "delta": "-4.0 pts",
      "impact": "Increases progression probability to AD",
      "direction": "worse"
    }
  ],
  "baseline": {
    "stage_probs": { "CN": 0.14, "MCI": 0.62, "AD": 0.24 },
    "conversion_risk": { "12m": 35, "24m": 58 },
    "p_ad": 24,
    "progression_risk": 48
  },
  "scenario": {
    "stage_probs": { "CN": 0.05, "MCI": 0.45, "AD": 0.50 },
    "conversion_risk": { "12m": 58, "24m": 78 },
    "p_ad": 50,
    "progression_risk": 72
  },
  "delta": {
    "p_ad": { "value": 26, "direction": "worse", "pct_str": "+26%", "sign_str": "+" },
    "progression_risk": { "value": 24, "direction": "worse", "pct_str": "+24%", "sign_str": "+" },
    "direction": "worse"
  },
  "headline_metrics": {
    "p_ad": {
      "metric_name": "P(AD at horizon)",
      "baseline_value": 24,
      "scenario_value": 50,
      "delta_value": 26,
      "direction": "worse",
      "formatted_baseline": "24%",
      "formatted_scenario": "50%",
      "formatted_delta": "+26%"
    },
    "progression_risk": {
      "metric_name": "progression risk",
      "baseline_value": 48,
      "scenario_value": 72,
      "delta_value": 24,
      "direction": "worse",
      "formatted_baseline": "48%",
      "formatted_scenario": "72%",
      "formatted_delta": "+24%"
    }
  },
  "forecast_reliability": "high",
  "scope_warnings": [],
  "disclaimer": "Scenario projections are research models and not clinical diagnostic guarantees."
}
```

### `GET /demo/cases`
Returns an array of 3 benchmark cases:
- Case 1: `case-1` Margaret Sterling (MCI -> AD progression)
- Case 2: `case-2` Arthur Pendelton (CN -> MCI conversion)
- Case 3: `case-3` Harold Finch (AD with lifestyle improvement)
Each case includes `story`, `expected_checklist`, and full longitudinal `visits` data.
