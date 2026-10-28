# Alzheimer's Digital Twin — Demo & Execution Guide

End-to-end multimodal digital twin for longitudinal Alzheimer's staging and progression forecasting.

---

## 1. Quickstart & Service Run Commands

### Backend (FastAPI Server)
```bash
cd server
# Ensure dependencies are installed
pip install fastapi uvicorn catboost pandas numpy pydantic scikit-learn pillow

# Launch API server on http://localhost:8000
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI documentation available at:
👉 **`http://localhost:8000/docs`**

---

## 2. Running Verification & Integration Tests

Run the full pytest suite to verify anti-leakage compliance, patient-builder flow, what-if scenario dry run, visit undo/reset, and legacy endpoints:

```bash
cd server
python -m pytest tests/ -v
```

Expected output:
- `tests/test_no_leakage.py`: 10/10 passed (zero leakage, temporal integrity)
- `tests/test_twin_api.py`: 7/7 passed (patient-builder, scenario, undo/reset, model card scope)
- `tests/test_part1_part2.py`: 3/3 passed (MRI upload, tempered ResNet softmax, reset/undo snapshot, real dates)
- `tests/test_plausibility.py`: 7/7 passed (archetypes CN, MCI, Mild AD, Moderate AD, monotonicity, scenario multipliers)
- `tests/test_risk_modifier.py`: 4/4 passed (bounds [0.85, 1.35], monotonicity, stage independence, scenario override)
- `tests/test_explain.py`: 5/5 passed (43 intake fields lineage, code-path audit, missing modality tolerance, narrative)
- `tests/test_legacy_routes.py`: 5/5 passed (/alele, /image, /extract-report, /base, /lstm-predict)

Total: **40 passed, 1 skipped (stub placeholder), 0 failed** across 41 tests.
All metrics synchronized from `server/final_metrics.json`.

---

## 3. Clinical Archetype Results (Derived from Training Envelope)

| Archetype | Key Inputs | Predicted Stage | P(CN) | P(MCI) | P(AD) | 12m MMSE Forecast | 12m Envelope Bounds | Reliability | Scope Warnings |
|---|---|---|---|---|---|---|---|---|---|
| **Cognitively Normal (CN)** | Age 72, MMSE 29, CDR 0, ADAS 5, FAQ 0 | **CN** | **0.933** | 0.053 | 0.014 | **28.5** | [26.1, 30.0] | `ok` | None |
| **Mild Cognitive Impairment (MCI)** | Age 72, MMSE 27, CDR 0.5, ADAS 10, FAQ 3 | **MCI** | 0.025 | **0.933** | 0.041 | **25.8** | [23.4, 28.3] | `ok` | None |
| **Mild AD** | Age 72, MMSE 20, CDR 1.0, ADAS 25, FAQ 15 | **AD** | 0.011 | 0.030 | **0.959** | **13.9** (24m: 13.1) | $\le \text{curr} + 1$ | `ok` | None |
| **Moderate AD** | Age 72, MMSE 16, CDR 1.0, ADAS 35, FAQ 18 | **AD** | 0.029 | 0.133 | **0.838** | **12.3** (6m: 15.7, 24m: 10.5) | $\le \text{curr} + 1$ | `low` | MMSE & TOTSCORE outside 5th–95th percentile training range |

---

## 4. Live API Walkthrough (cURL Examples)

### A. Create Twin via Patient-Builder Flow
```bash
curl -X POST "http://localhost:8000/twin" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Jane Doe",
    "demographics": {
      "age": 73.0,
      "sex": 1,
      "education_years": 16.0,
      "apoe4": 1
    },
    "scores": {
      "MMSE": 26.0,
      "CDGLOBAL": 0.5,
      "TOTSCORE": 16.0
    },
    "biomarkers": {
      "ABETA": 720.0,
      "TAU": 390.0,
      "PTAU": 38.0,
      "APVOLUME": 3200.0
    },
    "lifestyle": {
      "BMI": 26.5,
      "PhysicalActivity": 3,
      "DietQuality": 4,
      "SleepQuality": 3
    }
  }'
```
*Note*:
- Runs calibrated CatBoost stager at Tier T3 (Demographics + Neuropsych + CSF).
- Evaluates separate Kaggle CatBoost lifestyle risk model (`kaggle_clinical` modality chip).
- Evaluates Allele ensemble model (`allele_ensemble` modality chip).
- `APVOLUME` is never mapped to FreeSurfer hippocampus.

---

### B. Add Longitudinal Follow-up Visit
```bash
curl -X POST "http://localhost:8000/twin/<TWIN_ID>/visit" \
  -H "Content-Type: application/json" \
  -d '{
    "month": 12,
    "MMSE": 24.0,
    "CDGLOBAL": 0.5,
    "TOTSCORE": 18.5,
    "source": "user"
  }'
```

---

### C. Undo Last Visit
```bash
curl -X DELETE "http://localhost:8000/twin/<TWIN_ID>/visit/last"
```

---

### D. Reset Twin to Month-0 Baseline State
```bash
curl -X POST "http://localhost:8000/twin/<TWIN_ID>/reset"
```

---

### E. Run Counterfactual What-If Scenario (Dry Run)
```bash
curl -X POST "http://localhost:8000/twin/<TWIN_ID>/scenario" \
  -H "Content-Type: application/json" \
  -d '{
    "horizon_months": 12,
    "hypothetical_visit": {
      "MMSE": 22.0,
      "CDGLOBAL": 1.0
    },
    "decline_multiplier": 1.2
  }'
```
*Note*: Persists nothing. Returns baseline, scenario projection, deltas, and automated scope warnings for clinical inconsistency or implausible jumps.

---

### F. Query Progression Forecast (6m / 12m / 24m)
```bash
curl -X GET "http://localhost:8000/twin/<TWIN_ID>/forecast?horizon=24"
```

---

### G. Input Usage Matrix & Clinical Lineage
```bash
curl -X GET "http://localhost:8000/twin/<TWIN_ID>/input-usage"
```
Returns 43 intake fields with exact model lineage (`used_in_stage`, `used_in_forecast`, `used_in_lifestyle_modifier`, `used_in_symptom_screen`, `used_in_biomarker_ensemble`, `used_in_mri_fusion`, `recorded_only`, `missing`, `assumed_default`).

---

### H. Model Explainability & Narrative
```bash
curl -X GET "http://localhost:8000/twin/<TWIN_ID>/explain"
```
Returns CatBoost TreeSHAP stage drivers for predicted class, modality leave-one-out delta shifts, progression drivers, lifestyle modifier breakdown, Kaggle symptom screener, and 4-to-8 sentence clinical narrative.

---

### I. Structural MRI Slice Upload & Linear Fusion
```bash
curl -X POST "http://localhost:8000/twin/<TWIN_ID>/mri-image" \
  -F "file=@server/ml/image_model/test.jpg"
```
Runs temperature-scaled FastAI ResNet34 ($T=2.0$) and applies linear late fusion:
$$\text{final} = 0.8 \times P_{\text{adni}} + 0.2 \times P_{\text{mri\_mapped}}$$

---

### J. Model Card & Scope
```bash
curl -X GET "http://localhost:8000/model-card"
```
Returns held-out test metrics, valid clinical bounds (`age_range: [60.7, 84.7]`, `horizons_months: [6, 12, 24]`), training cohort specifications (`ADNI, 2123 subjects`), and ethical limitations. All numbers sourced from `server/final_metrics.json`.
