# Verification Report

**7 of 8 experiments verified against real data; Experiment 8 (Missing Modality Robustness) could not be validated due to a cross-cohort feature mismatch and has been excluded from paper claims — see Section [8] for detail and recommended paper-text changes.**

## Status of the 8 ML4H Experiments
- **Experiment 1 (Modality Ablation):** Regenerated from scratch.
- **Experiment 2 (Fusion Formula):** Regenerated from scratch.
- **Experiment 3 (Cross Validation):** Confirmed valid (synthetic data was limited to notebook fallback).
- **Experiment 4 (Case Studies):** Confirmed valid (real data).
- **Experiment 5 (SHAP Analysis):** Fixed. Re-run using real held-out clinical data for the SHAP baseline instead of random noise.
- **Experiment 6 (Bootstrap CI):** Confirmed valid.
- **Experiment 7 (OCR Evaluation):** Fixed. Recomputed based on actual `07_ocr_evaluation.csv`.
- **Experiment 8 (Missing Modality Robustness):** **SYNTHETIC-TAINTED**. Not salvageable — 32 of 33 clinical features required by catboost_alzheimer_model.pkl are absent from the merged ADNI/ADSP cohort used in this experiment. No numeric result from this experiment (neither the original 33.2%/98.6%/99.1% nor the recomputed 84.2%/94.4%/97.6%) should be cited anywhere in the paper. Recommend qualitative treatment as a limitation.
