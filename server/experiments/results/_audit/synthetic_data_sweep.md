# Task 0: Synthetic/Fallback Data Sweep

This report documents all occurrences of synthetic, fallback, dummy, mock, and `np.random` markers across the 8 experiments to determine their true data provenance.

| Experiment | File | Line | Context | Judgment |
| :--- | :--- | :--- | :--- | :--- |
| Exp 1 (Modality Ablation) | `exp01_modality_ablation.py` | 79 | `merged = pd.get_dummies(merged, columns=[col], dummy_na=False)` | **No synthetic data markers found** |
| Exp 2 (Fusion Formula) | `exp02_fusion_formula.py` | - | None | **No synthetic data markers found** |
| Exp 3 (Cross Validation) | `notebooks/03_cross_validation.ipynb` | 51, 62 | `df_clin = pd.DataFrame(np.random.randn(n_samples, len(scaled_cols)), columns=scaled_cols)` | **(ii) only a small fallback path was fake.** The Python script `exp03_cross_validation.py` loads real data `alzheimers_disease_data.csv`. The notebook was merely a demo. |
| Exp 4 (Case Studies) | `exp04_case_studies.py` | 9 | `# # Experiment 4 — Synthetic End-to-End Digital Twin Case Studies` | **(iii) harmless use.** The word "Synthetic" is just in the title; real patient records are sampled from `merged`. |
| Exp 5 (SHAP Analysis) | `exp05_shap_analysis.py` | 29 | `df_clin = pd.DataFrame(np.random.randn(200, len(cb_cols)), columns=cb_cols)` | **(i/ii) SHAP explainer sample was synthetic.** The underlying CatBoost model was loaded from disk, but the 200 samples passed to the SHAP explainer for the clinical model were generated using `np.random.randn`. The biomarker model SHAP used real data. |
| Exp 6 (Bootstrap CI) | `exp06_bootstrap_ci.py` | 74, 141 | `np.random.seed(42)` | **(iii) harmless use.** Used purely to seed the bootstrap resampling over real test data. |
| Exp 7 (OCR Evaluation) | `exp07_ocr_evaluation.py` | 14, 28, 37 | `# We evaluate the extraction logic on synthetic... "id": "synthetic_2.png"` | **(ii) synthetic hardcoded examples used.** The script uses hardcoded dictionaries to represent "synthetic" OCR results to demonstrate the pipeline. (Task 3 dictates we treat the existing CSV as the truth to process). |
| Exp 8 (Missing Modality) | `exp08_missing_modality_robustness.py` | 70-72 | `merged['Clinical_Risk'] = np.random.uniform(0, 1, size=len(merged))` | **(ii) small fallback path was fake.** Clinical_Risk was simulated with uniform noise to test degradation. |

## Conclusion for Task 2 Scope
For Experiment 5, the underlying models (`catboost_alzheimer_model.pkl` and `xgb_model.joblib`) were pre-trained and loaded from disk, not trained in this script on synthetic data. However, the data (`df_clin`) passed into the `shap.TreeExplainer` to compute the actual SHAP values for the clinical model was completely randomly generated. We will need to fix `exp05_shap_analysis.py` to sample its SHAP background from real held-out patient data, rather than retraining the model.
