Figure Manifest for research_paper/figures/
=============================================
All figures must be generated as both vector PDF and 300 dpi PNG.
Filenames must NEVER contain spaces.
Color palette must be color-blind safe (e.g. ColorBrewer, Seaborn colorblind, or viridis).
Font sizes must be >= 9 pt.

1. fig_architecture (.pdf, .png)
   - Title: System Architecture: Multimodal Digital Twin Pipeline
   - Data Source: System implementation and service routing in server/app/twin.py
   - Description: Intake -> Calibrated Stager (T1-T4) -> Multi-horizon Forecaster -> Counterfactual Scenario Lab -> Explainability & Plausibility -> Per-Visit Profiles. Includes ResNet34 MRI slice late fusion (w=0.20) and Lancet Commission lifestyle risk modifier.
   - Script: research_paper/scripts/make_figures.py (generate_architecture)

2. fig_data_pipeline (.pdf, .png)
   - Title: Longitudinal Data Pipeline and Leakage Prevention
   - Data Source: server/experiments/results/09_data_audit.md, server/data/splits.json, server/ml/twin/data_build.py
   - Description: Merging of ADNIMERGE, ADSP-PHC FreeSurfer, APOE tables, My_Table; backward-only joins; forbidden features blacklist; subject-level stratified 70/15/15 split.
   - Script: research_paper/scripts/make_figures.py (generate_data_pipeline)

3. fig_tier_ablation (.pdf, .png)
   - Title: Staging Tier Ablation on ADNI Held-Out Test Set (N=3,022 visits)
   - Data Source: server/experiments/results/exp09_stager_results.md
   - Description: Accuracy and Macro-F1 across Tiers T1, T1b, T2a, T2b, T3, T4, and T2b+MRI with 95% cluster-bootstrap confidence intervals.
   - Script: research_paper/scripts/make_figures.py (generate_tier_ablation)

4. fig_stager_cm_t1b (.pdf, .png)
   - Title: Tier T1b (Demographics + MMSE) Stager Confusion Matrix
   - Data Source: server/data/splits.json (test set, 456 subjects) + server/ml/twin/weights/stager_model.cbm
   - Description: 3x3 confusion matrix with integer counts and row-normalized percentages for classes CN, MCI, AD.
   - Script: research_paper/scripts/make_figures.py (generate_confusion_matrices)

5. fig_stager_cm_t2b (.pdf, .png)
   - Title: Tier T2b (Full Neuropsychological Battery) Stager Confusion Matrix
   - Data Source: server/data/splits.json (test set, 456 subjects) + server/ml/twin/weights/stager_model.cbm
   - Description: 3x3 confusion matrix with integer counts and row-normalized percentages for classes CN, MCI, AD.
   - Script: research_paper/scripts/make_figures.py (generate_confusion_matrices)

6. fig_stager_cm_t4 (.pdf, .png)
   - Title: Tier T4 (Full Multimodal + FreeSurfer MRI) Stager Confusion Matrix
   - Data Source: server/data/splits.json (test set, 456 subjects) + server/ml/twin/weights/stager_model.cbm
   - Description: 3x3 confusion matrix with integer counts and row-normalized percentages for classes CN, MCI, AD.
   - Script: research_paper/scripts/make_figures.py (generate_confusion_matrices)

7. fig_calibration (.pdf, .png)
   - Title: Probabilistic Calibration: Stager Platt Scaling and Conversion Risk
   - Data Source: server/experiments/results/exp09_stager_results.md, server/experiments/results/exp10_progression_results.md
   - Description: Two-panel reliability diagram: (a) Stager calibration across CN, MCI, AD with ECE; (b) Conversion hazard calibration at 6m, 12m, 24m with Brier skill scores.
   - Script: research_paper/scripts/make_figures.py (generate_calibration)

8. fig_forecast_mae (.pdf, .png)
   - Title: Longitudinal Forecasting Error vs. Standard Baselines
   - Data Source: server/experiments/results/exp10_progression_results.md
   - Description: Mean Absolute Error (MAE) with 95% cluster-CIs across 6m, 12m, 24m for MMSE, CDGLOBAL, and TOTSCORE (ADAS11) comparing Model B vs. Persistence vs. Slope Extrapolation.
   - Script: research_paper/scripts/make_figures.py (generate_forecast_mae)

9. fig_conversion_roc (.pdf, .png)
   - Title: Dementia Conversion Risk Discrimination (AUROC)
   - Data Source: server/experiments/results/exp10_progression_results.md
   - Description: ROC curves for dementia conversion at 12m (AUROC 0.7615) and 24m (AUROC 0.8111) comparing CatBoost progression model against the Stage+MMSE baseline.
   - Script: research_paper/scripts/make_figures.py (generate_conversion_roc)

10. fig_forest_paired_differences (.pdf, .png)
    - Title: Forest Plot of Paired Modality Differences with 95% Cluster-Bootstrap CIs
    - Data Source: server/experiments/results/exp09_stager_results.md, server/experiments/results/exp10_progression_results.md, server/final_metrics.json
    - Description: Effect sizes and 95% CIs for CSF staging increment, FreeSurfer staging increment, Masked vs Unmasked GRU, FreeSurfer progression increment, and Vitals inclusion, with zero null-effect reference line.
    - Script: research_paper/scripts/make_figures.py (generate_forest_plot)

11. fig_shap_importance (.pdf, .png)
    - Title: TreeSHAP Feature Attributions Grouped by Clinical Modality
    - Data Source: server/experiments/results/05_shap_top_features.json, server/experiments/results/09_data_audit.md
    - Description: Modality-stratified mean absolute SHAP attributions (Cognitive, Functional, Biomarker, FreeSurfer MRI, Demographics).
    - Script: research_paper/scripts/make_figures.py (generate_shap_importance)

12. fig_demo_trajectories (.pdf, .png)
    - Title: Longitudinal Demo Personas and Counterfactual Scenario Lab Simulation
    - Data Source: server/app/demo_patients.py, server/tests/test_scenario_direction.py
    - Description: Multi-panel plot: Case 1 (Eleanor Vance - Stable CN), Case 2 (Arthur Pendelton - CN->MCI converter), Case 3 (Margaret Sterling - Rapid MCI->AD converter), and Case 3 Scenario Lab comparison (Baseline vs Improved Lifestyle Intervention with 90% confidence band and multiplier annotations).
    - Script: research_paper/scripts/make_figures.py (generate_demo_trajectories)

13. fig_archetype_envelope (.pdf, .png)
    - Title: Clinical Archetype Forecasts and Empirical Plausibility Envelope
    - Data Source: server/final_metrics.json, server/ml/twin/weights/plausibility_envelope.json
    - Description: 24-month cognitive trajectories for CN, MCI, Mild AD, and Moderate AD plotted against the 5th-95th percentile training envelope, highlighting automated out-of-distribution scope warnings.
    - Script: research_paper/scripts/make_figures.py (generate_archetype_envelope)

14. fig_legacy_cm (.pdf, .png) [Appendix Only]
    - Title: Earlier-Phase Legacy Model Confusion Matrices (Phase 1 Baseline)
    - Data Source: research_paper/template/synopsis.tex (Biomarker ensemble: 394/5/202, 7/261/99, 224/68/456), Kaggle MRI ResNet34 slice matrix
    - Description: Legacy confusion matrices labeled "earlier phase, different data, not ADNI-validated".
    - Script: research_paper/scripts/make_figures.py (generate_legacy_cm)

15. fig_ui_dashboard (.png, .pdf) [Placeholder / User Capture]
    - Title: Patient Intake & Multi-Tier Staging Dashboard
    - Description: Interactive Next.js interface showing patient builder, active modality chips, and Platt-calibrated stage probabilities. Captured as per SCREENSHOTS.md.

16. fig_ui_why (.png, .pdf) [Placeholder / User Capture]
    - Title: Explainability Panel: TreeSHAP Drivers & Modality Ablation
    - Description: Next.js UI component rendering TreeSHAP feature contributions, leave-one-modality-out marginal shifts, and clinical narrative. Captured as per SCREENSHOTS.md.

17. fig_ui_scenario (.png, .pdf) [Placeholder / User Capture]
    - Title: Interactive Counterfactual Scenario Lab & Plausibility Guard
    - Description: Next.js UI showing what-if decline rate slider, lifestyle adjustment toggles, projected trajectory comparison, and envelope violation alerts. Captured as per SCREENSHOTS.md.

