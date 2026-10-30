# FIGURE_SPECS.md: Comprehensive Inventory of Publication Figures

This document specifies every figure used in the research paper, including its file naming, dimensions, data provenance, plot architecture, axis definitions, and key clinical takeaway. All figures are generated programmatically via `research_paper/scripts/make_new_figures.py` and `research_paper/scripts/make_figures.py` using vector PDF and 300 dpi PNG formats.

---

## 1. fig_pipeline_flow
- **Filename**: Drawn natively via TikZ in `main.tex` (no external raster file required; fallback handled by `\figph`).
- **Final Size**: Full width (6.5 in $\times$ 4.5 in).
- **Data Source**: System architecture of the Multimodal Digital Twin (`server/app/twin/`, `API_CONTRACT.md`).
- **Plot Type**: Directed hierarchical flowchart.
- **Axes / Nodes**: Intake $\to$ Scope & Validation $\to$ Multi-Tier Stager (Model A) $\to$ Auxiliary Multimodal Modules (ResNet-34 2D MRI, Vision-LLaMA OCR, Biomarker Ensemble, Symptom Screener) $\to$ Longitudinal Consultation Profile $\to$ Progression Forecasting (Model B) $\to$ Conversion Hazard Engine $\to$ Sequence Validation $\to$ Counterfactual Scenario Lab $\to$ Local Explainability (TreeSHAP & LOO) $\to$ Clinical Report Export.
- **Key Takeaway**: Illustrates the end-to-end clinical workflow, showing how point-in-time cross-sectional staging feeds into multi-horizon forecasting and scenario simulation under empirical missingness.

---

## 2. fig_baseline_characteristics
- **Filename**: `fig_baseline_characteristics.pdf`, `fig_baseline_characteristics.png`
- **Final Size**: Full width (6.5 in $\times$ 4.2 in), 2 rows $\times$ 3 columns.
- **Data Source**: `server/data/unified_visits.parquet` ($N=21,055$ time-axis visits; 8,425 CN, 8,363 MCI, 4,267 AD).
- **Plot Type**: Multi-panel boxplots and bar chart with colorblind-safe palette (Green: CN, Amber: MCI, Red: AD).
- **Axes**:
  - Panel (0,0): Patient Age (Years), [45, 95].
  - Panel (0,1): Education (Years), [5, 25].
  - Panel (0,2): APOE $\varepsilon4$ Carrier Share (%), [0, 75%].
  - Panel (1,0): MMSE Score (0–30), [10, 31].
  - Panel (1,1): CDR Sum of Boxes (0–18), [0, 14].
  - Panel (1,2): ADAS11-Cog Total Score (0–70), [0, 50].
- **Key Takeaway**: Demonstrates progressive cognitive and functional separation across syndromic stages while highlighting substantial overlap between CN and early MCI on bedside screening tests.

---

## 3. fig_diagnosis_transitions
- **Filename**: `fig_diagnosis_transitions.pdf`, `fig_diagnosis_transitions.png`
- **Final Size**: Half width (3.2 in $\times$ 2.8 in).
- **Data Source**: Consecutive consultation transitions from `server/data/unified_visits.parquet` ($N=18,021$ transitions).
- **Plot Type**: Annotated transition matrix heatmap (Blues colormap, row-normalized percentages).
- **Axes**: Y-axis = Current Stage (CN, MCI, AD); X-axis = Next Observed Stage (CN, MCI, AD).
- **Key Takeaway**: Confirms high diagnostic persistence (CN $\to$ CN: 94.5%, MCI $\to$ MCI: 86.7%, AD $\to$ AD: 90.5%) alongside clinical conversion rates (MCI $\to$ AD: 9.3%, CN $\to$ MCI: 4.9%) and diagnostic reversals (MCI $\to$ CN: 4.1%).

---

## 4. fig_missingness_by_modality
- **Filename**: `fig_missingness_by_modality.pdf`, `fig_missingness_by_modality.png`
- **Final Size**: Half width (3.2 in $\times$ 3.6 in).
- **Data Source**: Observational feature completeness across $N=21,055$ time-axis visits.
- **Plot Type**: Horizontal bar chart sorted by observational missingness rate.
- **Axes**: Y-axis = Clinical Modality; X-axis = Missingness Rate (%) [0–100%].
- **Key Takeaway**: Demographics are 100% complete, while invasive lumbar puncture CSF biomarkers ($A\beta_{42}/\tau$: 83.9% missing) and volumetric MRI (47.2% missing) are absent in the vast majority of visits, justifying the need for Model A's multi-tier dynamic routing.

---

## 5. fig_visit_distribution
- **Filename**: `fig_visit_distribution.pdf`, `fig_visit_distribution.png`
- **Final Size**: Full width (6.5 in $\times$ 2.6 in), 2 subplots.
- **Data Source**: Subject longitudinal records in `server/data/unified_visits.parquet` (3,034 subjects).
- **Plot Type**: Dual histogram.
  - Subplot A: Follow-up visits per subject (bins 1–20, median dashed line at 6 visits).
  - Subplot B: Follow-up interval duration (months) between consecutive visits (prominent peaks at 6 and 12 months).
- **Axes**: Subplot A: Visits per subject vs. Count; Subplot B: Interval (months) vs. Transition Count.
- **Key Takeaway**: Validates that while patients have a median of 6 consultations, visits follow irregular calendar intervals clustered around 6-month and 12-month clinical checkups.

---

## 6. fig_tier_per_class_recall
- **Filename**: `fig_tier_per_class_recall.pdf`, `fig_tier_per_class_recall.png`
- **Final Size**: Full width (6.5 in $\times$ 2.8 in).
- **Data Source**: Holdout test set ($N=3,022$) confusion matrices from `server/experiments/results/exp09_stager_results.md`.
- **Plot Type**: Grouped bar chart comparing CN, MCI, and AD recall across tiers T1b, T2b, and T4.
- **Axes**: X-axis = Diagnostic Tier; Y-axis = Class Recall (%) [0–115%].
- **Key Takeaway**: Reveals that bedside screening (Tier T1b) achieves 100% CN recall but collapses on MCI (0.0%) and AD (0.0%), whereas Tier T2b (adding CDR and neuropsych batteries) restores balanced sensitivity (88.4% CN, 69.7% MCI, 78.1% AD), which is further refined by Tier T4 volumetric MRI (81.4% AD).

---

## 7. fig_forecast_by_stage
- **Filename**: `fig_forecast_by_stage.pdf`, `fig_forecast_by_stage.png`
- **Final Size**: Full width (6.5 in $\times$ 2.8 in).
- **Data Source**: Progression Model B test set evaluation from `server/experiments/results/exp10_progression_results.md`.
- **Plot Type**: Grouped bar chart of MMSE Mean Absolute Error across 6m, 12m, and 24m horizons broken down by baseline anchor stage.
- **Axes**: X-axis = Forecast Horizon; Y-axis = MMSE MAE (Points) [0–3.8].
- **Key Takeaway**: Patients who are Cognitively Normal at baseline exhibit lower trajectory forecast errors (MAE: 0.72 at 6m, 1.34 at 24m) than baseline AD patients (1.45 at 6m, 3.18 at 24m), who experience greater biological variance.

---

## 8. fig_forecast_by_nvisits
- **Filename**: `fig_forecast_by_nvisits.pdf`, `fig_forecast_by_nvisits.png`
- **Final Size**: Half width (3.2 in $\times$ 2.8 in).
- **Data Source**: Test split longitudinal evaluation partitioned by prior visit history count.
- **Plot Type**: Multi-line error curve across horizons (6m, 12m, 24m) as a function of 1, 2, or 3+ prior visits.
- **Axes**: X-axis = Prior Consultations; Y-axis = MMSE MAE (Points) [0.5–2.8].
- **Key Takeaway**: Demonstrates that accumulated consultation history steadily reduces trajectory error (12m MAE drops from 1.58 at 1 visit to 1.36 with 3+ visits).

---

## 9. fig_pred_vs_actual_mmse12
- **Filename**: `fig_pred_vs_actual_mmse12.pdf`, `fig_pred_vs_actual_mmse12.png`
- **Final Size**: Half width (3.2 in $\times$ 3.0 in).
- **Data Source**: Model B 12-month MMSE regression on holdout test set ($N=1,716$).
- **Plot Type**: Scatter plot with transparency and identity line ($y = x$).
- **Axes**: X-axis = Actual Observed MMSE at Month 12; Y-axis = Model B Predicted MMSE. Range [12, 30.5].
- **Key Takeaway**: Confirms strong calibration along the diagonal ($R^2 = 0.74$, $\text{MAE} = 1.58$ points) without systematic under- or over-estimation across the cognitive spectrum.

---

## 10. fig_residual_hist_mmse12
- **Filename**: `fig_residual_hist_mmse12.pdf`, `fig_residual_hist_mmse12.png`
- **Final Size**: Half width (3.2 in $\times$ 2.8 in).
- **Data Source**: Empirical prediction residuals $(y - \hat{y})$ from Model B Month 12 MMSE test predictions.
- **Plot Type**: Histogram with Kernel Density Estimate and 5th/95th percentile dashed reference bounds.
- **Axes**: X-axis = Prediction Residual (Points) [-6, +6]; Y-axis = Probability Density.
- **Key Takeaway**: Residuals are tightly centered at zero with symmetric 90% empirical error bounds extending $\pm 2.4$ points.

---

## 11. fig_conversion_pr
- **Filename**: `fig_conversion_pr.pdf`, `fig_conversion_pr.png`
- **Final Size**: Half width (3.2 in $\times$ 2.8 in).
- **Data Source**: Progression hazard classifier evaluation on test split.
- **Plot Type**: Precision-Recall curves for 12-month and 24-month cognitive progression vs. cohort prevalence baseline.
- **Axes**: X-axis = Recall (Sensitivity) [0, 1.0]; Y-axis = Precision (Positive Predictive Value) [0, 1.05].
- **Key Takeaway**: Model achieves PR-AUC of 0.68 at 12 months and 0.76 at 24 months, substantially outperforming the baseline disease conversion prevalence (17.5%).

---

## 12. fig_risk_quintiles
- **Filename**: `fig_risk_quintiles.pdf`, `fig_risk_quintiles.png`
- **Final Size**: Half width (3.2 in $\times$ 2.8 in).
- **Data Source**: Test set conversion events grouped by predicted risk score quintiles (Q1 to Q5).
- **Plot Type**: Grouped bar chart comparing 12-month and 24-month observed progression rates.
- **Axes**: X-axis = Predicted Risk Quintile; Y-axis = Observed Event Rate (%) [0, 85%].
- **Key Takeaway**: Confirms strict monotonic risk stratification: lowest quintile (Q1) exhibits only a 2.1% (12m) and 4.8% (24m) progression rate, whereas top quintile (Q5) exhibits 54.2% (12m) and 76.4% (24m) conversion.

---

## 13. fig_seq_model_compare
- **Filename**: `fig_seq_model_compare.pdf`, `fig_seq_model_compare.png`
- **Final Size**: Half width (3.2 in $\times$ 2.8 in).
- **Data Source**: Standardized 846 sequence test windows evaluated across architectures (`server/scratch/run_mask_comparison.py`).
- **Plot Type**: Grouped bar chart comparing Zero-Imputed BiLSTM, Binary-Masked BiLSTM, and CatBoost on Accuracy and Macro-F1.
- **Axes**: X-axis = Model Architecture; Y-axis = Metric Score (%) [75, 88%].
- **Key Takeaway**: Shows that tabular CatBoost (84.6% Acc, 84.3% F1) matches or slightly exceeds Binary-Masked BiLSTM (84.0% Acc, 83.6% F1) without requiring sequence padding or imputation.

---

## 14. fig_seq_cm_variants
- **Filename**: `fig_seq_cm_variants.pdf`, `fig_seq_cm_variants.png`
- **Final Size**: Full width (6.5 in $\times$ 2.8 in), 2 subplots.
- **Data Source**: Confusion matrices from `run_mask_comparison.py` on 846 test sequences.
- **Plot Type**: Dual side-by-side heatmaps.
  - Subplot A: Zero-Imputed BiLSTM (Acc 83.4%).
  - Subplot B: Binary-Masked BiLSTM (Acc 84.0%).
- **Axes**: Y-axis = True Stage; X-axis = Predicted Stage (CN, MCI, AD).
- **Key Takeaway**: Demonstrates that binary masking reduces false MCI transitions by mitigating spurious signal corruption caused by zero-imputed missing values.

---

## 15. fig_seq_training_curves
- **Filename**: `fig_seq_training_curves.pdf`, `fig_seq_training_curves.png`
- **Final Size**: Full width (6.5 in $\times$ 2.7 in), 2 subplots.
- **Data Source**: Training epoch logs parsed from `lstm-gru-masking-no-masking.ipynb` (Cells 28 and 61).
- **Plot Type**: Line plots comparing Unmasked vs. Masked recurrent models across 20 epochs.
  - Subplot A: Training and validation cross-entropy loss.
  - Subplot B: Validation accuracy.
- **Axes**: X-axis = Epoch; Y-axis = Loss / Accuracy (%).
- **Key Takeaway**: Masked LSTM converges faster and maintains superior generalization stability, reaching 82.1% validation accuracy compared to the noisier trajectory of the unmasked model.

---

## 16. fig_scenario_tornado
- **Filename**: `fig_scenario_tornado.pdf`, `fig_scenario_tornado.png`
- **Final Size**: Half width (3.2 in $\times$ 3.4 in).
- **Data Source**: Scenario Lab sensitivity evaluation (`server/ml/twin/risk_modifier.py`).
- **Plot Type**: Horizontal tornado bar chart sorted by absolute effect magnitude.
- **Axes**: Y-axis = Parameter Modification; X-axis = Change in 12m MMSE Forecast ($\Delta$ Points) [-2.0, +2.0].
- **Key Takeaway**: Validates clinical priority: direct cognitive battery adjustments ($\pm 2$ MMSE) exert primary trajectory shifts ($\pm 1.5$ points), whereas lifestyle factor modifications modulate trajectories within literature-bounded bounds ($\pm 0.35$ to $\pm 0.42$ points).

---

## 17. fig_multiplier_sweep
- **Filename**: `fig_multiplier_sweep.pdf`, `fig_multiplier_sweep.png`
- **Final Size**: Half width (3.2 in $\times$ 2.8 in).
- **Data Source**: Lifestyle progression modifier parameter sweep ($M \in [0.85, 1.35]$).
- **Plot Type**: Multi-line longitudinal trajectory plot over 36 months.
- **Axes**: X-axis = Follow-Up Time (Months) [0, 36]; Y-axis = Simulated MMSE Score [19, 27].
- **Key Takeaway**: Illustrates how the lifestyle multiplier modulates the trajectory slope, producing divergent 36-month cognitive outcomes between protective ($0.85\times$) and adverse ($1.35\times$) profiles.

---

## 18. fig_lifestyle_sweep
- **Filename**: `fig_lifestyle_sweep.pdf`, `fig_lifestyle_sweep.png`
- **Final Size**: Full width (6.5 in $\times$ 2.7 in), 2 subplots.
- **Data Source**: 36-month counterfactual simulation comparing optimal vs. adverse lifestyle profiles.
- **Plot Type**: Dual trajectory line plots.
  - Subplot A: MMSE cognitive decline curves.
  - Subplot B: Dementia conversion hazard $P(\text{AD})$ (%) over time.
- **Axes**: X-axis = Months from Baseline; Y-axis = MMSE Score / $P(\text{AD})$ (%).
- **Key Takeaway**: Shows that aggressive lifestyle risk reduction attenuates 3-year cognitive decline by 2.8 MMSE points and reduces 36-month dementia conversion probability from 84% to 52%.

---

## 19. fig_shap_waterfall_case1
- **Filename**: `fig_shap_waterfall_case1.pdf`, `fig_shap_waterfall_case1.png`
- **Final Size**: Half width (3.2 in $\times$ 3.2 in).
- **Data Source**: Local TreeSHAP attributions for Demo Case 1 (Eleanor Vance, Stable CN).
- **Plot Type**: Horizontal bar chart of SHAP log-odds contributions.
- **Axes**: Y-axis = Clinical Feature & Value; X-axis = SHAP Log-Odds Contribution [-1.8, +0.6].
- **Key Takeaway**: Demonstrates that high baseline MMSE (29.0) and CDR Global (0.0) strongly drive log-odds away from AD (-1.42 and -1.15), explaining the model's high confidence in the CN stage.

---

## 20. fig_loo_modality
- **Filename**: `fig_loo_modality.pdf`, `fig_loo_modality.png`
- **Final Size**: Half width (3.2 in $\times$ 2.8 in).
- **Data Source**: Leave-one-modality-out counterfactual inference on a multimodal MCI/AD case.
- **Plot Type**: Horizontal bar chart of probability delta $\Delta P(\text{AD})$ when withholding specific modalities.
- **Axes**: Y-axis = Withheld Diagnostic Modality; X-axis = $\Delta P(\text{AD})$ (%) [-30, +5%].
- **Key Takeaway**: Quantifies the diagnostic reliance on specialized modalities: withholding CSF biomarkers drops $P(\text{AD})$ from 53% to 29% ($\Delta = -24\%$), while withholding volumetric MRI drops $P(\text{AD})$ to 34% ($\Delta = -19\%$).

---

## 21. fig_stage_prob_timeline_case1
- **Filename**: `fig_stage_prob_timeline_case1.pdf`, `fig_stage_prob_timeline_case1.png`
- **Final Size**: Half width (3.2 in $\times$ 2.6 in).
- **Data Source**: Eleanor Vance (Stable CN, Test Cohort) visits across M0, M6, M12, M24, M36.
- **Plot Type**: Stacked bar chart of calibrated probabilities $P(\text{CN})$, $P(\text{MCI})$, $P(\text{AD})$.
- **Axes**: X-axis = Consultation Month; Y-axis = Stage Probability (%) [0, 100%].
- **Key Takeaway**: Shows continuous, stable Cognitively Normal classification ($P(\text{CN}) \ge 93\%$) across 3 years of clinical follow-up.

---

## 22. fig_stage_prob_timeline_case2
- **Filename**: `fig_stage_prob_timeline_case2.pdf`, `fig_stage_prob_timeline_case2.png`
- **Final Size**: Half width (3.2 in $\times$ 2.6 in).
- **Data Source**: Arthur Pendelton (CN $\to$ MCI Converter, Test Cohort) visits across M0, M12, M24, M36.
- **Plot Type**: Stacked bar chart of calibrated probabilities.
- **Axes**: X-axis = Consultation Month; Y-axis = Stage Probability (%) [0, 100%].
- **Key Takeaway**: Highlights early detection of impending transition: $P(\text{CN})$ declines from 88% at baseline to 62% at Month 12, with definitive syndromic conversion to MCI ($P(\text{MCI}) = 72\%$) at Month 24.

---

## 23. fig_stage_prob_timeline_case3
- **Filename**: `fig_stage_prob_timeline_case3.pdf`, `fig_stage_prob_timeline_case3.png`
- **Final Size**: Half width (3.2 in $\times$ 2.6 in).
- **Data Source**: Margaret Sterling (MCI $\to$ AD Converter, Test Cohort) visits across M0, M6, M12, M24 plus counterfactual scenario.
- **Plot Type**: Stacked bar chart comparing observed disease progression against simulated lifestyle scenario.
- **Axes**: X-axis = Visit / Scenario; Y-axis = Stage Probability (%) [0, 100%].
- **Key Takeaway**: Rapid progression to AD ($P(\text{AD}) = 88\%$ at Month 24) is partially mitigated under the counterfactual intervention scenario, reducing $P(\text{AD})$ to 60%.

---

## 24. fig_trajectory_panels_case1
- **Filename**: `fig_trajectory_panels_case1.pdf`, `fig_trajectory_panels_case1.png`
- **Final Size**: Full width (6.5 in $\times$ 2.4 in), 3 subplots.
- **Data Source**: Eleanor Vance longitudinal measurements across M0–M36.
- **Plot Type**: 3-panel trajectory plots for MMSE, CDR Global, and ADAS11-Cog with 90% confidence bands.
- **Axes**: X-axis = Consultation Month; Y-axis = Test Scores (MMSE: 24–31, CDR: -0.2–1.2, ADAS11: 0–18).
- **Key Takeaway**: Depicts flat, preserved cognitive function over 36 months, with narrow uncertainty bands reflecting high stability.

---

## 25. fig_trajectory_panels_case2
- **Filename**: `fig_trajectory_panels_case2.pdf`, `fig_trajectory_panels_case2.png`
- **Final Size**: Full width (6.5 in $\times$ 2.4 in), 3 subplots.
- **Data Source**: Arthur Pendelton longitudinal measurements across M0–M36.
- **Plot Type**: 3-panel trajectory plots for MMSE, CDR Global, and ADAS11-Cog with 90% confidence bands.
- **Axes**: X-axis = Consultation Month; Y-axis = Test Scores (MMSE: 20–31, CDR: -0.2–1.2, ADAS11: 5–26).
- **Key Takeaway**: Captures progressive functional decline: MMSE drops from 29 to 24 while ADAS11 worsens from 8 to 17.5 points, accompanied by widening forecast uncertainty bands.

---

## 26. fig_trajectory_panels_case3
- **Filename**: `fig_trajectory_panels_case3.pdf`, `fig_trajectory_panels_case3.png`
- **Final Size**: Full width (6.5 in $\times$ 2.4 in), 3 subplots.
- **Data Source**: Margaret Sterling longitudinal measurements across M0–M24.
- **Plot Type**: 3-panel trajectory plots for MMSE, CDR Global, and ADAS11-Cog with 90% confidence bands.
- **Axes**: X-axis = Consultation Month; Y-axis = Test Scores (MMSE: 14–29, CDR: 0–2.0, ADAS11: 10–36).
- **Key Takeaway**: Demonstrates steep cognitive collapse into clinical dementia: MMSE plummets from 26 to 18 points within 24 months, with CDR Global crossing to 1.0 (moderate impairment).

---

## 27. fig_risk_over_visits
- **Filename**: `fig_risk_over_visits.pdf`, `fig_risk_over_visits.png`
- **Final Size**: Half width (3.2 in $\times$ 2.6 in).
- **Data Source**: Model progression hazard probabilities across clinical visits for Cases 1, 2, and 3.
- **Plot Type**: Multi-line progression risk plot across visits M0 to M36.
- **Axes**: X-axis = Consultation Month; Y-axis = Conversion Hazard (%) [0, 105%].
- **Key Takeaway**: Highlights dramatic clinical divergence: Case 1 maintains negligible risk ($\le 5\%$), Case 2 demonstrates escalating conversion risk (12% to 74%), and Case 3 reaches acute hazard (>90%).

---

## 28. fig_earlier_phase_summary
- **Filename**: `fig_earlier_phase_summary.pdf`, `fig_earlier_phase_summary.png`
- **Final Size**: Full width (6.5 in $\times$ 2.6 in).
- **Data Source**: Research project lineage synthesis across Phase 1, Phase 2, and Phase 3.
- **Plot Type**: Structured architectural phase comparison schema.
- **Axes**: Categorical phase blocks.
- **Key Takeaway**: Summarizes the technological progression: Phase 1 (cross-sectional multimodal screener) and Phase 2 (BiLSTM/GRU recurrent modeling) provided essential empirical baselines that culminated in Phase 3's production-ready, calibrated hierarchical digital twin.
