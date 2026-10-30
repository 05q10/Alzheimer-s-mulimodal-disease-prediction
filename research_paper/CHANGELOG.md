# CHANGELOG: Factual Auditing and Ground-Truth Corrections

This changelog records all empirical, mathematical, and algorithmic discrepancies identified between the draft manuscript (`main_before_v2.tex`) and the authoritative repository sources (`server/final_metrics.json`, `server/data/unified_visits.parquet`, `server/data/splits.json`, `server/ml/twin/stager.py`, `server/experiments/results/exp09_stager_results.md`, `server/experiments/results/exp10_progression_results.md`, and `lstm-gru-masking-no-masking.ipynb`).

---

## 1. Multi-Tier Stager Class Weights
- **Old Text**: Stated that class weights were removed during reconciliation, or variously referenced unweighted loss.
- **New Text**: Explicitly clarifies that active training code in `server/ml/twin/stager.py` (Line 173) enforces asymmetric class weighting: `class_weights=[1.0, 1.5, 2.0]` for the three classes `[0: CN, 1: MCI, 2: AD]`, deliberately penalizing false negatives on intermediate and dementia states.
- **Evidence / Source**: `server/ml/twin/stager.py` Line 173: `class_weights=[1.0, 1.5, 2.0]`.

---

## 2. Converter Share: Subject-Level vs. Visit-Level Reporting
- **Old Text**: Mentioned a single flat figure of 17.5% converters across splits without distinction.
- **New Text**: Distinguishes subject-level conversion from visit-level conversion. At the unique patient level, 17.5% of subjects convert across the cohort, which informed split stratification on `(baseline_DX, ever_converted)`. At the longitudinal visit level, 23.8% (train: 23.8%, val: 24.0%, test: 23.7%) of transition rows correspond to post-conversion or converting status.
- **Evidence / Source**: `server/final_metrics.json` (`data_provenance.splits`) and `server/experiments/results/09_data_audit.md`.

---

## 3. Probability Calibration: Temperature Scaling vs. Platt Scaling
- **Old Text**: Colloquially referred to probability calibration as "Platt scaling".
- **New Text**: Mathematically corrected to **single-parameter temperature scaling** on raw multi-class logits: $\hat{p}_k = \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}$ with scalar $T = 0.8554$ optimized via bounded NLL minimization ($T \in [0.1, 10.0]$).
- **Evidence / Source**: `server/ml/twin/stager.py` (`_fit_temperature_scaling`, Lines 195–215) and `server/ml/twin/weights/stager_meta.json` (`temperature: 0.855439985`).

---

## 4. Bootstrap Iterations and Resampling Design
- **Old Text**: Draft inconsistently cited 500 or 1,000 bootstrap iterations.
- **New Text**: Standardized to $B = 1,000$ patient-cluster bootstrap iterations (seed = 42) with replacement across all reported 95% confidence intervals to eliminate cluster-correlated intra-patient visit bias.
- **Evidence / Source**: `exp09_baseline_stager.py` and `table4_stager_results.tex` ($B=1,000$, seed=42).

---

## 5. CatBoost Hyperparameters for Stager and Progression Regressors
- **Old Text**: Mentioned generic tree parameters or left hyperparameter values unspecified.
- **New Text**: Documents exact hyperparameter configurations:
  - Model A (TwinStager): `iterations=600`, `depth=6`, `learning_rate=0.05`, `l2_leaf_reg=3.0`, `class_weights=[1.0, 1.5, 2.0]`, `early_stopping_rounds=50`.
  - Model B (Progression Regressors): `iterations=500`, `depth=6`, `learning_rate=0.03`, `l2_leaf_reg=4.0`, early stopping on validation MAE.
- **Evidence / Source**: `server/ml/twin/stager.py` and `server/ml/twin/progression.py`.

---

## 6. Tier T1b Bedside Screener Confusion Matrix Behavior
- **Old Text**: Hardcoded claim that Tier T1b "classifies all ambiguous cases into CN".
- **New Text**: Rigorously reports the empirical confusion matrix on $N=3,022$ holdout test visits: at Tier T1b (demographics + MMSE without CDR functional scores), the decision surface predicts CN for all test visits (CN recall = 1.000, MCI recall = 0.000, AD recall = 0.000, Macro-F1 = 0.656). The text accurately explains this clinical limitation: non-functional bedside screening alone lacks discriminative margin to separate early MCI without functional assessment.
- **Evidence / Source**: `server/experiments/results/exp09_stager_results.md` (Tier T1b CM: CN 1211/1211, MCI 0/1202, AD 0/609).

---

## 7. Sequence-Model Test Set Sample Size: 589 vs. 456 Subjects
- **Old Text**: Quoted a 589-subject test set for BiLSTM without clarifying why it differed from the 456-subject test split.
- **New Text**: Explicitly clarifies the cohort lineage:
  - The legacy exploratory notebook `lstm-gru-masking-no-masking.ipynb` used an unstratified 80/20 train/test split on 2,945 available subjects, yielding 589 test subjects (1,340 sliding sequence windows).
  - The standardized production pipeline uses patient-level stratified splitting on all 3,034 subjects, reserving 456 test subjects (846 sliding sequence windows; 3,022 time-axis visits). Both evaluations are reported with transparent labeling.
- **Evidence / Source**: `lstm-gru-masking-no-masking.ipynb` (Cell 18), `server/data/splits.json`, and `server/scratch/run_mask_comparison.py`.

---

## 8. Biomarker Ensemble Accuracy: 63.3% vs. 48.1%
- **Old Text**: Mentioned 63.3% in Methods while Appendix C listed 48.1%.
- **New Text**: Harmonizes and contextualizes both figures:
  - 63.3% represents the Optuna-tuned multi-algorithm ensemble (Random Forest, Extra Trees, LightGBM) on the balanced 3,000-row ADNI CSF subset in exploratory Phase 1.
  - 48.1% [0.435, 0.526] represents the single baseline Random Forest evaluated under strict un-imputed cluster bootstrap in `06_bootstrap_results.txt`.
- **Evidence / Source**: `server/final_metrics.json` (`legacy_biomarker_ensemble`), `server/experiments/results/06_bootstrap_results.txt`.

---

## 9. Kaggle Clinical Screener Accuracy
- **Old Text**: Variously cited Kaggle screener accuracy without exact confidence bounds.
- **New Text**: Reports exact test accuracy: 95.6% [0.935, 0.972] on the 2,149-subject Kaggle lifestyle screener cohort.
- **Evidence / Source**: `server/final_metrics.json` (`legacy_kaggle_screener`).

---

## 10. Progression Model B Feature Inputs: 10 Core Anchors
- **Old Text**: Ambiguously suggested Model B might consume dynamic longitudinal slopes or historical lags.
- **New Text**: Affirms with architectural certainty that `progression_gbm_delta_*.cbm` consumes **strictly 10 anchor features**:
  - 4 Demographics & Genetics: `entry_age`, `sex`, `education_years`, `apoe4_count`.
  - 6 Cognitive & Functional Batteries: `MMSE`, `CDGLOBAL`, `CDRSB`, `FAQTOTAL`, `TOTSCORE`, `MOCA`.
  Model B does not consume engineered trajectory slopes or sliding lags, ensuring instant zero-leakage evaluation on first consultation.
- **Evidence / Source**: `server/ml/twin/progression.py` (`FEATURE_COLS`) and CatBoost model metadata inspection.

---

## 11. Test Visit Count: 3,022 vs. 3,330 Rows
- **Old Text**: Draft text mentioned 3,330 visits in one place and 3,022 in another.
- **New Text**: Resolves the exact count: the raw Parquet file contains 3,194 visits for the 456 test subjects. Filtering for valid time-axis visits (`exclude_time_axis == 0`) and confirmed diagnostic labels (`DIAGNOSIS IN [1, 2, 3]`) yields exactly **3,022 time-axis test visits**.
- **Evidence / Source**: `server/experiments/results/exp09_stager_results.md` and `server/data/unified_visits.parquet`.


---

## 12. LaTeX Syntax & Overleaf Compilation Fixes
- **Issue**: Unescaped `&` within `\textbf{Clinical & Algorithmic Definition}` at line 177 in `main.tex` and within cell text of `table_intake_lineage_43.tex` caused a fatal compilation abort (`! Forbidden control sequence found while scanning use of \check@nocorr@. ! Missing } inserted.` and `Extra alignment tab has been changed to \cr`).
- **Fix**:
  - Escaped ampersands in `main.tex` lines 177, 573, 574, 709, 1117, 1123.
  - Escaped `&` in `table_intake_lineage_43.tex` at line 23 (`Executive \& visuospatial battery`) and in `scripts/make_numbers.py` and `scripts/build_full_paper.py`.
  - Audited all 26 table files in `research_paper/tables/` with automated validator (`scripts/validate_tables.py`), confirming 0 delimiter discrepancies across all tabular environments.
- **Evidence / Source**: `scripts/validate_tables.py` and Overleaf compilation logs.
