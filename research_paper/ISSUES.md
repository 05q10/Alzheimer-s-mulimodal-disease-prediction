# Repository Verification & Metric Audit: Discrepancies and Resolutions

**Date**: October 2026  
**Document**: `research_paper/ISSUES.md`  
**Purpose**: Document all numeric disagreements, cohort filtering differences, and methodological caveats across repository result files.

---

## 1. Cross-Sectional Stager (Model A) Discrepancy

### Description
Two different sets of test-set accuracy figures appear in the repository for the Model A TwinStager:
- **Source A (`server/final_metrics.json` lines 23–30)**:
  - T1a Demographics: 0.812
  - T1b (+MMSE): 0.841
  - T2a (+CDR): 0.887
  - T2b (+Neuropsych): 0.908
  - T3 (+CSF): 0.925
  - T4 (+FreeSurfer MRI): 0.938
  - Macro-F1: 0.894
- **Source B (`server/experiments/results/exp09_stager_results.md`)**:
  - T1 Demographics: 0.541 (Macro-F1: 0.449 [0.408, 0.486])
  - T1b (+MMSE): 0.687 (Macro-F1: 0.656 [0.623, 0.689])
  - T2a (+CDR): 0.790 (Macro-F1: 0.771 [0.742, 0.800])
  - T2b (+Neuropsych): 0.804 (Macro-F1: 0.788 [0.759, 0.817])
  - T3 (+CSF): 0.802 (Macro-F1: 0.785 [0.756, 0.814])
  - T4 (+FreeSurfer MRI): 0.811 (Macro-F1: 0.796 [0.769, 0.824])

### Cause & Resolution
- **Cause**: Source A reflects an early exploratory run where subjects with missing neuropsychological sub-tests were pre-filtered (`dropna`), leaving a high-completeness subset that artificially inflated diagnostic accuracy. Source B reflects the true multi-tier evaluation across all 3,022 held-out test visits (including natural non-random missingness).
- **Resolution in Paper**: In accordance with the prompt's truthfulness rule, both sources are documented. The paper reports the rigorous, un-imputed multi-tier evaluation from `exp09_stager_results.md` (Tier T4 Macro-F1 = 0.796 [0.769, 0.824], Accuracy = 0.811) as the true test metric, and notes the earlier filtered benchmark as an exploratory artifact.

---

## 2. Longitudinal Progression TOTSCORE (ADAS11) Cohort Expansion

### Description
In progression forecasting metrics, earlier summaries report 24-month TOTSCORE persistence MAE = 3.665 and GBM MAE = 3.292, whereas expanded tables report persistence MAE = 3.912 and GBM MAE = 3.518.

### Cause & Resolution
- **Cause**: Earlier exploratory runs enforced a simultaneous non-null constraint across all three cognitive targets (MMSE, CDGLOBAL, and TOTSCORE), restricting the evaluation cohort to ~1,120 visit pairs. The final standardized pipeline filters strictly per-target (`dropna(subset=[future_target, anchor_target])`), expanding the 24-month TOTSCORE test cohort to 1,578 visit pairs across 369 subjects.
- **Resolution in Paper**: Table~\ref{tab:forecast_mae} reports the standardized anchor evaluation on 664 anchors (GBM MAE = 3.120 [2.824, 3.441] vs persistence 3.665). The expanded cohort metrics (3.518 vs 3.912) are cited in Section 4.2 text to confirm that CatBoost decisively outperforms persistence across both cohort definitions.

---

## 3. Retraction of Experiment 8 (Missing Modality Robustness)

### Description
Experiment 8 of the legacy ML4H suite attempted to evaluate model robustness when modalities were artificially dropped.

### Cause & Resolution
- **Cause**: As audited in `server/experiments/results/VERIFICATION_REPORT.md`, Experiment 8 suffered from a severe cross-cohort feature incompatibility: 32 of 33 clinical features required by `catboost_alzheimer_model.pkl` were completely absent from the merged ADNI cohort used in the script. Synthetic random fallback data had tainted the results.
- **Resolution in Paper**: In accordance with strict truthfulness rules, Experiment 8 is completely excluded from all manuscript claims, tables, and figures. It is documented in the audit as an invalid run.

---

## 4. Kaggle 2D MRI Slice Accuracy Caveat

### Description
The 2D ResNet34 image classifier trained on Kaggle slices achieved 99.7% classification accuracy across four dementia classes.

### Cause & Resolution
- **Cause**: The Kaggle dataset consists of augmented 2D axial slices where adjacent slices from the same subject appear in both training and test sets (slice-level data leakage). Furthermore, the dataset was not derived from or validated on ADNI 3D volumetric scans.
- **Resolution in Paper**: We explicitly reject the 99.7% metric as unvalidated on clinical cohorts. In our system, the 2D image classifier is treated strictly as an auxiliary demonstration module, downweighted to $w=0.20$ late fusion with temperature scaling ($T=2.0$), and accompanied by a persistent UI disclaimer.

---

## 5. Status of Vitals Features (BMI & Blood Pressure)

### Description
BMI, Systolic BP, and Diastolic BP were evaluated for inclusion in the clinical staging feature set.

### Cause & Resolution
- **Finding**: Vitals are recorded in only 18.2% of ADNI longitudinal rows. Paired cluster-bootstrap evaluation demonstrated $\Delta\text{MAE} = -0.008$ (95% CI: [-0.031, 0.016], $p=0.42$).
- **Resolution**: Vitals features were formally excluded from model feature sets and classified as `recorded_only` in the 43-field lineage matrix.

---

## 6. Lifestyle Modifier Grounding

### Description
The lifestyle risk modifier applies relative risk multipliers across 10 modifiable factors (smoking, exercise, diet, sleep, etc.).

### Cause & Resolution
- **Grounding**: ADNI does not longitudinally record lifestyle behaviors. The modifier was not learned from ADNI; it is an external literature-grounded prior derived from the Lancet Commission on Dementia Prevention (Livingston et al., 2020, 2024 updates).
- **Resolution in Paper**: Explicitly labeled in the abstract, methods, and Table~\ref{tab:lifestyle_factors} as an external literature prior.

---

## 7. Bidirectional LSTM Sequence Stager Integration & Table Margin Audit

### Description
The Bidirectional LSTM model (`server/ml/lstm_model/model_weights/lstm_model.keras`) is deployed as an active operational endpoint (`POST /lstm-predict`) powering the frontend temporal prediction tool (`client/app/temporal-prediction/`), but was omitted from the initial draft of Table 3. In addition, wide tables overflowed into the right margin when rendered in single-column article format.

### Cause & Resolution
- **Cause**: The Phase 2 rewrite initially focused on the primary CatBoost tabular stager (Model A) and multi-horizon regressors (Model B). The BiLSTM sequence stager is an active sequential deep learning model trained on 3-visit sliding windows ($T \le 3$) of 5 longitudinal features (`entry_age`, `CDGLOBAL`, `MMSCORE`, `TOTSCORE`, `visit_month`). Furthermore, unconstrained table column specifications exceeded the 15.92 cm text width of the 1-inch-margin A4 layout.
- **Resolution**:
  1. Added the `BiLSTM Sequence Stager` explicitly to Table 3 (`table3_model_inventory.tex`), containing all 6 audited dimensions: Component Model, Architecture, Input Modalities, Deployment Status, Training Cohort, and Primary Metric.
  2. Integrated comprehensive narrative descriptions of the BiLSTM architecture in Section 1 (Introduction), Section 3.3 (Methodology), Section 4 (Results), and Appendix A (Model Architectures).
  3. Wrapped all 11 tables in `\begin{adjustbox}{max width=\linewidth}` and re-balanced column widths in `research_paper/scripts/make_numbers.py` and `research_paper/preamble.tex`, guaranteeing zero margin overflow on Overleaf and local LaTeX compilers.


---

## 8. Sequential Recurrent Models: 589 Subjects vs. 456 Subjects Split
### Description
`lstm-gru-masking-no-masking.ipynb` evaluated 589 test subjects (1,340 sliding sequence windows), whereas the standardized production pipeline splits the 3,034 cohort into 456 test subjects (846 sequence windows).

### Cause & Resolution
- **Cause**: The exploratory notebook used an unstratified 80/20 train/test split on 2,945 available subjects with complete sequence windows. The production pipeline enforces patient-level stratified splitting on `(baseline_DX, ever_converted)` across all 3,034 subjects (70/15/15 ratio), leaving 456 test subjects.
- **Resolution in Paper**: Both results are retained and explicitly contextualized: Table~\ref{tab:seq_model_compare} reports the controlled evaluation on the 846 standardized test sequence windows, while citing the exploratory 1,340-window findings as preliminary research evidence.

---

## 9. Biomarker Ensemble Accuracy: 63.3% vs. 48.1%
### Description
Earlier text reported 63.3% accuracy for the fluid biomarker model, while Appendix C listed 48.1% [0.435, 0.526].

### Cause & Resolution
- **Cause**: 63.3% reflects an Optuna-tuned multi-algorithm ensemble (Random Forest, Extra Trees, LightGBM) evaluated on a balanced 3,000-row ADNI CSF subset in exploratory Phase 1. 48.1% reflects the single-algorithm baseline Random Forest evaluated under strict cluster bootstrap across unselected visits in `06_bootstrap_results.txt`.
- **Resolution in Paper**: Both sources are cited with their respective sample sizes, algorithms, and validation protocols, removing any silent ambiguity.
