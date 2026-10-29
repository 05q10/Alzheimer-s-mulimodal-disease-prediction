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
