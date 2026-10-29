# TRIPOD+AI 2024 Checklist Mapping

**Guideline**: Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis + Artificial Intelligence (TRIPOD+AI)  
**Citation**: Collins et al., *BMJ* 2024;385:e078378. DOI: 10.1136/bmj-2023-078378  
**Manuscript**: *A Multimodal Longitudinal Digital Twin Framework for Alzheimer's Disease Staging, Trajectory Forecasting, and Decision Support*  

---

| Item No. | TRIPOD+AI Checklist Item | Manuscript Section Where Addressed | Status / Notes |
|---|---|---|---|
| **Title** | | | |
| 1 | Identify the study as developing, validating, or updating a machine learning / AI clinical prediction model, target condition, and outcome. | Title & Abstract | **Done**: Title clearly specifies longitudinal staging, forecasting, and digital twin for AD. |
| **Abstract** | | | |
| 2 | Provide a structured abstract (background, objectives, data sources, participants, ML methods, performance metrics, and clinical implications). | Abstract | **Done**: Structured abstract (~200 words) with background, methods, results (CIs cited), and conclusion. |
| **Introduction** | | | |
| 3a | Explain the medical context, clinical pathway, and rationale for developing the AI model. | Section 1 (Introduction) | **Done**: Discusses fragmented assessment across cognitive tests, CSF, and MRI. |
| 3b | Specify the intended user, clinical setting, and timing of model predictions. | Section 1 & Section 3 | **Done**: Outpatient neurology, memory clinics, screening at baseline and prospective follow-up visits. |
| 4 | State specific objectives, including whether model is diagnostic (staging) or prognostic (trajectory/conversion). | Section 1.1 (Key Contributions) | **Done**: Dual objective: Model A (cross-sectional diagnostic stager) and Model B (longitudinal prognostic progression). |
| **Methods: Data** | | | |
| 5a | Describe the source of data (study design, recruitment dates, multicenter cohort). | Section 3.1 & Table 1 | **Done**: ADNI observational multicenter cohort (ADNI-1, GO, 2, 3), 2005–2023. |
| 5b | Detail data linkage, merging rules, and temporal alignment across modalities. | Section 3.2 & Table 2 | **Done**: Backward-only `merge_asof` joins ($\le 180$ days), date reconstruction rules. |
| 6a | Specify eligibility criteria for participants (inclusion and exclusion criteria). | Section 3.1 & Table 1 | **Done**: Non-missing canonical `DIAGNOSIS`, valid baseline anchor, exclusion of time-axis artifacts. |
| 6b | Describe participant flow and demographics across development and validation partitions. | Section 3.1 & Table 1 | **Done**: 2,123 train, 455 validation, 456 test subjects; converter proportions reported. |
| **Methods: Outcome** | | | |
| 7a | Define diagnostic and prognostic outcomes precisely, including blinding to predictors. | Section 3.3 & Section 3.4 | **Done**: Canonical diagnosis (CN, MCI, AD) from ADNI expert panel; delta scores $\Delta\text{MMSE}, \Delta\text{CDR}, \Delta\text{ADAS11}$. |
| 7b | Report time horizons for prognostic outcomes. | Section 3.4 & Table 5 | **Done**: Strictly standardized at 6, 12, and 24 months. |
| **Methods: Predictors** | | | |
| 8a | Define all candidate predictors and measurement methods across modalities. | Section 3.3 & Table 9 | **Done**: Detailed across T1 to T4 tiers; 43-field intake lineage specified. |
| 8b | Detail handling of predictor measurement timing relative to outcome. | Section 3.2 | **Done**: Strict backward-looking constraint ($\text{scan\_date} \le \text{exam\_date}$). |
| **Methods: Sample Size** | | | |
| 9 | State the sample size and number of outcome events. | Section 3.1 & Table 1 | **Done**: 3,034 subjects, 22,168 visits; 532 conversion events across cohort. |
| **Methods: Missing Data** | | | |
| 10 | Describe missing data patterns and how missingness was handled during development and testing. | Section 3.2 & Section 3.3 | **Done**: Native CatBoost tree splits without statistical imputation; multi-tier routing (T1--T4). |
| **Methods: ML Modeling** | | | |
| 11a | Detail model architectures, algorithms, hyperparameters, and feature selection. | Section 3.3, 3.4, Appendix B | **Done**: CatBoost gradient-boosted trees, Platt calibration, delta-change regressors. |
| 11b | Describe data partitioning strategy (cross-validation, bootstrap, holdout). | Section 3.1 | **Done**: Subject-stratified group split (70/15/15) preserving converter balance. |
| 11c | Specify techniques used to avoid data leakage. | Section 3.2 | **Done**: Zero subject overlap, no future scans, blacklisting of 16 trajectory features. |
| **Methods: Evaluation** | | | |
| 12 | Specify performance metrics, confidence intervals, and baselines. | Section 3.4 & Section 4 | **Done**: Macro-F1, Accuracy, MAE, AUROC, Brier Skill Score, ECE; 1,000 cluster-bootstrap CIs. |
| **Results** | | | |
| 13 | Present baseline characteristics of participants across development and validation sets. | Table 1 & Table 4 | **Done**: Reported in Table 1 and Section 4.1. |
| 14 | Report absolute numbers of participants and events in each performance evaluation. | Table 4, 5, 6, 7 | **Done**: Sample sizes and anchor counts provided for every evaluation table. |
| 15 | Present model discrimination, calibration curves, and comparisons with baselines. | Section 4.1, 4.2, 4.3; Fig 3--8 | **Done**: Full calibration diagrams, forest plot, ROC curves, and persistence baselines. |
| 16 | Report all model ablations and incremental utility of modalities (including null findings). | Section 4.4, Table 7, Fig 7 | **Done**: Null findings for CSF biomarkers in staging and MRI in 12m forecasting fully disclosed. |
| **Discussion** | | | |
| 17 | Discuss methodological strengths, limitations, risk of bias, and generalizability. | Section 5 (Limitations) & Table 11 | **Done**: Single-cohort ADNI bias, Kaggle slice model disconnect, diagnostic circularity. |
| 18 | Discuss clinical relevance, decision support implications, and comparison with prior work. | Section 4.5 & Section 6 | **Done**: TreeSHAP attributions, Scenario Lab counterfactuals, plausibility envelope guards. |
| **Other Information** | | | |
| 19a | Provide information on accessibility of models, code, and supplementary materials. | Declarations & Appendix | **Done**: GitHub link, Overleaf package, data access via ADNI DUA. |
| 19b | State funding sources and role of funders. | Declarations | **Done**: Stated in Declarations. |
| 19c | Disclose competing interests. | Declarations | **Done**: None declared. |

