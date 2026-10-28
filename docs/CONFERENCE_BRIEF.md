# Academic Conference & Publication Brief

**Working Title**: A Multimodal Longitudinal Digital Twin Framework for Alzheimer's Disease Staging, Trajectory Forecasting, and Decision Support  
**Authors**: Mihit Singasane, Ved Thakker, Ria Talsania  
**Affiliation**: Department of Computer Engineering, Sardar Patel Institute of Technology  
**Project Repository**: `d:\family\MiniProject\Allele Prediction`  
**System Version**: `2.0.0-production`  

---

## 1. Abstract (150 Words)
Alzheimer's disease management requires synthesizing cognitive assessments, molecular biomarkers, and neuroimaging across irregular patient visits. We introduce an end-to-end multimodal digital twin framework that models cross-sectional disease stage, longitudinal cognitive trajectory, and conversion risk using 22,168 clinical visits across 3,034 subjects from the Alzheimer's Disease Neuroimaging Initiative (ADNI). A calibrated multi-tier CatBoost classifier dynamically routes patient data across diagnostic tiers, achieving a held-out Macro-F1 of 0.796 (95% CI: [0.769, 0.824]). Multi-horizon delta-regressors forecast 12-month MMSE progression with a Mean Absolute Error of 1.465 points (95% CI: [1.354, 1.581]), decisively outperforming persistence and slope baselines. Leakage controls enforce subject-stratified splits, backward-only imaging joins, and exclusion of forward-looking trajectory artifacts. The architecture features an interactive counterfactual scenario lab, TreeSHAP feature attributions, leave-one-modality-out marginal sensitivity, and empirical plausibility envelope guards. This system demonstrates how rigorous data hygiene and clinical interpretability can deliver reliable, transparent decision support for neurodegenerative disease monitoring.

---

## 2. Keywords (18 Keywords)
Digital Twin, Alzheimer's Disease, Mild Cognitive Impairment, Multimodal Fusion, Longitudinal Progression, Gradient Boosted Trees, CatBoost, ADNI, Data Leakage Prevention, Counterfactual Simulation, Explainable AI, TreeSHAP, Biomarkers, Neuropsychological Testing, FreeSurfer MRI, Clinical Decision Support, Medical Informatics, Reproducible ML.

---

## 3. Supported Paper Types & Upgrade Pathways

```
[Current State: Strong System/Demo Paper] ──> [+ External Validation] ──> [Full Research Paper]
```

- **System / Demonstration Paper (Ready Today)**:  
  *Fit*: Highly competitive today for software, demonstration, and system tracks (e.g., ML4H Demonstration Track, CHIL System Track, AMIA Informatics Summit).  
  *Justification*: Working full-stack Next.js and FastAPI system, 40 passing automated verification tests, real-time counterfactual scenario lab, 43-field lineage matrix, and sub-5ms deterministic inference.
- **Short Paper / Findings Track (Ready Today)**:  
  *Fit*: Strong fit for findings/extended abstract tracks (e.g., ML4H Findings, IEEE BIBM Short Paper).  
  *Justification*: Presents clear, methodologically sound negative and null findings: demonstrating that CSF biomarkers add negligible cross-sectional syndromic staging signal over complete neuropsychological batteries ($p > 0.05$), and that volumetric MRI does not statistically improve 12-month cognitive trajectory forecasting.
- **Full Research Paper (Requires Targeted Upgrades)**:  
  *Fit*: Main tracks of top-tier venues (ACM CHIL, IEEE JBHI, JAMIA, Lancet Digital Health).  
  *Required Upgrades*:
  1. **External Validation**: Test frozen models on an independent external cohort (e.g., AIBL or OASIS-3).
  2. **Replace 2D Kaggle MRI Model**: Retrain a native 3D volumetric convolutional network directly on ADNI T1 DICOMs.
  3. **Clinician User Study**: Formal evaluation with board-certified neurologists measuring decision impact and counterfactual plausibility.

---

## 4. Defensible Claims vs. Claims We Must NOT Make

### Defensible Claims (Backed by Verified Evidence)
1. **Accurate Multi-Tier Syndromic Staging Without Synthetic Imputation**: The multi-tier CatBoost stager achieves Macro-F1 = 0.796 (95% CI: [0.769, 0.824]) on the full held-out test set ($N=3,022$ visits) without statistical data imputation (`exp09_stager_results.md`).
2. **Superior Cognitive Trajectory Forecasting at 12 and 24 Months**: CatBoost delta-regressors forecast 12-month MMSE decline with MAE = 1.465 (95% CI: [1.354, 1.581]), beating persistence (1.603) and slope extrapolation (2.108); at 24 months, MMSE MAE is 1.560 vs persistence 1.739 (`exp10_progression_results.md`).
3. **Robust Conversion Hazard Discrimination**: Predicts progression to dementia with AUROC = 0.7615 (95% CI: [0.703, 0.819]) at 12 months and AUROC = 0.8111 (95% CI: [0.756, 0.861]) at 24 months, with positive Brier Skill Scores ($+0.1379$ and $+0.2088$) (`exp10_progression_results.md`).
4. **Honest Null Findings on Biomarker Staging Increment**: Demonstrates that adding CSF biomarkers to complete cognitive tests changes Macro-F1 by only $+0.0135$ (95% CI: `[-0.0009, 0.0320]`), which includes zero (`exp09_stager_results.md`).
5. **Zero Data Leakage Protocol**: Strict subject-level stratification, strictly backward-looking temporal joins (`mri_gap_months <= 0`), and complete exclusion of 16 future-derived columns verified by automated test assertions (`server/tests/test_no_leakage.py`).

### Claims We Must NOT Make (Do NOT Overclaim)
- ❌ **Do NOT claim 99.7% MRI diagnostic accuracy on ADNI**: The 99.7% metric came from an augmented 2D Kaggle slice dataset with slice leakage. In ADNI production, it is downweighted to a 20% late fusion with explicit disclaimers.
- ❌ **Do NOT claim the model beats persistence for short-term Global CDR**: At 6m and 12m, persistence achieves MAE of 0.061 and 0.112, beating the GBM model (0.064 and 0.113) (`exp10_progression_results.md`).
- ❌ **Do NOT claim the lifestyle modifier was learned from ADNI**: ADNI does not longitudinally record lifestyle factors. The modifier is a literature-informed prior from the Lancet Commission, capped at $[0.85, 1.35]$.
- ❌ **Do NOT cite Experiment 8 (Missing Modality Robustness)**: Experiment 8 was classified as synthetic-tainted due to a 32-feature cross-cohort mismatch (`VERIFICATION_REPORT.md`).
- ❌ **Do NOT claim FDA or CE medical device clearance**: The platform is an academic investigative research prototype.

---

## 5. Reviewer Strengths & Likely Criticisms with Honest Answers

### Anticipated Reviewer Strengths
- **Methodological Hygiene**: Zero subject overlap, no future-visit temporal leakage, and complete avoidance of forward-looking imputation artifacts.
- **Clinical Honesty**: Clear reporting of null and negative results (e.g., CSF staging increment spanning zero, persistence beating GBM on short-term CDR).
- **Comprehensive Interpretability**: Full 43-field intake lineage, TreeSHAP attributions, leave-one-modality-out marginal analyses, and plausibility boundary guards.

### Reviewer Criticisms & Rebuttals
- **Criticism 1**: *"The model is trained only on ADNI, which lacks racial, socioeconomic, and community health diversity."*  
  **Honest Rebuttal**: We concede single-cohort ADNI bias as a primary limitation in our Model Card and manuscript. We explicitly frame this system as a computational demonstration of multimodal data fusion rather than a generalized clinical device.
- **Criticism 2**: *"Why include a 2D Kaggle MRI classifier if you already have FreeSurfer volumetric features?"*  
  **Honest Rebuttal**: The 2D slice model serves strictly as an auxiliary image-upload demonstration. To prevent distorted predictions, its output is temperature-scaled ($T=2.0$), heavily downweighted to $w=0.20$, and accompanied by prominent UI disclaimers.
- **Criticism 3**: *"Why use CatBoost rather than deep sequence models like Transformers or LSTMs?"*  
  **Honest Rebuttal**: We explicitly benchmarked a Masked GRU architecture against CatBoost on identical test anchors (`exp10_progression_results.md`). While missingness masking improved GRU performance over unmasked GRUs, CatBoost achieved superior test MAE (1.465 vs 1.475), runs deterministically in under 5ms, and natively handles non-random clinical missingness without synthetic imputation.

---

## 6. Target Communities & Example Venues

> [!IMPORTANT]
> Official deadlines, rankings, page limits, double-blind policies, and physical locations vary by cycle. All authors must verify submission instructions directly on each venue's official website. Do not rely on historical dates or memory.

### 1. Digital Twins in Healthcare & Medical Cybernetics
- **IEEE Journal of Biomedical and Health Informatics (J-BHI)**: Premier venue for computational health architectures and longitudinal monitoring.
- **Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)**: Excellent for biomedical digital twin demonstrations.
- **Computers in Biology and Medicine (Elsevier)**: Focuses on computational models, multi-tier software architectures, and synthetic clinical simulation.

### 2. Clinical Machine Learning & Decision Support
- **Machine Learning for Healthcare (ML4H)**: Ideal fit for both the Demonstration Track and the Findings Track (focusing on null findings and data leakage controls).
- **ACM Conference on Health, Inference, and Learning (CHIL)**: Focuses on robust machine learning, causal inference, and real-world health deployment.
- **Artificial Intelligence in Medicine (AIIM - Elsevier)**: High-impact clinical AI venue valuing interpretability and multi-modal integration.

### 3. Medical AI & Biomedical Imaging
- **Medical Image Computing and Computer Assisted Intervention (MICCAI) Workshops**: Specialized workshops on neuroimaging genetics and multimodal fusion.
- **SPIE Medical Imaging**: Strong track on computer-aided diagnosis and digital pathology/radiology workflow integration.
- **Journal of Digital Imaging (Springer)**: Radiologic informatics, image-derived volumetrics, and decision support.

### 4. Neurology, Gerontology & Alzheimer's Research
- **Alzheimer's & Dementia: Diagnosis, Assessment & Disease Monitoring (DADM - Wiley)**: Open-access translational journal focusing on clinical utility of prognostic markers.
- **Neurobiology of Aging (Elsevier)**: Emphasizes empirical neuropsychological and structural progression trajectories.
- **Frontiers in Aging Neuroscience**: Rapid-review venue for computational neuroscience and biomarker modeling.

### 5. Health Informatics & Clinical Workflows
- **AMIA Annual Symposium / AMIA Informatics Summit**: Flagship clinical informatics conferences prioritizing clinician-facing tools and decision support.
- **Journal of the American Medical Informatics Association (JAMIA)**: Top-tier clinical informatics journal focusing on evaluation, governance, and workflow safety.
- **International Journal of Medical Informatics (IJMI)**: Addresses human-AI teaming, usability, and clinical dashboard design.

### 6. Student & Undergraduate Research Tracks
- **IEEE BIBM (Bioinformatics and Biomedicine) Student Research Track**: Excellent venue for engineering capstone and undergraduate research papers.
- **ACM SAC (Symposium on Applied Computing) Bioinformatics Track**: Broad applied computer science venue with strong student paper tracks.
- **IEEE MIT Undergraduate Research Technology Conference (URTC)**: Prestigious showcase for senior undergraduate systems.

---

## 7. Submission Readiness Checklist

- [ ] **Anonymization & Blinding**: Ensure author names, institutional affiliations (Sardar Patel Institute of Technology), and specific supervisor names are removed for double-blind tracks.
- [ ] **ADNI Data Use Statement**: Include the mandatory ADNI consortium acknowledgment text and state the data access approval credential.
- [ ] **Code & Data Availability Statement**: State that raw ADNI data cannot be redistributed per DUA guidelines; provide a GitHub repository link containing anonymized configuration scripts, synthetic demo profiles, and reproduction pipelines.
- [ ] **Ethics & Regulatory Disclaimer**: Explicitly state that the system is a research prototype without FDA/CE clearance and does not handle real unconsented PHI.
- [ ] **Figures & Architecture Diagrams**: Export high-resolution vector diagrams of the system architecture, TreeSHAP summary plots (`05_shap_catboost_bar.png`), and cluster-bootstrap calibration curves.
- [ ] **Metric Synchronization**: Verify that every cited accuracy, MAE, and confidence interval matches `server/final_metrics.json` and associated results files verbatim.
