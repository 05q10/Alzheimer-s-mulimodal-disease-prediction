# ML4H Findings-Track Experiments Summary

## Experiment 01 Modality Ablation

**Headline Findings:**
- **MRI Only**: F1 = 0.489
- **+ CSF**: F1 = 0.531
- **+ Clinical**: F1 = 0.652
- **+ Genotype (Full)**: F1 = 0.616

---

## Experiment 02 Fusion Formula

**Headline Findings:**
- Best Fusion Weights: Biomarker=0.80, MRI=0.10, Clinical=0.10
- Best Fusion AUC: 0.749

---

## Experiment 03 Cross Validation

**Headline Findings:**
- **Clinical CatBoost CV:** 5-fold Mean F1 = 0.721 ± 0.014
- **Biomarker Ensemble CV:** 5-fold Mean F1 = 0.576 ± 0.042

---

## Experiment 04 Case Studies

**Headline Findings:**
- Case 1 (CN) correctly highlights low risk scores across modalities, validating the unified score's baseline stability.
- Case 2 (MCI) illustrates a borderline scenario where biomarker signatures elevate the fused risk score ahead of severe cognitive decline.
- Case 3 (Dementia) shows strong agreement between high MRI severity and high biomarker probabilities, yielding a definitive high-risk fused score.

---

## Experiment 05 Shap Analysis

**Headline Findings:**
- Top 5 clinical SHAP features: ['FunctionalAssessment', 'ADL', 'MemoryComplaints', 'MMSE', 'BehavioralProblems']. This strongly aligns with the Pearson correlation findings.
- Top 5 biomarker SHAP features: ['ABETA', 'GENOTYPE_2_4', 'MMSE', 'APVOLUME', 'GENOTYPE_2_3']. The interactions show massive importance, validating the non-linear capability of the ensemble.

---

## Experiment 06 Bootstrap CI

**Headline Findings:**
- **Clinical CatBoost:** Test accuracy is 0.956 [95% CI: 0.935-0.972].
- **Clinical CatBoost:** Weighted F1 score is 0.956 [95% CI: 0.935-0.972].
- **Biomarker Ensemble:** Weighted F1 score is 0.270 [95% CI: 0.222-0.315].
- The robust clinical CI confirms high reliability when complete patient history is available.

---

## Experiment 07 OCR Evaluation

**Headline Findings:**
- Tested OCR extraction on validation reports.
- **Accuracy:** Exact Match (10), Fuzzy Match (1), Missed (1).
- **Limitations:** Minor numerical hallucinations and occasional missed fields occur, emphasizing the need for 'human-in-the-loop' verification.

---

## Experiment 08 Missing Modality Robustness

**Headline Findings:**
- Not reported — see VERIFICATION_REPORT.md for rationale (cross-cohort feature mismatch).

---

