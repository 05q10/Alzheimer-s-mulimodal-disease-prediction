# Data Provenance Audit

| Experiment | Data source claimed | Data source actually used | Real/Synthetic (per Task 0) | File exists? | Row count | SHA256 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Exp 1 | ADNIMERGE, ADSP, APOE | ADNIMERGE, ADSP, APOE | Real (No synthetic markers) |  |  |  |
| Exp 2 | ADNIMERGE, ADSP, APOE | ADNIMERGE, ADSP, APOE | Real (No synthetic markers) |  |  |  |
| Exp 3 | alzheimers_disease_data.csv, ADNIMERGE, ADSP, APOE | alzheimers_disease_data.csv, ADNIMERGE, ADSP, APOE | Real (small fallback in nb was fake) | True | 2149 | 66f51dbb5f00a4bf... |
| Exp 4 | ADNIMERGE, ADSP, APOE | ADNIMERGE, ADSP, APOE | Real (synthetic just in title) |  |  |  |
| Exp 5 | alzheimers_disease_data.csv (expected), ADNIMERGE, ADSP, APOE | np.random.randn, ADNIMERGE, ADSP, APOE | Synthetic background for clinical model SHAP |  |  |  |
| Exp 6 | alzheimers_disease_data.csv, ADNIMERGE, ADSP, APOE | alzheimers_disease_data.csv, ADNIMERGE, ADSP, APOE | Real (np.random seed used harmlessly) | True | 2149 | 66f51dbb5f00a4bf... |
| Exp 7 | 07_ocr_evaluation.csv | 07_ocr_evaluation.csv | Real per-field data from CSV (originally synthetic dict) | True | 12 | 2e1ed4c0b0c9e195b98226d94d6ce967acaafba81a25efac426d0fd5b6342a3c |
| Exp 8 | ADNIMERGE, ADSP, APOE | ADNIMERGE, ADSP, APOE, np.random.uniform | Synthetic-tainted (clinical risk simulated) |  |  |  |
