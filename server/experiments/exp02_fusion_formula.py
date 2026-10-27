import os
import joblib
import pandas as pd
import numpy as np
import catboost
from sklearn.metrics import roc_auc_score, f1_score
from data_loader import load_merged_data

# %% [markdown]
# # Experiment 2 — Fusion Formula for the Unified Digital Twin Risk Score
# **Goal:** Define and validate a formula combining clinical CatBoost, MRI severity, and Biomarker ensemble into one patient-level risk score.

# %%
print("Loading trained models...")
base_model_dir = "../ml/base_model"
alele_model_dir = "../ml/alele_model/model_7f/content/model_7f"

# Load Clinical Model
catboost_model = joblib.load(os.path.join(base_model_dir, "catboost_alzheimer_model.pkl"))
cb_minmax = joblib.load(os.path.join(base_model_dir, "minmax.pkl"))
cb_std = joblib.load(os.path.join(base_model_dir, "standard.pkl"))

# Load Biomarker Ensemble
xgb_model = joblib.load(os.path.join(alele_model_dir, "xgb_model.joblib"))
rf_model  = joblib.load(os.path.join(alele_model_dir, "rf_model.joblib"))
nn_model  = joblib.load(os.path.join(alele_model_dir, "nn_model.joblib"))
blend_weights = joblib.load(os.path.join(alele_model_dir, "blend_weights_optimized.joblib"))
scaler = joblib.load(os.path.join(alele_model_dir, "scaler.joblib"))
le_dx  = joblib.load(os.path.join(alele_model_dir, "label_encoder.joblib"))
trained_columns = joblib.load(os.path.join(alele_model_dir, "trained_columns.joblib"))

# %% [markdown]
# ## Process ADNI Cohort

# %%
data_dir = "../data"
merged, dx_col = load_merged_data(data_dir)
merged = merged.dropna(subset=[dx_col])
valid_dx = ['CN', 'MCI', 'Dementia']
merged[dx_col] = merged[dx_col].replace({'AD': 'Dementia'})
merged = merged[merged[dx_col].isin(valid_dx)]

print(f"Loaded ADNI Cohort: {merged.shape[0]} patients.")

# 1. Biomarker Predictions
def get_biomarker_prob(df):
    new_patient = df[['ABETA', 'TAU', 'MMSE', 'APVOLUME']].copy()
    genotype_col = [c for c in df.columns if 'APOE' in c.upper()][0]
    new_patient['GENOTYPE'] = df[genotype_col]
    
    new_patient['ABETA_TAU_RATIO'] = new_patient['ABETA'] / new_patient['TAU']
    new_patient['MMSE_TAU_INTERACT'] = new_patient['MMSE'] * new_patient['TAU']
    
    encoded = pd.get_dummies(new_patient)
    aligned_input = pd.DataFrame(np.zeros((len(df), len(trained_columns))), columns=trained_columns)
    for c in encoded.columns:
        if c in aligned_input.columns:
            aligned_input[c] = encoded[c].values
            
    aligned_input_scaled = scaler.transform(aligned_input.fillna(0))
    
    # Predict
    try: n_xgb = xgb_model.n_features_in_
    except: n_xgb = xgb_model.get_booster().num_features()
    
    p_xgb = xgb_model.predict_proba(aligned_input_scaled[:, :n_xgb])
    p_rf  = rf_model.predict_proba(aligned_input_scaled[:, :rf_model.n_features_in_])
    p_nn  = nn_model.predict_proba(aligned_input_scaled[:, :nn_model.n_features_in_])
    
    blend_probs = (blend_weights['w_xgb'] * p_xgb + blend_weights['w_rf'] * p_rf + blend_weights['w_nn'] * p_nn)
    
    # We want probability of Dementia (or Dementia + MCI)
    dem_idx = list(le_dx.classes_).index('Dementia')
    mci_idx = list(le_dx.classes_).index('MCI')
    return blend_probs[:, dem_idx] + 0.5 * blend_probs[:, mci_idx] # weighted sum for risk

merged['Biomarker_Risk'] = get_biomarker_prob(merged)

# 2. MRI Severity Proxy (Ordinal 0-3)
# We use inverse APVOLUME quantiles as a severity proxy (lower volume = higher severity)
quantiles = merged['APVOLUME'].quantile([0.25, 0.5, 0.75]).values
def mri_severity(v):
    if v < quantiles[0]: return 3.0
    if v < quantiles[1]: return 2.0
    if v < quantiles[2]: return 1.0
    return 0.0

merged['MRI_Severity'] = merged['APVOLUME'].apply(mri_severity) / 3.0 # normalized 0-1

# 3. Clinical Prediction (Simulated missing for ADNI, or using placeholders if we had overlapping features)
# Since they are disjoint, we set Clinical to NaN for this cohort to test missingness robustness.
merged['Clinical_Risk'] = np.nan

# %% [markdown]
# ## Fusion Formula Calibration

# %%
def fused_risk(row, w_bio=0.4, w_mri=0.3, w_clin=0.3):
    weights = []
    scores = []
    
    if not np.isnan(row['Biomarker_Risk']):
        weights.append(w_bio)
        scores.append(row['Biomarker_Risk'])
        
    if not np.isnan(row['MRI_Severity']):
        weights.append(w_mri)
        scores.append(row['MRI_Severity'])
        
    if not np.isnan(row['Clinical_Risk']):
        weights.append(w_clin)
        scores.append(row['Clinical_Risk'])
        
    if sum(weights) == 0:
        return 0.0
    
    # Renormalize weights over available modalities
    weights = np.array(weights) / sum(weights)
    return np.sum(weights * np.array(scores))

# Grid Search on ADNI
best_auc = 0
best_w = (0.33, 0.33, 0.33)

y_true = (merged[dx_col] == 'Dementia').astype(int)

for wb in np.linspace(0.1, 0.9, 9):
    for wm in np.linspace(0.1, 0.9, 9):
        wc = 1.0 - wb - wm
        if wc < 0: continue
        
        merged['Temp_Risk'] = merged.apply(lambda r: fused_risk(r, wb, wm, wc), axis=1)
        try:
            auc = roc_auc_score(y_true, merged['Temp_Risk'])
            if auc > best_auc:
                best_auc = auc
                best_w = (wb, wm, wc)
        except:
            pass

print(f"Best Weights (Bio, MRI, Clin): {best_w} -> AUC: {best_auc:.3f}")

merged['Final_Risk_Score'] = merged.apply(lambda r: fused_risk(r, *best_w), axis=1)

# %% [markdown]
# ## Patient Case Studies

# %%
sample_cases = merged.sample(10, random_state=42)[['RID', 'ABETA', 'TAU', 'APVOLUME', dx_col, 'Biomarker_Risk', 'MRI_Severity', 'Clinical_Risk', 'Final_Risk_Score']]
sample_cases.to_csv("results/02_fusion_cases.csv", index=False)
print("\n--- Example Patients ---")
print(sample_cases)

import json
import datetime

fusion_info = {
    "best_weights": {
        "Biomarker": float(best_w[0]),
        "MRI": float(best_w[1]),
        "Clinical": float(best_w[2])
    },
    "best_auc": float(best_auc),
    "generated_at": datetime.datetime.now().isoformat()
}

with open("results/02_fusion_formula.json", "w") as f:
    json.dump(fusion_info, f, indent=4)

# %% [markdown]
# # Summary
# 
# **Headline Findings:**
