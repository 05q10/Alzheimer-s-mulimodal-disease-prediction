import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from data_loader import load_merged_data

import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# # Experiment 3 — 5-Fold Cross-Validation with Mean ± Std
# **Goal:** Replace single 80/20 train/test split results with 5-fold CV for both the CatBoost clinical model and the ADNI biomarker ensemble.

# %%
print("Setting up for CV...")
df_clin_raw = pd.read_csv('../data/alzheimers_disease_data.csv')
y_clin = df_clin_raw['Diagnosis'].values
cb_cols = ['Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality',
    'SleepQuality', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal',
    'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides',
    'MMSE', 'FunctionalAssessment', 'ADL']
df_clin = df_clin_raw[cb_cols]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cb_f1_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(df_clin, y_clin)):
    X_train, X_val = df_clin.iloc[train_idx], df_clin.iloc[val_idx]
    y_train, y_val = y_clin[train_idx], y_clin[val_idx]
    
    # Paper exact hyperparameters
    cb = CatBoostClassifier(iterations=100, learning_rate=0.01, depth=6, eval_metric='F1', verbose=0)
    cb.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=10)
    
    preds = cb.predict(X_val)
    f1 = f1_score(y_val, preds, average='weighted' if len(set(y_clin)) > 2 else 'binary')
    cb_f1_scores.append(f1)

cb_mean_f1, cb_std_f1 = np.mean(cb_f1_scores), np.std(cb_f1_scores)

# %% [markdown]
# ## ADNI Biomarker Ensemble CV

# %%
data_dir = "../data"
merged, dx_col = load_merged_data(data_dir)
merged = merged.dropna(subset=[dx_col])
valid_dx = ['CN', 'MCI', 'Dementia']
merged[dx_col] = merged[dx_col].replace({'AD': 'Dementia'})
merged = merged[merged[dx_col].isin(valid_dx)]

# Prepare features
features = ['ABETA', 'TAU', 'ABETA_TAU_RATIO', 'MMSE', 'MMSE_TAU_INTERACT', 'APVOLUME']
genotype_cols = [c for c in merged.columns if 'APOE' in c.upper()]
features += genotype_cols

# Fill missing
for col in features:
    if pd.api.types.is_numeric_dtype(merged[col]):
        merged[col] = merged[col].fillna(merged[col].median())

# For genotype if not numeric, we one-hot encode.
le = LabelEncoder()
y_bio = le.fit_transform(merged[dx_col])
X_bio_raw = merged[features].copy()
# Identify non-numeric columns
non_num = X_bio_raw.select_dtypes(exclude=[np.number]).columns
if len(non_num) > 0:
    X_bio = pd.get_dummies(X_bio_raw, columns=non_num)
else:
    X_bio = X_bio_raw
X_bio = X_bio.fillna(0)

# Pre-load optimal weights instead of re-optimizing per fold to save time, or use equal weights
# as a proxy for the ensemble behavior in CV.
w_xgb, w_rf, w_mlp = 0.4, 0.3, 0.3 

bio_f1_scores = []
bio_acc_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_bio, y_bio)):
    X_train, X_val = X_bio.iloc[train_idx], X_bio.iloc[val_idx]
    y_train, y_val = y_bio[train_idx], y_bio[val_idx]
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    
    # PCA
    pca = PCA(n_components=0.98, random_state=42)
    X_train_pca = pca.fit_transform(X_train_s)
    X_val_pca = pca.transform(X_val_s)
    
    # SMOTE (only on train)
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_pca, y_train)
    
    # Models
    xgb = XGBClassifier(learning_rate=0.05, max_depth=7, n_estimators=100, subsample=0.9, colsample_bytree=0.9, random_state=42, eval_metric='mlogloss')
    rf = RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_split=4, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=50, random_state=42)
    
    xgb.fit(X_train_sm, y_train_sm)
    rf.fit(X_train_sm, y_train_sm)
    mlp.fit(X_train_sm, y_train_sm)
    
    # Predict
    p_xgb = xgb.predict_proba(X_val_pca)
    p_rf = rf.predict_proba(X_val_pca)
    p_mlp = mlp.predict_proba(X_val_pca)
    
    blend = w_xgb * p_xgb + w_rf * p_rf + w_mlp * p_mlp
    preds = np.argmax(blend, axis=1)
    
    bio_f1_scores.append(f1_score(y_val, preds, average='weighted'))
    bio_acc_scores.append(accuracy_score(y_val, preds))

bio_mean_f1, bio_std_f1 = np.mean(bio_f1_scores), np.std(bio_f1_scores)

# %% [markdown]
# ## Results & Visualization

# %%
results_df = pd.DataFrame([
    {"Model": "Clinical CatBoost", "Metric": "Weighted F1", "Mean": cb_mean_f1, "Std": cb_std_f1, "Min": np.min(cb_f1_scores), "Max": np.max(cb_f1_scores)},
    {"Model": "Biomarker Ensemble", "Metric": "Weighted F1", "Mean": bio_mean_f1, "Std": bio_std_f1, "Min": np.min(bio_f1_scores), "Max": np.max(bio_f1_scores)},
    {"Model": "Biomarker Ensemble", "Metric": "Accuracy", "Mean": np.mean(bio_acc_scores), "Std": np.std(bio_acc_scores), "Min": np.min(bio_acc_scores), "Max": np.max(bio_acc_scores)}
])

results_df.to_csv("results/03_cross_validation.csv", index=False)
print("\n--- CV Results ---")
print(results_df)

plt.figure(figsize=(8, 5))
plt.boxplot([cb_f1_scores, bio_f1_scores], labels=['Clinical CatBoost', 'Biomarker Ensemble'])
plt.title('5-Fold Cross-Validation F1 Scores')
plt.ylabel('Weighted F1 Score')
plt.ylim(0, 1.0)
plt.savefig("figures/03_cross_validation_f1.png", dpi=300)
plt.show()

# %% [markdown]
# # Summary
# 
# **Headline Findings:**
# - **Clinical CatBoost CV:** 5-fold Mean F1 = {cb_mean_f1:.3f} ± {cb_std_f1:.3f}
# - **Biomarker Ensemble CV:** 5-fold Mean F1 = {bio_mean_f1:.3f} ± {bio_std_f1:.3f}
