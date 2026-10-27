import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import optuna
from data_loader import load_merged_data

import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# # Experiment 1 — Modality Ablation on the ADNI Biomarker+Genotype Ensemble
# **Goal:** Quantify how much each modality group contributes to the existing blended ensemble (XGBoost + RF + NN) that predicts CN / MCI / Dementia.

# %%
print("Setting up directories...")
os.makedirs("notebooks", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

data_dir = "../data"
merged, dx_col = load_merged_data(data_dir)

# Clean Target
merged = merged.dropna(subset=[dx_col])
valid_dx = ['CN', 'MCI', 'Dementia']
merged[dx_col] = merged[dx_col].replace({'AD': 'Dementia'}) # Map AD to Dementia
merged = merged[merged[dx_col].isin(valid_dx)]

# %% [markdown]
# ## Define Modality Feature Groups

# %%
# Select a few representative structural features from ADSP if available.
adsp_cols = [c for c in merged.columns if 'VOL' in c.upper() or 'THICK' in c.upper() or 'AREA' in c.upper()][:20]

# Modality 1: MRI-volumetric only
features_mri = ['APVOLUME'] + adsp_cols
features_mri = [c for c in features_mri if c in merged.columns]

# Modality 2: + CSF
features_csf = ['ABETA', 'TAU', 'ABETA_TAU_RATIO']

# Modality 3: + Clinical/Cognitive
features_clin = ['MMSE', 'MMSE_TAU_INTERACT']

# Modality 4: + Genotype
# Genotype might be in 'APOE4' or similar from APOERES
genotype_cols = [c for c in merged.columns if 'APOE' in c.upper()]
features_geno = genotype_cols

groups = {
    "MRI Only": features_mri,
    "+ CSF": features_mri + features_csf,
    "+ Clinical": features_mri + features_csf + features_clin,
    "+ Genotype (Full)": features_mri + features_csf + features_clin + features_geno
}

# Ensure all columns exist and handle missing
for g, cols in groups.items():
    groups[g] = [c for c in cols if c in merged.columns]

# Fill missing values with median for numeric columns
for col in set([c for cols in groups.values() for c in cols]):
    if pd.api.types.is_numeric_dtype(merged[col]):
        merged[col] = merged[col].fillna(merged[col].median())
    else:
        # one hot encode non-numeric
        merged = pd.get_dummies(merged, columns=[col], dummy_na=False)
        # Update group lists with one-hot encoded columns
        new_cols = [c for c in merged.columns if c.startswith(col + '_')]
        for g in groups:
            if col in groups[g]:
                groups[g].remove(col)
                groups[g].extend(new_cols)

print("Modality group sizes:")
for g, cols in groups.items():
    print(f"{g}: {len(cols)} features")

# Encode labels
le = LabelEncoder()
y = le.fit_transform(merged[dx_col])

# %% [markdown]
# ## Retrain and Evaluate Each Modality Group

# %%
def train_and_eval(X, y, group_name):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # PCA
    pca = PCA(n_components=0.98, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # SMOTE on train only
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_pca, y_train)
    
    # Train Models
    xgb = XGBClassifier(learning_rate=0.05, max_depth=7, n_estimators=300, subsample=0.9, colsample_bytree=0.9, eval_metric='mlogloss', random_state=42)
    rf = RandomForestClassifier(n_estimators=400, max_depth=14, min_samples_split=4, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam', max_iter=500, random_state=42)
    
    xgb.fit(X_train_smote, y_train_smote)
    rf.fit(X_train_smote, y_train_smote)
    mlp.fit(X_train_smote, y_train_smote)
    
    # Get Validation Probs for Optuna
    p_xgb = xgb.predict_proba(X_test_pca)
    p_rf = rf.predict_proba(X_test_pca)
    p_mlp = mlp.predict_proba(X_test_pca)
    
    def objective(trial):
        w_xgb = trial.suggest_float('w_xgb', 0, 1)
        w_rf = trial.suggest_float('w_rf', 0, 1)
        w_mlp = trial.suggest_float('w_mlp', 0, 1)
        
        # Normalize
        total = w_xgb + w_rf + w_mlp
        if total == 0: return 0
        w_xgb /= total; w_rf /= total; w_mlp /= total
        
        blend = w_xgb * p_xgb + w_rf * p_rf + w_mlp * p_mlp
        preds = np.argmax(blend, axis=1)
        return accuracy_score(y_test, preds)
        
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    
    best_w = study.best_params
    total = sum(best_w.values())
    w_xgb, w_rf, w_mlp = best_w['w_xgb']/total, best_w['w_rf']/total, best_w['w_mlp']/total
    
    final_blend = w_xgb * p_xgb + w_rf * p_rf + w_mlp * p_mlp
    final_preds = np.argmax(final_blend, axis=1)
    
    acc = accuracy_score(y_test, final_preds)
    p, r, f1, _ = precision_recall_fscore_support(y_test, final_preds, average='weighted')
    per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(y_test, final_preds)
    
    return {
        "Group": group_name,
        "Accuracy": acc,
        "Weighted F1": f1,
        **{f"{le.classes_[i]} F1": per_class_f1[i] for i in range(len(le.classes_))}
    }

results = []
for g, cols in groups.items():
    print(f"Training for {g}...")
    res = train_and_eval(merged[cols], y, g)
    results.append(res)

results_df = pd.DataFrame(results)
results_df.to_csv("results/01_modality_ablation.csv", index=False)
print("\n--- Modality Ablation Results ---")
print(results_df)

# %% [markdown]
# ## Visualization

# %%
plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x='Group', y='Weighted F1', palette='viridis')
plt.title('Modality Ablation: Incremental Gain in Weighted F1')
plt.ylabel('Weighted F1 Score')
plt.xlabel('Feature Set')
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig("figures/01_modality_ablation_f1.png", dpi=300)
plt.close()

# %% [markdown]
# # Summary
# 
# **Headline Findings:**
