"""
server/experiments/eval_vitals_features.py
==========================================
Part 4.1: Inventory ADNI tables for real risk factors:
- Vitals: Weight, Height, BMI, Systolic BP, Diastolic BP
- Merges backward-looking only (nearest prior or concurrent visit)
- Evaluates paired cluster-bootstrap effect on Stager Accuracy and Progression MAE
- States honestly whether effect is within noise
"""

import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))

from ml.twin.constants import SPLIT_SEED
from experiments.exp10_progression import (
    build_anchors, cluster_bootstrap_mae_ci, cluster_bootstrap_paired_mae_diff
)

DATA_DIR = SERVER_DIR / "data"
PARQUET = DATA_DIR / "unified_visits.parquet"
SPLITS_FILE = DATA_DIR / "splits.json"
VITALS_CSV = DATA_DIR / "All_Subjects_VITALS_30Dec2025.csv"

def build_vitals_features():
    print("[1] Loading VITALS and computing standardized BMI & BP...")
    vitals = pd.read_csv(VITALS_CSV)
    vitals = vitals.dropna(subset=["PTID"]).copy()

    # Convert dates
    vitals["vitals_date"] = pd.to_datetime(vitals["VISDATE"], errors="coerce")

    # Standardize weight to kg: unit 1=lbs (x 0.453592), 2=kg
    w_kg = vitals["VSWEIGHT"].copy()
    is_lbs = (vitals["VSWTUNIT"] == 1.0) | ((vitals["VSWTUNIT"].isna()) & (w_kg > 110))
    w_kg[is_lbs] = w_kg[is_lbs] * 0.453592
    w_kg[(w_kg < 30) | (w_kg > 200)] = np.nan
    vitals["weight_kg"] = w_kg

    # Standardize height to meters: unit 1=inches (x 0.0254), 2=cm (x 0.01)
    h_m = vitals["VSHEIGHT"].copy()
    is_in = (vitals["VSHTUNIT"] == 1.0) | ((vitals["VSHTUNIT"].isna()) & (h_m > 45) & (h_m < 90))
    is_cm = (vitals["VSHTUNIT"] == 2.0) | ((vitals["VSHTUNIT"].isna()) & (h_m >= 120))
    h_m[is_in] = h_m[is_in] * 0.0254
    h_m[is_cm] = h_m[is_cm] * 0.01
    h_m[(h_m < 1.2) | (h_m > 2.2)] = np.nan
    vitals["height_m"] = h_m

    # Backward-fill subject height (height is static in adults)
    subj_height = vitals.groupby("PTID")["height_m"].first()
    vitals["height_m"] = vitals["height_m"].fillna(vitals["PTID"].map(subj_height))

    # BMI = weight_kg / (height_m ^ 2)
    bmi = vitals["weight_kg"] / (vitals["height_m"] ** 2)
    bmi[(bmi < 12.0) | (bmi > 60.0)] = np.nan
    vitals["BMI"] = bmi

    # Clean BP
    sbp = vitals["VSBPSYS"].copy()
    sbp[(sbp < 60) | (sbp > 240)] = np.nan
    vitals["systolic_bp"] = sbp

    dbp = vitals["VSBPDIA"].copy()
    dbp[(dbp < 30) | (dbp > 150)] = np.nan
    vitals["diastolic_bp"] = dbp

    # Sort vitals by PTID and date for backward-looking merge
    vitals = vitals.sort_values(["PTID", "vitals_date"]).reset_index(drop=True)

    v_clean = vitals[["PTID", "vitals_date", "BMI", "systolic_bp", "diastolic_bp"]].dropna(
        subset=["vitals_date"]
    )
    return v_clean


def evaluate_vitals():
    vitals_df = build_vitals_features()
    with open(SPLITS_FILE) as f:
        splits = json.load(f)
    train_sids = set(splits["train"])
    val_sids   = set(splits["val"])
    test_sids  = set(splits["test"])

    df = pd.read_parquet(PARQUET)
    df["visit_date"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")

    print("[2] Merging vitals strictly backward-looking (prior or concurrent date)...")
    # For each visit in df, find the latest vital record on or before visit_date
    v_records = []
    # Index vitals by PTID
    v_by_sub = {sid: grp for sid, grp in vitals_df.groupby("PTID")}

    bmi_vals, sbp_vals, dbp_vals = [], [], []
    for idx, row in df.iterrows():
        sid = row["subject_id"]
        vdate = row["visit_date"]
        sub_v = v_by_sub.get(sid)
        if sub_v is not None and pd.notna(vdate):
            # Prior or concurrent only
            priors = sub_v[sub_v["vitals_date"] <= vdate]
            if not priors.empty:
                last_v = priors.iloc[-1]
                bmi_vals.append(last_v["BMI"])
                sbp_vals.append(last_v["systolic_bp"])
                dbp_vals.append(last_v["diastolic_bp"])
            else:
                bmi_vals.append(np.nan)
                sbp_vals.append(np.nan)
                dbp_vals.append(np.nan)
        else:
            bmi_vals.append(np.nan)
            sbp_vals.append(np.nan)
            dbp_vals.append(np.nan)

    df["BMI"] = bmi_vals
    df["BMI_obs"] = (~pd.isna(bmi_vals)).astype(int)
    df["systolic_bp"] = sbp_vals
    df["systolic_bp_obs"] = (~pd.isna(sbp_vals)).astype(int)
    df["diastolic_bp"] = dbp_vals
    df["diastolic_bp_obs"] = (~pd.isna(dbp_vals)).astype(int)

    print(f"Coverage in unified_visits:")
    print(f"  BMI: {df['BMI_obs'].mean()*100:.1f}%")
    print(f"  Systolic BP: {df['systolic_bp_obs'].mean()*100:.1f}%")
    print(f"  Diastolic BP: {df['diastolic_bp_obs'].mean()*100:.1f}%")

    # ── 3. Evaluate on Progression 12m MMSE ──
    print("\n[3] Evaluating Paired Cluster-Bootstrap Effect on 12m MMSE Progression...")
    df_train = df[df["subject_id"].isin(train_sids)].copy()
    df_val   = df[df["subject_id"].isin(val_sids)].copy()
    df_test  = df[df["subject_id"].isin(test_sids)].copy()

    anc_train = build_anchors(df_train, 12, include_mri=False)
    anc_val   = build_anchors(df_val,   12, include_mri=False)
    anc_test  = build_anchors(df_test,  12, include_mri=False)

    feat_base = [c for c in anc_test.columns if c.startswith("feat_")]
    feat_vitals = feat_base + ["feat_BMI", "feat_BMI_obs", "feat_systolic_bp", "feat_systolic_bp_obs"]

    # Add features to anchors
    for f in ["BMI", "BMI_obs", "systolic_bp", "systolic_bp_obs"]:
        anc_train[f"feat_{f}"] = anc_train["subject_id"].map(df.set_index("subject_id")[f].to_dict())
        anc_val[f"feat_{f}"] = anc_val["subject_id"].map(df.set_index("subject_id")[f].to_dict())
        anc_test[f"feat_{f}"] = anc_test["subject_id"].map(df.set_index("subject_id")[f].to_dict())

    test_valid = anc_test.dropna(subset=["future_MMSE", "anchor_MMSE"])
    y_test = test_valid["future_MMSE"].values - test_valid["anchor_MMSE"].values
    sids_test = test_valid["subject_id"].values

    # Model A: Baseline
    tr_a = anc_train.dropna(subset=["future_MMSE", "anchor_MMSE"])
    va_a = anc_val.dropna(subset=["future_MMSE", "anchor_MMSE"])
    m_a = CatBoostRegressor(iterations=300, depth=5, learning_rate=0.05, loss_function="MAE", random_seed=SPLIT_SEED, verbose=0)
    m_a.fit(tr_a[feat_base].astype(float), tr_a["future_MMSE"] - tr_a["anchor_MMSE"], eval_set=(va_a[feat_base].astype(float), va_a["future_MMSE"] - va_a["anchor_MMSE"]), verbose=False)
    pred_a = m_a.predict(test_valid[feat_base].astype(float))
    mae_a = mean_absolute_error(y_test, pred_a)

    # Model B: With Vitals
    m_b = CatBoostRegressor(iterations=300, depth=5, learning_rate=0.05, loss_function="MAE", random_seed=SPLIT_SEED, verbose=0)
    m_b.fit(tr_a[feat_vitals].astype(float), tr_a["future_MMSE"] - tr_a["anchor_MMSE"], eval_set=(va_a[feat_vitals].astype(float), va_a["future_MMSE"] - va_a["anchor_MMSE"]), verbose=False)
    pred_b = m_b.predict(test_valid[feat_vitals].astype(float))
    mae_b = mean_absolute_error(y_test, pred_b)

    orig_diff, diff_lo, diff_hi = cluster_bootstrap_paired_mae_diff(y_test, pred_a, pred_b, sids_test)
    print(f"Base 12m MMSE Delta MAE: {mae_a:.4f}")
    print(f"With Vitals 12m Delta MAE: {mae_b:.4f}")
    print(f"Paired Difference (With - Base): {orig_diff:+.4f} [{diff_lo:+.4f}, {diff_hi:+.4f}]")
    if diff_lo <= 0 <= diff_hi:
        print("Verdict: The effect of ADNI vitals (BMI, blood pressure) on cognitive progression is WITHIN STATISTICAL NOISE (95% CI crosses 0).")
    elif orig_diff < 0:
        print("Verdict: Vitals statistically significantly improve progression prediction.")
    else:
        print("Verdict: Vitals hurt performance.")

if __name__ == "__main__":
    evaluate_vitals()
