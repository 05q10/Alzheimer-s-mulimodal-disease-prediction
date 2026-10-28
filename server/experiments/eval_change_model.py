"""
server/experiments/eval_change_model.py
=======================================
Compare predicting LEVEL vs predicting CHANGE (delta = future - anchor)
for progression forecasting GBM.
Computes Test MAE, 95% Cluster-Bootstrap CIs, and paired differences.
"""

import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))

from ml.twin.constants import FEATURE_TIERS, FORBIDDEN_FEATURES, SPLIT_SEED
from experiments.exp10_progression import (
    build_anchors, cluster_bootstrap_mae_ci, cluster_bootstrap_paired_mae_diff
)

DATA_DIR = SERVER_DIR / "data"
PARQUET = DATA_DIR / "unified_visits.parquet"
SPLITS_FILE = DATA_DIR / "splits.json"
WEIGHTS_DIR = SERVER_DIR / "ml" / "twin" / "weights"

HORIZONS = [6, 12, 24]
TARGETS = ["MMSE", "CDGLOBAL", "TOTSCORE"]

def run_comparison():
    with open(SPLITS_FILE) as f:
        splits = json.load(f)
    train_sids = set(splits["train"])
    val_sids   = set(splits["val"])
    test_sids  = set(splits["test"])

    df = pd.read_parquet(PARQUET)
    df = df[df["exclude_time_axis"] == 0].copy()
    df = df.dropna(subset=["DIAGNOSIS"])
    if "MMSCORE" in df.columns:
        if "MMSE" in df.columns:
            df["MMSE"] = df["MMSCORE"].fillna(df["MMSE"])
        else:
            df["MMSE"] = df["MMSCORE"]
        df = df.drop(columns=["MMSCORE"])

    for col in TARGETS:
        if f"{col}_obs" not in df.columns:
            df[f"{col}_obs"] = df[col].notna().astype(int)

    df_train = df[df["subject_id"].isin(train_sids)].copy()
    df_val   = df[df["subject_id"].isin(val_sids)].copy()
    df_test  = df[df["subject_id"].isin(test_sids)].copy()

    results = []

    for h in HORIZONS:
        print(f"\n--- HORIZON {h}m ---")
        anc_train = build_anchors(df_train, h, include_mri=False)
        anc_val   = build_anchors(df_val,   h, include_mri=False)
        anc_test  = build_anchors(df_test,  h, include_mri=False)

        feat_cols = [c for c in anc_test.columns if c.startswith("feat_")]

        for target in TARGETS:
            y_col = f"future_{target}"
            anc_col = f"anchor_{target}"

            # Prepare train / val
            train = anc_train[feat_cols + [y_col, anc_col]].dropna(subset=[y_col, anc_col]).copy()
            val   = anc_val[feat_cols + [y_col, anc_col]].dropna(subset=[y_col, anc_col]).copy()
            test  = anc_test[feat_cols + [y_col, anc_col, "subject_id"]].dropna(subset=[y_col, anc_col]).copy()

            train["delta"] = train[y_col] - train[anc_col]
            val["delta"]   = val[y_col] - val[anc_col]
            test["delta"]  = test[y_col] - test[anc_col]

            X_tr = train[feat_cols].astype(float)
            X_va = val[feat_cols].astype(float)
            X_te = test[feat_cols].astype(float)

            y_tr_lvl = train[y_col].values
            y_va_lvl = val[y_col].values
            y_te_lvl = test[y_col].values

            y_tr_d = train["delta"].values
            y_va_d = val["delta"].values
            y_te_d = test["delta"].values

            sids_te = test["subject_id"].values
            anc_te = test[anc_col].values

            # Level model
            lvl_model = CatBoostRegressor(
                iterations=400, depth=5, learning_rate=0.05,
                loss_function="MAE", random_seed=SPLIT_SEED,
                verbose=0, early_stopping_rounds=40,
            )
            lvl_model.fit(X_tr, y_tr_lvl, eval_set=(X_va, y_va_lvl), verbose=False)
            pred_lvl = lvl_model.predict(X_te)
            mae_lvl = mean_absolute_error(y_te_lvl, pred_lvl)
            ci_lvl = cluster_bootstrap_mae_ci(y_te_lvl, pred_lvl, sids_te)

            # Change (delta) model
            # Note: Monotone constraints where clinically justifiable:
            # e.g., for MMSE delta, higher baseline FAQ or CDRSB should not INCREASE delta (i.e. more impairment -> less improvement or more decline)
            d_model = CatBoostRegressor(
                iterations=400, depth=5, learning_rate=0.05,
                loss_function="MAE", random_seed=SPLIT_SEED,
                verbose=0, early_stopping_rounds=40,
            )
            d_model.fit(X_tr, y_tr_d, eval_set=(X_va, y_va_d), verbose=False)
            pred_d = d_model.predict(X_te)
            pred_from_d = anc_te + pred_d
            if target == "MMSE":
                pred_from_d = np.clip(pred_from_d, 0.0, 30.0)
            elif target == "CDGLOBAL":
                pred_from_d = np.clip(pred_from_d, 0.0, 3.0)
            elif target == "TOTSCORE":
                pred_from_d = np.clip(pred_from_d, 0.0, 70.0)

            mae_d = mean_absolute_error(y_te_lvl, pred_from_d)
            ci_d = cluster_bootstrap_mae_ci(y_te_lvl, pred_from_d, sids_te)

            orig_diff, diff_lo, diff_hi = cluster_bootstrap_paired_mae_diff(
                y_te_lvl, pred_lvl, pred_from_d, sids_te
            )

            res = {
                "horizon": h,
                "target": target,
                "mae_level": round(float(mae_lvl), 3),
                "ci_level": ci_lvl,
                "mae_delta": round(float(mae_d), 3),
                "ci_delta": ci_d,
                "diff_mae": round(orig_diff, 4),
                "diff_ci": [round(diff_lo, 4), round(diff_hi, 4)],
            }
            results.append(res)
            print(f"{target}@{h}m: Level MAE = {mae_lvl:.3f} {ci_lvl} | Delta MAE = {mae_d:.3f} {ci_d} | Diff = {orig_diff:+.4f} [{diff_lo:+.4f}, {diff_hi:+.4f}]")

            # Save the change model if good
            d_model.save_model(str(WEIGHTS_DIR / f"progression_gbm_delta_{target}_{h}m.cbm"))

    res_df = pd.DataFrame(results)
    print("\nSUMMARY:")
    print(res_df.to_string())

if __name__ == "__main__":
    run_comparison()
