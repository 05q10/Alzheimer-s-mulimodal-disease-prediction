"""
server/experiments/build_plausibility_envelope.py
=================================================
Computes empirical envelope of changes by stage and anchor score tercile from TRAIN split only.
Targets: MMSE, CDGLOBAL, TOTSCORE
Horizons: 6, 12, 24 months
Outputs: server/ml/twin/weights/plausibility_envelope.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

SERVER_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SERVER_DIR / "data"
WEIGHTS_DIR = SERVER_DIR / "ml" / "twin" / "weights"

PARQUET_FILE = DATA_DIR / "unified_visits.parquet"
SPLITS_FILE = DATA_DIR / "splits.json"

DX_MAP = {1: "CN", 2: "MCI", 3: "AD"}
TARGETS = ["MMSE", "CDGLOBAL", "TOTSCORE"]
HORIZONS = [6, 12, 24]


def compute_envelope():
    print("Loading splits and training data...")
    with open(SPLITS_FILE) as f:
        splits = json.load(f)
    train_sids = set(splits["train"])

    df = pd.read_parquet(PARQUET_FILE)
    df_train = df[df["subject_id"].isin(train_sids)].copy()
    print(f"Train set: {len(df_train)} visits across {len(train_sids)} subjects.")

    # 1. Compute 5th-95th percentile training ranges
    input_ranges = {}
    for col in ["entry_age", "MMSE", "CDGLOBAL", "TOTSCORE", "FAQTOTAL"]:
        if col in df_train.columns:
            vals = df_train[col].dropna()
            p5, p95 = np.percentile(vals, [5, 95])
            name = "age" if col == "entry_age" else col
            input_ranges[name] = [round(float(p5), 1), round(float(p95), 1)]

    # 2. Build pairs for each horizon on train
    envelope = {
        "horizons": {},
        "training_input_ranges": input_ranges,
    }

    for h in HORIZONS:
        envelope["horizons"][str(h)] = {}
        records = []
        for sid, grp in df_train.groupby("subject_id"):
            grp = grp.sort_values("visit_month").reset_index(drop=True)
            for i, anchor in grp.iterrows():
                target_m = anchor["visit_month"] + h
                diffs = (grp["visit_month"] - target_m).abs()
                cands = grp[(grp["visit_month"] > anchor["visit_month"]) & (diffs <= 3)]
                if cands.empty:
                    continue
                future = cands.loc[diffs[cands.index].idxmin()]
                stage = DX_MAP.get(int(anchor.get("DIAGNOSIS", 0)), None)
                if not stage:
                    continue

                rec = {
                    "stage": stage,
                    "anchor_MMSE": anchor.get("MMSE"),
                    "future_MMSE": future.get("MMSE"),
                    "anchor_CDGLOBAL": anchor.get("CDGLOBAL"),
                    "future_CDGLOBAL": future.get("CDGLOBAL"),
                    "anchor_TOTSCORE": anchor.get("TOTSCORE"),
                    "future_TOTSCORE": future.get("TOTSCORE"),
                }
                records.append(rec)

        hdf = pd.DataFrame(records)

        for target in TARGETS:
            anchor_col = f"anchor_{target}"
            future_col = f"future_{target}"
            valid = hdf.dropna(subset=[anchor_col, future_col]).copy()
            valid["delta"] = valid[future_col] - valid[anchor_col]

            # Overall percentiles
            p5, p25, p50, p75, p95 = np.percentile(valid["delta"], [5, 25, 50, 75, 95])
            overall_pcts = {
                "p5": round(float(p5), 2),
                "p25": round(float(p25), 2),
                "p50": round(float(p50), 2),
                "p75": round(float(p75), 2),
                "p95": round(float(p95), 2),
            }

            # By stage percentiles
            by_stage = {}
            for stg in ["CN", "MCI", "AD"]:
                stg_df = valid[valid["stage"] == stg]
                if len(stg_df) >= 10:
                    s_p5, s_p25, s_p50, s_p75, s_p95 = np.percentile(stg_df["delta"], [5, 25, 50, 75, 95])
                    by_stage[stg] = {
                        "p5": round(float(s_p5), 2),
                        "p25": round(float(s_p25), 2),
                        "p50": round(float(s_p50), 2),
                        "p75": round(float(s_p75), 2),
                        "p95": round(float(s_p95), 2),
                    }
                else:
                    by_stage[stg] = dict(overall_pcts)

            # By tercile within target
            by_tercile = {}
            try:
                valid["tercile"] = pd.qcut(valid[anchor_col], q=3, labels=["low", "mid", "high"], duplicates="drop")
                for terc in ["low", "mid", "high"]:
                    tdf = valid[valid["tercile"] == terc]
                    if len(tdf) >= 10:
                        t_p5, t_p25, t_p50, t_p75, t_p95 = np.percentile(tdf["delta"], [5, 25, 50, 75, 95])
                        by_tercile[terc] = {
                            "p5": round(float(t_p5), 2),
                            "p25": round(float(t_p25), 2),
                            "p50": round(float(t_p50), 2),
                            "p75": round(float(t_p75), 2),
                            "p95": round(float(t_p95), 2),
                        }
            except Exception:
                pass

            envelope["horizons"][str(h)][target] = {
                "overall": overall_pcts,
                "by_stage": by_stage,
                "by_tercile": by_tercile,
            }

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = WEIGHTS_DIR / "plausibility_envelope.json"
    with open(out_path, "w") as f:
        json.dump(envelope, f, indent=2)

    print(f"Plausibility envelope saved to {out_path}")
    return envelope


if __name__ == "__main__":
    compute_envelope()
