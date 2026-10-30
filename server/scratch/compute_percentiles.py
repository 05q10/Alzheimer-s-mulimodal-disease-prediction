import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
import sys

scratch_dir = Path(__file__).resolve().parent
server_dir = scratch_dir.parent
sys.path.insert(0, str(scratch_dir))
sys.path.insert(0, str(server_dir))

from ml.twin.stager import get_stager
from ml.twin.progression import get_progression
from check_eligibility import eligible_subjects, trajectories

print("Loading splits and unified visits...", flush=True)
with open("server/data/splits.json") as f:
    splits = json.load(f)

test_subjects = set(splits["test"])
df = pd.read_parquet("server/data/unified_visits.parquet")

stager = get_stager()
prog = get_progression()

test_df = df[df["subject_id"].isin(test_subjects)].copy()

all_anchor_errors = {
    "MMSE": {"6": [], "12": [], "24": []},
    "CDGLOBAL": {"6": [], "12": [], "24": []},
    "TOTSCORE": {"6": [], "12": [], "24": []},
}

print("Computing test anchor predictions for error distributions...", flush=True)
for sid, grp in test_df.groupby("subject_id"):
    v_sorted = grp.sort_values("visit_month")
    row0 = v_sorted.iloc[0]
    sex = float(row0.get("sex") if pd.notna(row0.get("sex")) else row0.get("PTGENDER", 0))
    edu = float(row0.get("education_years") if pd.notna(row0.get("education_years")) else row0.get("PTEDUCAT", 14))
    apoe = float(row0.get("APOE4", 0) if pd.notna(row0.get("APOE4")) else 0)

    for i, anchor in v_sorted.iterrows():
        a_m = anchor["visit_month"]
        if pd.isna(anchor["MMSE"]) or pd.isna(anchor["CDGLOBAL"]):
            continue

        inputs = {
            "entry_age": float(anchor.get("age_at_visit") or anchor.get("AGE") or 70.0),
            "sex": sex,
            "education_years": edu,
            "apoe4_count": apoe,
            "MMSE": float(anchor["MMSE"]),
            "CDGLOBAL": float(anchor["CDGLOBAL"]),
        }
        if pd.notna(anchor.get("TOTSCORE")):
            inputs["TOTSCORE"] = float(anchor["TOTSCORE"])
        if pd.notna(anchor.get("CDRSB")):
            inputs["CDRSB"] = float(anchor["CDRSB"])
        if pd.notna(anchor.get("MOCA")):
            inputs["MOCA"] = float(anchor["MOCA"])
        if pd.notna(anchor.get("FAQTOTAL")):
            inputs["FAQTOTAL"] = float(anchor["FAQTOTAL"])

        s_res = stager.predict_one(inputs)
        probs_dict = s_res["probs"]
        fc = prog.predict_forecast(inputs, probs_dict)

        for h_str, h_int in [("6", 6), ("12", 12), ("24", 24)]:
            fut = v_sorted[v_sorted["visit_month"].between(a_m + h_int - 1.5, a_m + h_int + 1.5)]
            if len(fut) > 0:
                fut_row = fut.iloc[0]
                if pd.notna(fut_row.get("MMSE")) and fc.get(h_str, {}).get("MMSE_mean") is not None:
                    err = abs(float(fut_row["MMSE"]) - float(fc[h_str]["MMSE_mean"]))
                    all_anchor_errors["MMSE"][h_str].append((sid, a_m, err))
                if pd.notna(fut_row.get("CDGLOBAL")) and fc.get(h_str, {}).get("CDGLOBAL_mean") is not None:
                    err = abs(float(fut_row["CDGLOBAL"]) - float(fc[h_str]["CDGLOBAL_mean"]))
                    all_anchor_errors["CDGLOBAL"][h_str].append((sid, a_m, err))
                if pd.notna(fut_row.get("TOTSCORE")) and fc.get(h_str, {}).get("TOTSCORE_mean") is not None:
                    err = abs(float(fut_row["TOTSCORE"]) - float(fc[h_str]["TOTSCORE_mean"]))
                    all_anchor_errors["TOTSCORE"][h_str].append((sid, a_m, err))

print("\nError distributions computed:", flush=True)
for tgt in ["MMSE", "CDGLOBAL", "TOTSCORE"]:
    for h in ["6", "12", "24"]:
        arr = [x[2] for x in all_anchor_errors[tgt][h]]
        if arr:
            p50 = np.median(arr)
            p75 = np.percentile(arr, 75)
            p90 = np.percentile(arr, 90)
            print(f"  {tgt} {h}m (N={len(arr)}): Mean MAE={np.mean(arr):.2f}, Median={p50:.2f}, 75th={p75:.2f}, 90th={p90:.2f}", flush=True)

chosen_ids = {
    "A (Stable CN)": "022_S_4291",
    "B (CN -> MCI)": "031_S_4218",
    "C (MCI -> AD)": "035_S_0997",
    "D (Stable MCI)": "022_S_4444",
}

print("\n=== SELECTED PATIENTS ERROR PERCENTILES ===", flush=True)
for label, sid in chosen_ids.items():
    print(f"\n--- {label} (Subject {sid}) ---", flush=True)
    for tgt in ["MMSE", "CDGLOBAL", "TOTSCORE"]:
        for h in ["6", "12", "24"]:
            pop_errors = [x[2] for x in all_anchor_errors[tgt][h]]
            pat_errors = [x[2] for x in all_anchor_errors[tgt][h] if x[0] == sid]
            if pat_errors and pop_errors:
                pe = pat_errors[0]
                pct = (np.array(pop_errors) < pe).mean() * 100.0
                print(f"  {tgt} at {h}m: Abs Error = {pe:.2f} (Percentile: {pct:.1f}%)", flush=True)

# Search for Hard Case candidates
print("\n=== HARD CASE CANDIDATES (>75th pct MMSE error) ===", flush=True)
hard_candidates = []
for sid, grp in eligible_subjects:
    pat_12m = [x[2] for x in all_anchor_errors["MMSE"]["12"] if x[0] == sid and x[1] == 0]
    pop_12m = [x[2] for x in all_anchor_errors["MMSE"]["12"]]
    if pat_12m and pop_12m:
        pe12 = pat_12m[0]
        pct12 = (np.array(pop_12m) < pe12).mean() * 100.0
        if pct12 > 75.0:
            dxs = [int(grp[m]["DIAGNOSIS"]) for m in [0, 6, 12, 24]]
            hard_candidates.append((sid, pct12, pe12, dxs))

hard_candidates.sort(key=lambda x: -x[1])
for sid, pct, pe, dxs in hard_candidates[:5]:
    print(f"  Subject: {sid} | 12m MMSE Error: {pe:.2f} ({pct:.1f}th pct) | Trajectory DX: {dxs}", flush=True)
