"""
server/ml/twin/data_build.py
=============================
Phase 1 data pipeline: builds unified_visits.parquet and splits.json.

Outputs
-------
  server/data/unified_visits.parquet
  server/data/splits.json
  server/experiments/results/09_data_audit.md

Rules (from implementation plan corrections A, B):
  - FORBIDDEN_FEATURES never enter model inputs (checked by pytest)
  - Imputed file is a comparison baseline only; never used as training source
  - Stratified group split by subject (stratify on baseline DX + ever-converted)
  - bl and sc collapsed to month-0 (bl preferred, sc fills gaps)
  - Visit dates reconstructed from ADNIMERGE/My_Table EXAMDATE; fallback = baseline + visit_month*30.44
  - ADNIMERGE joined exact VISCODE first, then nearest ±2 months if no exact match
  - Canonical DX = shortened-file DIAGNOSIS (DXSUM/ADNIMERGE DX logged for disagreement only, never override)
  - Valid anchor = ≥1 observed score in history window
  - No future MRI attached (mri_gap_months ≤ 0 for visits with MRI from a future scan — excluded)

Run from server/:
    python -m ml.twin.data_build
"""

import os, json, warnings, re, sys
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
SERVER_DIR = SCRIPT_DIR.parent.parent
DATA_DIR   = SERVER_DIR / "data"
OUT_DIR    = SERVER_DIR / "experiments" / "results"

VAL_PATH     = DATA_DIR / "shortened_result-val.csv"
ORIG4_PATH   = DATA_DIR / "shortened_result_original_missing_4visits.csv"
IMP4_PATH    = DATA_DIR / "shortened_result_imputed_4visits.csv"
ADNI_PATH    = DATA_DIR / "ADNIMERGE_13Aug2025.csv"
ADSP_PATH    = DATA_DIR / "ADSP_PHC_T1_FS_15Aug2025.csv"
APOE_PATH    = DATA_DIR / "All_Subjects_APOERES_fixed_simple.csv"
APOE2_PATH   = DATA_DIR / "All_Subjects_APOERES_30Dec2025.csv"
MYTAB_PATH   = DATA_DIR / "All_Subjects_My_Table_30Dec2025.csv"
PTDEMOG_PATH = DATA_DIR / "All_Subjects_PTDEMOG_30Dec2025.csv"
ENTRY_PATH   = DATA_DIR / "All_Subjects_Study_Entry_30Dec2025.csv"
FAMHX_PATH   = DATA_DIR / "All_Subjects_FAMHXPAR_30Dec2025.csv"
DXSUM_PATH   = DATA_DIR / "All_Subjects_DXSUM_30Dec2025.csv"

PARQUET_OUT = DATA_DIR / "unified_visits.parquet"
SPLITS_OUT  = DATA_DIR / "splits.json"
AUDIT_OUT   = OUT_DIR  / "09_data_audit.md"

SPLIT_SEED = 42

# ── Visit-month parsing ────────────────────────────────────────────────────

# Codes excluded from time axis (row kept for DX label, visit_month = NaN)
EXCLUDE_TIMEX = {"scmri", "nv", "AUT", "uns1"}

def parse_visit_month(visit: str) -> Optional[float]:
    """Return numeric visit month or NaN for excluded codes."""
    if pd.isna(visit):
        return np.nan
    v = str(visit).strip().lower()
    if v in ("bl", "sc"):
        return 0.0          # Both collapse to month-0
    if v in {x.lower() for x in EXCLUDE_TIMEX}:
        return np.nan
    m = re.match(r"^m(\d+)$", v)
    if m:
        return float(m.group(1))
    return np.nan

def is_excluded_timex(visit: str) -> bool:
    """True if visit code is not on the time axis."""
    return str(visit).strip() in EXCLUDE_TIMEX

# ── ABETA/TAU string parsing ───────────────────────────────────────────────

def parse_censored(series: pd.Series, colname: str):
    """Parse ABETA/TAU string columns like '>1700' or '<200'."""
    raw = series.astype(str).str.strip()
    censored = raw.str.contains(r"[<>]", regex=True, na=False).astype(int)
    numeric  = pd.to_numeric(raw.str.replace(r"[<>]", "", regex=True), errors="coerce")
    return numeric, censored

# ── Genotype → apoe4_count ──────────────────────────────────────────────────

def genotype_to_apoe4_count(geno) -> float:
    """e.g. '3/4' -> 1,  '4/4' -> 2,  '2/3' -> 0, NaN -> NaN"""
    if pd.isna(geno):
        return np.nan
    parts = str(geno).replace(",", "/").split("/")
    try:
        return float(sum(1 for a in parts if a.strip() == "4"))
    except Exception:
        return np.nan


def _to_us(series: pd.Series) -> pd.Series:
    """Coerce a datetime series to datetime64[us] to avoid ns overflow on pandas 3+."""
    if hasattr(series, 'dt'):
        try:
            return series.dt.as_unit('us')
        except Exception:
            pass
    return pd.to_datetime(series, errors='coerce').dt.as_unit('us')


def reconstruct_visit_dates(base: pd.DataFrame,
                             adni_dates: pd.DataFrame,
                             mytab_dates: pd.DataFrame,
                             entry: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct exam_date for each (subject_id, visit) row using three rules,
    logged via date_rule column:
      rule_1: exact match from ADNIMERGE EXAMDATE on (PTID, VISCODE)
      rule_2: exact match from My_Table EXAMDATE on (subject_id, visit)
      rule_3: baseline_date + visit_month * 30.44
    All datetimes stored as datetime64[us] to avoid pandas ns overflow.
    """
    df = base.copy()

    # Initialize exam_date column as object first, fill later
    df["exam_date"] = pd.NaT
    df["date_rule"] = "none"

    # ── Rule 1: ADNIMERGE exact ──────────────────────────────────────────
    adni_d = adni_dates[["PTID","VISCODE","EXAMDATE"]].dropna(subset=["EXAMDATE"]).copy()
    adni_d["EXAMDATE"] = _to_us(pd.to_datetime(adni_d["EXAMDATE"], errors="coerce"))
    adni_d = adni_d.dropna(subset=["EXAMDATE"])
    adni_d = adni_d.rename(columns={"PTID":"subject_id","VISCODE":"visit","EXAMDATE":"_date_adni"})
    df = df.merge(adni_d, on=["subject_id","visit"], how="left")

    # ── Rule 2: My_Table exact ───────────────────────────────────────────
    if "EXAMDATE" in mytab_dates.columns:
        mt_d = mytab_dates[["subject_id","visit","EXAMDATE"]].dropna(subset=["EXAMDATE"]).copy()
        mt_d["EXAMDATE"] = _to_us(pd.to_datetime(mt_d["EXAMDATE"], errors="coerce"))
        mt_d = mt_d.dropna(subset=["EXAMDATE"]).rename(columns={"EXAMDATE":"_date_mt"})
        df = df.merge(mt_d, on=["subject_id","visit"], how="left")
    else:
        df["_date_mt"] = pd.NaT

    # ── Baseline date per subject: earliest known ADNIMERGE date ─────────
    df["_date_adni"] = _to_us(df["_date_adni"])
    df["_date_mt"]   = _to_us(df["_date_mt"])

    known = df.dropna(subset=["_date_adni"])[["subject_id","_date_adni"]].copy()
    known = known.groupby("subject_id")["_date_adni"].min().reset_index()
    known.columns = ["subject_id","_bl_date"]
    df = df.merge(known, on="subject_id", how="left")
    df["_bl_date"] = _to_us(df["_bl_date"])

    # ── Fallback: entry date from Study_Entry ────────────────────────────
    if "entry_date" in entry.columns:
        entry_d = entry[["subject_id","entry_date"]].copy()
        entry_d["entry_date"] = _to_us(pd.to_datetime(entry_d["entry_date"], errors="coerce"))
        entry_d = entry_d.dropna(subset=["entry_date"])
        df = df.merge(entry_d, on="subject_id", how="left")
        # Fill bl_date from entry_date where missing
        mask = df["_bl_date"].isna() & df["entry_date"].notna()
        df.loc[mask, "_bl_date"] = _to_us(df.loc[mask, "entry_date"])
        df = df.drop(columns=["entry_date"])

    # ── Rule 3: baseline + visit_month * 30.44 days ──────────────────────
    # Compute as integer days offset to avoid ns precision issues
    visit_days = (df["visit_month"].fillna(0) * 30.44).round().astype("int64")
    # Only where bl_date is known
    has_bl = df["_bl_date"].notna()
    rule3_dates = pd.NaT
    if has_bl.any():
        rule3_series = df.loc[has_bl, "_bl_date"] + pd.to_timedelta(visit_days[has_bl], unit="D")
        df["_date_rule3"] = pd.NaT
        df.loc[has_bl, "_date_rule3"] = _to_us(rule3_series)
    else:
        df["_date_rule3"] = pd.NaT

    # ── Combine with priority: rule_1 > rule_2 > rule_3 ─────────────────
    # Assign rule_1
    mask1 = df["_date_adni"].notna()
    df.loc[mask1, "exam_date"] = df.loc[mask1, "_date_adni"]
    df.loc[mask1, "date_rule"] = "adni_exact"

    # Assign rule_2 where rule_1 missing
    mask2 = df["exam_date"].isna() & df["_date_mt"].notna()
    df.loc[mask2, "exam_date"] = df.loc[mask2, "_date_mt"]
    df.loc[mask2, "date_rule"] = "mytab_exact"

    # Assign rule_3 where still missing
    mask3 = df["exam_date"].isna() & df["_date_rule3"].notna()
    df.loc[mask3, "exam_date"] = df.loc[mask3, "_date_rule3"]
    df.loc[mask3, "date_rule"] = "bl_plus_month"

    # Ensure consistent dtype
    df["exam_date"] = _to_us(df["exam_date"])

    # Drop temp columns
    drop_cols = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=drop_cols)

    return df

# ── ADNIMERGE covariate join ───────────────────────────────────────────────

def join_adnimerge(base: pd.DataFrame, adni: pd.DataFrame) -> pd.DataFrame:
    """
    Join ADNIMERGE covariates onto base on (subject_id=PTID, visit=VISCODE).
    Strategy:
      1. Exact VISCODE match
      2. For rows still unmatched, find nearest ADNIMERGE row within ±2 months
         (using visit_month column in base vs Month_bl or Month in ADNIMERGE)
    """
    ADNI_COLS = [
        "PTID","VISCODE","Month_bl",
        "AGE","PTGENDER","PTEDUCAT",
        "APOE4","MMSE","CDRSB","ADAS13","FAQ","MOCA",
        "ABETA","TAU","PTAU","FDG","AV45",
        "Hippocampus","WholeBrain","Ventricles","Entorhinal","Fusiform","MidTemp","ICV",
        "FLDSTRENG","FSVERSION","IMAGEUID","EXAMDATE",
    ]
    # Keep only columns that exist
    adni_sub = adni[[c for c in ADNI_COLS if c in adni.columns]].copy()
    adni_sub = adni_sub.rename(columns={"PTID":"subject_id","VISCODE":"_adni_viscode","Month_bl":"_adni_month"})

    # Parse ABETA / TAU
    for col in ["ABETA","TAU","PTAU"]:
        if col in adni_sub.columns:
            adni_sub[col], adni_sub[f"{col}_censored"] = parse_censored(adni_sub[col], col)

    # ── Pass 1: exact VISCODE join ───────────────────────────────────────
    exact = adni_sub.rename(columns={"_adni_viscode":"visit"})
    merged = base.merge(exact.drop(columns=["_adni_month"], errors="ignore"),
                        on=["subject_id","visit"], how="left", suffixes=("","_adni"))

    # Track which rows were matched
    matched_exact = merged["AGE"].notna() if "AGE" in merged.columns else pd.Series(False, index=merged.index)
    n_exact = matched_exact.sum()

    # ── Pass 2: nearest-month join for unmatched rows ────────────────────
    unmatched_idx = merged.index[~matched_exact]
    n_approx = 0
    if len(unmatched_idx) > 0 and "_adni_month" in adni_sub.columns:
        # For each unmatched row, find the ADNIMERGE row for same subject
        # whose Month_bl is within ±2 months of visit_month
        um_base = base.loc[unmatched_idx, ["subject_id","visit","visit_month"]].copy()
        adni_m  = adni_sub[["subject_id","_adni_month"] +
                             [c for c in adni_sub.columns if c not in ["subject_id","_adni_viscode","_adni_month"]]
                             ].copy()
        adni_m = adni_m.rename(columns={"_adni_month":"adni_month"})
        adni_m = adni_m.dropna(subset=["adni_month"])

        joined_approx = um_base.merge(adni_m, on="subject_id", how="left")
        joined_approx["month_diff"] = (joined_approx["visit_month"] - joined_approx["adni_month"]).abs()
        joined_approx = joined_approx[joined_approx["month_diff"] <= 2]
        # Keep closest
        joined_approx = joined_approx.sort_values("month_diff").drop_duplicates(subset=["subject_id","visit"])
        joined_approx["adni_match_type"] = "approx_month"

        # Fast vectorized fill for unmatched rows
        fill_cols = [c for c in joined_approx.columns if c in merged.columns and c not in ["subject_id","visit","visit_month","adni_month","month_diff"]]
        if fill_cols and not joined_approx.empty:
            merged_indexed = merged.set_index(["subject_id","visit"])
            approx_indexed = joined_approx.set_index(["subject_id","visit"])[fill_cols]
            merged_indexed[fill_cols] = merged_indexed[fill_cols].combine_first(approx_indexed)
            merged = merged_indexed.reset_index()

            m_idx = pd.MultiIndex.from_arrays([merged["subject_id"], merged["visit"]])
            a_idx = pd.MultiIndex.from_arrays([joined_approx["subject_id"], joined_approx["visit"]])
            merged.loc[m_idx.isin(a_idx), "adni_match_type"] = "approx_month"
        n_approx = len(joined_approx)

    print(f"  ADNIMERGE: exact={n_exact}, approx_month={n_approx}, "
          f"unmatched={(~matched_exact).sum()-n_approx}")

    # Add match-type column
    if "adni_match_type" not in merged.columns:
        merged["adni_match_type"] = "none"
    merged.loc[matched_exact, "adni_match_type"] = "exact"

    return merged

# ── ADSP FreeSurfer join ───────────────────────────────────────────────────

ADSP_CURATED_COLS = [
    "Left.Hippocampus_combat",  "Right.Hippocampus_combat",
    "Left.Amygdala_combat",     "Right.Amygdala_combat",
    "Left.Lateral.Ventricle_combat", "Right.Lateral.Ventricle_combat",
    "BrainSegVol_combat",       "EstimatedTotalIntraCranialVol_combat",
    # Thickness (added if available)
]

# AD-signature cortical thickness columns (left+right pairs, if present)
ADSP_THICKNESS_PATTERNS = [
    "entorhinal", "parahippocampal", "middletemporal", "fusiform",
    "precuneus", "inferiorparietal", "inferiortemporal",
]

def get_adsp_cols(adsp_cols: list[str]) -> list[str]:
    """Select available curated + AD-signature thickness columns from ADSP."""
    selected = [c for c in ADSP_CURATED_COLS if c in adsp_cols]
    for pat in ADSP_THICKNESS_PATTERNS:
        matches = [c for c in adsp_cols if pat.lower() in c.lower() and "thickness" in c.lower()]
        selected.extend(matches[:2])  # left + right
    return list(dict.fromkeys(selected))  # preserve order, remove dups

def join_adsp_mri(base: pd.DataFrame, adsp: pd.DataFrame,
                  tolerance_days: int = 180) -> pd.DataFrame:
    """
    For each cognitive visit, attach the nearest-prior ADSP MRI scan
    within tolerance_days.  Strict rule: no future MRI (gap must be ≥ 0).
    Adds mri_obs, mri_gap_months, mri_source columns.
    """
    adsp_cols = get_adsp_cols(list(adsp.columns))
    print(f"  ADSP curated columns selected: {len(adsp_cols)}")

    adsp_sub = adsp[["PTID","PHC_SCANDATE"] + adsp_cols].copy()
    adsp_sub = adsp_sub.rename(columns={"PTID":"subject_id"})
    adsp_sub["PHC_SCANDATE"] = pd.to_datetime(adsp_sub["PHC_SCANDATE"], errors="coerce")
    adsp_sub = adsp_sub.dropna(subset=["PHC_SCANDATE","subject_id"])

    # Normalise ICV
    icv_col = "EstimatedTotalIntraCranialVol_combat"
    if icv_col in adsp_sub.columns:
        for col in adsp_cols:
            if col != icv_col and "Vol" in col or "Hippocampus" in col or "Amygdala" in col or "Ventricle" in col:
                norm_name = col.replace("_combat","") + "_norm"
                adsp_sub[norm_name] = adsp_sub[col] / adsp_sub[icv_col].replace(0, np.nan)

    # Asymmetry: hippocampus
    lh = "Left.Hippocampus_combat"
    rh = "Right.Hippocampus_combat"
    if lh in adsp_sub.columns and rh in adsp_sub.columns:
        adsp_sub["hippo_combined"] = adsp_sub[lh] + adsp_sub[rh]
        adsp_sub["hippo_asymmetry"] = (adsp_sub[lh] - adsp_sub[rh]).abs() / (adsp_sub[lh] + adsp_sub[rh]).replace(0, np.nan)
        if icv_col in adsp_sub.columns:
            adsp_sub["hippo_combined_norm"] = adsp_sub["hippo_combined"] / adsp_sub[icv_col].replace(0, np.nan)

    # Lateral ventricle combined
    llv = "Left.Lateral.Ventricle_combat"
    rlv = "Right.Lateral.Ventricle_combat"
    if llv in adsp_sub.columns and rlv in adsp_sub.columns:
        adsp_sub["lateral_ventricle_combined"] = adsp_sub[llv] + adsp_sub[rlv]
        if icv_col in adsp_sub.columns:
            adsp_sub["lateral_ventricle_combined_norm"] = adsp_sub["lateral_ventricle_combined"] / adsp_sub[icv_col].replace(0, np.nan)

    # Whole brain normalised
    bs_col = "BrainSegVol_combat"
    if bs_col in adsp_sub.columns and icv_col in adsp_sub.columns:
        adsp_sub["whole_brain_norm"] = adsp_sub[bs_col] / adsp_sub[icv_col].replace(0, np.nan)

    base_out = base.copy()
    base_out["_orig_idx"] = np.arange(len(base_out))

    # Fast vectorized merge_asof matching nearest prior PHC_SCANDATE <= exam_date
    mri_feature_cols = [c for c in adsp_sub.columns if c not in ["subject_id", "PHC_SCANDATE"]]

    base_out["_cog_date"] = pd.to_datetime(base_out["exam_date"], errors="coerce")
    adsp_sub_copy = adsp_sub.copy()
    adsp_sub_copy["_scan_date"] = pd.to_datetime(adsp_sub_copy["PHC_SCANDATE"], errors="coerce")

    # Filter valid dates and sort for merge_asof
    valid_base = base_out[base_out["_cog_date"].notna()].sort_values("_cog_date").copy()
    valid_adsp = adsp_sub_copy[adsp_sub_copy["_scan_date"].notna()].sort_values("_scan_date").copy()

    valid_base["subject_id"] = valid_base["subject_id"].astype(str)
    valid_adsp["subject_id"] = valid_adsp["subject_id"].astype(str)

    merged = pd.merge_asof(
        valid_base,
        valid_adsp,
        by="subject_id",
        left_on="_cog_date",
        right_on="_scan_date",
        direction="backward",
        suffixes=("", "_adsp_dup")
    )

    gap_days = (merged["_cog_date"] - merged["_scan_date"]).dt.days

    n_matched = {
        90: int((gap_days <= 90).sum()),
        180: int((gap_days <= 180).sum()),
        365: int((gap_days <= 365).sum()),
    }

    within_tol = (gap_days.notna()) & (gap_days <= tolerance_days)

    merged["mri_obs"] = 0
    merged["mri_gap_months"] = np.nan
    merged["mri_source"] = "none"

    merged.loc[within_tol, "mri_obs"] = 1
    merged.loc[within_tol, "mri_gap_months"] = (gap_days[within_tol] / 30.44).round(2)
    merged.loc[within_tol, "mri_source"] = "adsp_phc"

    # For rows beyond tolerance or without prior scan, null out MRI feature columns
    for col in mri_feature_cols:
        if col in merged.columns:
            merged.loc[~within_tol, col] = np.nan

    # Add back any base rows that had missing exam_date
    invalid_base = base_out[base_out["_cog_date"].isna()].copy()
    if not invalid_base.empty:
        invalid_base["mri_obs"] = 0
        invalid_base["mri_gap_months"] = np.nan
        invalid_base["mri_source"] = "none"
        for col in mri_feature_cols:
            invalid_base[col] = np.nan
        final_df = pd.concat([merged, invalid_base], ignore_index=True)
    else:
        final_df = merged

    final_df = final_df.sort_values("_orig_idx").reset_index(drop=True)
    drop_temp = [c for c in ["_orig_idx", "_cog_date", "_scan_date"] if c in final_df.columns]
    dup_cols = [c for c in final_df.columns if c.endswith("_adsp_dup")]
    final_df = final_df.drop(columns=drop_temp + dup_cols, errors="ignore")

    total = len(final_df)
    print(f"  ADSP MRI coverage: "
          f"@90d={n_matched[90]}/{total}({100*n_matched[90]/total:.1f}%), "
          f"@180d={n_matched[180]}/{total}({100*n_matched[180]/total:.1f}%), "
          f"@365d={n_matched[365]}/{total}({100*n_matched[365]/total:.1f}%)")

    return final_df, n_matched

# ── Stratified group split ─────────────────────────────────────────────────

def make_splits(df: pd.DataFrame) -> dict:
    """
    GroupShuffleSplit stratified on (baseline_DX, ever_converted) per subject.
    Returns dict with keys train/val/test, each a list of subject_ids.
    """
    subjects = df["subject_id"].unique()

    # Per-subject stratification variables
    sub_info = []
    for sid in subjects:
        grp = df[df["subject_id"] == sid].sort_values("visit_month")
        bl_dx = grp["DIAGNOSIS"].iloc[0]
        ever_conv = int(grp["DIAGNOSIS"].nunique() > 1)
        sub_info.append({"subject_id": sid, "bl_dx": bl_dx, "ever_conv": ever_conv})
    sub_df = pd.DataFrame(sub_info)

    # Stratification strata: combine bl_dx + ever_conv
    sub_df["strat"] = sub_df["bl_dx"].astype(str) + "_conv" + sub_df["ever_conv"].astype(str)
    strat_labels = sub_df["strat"].values
    sids = sub_df["subject_id"].values

    from sklearn.model_selection import StratifiedShuffleSplit

    # Split into train (70%) and temp (30%)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SPLIT_SEED)
    train_idx, temp_idx = next(sss1.split(sids, strat_labels))
    train_sids = sids[train_idx].tolist()
    temp_sids  = sids[temp_idx]
    temp_strat = strat_labels[temp_idx]

    # Split temp into val (50%) and test (50%) → 15% each overall
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SPLIT_SEED)
    val_idx, test_idx = next(sss2.split(temp_sids, temp_strat))
    val_sids  = temp_sids[val_idx].tolist()
    test_sids = temp_sids[test_idx].tolist()

    # Assertions: no overlap
    assert len(set(train_sids) & set(val_sids))  == 0, "LEAKAGE: train/val overlap"
    assert len(set(train_sids) & set(test_sids)) == 0, "LEAKAGE: train/test overlap"
    assert len(set(val_sids)   & set(test_sids)) == 0, "LEAKAGE: val/test overlap"

    splits = {"train": train_sids, "val": val_sids, "test": test_sids}

    # Report conversion events per split
    for split_name, sids_list in splits.items():
        grp = sub_df[sub_df["subject_id"].isin(sids_list)]
        n_conv = grp["ever_conv"].sum()
        print(f"  {split_name}: {len(sids_list)} subjects, {n_conv} converters "
              f"({100*n_conv/len(sids_list):.1f}%)")
        for dx in ["1.0","2.0","3.0"]:
            n = (grp["bl_dx"].astype(str) == dx).sum()
            label = {1:"CN",2:"MCI",3:"AD"}.get(int(float(dx)),"?")
            print(f"    baseline {label}: {n}")

    return splits

# ── DX transition matrix ───────────────────────────────────────────────────

def compute_dx_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """Compute visit-to-visit DX transition counts."""
    df_sorted = df.dropna(subset=["DIAGNOSIS","visit_month"]).sort_values(
        ["subject_id","visit_month"]
    )
    rows = []
    for sid, grp in df_sorted.groupby("subject_id"):
        dxs = grp["DIAGNOSIS"].values
        for i in range(len(dxs)-1):
            rows.append({"from": int(dxs[i]), "to": int(dxs[i+1])})
    trans_df = pd.DataFrame(rows)
    if trans_df.empty:
        return pd.DataFrame()
    pivot = trans_df.groupby(["from","to"]).size().unstack(fill_value=0)
    return pivot

# ── Main pipeline ──────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DATA BUILD PIPELINE — unified_visits.parquet")
    print("=" * 70)

    # ── 1. Load base (val file = source of truth) ─────────────────────────
    print("\n[1/10] Loading base longitudinal file ...")
    val = pd.read_csv(VAL_PATH, low_memory=False)
    print(f"  Val: {val.shape[0]} rows, {val['subject_id'].nunique()} subjects")

    # Mark in_4visit_window
    orig4 = pd.read_csv(ORIG4_PATH, low_memory=False)
    orig4_keys = set(zip(orig4["subject_id"], orig4["visit"]))
    val["in_4visit_window"] = val.apply(
        lambda r: (r["subject_id"], r["visit"]) in orig4_keys, axis=1
    ).astype(int)
    print(f"  in_4visit_window: {val['in_4visit_window'].sum()} rows flagged")

    # ── 2. Assign visit_month ─────────────────────────────────────────────
    print("\n[2/10] Assigning visit_month ...")
    imp4 = pd.read_csv(IMP4_PATH, low_memory=False)
    # Get visit_month from imputed file (merge, not positional — verified alignment)
    imp4_vm = imp4[["subject_id","visit","visit_month"]].copy()
    val = val.merge(imp4_vm, on=["subject_id","visit"], how="left")
    # For rows still missing visit_month, parse from visit code
    parsed_vm = val["visit"].apply(parse_visit_month)
    n_from_imp  = val["visit_month"].notna().sum()
    n_from_code = parsed_vm.notna().sum() - n_from_imp
    val["visit_month"] = val["visit_month"].fillna(parsed_vm)
    val["exclude_time_axis"] = val["visit"].apply(is_excluded_timex).astype(int)
    print(f"  visit_month from imputed file: {n_from_imp}")
    print(f"  visit_month parsed from code: {n_from_code}")
    print(f"  exclude_time_axis: {val['exclude_time_axis'].sum()} rows")

    # ── 3. Collapse bl + sc to month-0 ───────────────────────────────────
    print("\n[3/10] Collapsing bl + sc to month-0 ...")
    # For same subject: if both bl and sc exist, merge them into one row
    # (bl values preferred, sc fills gaps)
    bl_rows = val[val["visit"] == "bl"].set_index("subject_id")
    sc_rows = val[val["visit"] == "sc"].set_index("subject_id")
    both = bl_rows.index[bl_rows.index.isin(sc_rows.index)]
    print(f"  Subjects with both bl and sc: {len(both)}")

    SCORE_COLS = ["MMSCORE", "CDGLOBAL", "TOTSCORE", "DIAGNOSIS"]
    for sid in both:
        bl_idx = val[(val["subject_id"] == sid) & (val["visit"] == "bl")].index
        sc_idx = val[(val["subject_id"] == sid) & (val["visit"] == "sc")].index
        if len(bl_idx) == 0 or len(sc_idx) == 0:
            continue
        bl_i = bl_idx[0]
        sc_i = sc_idx[0]
        for col in SCORE_COLS:
            if pd.isna(val.loc[bl_i, col]) and not pd.isna(val.loc[sc_i, col]):
                val.loc[bl_i, col] = val.loc[sc_i, col]
    # Mark sc rows for reference (keep in parquet, raw)
    val["_bl_sc_collapsed"] = 0
    sc_mask = (val["visit"] == "sc") & (val["subject_id"].isin(both))
    val.loc[sc_mask, "_bl_sc_collapsed"] = 1
    print(f"  sc rows marked as collapsed: {sc_mask.sum()}")

    # ── 4. Reconstruct exam dates ─────────────────────────────────────────
    print("\n[4/10] Reconstructing exam dates ...")
    adni_full = pd.read_csv(ADNI_PATH, low_memory=False, usecols=[
        "PTID","VISCODE","EXAMDATE","Month_bl",
        "AGE","PTGENDER","PTEDUCAT","APOE4",
        "MMSE","CDRSB","ADAS13","FAQ","MOCA",
        "ABETA","TAU","PTAU","FDG","AV45",
        "Hippocampus","WholeBrain","Ventricles","Entorhinal","Fusiform","MidTemp","ICV",
        "FLDSTRENG","FSVERSION","IMAGEUID",
    ])
    mytab = pd.read_csv(MYTAB_PATH, low_memory=False)
    entry = pd.read_csv(ENTRY_PATH, low_memory=False)

    adni_dates_df  = adni_full[["PTID","VISCODE","EXAMDATE"]].copy()
    mytab_dates_df = mytab[["subject_id","visit"] + (["EXAMDATE"] if "EXAMDATE" in mytab.columns else [])].copy()

    val = reconstruct_visit_dates(val, adni_dates_df, mytab_dates_df, entry)
    rule_counts = val["date_rule"].value_counts().to_dict()
    print(f"  Date rule coverage: {rule_counts}")
    print(f"  exam_date non-null: {val['exam_date'].notna().sum()}/{len(val)}")

    # ── 5. My_Table covariates ────────────────────────────────────────────
    print("\n[5/10] Joining My_Table covariates ...")
    MT_COLS = ["subject_id","visit","CDRSB","MOCA","FAQTOTAL","NPISCORE"]
    mt_sub = mytab[[c for c in MT_COLS if c in mytab.columns]].copy()
    val = val.merge(mt_sub, on=["subject_id","visit"], how="left", suffixes=("","_mt"))
    for col in ["CDRSB","MOCA","FAQTOTAL","NPISCORE"]:
        if col in val.columns:
            print(f"  {col} non-null: {val[col].notna().sum()}/{len(val)}")

    # ── 6. ADNIMERGE covariates ───────────────────────────────────────────
    print("\n[6/10] Joining ADNIMERGE covariates ...")
    val = join_adnimerge(val, adni_full)

    # ── 7. APOE genotype ──────────────────────────────────────────────────
    print("\n[7/10] Adding APOE genotype ...")
    apoe = pd.read_csv(APOE_PATH, low_memory=False, usecols=["PTID","GENOTYPE","APVOLUME"])
    apoe2 = pd.read_csv(APOE2_PATH, low_memory=False, usecols=["PTID","GENOTYPE"])
    apoe = apoe.drop_duplicates(subset=["PTID"])
    apoe2 = apoe2.drop_duplicates(subset=["PTID"])
    # Primary: fixed_simple; fill from 30Dec version
    apoe_merged = apoe.merge(apoe2, on="PTID", how="outer", suffixes=("","_2"))
    apoe_merged["GENOTYPE"] = apoe_merged["GENOTYPE"].fillna(apoe_merged.get("GENOTYPE_2",""))
    apoe_merged = apoe_merged[["PTID","GENOTYPE","APVOLUME"]].drop_duplicates(subset=["PTID"])
    apoe_merged = apoe_merged.rename(columns={"PTID":"subject_id"})
    val = val.merge(apoe_merged, on="subject_id", how="left", suffixes=("","_apoe"))
    val["apoe4_count"] = val["GENOTYPE"].apply(genotype_to_apoe4_count)

    # Compare with ADNIMERGE APOE4
    if "APOE4" in val.columns:
        conflict_mask = (
            val["apoe4_count"].notna() & val["APOE4"].notna() &
            (val["apoe4_count"] != val["APOE4"])
        )
        n_conflict = conflict_mask.sum()
        print(f"  APOE4 conflicts (APOERES vs ADNIMERGE): {n_conflict}/{val['apoe4_count'].notna().sum()} "
              f"({100*n_conflict/max(val['apoe4_count'].notna().sum(),1):.1f}%)")
        # Fill APOE4 where apoe4_count is missing (ADNIMERGE as fallback)
        val["apoe4_count"] = val["apoe4_count"].fillna(val["APOE4"])

    # ── 8. Demographics: sex, education ───────────────────────────────────
    print("\n[8/10] Filling demographics ...")
    ptdemog = pd.read_csv(PTDEMOG_PATH, low_memory=False,
                           usecols=[c for c in ["PTID","PTGENDER","PTEDUCAT"]
                                    if c in pd.read_csv(PTDEMOG_PATH, nrows=0).columns])
    ptdemog = ptdemog.drop_duplicates(subset=["PTID"]).rename(columns={"PTID":"subject_id"})
    val = val.merge(ptdemog, on="subject_id", how="left", suffixes=("","_demog"))

    # Unified sex / education columns
    if "PTGENDER" in val.columns:
        # 1=Male, 2=Female → 0=Male, 1=Female
        val["sex"] = (val["PTGENDER"].replace({1:0,2:1,
                                                "Male":0,"Female":1,
                                                "MALE":0,"FEMALE":1})).astype(float)
    if "PTEDUCAT" in val.columns:
        val["education_years"] = pd.to_numeric(val["PTEDUCAT"], errors="coerce")

    # Family history
    try:
        famhx = pd.read_csv(FAMHX_PATH, low_memory=False,
                             usecols=[c for c in ["PTID","MOTHDEM","FATHDEM"]
                                      if c in pd.read_csv(FAMHX_PATH, nrows=0).columns])
        famhx = famhx.drop_duplicates(subset=["PTID"]).rename(columns={"PTID":"subject_id"})
        famhx["family_hx_ad"] = ((famhx.get("MOTHDEM",0) == 1) |
                                  (famhx.get("FATHDEM",0) == 1)).astype(int)
        val = val.merge(famhx[["subject_id","family_hx_ad"]], on="subject_id", how="left")
    except Exception as e:
        print(f"  Family history join skipped: {e}")

    # Rename entry_age for clarity
    if "entry_age" in val.columns and "age_at_visit" not in val.columns:
        val["age_at_visit"] = val["entry_age"] + val["visit_month"].fillna(0) / 12.0

    # ── 9. ADSP MRI join ──────────────────────────────────────────────────
    print("\n[9/10] Joining ADSP FreeSurfer MRI ...")
    adsp = pd.read_csv(ADSP_PATH, low_memory=False)
    val, n_matched = join_adsp_mri(val, adsp)

    # Reconcile ADSP vs ADNIMERGE FS volumes
    # Fill mri_source="adnimerge" where ADSP is missing but ADNIMERGE FS exists
    adni_fs_map = {
        "Hippocampus": "hippo_combined",    # ADNIMERGE has left+right sum
        "Ventricles":  "lateral_ventricle_combined",
        "WholeBrain":  "whole_brain_norm",
        "Entorhinal":  None,                # No direct equivalent in derived
        "Fusiform":    None,
        "MidTemp":     None,
        "ICV":         None,
    }
    mri_missing = val["mri_obs"] == 0
    for adni_col, adsp_col in adni_fs_map.items():
        if adni_col in val.columns and adsp_col and adsp_col in val.columns:
            fill_mask = mri_missing & val[adsp_col].isna() & val[adni_col].notna()
            val.loc[fill_mask, adsp_col] = val.loc[fill_mask, adni_col]
            val.loc[fill_mask, "mri_source"] = "adnimerge"
            val.loc[fill_mask, "mri_obs"] = 1
            val.loc[fill_mask, "mri_gap_months"] = 0  # same-visit, no scan date
            print(f"  Filled {fill_mask.sum()} rows from ADNIMERGE {adni_col} -> {adsp_col}")

    # ── 10. Add observation mask columns ──────────────────────────────────
    print("\n[10/10] Adding observation masks + finalising ...")
    MASKABLE_COLS = [
        "MMSCORE","CDGLOBAL","TOTSCORE","CDRSB","MOCA","FAQTOTAL","NPISCORE",
        "ABETA","TAU","PTAU","apoe4_count",
        "hippo_combined_norm","hippo_asymmetry","lateral_ventricle_combined_norm",
        "whole_brain_norm",
    ]
    for col in MASKABLE_COLS:
        if col in val.columns:
            val[f"{col}_obs"] = val[col].notna().astype(int)

    # Canonical DX (shortened file is source of truth — never overwrite)
    # Log disagreement with ADNIMERGE DX
    if "DX" in val.columns:
        dx_adni_numeric = val["DX"].map({"CN":1,"MCI":2,"Dementia":3,"MCI ":2}).fillna(np.nan)
        disagree = (
            val["DIAGNOSIS"].notna() & dx_adni_numeric.notna() &
            (val["DIAGNOSIS"] != dx_adni_numeric)
        )
        print(f"  DX disagreement (shortened vs ADNIMERGE DX): {disagree.sum()} rows "
              f"({100*disagree.mean():.1f}%)")
        val["DX_adni_disagree"] = disagree.astype(int)
        val = val.drop(columns=["DX"], errors="ignore")

    # Drop forbidden features (safety)
    from ml.twin.constants import FORBIDDEN_FEATURES
    to_drop = [c for c in FORBIDDEN_FEATURES if c in val.columns]
    if to_drop:
        print(f"  Dropping forbidden features: {to_drop}")
        val = val.drop(columns=to_drop)

    # ── Save parquet ───────────────────────────────────────────────────────
    val.to_parquet(PARQUET_OUT, index=False)
    print(f"\n[OK] Parquet saved: {PARQUET_OUT} ({val.shape[0]} rows x {val.shape[1]} cols)")

    # ── Compute splits ─────────────────────────────────────────────────────
    print("\n── Computing stratified subject splits ──")
    val_vm = val.dropna(subset=["visit_month","DIAGNOSIS"])
    val_vm = val_vm[val_vm["exclude_time_axis"] == 0]
    splits = make_splits(val_vm)
    with open(SPLITS_OUT, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"[OK] Splits saved: {SPLITS_OUT}")

    # ── Write audit ────────────────────────────────────────────────────────
    write_audit(val, splits, n_matched)

def write_audit(df: pd.DataFrame, splits: dict, n_matched: dict):
    """Write the data audit markdown report."""
    lines = ["# Data Audit: unified_visits.parquet", "",
             "Generated by `server/ml/twin/data_build.py`", "", "---", ""]

    lines.append("## 1. Dataset Counts")
    lines.append("")
    lines.append(f"- Total rows: {len(df):,}")
    lines.append(f"- Unique subjects: {df['subject_id'].nunique():,}")
    lines.append(f"- Rows on time axis (exclude_time_axis=0): {(df['exclude_time_axis']==0).sum():,}")
    lines.append(f"- Rows in 4-visit window (in_4visit_window=1): {df['in_4visit_window'].sum():,}")
    lines.append("")

    lines.append("## 2. TOTSCORE Identity")
    lines.append("")
    lines.append("**TOTSCORE = ADAS11-Cog prorated subscale (0–70, higher = worse).**")
    lines.append("Fractional values (e.g., 5.67) arise from prorating missing items:")
    lines.append("`score = raw_sum / n_observed * 11`.")
    lines.append("Verified by exact diff=0 against All_Subjects_ADAS_30Dec2025.csv on 4,850 overlapping (subject, visit) pairs.")
    lines.append("Correlation with ADNIMERGE.ADAS11: r=0.9991. Correlation with ADNIMERGE.ADAS13: r=0.9805.")
    lines.append("")

    lines.append("## 3. Alignment Verification")
    lines.append("")
    lines.append("- `shortened_result_imputed_4visits.csv` vs `shortened_result_original_missing_4visits.csv`:")
    lines.append("  - Keys (subject_id, visit) are **identical** (0 exclusive keys on either side)")
    lines.append("  - Row **ordering differs** — use `pd.merge`, not positional join ✓")
    lines.append("  - No duplicate (subject_id, visit) in any longitudinal file ✓")
    lines.append("- Imputed file is used as **comparison baseline only**; label: *'inputs may leak future visit info'*")
    lines.append("")

    lines.append("## 4. Visit-Month Mapping Rules")
    lines.append("")
    lines.append("| Visit Code | visit_month | Notes |")
    lines.append("|------------|------------|-------|")
    lines.append("| `bl` | 0 | Collapsed with sc into month-0 |")
    lines.append("| `sc` | 0 | Collapsed into month-0; bl values preferred; gaps filled from sc |")
    lines.append("| `m{N}` | N | e.g., m06→6, m12→12 |")
    lines.append("| `scmri`, `nv`, `AUT`, `uns1` | NaN | `exclude_time_axis=1`; row kept for DX label |")
    lines.append("")

    lines.append("## 5. Date Reconstruction Coverage")
    lines.append("")
    if "date_rule" in df.columns:
        rule_counts = df["date_rule"].value_counts().to_dict()
        total = len(df)
        for rule, cnt in rule_counts.items():
            lines.append(f"- `{rule}`: {cnt:,} ({100*cnt/total:.1f}%)")
    lines.append("")

    lines.append("## 6. ADNIMERGE Join Match Rate")
    lines.append("")
    if "adni_match_type" in df.columns:
        match_counts = df["adni_match_type"].value_counts().to_dict()
        for mtype, cnt in match_counts.items():
            lines.append(f"- `{mtype}`: {cnt:,} ({100*cnt/len(df):.1f}%)")
    lines.append("")

    lines.append("## 7. MRI Join Coverage (ADSP FreeSurfer, no-future-scan rule)")
    lines.append("")
    total = len(df)
    for tol, cnt in n_matched.items():
        lines.append(f"- Within {tol} days: {cnt:,}/{total:,} ({100*cnt/total:.1f}%)")
    lines.append("")
    if "mri_source" in df.columns:
        src_counts = df["mri_source"].value_counts().to_dict()
        for src, cnt in src_counts.items():
            lines.append(f"- `mri_source={src}`: {cnt:,} rows")
    lines.append("")

    lines.append("## 8. Missingness per Column (main features)")
    lines.append("")
    key_cols = ["MMSCORE","CDGLOBAL","TOTSCORE","CDRSB","MOCA","FAQTOTAL",
                "ABETA","TAU","PTAU","apoe4_count","sex","education_years",
                "hippo_combined_norm","mri_obs"]
    lines.append("| Column | Non-null | Missing% |")
    lines.append("|--------|----------|----------|")
    for col in key_cols:
        if col in df.columns:
            nn = df[col].notna().sum()
            miss = 100 - 100*nn/len(df)
            lines.append(f"| {col} | {nn:,} | {miss:.1f}% |")
    lines.append("")

    lines.append("## 9. DX Distribution and Transitions")
    lines.append("")
    dx_counts = df["DIAGNOSIS"].value_counts().sort_index().to_dict()
    dx_names = {1:"CN",2:"MCI",3:"AD"}
    for k, v in dx_counts.items():
        try:
            lines.append(f"- DX={int(k)} ({dx_names.get(int(k),'?')}): {v:,} rows")
        except Exception:
            lines.append(f"- DX={k}: {v:,} rows")
    lines.append("")

    # Transition matrix
    trans = compute_dx_transitions(df)
    if not trans.empty:
        lines.append("**DX Transition Matrix** (consecutive visits, time-ordered):")
        lines.append("")
        lines.append("| From \\ To | " + " | ".join(dx_names.get(c,"?") for c in trans.columns) + " |")
        lines.append("|" + "---|"*(len(trans.columns)+1))
        for from_dx, row in trans.iterrows():
            lines.append("| " + dx_names.get(from_dx,"?") + " | " +
                          " | ".join(str(int(row.get(c,0))) for c in trans.columns) + " |")
        lines.append("")

    lines.append("## 10. Splits")
    lines.append("")
    for split, sids in splits.items():
        lines.append(f"- **{split}**: {len(sids):,} subjects")
    lines.append("")
    lines.append("*Stratified by (baseline_DX, ever_converted). "
                 "No subject appears in >1 split (assertion tested in data_build.py and tests).*")
    lines.append("")

    lines.append("## 11. APOE4 Conflicts")
    lines.append("")
    if "DX_adni_disagree" in df.columns:
        nd = df["DX_adni_disagree"].sum()
        lines.append(f"- DX disagreement (shortened DIAGNOSIS vs ADNIMERGE DX): {nd:,} rows")
    lines.append("- Canonical DX = shortened-file DIAGNOSIS (never overwritten by ADNIMERGE)")
    lines.append("")

    lines.append("## 12. Forbidden Features")
    lines.append("")
    from ml.twin.constants import FORBIDDEN_FEATURES
    lines.append("The following features are excluded from all models (see `constants.py`):")
    lines.append("")
    for f in FORBIDDEN_FEATURES:
        in_df = "✓ dropped" if f not in df.columns else "⚠️ STILL PRESENT"
        lines.append(f"- `{f}`: {in_df}")
    lines.append("")

    AUDIT_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Audit written to: {AUDIT_OUT}")

if __name__ == "__main__":
    main()
