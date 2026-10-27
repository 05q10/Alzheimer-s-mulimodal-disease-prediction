"""
server/ml/twin/constants.py
===========================
Global constants shared by all twin modules.

FORBIDDEN_FEATURES:
  These columns MUST NEVER appear in any model's feature list.
  They are either:
    - Computed over the subject's whole history including future visits
      (num_visits, num_dirty_*, *_traj_type, imputation_confidence)
    - Label-derived / DX criteria definitions (ADNIMERGE DX, DX_bl)
    - Imputation artefacts that may leak future values
  A pytest in server/tests/test_no_leakage.py imports this list
  and fails if any forbidden feature appears in any saved feature list.
"""

FORBIDDEN_FEATURES = [
    # Computed over full subject history (leaks future)
    "num_visits",
    "num_dirty_rows",
    "num_dirty_mmse",
    "num_dirty_cdglobal",
    "num_dirty_totscore",
    # Imputation metadata (may encode future visit pattern)
    "CDGLOBAL_traj_type",
    "MMSCORE_traj_type",
    "TOTSCORE_traj_type",
    "CDGLOBAL_is_missing",
    "MMSCORE_is_missing",
    "TOTSCORE_is_missing",
    "imputation_confidence",
    # ADNIMERGE DX labels (partly define DX criteria)
    "DX",
    "DX_bl",
    "BCPREDX",    # clinician baseline-change prediction (label-derived)
    # Derived from forbidden fields
    "num_visits_so_far",  # safe version computed at anchor — allowed; raw is not
]

# TOTSCORE identity (verified by exact match against All_Subjects_ADAS_30Dec2025.csv)
TOTSCORE_IDENTITY = "ADAS11_cog_prorated"
TOTSCORE_DESCRIPTION = (
    "ADAS-Cog 11-item subscale (0-70, higher = worse). "
    "Fractional values arise from prorating over observed items: "
    "score = raw_sum / n_observed * 11. "
    "Verified by exact diff=0 against All_Subjects_ADAS_30Dec2025.csv "
    "on 4,850 overlapping (subject, visit) pairs."
)

# DX encoding in the shortened longitudinal files
DX_MAP = {1: "CN", 2: "MCI", 3: "AD"}  # DIAGNOSIS column in shortened files
DX_MAP_INV = {"CN": 1, "MCI": 2, "AD": 3}

# Visit-month parsing rules (documented as required by prompt)
VISIT_MONTH_RULES = {
    "bl": 0,
    "sc": 0,       # collapsed into month-0 (bl preferred, sc fills gaps)
    "scmri": None, # excluded from time axis; row kept for DX only
    "nv": None,
    "AUT": None,
    "uns1": None,
    # m{N} -> N  (parsed programmatically)
}

# MRI temporal join tolerance
MRI_JOIN_TOLERANCE_DAYS = 180   # ±180 days; no future scans (strict <=0 offset allowed)

# Feature tiers for the baseline stager
FEATURE_TIERS = {
    "T1":  ["entry_age", "sex", "education_years", "apoe4_count"],
    "T1a": ["entry_age", "sex", "education_years", "apoe4_count"],
    "T1b": ["entry_age", "sex", "education_years", "apoe4_count", "MMSE"],
    "T2a": ["entry_age", "sex", "education_years", "apoe4_count",
            "MMSE", "CDGLOBAL", "CDRSB"],
    "T2b": ["entry_age", "sex", "education_years", "apoe4_count",
            "MMSE", "CDGLOBAL", "CDRSB", "FAQTOTAL", "TOTSCORE", "MOCA"],
    "T2":  ["entry_age", "sex", "education_years", "apoe4_count",
            "MMSE", "CDGLOBAL", "CDRSB", "FAQTOTAL", "TOTSCORE", "MOCA"],
    "T3":  ["entry_age", "sex", "education_years", "apoe4_count",
            "MMSE", "CDGLOBAL", "CDRSB", "FAQTOTAL", "TOTSCORE", "MOCA",
            "ABETA", "TAU", "PTAU", "ABETA_censored", "TAU_censored"],
    "T4":  ["entry_age", "sex", "education_years", "apoe4_count",
            "MMSE", "CDGLOBAL", "CDRSB", "FAQTOTAL", "TOTSCORE", "MOCA",
            "ABETA", "TAU", "PTAU", "ABETA_censored", "TAU_censored",
            "hippo_combined_norm", "hippo_asymmetry",
            "lh_entorhinal_thickness", "rh_entorhinal_thickness",
            "lh_amygdala_norm", "rh_amygdala_norm",
            "lateral_ventricle_combined_norm",
            "whole_brain_norm",
            "lh_midtemporal_thickness", "rh_midtemporal_thickness",
            "lh_fusiform_thickness", "rh_fusiform_thickness",
            "lh_precuneus_thickness", "rh_precuneus_thickness",
            "mri_obs", "mri_gap_months"],
}

# ResNet MRI model label mapping (for late fusion)
RESNET_TO_DX = {
    "NonDemented": "CN",
    "VeryMild": "MCI",
    "Mild": "AD",
    "Moderate": "AD",
}

# Late fusion weights (ADNI stager 80%, ResNet 20%)
# ResNet was trained on an augmented non-ADNI Kaggle set.
# Its published 99.7% accuracy is INFLATED and not validated on ADNI.
# These weights are demo-only constants, not clinically validated.
FUSION_WEIGHTS = {"adni": 0.80, "resnet": 0.20}
FUSION_DISCLAIMER = (
    "The ResNet MRI model was trained on a non-ADNI Kaggle dataset. "
    "Its accuracy on ADNI subjects is unknown. The 80/20 fusion is "
    "for demonstration purposes only and is NOT clinically validated."
)

# GRU ship rule parameters
GRU_SHIP_MIN_F1_MARGIN = 0.03       # GRU must beat GBM by at least this
GRU_SHIP_MMSE_MAE_TOLERANCE = 0.0   # GRU must not be worse on MMSE MAE @+12m

# Data splits
SPLIT_SEED = 42
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
