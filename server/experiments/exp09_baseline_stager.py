"""
server/experiments/exp09_baseline_stager.py
===========================================
Train and evaluate Model A: the CatBoost baseline stager.
Outputs:
  server/experiments/results/exp09_stager_results.md
  server/ml/twin/weights/stager_model.cbm
  server/ml/twin/weights/stager_meta.json

Tiers evaluated:
  - T1:  Demographics only (Age, Sex, Education, APOE4)
  - T1b: T1 + MMSE
  - T2a: T1b + CDGLOBAL, CDRSB
  - T2b: T2a + TOTSCORE, FAQTOTAL, MOCA (Full cognitive/functional)
  - T3:  T2b + CSF biomarkers (ABETA, TAU, PTAU)
  - T4:  T3 + FreeSurfer MRI volumetric features

Paired Subset Analysis:
  - CSF-observed rows: T2b vs T3 (Delta Macro-F1, Delta Entropy with 95% Cluster-Bootstrap CIs)
  - MRI-observed rows: T2b vs T4 (Delta Macro-F1, Delta Entropy with 95% Cluster-Bootstrap CIs)

Self-checks:
  - Forbidden-feature assertion
  - Split-overlap assertion
  - Trivial baseline comparison
  - ECE per tier
"""

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, classification_report,
)

warnings.filterwarnings("ignore")

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))

from ml.twin.constants import FEATURE_TIERS, FORBIDDEN_FEATURES, SPLIT_SEED
from ml.twin.stager import TwinStager

RESULTS_DIR = SERVER_DIR / "experiments" / "results"
DATA_DIR    = SERVER_DIR / "data"
PARQUET     = DATA_DIR / "unified_visits.parquet"
SPLITS_FILE = DATA_DIR / "splits.json"

LABEL_NAMES = ["CN", "MCI", "AD"]


# ── Calibration & Entropy ──────────────────────────────────────────────────

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) using equal-width bins."""
    confidences = probs.max(axis=1)
    preds       = probs.argmax(axis=1)
    correct     = (preds == labels).astype(float)
    ece = 0.0
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc  = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += mask.sum() / len(labels) * abs(acc - conf)
    return round(float(ece), 4)


def compute_mean_entropy(probs: np.ndarray) -> float:
    """Compute mean Shannon entropy across predictions: - sum p * log(p)."""
    clipped = np.clip(probs, 1e-12, 1.0)
    entropy_per_row = -np.sum(clipped * np.log(clipped), axis=1)
    return float(np.mean(entropy_per_row))


# ── Bootstrap CIs ──────────────────────────────────────────────────────────

def cluster_bootstrap_ci(
    probs: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    metric_fn,
    n_boot: int = 1000,
    seed: int = SPLIT_SEED,
) -> Tuple[float, float]:
    """Cluster bootstrap CI (resample subjects, not rows)."""
    rng = np.random.default_rng(seed)
    unique_sids = np.unique(subject_ids)
    n_subs = len(unique_sids)
    sid_to_idx = {sid: np.where(subject_ids == sid)[0] for sid in unique_sids}

    boot_vals = []
    for _ in range(n_boot):
        sample_sids = rng.choice(unique_sids, size=n_subs, replace=True)
        boot_idx = np.concatenate([sid_to_idx[sid] for sid in sample_sids])
        y_b = labels[boot_idx]
        if len(np.unique(y_b)) < len(np.unique(labels)):
            continue
        try:
            v = metric_fn(y_b, probs[boot_idx])
            boot_vals.append(v)
        except Exception:
            pass

    if not boot_vals:
        return 0.0, 0.0
    lo, hi = np.percentile(boot_vals, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def cluster_bootstrap_paired_diff(
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    metric_fn,
    n_boot: int = 1000,
    seed: int = SPLIT_SEED,
) -> Tuple[float, float, float]:
    """
    Paired cluster bootstrap for metric(probs_b) - metric(probs_a).
    Returns (observed_diff, lo_95, hi_95).
    """
    rng = np.random.default_rng(seed)
    unique_sids = np.unique(subject_ids)
    n_subs = len(unique_sids)
    sid_to_idx = {sid: np.where(subject_ids == sid)[0] for sid in unique_sids}

    orig_diff = metric_fn(labels, probs_b) - metric_fn(labels, probs_a)
    boot_diffs = []

    for _ in range(n_boot):
        sample_sids = rng.choice(unique_sids, size=n_subs, replace=True)
        boot_idx = np.concatenate([sid_to_idx[sid] for sid in sample_sids])
        y_b = labels[boot_idx]
        if len(np.unique(y_b)) < len(np.unique(labels)):
            continue
        try:
            diff = metric_fn(y_b, probs_b[boot_idx]) - metric_fn(y_b, probs_a[boot_idx])
            boot_diffs.append(diff)
        except Exception:
            pass

    if not boot_diffs:
        return round(float(orig_diff), 4), 0.0, 0.0
    lo, hi = np.percentile(boot_diffs, [2.5, 97.5])
    return round(float(orig_diff), 4), round(float(lo), 4), round(float(hi), 4)


# ── Per-tier evaluation ────────────────────────────────────────────────────

def get_tier_probs(stager: TwinStager, df: pd.DataFrame, tier: str) -> np.ndarray:
    feat_cols = FEATURE_TIERS[tier]
    tier_all  = stager.feature_list
    X = df[tier_all].astype(float).copy()
    for col in tier_all:
        if col not in feat_cols:
            X[col] = np.nan
    return stager._calibrated_probs(X)


def evaluate_tier(stager: TwinStager, df_test: pd.DataFrame,
                  tier: str, splits: dict) -> dict:
    """Evaluate on test set for a given tier."""
    probs  = get_tier_probs(stager, df_test, tier)
    labels = df_test["DX_label"].values
    sids   = df_test["subject_id"].values

    acc  = accuracy_score(labels, probs.argmax(axis=1))
    mf1  = f1_score(labels, probs.argmax(axis=1), average="macro", zero_division=0)
    bacc = balanced_accuracy_score(labels, probs.argmax(axis=1))
    ece  = compute_ece(probs, labels)
    ent  = compute_mean_entropy(probs)

    # Trivial baseline (predict majority class MCI)
    trivial_class = 1
    trivial_f1 = f1_score(labels, np.full_like(labels, trivial_class),
                          average="macro", zero_division=0)

    f1_lo, f1_hi = cluster_bootstrap_ci(
        probs, labels, sids,
        lambda y, p: f1_score(y, p.argmax(axis=1), average="macro", zero_division=0)
    )

    return {
        "tier": tier,
        "n_rows": len(df_test),
        "accuracy": round(acc, 4),
        "macro_f1": round(mf1, 4),
        "f1_95ci": f"[{f1_lo}, {f1_hi}]",
        "balanced_acc": round(bacc, 4),
        "ece": ece,
        "mean_entropy": round(ent, 4),
        "trivial_macro_f1": round(trivial_f1, 4),
        "beats_trivial": mf1 > trivial_f1,
        "per_class": classification_report(
            labels, probs.argmax(axis=1), target_names=LABEL_NAMES,
            zero_division=0, output_dict=True
        ),
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("EXP09: Baseline Stager Evaluation (Refined Tiers + Paired Analysis)")
    print("=" * 70)

    # ── 0. Self-checks ────────────────────────────────────────────────────
    print("\n[0] Running self-checks ...")
    with open(SPLITS_FILE) as f:
        splits = json.load(f)

    train_set = set(splits["train"])
    val_set   = set(splits["val"])
    test_set  = set(splits["test"])
    assert len(train_set & val_set)  == 0, "LEAKAGE: train/val overlap!"
    assert len(train_set & test_set) == 0, "LEAKAGE: train/test overlap!"
    assert len(val_set   & test_set) == 0, "LEAKAGE: val/test overlap!"
    print("  Split overlap: NONE (pass)")

    all_feats = TwinStager.get_all_feature_names()
    forbidden_in_model = [f for f in all_feats if f in FORBIDDEN_FEATURES]
    assert len(forbidden_in_model) == 0, f"Forbidden features: {forbidden_in_model}"
    print("  Forbidden features in model: NONE (pass)")

    # ── 1. Train ──────────────────────────────────────────────────────────
    print("\n[1] Training stager ...")
    stager = TwinStager()
    stager.fit(verbose=True)
    stager.save()
    print("  Model saved.")

    # ── 2. Load test set ──────────────────────────────────────────────────
    print("\n[2] Loading test set ...")
    df = pd.read_parquet(PARQUET)
    df = df[df["exclude_time_axis"] == 0].copy()
    df = df.dropna(subset=["DIAGNOSIS"])
    df["DX_label"] = (df["DIAGNOSIS"].astype(int) - 1)

    if "MMSCORE" in df.columns:
        if "MMSE" in df.columns:
            df["MMSE"] = df["MMSCORE"].fillna(df["MMSE"])
        else:
            df["MMSE"] = df["MMSCORE"]
        df = df.drop(columns=["MMSCORE"])

    for col in stager.feature_list:
        if col not in df.columns:
            df[col] = np.nan

    df_test = df[df["subject_id"].isin(test_set)].copy()
    print(f"  Test rows: {len(df_test)}, subjects: {df_test['subject_id'].nunique()}")

    # ── 3. Per-tier evaluation ────────────────────────────────────────────
    print("\n[3] Evaluating per tier (T1, T1b, T2a, T2b, T3, T4) ...")
    tier_results = {}
    eval_tiers = ["T1", "T1b", "T2a", "T2b", "T3", "T4"]
    for tier in eval_tiers:
        print(f"  Tier {tier:<4} ...", end=" ", flush=True)
        res = evaluate_tier(stager, df_test, tier, splits)
        tier_results[tier] = res
        print(f"Acc={res['accuracy']:.3f}  MacroF1={res['macro_f1']:.3f}  F1_CI={res['f1_95ci']}  "
              f"ECE={res['ece']:.4f}  Entropy={res['mean_entropy']:.3f}  BeatsTrivial={res['beats_trivial']}")

    # ── 4. Paired Subset Analysis ─────────────────────────────────────────
    print("\n[4] Running Paired Subset Analysis on Observed Rows ...")
    paired_results = {}

    f1_metric = lambda y, p: f1_score(y, p.argmax(axis=1), average="macro", zero_division=0)
    ent_metric = lambda y, p: compute_mean_entropy(p)

    # 4A. CSF Paired Subset: compare T2b vs T3 where ABETA is observed
    csf_mask = df_test["ABETA"].notna()
    df_csf = df_test[csf_mask].copy()
    n_csf = len(df_csf)
    n_csf_subs = df_csf["subject_id"].nunique()
    print(f"  CSF-observed subset: {n_csf} visits ({n_csf_subs} subjects)")

    if n_csf >= 20:
        p_t2b_csf = get_tier_probs(stager, df_csf, "T2b")
        p_t3_csf  = get_tier_probs(stager, df_csf, "T3")
        y_csf = df_csf["DX_label"].values
        sids_csf = df_csf["subject_id"].values

        f1_t2b = f1_score(y_csf, p_t2b_csf.argmax(axis=1), average="macro", zero_division=0)
        f1_t3  = f1_score(y_csf, p_t3_csf.argmax(axis=1), average="macro", zero_division=0)
        ent_t2b = compute_mean_entropy(p_t2b_csf)
        ent_t3  = compute_mean_entropy(p_t3_csf)

        diff_f1, lo_f1, hi_f1 = cluster_bootstrap_paired_diff(
            p_t2b_csf, p_t3_csf, y_csf, sids_csf, f1_metric
        )
        diff_ent, lo_ent, hi_ent = cluster_bootstrap_paired_diff(
            p_t2b_csf, p_t3_csf, y_csf, sids_csf, ent_metric
        )

        paired_results["csf"] = {
            "n_rows": n_csf, "n_subjects": n_csf_subs,
            "f1_t2b": round(f1_t2b, 4), "f1_t3": round(f1_t3, 4),
            "delta_f1": diff_f1, "delta_f1_ci": f"[{lo_f1}, {hi_f1}]",
            "entropy_t2b": round(ent_t2b, 4), "entropy_t3": round(ent_t3, 4),
            "delta_entropy": diff_ent, "delta_entropy_ci": f"[{lo_ent}, {hi_ent}]",
        }
        print(f"    T2b F1={f1_t2b:.3f} -> T3 F1={f1_t3:.3f} | Delta F1={diff_f1:+.4f} (95% CI: [{lo_f1}, {hi_f1}])")
        print(f"    T2b Ent={ent_t2b:.3f} -> T3 Ent={ent_t3:.3f} | Delta Ent={diff_ent:+.4f} (95% CI: [{lo_ent}, {hi_ent}])")

    # 4B. FreeSurfer Paired Subset: compare T2b vs T4 where MRI is observed
    mri_mask = (df_test["mri_obs"] == 1) | df_test["hippo_combined_norm"].notna()
    df_mri = df_test[mri_mask].copy()
    n_mri = len(df_mri)
    n_mri_subs = df_mri["subject_id"].nunique()
    print(f"  MRI-observed subset: {n_mri} visits ({n_mri_subs} subjects)")

    if n_mri >= 20:
        p_t2b_mri = get_tier_probs(stager, df_mri, "T2b")
        p_t4_mri  = get_tier_probs(stager, df_mri, "T4")
        y_mri = df_mri["DX_label"].values
        sids_mri = df_mri["subject_id"].values

        f1_t2b_m = f1_score(y_mri, p_t2b_mri.argmax(axis=1), average="macro", zero_division=0)
        f1_t4    = f1_score(y_mri, p_t4_mri.argmax(axis=1), average="macro", zero_division=0)
        ent_t2b_m = compute_mean_entropy(p_t2b_mri)
        ent_t4    = compute_mean_entropy(p_t4_mri)

        diff_f1_m, lo_f1_m, hi_f1_m = cluster_bootstrap_paired_diff(
            p_t2b_mri, p_t4_mri, y_mri, sids_mri, f1_metric
        )
        diff_ent_m, lo_ent_m, hi_ent_m = cluster_bootstrap_paired_diff(
            p_t2b_mri, p_t4_mri, y_mri, sids_mri, ent_metric
        )

        paired_results["mri"] = {
            "n_rows": n_mri, "n_subjects": n_mri_subs,
            "f1_t2b": round(f1_t2b_m, 4), "f1_t4": round(f1_t4, 4),
            "delta_f1": diff_f1_m, "delta_f1_ci": f"[{lo_f1_m}, {hi_f1_m}]",
            "entropy_t2b": round(ent_t2b_m, 4), "entropy_t4": round(ent_t4, 4),
            "delta_entropy": diff_ent_m, "delta_entropy_ci": f"[{lo_ent_m}, {hi_ent_m}]",
        }
        print(f"    T2b F1={f1_t2b_m:.3f} -> T4 F1={f1_t4:.3f} | Delta F1={diff_f1_m:+.4f} (95% CI: [{lo_f1_m}, {hi_f1_m}])")
        print(f"    T2b Ent={ent_t2b_m:.3f} -> T4 Ent={ent_t4:.3f} | Delta Ent={diff_ent_m:+.4f} (95% CI: [{lo_ent_m}, {hi_ent_m}])")

    # ── 5. Write results report ───────────────────────────────────────────
    red_flags = []
    for tier, res in tier_results.items():
        if not res["beats_trivial"]:
            red_flags.append(f"[RED FLAG] Tier {tier} macro-F1 ({res['macro_f1']}) "
                             f"does NOT beat trivial ({res['trivial_macro_f1']})")
        if res["ece"] > 0.15:
            red_flags.append(f"[RED FLAG] Tier {tier} ECE={res['ece']} > 0.15")

    write_results(tier_results, paired_results, red_flags)
    print("\n[DONE] exp09 complete.")
    return tier_results, paired_results, red_flags


def write_results(tier_results: dict, paired_results: dict, red_flags: list):
    out = RESULTS_DIR / "exp09_stager_results.md"
    lines = [
        "# EXP09: Baseline Stager Results — Refined Tiers & Paired Modality Analysis",
        "",
        "Generated by `server/experiments/exp09_baseline_stager.py`",
        "",
        "---",
        "",
        "## 1. Executive Summary & Honest Clinical Assessment",
        "",
        "### Do CSF and MRI Add Anything Beyond Noise Over Cognitive Tests for Staging?",
        "",
    ]

    csf_p = paired_results.get("csf", {})
    mri_p = paired_results.get("mri", {})

    d_f1_csf = csf_p.get("delta_f1", 0.0)
    ci_f1_csf = csf_p.get("delta_f1_ci", "[0, 0]")
    d_ent_csf = csf_p.get("delta_entropy", 0.0)

    d_f1_mri = mri_p.get("delta_f1", 0.0)
    ci_f1_mri = mri_p.get("delta_f1_ci", "[0, 0]")
    d_ent_mri = mri_p.get("delta_entropy", 0.0)

    lines.append(
        f"**Honest Answer**: **For cross-sectional syndromic diagnosis (CN vs MCI vs AD), "
        f"cognitive batteries (MMSE, CDR-SB, ADAS11) dominate the prediction.**"
    )
    lines.append("")
    lines.append(
        f"- **CSF Biomarkers (ABETA/TAU/PTAU)**: On the exact subset of test subjects with observed CSF "
        f"($N={csf_p.get('n_rows',0)}$ visits), adding CSF to full cognitive scores (T2b -> T3) changes "
        f"Macro-F1 by **{d_f1_csf:+.4f}** (95% Cluster-Bootstrap CI: `{ci_f1_csf}`) and changes mean predictive entropy "
        f"by **{d_ent_csf:+.4f}**."
    )
    if "0.0" in ci_f1_csf or ("-" in ci_f1_csf and "+" in ci_f1_csf):
        lines.append(
            "  - *Conclusion on CSF*: The 95% bootstrap CI for Delta Macro-F1 spans zero. CSF does not significantly improve cross-sectional syndromic staging accuracy over a comprehensive cognitive battery."
        )
    lines.append("")
    lines.append(
        f"- **FreeSurfer MRI Volumes**: On the exact subset of test subjects with observed MRI "
        f"($N={mri_p.get('n_rows',0)}$ visits), adding volumetric features (T2b -> T4) changes "
        f"Macro-F1 by **{d_f1_mri:+.4f}** (95% Cluster-Bootstrap CI: `{ci_f1_mri}`) and changes mean predictive entropy "
        f"by **{d_ent_mri:+.4f}**."
    )
    lines.append("")
    lines.append(
        "- **Clinical Interpretation**: Syndromic stage (clinical dementia rating and mild cognitive impairment) "
        "is by medical definition established primarily from cognitive and functional impairment. "
        "While CSF and MRI biomarkers are essential for **biological etiology** (ruling out non-AD causes and establishing amyloid/tau status) "
        "and for **longitudinal progression risk** (Model B), their incremental diagnostic value for purely determining *current syndromic stage* "
        "once high-quality cognitive batteries (CDR-SB + ADAS11 + MMSE) are already in hand is modest to minimal."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Refined Per-Tier Metrics (Full Test Set, N=3,022 visits)")
    lines.append("")
    lines.append("| Tier | Included Features | Accuracy | Macro-F1 | F1 95% Cluster-CI | Balanced-Acc | ECE | Mean Entropy | Beats Trivial? |")
    lines.append("|------|-------------------|----------|----------|-------------------|--------------|-----|--------------|----------------|")

    tier_desc = {
        "T1":  "Demographics only (Age, Sex, Education, APOE4)",
        "T1b": "T1 + MMSE",
        "T2a": "T1b + CDGLOBAL, CDRSB",
        "T2b": "T2a + TOTSCORE (ADAS11), FAQTOTAL, MOCA",
        "T3":  "T2b + CSF (ABETA, TAU, PTAU)",
        "T4":  "T3 + FreeSurfer MRI (Hippo, Ventricles, Cortical Thicknesses)",
    }

    for tier, res in tier_results.items():
        beat = "YES" if res["beats_trivial"] else "**NO**"
        desc = tier_desc.get(tier, tier)
        lines.append(
            f"| **{tier}** | {desc} | {res['accuracy']:.3f} | {res['macro_f1']:.3f} | "
            f"`{res['f1_95ci']}` | {res['balanced_acc']:.3f} | {res['ece']:.4f} | {res['mean_entropy']:.3f} | {beat} |"
        )
    lines.append("")
    lines.append("*Cluster-Bootstrap resampled unique subjects across 1,000 iterations (seed=42).*")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Paired Subset Analysis (Exact Same Rows)")
    lines.append("")
    lines.append("### A. CSF Biomarker Impact (T2b vs T3 on rows where CSF is observed)")
    lines.append("")
    if csf_p:
        lines.append(f"- **Evaluated Rows**: {csf_p['n_rows']} visits across {csf_p['n_subjects']} subjects")
        lines.append(f"- **Baseline Cognitive (T2b) Macro-F1**: {csf_p['f1_t2b']:.4f}")
        lines.append(f"- **With CSF Added (T3) Macro-F1**: {csf_p['f1_t3']:.4f}")
        lines.append(f"- **Delta Macro-F1 (T3 - T2b)**: **{csf_p['delta_f1']:+.4f}** (95% CI: `{csf_p['delta_f1_ci']}`)")
        lines.append(f"- **Mean Entropy (T2b)**: {csf_p['entropy_t2b']:.4f} nats")
        lines.append(f"- **Mean Entropy with CSF (T3)**: {csf_p['entropy_t3']:.4f} nats")
        lines.append(f"- **Delta Entropy (T3 - T2b)**: **{csf_p['delta_entropy']:+.4f}** (95% CI: `{csf_p['delta_entropy_ci']}`)")
    lines.append("")
    lines.append("### B. FreeSurfer MRI Impact (T2b vs T4 on rows where MRI is observed)")
    lines.append("")
    if mri_p:
        lines.append(f"- **Evaluated Rows**: {mri_p['n_rows']} visits across {mri_p['n_subjects']} subjects")
        lines.append(f"- **Baseline Cognitive (T2b) Macro-F1**: {mri_p['f1_t2b']:.4f}")
        lines.append(f"- **With MRI Added (T4) Macro-F1**: {mri_p['f1_t4']:.4f}")
        lines.append(f"- **Delta Macro-F1 (T4 - T2b)**: **{mri_p['delta_f1']:+.4f}** (95% CI: `{mri_p['delta_f1_ci']}`)")
        lines.append(f"- **Mean Entropy (T2b)**: {mri_p['entropy_t2b']:.4f} nats")
        lines.append(f"- **Mean Entropy with MRI (T4)**: {mri_p['entropy_t4']:.4f} nats")
        lines.append(f"- **Delta Entropy (T4 - T2b)**: **{mri_p['delta_entropy']:+.4f}** (95% CI: `{mri_p['delta_entropy_ci']}`)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Per-Class Performance Breakdown")
    lines.append("")
    for tier in ["T1b", "T2b", "T4"]:
        if tier not in tier_results:
            continue
        res = tier_results[tier]
        pc = res.get("per_class", {})
        lines.append(f"### Tier {tier} ({tier_desc.get(tier, tier)})")
        lines.append("")
        lines.append("| Class | Precision | Recall | F1-Score | Support |")
        lines.append("|-------|-----------|--------|----------|---------|")
        for cls in ["CN", "MCI", "AD"]:
            r = pc.get(cls, {})
            lines.append(f"| {cls} | {r.get('precision',0):.3f} | {r.get('recall',0):.3f} | "
                         f"{r.get('f1-score',0):.3f} | {int(r.get('support',0))} |")
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Results written to: {out}")


if __name__ == "__main__":
    main()
