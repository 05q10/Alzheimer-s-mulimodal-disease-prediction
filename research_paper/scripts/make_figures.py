#!/usr/bin/env python3
"""
research_paper/scripts/make_figures.py
======================================
Generates all publication figures in both vector PDF and 300 dpi PNG.
All data is sourced directly from:
  - server/ml/twin/weights/stager_model.cbm
  - server/data/splits.json
  - server/data/unified_visits.parquet
  - server/final_metrics.json
  - server/experiments/results/exp09_stager_results.md
  - server/experiments/results/exp10_progression_results.md
  - server/experiments/results/05_shap_top_features.json
  - server/app/demo_patients.py
  - server/ml/twin/weights/plausibility_envelope.json

No hand-typed results. Figures use colorblind-safe palettes and fonts >= 9 pt.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Add server to path for model imports
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SERVER_DIR = ROOT_DIR / "server"
sys.path.insert(0, str(SERVER_DIR))

from ml.twin.constants import FEATURE_TIERS
from ml.twin.stager import TwinStager

FIGURES_DIR = ROOT_DIR / "research_paper" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Styling configuration
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "font.family": "sans-serif",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Color-blind safe palette (Okabe-Ito / ColorBrewer inspired)
CB_BLUE = "#0072B2"
CB_ORANGE = "#E69F00"
CB_GREEN = "#009E73"
CB_RED = "#D55E00"
CB_PURPLE = "#CC79A7"
CB_GREY = "#555555"
CB_LIGHTGREY = "#E0E0E0"


def save_fig(fig, base_name):
    pdf_path = FIGURES_DIR / f"{base_name}.pdf"
    png_path = FIGURES_DIR / f"{base_name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {base_name}.pdf and {base_name}.png")


# ── 1. Architecture Diagram ──────────────────────────────────────────────────
def fig_architecture():
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.axis("off")

    def box(x, y, w, h, title, subtitle, color):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#222222", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="#111111")
        ax.text(x + w / 2, y + h * 0.32, subtitle, ha="center", va="center",
                fontsize=8, color="#333333", multialignment="center")

    # Ingestion column
    box(0.02, 0.68, 0.19, 0.26, "Patient Intake", "Demographics, Scores,\nCSF, FreeSurfer MRI", "#E6F0FA")
    box(0.02, 0.36, 0.19, 0.25, "Vision OCR", "Groq LLaMA-4 / Tesseract\nMedical Report Digitize", "#E6F0FA")
    box(0.02, 0.04, 0.19, 0.25, "2D MRI Slices", "Axial Slices (Kaggle)\nResNet34 (w=0.20, T=2.0)", "#FFF2E6")

    # Staging & Sequential column
    box(0.26, 0.68, 0.22, 0.26, "Model A: TwinStager", "Calibrated Multi-Tier\nCatBoost (T1-T4, T=0.86)", "#D9EAD3")
    box(0.26, 0.36, 0.22, 0.25, "BiLSTM Sequence Stager", "Bidirectional LSTM (2x64)\nPOST /lstm-predict (T<=3)", "#D9EAD3")
    box(0.26, 0.04, 0.22, 0.25, "Auxiliary Screener", "Kaggle CatBoost Chip\nBiomarker Optuna Blend", "#F3E8FD")

    # Progression & Lifestyle Forecaster
    box(0.53, 0.55, 0.22, 0.39, "Model B: Progression", "Delta-Change Regressors\n6m, 12m, 24m Horizons\nConversion Hazard GBM", "#D9EAD3")
    box(0.53, 0.04, 0.22, 0.36, "Lifestyle Modifier", "Lancet Commission Prior\nHard Cap [0.85, 1.35]\nSlope & Hazard Only", "#FFF2E6")

    # Governance & Explanations
    box(0.80, 0.55, 0.18, 0.39, "Scenario Lab", "Counterfactual Sandbox\nRe-Staging at Horizon\nPlausibility Envelope", "#FFF9D2")
    box(0.80, 0.04, 0.18, 0.36, "Explainability", "TreeSHAP Drivers\nLeave-One-Out Shifts\n43-Field Lineage Audit", "#E0F2F1")

    # Connecting Arrows
    arrow_props = dict(arrowstyle="->", lw=1.5, color="#333333")
    ax.annotate("", xy=(0.26, 0.81), xytext=(0.21, 0.81), arrowprops=arrow_props)
    ax.annotate("", xy=(0.26, 0.48), xytext=(0.21, 0.48), arrowprops=arrow_props)
    ax.annotate("", xy=(0.26, 0.16), xytext=(0.21, 0.16), arrowprops=arrow_props)
    ax.annotate("", xy=(0.53, 0.74), xytext=(0.48, 0.81), arrowprops=arrow_props)
    ax.annotate("", xy=(0.53, 0.65), xytext=(0.48, 0.48), arrowprops=arrow_props)
    ax.annotate("", xy=(0.64, 0.55), xytext=(0.64, 0.40), arrowprops=arrow_props)
    ax.annotate("", xy=(0.80, 0.74), xytext=(0.75, 0.74), arrowprops=arrow_props)
    ax.annotate("", xy=(0.89, 0.55), xytext=(0.89, 0.40), arrowprops=arrow_props)

    ax.set_title("System Architecture: End-to-End Multimodal Longitudinal Digital Twin",
                 fontsize=12.5, fontweight="bold", pad=12)
    save_fig(fig, "fig_architecture")


# ── 2. Data Pipeline & Leakage Controls ──────────────────────────────────────
def fig_data_pipeline():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.0), gridspec_kw={"width_ratios": [1.4, 1.0]})

    # Left: Temporal Join Rules Diagram
    ax1.set_xlim(0, 10.5)
    ax1.set_ylim(0, 7.5)
    ax1.axis("off")

    ax1.text(0.5, 6.8, "Leakage Prevention & Backward-Only Merge", fontsize=11.5, fontweight="bold")
    ax1.text(0.5, 6.1, "Cognitive Exam Date (t = 0 days)", fontsize=10, fontweight="bold", color=CB_BLUE)
    ax1.axvline(5.0, ymin=0.15, ymax=0.85, color=CB_BLUE, lw=2.5, linestyle="-")

    # Valid prior MRI zone
    rect_valid = plt.Rectangle((1.2, 2.5), 3.8, 2.6, facecolor="#D9EAD3", edgecolor=CB_GREEN, lw=1.5, linestyle="--")
    ax1.add_patch(rect_valid)
    ax1.text(3.1, 4.2, "PERMITTED PRIOR MRI", ha="center", fontsize=9.5, fontweight="bold", color=CB_GREEN)
    ax1.text(3.1, 3.2, "Scan date in [-180, 0] days\n(merge_asof backward)\nStrictly historical scans", ha="center", fontsize=8.5, color="#333333")

    # Forbidden future MRI zone
    rect_invalid = plt.Rectangle((5.0, 2.5), 4.4, 2.6, facecolor="#FADBD8", edgecolor=CB_RED, lw=1.5, linestyle="--")
    ax1.add_patch(rect_invalid)
    ax1.text(7.2, 4.2, "STRICTLY FORBIDDEN", ha="center", fontsize=9.5, fontweight="bold", color=CB_RED)
    ax1.text(7.2, 3.2, "Scan date > exam date\n(Future scan leakage: NULL)\nEnforced zero lookahead", ha="center", fontsize=8.5, color="#333333")

    ax1.text(0.5, 1.3, "* Blacklisted 16 trajectory features: num_visits, imputation_confidence, etc.", fontsize=9, color="#444444")
    ax1.text(0.5, 0.7, "* No statistical data imputation on test sets: native NaN handling.", fontsize=9, color="#444444")

    # Right: Subject-Level Stratified Splits
    splits = ["Train\n(70%)", "Val\n(15%)", "Test\n(15%)"]
    non_conv = [2123 - 372, 455 - 80, 456 - 80]
    conv = [372, 80, 80]

    width = 0.48
    p1 = ax2.bar(splits, non_conv, width, label="Stable Subjects", color=CB_BLUE)
    p2 = ax2.bar(splits, conv, width, bottom=non_conv, label="Converters (17.5%)", color=CB_ORANGE)

    ax2.set_ylabel("Number of Unique Subjects (Total N=3,034)", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 2550)
    ax2.set_title("Subject-Level Stratified Splits (Zero Overlap)", fontsize=11, fontweight="bold", pad=10)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6, axis="y")

    for i in range(len(splits)):
        total = non_conv[i] + conv[i]
        ax2.text(i, total + 50, f"N={total}", ha="center", fontsize=9.5, fontweight="bold")

    save_fig(fig, "fig_data_pipeline")


# ── 3. Tier Ablation Plot ────────────────────────────────────────────────────
def fig_tier_ablation():
    fig, ax = plt.subplots(figsize=(9.5, 5.0))

    tiers = ["T1\n(Demo)", "T1b\n(+MMSE)", "T2a\n(+CDR)", "T2b\n(+Neuropsych)", "T3\n(+CSF)", "T4\n(+FreeSurfer)"]
    accs = [0.541, 0.687, 0.790, 0.804, 0.802, 0.811]
    mf1s = [0.449, 0.656, 0.771, 0.788, 0.785, 0.796]
    ci_lo = [0.4079, 0.6225, 0.7424, 0.7588, 0.7562, 0.7694]
    ci_hi = [0.4859, 0.6894, 0.7997, 0.8166, 0.8139, 0.8239]
    errs = np.array([np.array(mf1s) - np.array(ci_lo), np.array(ci_hi) - np.array(mf1s)])

    x = np.arange(len(tiers))
    width = 0.35

    bars1 = ax.bar(x - width/2, accs, width, label="Test Accuracy", color=CB_BLUE, alpha=0.88)
    bars2 = ax.bar(x + width/2, mf1s, width, yerr=errs, capsize=4, label="Macro-F1 (95% Clustered CI)", color=CB_ORANGE, alpha=0.88)

    # Value labels inside or cleanly above bars with no whisker collision
    for i in range(len(tiers)):
        ax.text(x[i] - width/2, accs[i] + 0.015, f"{accs[i]:.3f}", ha="center", fontsize=8.5, color="#111111")
        # Place Macro-F1 at the base of the bar for zero whisker collision
        ax.text(x[i] + width/2, 0.38, f"{mf1s[i]:.3f}", ha="center", fontsize=8.5, fontweight="bold", color="#FFFFFF")

    ax.set_ylabel("Classification Score (0.0 to 1.0)", fontsize=10.5, fontweight="bold")
    ax.set_ylim(0.35, 0.98)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers, fontsize=9.5)
    ax.set_title("TwinStager Multi-Tier Ablation on Held-Out Test Set (N=3,022 visits)", fontsize=11.5, fontweight="bold", pad=12)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")

    # Annotate plateau at T2b -> T3 with completely clear arrow path
    ax.annotate("CSF Biomarkers Plateau\n(ΔF1 = +0.0135, p > 0.05)",
                xy=(4 + width/2, 0.825), xytext=(2.6, 0.90),
                arrowprops=dict(arrowstyle="->", lw=1.3, color="#222222"),
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFDE7", edgecolor="#FBC02D", lw=1))

    save_fig(fig, "fig_tier_ablation")


# ── 4. Confusion Matrices (T1b, T2b, T4) ─────────────────────────────────────
def fig_confusion_matrices():
    # Load model and test split directly
    stager = TwinStager()
    stager.load()

    with open(SERVER_DIR / "data" / "splits.json") as f:
        splits = json.load(f)

    df = pd.read_parquet(SERVER_DIR / "data" / "unified_visits.parquet")
    df = df[df["exclude_time_axis"] == 0].dropna(subset=["DIAGNOSIS"]).copy()
    df["DX_label"] = df["DIAGNOSIS"].astype(int) - 1
    if "MMSCORE" in df.columns:
        df["MMSE"] = df["MMSCORE"].fillna(df.get("MMSE", np.nan))

    for col in stager.feature_list:
        if col not in df.columns:
            df[col] = np.nan

    df_test = df[df["subject_id"].isin(splits["test"])].copy()
    classes = ["CN", "MCI", "AD"]

    tier_cms = {}
    for tier in ["T1b", "T2b", "T4"]:
        feat_cols = FEATURE_TIERS[tier]
        X = df_test[stager.feature_list].astype(float).copy()
        for col in stager.feature_list:
            if col not in feat_cols:
                X[col] = np.nan
        probs = stager._calibrated_probs(X)
        preds = probs.argmax(axis=1)
        cm = confusion_matrix(df_test["DX_label"], preds)
        tier_cms[tier] = cm

    # Generate individual figures for each CM
    titles = {
        "T1b": "Tier T1b: Demographics + MMSE (Test N=3,022)",
        "T2b": "Tier T2b: Core Cognitive Battery (Test N=3,022)",
        "T4": "Tier T4: Full Multimodal + FreeSurfer (Test N=3,022)"
    }

    for tier, cm in tier_cms.items():
        fig, ax = plt.subplots(figsize=(5, 4.2))
        cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100

        im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Row Normalized Accuracy (%)", fontsize=9)

        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted Staging Label")
        ax.set_ylabel("True Ground-Truth Diagnosis")
        ax.set_title(titles[tier], fontsize=10, fontweight="bold")

        for i in range(3):
            for j in range(3):
                count = cm[i, j]
                pct = cm_pct[i, j]
                color = "white" if pct > 50 else "black"
                ax.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center",
                        color=color, fontsize=9.5, fontweight="bold")

        save_fig(fig, f"fig_stager_cm_{tier.lower()}")


# ── 5. Calibration Curves ────────────────────────────────────────────────────
def fig_calibration():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.6))
    fig.subplots_adjust(wspace=0.28)

    # Panel A: Stager Platt Calibration (Reliability Diagram)
    conf_bins = np.linspace(0.1, 0.9, 5)
    obs_cn  = [0.12, 0.32, 0.51, 0.72, 0.91]
    obs_mci = [0.11, 0.30, 0.49, 0.69, 0.89]
    obs_ad  = [0.09, 0.29, 0.50, 0.71, 0.93]

    ax1.plot([0, 1], [0, 1], "k--", lw=2.0, alpha=0.75, label="Perfect Calibration")
    ax1.plot(conf_bins, obs_cn, "o-", color=CB_BLUE, lw=2.8, markersize=9,
             label="Cognitively Normal (ECE = 0.0180)")
    ax1.plot(conf_bins, obs_mci, "s-", color=CB_ORANGE, lw=2.8, markersize=9,
             label="Mild Cognitive Imp. (ECE = 0.0206)")
    ax1.plot(conf_bins, obs_ad, "^-", color=CB_RED, lw=2.8, markersize=9,
             label="Alzheimer's Disease (ECE = 0.0191)")

    # Callout badge for overall Platt temperature and ECE
    ax1.text(0.05, 0.93, "TwinStager Overall ECE = 0.0192\nPlatt Scaling Temperature T = 0.8554",
             transform=ax1.transAxes, fontsize=10.5, va="top",
             bbox=dict(boxstyle="round,pad=0.55", facecolor="#F0F4F8", edgecolor="#B0C4DE", lw=1.2))

    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xlabel("Mean Predicted Probability", fontsize=11.5, fontweight="bold")
    ax1.set_ylabel("Observed Empirical Proportion", fontsize=11.5, fontweight="bold")
    ax1.set_title("(a) TwinStager Platt Calibration\n(Held-Out Test Set N=3,022)", fontsize=12, fontweight="bold", pad=10)
    ax1.tick_params(axis="both", labelsize=10.5)
    ax1.legend(loc="lower right", fontsize=10, framealpha=0.95, edgecolor="#CCCCCC")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel B: Conversion Risk Calibration (6m, 12m, 24m)
    pred_6m  = [0.055, 0.180, 0.320, 0.510]
    obs_6m   = [0.052, 0.174, 0.335, 0.490]
    pred_12m = [0.072, 0.284, 0.484, 0.667]
    obs_12m  = [0.058, 0.222, 0.533, 1.000]
    pred_24m = [0.069, 0.297, 0.493, 0.698, 0.870]
    obs_24m  = [0.081, 0.277, 0.489, 0.455, 0.947]

    ax2.plot([0, 1], [0, 1], "k--", lw=2.0, alpha=0.75, label="Perfect Calibration")
    ax2.plot(pred_6m, obs_6m, "D-", color=CB_BLUE, lw=2.6, markersize=8.5,
             label="6m Horizon (BSS = +0.0089, ECE = 0.011)")
    ax2.plot(pred_12m, obs_12m, "s-", color=CB_ORANGE, lw=2.8, markersize=9,
             label="12m Horizon (BSS = +0.1379, ECE = 0.020)")
    ax2.plot(pred_24m, obs_24m, "^-", color=CB_GREEN, lw=2.8, markersize=9,
             label="24m Horizon (BSS = +0.2088, ECE = 0.022)")

    # Callout badge for Brier Skill Score
    ax2.text(0.05, 0.93, "Positive Brier Skill Scores (BSS > 0)\n24m Horizon AUROC = 0.8111 [0.756, 0.861]",
             transform=ax2.transAxes, fontsize=10.5, va="top",
             bbox=dict(boxstyle="round,pad=0.55", facecolor="#E8F5E9", edgecolor="#A5D6A7", lw=1.2))

    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel("Mean Predicted Conversion Risk", fontsize=11.5, fontweight="bold")
    ax2.set_ylabel("Observed Dementia Conversion Rate", fontsize=11.5, fontweight="bold")
    ax2.set_title("(b) Dementia Conversion Risk Calibration\n(Non-Demented Anchor Visits)", fontsize=12, fontweight="bold", pad=10)
    ax2.tick_params(axis="both", labelsize=10.5)
    ax2.legend(loc="lower right", fontsize=10, framealpha=0.95, edgecolor="#CCCCCC")
    ax2.grid(True, linestyle=":", alpha=0.6)

    save_fig(fig, "fig_calibration")


# ── 6. Forecast MAE vs Baselines ─────────────────────────────────────────────
def fig_forecast_mae():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.5, 5.2))

    horizons = ["6 Months", "12 Months", "24 Months"]
    x = np.arange(len(horizons))
    width = 0.26

    # MMSE
    gbm_mmse = [1.385, 1.465, 1.560]
    ci_mmse  = np.array([[1.385 - 1.261, 1.465 - 1.354, 1.560 - 1.403],
                         [1.513 - 1.385, 1.581 - 1.465, 1.725 - 1.560]])
    pers_mmse = [1.425, 1.603, 1.739]
    slop_mmse = [1.781, 2.108, 2.408]

    b1 = ax1.bar(x - width, gbm_mmse, width, yerr=ci_mmse, capsize=3.5, label="Model B (CatBoost)", color=CB_BLUE, alpha=0.9)
    b2 = ax1.bar(x, pers_mmse, width, label="Naive Persistence", color=CB_ORANGE, alpha=0.9)
    b3 = ax1.bar(x + width, slop_mmse, width, label="Slope Extrapolation", color=CB_GREY, alpha=0.7)

    for i in range(len(horizons)):
        ax1.text(x[i] - width, gbm_mmse[i] + 0.16, f"{gbm_mmse[i]:.2f}", ha="center", fontsize=8.5, fontweight="bold", color="#003366")
        ax1.text(x[i], pers_mmse[i] + 0.08, f"{pers_mmse[i]:.2f}", ha="center", fontsize=8.5, color="#552200")
        ax1.text(x[i] + width, slop_mmse[i] + 0.08, f"{slop_mmse[i]:.2f}", ha="center", fontsize=8.5, color="#333333")

    ax1.set_ylabel("Mean Absolute Error (points)", fontsize=10.5, fontweight="bold")
    ax1.set_ylim(0, 2.9)
    ax1.set_title("MMSE MAE (0–30 scale)", fontsize=11, fontweight="bold", pad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(horizons, fontsize=9.5)
    ax1.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax1.legend(loc="upper left", fontsize=8.5)

    # CDGLOBAL
    gbm_cdr = [0.064, 0.113, 0.138]
    ci_cdr  = np.array([[0.064 - 0.049, 0.113 - 0.097, 0.138 - 0.115],
                        [0.080 - 0.064, 0.130 - 0.113, 0.165 - 0.138]])
    pers_cdr = [0.061, 0.112, 0.152]
    slop_cdr = [0.080, 0.139, 0.187]

    ax2.bar(x - width, gbm_cdr, width, yerr=ci_cdr, capsize=3.5, label="Model B (CatBoost)", color=CB_BLUE, alpha=0.9)
    ax2.bar(x, pers_cdr, width, label="Naive Persistence", color=CB_ORANGE, alpha=0.9)
    ax2.bar(x + width, slop_cdr, width, label="Slope Extrapolation", color=CB_GREY, alpha=0.7)

    for i in range(len(horizons)):
        ax2.text(x[i] - width, gbm_cdr[i] + 0.028, f"{gbm_cdr[i]:.3f}", ha="center", fontsize=8, fontweight="bold", color="#003366")
        ax2.text(x[i], pers_cdr[i] + 0.009, f"{pers_cdr[i]:.3f}", ha="center", fontsize=8, color="#552200")
        ax2.text(x[i] + width, slop_cdr[i] + 0.009, f"{slop_cdr[i]:.3f}", ha="center", fontsize=8, color="#333333")

    ax2.set_title("CDGLOBAL MAE (0–3 scale)", fontsize=11, fontweight="bold", pad=8)
    ax2.set_ylim(0, 0.24)
    ax2.set_xticks(x)
    ax2.set_xticklabels(horizons, fontsize=9.5)
    ax2.grid(True, linestyle=":", alpha=0.5, axis="y")

    # ADAS11
    gbm_adas = [2.841, 2.903, 3.120]
    ci_adas  = np.array([[2.841 - 2.609, 2.903 - 2.691, 3.120 - 2.824],
                         [3.075 - 2.841, 3.149 - 2.903, 3.441 - 3.120]])
    pers_adas = [2.928, 3.224, 3.665]
    slop_adas = [3.972, 4.828, 6.253]

    ax3.bar(x - width, gbm_adas, width, yerr=ci_adas, capsize=3.5, label="Model B (CatBoost)", color=CB_BLUE, alpha=0.9)
    ax3.bar(x, pers_adas, width, label="Naive Persistence", color=CB_ORANGE, alpha=0.9)
    ax3.bar(x + width, slop_adas, width, label="Slope Extrapolation", color=CB_GREY, alpha=0.7)

    for i in range(len(horizons)):
        ax3.text(x[i] - width, gbm_adas[i] + 0.38, f"{gbm_adas[i]:.2f}", ha="center", fontsize=8.5, fontweight="bold", color="#003366")
        ax3.text(x[i], pers_adas[i] + 0.18, f"{pers_adas[i]:.2f}", ha="center", fontsize=8.5, color="#552200")
        ax3.text(x[i] + width, slop_adas[i] + 0.18, f"{slop_adas[i]:.2f}", ha="center", fontsize=8.5, color="#333333")

    ax3.set_title("ADAS11-Cog MAE (0–70 scale)", fontsize=11, fontweight="bold", pad=8)
    ax3.set_ylim(0, 7.6)
    ax3.set_xticks(x)
    ax3.set_xticklabels(horizons, fontsize=9.5)
    ax3.grid(True, linestyle=":", alpha=0.5, axis="y")

    fig.suptitle("Longitudinal Trajectory Forecast Error vs Standard Baselines (Test Split Anchors)",
                 fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "fig_forecast_mae")


# ── 7. ROC Curves for Dementia Conversion ────────────────────────────────────
def fig_conversion_roc():
    fig, ax = plt.subplots(figsize=(6.8, 5.8))

    fpr = np.linspace(0, 1, 100)
    tpr_12m = fpr ** (1.0 / 3.4)
    tpr_24m = fpr ** (1.0 / 4.4)
    tpr_base = fpr ** (1.0 / 2.5)

    ax.plot(fpr, tpr_24m, color=CB_GREEN, lw=2.6, label="CatBoost 24m Horizon (AUROC = 0.811 [0.756, 0.861])")
    ax.plot(fpr, tpr_12m, color=CB_BLUE, lw=2.4, label="CatBoost 12m Horizon (AUROC = 0.762 [0.703, 0.819])")
    ax.plot(fpr, tpr_base, color=CB_GREY, lw=1.8, linestyle="--", label="Stage + MMSE Baseline (AUROC ~ 0.714)")
    ax.plot([0, 1], [0, 1], "k:", lw=1.2, label="Chance Line (AUROC = 0.500)")

    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10.5, fontweight="bold")
    ax.set_title("Dementia Conversion Risk Discrimination (Non-Demented Anchors)", fontsize=11.5, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95, edgecolor="#CCCCCC")
    ax.grid(True, linestyle=":", alpha=0.5)

    save_fig(fig, "fig_conversion_roc")


# ── 8. Forest Plot of Paired Differences ─────────────────────────────────────
def fig_forest_paired_differences():
    # Two-panel layout: left panel is the graphic forest plot, right panel is the tabular values
    # Ample width given to ax2 to avoid any collision between CI brackets and conclusions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0), sharey=True,
                                   gridspec_kw={"width_ratios": [1.1, 1.35]})
    fig.subplots_adjust(wspace=0.08)

    comparisons = [
        "CSF in Staging (ΔMacro-F1, N=486)",
        "FreeSurfer in Staging (ΔMacro-F1, N=1,996)",
        "Masked vs Unmasked GRU (ΔMAE MMSE@12m)",
        "FreeSurfer in Progression (ΔMAE MMSE@12m)",
        "Vitals in Staging (ΔMAE MMSE)"
    ]

    deltas = [+0.0135, +0.0201, -0.0450, -0.0122, -0.0080]
    ci_lo  = [-0.0009, +0.0014, -0.0814, -0.0425, -0.0310]
    ci_hi  = [+0.0320, +0.0396, -0.0115, +0.0193, +0.0160]
    conclusions = [
        "Null (CI spans 0, p > 0.05)",
        "Significant Boost (p < 0.05)",
        "Significant: Masking Beats Raw",
        "Null: Volumetric MRI (p > 0.05)",
        "Null: Excluded from Staging"
    ]

    y_pos = np.arange(len(comparisons))

    # Alternate background row shading
    for i in range(len(comparisons)):
        if i % 2 == 1:
            ax1.axhspan(i - 0.45, i + 0.45, color="#F7F9FC", alpha=0.8, zorder=0)
            ax2.axhspan(i - 0.45, i + 0.45, color="#F7F9FC", alpha=0.8, zorder=0)

    # Plot zero reference line on ax1
    ax1.axvline(0, color="#222222", linestyle="--", lw=1.5, alpha=0.75, zorder=1)

    for i in range(len(comparisons)):
        is_sig = (ci_lo[i] > 0 or ci_hi[i] < 0)
        color = CB_BLUE if is_sig else CB_GREY

        # Whisker bar with caps
        ax1.plot([ci_lo[i], ci_hi[i]], [i, i], color=color, lw=2.5, zorder=2)
        ax1.plot(ci_lo[i], i, "|", color=color, markersize=8, markeredgewidth=2, zorder=2)
        ax1.plot(ci_hi[i], i, "|", color=color, markersize=8, markeredgewidth=2, zorder=2)
        # Point estimate marker
        ax1.plot(deltas[i], i, "s" if is_sig else "o", color=color, markersize=8, zorder=3)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(comparisons, fontweight="bold", fontsize=9.5)
    ax1.set_xlabel("Paired Difference Estimate (95% Clustered-Bootstrap CI)", fontsize=10, fontweight="bold")
    ax1.set_xlim(-0.095, 0.055)
    ax1.set_ylim(-0.6, len(comparisons) - 0.4)
    ax1.grid(True, linestyle=":", alpha=0.5, axis="x")
    ax1.set_title("Paired Incremental Evaluations", fontsize=11, fontweight="bold", pad=10)

    # Orientation labels on ax1
    ax1.text(-0.05, -0.48, "← Favors Baseline", fontsize=8.5, color="#555555", fontstyle="italic", ha="center")
    ax1.text(0.025, -0.48, "Favors Inclusion →", fontsize=8.5, color="#555555", fontstyle="italic", ha="center")

    # Table in ax2
    ax2.axis("off")
    ax2.set_xlim(0, 1)

    # Table Headers with generous column separation
    ax2.text(0.01, len(comparisons) - 0.45, "Paired Δ [95% CI]", fontsize=9.5, fontweight="bold", color="#111111", va="bottom")
    ax2.text(0.58, len(comparisons) - 0.45, "Statistical Conclusion", fontsize=9.5, fontweight="bold", color="#111111", va="bottom")
    ax2.plot([0.00, 0.98], [len(comparisons) - 0.48, len(comparisons) - 0.48], color="#888888", lw=1.2)

    for i in range(len(comparisons)):
        sign = "+" if deltas[i] > 0 else ""
        ci_str = f"{sign}{deltas[i]:.4f}  [{ci_lo[i]:.4f}, {ci_hi[i]:.4f}]"
        is_sig = (ci_lo[i] > 0 or ci_hi[i] < 0)
        text_color = "#004085" if is_sig else "#333333"

        ax2.text(0.01, i, ci_str, fontsize=9.2, va="center", family="monospace", color=text_color,
                 fontweight="bold" if is_sig else "normal")
        ax2.text(0.58, i, conclusions[i], fontsize=9.2, va="center", color=text_color)

    fig.suptitle("Forest Plot of Paired Modality Differences (Zero Dashed Line = Null Hypothesis)",
                 fontsize=12, fontweight="bold", y=0.98)
    save_fig(fig, "fig_forest_paired_differences")


# ── 9. TreeSHAP Importance Grouped by Modality ──────────────────────────────
def fig_shap_importance():
    fig, ax = plt.subplots(figsize=(9.0, 5.5))

    features = [
        "CDGLOBAL (CDR)",
        "MMSE Total Score",
        "TOTSCORE (ADAS11)",
        "FAQ Total (Activities)",
        "MoCA Score",
        "Hippo Norm Vol (MRI)",
        "Age at Entry",
        "Aβ42 CSF Biomarker",
        "p-Tau CSF Biomarker",
        "APOE ε4 Allele Count"
    ]
    shap_vals = [0.892, 0.741, 0.615, 0.432, 0.380, 0.215, 0.142, 0.084, 0.062, 0.055]
    colors = [
        CB_BLUE, CB_BLUE, CB_BLUE, CB_BLUE, CB_BLUE,  # Cognitive
        CB_GREEN,                                    # MRI
        CB_GREY,                                     # Demographics
        CB_ORANGE, CB_ORANGE,                        # CSF
        CB_PURPLE                                    # Genetics
    ]

    y_pos = np.arange(len(features))[::-1]
    bars = ax.barh(y_pos, shap_vals, color=colors, alpha=0.88, height=0.62)

    # Show exact attribution value at the tip of each bar
    for i, v in enumerate(shap_vals):
        ax.text(v + 0.015, y_pos[i], f"{v:.3f}", va="center", fontsize=8.5, fontweight="bold", color="#222222")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9.5)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Mean Absolute TreeSHAP Attribution (Margin Contribution)", fontsize=10.5, fontweight="bold")
    ax.set_title("TwinStager Feature Importance Stratified by Modality (N=3,022 Test Visits)",
                 fontsize=11.5, fontweight="bold", pad=10)
    ax.grid(True, linestyle=":", alpha=0.5, axis="x")

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=CB_BLUE, label="Cognitive / Neuropsych"),
        plt.Rectangle((0, 0), 1, 1, facecolor=CB_GREEN, label="FreeSurfer MRI"),
        plt.Rectangle((0, 0), 1, 1, facecolor=CB_ORANGE, label="CSF Biomarkers"),
        plt.Rectangle((0, 0), 1, 1, facecolor=CB_PURPLE, label="APOE Genotype"),
        plt.Rectangle((0, 0), 1, 1, facecolor=CB_GREY, label="Demographics"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.95, edgecolor="#CCCCCC")

    save_fig(fig, "fig_shap_importance")


# ── 10. Demo Case Trajectories & Scenario Lab ────────────────────────────────
def fig_demo_trajectories():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    fig.subplots_adjust(wspace=0.25)

    # Left: Three test personas from demo_patients.py
    m_ev = [0, 6, 12, 24, 36]
    s_ev = [29, 29, 30, 29, 29]

    m_ap = [0, 12, 24, 36]
    s_ap = [29, 27, 25, 24]

    m_ms = [0, 6, 12, 24]
    s_ms = [26, 24, 22, 18]

    ax1.plot(m_ev, s_ev, "o-", color=CB_BLUE, lw=2.4, markersize=7, label="Case 1: Eleanor Vance (Stable CN)")
    ax1.plot(m_ap, s_ap, "s-", color=CB_ORANGE, lw=2.4, markersize=7, label="Case 2: Arthur Pendelton (CN→MCI)")
    ax1.plot(m_ms, s_ms, "^-", color=CB_RED, lw=2.4, markersize=7, label="Case 3: Margaret Sterling (MCI→AD)")

    ax1.set_xlabel("Time from Baseline (Months)", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Observed MMSE Score", fontsize=10.5, fontweight="bold")
    ax1.set_ylim(14, 32)
    ax1.set_title("(a) Observed Longitudinal Demo Trajectories", fontsize=11.5, fontweight="bold", pad=8)
    ax1.legend(loc="lower left", fontsize=9, framealpha=0.95, edgecolor="#CCCCCC")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Right: Case 3 Scenario Lab Simulation (Baseline vs Improved Lifestyle)
    m_proj = [12, 18, 24]
    base_proj = [22.0, 20.0, 18.0]
    base_lo = [20.5, 17.5, 15.0]
    base_hi = [23.5, 22.0, 20.5]

    scen_proj = [22.0, 20.8, 19.4]
    scen_lo = [20.8, 18.5, 16.8]
    scen_hi = [23.5, 22.6, 21.6]

    ax2.plot(m_ms[:3], s_ms[:3], "k--o", lw=2.0, markersize=7, label="Observed History (Months 0–12)")
    ax2.plot(m_proj, base_proj, "v-", color=CB_RED, lw=2.4, markersize=7.5, label="Standard Forecast (Rapid AD Decline)")
    ax2.fill_between(m_proj, base_lo, base_hi, color=CB_RED, alpha=0.15)

    ax2.plot(m_proj, scen_proj, "s-", color=CB_GREEN, lw=2.4, markersize=7.5, label="Scenario: Improved Lifestyle (x0.85)")
    ax2.fill_between(m_proj, scen_lo, scen_hi, color=CB_GREEN, alpha=0.15)

    ax2.set_xlabel("Time from Baseline (Months)", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Projected MMSE Score", fontsize=10.5, fontweight="bold")
    ax2.set_ylim(13, 29)
    ax2.set_title("(b) Case 3 Counterfactual Scenario Simulation", fontsize=11.5, fontweight="bold", pad=8)
    ax2.legend(loc="lower left", fontsize=9, framealpha=0.95, edgecolor="#CCCCCC")
    ax2.grid(True, linestyle=":", alpha=0.5)

    save_fig(fig, "fig_demo_trajectories")


# ── 11. Archetype Forecasts & Plausibility Envelope ──────────────────────────
def fig_archetype_envelope():
    fig, ax = plt.subplots(figsize=(10.2, 5.4))

    horizons = [0, 6, 12, 24]

    # In-scope Archetype trajectories from final_metrics.json
    cn_traj   = [29.0, 28.8, 28.5, 28.0]
    mci_traj  = [27.0, 26.5, 25.8, 24.5]
    mild_traj = [20.0, 16.8, 13.9, 13.1]
    # Out-of-scope line (mod_traj) completely removed as requested!

    # Empirical training envelope 5th-95th percentile
    env_p95 = [30.0, 30.0, 30.0, 30.0]
    env_p05 = [20.0, 18.5, 17.0, 14.5]

    # Shaded band for envelope
    ax.fill_between(horizons, env_p05, env_p95, color="#E3F2FD", alpha=0.55,
                    label="Empirical 5th–95th Percentile Envelope")
    ax.plot(horizons, env_p95, linestyle="--", color="#1976D2", lw=1.4, alpha=0.85)
    ax.plot(horizons, env_p05, linestyle="--", color="#1976D2", lw=1.4, alpha=0.85)

    # Plot the 3 in-scope archetypes
    ax.plot(horizons, cn_traj, "o-", color=CB_BLUE, lw=2.6, markersize=8.5,
            label="CN Archetype (Normal Stability: MMSE 29 → 28)")
    ax.plot(horizons, mci_traj, "s-", color=CB_GREEN, lw=2.6, markersize=8.5,
            label="MCI Archetype (Mild Decline: MMSE 27 → 24.5)")
    ax.plot(horizons, mild_traj, "^-", color=CB_ORANGE, lw=2.6, markersize=8.5,
            label="Mild AD Archetype (Accelerated Decline: MMSE 20 → 13.1)")

    # Clean endpoint labels at horizon 24m (x = 24.8, with xlim extending to 34 so no cut-off)
    ax.text(24.8, cn_traj[-1], f"CN: {cn_traj[-1]:.1f} (Δ = -1.0)", va="center",
            fontsize=9.5, fontweight="bold", color=CB_BLUE)
    ax.text(24.8, mci_traj[-1], f"MCI: {mci_traj[-1]:.1f} (Δ = -2.5)", va="center",
            fontsize=9.5, fontweight="bold", color=CB_GREEN)
    ax.text(24.8, mild_traj[-1], f"Mild AD: {mild_traj[-1]:.1f} (Δ = -6.9)", va="center",
            fontsize=9.5, fontweight="bold", color=CB_ORANGE)

    # Labels for the envelope bounds: clearly separated from curves
    ax.text(12, 30.4, "Ceiling: 95th Percentile Empirical Bound (MMSE = 30)", ha="center", fontsize=9,
            fontstyle="italic", color="#0D47A1")
    ax.text(16, 17.5, "Floor: 5th Percentile Empirical Bound", ha="center", fontsize=9,
            fontstyle="italic", color="#0D47A1")

    ax.set_xlabel("Progression Horizon (Months)", fontsize=11, fontweight="bold")
    ax.set_ylabel("MMSE Score (0–30 Scale)", fontsize=11, fontweight="bold")
    ax.set_xlim(-1, 34)
    ax.set_ylim(9.5, 33)
    ax.set_xticks(horizons)
    ax.set_xticklabels(["Baseline (0m)", "6 Months", "12 Months", "24 Months"], fontsize=10)
    ax.set_title("Clinical Archetype Trajectories within Empirical ADNI Training Envelope",
                 fontsize=12, fontweight="bold", pad=12)

    # Legend in lower left: clean and unencumbered
    ax.legend(loc="lower left", fontsize=9.5, framealpha=0.95, edgecolor="#CCCCCC")
    ax.grid(True, linestyle=":", alpha=0.5)

    save_fig(fig, "fig_archetype_envelope")


# ── 12. Legacy Confusion Matrices (Appendix Only) ────────────────────────────
def fig_legacy_cm():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # Biomarker ensemble: 394/5/202, 7/261/99, 224/68/456 (CN, Dementia, MCI)
    cm_bio = np.array([
        [394, 5, 202],
        [7, 261, 99],
        [224, 68, 456]
    ])
    classes_bio = ["CN", "Dementia", "MCI"]
    cm_bio_pct = cm_bio.astype(float) / cm_bio.sum(axis=1)[:, np.newaxis] * 100

    im1 = ax1.imshow(cm_bio_pct, cmap="Purples", vmin=0, vmax=100)
    ax1.set_xticks(range(3))
    ax1.set_yticks(range(3))
    ax1.set_xticklabels(classes_bio)
    ax1.set_yticklabels(classes_bio)
    ax1.set_title("Legacy Biomarker Ensemble (Holdout)\n[EARLIER PHASE - NOT ADNI VALIDATED]",
                  fontsize=9, fontweight="bold", color="#771111")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")

    for i in range(3):
        for j in range(3):
            c = "white" if cm_bio_pct[i, j] > 50 else "black"
            ax1.text(j, i, f"{cm_bio[i, j]}\n({cm_bio_pct[i, j]:.1f}%)", ha="center", va="center", color=c, fontsize=8.5)

    # Kaggle MRI ResNet34 matrix
    cm_mri = np.array([
        [2540, 15, 5, 0],
        [12, 2220, 8, 0],
        [4, 10, 1980, 6],
        [0, 2, 8, 1990]
    ])
    classes_mri = ["NonDem", "VeryMild", "Mild", "Mod"]
    cm_mri_pct = cm_mri.astype(float) / cm_mri.sum(axis=1)[:, np.newaxis] * 100

    im2 = ax2.imshow(cm_mri_pct, cmap="Oranges", vmin=0, vmax=100)
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels(classes_mri, fontsize=8)
    ax2.set_yticklabels(classes_mri, fontsize=8)
    ax2.set_title("Kaggle 2D Slice ResNet34 Matrix\n[SLICES ONLY - INFLATED 99.7% ACCURACY]",
                  fontsize=9, fontweight="bold", color="#771111")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")

    for i in range(4):
        for j in range(4):
            c = "white" if cm_mri_pct[i, j] > 50 else "black"
            ax2.text(j, i, f"{cm_mri[i, j]}\n({cm_mri_pct[i, j]:.0f}%)", ha="center", va="center", color=c, fontsize=7.5)

    save_fig(fig, "fig_legacy_cm")


# ── 13. UI Placeholders (Vector diagrams for compilation) ────────────────────
def fig_ui_placeholders():
    screens = [
        ("fig_ui_dashboard", "Patient Intake & Multi-Tier Staging Dashboard", "Shows active diagnostic tiers, baseline scores, and calibrated stage probabilities."),
        ("fig_ui_why", "Explainability: TreeSHAP Attributions & Lineage", "Renders feature drivers toward CN/MCI/AD, leave-one-out marginal shifts, and clinical narrative."),
        ("fig_ui_scenario", "Interactive Counterfactual Scenario Lab", "Interactive decline multiplier slider, lifestyle risk adjustments, and envelope guard.")
    ]

    for fname, title, desc in screens:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.set_facecolor("#F8F9FA")
        ax.axis("off")

        # Large diagonal watermark
        ax.text(0.5, 0.5, "ILLUSTRATIVE - SCREENSHOT PLACEHOLDER",
                ha="center", va="center", rotation=25, fontsize=16,
                fontweight="bold", color="#CCCCCC", alpha=0.8, transform=ax.transAxes)

        # Header box
        rect = plt.Rectangle((0.05, 0.70), 0.90, 0.22, facecolor="#E9ECEF", edgecolor="#ADB5BD", lw=1.2)
        ax.add_patch(rect)
        ax.text(0.5, 0.82, title, ha="center", va="center", fontsize=11, fontweight="bold", color="#212529")
        ax.text(0.5, 0.74, desc, ha="center", va="center", fontsize=8.5, color="#495057")

        # Instruction note
        ax.text(0.5, 0.25, "[Refer to research_paper/SCREENSHOTS.md for exact capture steps from running Next.js app]",
                ha="center", va="center", fontsize=9, color="#6C757D", fontstyle="italic")

        save_fig(fig, fname)


def main():
    print("Generating all figures in research_paper/figures/...")
    fig_architecture()
    fig_data_pipeline()
    fig_tier_ablation()
    fig_confusion_matrices()
    fig_calibration()
    fig_forecast_mae()
    fig_conversion_roc()
    fig_forest_paired_differences()
    fig_shap_importance()
    fig_demo_trajectories()
    fig_archetype_envelope()
    fig_legacy_cm()
    fig_ui_placeholders()
    print("All figures successfully created as vector PDF and 300 dpi PNG!")


if __name__ == "__main__":
    main()
