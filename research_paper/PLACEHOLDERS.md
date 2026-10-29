# Illustrative Placeholders Manifest

**Date**: October 2026  
**Document**: `research_paper/PLACEHOLDERS.md`  
**Rule**: Any figure that cannot be computed directly from saved offline model weights or observational splits is clearly marked as an illustrative placeholder with a prominent diagonal watermark. In `make final` mode, the build script verifies that no unreplaced placeholder remains.

---

## Catalog of Illustrative Placeholders

### 1. `fig_ui_dashboard` (PDF / PNG)
- **Title**: Patient Intake & Multi-Tier Staging Dashboard
- **Location in Paper**: Referenced via `\figph{fig_ui_dashboard}{...}{fig:ui_dash}` in Section 4.5.
- **Why It Is a Placeholder**: The dashboard is a live interactive Next.js 14 web application interface. Static computational scripts cannot capture the dynamic web browser UI rendering, user interactions, or styling directly.
- **Current Content**: A clean vector diagram with a diagonal watermark: `ILLUSTRATIVE - SCREENSHOT PLACEHOLDER`, specifying the dashboard layout, tier chips, and calibrated stage probabilities.
- **How to Replace with Real Capture**: Follow the exact step-by-step instructions in [`research_paper/SCREENSHOTS.md`](file:///d:/family/MiniProject/Allele%20Prediction/research_paper/SCREENSHOTS.md) to launch the frontend (`npm run dev`) and capture the browser window at `http://localhost:3000/twin/[id]`.

---

### 2. `fig_ui_why` (PDF / PNG)
- **Title**: Explainability Panel: TreeSHAP Feature Drivers & Modality Lineage
- **Location in Paper**: Referenced via `\figph{fig_ui_why}{...}{fig:ui_why}` in Section 4.5.
- **Why It Is a Placeholder**: Renders the clinician-facing explainability drawer (`WhyThisStagePanel.tsx`) in the live Next.js web application.
- **Current Content**: Vector diagram watermarked `ILLUSTRATIVE - SCREENSHOT PLACEHOLDER` detailing TreeSHAP bar contributions and leave-one-out probability deltas.
- **How to Replace with Real Capture**: Launch the web application, open a patient twin, expand the "Why this Stage?" panel, and capture the component as detailed in `SCREENSHOTS.md`.

---

### 3. `fig_ui_scenario` (PDF / PNG)
- **Title**: Interactive Counterfactual Scenario Lab Interface
- **Location in Paper**: Referenced via `\figph{fig_ui_scenario}{...}{fig:ui_scen}` in Section 4.5.
- **Why It Is a Placeholder**: Renders the interactive what-if decline rate slider, lifestyle adjustment toggles, and plausibility envelope violation alerts in the live frontend (`Step3ScenarioLab.tsx`).
- **Current Content**: Vector diagram watermarked `ILLUSTRATIVE - SCREENSHOT PLACEHOLDER` showing decline multiplier controls and counterfactual comparison curves.
- **How to Replace with Real Capture**: Open the Scenario Lab tab in the frontend, adjust the decline rate slider, and capture the interface as detailed in `SCREENSHOTS.md`.

---

## Verification Summary
- **Total Scientific Results Figures**: 14 figures (all generated from real observational test splits and saved model weights).
- **Total UI Screenshot Placeholders**: 3 figures (generated as watermarked vector placeholders; replaceable via `SCREENSHOTS.md`).
- **Synthetic or Invented Results**: **Zero**.

