# Clinical Web Application Screenshot Capture Guide

**Date**: October 2026  
**Document**: `research_paper/SCREENSHOTS.md`  
**Target Figures**: `fig_ui_dashboard`, `fig_ui_why`, `fig_ui_scenario`

This guide details the exact procedure for capturing high-resolution clinical UI screenshots from the running Next.js application to replace the illustrative vector placeholders.

---

## 1. Launching the Local Services

### Step 1: Start the Backend FastAPI Server
Open a terminal and execute:
```bash
cd server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Verify backend readiness by navigating to `http://localhost:8000/docs` in your browser.

### Step 2: Start the Next.js Frontend Server
Open a second terminal and execute:
```bash
cd client
npm run dev
```
The application will launch on `http://localhost:3000`.

---

## 2. Screenshot Capture Protocol

For crisp academic figures:
- Use a desktop browser (Chrome, Edge, or Firefox) maximized on a 1080p or 4K display.
- Set browser zoom to 100%.
- Capture using your system's lossless screenshot tool (e.g. Snipping Tool on Windows, `Win + Shift + S`).

---

### Capture 1: `fig_ui_dashboard.png` (Patient Intake & Staging Dashboard)
1. Open `http://localhost:3000/demos` and select **Margaret Sterling (MCI to AD Converter)** or create a twin via the intake wizard.
2. Navigate to the main patient twin view at `http://localhost:3000/twin/[twin_id]`.
3. Ensure the top section is visible, showing:
   - Patient persona banner (Demographics, APOE status, Entry Age).
   - Active diagnostic modality chips (Demographics, Neuropsych, CSF, MRI).
   - The Stage Card displaying primary stage (`MCI` or `AD`), confidence gauge, and calibrated probabilities ($P(\text{CN})$, $P(\text{MCI})$, $P(\text{AD})$).
4. Capture a rectangular window of this area.
5. Save the file directly as:  
   `research_paper/figures/fig_ui_dashboard.png`

---

### Capture 2: `fig_ui_why.png` (Explainability & Lineage Panel)
1. In the same patient twin view, scroll down or click on the **"Why This Stage?"** explainability card.
2. Ensure the panel expands to show:
   - The top TreeSHAP feature attribution bars (e.g., CDR, MMSE, ADAS11).
   - The **Leave-One-Modality-Out** sensitivity table showing marginal probability deltas without advanced scores or biomarkers.
   - The deterministic 4-to-8 sentence clinical narrative.
3. Capture a clear, legible screenshot of this drawer.
4. Save the file directly as:  
   `research_paper/figures/fig_ui_why.png`

---

### Capture 3: `fig_ui_scenario.png` (Counterfactual Scenario Lab)
1. In the patient twin view, switch to the **"Scenario Lab"** tab (`Step3ScenarioLab.tsx`).
2. Adjust the **Decline Rate Multiplier** slider (e.g., set to $0.85\times$ or $1.20\times$) or toggle lifestyle risk interventions (e.g., improved exercise or smoking cessation).
3. Ensure the view displays:
   - The slider controls and hypothetical follow-up inputs.
   - The dual trajectory projection comparing the standard baseline forecast against the counterfactual scenario.
   - The plausibility envelope guard status badge (showing `ok` or the yellow/red scope warning if inputs are extreme).
4. Capture the scenario comparison interface.
5. Save the file directly as:  
   `research_paper/figures/fig_ui_scenario.png`

---

## 3. Post-Capture Vector PDF Conversion

After saving the three PNG files into `research_paper/figures/`, run the following one-line command to generate the matching vector/lossless PDF versions:

```bash
python -c "from PIL import Image; [Image.open(f'research_paper/figures/{name}.png').convert('RGB').save(f'research_paper/figures/{name}.pdf') for name in ['fig_ui_dashboard', 'fig_ui_why', 'fig_ui_scenario']]; print('Successfully converted screenshots to PDF.')"
```

Once converted, your paper can be compiled in final mode (`make final`).
