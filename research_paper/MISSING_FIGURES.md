# Missing Figures & Live UI Capture Manifest

**Date**: October 2026  
**Document**: `research_paper/MISSING_FIGURES.md`

All quantitative machine learning results figures have been generated directly from the repository's saved model weights and the held-out test split. The only figures requiring manual capture are user interface (UI) screenshots from the live web application:

| Figure Key | Expected File Names | Description | Status | Replacement Procedure |
|---|---|---|---|---|
| `fig_ui_dashboard` | `fig_ui_dashboard.pdf`, `fig_ui_dashboard.png` | Patient intake and multi-tier staging dashboard | Vector placeholder generated | Capture from `http://localhost:3000/twin/[id]` (refer to `SCREENSHOTS.md`) |
| `fig_ui_why` | `fig_ui_why.pdf`, `fig_ui_why.png` | Explainability panel: TreeSHAP drivers and modality lineage | Vector placeholder generated | Capture from `WhyThisStagePanel` in running web app (refer to `SCREENSHOTS.md`) |
| `fig_ui_scenario` | `fig_ui_scenario.pdf`, `fig_ui_scenario.png` | Counterfactual Scenario Lab simulator | Vector placeholder generated | Capture from Scenario Lab tab in running web app (refer to `SCREENSHOTS.md`) |

---

## Instructions for Authors Prior to Final Publication
1. Run the local development servers (`server` on port 8000, `client` on port 3000).
2. Follow the capture guidelines in `research_paper/SCREENSHOTS.md`.
3. Save the captured PNG images directly over `research_paper/figures/fig_ui_dashboard.png`, `fig_ui_why.png`, and `fig_ui_scenario.png`.
4. Convert each PNG to vector/lossless PDF using:
   ```bash
   python -c "from PIL import Image; [Image.open(f'research_paper/figures/{name}.png').convert('RGB').save(f'research_paper/figures/{name}.pdf') for name in ['fig_ui_dashboard', 'fig_ui_why', 'fig_ui_scenario']]"
   ```
5. Rebuild the paper in final mode (`make final`).

