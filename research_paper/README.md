# Alzheimer's Multimodal Digital Twin — Research Paper Package

**Manuscript Title**: *A Multimodal Longitudinal Digital Twin Framework for Alzheimer's Disease Staging, Trajectory Forecasting, and Decision Support*  
**Authors**: Mihit Singasane, Ved Thakker, Ria Talsania, Swapnali Kurhade  
**Affiliation**: Department of Computer Engineering, Sardar Patel Institute of Technology, Mumbai, India  
**Target Package Directory**: `research_paper/`  
**System Version**: `2.0.0-production`  

---

## 1. Package Structure

```
research_paper/
  ├── main.tex                    # Primary LaTeX manuscript (condensed journal article format)
  ├── preamble.tex                # Package imports, geometry, styling, and \figph macro
  ├── numbers.tex                 # Auto-generated LaTeX macros (\newcommand) from metric files
  ├── references.bib              # Verified BibTeX bibliography with DOIs
  ├── figures/                    # Publication figures (both vector PDF and 300 dpi PNG)
  │     ├── README.txt            # Manifest listing exact filenames, sources, and scripts
  │     ├── fig_architecture.*    # End-to-end system architecture diagram
  │     ├── fig_data_pipeline.*   # Data merging and leakage control diagram
  │     ├── fig_tier_ablation.*   # Multi-tier staging performance with 95% CIs
  │     ├── fig_stager_cm_*.*     # Test confusion matrices for T1b, T2b, and T4
  │     ├── fig_calibration.*     # Staging and conversion reliability diagrams
  │     ├── fig_forecast_mae.*    # Multi-horizon MAE vs persistence and slope baselines
  │     ├── fig_conversion_roc.*  # Dementia conversion ROC curves (AUROC 0.811 at 24m)
  │     ├── fig_forest_*.*        # Forest plot of paired differences with zero line
  │     ├── fig_shap_importance.* # TreeSHAP feature attributions by modality
  │     ├── fig_demo_*.*          # Demo trajectories and Case 3 scenario simulation
  │     ├── fig_archetype_*.*     # Archetype trajectories vs 5th–95th percentile envelope
  │     ├── fig_legacy_cm.*       # Earlier-phase legacy confusion matrices (Appendix)
  │     └── fig_ui_*.*            # Watermarked vector placeholders for live UI capture
  ├── tables/                     # Auto-generated LaTeX tables
  │     ├── table1_cohort_splits.tex
  │     ├── table2_data_sources.tex
  │     ├── table3_model_inventory.tex
  │     ├── table4_stager_results.tex
  │     ├── table5_forecast_mae.tex
  │     ├── table6_conversion_metrics.tex
  │     ├── table7_paired_ablations.tex
  │     ├── table8_archetype_plausibility.tex
  │     ├── table9_lineage_summary.tex
  │     ├── table10_lifestyle_factors.tex
  │     └── table11_limitations.tex
  ├── scripts/                    # Generation scripts (zero hand-typed numbers)
  │     ├── make_numbers.py       # Extracts metrics and writes numbers.tex + tables/
  │     └── make_figures.py       # Computes and renders all vector PDFs and 300 dpi PNGs
  ├── Makefile                    # Build automation supporting draft, final, and zip
  ├── ISSUES.md                   # Audit of numerical disagreements and cohort differences
  ├── PLACEHOLDERS.md             # Detailed documentation of illustrative placeholders
  ├── MISSING_FIGURES.md          # Manifest of figures requiring live browser capture
  ├── SCREENSHOTS.md              # Step-by-step instructions for capturing UI screenshots
  ├── TRIPOD_AI_checklist.md      # Full TRIPOD+AI 2024 checklist mapping
  ├── REFERENCE_AUDIT.md          # Complete bibliography audit and DOI cross-reference
  ├── template/                   # Preserved original student template and draft files
  └── research_paper.zip          # Self-contained bundle ready for Overleaf upload
```

---

## 2. Compilation Instructions

### A. Overleaf Compilation (Recommended)
1. Download or locate `research_paper/research_paper.zip`.
2. Open [Overleaf](https://www.overleaf.com), click **New Project** $\rightarrow$ **Upload Project**, and select `research_paper.zip`.
3. Set the compiler to **pdfLaTeX** (default in Overleaf).
4. Click **Recompile**. The paper compiles cleanly into a camera-ready PDF.

### B. Local Compilation (Linux / macOS / Windows with TeX Live)
To regenerate metrics, figures, and compile the manuscript:
```bash
# 1. Regenerate numbers and LaTeX tables from repository result files
python scripts/make_numbers.py

# 2. Regenerate all vector PDFs and 300 dpi PNG figures
python scripts/make_figures.py

# 3. Compile in DRAFT mode (allows illustrative UI placeholders)
make draft
# Or manually:
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

# 4. Compile in FINAL mode (verifies that no placeholders remain)
make final
```

*Note on Local TeX Environment*: If no local LaTeX compiler is installed on your workstation, you do not need to install a multi-gigabyte distribution. Simply upload `research_paper.zip` directly to Overleaf.

---

## 3. How to Port to Journal Templates (Springer Nature / BMC)

To port `main.tex` to a specific journal publisher class (e.g., Springer Nature `sn-jnl.cls` or BMC Bioinformatics):

1. **Document Class Replacement**:
   Replace:
   ```latex
   \documentclass[11pt,a4paper]{article}
   \input{preamble.tex}
   ```
   with the publisher's template preamble:
   ```latex
   \documentclass[pdflatex,sn-basic]{sn-jnl}
   \input{numbers.tex}
   ```
2. **Author Affiliations**:
   Convert the standard `\author` block to the journal's structured author syntax:
   ```latex
   \author[1]{\fnm{Mihit} \sur{Singasane}}\email{mihit.singasane@spit.ac.in}
   \author[1]{\fnm{Ved} \sur{Thakker}}\email{ved.thakker@spit.ac.in}
   \author[1]{\fnm{Ria} \sur{Talsania}}\email{ria.talsania@spit.ac.in}
   \author*[1]{\fnm{Swapnali} \sur{Kurhade}}\email{swapnali.kurhade@spit.ac.in}
   \affil[1]{\orgdiv{Department of Computer Engineering}, \orgname{Sardar Patel Institute of Technology}, \orgaddress{\city{Mumbai}, \country{India}}}
   ```
3. **Tables and Figures**:
   All tables (`tables/*.tex`) use standard `booktabs` syntax (`\toprule`, `\midrule`, `\bottomrule`), which is natively supported across Springer Nature, Elsevier, IEEE, and BMC templates without modification.
4. **Declarations Section**:
   The `Declarations` section in `main.tex` already adheres to the standard BioMed Central (BMC) and Springer Nature subheadings:
   - *Ethics approval and consent to participate*
   - *Consent for publication*
   - *Availability of data and materials*
   - *Competing interests*
   - *Funding*
   - *Authors' contributions*
   - *Acknowledgements*

---

## 4. Author Action Items Prior to Submission

Before submitting the paper to a journal or conference, the authors must complete the following items:

1. **Capture Real UI Screenshots**: Follow `SCREENSHOTS.md` to capture `fig_ui_dashboard.png`, `fig_ui_why.png`, and `fig_ui_scenario.png` from the running Next.js application, convert them to PDF, and replace the placeholders.
2. **Official ADNI Acknowledgment Text**: Copy and paste the current mandatory acknowledgment paragraph and author-byline formatting verbatim from the official ADNI website ([adni.loni.usc.edu](https://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Data_Use_Agreement.pdf)) into the Declarations section.
3. **Ethics Approval Protocol Number**: If required by the target journal, insert the specific Institutional Review Board (IRB) approval protocol reference numbers for your institution.
4. **Funding & Competing Interests**: If external grants funded this project, replace the placeholder in Declarations with the formal grant number and agency name.
5. **Double-Blind Anonymization**: If submitting to a double-blind peer-review venue (e.g., ACM CHIL or MICCAI), comment out author names, institutional affiliations, and the GitHub repository link.
