# research_paper/Makefile
# =======================

PYTHON = python
TEX = pdflatex
BIB = bibtex

.PHONY: all draft final figures numbers zip clean check_placeholders

all: draft

numbers:
	$(PYTHON) scripts/make_numbers.py

figures:
	$(PYTHON) scripts/make_figures.py

check_placeholders:
	@echo "Checking for unresolved illustrative placeholders..."
	@$(PYTHON) -c "import os, sys; \
placeholders = ['fig_ui_dashboard.png', 'fig_ui_why.png', 'fig_ui_scenario.png']; \
found = [p for p in placeholders if os.path.exists(os.path.join('figures', p)) and 'ILLUSTRATIVE' in open(os.path.join('figures', p.replace('.png', '.pdf')), 'rb').read().decode('latin1', errors='ignore')]; \
if found: \
    print('ERROR: Final build failed! Unresolved placeholders detected: ' + ', '.join(found)); \
    print('Refer to research_paper/SCREENSHOTS.md for live capture instructions.'); \
    sys.exit(1); \
else: \
    print('All placeholders resolved. Proceeding to final compilation.')"

draft: numbers figures
	@echo "Building paper in DRAFT mode (placeholders permitted)..."
	@if command -v $(TEX) >/dev/null 2>&1; then \
		$(TEX) -interaction=nonstopmode main.tex; \
		$(BIB) main; \
		$(TEX) -interaction=nonstopmode main.tex; \
		$(TEX) -interaction=nonstopmode main.tex; \
		echo "Draft PDF successfully compiled: main.pdf"; \
	else \
		echo "NOTE: pdflatex not found in local PATH. Preparing Overleaf bundle..."; \
	fi
	@make zip

final: check_placeholders numbers figures
	@echo "Building paper in FINAL mode..."
	@if command -v $(TEX) >/dev/null 2>&1; then \
		$(TEX) -interaction=nonstopmode main.tex; \
		$(BIB) main; \
		$(TEX) -interaction=nonstopmode main.tex; \
		$(TEX) -interaction=nonstopmode main.tex; \
		echo "Final PDF successfully compiled: main.pdf"; \
	else \
		echo "NOTE: pdflatex not found in local PATH. Preparing Overleaf bundle..."; \
	fi
	@make zip

zip:
	@echo "Creating research_paper.zip for Overleaf upload..."
	@$(PYTHON) -c "import zipfile, os; \
files_to_zip = ['main.tex', 'preamble.tex', 'numbers.tex', 'references.bib']; \
with zipfile.ZipFile('research_paper.zip', 'w', zipfile.ZIP_DEFLATED) as z: \
    for f in files_to_zip: \
        if os.path.exists(f): z.write(f, f); \
    for root, dirs, files in os.walk('figures'): \
        for f in files: \
            p = os.path.join(root, f); z.write(p, p); \
    for root, dirs, files in os.walk('tables'): \
        for f in files: \
            p = os.path.join(root, f); z.write(p, p); \
print('Successfully created research_paper.zip')"

clean:
	rm -f *.aux *.log *.bbl *.blg *.out *.toc *.synctex.gz
