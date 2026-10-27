"""
server/ml/twin/inventory.py
============================
Step 0 of the data pipeline: inventory every CSV in server/data/ and
server/ml/alele_model/, profile each one, and write the report to
server/experiments/results/00_data_inventory.md.

Run from server/:
    python -m ml.twin.inventory
"""

import os, hashlib, glob, re
from pathlib import Path
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
SERVER_DIR = SCRIPT_DIR.parent.parent            # server/
DATA_DIR   = SERVER_DIR / "data"
ALELE_DIR  = SERVER_DIR / "ml" / "alele_model"
OUT_DIR    = SERVER_DIR / "experiments" / "results"
OUT_FILE   = OUT_DIR / "00_data_inventory.md"

# Pre-existing knowledge from survey (Use/Skip/Partial decisions)
DECISIONS = {
    "shortened_result-val.csv":                      ("USE",     "Full longitudinal truth; 3,034 subjects; all DIAGNOSIS non-missing"),
    "shortened_result_original_missing_4visits.csv": ("USE",     "Train base; ≤4 visits; 40-42% missing scores; no leakage from future"),
    "shortened_result_imputed_4visits.csv":          ("USE*",    "Comparison baseline ONLY — imputation may use future visits (label 'may leak future info')"),
    "ADNIMERGE_13Aug2025.csv":                       ("USE",     "77.6% subject overlap; key covariates + FreeSurfer volumes"),
    "ADSP_PHC_T1_FS_15Aug2025.csv":                  ("USE",     "76.3% overlap; combat-harmonised FreeSurfer MRI; canonical MRI source"),
    "All_Subjects_APOERES_fixed_simple.csv":         ("USE",     "Primary APOE genotype source; more complete than ADNIMERGE APOE4"),
    "All_Subjects_APOERES_30Dec2025.csv":            ("PARTIAL", "Gap-fill for fixed_simple; compare genotypes and log conflicts"),
    "All_Subjects_My_Table_30Dec2025.csv":           ("USE",     "100% subject overlap; adds CDRSB, MOCA, FAQTOTAL for all subjects"),
    "All_Subjects_ADAS_30Dec2025.csv":               ("PARTIAL", "Confirms TOTSCORE=ADAS11_cog_prorated; use for audit only"),
    "All_Subjects_DXSUM_30Dec2025.csv":              ("PARTIAL", "Secondary DX source; used only to log disagreement, never to overwrite"),
    "CDR_15Aug2025.csv":                             ("PARTIAL", "Secondary CDR source for gap-filling"),
    "MMSE_15Aug2025.csv":                            ("PARTIAL", "Secondary MMSE source for gap-filling"),
    "All_Subjects_PTDEMOG_30Dec2025.csv":            ("USE",     "Demographics: sex and education for subjects missing from ADNIMERGE"),
    "All_Subjects_BLCHANGE_30Dec2025.csv":           ("SKIP",    "Clinician baseline-change flags; redundant with DX"),
    "cleaned_blchange.csv":                          ("SKIP",    "Same data as BLCHANGE de-duped; redundant"),
    "All_Subjects_ADSXLIST_30Dec2025.csv":           ("SKIP",    "Adverse event/symptom list; not AD modelling features"),
    "LABTESTS_15Aug2025.csv":                        ("SKIP",    "Generic lab flags; not AD-specific biomarkers"),
    "All_Subjects_VITALS_30Dec2025.csv":             ("SKIP",    "Height/weight/BP; not core AD features for this pipeline"),
    "All_Subjects_FAMHXPAR_30Dec2025.csv":           ("PARTIAL", "MOTHDEM+FATHDEM → binary family_hx_ad flag; sparse"),
    "All_Subjects_Study_Entry_30Dec2025.csv":        ("USE",     "entry_age baseline; join key for date reconstruction"),
    "alzheimers_disease_data.csv":                   ("SKIP",    "No ADNI-linkable IDs; Kaggle lifestyle cohort only (separate base_model)"),
}

# Column family classification keywords
FAMILY_RULES = {
    "ids":        ["RID","PTID","SUBJID","subject_id","ID","SITEID","PHC_LONIUID","PHC_Visit"],
    "dates":      ["DATE","date","EXAMDATE","VISDATE","APTESTDT","USERDATE","update_stamp","SCANDATE"],
    "demo":       ["AGE","age","GENDER","SEX","EDUC","RACE","ETHN","MARR","BIRTH","DOB","PTGENDER","PTDOB","PTEDUCAT","PTETHCAT","PTRACCAT","PHC_Sex","PHC_Edu","PHC_Age","PHC_Race","PHC_Eth"],
    "cognitive":  ["MMSE","MMSCORE","CDGLOBAL","CDRSB","ADAS","TOTSCORE","TOTAL13","FAQ","MOCA","RAVLT","ADL","GDTOTAL","NPISCORE","KEYMED","LDELTA","DIGITSCOR","TRABSCOR","FAQTOTAL","mPACC","EcogPt","EcogSP","BCMMSE","BCADAS"],
    "csf":        ["ABETA","TAU","PTAU","FDG","PIB","AV45","FBB"],
    "mri_vol":    ["Hippocampus","Ventricles","WholeBrain","Entorhinal","Fusiform","MidTemp","ICV","BrainSeg","CortexVol","CSF_","Amygdala","Accumbens","Caudate","Putamen","Pallidum","Thalamus","combat","volume","Volume"],
    "cortical_th":["thickness","_thickness"],
    "surface":    ["_area","_surface","SurfaceArea"],
    "qc":         ["QC","qc","HAS_QC","APUSABLE","APRECEIVE","APRESAMP","APAMBTEMP"],
    "genotype":   ["GENOTYPE","APOE","APVOLUME","apoe","genotype"],
    "diagnosis":  ["DIAGNOSIS","DX","DXNORM","DXMCI","DXAD","DXPARK","DXDEP","DXCONF","DXDSEV","diagnosis"],
    "study_meta": ["PHASE","COLPROT","ORIGPROT","VISCODE","VISCODE2","IMAGEUID","FLDSTRENG","FSVERSION","PHC_Scanner","PHC_Field"],
}

def classify_column(col: str) -> str:
    col_lower = col.lower()
    for family, keywords in FAMILY_RULES.items():
        for kw in keywords:
            if kw.lower() in col_lower:
                return family
    return "other"

def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def profile_csv(path: Path) -> dict:
    """Profile a CSV file without loading the whole thing into memory."""
    fname = path.name
    size_mb = path.stat().st_size / 1e6
    md5 = file_md5(str(path))

    # Load with low_memory, profile first chunk for column types
    try:
        df = pd.read_csv(path, low_memory=False, nrows=100000)
    except Exception as e:
        return {"file": fname, "error": str(e)}

    # Full row count
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            nrows = sum(1 for _ in f) - 1
    except Exception:
        nrows = len(df)

    ncols = len(df.columns)

    # Column families
    families: dict[str, list[str]] = {}
    for col in df.columns:
        fam = classify_column(col)
        families.setdefault(fam, []).append(col)

    # Missingness (top 10 offenders among nullable)
    miss = df.isnull().mean().sort_values(ascending=False)
    miss_top = miss[miss > 0].head(10).round(3).to_dict()

    # Candidate join keys (unique, low-null columns)
    key_candidates = []
    for col in df.columns:
        if df[col].isnull().mean() < 0.01:
            n_unique = df[col].nunique()
            if 1 < n_unique <= nrows:
                key_candidates.append(f"{col}(unique={n_unique})")

    # Date range
    date_cols = families.get("dates", [])
    date_range = {}
    for dc in date_cols[:3]:
        try:
            parsed = pd.to_datetime(df[dc], errors="coerce")
            if parsed.notna().sum() > 10:
                date_range[dc] = f"{parsed.min().date()} – {parsed.max().date()}"
        except Exception:
            pass

    # Duplicate key check
    dup_info = []
    for combo in [["subject_id","visit"], ["PTID","VISCODE"], ["RID","VISCODE"], ["subject_id"]]:
        if all(c in df.columns for c in combo):
            ndups = df.duplicated(subset=combo).sum()
            dup_info.append(f"{'+'.join(combo)}: {ndups} dups")
            break

    decision, reason = DECISIONS.get(fname, ("?", "Not pre-classified"))

    return {
        "file": fname,
        "size_mb": round(size_mb, 2),
        "md5": md5,
        "rows": nrows,
        "cols": ncols,
        "families": families,
        "miss_top": miss_top,
        "key_candidates": key_candidates[:6],
        "date_range": date_range,
        "dup_info": dup_info,
        "decision": decision,
        "reason": reason,
    }

def compute_overlap(path: Path, val_subjects: set) -> str:
    """Count how many val subjects appear in this file (best-effort)."""
    try:
        for id_col in ["subject_id", "PTID", "RID"]:
            try:
                df = pd.read_csv(path, usecols=[id_col], low_memory=False)
                ids = set(df[id_col].dropna().astype(str).unique())
                n = len(val_subjects & ids)
                return f"{n}/{len(val_subjects)} ({100*n/len(val_subjects):.1f}%)"
            except (ValueError, KeyError):
                continue
    except Exception:
        pass
    return "N/A"

def md_table(rows: list[dict], cols: list[str]) -> str:
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines  = [header, sep]
    for r in rows:
        line = "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |"
        lines.append(line)
    return "\n".join(lines)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all CSVs — data dir and alele_model dir
    csv_paths_raw = list(DATA_DIR.glob("*.csv")) + list(ALELE_DIR.glob("*.csv"))
    csv_paths_raw += list(ALELE_DIR.glob("**/*.csv"))
    # Deduplicate by MD5 (some CSVs are copied between data/ and alele_model/)
    seen_md5s: dict[str, Path] = {}
    dedup_paths: list[Path] = []
    for p in sorted(set(csv_paths_raw)):
        try:
            m = file_md5(str(p))
        except Exception:
            continue
        if m not in seen_md5s:
            seen_md5s[m] = p
            dedup_paths.append(p)
        else:
            print(f"  [DUPLICATE] {p.name} == {seen_md5s[m].name} (same MD5, skipping)")

    # Load val subjects for overlap computation
    val_path = DATA_DIR / "shortened_result-val.csv"
    val_df = pd.read_csv(val_path, usecols=["subject_id"], low_memory=False)
    val_subjects = set(val_df["subject_id"].astype(str).unique())
    print(f"Val subjects: {len(val_subjects)}")

    profiles = []
    for p in dedup_paths:
        print(f"  Profiling: {p.name} ...", end=" ", flush=True)
        prof = profile_csv(p)
        prof["val_overlap"] = compute_overlap(p, val_subjects)
        profiles.append(prof)
        print(f"{prof.get('rows','?')} rows x {prof.get('cols','?')} cols  [{prof['decision']}]")

    # ── Write markdown report ──────────────────────────────────────────────
    lines = ["# Data Inventory Report", "",
             f"Generated by `server/ml/twin/inventory.py`", "",
             f"**Val reference**: `shortened_result-val.csv` — {len(val_subjects)} subjects", "",
             "---", ""]

    # Summary table
    summary_rows = []
    for p in profiles:
        summary_rows.append({
            "File": p["file"],
            "MB": p.get("size_mb", "?"),
            "Rows": p.get("rows", "?"),
            "Cols": p.get("cols", "?"),
            "Val Overlap": p.get("val_overlap", "N/A"),
            "Decision": p.get("decision", "?"),
            "Reason": p.get("reason", ""),
        })
    lines.append("## Summary Table")
    lines.append("")
    lines.append(md_table(summary_rows, ["File","MB","Rows","Cols","Val Overlap","Decision","Reason"]))
    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-file detail
    lines.append("## Per-File Detail")
    lines.append("")
    for p in profiles:
        lines.append(f"### `{p['file']}`")
        lines.append("")
        if "error" in p:
            lines.append(f"> ⚠️ Error profiling: {p['error']}")
            lines.append("")
            continue
        lines.append(f"- **Size**: {p['size_mb']} MB | **Rows**: {p['rows']} | **Cols**: {p['cols']}")
        lines.append(f"- **MD5**: `{p['md5']}`")
        lines.append(f"- **Decision**: **{p['decision']}** — {p['reason']}")
        lines.append(f"- **Val subject overlap**: {p['val_overlap']}")
        if p["dup_info"]:
            lines.append(f"- **Duplicate key check**: {'; '.join(p['dup_info'])}")
        if p["date_range"]:
            lines.append(f"- **Date range**: {p['date_range']}")
        # Column families
        lines.append("- **Column families**:")
        fams = p.get("families", {})
        for fam, cols in fams.items():
            if cols:
                preview = ", ".join(cols[:8])
                more = f" … (+{len(cols)-8})" if len(cols) > 8 else ""
                lines.append(f"  - `{fam}` ({len(cols)}): {preview}{more}")
        # Missingness
        miss = p.get("miss_top", {})
        if miss:
            lines.append("- **Top missing columns** (fraction):")
            for col, frac in list(miss.items())[:8]:
                lines.append(f"  - `{col}`: {frac:.1%}")
        # Key candidates
        keys = p.get("key_candidates", [])
        if keys:
            lines.append(f"- **Key candidates**: {', '.join(keys)}")
        lines.append("")

    # Duplicate files section
    lines.append("---")
    lines.append("")
    lines.append("## Duplicate File Report (same MD5)")
    lines.append("")
    all_md5s: dict[str, list[Path]] = {}
    for p in csv_paths_raw:
        try:
            m = file_md5(str(p))
            all_md5s.setdefault(m, []).append(p)
        except Exception:
            pass
    found_dups = False
    for m, paths in all_md5s.items():
        if len(paths) > 1:
            found_dups = True
            names = [str(p.relative_to(SERVER_DIR.parent)) for p in paths]
            lines.append(f"- MD5 `{m[:8]}…`: {' == '.join(names)}")
    if not found_dups:
        lines.append("No exact duplicate files found.")
    lines.append("")

    # FORBIDDEN_FEATURES reminder
    from ml.twin.constants import FORBIDDEN_FEATURES
    lines.append("---")
    lines.append("")
    lines.append("## Forbidden Features (must never appear in model inputs)")
    lines.append("")
    lines.append("These columns are excluded from ALL models. See `ml/twin/constants.py`.")
    lines.append("")
    for f in FORBIDDEN_FEATURES:
        lines.append(f"- `{f}`")
    lines.append("")

    OUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] Inventory written to: {OUT_FILE}")
    return profiles

if __name__ == "__main__":
    main()
