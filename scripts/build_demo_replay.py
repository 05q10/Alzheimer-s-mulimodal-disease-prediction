"""
scripts/build_demo_replay.py
============================
Builds verified, de-identified Digital Twin demo replay datasets
from REAL held-out ADNI test subjects.
"""

import os
import sys
import json
import shutil
import random
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

# Ensure server is on sys.path
repo_root = Path(__file__).resolve().parent.parent
server_dir = repo_root / "server"
sys.path.insert(0, str(server_dir))

DATA_DIR = server_dir / "data"
SPLITS_PATH = DATA_DIR / "splits.json"
UNIFIED_PATH = DATA_DIR / "unified_visits.parquet"
REPLAY_DIR = DATA_DIR / "demo_replay"
IMAGES_DIR = REPLAY_DIR / "images"

# Environment variable for MRI root with fallback
DEMO_MRI_ROOT = Path(os.environ.get("DEMO_MRI_ROOT", str(DATA_DIR / "archive (1)" / "combined_images")))

CASES_CONFIG = [
    {
        "case_id": "case_a_stable_cn",
        "alias": "Eleanor Vance",
        "adni_subject_id": "022_S_4291",
        "trajectory_type": "Stable Cognitively Normal (CN)",
        "baseline_stage": "CN",
        "brief": (
            "Patient is a 76-year-old female with 16 years of completed education and zero APOE ε4 risk alleles. "
            "Evaluated at baseline in the Cognitively Normal (CN) stage, she remains cognitively intact across all four follow-up visits over 24 months. "
            "This demo case illustrates the digital twin's ability to maintain high specificity, correctly projecting long-term stability without false-positive progression alerts."
        ),
    },
    {
        "case_id": "case_b_cn_to_mci",
        "alias": "Arthur Pendelton",
        "adni_subject_id": "031_S_4218",
        "trajectory_type": "CN Converting to MCI",
        "baseline_stage": "CN",
        "brief": (
            "Patient is an 81-year-old male with 18 years of education and one APOE ε4 risk allele. "
            "Initial bedside screening placed him in the normal category, but subtle neuropsychological deficits emerged by month 12, culminating in syndromic conversion to MCI by month 24. "
            "This demo highlights early conversion detection and demonstrates how progression hazard and cognitive interval bands widen prior to clinical stage reclassification."
        ),
    },
    {
        "case_id": "case_c_mci_to_ad",
        "alias": "Margaret Sterling",
        "adni_subject_id": "035_S_0997",
        "trajectory_type": "MCI Converting to AD",
        "baseline_stage": "MCI",
        "brief": (
            "Patient is an 80-year-old female with 14 years of education and two APOE ε4 risk alleles (high genetic risk). "
            "Entering the study with Mild Cognitive Impairment, she exhibits progressive functional and memory decline over two years, transitioning into Alzheimer's dementia by month 24. "
            "This demo illustrates dynamic risk trajectory tracking and multi-horizon cognitive score decay forecasting in high-risk progressive phenotypes."
        ),
    },
    {
        "case_id": "case_d_stable_mci",
        "alias": "Harold Finch",
        "adni_subject_id": "022_S_4444",
        "trajectory_type": "Stable Amnestic MCI",
        "baseline_stage": "MCI",
        "brief": (
            "Patient is a 78-year-old male with 16 years of education and one APOE ε4 allele presenting with established mild memory impairment. "
            "Across 24 months of clinical observation, his cognitive battery scores and CDR Global rating remain remarkably indolent, exhibiting no transition to dementia. "
            "This demo demonstrates how the model correctly avoids over-predicting rapid conversion in patients with stable, non-progressing cognitive impairment."
        ),
    },
    {
        "case_id": "case_hard_cognitive_volatility",
        "alias": "Clara Hughes",
        "adni_subject_id": "013_S_1120",
        "trajectory_type": "MCI with High Cognitive Volatility (Hard Case)",
        "baseline_stage": "MCI",
        "brief": (
            "Patient is an 80-year-old male with 16 years of education and one APOE ε4 allele presenting with amnestic MCI. "
            "His longitudinal trajectory exhibits sharp episodic swings in MMSE (falling by 6 points within 12 months before stabilizing), presenting a difficult prognostic challenge. "
            "This benchmark hard case (12-month MMSE error in the 98th percentile) demonstrates the necessity of calibrated 90% confidence bands and reliability flags when handling volatile disease trajectories."
        ),
    },
]

DX_MAP = {1: "CN", 2: "MCI", 3: "AD"}
MRI_FOLDER_MAP = {
    1: "NonDemented",
    2: "VeryMildDemented",
    3: "MildDemented",  # or ModerateDemented
}


def build_replay_dataset():
    print("=== BUILDING VERIFIED DEMO REPLAY DATASET ===")

    # 1. Load splits and enforce non-leakage assertions
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    print(f"Loaded splits: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")

    for c in CASES_CONFIG:
        sid = c["adni_subject_id"]
        assert sid in test_set, f"CRITICAL: Subject {sid} is NOT in the test set!"
        assert sid not in train_set, f"CRITICAL LEAKAGE: Subject {sid} found in training set!"
        assert sid not in val_set, f"CRITICAL LEAKAGE: Subject {sid} found in validation set!"
    print("All 5 replay subjects strictly verified as held-out test subjects.")

    # 2. Load unified visits
    df = pd.read_parquet(UNIFIED_PATH)
    print(f"Loaded unified visits: {len(df)} total visits")

    # Prepare directories
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    today = datetime.now(timezone.utc).date()
    # Baseline is exactly 26 months ago so month 24 visit is 2 months ago (strictly past)
    baseline_date = today - timedelta(days=round(26 * 30.44))

    # Deterministic image pool
    rng = random.Random(42)
    mri_files_by_folder = {}
    if DEMO_MRI_ROOT.exists():
        for fld in ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]:
            fld_path = DEMO_MRI_ROOT / fld
            if fld_path.exists():
                imgs = sorted(list(fld_path.glob("*.jpg")) + list(fld_path.glob("*.png")))
                mri_files_by_folder[fld] = imgs
                print(f"Found {len(imgs)} images in {fld}")
    else:
        print(f"Warning: DEMO_MRI_ROOT ({DEMO_MRI_ROOT}) does not exist. Illustrative MRI slices will be placeholders.")

    index_cases = []

    for c in CASES_CONFIG:
        case_id = c["case_id"]
        sid = c["adni_subject_id"]
        alias = c["alias"]
        traj = c["trajectory_type"]
        b_stage = c["baseline_stage"]
        brief = c["brief"]

        sub_df = df[df["subject_id"] == sid].sort_values("visit_month")
        row0 = sub_df.iloc[0]

        # Extract fixed demographics
        raw_sex = row0.get("sex") if pd.notna(row0.get("sex")) else row0.get("PTGENDER", 0)
        sex = int(raw_sex)
        raw_edu = row0.get("education_years") if pd.notna(row0.get("education_years")) else row0.get("PTEDUCAT", 14.0)
        education_years = float(raw_edu)
        raw_apoe = row0.get("APOE4", 0) if pd.notna(row0.get("APOE4")) else 0
        apoe4_count = int(raw_apoe)

        base_age = float(row0.get("entry_age") or row0.get("AGE") or 70.0)
        rounded_base_age = round(base_age)

        # Select the 4 visits: months 0, 6, 12, 24
        target_months = [0, 6, 12, 24]
        visits_data = []

        for v_idx, tm in enumerate(target_months):
            cand = sub_df[sub_df["visit_month"].between(tm - 1.5, tm + 1.5)]
            cand = cand[cand["MMSE"].notna() & cand["CDGLOBAL"].notna() & cand["TOTSCORE"].notna()]
            assert len(cand) > 0, f"Subject {sid} missing valid record for target month {tm}"
            
            # Pick closest
            idx_closest = (cand["visit_month"] - tm).abs().idxmin()
            v_row = cand.loc[idx_closest]

            actual_m = tm
            exam_date = (baseline_date + timedelta(days=round(actual_m * 30.44))).strftime("%Y-%m-%d")
            v_age = round(rounded_base_age + (actual_m / 12.0), 1)
            true_dx_int = int(v_row["DIAGNOSIS"])
            true_dx_str = DX_MAP.get(true_dx_int, "Unknown")

            # Cognitive scores
            mmse = float(v_row["MMSE"])
            cdglobal = float(v_row["CDGLOBAL"])
            totscore = float(v_row["TOTSCORE"])
            cdrsb = float(v_row["CDRSB"]) if pd.notna(v_row.get("CDRSB")) else None
            faq = float(v_row["FAQTOTAL"]) if pd.notna(v_row.get("FAQTOTAL")) else (
                float(v_row["FAQ"]) if pd.notna(v_row.get("FAQ")) else None
            )
            moca = float(v_row["MOCA"]) if pd.notna(v_row.get("MOCA")) else (
                float(v_row["MOCA_adni"]) if pd.notna(v_row.get("MOCA_adni")) else None
            )

            # Fluid Biomarkers (only if actually observed)
            abeta = float(v_row["ABETA"]) if pd.notna(v_row.get("ABETA")) else None
            tau = float(v_row["TAU"]) if pd.notna(v_row.get("TAU")) else None
            ptau = float(v_row["PTAU"]) if pd.notna(v_row.get("PTAU")) else None

            # FreeSurfer Volumetrics (only if actually observed)
            hippo_norm = float(v_row["hippo_combined_norm"]) if pd.notna(v_row.get("hippo_combined_norm")) else None
            vent_norm = float(v_row["lateral_ventricle_combined_norm"]) if pd.notna(v_row.get("lateral_ventricle_combined_norm")) else None
            wb_norm = float(v_row["whole_brain_norm"]) if pd.notna(v_row.get("whole_brain_norm")) else None
            icv = float(v_row["EstimatedTotalIntraCranialVol_combat"]) if pd.notna(v_row.get("EstimatedTotalIntraCranialVol_combat")) else None

            # Deterministic illustrative MRI slice
            mri_meta = None
            fld_name = MRI_FOLDER_MAP.get(true_dx_int, "NonDemented")
            if fld_name in mri_files_by_folder and mri_files_by_folder[fld_name]:
                pool = mri_files_by_folder[fld_name]
                # Seeded pick per case and visit
                seed_val = int(f"{hash(case_id) % 10000}{v_idx}")
                slice_idx = random.Random(seed_val).randint(0, len(pool) - 1)
                src_img = pool[slice_idx]
                dest_filename = f"{case_id}_v{v_idx}{src_img.suffix}"
                dest_path = IMAGES_DIR / dest_filename
                shutil.copy2(src_img, dest_path)

                mri_meta = {
                    "image_file": dest_filename,
                    "image_rel_path": f"demo_replay/images/{dest_filename}",
                    "source": "Kaggle 2D Axial MRI Dataset",
                    "pairing_type": "illustrative_slice_by_true_dx",
                    "matched_diagnosis": true_dx_str,
                    "disclaimer": "Illustrative pairing from Kaggle dataset; not the patient's personal scan."
                }

            visit_obj = {
                "visit_index": v_idx,
                "month": actual_m,
                "exam_date": exam_date,
                "age_at_visit": v_age,
                "true_dx": true_dx_int,
                "true_dx_str": true_dx_str,
                "scores": {
                    "MMSE": mmse,
                    "CDGLOBAL": cdglobal,
                    "CDRSB": cdrsb,
                    "TOTSCORE": totscore,
                    "FAQ": faq,
                    "MOCA": moca,
                },
                "biomarkers": {
                    "ABETA": abeta,
                    "TAU": tau,
                    "PTAU": ptau,
                },
                "mri_volumes": {
                    "hippo_combined_norm": hippo_norm,
                    "lateral_ventricle_combined_norm": vent_norm,
                    "whole_brain_norm": wb_norm,
                    "icv": icv,
                },
                "lifestyle": {
                    "alcohol": None,
                    "diet_quality": None,
                    "sleep_quality": None,
                    "physical_activity": None,
                    "smoking": None,
                    "hypertension": None,
                    "diabetes": None,
                    "depression": None,
                    "head_injury": None,
                    "cardiovascular_disease": None,
                    "note": "not recorded in ADNI; lifestyle factor is neutral (1.0) and cannot distort replay"
                },
                "mri_slice": mri_meta
            }
            visits_data.append(visit_obj)

        case_document = {
            "case_id": case_id,
            "alias": alias,
            "trajectory_type": traj,
            "baseline_stage": b_stage,
            "brief": brief,
            "baseline_date": baseline_date.strftime("%Y-%m-%d"),
            "demographics": {
                "age": rounded_base_age,
                "sex": sex,
                "education_years": education_years,
                "apoe4_count": apoe4_count
            },
            "num_visits": len(visits_data),
            "visits": visits_data,
            "data_provenance": {
                "source": "ADNI Held-Out Test Split",
                "non_leakage_guarantee": "Subject verified strictly disjoint from train and validation splits",
                "de_identification": "Name is invented; age rounded; dates shifted relative to today; subject ID scrubbed"
            }
        }

        # Write case JSON
        out_path = REPLAY_DIR / f"{case_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(case_document, f, indent=2)
        print(f"Wrote {case_id}.json ({len(visits_data)} visits)")

        index_cases.append({
            "case_id": case_id,
            "alias": alias,
            "trajectory_type": traj,
            "baseline_stage": b_stage,
            "brief": brief,
            "demographics": case_document["demographics"],
            "num_visits": len(visits_data),
            "timeline_months": target_months
        })

    # Write cases index (for GET /demo/cases)
    index_path = REPLAY_DIR / "cases_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_cases, f, indent=2)
    print(f"Wrote cases_index.json with {len(index_cases)} cases.")
    print("=== REPLAY BUILD COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    build_replay_dataset()

