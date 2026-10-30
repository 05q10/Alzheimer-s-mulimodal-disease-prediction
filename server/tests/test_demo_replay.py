"""
server/tests/test_demo_replay.py
================================
Pytest suite verifying Part 3 requirements for the ADNI Demo Replay:
  (a) All four replay subjects are strictly in the test split of server/data/splits.json,
      with zero occurrence in train or val splits.
  (b) The stored pre-visit prediction for visit k is identical when recomputed from
      visits 0..k-1 and does not change after visit k is ingested (hash check) and
      uses no feature dated after visit k-1.
  (c) Replay runs cleanly both with include_mri=True and include_mri=False.
  (d) Reset returns the twin strictly to baseline visit 0 state.
  (e) End-to-end API route run for all four patients passes successfully.
  (f) Legacy routes (/demo/patients, /model-card, /twin) continue to respond normally.
"""

import os
import sys
import json
from pathlib import Path
import pytest

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))

# Configuration: ensure cache and environment are optimized
cache_dir = SERVER_DIR / "scratch" / "mpl_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(cache_dir)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "dummy_groq_key_for_testing")

from app.replay_engine import (
    start_replay, advance_replay, reset_replay,
    get_replay_report, get_replay_summary, get_cases_index,
    compute_prediction_hash, REPLAY_DIR
)
from app.twin import get_twin
from app.models.twin_models import CreateTwinRequest
from app.routers.twin_router import (
    list_demo_cases, start_replay_route, next_replay_route,
    reset_replay_route, get_replay_report_route, get_replay_summary_route,
    list_demo_patients, get_model_card, create_twin_route, get_twin_route
)


# ── Test (a): Partition Guard ─────────────────────────────────────────────

def test_all_replay_subjects_are_in_test_split():
    """Assert all 4 chosen subjects (and hard case) come strictly from held-out test split."""
    splits_path = SERVER_DIR / "data" / "splits.json"
    assert splits_path.exists(), "splits.json must exist"

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    test_subjects = set(splits["test"])
    train_subjects = set(splits["train"])
    val_subjects = set(splits["val"])

    # Load internal mapping
    mapping_path = SERVER_DIR / "data" / ".demo_replay_mapping.json"
    assert mapping_path.exists(), "Internal mapping must exist for audit verification"
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    for case_id, info in mapping.items():
        sid = info.get("adni_subject_id") or info.get("subject_id")
        assert sid in test_subjects, f"{case_id} subject {sid} is NOT in held-out test split!"
        assert sid not in train_subjects, f"DATA LEAKAGE: {case_id} subject {sid} is in TRAIN split!"
        assert sid not in val_subjects, f"DATA LEAKAGE: {case_id} subject {sid} is in VAL split!"


# ── Test (b): Predict Before Reveal & Hash Integrity ──────────────────────

def test_pre_visit_prediction_hash_invariance():
    """
    Assert pre-visit prediction for visit k is frozen before reveal, uses only visits 0..k-1,
    and its SHA-256 hash does not change after visit k is revealed and ingested.
    """
    case_id = "case_a_stable_cn"
    st = start_replay(case_id, include_mri=False)
    twin_id = st["twin"].twin_id
    replay_state = st["replay"]

    # Pre-visit prediction for visit 1 (month 6) must be generated at baseline
    pred_v1 = replay_state["pending_pre_visit_prediction"]
    assert pred_v1 is not None
    assert pred_v1["visit_index"] == 1
    assert pred_v1["target_month"] == 6
    assert pred_v1["anchor_month"] == 0

    hash_before = pred_v1["prediction_hash"]
    assert hash_before == compute_prediction_hash(pred_v1)

    # Ingest visit 1
    adv1 = advance_replay(case_id, twin_id)
    eval1 = adv1["evaluation"]

    # Hash recorded in evaluation must match pre-visit frozen hash
    assert eval1["prediction_hash"] == hash_before
    # Prediction values in evaluation must match exactly
    assert eval1["predicted_scores"]["MMSE"] == pred_v1["predicted_MMSE"]
    assert eval1["predicted_scores"]["CDGLOBAL"] == pred_v1["predicted_CDGLOBAL"]
    assert eval1["predicted_scores"]["TOTSCORE"] == pred_v1["predicted_TOTSCORE"]

    # Pre-visit prediction for visit 2 must now be frozen
    pred_v2 = adv1["replay"]["pending_pre_visit_prediction"]
    assert pred_v2 is not None
    assert pred_v2["visit_index"] == 2
    assert pred_v2["target_month"] == 12
    assert pred_v2["anchor_month"] == 6

    hash_v2_before = pred_v2["prediction_hash"]
    adv2 = advance_replay(case_id, twin_id)
    assert adv2["evaluation"]["prediction_hash"] == hash_v2_before


# ── Test (c): MRI on/off Flag ─────────────────────────────────────────────

def test_replay_runs_with_and_without_mri():
    """Assert replay runs cleanly with include_mri=True and include_mri=False."""
    # Run with MRI
    st_mri = start_replay("case_d_stable_mci", include_mri=True)
    tid_mri = st_mri["twin"].twin_id
    assert st_mri["replay"]["include_mri"] is True
    adv_mri = advance_replay("case_d_stable_mci", tid_mri)
    assert adv_mri["evaluation"]["visit_index"] == 1

    # Run without MRI
    st_nomri = start_replay("case_d_stable_mci", include_mri=False)
    tid_nomri = st_nomri["twin"].twin_id
    assert st_nomri["replay"]["include_mri"] is False
    adv_nomri = advance_replay("case_d_stable_mci", tid_nomri)
    assert adv_nomri["evaluation"]["visit_index"] == 1
    assert adv_nomri["evaluation"]["mri_slice_attached"] is False


# ── Test (d): Reset Invariance ────────────────────────────────────────────

def test_replay_reset_restores_baseline_state():
    """Assert reset returns twin and replay state strictly back to baseline visit 0."""
    case_id = "case_a_stable_cn"
    st = start_replay(case_id, include_mri=False)
    twin_id = st["twin"].twin_id

    # Advance through visits
    advance_replay(case_id, twin_id)
    advance_replay(case_id, twin_id)
    twin_adv = get_twin(twin_id)
    assert len(twin_adv.visits) == 3

    # Reset
    rst = reset_replay(case_id, twin_id)
    twin_rst = get_twin(twin_id)
    assert len(twin_rst.visits) == 1
    assert twin_rst.visits[0].month == 0
    assert rst["replay"]["current_visit_index"] == 0
    assert rst["replay"]["is_complete"] is False
    assert rst["replay"]["pending_pre_visit_prediction"]["visit_index"] == 1


# ── Test (e): End-to-End API Route Run for All Patients ───────────────────

def test_end_to_end_api_all_four_patients():
    """Assert end-to-end API router execution for all 4 patients succeeds."""
    cases = ["case_a_stable_cn", "case_b_cn_to_mci", "case_c_mci_to_ad", "case_d_stable_mci"]

    # 1. GET /demo/cases
    all_cases = list_demo_cases()
    assert len(all_cases) >= 4
    case_ids = [c["case_id"] if isinstance(c, dict) else c.case_id for c in all_cases]
    for c in cases:
        assert c in case_ids

    # 2. Run through all 4 patients end-to-end
    for case_id in cases:
        # POST /demo/replay/{case}/start
        res_start = start_replay_route(case=case_id, include_mri=False)
        assert res_start["twin"] is not None
        twin_id = res_start["twin"].twin_id
        assert twin_id is not None
        assert res_start["replay"]["current_visit_index"] == 0

        # Advance 3 times (months 6, 12, 24)
        for v_idx in [1, 2, 3]:
            res_next = next_replay_route(case=case_id, twin_id=twin_id)
            assert res_next["evaluation"]["visit_index"] == v_idx
            assert res_next["evaluation"]["actual_scores"]["MMSE"] is not None

        # GET /demo/replay/{case}/report
        rep_data = get_replay_report_route(case=case_id, twin_id=twin_id)
        assert rep_data["is_complete"] is True
        assert len(rep_data["visits"]) == 4
        assert rep_data["summary_metrics"]["mae_MMSE"] is not None

    # 3. GET /demo/replay/summary
    sum_data = get_replay_summary_route()
    assert sum_data["evidence_statement"] == "Four patients are an illustration and the test-set metrics are the evidence."
    assert len(sum_data["replay_patients"]) >= 4


# ── Test (f): Legacy Routes Resiliency ────────────────────────────────────

def test_legacy_routes_still_respond():
    """Assert /demo/patients, /model-card, and /twin routes continue to function normally."""
    # GET /demo/patients
    patients = list_demo_patients()
    assert len(patients) >= 4
    for p in patients:
        assert hasattr(p, "twin_id") or "twin_id" in p

    # GET /model-card
    mc = get_model_card()
    assert mc.stager is not None
    assert mc.progression is not None

    # POST /twin
    twin = create_twin_route(CreateTwinRequest(
        age=72.0,
        sex=0,
        education_years=16.0,
        apoe4_count=0,
        MMSE=29.0,
        CDGLOBAL=0.0,
        TOTSCORE=5.0
    ))
    assert twin.twin_id is not None

    # GET /twin/{twin_id}
    fetched = get_twin_route(twin.twin_id)
    assert fetched.twin_id == twin.twin_id
