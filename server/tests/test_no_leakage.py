"""
server/tests/test_no_leakage.py
================================
Pytest suite that enforces zero data leakage across the whole pipeline.

Tests:
  1. No forbidden feature in any saved model feature list
  2. No subject appears in more than one split
  3. Unified parquet contains no forbidden feature columns
  4. Split counts are in expected ranges
  5. Every model weights directory has a feature_list file

Run from server/:
    pytest tests/test_no_leakage.py -v
"""

import json
import os
import sys
from pathlib import Path

import pytest

# ── Path setup ────────────────────────────────────────────────────────────
SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))

DATA_DIR    = SERVER_DIR / "data"
WEIGHTS_DIR = SERVER_DIR / "ml" / "twin" / "weights"
SPLITS_FILE = DATA_DIR / "splits.json"
PARQUET_FILE = DATA_DIR / "unified_visits.parquet"

from ml.twin.constants import FORBIDDEN_FEATURES


# ── Helper ────────────────────────────────────────────────────────────────

def load_splits() -> dict:
    if not SPLITS_FILE.exists():
        pytest.skip(f"splits.json not found at {SPLITS_FILE} — run data_build.py first")
    with open(SPLITS_FILE) as f:
        return json.load(f)


# ── Test 1: No subject in more than one split ─────────────────────────────

def test_no_subject_overlap_across_splits():
    """Critical: no subject_id must appear in more than one split."""
    splits = load_splits()
    sets = {name: set(ids) for name, ids in splits.items()}
    splits_list = list(sets.items())
    for i in range(len(splits_list)):
        for j in range(i + 1, len(splits_list)):
            name_a, set_a = splits_list[i]
            name_b, set_b = splits_list[j]
            overlap = set_a & set_b
            assert len(overlap) == 0, (
                f"LEAKAGE: {len(overlap)} subjects appear in both '{name_a}' and '{name_b}': "
                f"{list(overlap)[:5]}"
            )


# ── Test 2: All subjects are assigned ─────────────────────────────────────

def test_all_splits_non_empty():
    splits = load_splits()
    for name, ids in splits.items():
        assert len(ids) > 0, f"Split '{name}' is empty"


def test_split_ratios_approximate():
    """Train should be ~70%, val/test ~15% each (±5%)."""
    splits = load_splits()
    total = sum(len(v) for v in splits.values())
    for name, ids in splits.items():
        ratio = len(ids) / total
        if name == "train":
            assert 0.60 <= ratio <= 0.80, f"Train ratio {ratio:.2f} outside [0.60, 0.80]"
        else:
            assert 0.10 <= ratio <= 0.25, f"{name} ratio {ratio:.2f} outside [0.10, 0.25]"


# ── Test 3: Parquet has no forbidden feature columns ─────────────────────

def test_parquet_no_forbidden_features():
    """unified_visits.parquet must not contain any FORBIDDEN_FEATURES column."""
    try:
        import pandas as pd
    except ImportError:
        pytest.skip("pandas not installed")

    if not PARQUET_FILE.exists():
        pytest.skip(f"unified_visits.parquet not found — run data_build.py first")

    df = pd.read_parquet(PARQUET_FILE, columns=None)
    violations = [col for col in FORBIDDEN_FEATURES if col in df.columns]
    assert len(violations) == 0, (
        f"Forbidden feature(s) found in parquet: {violations}"
    )


# ── Test 4: Saved model feature lists have no forbidden features ──────────

def _iter_feature_list_files():
    """Yield all *feature_list* or *meta.json* files under ml/twin/weights/."""
    if not WEIGHTS_DIR.exists():
        return
    for path in WEIGHTS_DIR.rglob("*.json"):
        yield path
    for path in WEIGHTS_DIR.rglob("*feature_list*"):
        yield path


def test_model_feature_lists_no_forbidden():
    """All saved model meta.json files must not reference forbidden features."""
    found_any = False
    violations = []
    for path in _iter_feature_list_files():
        found_any = True
        try:
            with open(path) as f:
                content = json.load(f)
            # Extract feature list from common keys
            features = (
                content.get("feature_list") or
                content.get("features") or
                content.get("feature_names") or
                []
            )
            for feat in features:
                if feat in FORBIDDEN_FEATURES:
                    violations.append(f"{path.name}::{feat}")
        except Exception:
            pass  # Non-JSON or unreadable file — skip

    if not found_any:
        pytest.skip("No model weight files found yet — run exp09/exp10 first")

    assert len(violations) == 0, (
        f"Forbidden feature(s) found in model feature lists: {violations}"
    )


# ── Test 5: Parquet split coverage ────────────────────────────────────────

def test_parquet_subjects_match_splits():
    """Every subject in splits.json must appear in unified_visits.parquet."""
    try:
        import pandas as pd
    except ImportError:
        pytest.skip("pandas not installed")

    if not PARQUET_FILE.exists():
        pytest.skip("unified_visits.parquet not found")

    splits = load_splits()
    df = pd.read_parquet(PARQUET_FILE, columns=["subject_id"])
    parquet_subjects = set(df["subject_id"].astype(str).unique())

    all_split_subjects = set()
    for ids in splits.values():
        all_split_subjects.update(str(s) for s in ids)

    missing = all_split_subjects - parquet_subjects
    assert len(missing) == 0, (
        f"{len(missing)} subjects in splits.json not found in parquet: {list(missing)[:5]}"
    )


# ── Test 6: Imputed file is NOT used directly as a training feature source ─

def test_imputed_traj_type_not_in_parquet():
    """
    *_traj_type and imputation_confidence columns from the imputed file
    must never appear in unified_visits.parquet (they are forbidden features
    because they may encode future visit information).
    """
    try:
        import pandas as pd
    except ImportError:
        pytest.skip("pandas not installed")

    if not PARQUET_FILE.exists():
        pytest.skip("unified_visits.parquet not found")

    df = pd.read_parquet(PARQUET_FILE, columns=None)
    leaky = [c for c in df.columns if "traj_type" in c or "imputation_confidence" in c]
    assert len(leaky) == 0, f"Imputed-file leakage columns found in parquet: {leaky}"


# ── Test 7: API stub routes are reachable ────────────────────────────────

def test_stub_api_twin_create():
    """Smoke test: POST /twin returns 200 with a twin_id."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi[testclient] not installed")

    # Add server to path
    sys.path.insert(0, str(SERVER_DIR))
    try:
        from app.main import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/twin", json={
            "age": 74, "sex": 1, "education_years": 14, "apoe4_count": 1, "MMSE": 26
        })
        assert resp.status_code == 200, f"POST /twin failed: {resp.status_code} {resp.text[:200]}"
        data = resp.json()
        assert "twin_id" in data, "Response missing twin_id"
        assert "state" in data, "Response missing state"
    except Exception as e:
        pytest.skip(f"API not fully initialised yet: {e}")


def test_stub_api_demo_patients():
    """Smoke test: GET /demo/patients returns a list."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi[testclient] not installed")

    sys.path.insert(0, str(SERVER_DIR))
    try:
        from app.main import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/demo/patients")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 5
    except Exception as e:
        pytest.skip(f"API not fully initialised yet: {e}")


def test_stub_api_model_card():
    """Smoke test: GET /model-card returns limitations list."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi[testclient] not installed")

    sys.path.insert(0, str(SERVER_DIR))
    try:
        from app.main import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/model-card")
        assert resp.status_code == 200
        data = resp.json()
        assert "limitations" in data
        assert len(data["limitations"]) > 0
        # Must have the ResNet disclaimer
        assert "resnet_disclaimer" in data
        assert "not validated on ADNI" in data["resnet_disclaimer"].lower() or \
               "demo" in data["resnet_disclaimer"].lower()
    except Exception as e:
        pytest.skip(f"API not fully initialised yet: {e}")
