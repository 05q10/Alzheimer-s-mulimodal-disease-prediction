import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
SERVER_DIR = Path(__file__).resolve().parent.parent
TEST_JPG = SERVER_DIR / "ml" / "image_model" / "test.jpg"


def test_mri_upload_and_tempered_softmax():
    """PART 1.1: Upload test.jpg to /twin/{id}/mri-image and verify tempered softmax & mapped probs."""
    res = client.post("/twin", json={"name": "MRI Test Patient", "age": 72.0, "MMSE": 26.0, "CDGLOBAL": 0.5})
    assert res.status_code == 200, res.text
    twin_id = res.json()["twin_id"]
    adni_probs = res.json()["state"]["stage_probs"]

    assert TEST_JPG.exists(), f"Missing test image at {TEST_JPG}"
    with open(TEST_JPG, "rb") as f:
        upload_res = client.post(
            f"/twin/{twin_id}/mri-image",
            files={"file": ("test.jpg", f, "image/jpeg")}
        )

    assert upload_res.status_code == 200, upload_res.text
    data = upload_res.json()
    assert "mri" in data
    assert "state" in data

    mri = data["mri"]
    assert "predicted_class" in mri
    assert "probs" in mri
    assert "mapped_probs" in mri

    # Tempering check: confidence must not be 1.00 (strictly < 1.0)
    top_prob = max(mri["probs"].values())
    assert top_prob < 1.0, f"Confidence {top_prob} is not tempered (should be < 1.0)"

    # Verify probs keys
    for k in ["NonDemented", "VeryMild", "Mild", "Moderate"]:
        assert k in mri["probs"]

    # Verify mapped probs sum to ~1.0
    mapped = mri["mapped_probs"]
    for k in ["CN", "MCI", "AD"]:
        assert k in mapped
    assert abs(sum(mapped.values()) - 1.0) < 0.01

    # PART 2: Check late fusion
    state = data["state"]
    assert "fusion" in state
    fusion = state["fusion"]
    assert fusion is not None
    assert fusion["weight"] == 0.2
    assert "adni_probs" in fusion
    assert "mri_probs" in fusion
    assert "final_probs" in fusion

    # Linear blend verification
    w = 0.2
    expected_cn = (1 - w) * adni_probs["CN"] + w * mapped["CN"]
    assert abs(fusion["final_probs"]["CN"] - expected_cn) < 0.05


def test_reset_and_undo_snapshot():
    """PART 1.2: Add 2 visits, reset, verify exact restoration. Test undo recomputation consistency."""
    # Create twin
    create_res = client.post("/twin", json={
        "name": "Jane Reset Test",
        "demographics": {"age": 71.0, "sex": 1, "education_years": 16.0, "apoe4": 1},
        "scores": {"MMSE": 27.0, "CDGLOBAL": 0.5, "TOTSCORE": 12.0},
        "lifestyle": {"PhysicalActivity": 3, "DietQuality": 4}
    })
    assert create_res.status_code == 200
    twin_init = create_res.json()
    twin_id = twin_init["twin_id"]
    init_probs = twin_init["state"]["stage_probs"]

    today = datetime.now(timezone.utc).date()
    v1_date = (today - timedelta(days=180)).strftime("%Y-%m-%d")
    v2_date = today.strftime("%Y-%m-%d")

    # Add Visit 1
    r_v1 = client.post(f"/twin/{twin_id}/visit", json={"exam_date": v1_date, "MMSE": 25.0, "CDGLOBAL": 0.5})
    assert r_v1.status_code == 200
    v1_probs = r_v1.json()["state"]["stage_probs"]

    # Add Visit 2
    r_v2 = client.post(f"/twin/{twin_id}/visit", json={"exam_date": v2_date, "MMSE": 22.0, "CDGLOBAL": 1.0})
    assert r_v2.status_code == 200

    # Call /reset
    r_reset = client.post(f"/twin/{twin_id}/reset")
    assert r_reset.status_code == 200
    reset_data = r_reset.json()

    # Probabilities after reset MUST equal initial probabilities exactly
    assert reset_data["state"]["stage_probs"]["CN"] == init_probs["CN"]
    assert reset_data["state"]["stage_probs"]["MCI"] == init_probs["MCI"]
    assert reset_data["state"]["stage_probs"]["AD"] == init_probs["AD"]
    assert len(reset_data["visits"]) == 1
    assert reset_data["scenarios"] == []

    # Check history event is "reset"
    last_event = reset_data["history"][-1]
    assert last_event["event"] == "reset"
    assert "recorded_at" in last_event

    # Now re-add Visit 1 and Visit 2, then test Undo
    client.post(f"/twin/{twin_id}/visit", json={"exam_date": v1_date, "MMSE": 25.0, "CDGLOBAL": 0.5})
    client.post(f"/twin/{twin_id}/visit", json={"exam_date": v2_date, "MMSE": 22.0, "CDGLOBAL": 1.0})

    # Undo Visit 2
    r_undo = client.delete(f"/twin/{twin_id}/visit/last")
    assert r_undo.status_code == 200
    undo_data = r_undo.json()

    # After undoing Visit 2, probabilities should match Visit 1
    assert abs(undo_data["state"]["stage_probs"]["CN"] - v1_probs["CN"]) < 1e-4
    assert abs(undo_data["state"]["stage_probs"]["MCI"] - v1_probs["MCI"]) < 1e-4
    assert abs(undo_data["state"]["stage_probs"]["AD"] - v1_probs["AD"]) < 1e-4


def test_real_dates_and_validation():
    """PART 1.3: Real dates, baseline_date, future date rejection, month & age derivations."""
    base_date = "2024-01-15"
    res = client.post("/twin", json={
        "name": "Date Test Patient",
        "baseline_date": base_date,
        "age": 70.0,
        "MMSE": 28.0,
        "CDGLOBAL": 0.0
    })
    assert res.status_code == 200
    data = res.json()
    twin_id = data["twin_id"]
    assert data["baseline_date"] == base_date
    assert data["visits"][0]["exam_date"] == base_date
    assert data["visits"][0]["age"] == 70.0

    # Verify forecast points have month and date
    fc = data["state"]["forecast"]
    assert "12" in fc
    assert fc["12"]["month"] == 12
    assert fc["12"]["date"] is not None

    # Observed visit with future date must be rejected
    future_date = (datetime.now(timezone.utc).date() + timedelta(days=30)).strftime("%Y-%m-%d")
    r_future = client.post(f"/twin/{twin_id}/visit", json={
        "exam_date": future_date,
        "MMSE": 26.0,
        "source": "user"
    })
    assert r_future.status_code == 422, "Future date for observed visit must be rejected"

    # Valid visit ~6 months after baseline
    exam_date = "2024-07-15"
    r_valid = client.post(f"/twin/{twin_id}/visit", json={
        "exam_date": exam_date,
        "MMSE": 27.0,
        "source": "user"
    })
    assert r_valid.status_code == 200
    twin_data = client.get(f"/twin/{twin_id}").json()
    last_v = twin_data["visits"][-1]
    assert last_v["exam_date"] == exam_date
    assert last_v["month"] == 6
    assert round(last_v["age"], 1) == 70.5
