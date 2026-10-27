"""
End-to-end integration tests for Alzheimer's Digital Twin API.
Tests strict compliance with API_CONTRACT.md, patient-builder flow, scenario engine,
visit undo/reset, model-card scope metadata, and legacy route compatibility.
"""

import os
import json
import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_get_demo_patients():
    """Verify demo patients endpoint returns valid personas with scrubbed IDs."""
    response = client.get("/demo/patients")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 5

    for p in data:
        # Crucial privacy requirement: never expose real ADNI subject_id
        assert "subject_id" not in p
        assert "RID" not in p
        assert "PTID" not in p
        assert "twin_id" in p
        assert "persona_label" in p
        assert "baseline_stage" in p
        assert "is_synthetic" in p

        # Check synthetic persona flags
        if "Synthetic" in p.get("persona_label", ""):
            assert p["is_synthetic"] is True


def test_twin_lifecycle_and_multimodal_refinement():
    """
    Test full twin lifecycle:
    1. Create twin from basic info (Tier T1b: demo + MMSE)
    2. Add follow-up visit
    3. Attach CSF biomarkers (Refines to Tier T3)
    4. Attach FreeSurfer MRI volumes (Refines to Tier T4)
    5. Request Progression Forecast
    6. Run What-If Simulation
    """
    # 1. Create Twin
    create_payload = {
        "age": 74.0,
        "sex": 0,
        "education_years": 16.0,
        "apoe4_count": 1,
        "MMSE": 26.0,
        "is_synthetic": True,
        "persona_label": "Test Subject"
    }

    create_res = client.post("/twin", json=create_payload)
    assert create_res.status_code == 200
    twin = create_res.json()
    twin_id = twin["twin_id"]
    assert twin_id is not None
    assert twin["state"]["tier_used"] == "T1b"
    assert "CN" in twin["state"]["stage_probs"]
    assert "MCI" in twin["state"]["stage_probs"]
    assert "AD" in twin["state"]["stage_probs"]
    prob_sum = sum(twin["state"]["stage_probs"].values())
    assert 0.99 <= prob_sum <= 1.01

    # 2. Add Visit
    visit_payload = {
        "month": 12,
        "MMSE": 24.0,
        "CDGLOBAL": 0.5,
        "source": "user"
    }
    visit_res = client.post(f"/twin/{twin_id}/visit", json=visit_payload)
    assert visit_res.status_code == 200
    visit_data = visit_res.json()
    assert visit_data["twin_id"] == twin_id
    assert "prob_diff" in visit_data
    assert "state" in visit_data

    # 3. Attach CSF Biomarkers
    bio_payload = {
        "ABETA": 750.0,
        "TAU": 380.0,
        "PTAU": 39.0
    }
    bio_res = client.post(f"/twin/{twin_id}/biomarkers", json=bio_payload)
    assert bio_res.status_code == 200
    bio_data = bio_res.json()
    assert bio_data["modalities"]["biomarkers"]["present"] is True
    assert bio_data["state"]["tier_used"] == "T3"

    # 4. Attach MRI Volumes
    mri_payload = {
        "hippocampus": 6200.0,
        "entorhinal": 3100.0,
        "ventricles": 39000.0,
        "whole_brain": 1010000.0,
        "icv": 1500000.0
    }
    mri_res = client.post(f"/twin/{twin_id}/mri-volumes", json=mri_payload)
    assert mri_res.status_code == 200
    mri_data = mri_res.json()
    assert mri_data["modalities"]["mri_volumes"]["present"] is True
    assert mri_data["state"]["tier_used"] == "T4"

    # 5. Progression Forecast
    forecast_res = client.get(f"/twin/{twin_id}/forecast?horizon=24")
    assert forecast_res.status_code == 200
    forecast_data = forecast_res.json()
    assert forecast_data["twin_id"] == twin_id
    assert forecast_data["horizon_months"] == 24
    assert "forecast" in forecast_data
    assert "MMSE_mean" in forecast_data["forecast"]
    assert "conversion_prob" in forecast_data["forecast"]

    # 6. Counterfactual Simulation
    sim_res = client.post(f"/twin/{twin_id}/simulate", json={"decline_multiplier": 1.5})
    assert sim_res.status_code == 200
    sim_data = sim_res.json()
    assert sim_data["twin_id"] == twin_id
    assert "simulated_forecast" in sim_data
    assert "disclaimer" in sim_data
    assert "illustrative" in sim_data["disclaimer"].lower() or "clinical prediction" in sim_data["disclaimer"].lower()


def test_patient_builder_flow_and_modality_chips():
    """
    Test patient-builder structured creation:
    - nested demographics, scores, lifestyle, and allele biomarkers
    - runs Kaggle CatBoost filling kaggle_clinical chip
    - runs Allele ensemble filling allele_ensemble chip
    - asserts APVOLUME is not mapped to FreeSurfer hippocampus
    """
    builder_payload = {
        "name": "Evelyn Reed",
        "demographics": {
            "age": 73.0,
            "sex": 1,
            "education_years": 16.0,
            "apoe4": 1
        },
        "scores": {
            "MMSE": 25.0,
            "CDGLOBAL": 0.5,
            "TOTSCORE": 17.0,
            "CDRSB": 2.0,
            "MOCA": 23.0,
            "FAQ": 4.0
        },
        "biomarkers": {
            "ABETA": 710.0,
            "TAU": 410.0,
            "PTAU": 40.0,
            "APVOLUME": 3250.0  # Kept strictly separate from FreeSurfer hippocampus
        },
        "lifestyle": {
            "BMI": 26.2,
            "PhysicalActivity": 3,
            "DietQuality": 4,
            "SleepQuality": 3
        }
    }

    res = client.post("/twin", json=builder_payload)
    assert res.status_code == 200
    twin = res.json()

    assert twin["name"] == "Evelyn Reed"
    # Highest tier allowed by demographics + neuropsych + CSF is T3
    assert twin["state"]["tier_used"] == "T3"

    # Verify lifestyle chip filled by Kaggle CatBoost
    assert twin["modalities"]["kaggle_clinical"]["present"] is True
    assert "lifestyle_risk" in twin["modalities"]["kaggle_clinical"]["latest"]

    # Verify allele ensemble chip filled
    assert twin["modalities"]["allele_ensemble"]["present"] is True

    # Assert APVOLUME was NOT mapped to FreeSurfer MRI volumes
    assert twin["modalities"]["mri_volumes"]["present"] is False


def test_scenario_endpoint_and_dry_run():
    """
    Verify POST /twin/{id}/scenario:
    - Rejects horizons other than 6, 12, 24 with 422
    - Dry run: persists nothing to disk or memory
    - Scope warnings produced for impossible jump (>10 pts/yr) and MMSE/CDR mismatch
    """
    # 1. Create twin
    create_res = client.post("/twin", json={
        "age": 75.0,
        "sex": 0,
        "education_years": 14.0,
        "apoe4_count": 0,
        "MMSE": 27.0,
        "CDGLOBAL": 0.0
    })
    assert create_res.status_code == 200
    twin = create_res.json()
    twin_id = twin["twin_id"]
    orig_stage_probs = twin["state"]["stage_probs"]

    # 2. Reject invalid horizon with 422
    invalid_res = client.post(f"/twin/{twin_id}/scenario", json={"horizon_months": 18})
    assert invalid_res.status_code == 422

    # 3. Valid scenario with extreme hypothetical jump (MMSE 27 -> 12 over 12 months)
    # and inconsistent CDR (MMSE 12 with CDGLOBAL 0.0)
    scen_payload = {
        "horizon_months": 12,
        "hypothetical_visit": {
            "MMSE": 12.0,
            "CDGLOBAL": 0.0
        },
        "decline_multiplier": 1.5
    }
    scen_res = client.post(f"/twin/{twin_id}/scenario", json=scen_payload)
    assert scen_res.status_code == 200
    scen_data = scen_res.json()

    assert scen_data["twin_id"] == twin_id
    assert scen_data["horizon_months"] == 12
    assert "baseline" in scen_data
    assert "scenario" in scen_data
    assert "delta" in scen_data
    assert "scope_warnings" in scen_data
    assert len(scen_data["scope_warnings"]) > 0

    # Scope warnings must flag the implausible jump and score inconsistency
    warning_text = " ".join(scen_data["scope_warnings"]).lower()
    assert "pts/yr" in warning_text or "implausible" in warning_text or "inconsistent" in warning_text

    # Verify DRY RUN: twin in memory and on disk was NOT mutated
    get_res = client.get(f"/twin/{twin_id}")
    assert get_res.status_code == 200
    current_twin = get_res.json()
    assert current_twin["state"]["stage_probs"] == orig_stage_probs


def test_undo_and_reset():
    """Verify DELETE /twin/{id}/visit/last (undo) and POST /twin/{id}/reset."""
    create_res = client.post("/twin", json={
        "age": 72.0,
        "sex": 1,
        "education_years": 16.0,
        "apoe4_count": 0,
        "MMSE": 28.0
    })
    twin = create_res.json()
    twin_id = twin["twin_id"]
    assert len(twin["visits"]) == 1

    # Add follow-up visit month 12
    v1_res = client.post(f"/twin/{twin_id}/visit", json={"month": 12, "MMSE": 26.0})
    assert v1_res.status_code == 200

    # Add follow-up visit month 24
    v2_res = client.post(f"/twin/{twin_id}/visit", json={"month": 24, "MMSE": 22.0})
    assert v2_res.status_code == 200

    check_twin = client.get(f"/twin/{twin_id}").json()
    assert len(check_twin["visits"]) == 3

    # Undo last visit (month 24)
    undo_res = client.delete(f"/twin/{twin_id}/visit/last")
    assert undo_res.status_code == 200
    undone_twin = undo_res.json()
    assert len(undone_twin["visits"]) == 2
    assert undone_twin["visits"][-1]["month"] == 12

    # Reset to baseline
    reset_res = client.post(f"/twin/{twin_id}/reset")
    assert reset_res.status_code == 200
    reset_twin = reset_res.json()
    assert len(reset_twin["visits"]) == 1
    assert reset_twin["visits"][0]["month"] == 0


def test_model_card_scope():
    """Verify GET /model-card contains training split scope metadata."""
    res = client.get("/model-card")
    assert res.status_code == 200
    card = res.json()
    assert "scope" in card
    scope = card["scope"]
    assert "age_range" in scope
    assert scope["age_range"] == [60.7, 84.7]
    assert "horizons_months" in scope
    assert scope["horizons_months"] == [6, 12, 24]
    assert "score_ranges" in scope
    assert scope["score_ranges"]["MMSE"] == [0, 30]
    assert "training_cohort" in scope
    assert "2123 subjects" in scope["training_cohort"]


def test_legacy_routes_still_respond():
    """Ensure /alele, /image, /extract-report, /base, /lstm-predict continue to respond."""
    # 1. /alele
    alele_payload = {
        "ABETA": 800.0,
        "TAU": 300.0,
        "MMSE": 28.0,
        "APVOLUME": 3500.0,
        "GENOTYPE": "3,3"
    }
    r = client.post("/alele", json=alele_payload)
    assert r.status_code in [200, 503]

    # 2. /image
    img = Image.new("RGB", (224, 224), color="black")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    r = client.post("/image", files={"file": ("scan.jpg", buf, "image/jpeg")})
    assert r.status_code in [200, 503]

    # 3. /extract-report
    fake_pdf = b"%PDF-1.4 simulated blood report"
    r = client.post("/extract-report", files={"file": ("report.pdf", fake_pdf, "application/pdf")})
    assert r.status_code in [200, 503]

    # 4. /base
    r = client.post("/base", json={"Age": 70, "MMSE": 26, "BMI": 25.0})
    assert r.status_code in [200, 503]

    # 5. /lstm-predict
    lstm_payload = {
        "visits": [
            {"visit_month": 0, "entry_age": 70.0, "CDGLOBAL": 0.0, "MMSCORE": 28.0, "TOTSCORE": 10.0}
        ]
    }
    r = client.post("/lstm-predict", json=lstm_payload)
    assert r.status_code in [200, 503]
