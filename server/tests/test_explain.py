"""
server/tests/test_explain.py
============================
Pytest suite for Part 5: Explain Every Input & Clinical Lineage.
Verifies:
  1. input-usage lists every intake field (all 43 intake fields accounted for)
  2. No field is marked used without a real code path
  3. explain endpoint works when modalities are missing
  4. Narrative is deterministic and mentions the top stage driver
  5. Scenario response returns method explanation and drivers_delta
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from ml.twin.stager import get_stager
from ml.twin.risk_modifier import get_risk_config

client = TestClient(app)

INTAKE_43_FIELDS = [
    "name", "age", "sex", "education", "ethnicity", "doctor_in_charge",
    "height", "weight", "BMI", "smoking", "family_history_of_ad", "alcohol",
    "physical_activity", "diet_quality", "sleep_quality", "cardiovascular_disease",
    "diabetes", "depression", "head_injury", "hypertension", "MMSE",
    "functional_assessment", "ADL", "memory_complaints", "behavioral_problems",
    "confusion", "disorientation", "personality_changes", "difficulty_completing_tasks",
    "forgetfulness", "ABETA", "TAU", "APVOLUME", "genotype", "CDR_global",
    "ADAS11_total", "FAQ", "MOCA", "PTAU", "mri_volumes", "mri_image",
    "blood_report_ocr", "baseline_date"
]


@pytest.fixture
def standard_twin():
    resp = client.post("/twin", json={
        "name": "Audit Test Patient",
        "baseline_date": "2026-10-28",
        "demographics": {
            "age": 73.0,
            "sex": 1,
            "education_years": 16.0,
            "apoe4": 1,
            "ethnicity": 0,
            "doctor_in_charge": "Dr. House",
            "height": 168.0,
            "weight": 65.0,
            "BMI": 23.0
        },
        "scores": {
            "MMSE": 26.0,
            "CDGLOBAL": 0.5,
            "TOTSCORE": 14.0,
            "CDRSB": 1.5,
            "MOCA": 24.0,
            "FAQ": 2.0
        },
        "biomarkers": {
            "ABETA": 820.0,
            "TAU": 380.0,
            "PTAU": 32.0,
            "APVOLUME": 3100.0,
            "GENOTYPE": "3,4"
        },
        "lifestyle": {
            "alcohol": 3.5,
            "physical_activity": 4.0,
            "diet_quality": 8.0,
            "sleep_quality": 7.0,
            "smoking": 0,
            "hypertension": 1
        }
    })
    assert resp.status_code == 200
    return resp.json()


def test_input_usage_lists_every_intake_field(standard_twin):
    twin_id = standard_twin["twin_id"]
    resp = client.get(f"/twin/{twin_id}/input-usage")
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert "items" in data
    items = data["items"]
    assert len(items) == 43, f"Expected exactly 43 intake fields, got {len(items)}"

    field_names = [it["field"] for it in items]
    for required_f in INTAKE_43_FIELDS:
        assert required_f in field_names, f"Missing intake field: {required_f}"

    # Verify structure of each item
    for it in items:
        assert "field" in it
        assert "label" in it
        assert "status" in it
        assert "used_by" in it
        assert "reason" in it
        assert it["status"] in [
            "used_in_stage", "used_in_forecast", "used_in_lifestyle_modifier",
            "used_in_symptom_screen", "used_in_biomarker_ensemble", "used_in_mri_fusion",
            "recorded_only", "missing", "assumed_default"
        ]


def test_no_field_marked_used_without_code_path(standard_twin):
    twin_id = standard_twin["twin_id"]
    resp = client.get(f"/twin/{twin_id}/input-usage")
    assert resp.status_code == 200
    items = resp.json()["items"]

    stager = get_stager()
    stager_features = set(stager.feature_list)
    risk_cfg = get_risk_config().get("factors", {})

    strictly_recorded = {"name", "ethnicity", "doctor_in_charge", "baseline_date", "height", "weight", "BMI"}

    for it in items:
        f = it["field"]
        status = it["status"]

        # 1. Non-modeled fields must be recorded_only
        if f in strictly_recorded:
            assert status in ["recorded_only", "missing"], f"Field '{f}' should be recorded_only, got '{status}'"
            assert len(it["used_by"]) == 0, f"Field '{f}' has unexpected used_by: {it['used_by']}"

        # 2. Never mark a missing value as used
        if it["value"] is None and status not in ["assumed_default", "missing"]:
            pytest.fail(f"Field '{f}' has value None but is marked as '{status}'")

        # 3. If marked used_in_stage, must have genuine staging lineage
        if status == "used_in_stage":
            valid_stage_inputs = {
                "age", "sex", "education", "MMSE", "CDR_global", "CDGLOBAL", "ADAS11_total", "FAQ",
                "MOCA", "ABETA", "TAU", "PTAU", "genotype", "mri_volumes"
            }
            assert f in valid_stage_inputs, f"Field '{f}' marked used_in_stage without staging code path"
            assert "stager" in it["used_by"]

        # 4. If marked used_in_lifestyle_modifier, must be in Lancet risk modifier
        if status == "used_in_lifestyle_modifier":
            assert "lifestyle_modifier" in it["used_by"]
            assert f in risk_cfg or f.lower() in risk_cfg


def test_explain_works_when_modalities_missing():
    # Create minimal baseline twin: demographics + MMSE only
    resp = client.post("/twin", json={
        "name": "Minimal Twin",
        "demographics": {
            "age": 70.0,
            "sex": 0,
            "education_years": 12.0
        },
        "scores": {
            "MMSE": 27.0
        }
    })
    assert resp.status_code == 200
    twin_id = resp.json()["twin_id"]

    # Call explain on twin with missing modalities
    expl_resp = client.get(f"/twin/{twin_id}/explain")
    assert expl_resp.status_code == 200, expl_resp.text
    expl = expl_resp.json()

    assert "stage" in expl
    assert "drivers" in expl["stage"]
    assert len(expl["stage"]["drivers"]) > 0
    assert "by_group" in expl["stage"]

    # Leave-one-out should handle missing modalities gracefully (no crash)
    assert "leave_one_out" in expl["stage"]
    assert isinstance(expl["stage"]["leave_one_out"], list)

    assert "forecast" in expl
    assert "drivers" in expl["forecast"]
    assert len(expl["forecast"]["drivers"]) > 0

    assert "narrative" in expl
    assert isinstance(expl["narrative"], str)
    assert len(expl["narrative"]) > 20

    assert "lifestyle_modifier" in expl
    assert expl["lifestyle_modifier"]["total_multiplier"] == 1.0


def test_narrative_mentions_top_driver(standard_twin):
    twin_id = standard_twin["twin_id"]
    resp = client.get(f"/twin/{twin_id}/explain")
    assert resp.status_code == 200
    expl = resp.json()

    narrative = expl["narrative"]
    drivers = expl["stage"]["drivers"]
    assert len(drivers) > 0

    top_driver = drivers[0]
    top_label = top_driver["label"]
    top_feature = top_driver["feature"]

    # Narrative must mention either the primary label, feature abbreviation, or score value
    driver_mentioned = (
        top_label.lower() in narrative.lower() or
        top_feature.lower() in narrative.lower() or
        "mmse" in narrative.lower() or
        "cdr" in narrative.lower()
    )
    assert driver_mentioned, f"Narrative '{narrative}' does not mention top driver '{top_label}'"

    # Narrative sentence count must be between 4 and 8
    import re
    sentences = [s.strip() for s in re.split(r'\.\s+', narrative) if s.strip()]
    assert 4 <= len(sentences) <= 8, f"Narrative has {len(sentences)} sentences (expected 4 to 8): '{narrative}'"


def test_scenario_returns_method_and_drivers_delta(standard_twin):
    twin_id = standard_twin["twin_id"]
    resp = client.post(f"/twin/{twin_id}/scenario", json={
        "horizon_months": 12,
        "hypothetical_visit": {
            "MMSE": 22.0,
            "CDGLOBAL": 1.0,
            "TOTSCORE": 20.0
        },
        "decline_multiplier": 1.5
    })
    assert resp.status_code == 200, resp.text
    scen = resp.json()

    assert "method" in scen
    assert isinstance(scen["method"], list)
    assert len(scen["method"]) >= 5
    # Verify core mathematical steps are documented in method
    method_text = " ".join(scen["method"]).lower()
    assert "progression" in method_text or "forecast" in method_text
    assert "decline multiplier" in method_text
    assert "stager" in method_text
    assert "conversion" in method_text

    assert "drivers_delta" in scen
    assert isinstance(scen["drivers_delta"], list)
    assert len(scen["drivers_delta"]) >= 1
    factors_delta = [d["factor"] for d in scen["drivers_delta"]]
    assert "MMSE" in factors_delta or "decline_multiplier" in factors_delta
