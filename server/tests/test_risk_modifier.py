"""
server/tests/test_risk_modifier.py
==================================
Tests for Part 4: Risk Factors and Lifestyle Modifier.
Verifies:
  1. Modifier is strictly bounded within hard caps [0.85, 1.35].
  2. Modifier is monotone (worse lifestyle never lowers risk).
  3. Has zero effect when no lifestyle inputs are given (multiplier = 1.0).
  4. Never alters current clinical stage or baseline stage probabilities.
  5. Scenario endpoint accepts lifestyle overrides and visibly modulates forecast.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from ml.twin.risk_modifier import compute_lifestyle_modifier

client = TestClient(app)


def test_modifier_bounds_and_null_effect():
    """Modifier has no effect when empty and remains strictly bounded in [0.85, 1.35]."""
    # 1. Null / empty input
    res_none = compute_lifestyle_modifier(None)
    assert res_none["decline_multiplier"] == 1.0
    assert res_none["hazard_multiplier"] == 1.0
    assert len(res_none["contributions"]) == 0

    res_empty = compute_lifestyle_modifier({})
    assert res_empty["decline_multiplier"] == 1.0
    assert res_empty["hazard_multiplier"] == 1.0

    # 2. Extreme bad lifestyle (everything adverse)
    extreme_bad = {
        "smoking": 1,
        "alcohol": 35,
        "physical_activity": 1,
        "diet_quality": 1,
        "sleep_quality": 1,
        "hypertension": 1,
        "diabetes": 1,
        "depression": 1,
        "head_injury": 1,
        "cardiovascular_disease": 1,
    }
    res_bad = compute_lifestyle_modifier(extreme_bad)
    assert res_bad["decline_multiplier"] <= 1.35, f"Decline exceeded cap: {res_bad['decline_multiplier']}"
    assert res_bad["decline_multiplier"] == 1.35, f"Should hit 1.35 cap: {res_bad['decline_multiplier']}"
    assert res_bad["hazard_multiplier"] == 1.35
    assert len(res_bad["contributions"]) > 5

    # 3. Extreme protective lifestyle
    extreme_good = {
        "smoking": 0,
        "alcohol": 0,
        "physical_activity": 5,
        "diet_quality": 5,
        "sleep_quality": 5,
        "hypertension": 0,
        "diabetes": 0,
        "depression": 0,
        "head_injury": 0,
        "cardiovascular_disease": 0,
    }
    res_good = compute_lifestyle_modifier(extreme_good)
    assert res_good["decline_multiplier"] >= 0.85, f"Decline below cap: {res_good['decline_multiplier']}"
    assert res_good["decline_multiplier"] < 1.0, f"Protective lifestyle should reduce decline: {res_good['decline_multiplier']}"


def test_modifier_monotonicity():
    """Worse lifestyle must never lower risk."""
    # Physical activity: 5 (high) -> 3 (moderate) -> 1 (sedentary)
    mult_act5 = compute_lifestyle_modifier({"physical_activity": 5})["decline_multiplier"]
    mult_act3 = compute_lifestyle_modifier({"physical_activity": 3})["decline_multiplier"]
    mult_act1 = compute_lifestyle_modifier({"physical_activity": 1})["decline_multiplier"]
    assert mult_act5 <= mult_act3 <= mult_act1, f"Activity order failed: {mult_act5}, {mult_act3}, {mult_act1}"

    # Alcohol: 0 -> 18 -> 30 units/week
    mult_alc0 = compute_lifestyle_modifier({"alcohol": 0})["decline_multiplier"]
    mult_alc18 = compute_lifestyle_modifier({"alcohol": 18})["decline_multiplier"]
    mult_alc30 = compute_lifestyle_modifier({"alcohol": 30})["decline_multiplier"]
    assert mult_alc0 <= mult_alc18 <= mult_alc30, f"Alcohol order failed: {mult_alc0}, {mult_alc18}, {mult_alc30}"

    # Smoking: 0 -> 1
    mult_smk0 = compute_lifestyle_modifier({"smoking": 0})["decline_multiplier"]
    mult_smk1 = compute_lifestyle_modifier({"smoking": 1})["decline_multiplier"]
    assert mult_smk0 <= mult_smk1, f"Smoking order failed: {mult_smk0}, {mult_smk1}"


def test_lifestyle_never_changes_current_stage_probabilities():
    """Lifestyle modifier must modulate forecast decline but NEVER alter current stage probs."""
    base_payload = {
        "age": 73,
        "sex": 0,
        "education_years": 16,
        "scores": {
            "MMSE": 25.0,
            "CDGLOBAL": 0.5,
            "TOTSCORE": 15.0,
        }
    }

    # 1. Neutral twin
    res_neutral = client.post("/twin", json=base_payload).json()
    probs_neutral = res_neutral["state"]["stage_probs"]
    stage_neutral = res_neutral["state"]["stage"]

    # 2. Twin with high adverse lifestyle
    payload_bad = dict(base_payload)
    payload_bad["lifestyle"] = {
        "smoking": 1,
        "alcohol": 30,
        "physical_activity": 1,
        "hypertension": 1,
        "diabetes": 1,
    }
    res_bad = client.post("/twin", json=payload_bad).json()
    probs_bad = res_bad["state"]["stage_probs"]
    stage_bad = res_bad["state"]["stage"]

    # Stage and probabilities must be EXACTLY identical
    assert stage_neutral == stage_bad, f"Stage differed: {stage_neutral} vs {stage_bad}"
    assert probs_neutral["CN"] == probs_bad["CN"], f"P(CN) differed: {probs_neutral['CN']} vs {probs_bad['CN']}"
    assert probs_neutral["MCI"] == probs_bad["MCI"], f"P(MCI) differed: {probs_neutral['MCI']} vs {probs_bad['MCI']}"
    assert probs_neutral["AD"] == probs_bad["AD"], f"P(AD) differed: {probs_neutral['AD']} vs {probs_bad['AD']}"

    # But forecast trajectory should reflect faster decline
    fc_neutral_12 = res_neutral["state"]["forecast"]["12"]["MMSE_mean"]
    fc_bad_12 = res_bad["state"]["forecast"]["12"]["MMSE_mean"]
    assert fc_bad_12 < fc_neutral_12, f"Bad lifestyle did not worsen 12m MMSE: bad={fc_bad_12}, neutral={fc_neutral_12}"


def test_scenario_with_lifestyle_override():
    """Scenario endpoint accepts lifestyle overrides, modestly changing future projections."""
    twin_res = client.post("/twin", json={
        "age": 71,
        "sex": 1,
        "education_years": 14,
        "scores": {"MMSE": 26.0, "CDGLOBAL": 0.5, "TOTSCORE": 14.0},
        "lifestyle": {"physical_activity": 3, "alcohol": 5}
    }).json()
    twin_id = twin_res["twin_id"]

    # Run scenario with high activity + healthy diet override
    scen_healthy = client.post(f"/twin/{twin_id}/scenario", json={
        "horizon_months": 12,
        "lifestyle": {"physical_activity": 5, "diet_quality": 5, "alcohol": 0}
    }).json()

    # Run scenario with heavy alcohol + sedentary override
    scen_unhealthy = client.post(f"/twin/{twin_id}/scenario", json={
        "horizon_months": 12,
        "lifestyle": {"physical_activity": 1, "alcohol": 35, "smoking": 1}
    }).json()

    fc_h = scen_healthy["scenario"]["forecast"]["12"]
    fc_u = scen_unhealthy["scenario"]["forecast"]["12"]

    # Healthy lifestyle scenario achieves higher MMSE (slower decline) than unhealthy
    assert fc_h["MMSE_mean"] > fc_u["MMSE_mean"], f"Healthy MMSE ({fc_h['MMSE_mean']}) <= Unhealthy MMSE ({fc_u['MMSE_mean']})"
