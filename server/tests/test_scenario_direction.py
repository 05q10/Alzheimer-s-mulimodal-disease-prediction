import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_sweep_cognitive_scores_monotonic():
    """Worse cognitive/functional scores never lower P(AD at horizon) or progression risk."""
    r = client.post("/twin", json={
        "demographics": {"age": 72.0, "sex": 1, "education_years": 16.0, "apoe4_count": 1},
        "scores": {"MMSE": 24.0, "CDGLOBAL": 0.5, "TOTSCORE": 18.0, "FAQ": 6.0},
    })
    assert r.status_code == 200
    twin_id = r.json()["twin_id"]

    # Sweep from mild to severe
    score_sweep = [
        {"MMSE": 28.0, "CDGLOBAL": 0.0, "TOTSCORE": 8.0, "FAQ": 0.0},
        {"MMSE": 25.0, "CDGLOBAL": 0.5, "TOTSCORE": 15.0, "FAQ": 4.0},
        {"MMSE": 22.0, "CDGLOBAL": 0.5, "TOTSCORE": 22.0, "FAQ": 9.0},
        {"MMSE": 19.0, "CDGLOBAL": 1.0, "TOTSCORE": 28.0, "FAQ": 15.0},
        {"MMSE": 15.0, "CDGLOBAL": 1.0, "TOTSCORE": 36.0, "FAQ": 20.0},
    ]

    prev_p_ad = -1.0
    prev_headline_risk = -1.0

    for scores in score_sweep:
        resp = client.post(f"/twin/{twin_id}/scenario", json={
            "horizon_months": 12,
            "hypothetical_visit": scores,
        })
        assert resp.status_code == 200
        data = resp.json()

        p_ad = data["scenario"]["p_ad_at_horizon"]
        headline_risk = data["scenario"]["headline_risk_value"]
        delta = data["delta"]

        assert "direction" in delta, "delta must contain 'direction'"
        assert "p_ad_at_horizon" in delta["direction"]
        assert delta["direction"]["p_ad_at_horizon"] in ["worse", "better", "same"]

        if prev_p_ad >= 0:
            assert p_ad >= prev_p_ad - 1e-4, f"Worse scores lowered P(AD at horizon): {p_ad} < {prev_p_ad}"
        if prev_headline_risk >= 0:
            assert headline_risk >= prev_headline_risk - 1e-4, f"Worse scores lowered headline risk: {headline_risk} < {prev_headline_risk}"

        prev_p_ad = p_ad
        prev_headline_risk = headline_risk


def test_sweep_decline_multiplier_monotonic():
    """A higher decline_multiplier never lowers P(AD at horizon) or progression risk."""
    # 1. MCI patient
    r_mci = client.post("/twin", json={
        "demographics": {"age": 70.0, "sex": 0, "education_years": 14.0, "apoe4_count": 1},
        "scores": {"MMSE": 24.0, "CDGLOBAL": 0.5, "TOTSCORE": 17.0, "FAQ": 5.0},
    })
    twin_mci = r_mci.json()["twin_id"]

    mults = [0.5, 0.8, 1.0, 1.2, 1.5]
    prev_risk = -1.0
    for m in mults:
        resp = client.post(f"/twin/{twin_mci}/scenario", json={
            "horizon_months": 12,
            "decline_multiplier": m,
        })
        assert resp.status_code == 200
        data = resp.json()
        h_risk = data["scenario"]["headline_risk_value"]
        assert data["delta"]["direction"]["headline_risk"] in ["better", "same", "worse"]

        if prev_risk >= 0:
            assert h_risk >= prev_risk - 1e-4, f"Higher multiplier {m} lowered MCI conversion risk: {h_risk} < {prev_risk}"
        prev_risk = h_risk

    # 2. AD patient
    r_ad = client.post("/twin", json={
        "demographics": {"age": 75.0, "sex": 1, "education_years": 12.0, "apoe4_count": 2},
        "scores": {"MMSE": 18.0, "CDGLOBAL": 1.0, "TOTSCORE": 28.0, "FAQ": 16.0},
    })
    twin_ad = r_ad.json()["twin_id"]

    prev_prog_risk = -1.0
    for m in mults:
        resp = client.post(f"/twin/{twin_ad}/scenario", json={
            "horizon_months": 12,
            "decline_multiplier": m,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["scenario"]["headline_risk_metric"] == "progression_risk"
        prog_risk = data["scenario"]["headline_risk_value"]

        if prev_prog_risk >= 0:
            assert prog_risk >= prev_prog_risk - 1e-4, f"Higher multiplier {m} lowered AD progression risk: {prog_risk} < {prev_prog_risk}"
        prev_prog_risk = prog_risk


def test_sweep_lifestyle_monotonic():
    """Worse lifestyle never lowers forecast decline or conversion hazard, and never changes current-stage probabilities."""
    r = client.post("/twin", json={
        "demographics": {"age": 71.0, "sex": 0, "education_years": 16.0, "apoe4_count": 0},
        "scores": {"MMSE": 26.0, "CDGLOBAL": 0.5, "TOTSCORE": 14.0, "FAQ": 3.0},
    })
    twin_id = r.json()["twin_id"]

    # 1. Alcohol sweep (0 to 35 units/week)
    prev_haz = -1.0
    for alc in [0, 7, 14, 21, 35]:
        resp = client.post(f"/twin/{twin_id}/scenario", json={"horizon_months": 12, "lifestyle": {"alcohol": alc}}).json()
        haz = resp["scenario"]["forecast"]["12"]["conversion_prob"]
        anchor_probs = resp["scenario"]["anchor_stage_probs"]
        assert resp["scenario"]["anchor_stage_probs"] == resp["baseline"]["anchor_stage_probs"], "Lifestyle changed anchor stage probs!"
        if prev_haz >= 0:
            assert haz >= prev_haz - 1e-4, f"More alcohol {alc} lowered conversion hazard: {haz} < {prev_haz}"
        prev_haz = haz

    # 2. Physical activity sweep (8 down to 0 hrs/week)
    prev_haz = -1.0
    for act in [8, 5, 3, 1, 0]:
        resp = client.post(f"/twin/{twin_id}/scenario", json={"horizon_months": 12, "lifestyle": {"physical_activity": act}}).json()
        haz = resp["scenario"]["forecast"]["12"]["conversion_prob"]
        assert resp["scenario"]["anchor_stage_probs"] == resp["baseline"]["anchor_stage_probs"]
        if prev_haz >= 0:
            assert haz >= prev_haz - 1e-4, f"Less activity {act} lowered conversion hazard: {haz} < {prev_haz}"
        prev_haz = haz

    # 3. Diet quality sweep (9 down to 1 score)
    prev_haz = -1.0
    for diet in [9, 7, 5, 3, 1]:
        resp = client.post(f"/twin/{twin_id}/scenario", json={"horizon_months": 12, "lifestyle": {"diet_quality": diet}}).json()
        haz = resp["scenario"]["forecast"]["12"]["conversion_prob"]
        assert resp["scenario"]["anchor_stage_probs"] == resp["baseline"]["anchor_stage_probs"]
        if prev_haz >= 0:
            assert haz >= prev_haz - 1e-4, f"Worse diet {diet} lowered conversion hazard: {haz} < {prev_haz}"
        prev_haz = haz

    # 4. Sleep quality sweep (9 down to 1 score)
    prev_haz = -1.0
    for slp in [9, 7, 5, 3, 1]:
        resp = client.post(f"/twin/{twin_id}/scenario", json={"horizon_months": 12, "lifestyle": {"sleep_quality": slp}}).json()
        haz = resp["scenario"]["forecast"]["12"]["conversion_prob"]
        assert resp["scenario"]["anchor_stage_probs"] == resp["baseline"]["anchor_stage_probs"]
        if prev_haz >= 0:
            assert haz >= prev_haz - 1e-4, f"Worse sleep {slp} lowered conversion hazard: {haz} < {prev_haz}"
        prev_haz = haz

    # 5. Health conditions: smoking, hypertension, diabetes, depression, head injury, CVD
    for cond in ["smoking", "hypertension", "diabetes", "depression", "head_injury", "cardiovascular_disease"]:
        resp_off = client.post(f"/twin/{twin_id}/scenario", json={"horizon_months": 12, "lifestyle": {cond: 0}}).json()
        resp_on = client.post(f"/twin/{twin_id}/scenario", json={"horizon_months": 12, "lifestyle": {cond: 1}}).json()

        haz_off = resp_off["scenario"]["forecast"]["12"]["conversion_prob"]
        haz_on = resp_on["scenario"]["forecast"]["12"]["conversion_prob"]
        mmse_off = resp_off["scenario"]["forecast"]["12"]["MMSE_mean"]
        mmse_on = resp_on["scenario"]["forecast"]["12"]["MMSE_mean"]

        assert haz_on >= haz_off - 1e-4, f"{cond}=1 lowered conversion hazard: {haz_on} < {haz_off}"
        assert mmse_on <= mmse_off + 1e-4, f"{cond}=1 improved MMSE forecast: {mmse_on} > {mmse_off}"
        assert resp_on["scenario"]["anchor_stage_probs"] == resp_on["baseline"]["anchor_stage_probs"]


def test_mild_ad_plausible_annual_decline():
    """Mild AD (MMSE 20) forecast at 12m must be bounded to typical ADNI 2-4 points/year decline."""
    r = client.post("/twin", json={
        "demographics": {"age": 75.0, "sex": 0, "education_years": 14.0, "apoe4_count": 1},
        "scores": {"MMSE": 20.0, "CDGLOBAL": 1.0, "TOTSCORE": 25.0, "FAQ": 15.0},
    })
    twin_id = r.json()["twin_id"]

    resp = client.post(f"/twin/{twin_id}/scenario", json={"horizon_months": 12}).json()
    fc_12 = resp["scenario"]["forecast"]["12"]
    mmse_12m = fc_12["MMSE_mean"]
    annual_drop = 20.0 - mmse_12m

    # Must be in plausible ADNI range [1.5, 4.5] points/year, NOT dropping 6+ points to 13.9
    assert 1.5 <= annual_drop <= 4.5, f"Mild AD annual drop ({annual_drop}) fell outside ADNI [1.5, 4.5] pts/yr range (forecast MMSE={mmse_12m})"
    assert mmse_12m >= 15.5, f"Forecast MMSE={mmse_12m} is too steep for 12m"


def test_delta_direction_contract():
    """Verify delta = scenario - baseline and directions: worse | better | same."""
    r = client.post("/twin", json={
        "demographics": {"age": 70.0, "sex": 1, "education_years": 16.0, "apoe4_count": 1},
        "scores": {"MMSE": 24.0, "CDGLOBAL": 0.5, "TOTSCORE": 18.0, "FAQ": 6.0},
    })
    twin_id = r.json()["twin_id"]

    # Worse scenario
    resp_worse = client.post(f"/twin/{twin_id}/scenario", json={
        "horizon_months": 12,
        "decline_multiplier": 1.5,
        "lifestyle": {"smoking": 1, "alcohol": 30}
    }).json()

    delta_w = resp_worse["delta"]
    assert "direction" in delta_w
    assert "headline_risk" in delta_w["direction"]
    assert delta_w["direction"]["headline_risk"] == "worse"
    assert delta_w["direction"]["MMSE_mean"] == "worse"

    # Better scenario
    resp_better = client.post(f"/twin/{twin_id}/scenario", json={
        "horizon_months": 12,
        "decline_multiplier": 0.5,
        "lifestyle": {"smoking": 0, "alcohol": 0, "physical_activity": 8, "diet_quality": 9}
    }).json()

    delta_b = resp_better["delta"]
    assert delta_b["direction"]["headline_risk"] == "better"
    assert delta_b["direction"]["MMSE_mean"] == "better"
