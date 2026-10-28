"""
server/tests/test_plausibility.py
=================================
Clinical Archetype, Plausibility Envelope, Monotonicity, and Scenario tests.
Bounds are derived directly from weights/plausibility_envelope.json.
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from ml.twin.stager import get_stager
from ml.twin.progression import get_progression

client = TestClient(app)

SERVER_DIR = Path(__file__).resolve().parent.parent
ENVELOPE_FILE = SERVER_DIR / "ml" / "twin" / "weights" / "plausibility_envelope.json"

@pytest.fixture(scope="module")
def envelope():
    assert ENVELOPE_FILE.exists(), f"Envelope file missing at {ENVELOPE_FILE}"
    with open(ENVELOPE_FILE) as f:
        return json.load(f)


def test_archetype_cn(envelope):
    """CN: age 72, MMSE 29, CDR 0, ADAS11 5, FAQ 0 -> P(CN) >= 0.7, 12m MMSE change inside the CN envelope."""
    stager = get_stager()
    progression = get_progression()

    inputs = {"entry_age": 72, "MMSE": 29.0, "CDGLOBAL": 0.0, "TOTSCORE": 5.0, "FAQTOTAL": 0.0}
    stage_res = stager.predict_one(inputs)
    assert stage_res["probs"]["CN"] >= 0.70, f"Expected P(CN) >= 0.7, got {stage_res['probs']['CN']}"

    fc = progression.predict_forecast(inputs, stage_res["probs"])
    m12_pred = fc["12"]["MMSE_mean"]
    change_12m = m12_pred - 29.0

    cn_env = envelope["horizons"]["12"]["MMSE"]["by_stage"]["CN"]
    p5 = cn_env["p5"]
    p95 = cn_env["p95"]

    assert p5 <= change_12m <= p95, f"12m MMSE change ({change_12m}) outside CN envelope [{p5}, {p95}]"


def test_archetype_mci():
    """MCI: MMSE 27, CDR 0.5, ADAS11 10, FAQ 3 -> MCI top class, 12m MMSE not above current + 1."""
    stager = get_stager()
    progression = get_progression()

    inputs = {"MMSE": 27.0, "CDGLOBAL": 0.5, "TOTSCORE": 10.0, "FAQTOTAL": 3.0}
    stage_res = stager.predict_one(inputs)
    assert stage_res["stage"] == "MCI", f"Expected top class MCI, got {stage_res['stage']}"
    assert stage_res["probs"]["MCI"] > stage_res["probs"]["CN"], "MCI prob should exceed CN"
    assert stage_res["probs"]["MCI"] > stage_res["probs"]["AD"], "MCI prob should exceed AD"

    fc = progression.predict_forecast(inputs, stage_res["probs"])
    assert fc["12"]["MMSE_mean"] <= 27.0 + 1.0, f"12m MMSE ({fc['12']['MMSE_mean']}) above current + 1"


def test_archetype_mild_ad():
    """Mild AD: MMSE 20, CDR 1, ADAS11 25, FAQ 15 -> P(AD) >= 0.7, 12m and 24m MMSE <= current + 1."""
    stager = get_stager()
    progression = get_progression()

    inputs = {"MMSE": 20.0, "CDGLOBAL": 1.0, "TOTSCORE": 25.0, "FAQTOTAL": 15.0}
    stage_res = stager.predict_one(inputs)
    assert stage_res["probs"]["AD"] >= 0.70, f"Expected P(AD) >= 0.7, got {stage_res['probs']['AD']}"

    fc = progression.predict_forecast(inputs, stage_res["probs"])
    assert fc["12"]["MMSE_mean"] <= 20.0 + 1.0, f"12m MMSE ({fc['12']['MMSE_mean']}) above current + 1"
    assert fc["24"]["MMSE_mean"] <= 20.0 + 1.0, f"24m MMSE ({fc['24']['MMSE_mean']}) above current + 1"


def test_archetype_moderate_ad():
    """Moderate AD: MMSE 16, CDR 1, ADAS11 35 -> P(AD) >= 0.8, forecasts at 6/12/24m never above current + 1, reliability 'low'."""
    stager = get_stager()
    progression = get_progression()

    inputs = {"MMSE": 16.0, "CDGLOBAL": 1.0, "TOTSCORE": 35.0}
    stage_res = stager.predict_one(inputs)
    assert stage_res["probs"]["AD"] >= 0.80, f"Expected P(AD) >= 0.8, got {stage_res['probs']['AD']}"

    fc = progression.predict_forecast(inputs, stage_res["probs"])
    assert fc["6"]["MMSE_mean"] <= 16.0 + 1.0, f"6m MMSE ({fc['6']['MMSE_mean']}) above current + 1"
    assert fc["12"]["MMSE_mean"] <= 16.0 + 1.0, f"12m MMSE ({fc['12']['MMSE_mean']}) above current + 1"
    assert fc["24"]["MMSE_mean"] <= 16.0 + 1.0, f"24m MMSE ({fc['24']['MMSE_mean']}) above current + 1"
    assert fc["forecast_reliability"] == "low", f"Expected reliability 'low', got {fc['forecast_reliability']}"


def test_inconsistent_scores_warning():
    """Inconsistent: MMSE 29 with CDR 2 -> scope_warning."""
    res = client.post("/twin", json={
        "age": 72,
        "sex": 0,
        "education_years": 16,
        "scores": {
            "MMSE": 29.0,
            "CDGLOBAL": 2.0,
        }
    })
    assert res.status_code == 200, res.text
    data = res.json()
    warnings = data["state"]["scope_warnings"]
    assert any("Inconsistent" in w or "conflict" in w or "Discordant" in w for w in warnings), f"Missing inconsistency warning in {warnings}"


def test_stager_monotonicity():
    """Monotonicity: lowering MMSE must not raise P(CN); raising ADAS11 must not lower P(AD)."""
    stager = get_stager()

    # 1. Lowering MMSE must not raise P(CN)
    p_cns = []
    for m in range(30, 14, -1):
        res = stager.predict_one({"MMSE": m, "entry_age": 72})
        p_cns.append(res["probs"]["CN"])
    for i in range(len(p_cns) - 1):
        assert p_cns[i] >= p_cns[i + 1], f"Lowering MMSE raised P(CN): {p_cns[i]} -> {p_cns[i+1]}"

    # 2. Raising ADAS11 must not lower P(AD)
    p_ads = []
    for a in range(5, 55, 5):
        res = stager.predict_one({"TOTSCORE": a, "entry_age": 72})
        p_ads.append(res["probs"]["AD"])
    for i in range(len(p_ads) - 1):
        assert p_ads[i] <= p_ads[i + 1], f"Raising ADAS lowered P(AD): {p_ads[i]} -> {p_ads[i+1]}"


def test_scenario_decline_multiplier_monotonicity():
    """Scenario: decline_multiplier 1.5 is equal or worse than 1.0, which is equal or worse than 0.5, for every target."""
    create_res = client.post("/twin", json={
        "age": 74,
        "sex": 0,
        "education_years": 16,
        "scores": {
            "MMSE": 26.0,
            "CDGLOBAL": 0.5,
            "TOTSCORE": 18.0,
        }
    })
    assert create_res.status_code == 200
    twin_id = create_res.json()["twin_id"]

    res_05 = client.post(f"/twin/{twin_id}/scenario", json={"horizon_months": 12, "decline_multiplier": 0.5}).json()
    res_10 = client.post(f"/twin/{twin_id}/scenario", json={"horizon_months": 12, "decline_multiplier": 1.0}).json()
    res_15 = client.post(f"/twin/{twin_id}/scenario", json={"horizon_months": 12, "decline_multiplier": 1.5}).json()

    fc_05 = res_05["scenario"]["forecast"]["12"]
    fc_10 = res_10["scenario"]["forecast"]["12"]
    fc_15 = res_15["scenario"]["forecast"]["12"]

    # MMSE: lower is worse -> mmse_15 <= mmse_10 <= mmse_05
    assert fc_15["MMSE_mean"] <= fc_10["MMSE_mean"] <= fc_05["MMSE_mean"], (
        f"MMSE order violated: 1.5={fc_15['MMSE_mean']}, 1.0={fc_10['MMSE_mean']}, 0.5={fc_05['MMSE_mean']}"
    )

    # CDGLOBAL: higher is worse -> cdg_15 >= cdg_10 >= cdg_05
    assert fc_15["CDGLOBAL_mean"] >= fc_10["CDGLOBAL_mean"] >= fc_05["CDGLOBAL_mean"], (
        f"CDGLOBAL order violated: 1.5={fc_15['CDGLOBAL_mean']}, 1.0={fc_10['CDGLOBAL_mean']}, 0.5={fc_05['CDGLOBAL_mean']}"
    )

    # TOTSCORE: higher is worse -> tot_15 >= tot_10 >= tot_05
    assert fc_15["TOTSCORE_mean"] >= fc_10["TOTSCORE_mean"] >= fc_05["TOTSCORE_mean"], (
        f"TOTSCORE order violated: 1.5={fc_15['TOTSCORE_mean']}, 1.0={fc_10['TOTSCORE_mean']}, 0.5={fc_05['TOTSCORE_mean']}"
    )
