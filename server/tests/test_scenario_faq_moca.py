"""
server/tests/test_scenario_faq_moca.py
======================================
Automated regression tests verifying that:
1. CatBoost TwinStager is directly sensitive to FAQ and MoCA.
2. CatBoost TwinProgression is directly sensitive to FAQ and MoCA.
3. Scenario Lab (POST /twin/{id}/scenario) propagates FAQ and MoCA overrides into re-staging and conversion risks.
4. GET /twin/{twin_id}/visit/{idx}/mri serves structural MRI slices.
5. Legacy twin alias resolution auto-initializes twin-case-1 with explainability and input usage.
"""

import pytest
from app.twin import get_stager, get_progression, create_twin, run_scenario, get_twin, get_twin_explain, get_input_usage
from app.models.twin_models import CreateTwinRequest, ScenarioRequest
from app.routers.twin_router import get_visit_mri_route


def test_stager_sensitivity_to_faq_and_moca():
    st = get_stager()
    base_input = {
        "entry_age": 72.0, "sex": 0, "education_years": 16.0, "apoe4_count": 1,
        "MMSE": 26.0, "CDGLOBAL": 0.5, "TOTSCORE": 12.0, "FAQ": 0.0, "MOCA": 28.0
    }
    impaired_input = {
        "entry_age": 72.0, "sex": 0, "education_years": 16.0, "apoe4_count": 1,
        "MMSE": 26.0, "CDGLOBAL": 0.5, "TOTSCORE": 12.0, "FAQ": 27.0, "MOCA": 18.0
    }
    res_base = st.predict_one(base_input)
    res_imp = st.predict_one(impaired_input)

    assert res_imp["probs"]["AD"] > res_base["probs"]["AD"] * 2.0, (
        f"AD prob must increase substantially with severe FAQ/MoCA impairment. Got base={res_base['probs']['AD']}, imp={res_imp['probs']['AD']}"
    )


def test_progression_sensitivity_to_faq_and_moca():
    pr = get_progression()
    base_visit = {
        "entry_age": 72.0, "sex": 0, "education_years": 16.0, "apoe4_count": 1,
        "MMSE": 26.0, "CDGLOBAL": 0.5, "TOTSCORE": 12.0, "FAQ": 0.0, "MOCA": 28.0
    }
    imp_visit = {
        "entry_age": 72.0, "sex": 0, "education_years": 16.0, "apoe4_count": 1,
        "MMSE": 26.0, "CDGLOBAL": 0.5, "TOTSCORE": 12.0, "FAQ": 27.0, "MOCA": 18.0
    }
    fc_base = pr.predict_forecast(base_visit)
    fc_imp = pr.predict_forecast(imp_visit)

    assert fc_imp["12"]["conversion_prob"] > fc_base["12"]["conversion_prob"], (
        f"12m conversion prob must increase with severe FAQ/MoCA. Got base={fc_base['12']['conversion_prob']}, imp={fc_imp['12']['conversion_prob']}"
    )


def test_scenario_lab_faq_moca_override():
    req = CreateTwinRequest(
        name="Scenario Test Twin",
        age=72.0, sex=0, education_years=16.0, apoe4_count=1,
        MMSE=26.0, CDGLOBAL=0.5, TOTSCORE=12.0, FAQ=0.0, MOCA=28.0
    )
    twin = create_twin(req)

    # 1. Normal FAQ/MOCA
    scen_normal = run_scenario(twin.twin_id, ScenarioRequest(
        horizon_months=12,
        hypothetical_visit={"MMSE": 26.0, "CDGLOBAL": 0.5, "TOTSCORE": 12.0, "FAQ": 0.0, "MOCA": 28.0}
    ))

    # 2. Impaired FAQ (27) and MOCA (18)
    scen_impaired = run_scenario(twin.twin_id, ScenarioRequest(
        horizon_months=12,
        hypothetical_visit={"MMSE": 26.0, "CDGLOBAL": 0.5, "TOTSCORE": 12.0, "FAQ": 27.0, "MOCA": 18.0}
    ))

    p_ad_normal = scen_normal.scenario["stage_probs"]["AD"]
    p_ad_imp = scen_impaired.scenario["stage_probs"]["AD"]

    assert p_ad_imp > p_ad_normal * 2.0, (
        f"Scenario Lab must reflect FAQ/MoCA overrides in stage distribution. Got base={p_ad_normal}, imp={p_ad_imp}"
    )


def test_demo_twin_auto_resolution_and_mri():
    twin = get_twin("twin-case-1")
    assert twin.twin_id == "twin-case-1"
    assert len(twin.visits) >= 2

    # Verify explain and input usage
    exp = get_twin_explain("twin-case-1")
    assert exp.twin_id == "twin-case-1"
    iu = get_input_usage("twin-case-1")
    assert len(iu.items) == 43

    # Verify MRI route
    res0 = get_visit_mri_route("twin-case-1", 0)
    assert res0.status_code == 200 or hasattr(res0, "path")
    assert "case_c_mci_to_ad_v0.jpg" in str(res0.path)

