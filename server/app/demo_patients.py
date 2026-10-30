"""
server/app/demo_patients.py
===========================
Pre-populated clinical personas for demonstration and evaluation.
- All real personas are derived strictly from subjects in the TEST SPLIT only.
- Real subject_ids are completely scrubbed (privacy & non-identification).
- Includes synthetic personas explicitly flagged with is_synthetic=True.
"""

from typing import List, Dict, Any
from app.models.twin_models import DemoPatientEntry, CreateTwinRequest, VisitRecord, AddVisitRequest
from app.twin import create_twin, add_visit

DEMO_PERSONAS = [
    {
        "alias": "Eleanor Vance",
        "persona_label": "Eleanor Vance — Stable Cognitively Normal (Test Cohort)",
        "is_synthetic": False,
        "baseline_stage": "CN",
        "trajectory": "Stable CN over 24 months",
        "demographics": {"age": 76.0, "sex": 1, "education_years": 16.0, "apoe4_count": 0},
        "visits": [
            {"month": 0, "MMSE": 30.0, "CDGLOBAL": 0.0, "TOTSCORE": 3.0, "DX_obs": 1},
            {"month": 6, "MMSE": 29.0, "CDGLOBAL": 0.0, "TOTSCORE": 6.0, "DX_obs": 1},
            {"month": 12, "MMSE": 29.0, "CDGLOBAL": 0.0, "TOTSCORE": 2.0, "DX_obs": 1},
            {"month": 24, "MMSE": 30.0, "CDGLOBAL": 0.0, "TOTSCORE": 3.3, "DX_obs": 1},
        ],
    },
    {
        "alias": "Arthur Pendelton",
        "persona_label": "Arthur Pendelton — CN to MCI Converter (Test Cohort)",
        "is_synthetic": False,
        "baseline_stage": "CN",
        "trajectory": "Converts CN -> MCI at Month 12/24",
        "demographics": {"age": 81.0, "sex": 0, "education_years": 18.0, "apoe4_count": 1},
        "visits": [
            {"month": 0, "MMSE": 29.0, "CDGLOBAL": 0.0, "TOTSCORE": 9.7, "DX_obs": 1},
            {"month": 6, "MMSE": 28.0, "CDGLOBAL": 0.0, "TOTSCORE": 6.7, "DX_obs": 1},
            {"month": 12, "MMSE": 26.0, "CDGLOBAL": 0.0, "TOTSCORE": 8.3, "DX_obs": 2},
            {"month": 24, "MMSE": 29.0, "CDGLOBAL": 0.5, "TOTSCORE": 8.3, "DX_obs": 2},
        ],
    },
    {
        "alias": "Margaret Sterling",
        "persona_label": "Margaret Sterling — MCI to AD Converter (Test Cohort)",
        "is_synthetic": False,
        "baseline_stage": "MCI",
        "trajectory": "Progresses MCI -> AD by Month 24",
        "demographics": {"age": 80.0, "sex": 1, "education_years": 14.0, "apoe4_count": 2},
        "visits": [
            {"month": 0, "MMSE": 24.0, "CDGLOBAL": 0.5, "TOTSCORE": 11.0, "DX_obs": 2},
            {"month": 6, "MMSE": 24.0, "CDGLOBAL": 0.5, "TOTSCORE": 17.0, "DX_obs": 2},
            {"month": 12, "MMSE": 24.0, "CDGLOBAL": 0.5, "TOTSCORE": 15.0, "DX_obs": 2},
            {"month": 24, "MMSE": 22.0, "CDGLOBAL": 0.5, "TOTSCORE": 19.3, "DX_obs": 3},
        ],
    },
    {
        "alias": "Harold Finch",
        "persona_label": "Harold Finch — Stable Amnestic MCI (Test Cohort)",
        "is_synthetic": False,
        "baseline_stage": "MCI",
        "trajectory": "Stable Amnestic MCI over 24 months",
        "demographics": {"age": 78.0, "sex": 0, "education_years": 16.0, "apoe4_count": 1},
        "visits": [
            {"month": 0, "MMSE": 29.0, "CDGLOBAL": 0.5, "TOTSCORE": 6.3, "DX_obs": 2},
            {"month": 6, "MMSE": 27.0, "CDGLOBAL": 0.5, "TOTSCORE": 7.0, "DX_obs": 2},
            {"month": 12, "MMSE": 29.0, "CDGLOBAL": 0.5, "TOTSCORE": 4.0, "DX_obs": 2},
            {"month": 24, "MMSE": 29.0, "CDGLOBAL": 0.5, "TOTSCORE": 5.7, "DX_obs": 2},
        ],
    },
    {
        "alias": "Dr. Walter Bishop",
        "persona_label": "Dr. Walter Bishop — Synthetic High APOE4 Carrier",
        "is_synthetic": True,
        "baseline_stage": "MCI",
        "trajectory": "Synthetic High-Risk Simulation Scenario",
        "demographics": {"age": 72.0, "sex": 0, "education_years": 20.0, "apoe4_count": 2},
        "visits": [
            {"month": 0, "MMSE": 26.0, "CDGLOBAL": 0.5, "TOTSCORE": 14.0, "DX_obs": 2},
            {"month": 12, "MMSE": 23.0, "CDGLOBAL": 0.5, "TOTSCORE": 18.0, "DX_obs": 2},
        ],
    },
]

# Track initialized demo twin IDs
_demo_twin_ids: List[DemoPatientEntry] = []


def initialize_demo_patients() -> List[DemoPatientEntry]:
    """Ensure demo twin instances exist in memory and disk storage."""
    global _demo_twin_ids
    if _demo_twin_ids:
        return _demo_twin_ids

    entries = []
    for p in DEMO_PERSONAS:
        v0 = p["visits"][0]
        req = CreateTwinRequest(
            age=p["demographics"]["age"],
            sex=p["demographics"]["sex"],
            education_years=p["demographics"]["education_years"],
            apoe4_count=p["demographics"]["apoe4_count"],
            MMSE=v0["MMSE"],
            CDGLOBAL=v0["CDGLOBAL"],
            TOTSCORE=v0["TOTSCORE"],
            is_synthetic=p["is_synthetic"],
            persona_label=p["persona_label"],
        )
        twin = create_twin(req)

        # Add remaining visits
        for v in p["visits"][1:]:
            add_visit(twin.twin_id, AddVisitRequest(
                month=v["month"],
                MMSE=v["MMSE"],
                CDGLOBAL=v["CDGLOBAL"],
                TOTSCORE=v["TOTSCORE"],
                DX_obs=v["DX_obs"],
                source="adni" if not p["is_synthetic"] else "synthetic"
            ))

        entry = DemoPatientEntry(
            twin_id=twin.twin_id,
            persona_label=p["persona_label"],
            is_synthetic=p["is_synthetic"],
            baseline_stage=p["baseline_stage"],
            trajectory=p["trajectory"],
            n_visits=len(p["visits"])
        )
        entries.append(entry)

    _demo_twin_ids = entries
    return _demo_twin_ids
