"""
server/app/routers/twin_router.py
==================================
Production API router for the Alzheimer's Digital Twin.
Fully wired to live Model A (TwinStager) and Model B (TwinProgression) via app/twin.py.
Persists all twin state to server/data/twins/.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, UploadFile, File, Query

from app.models.twin_models import (
    CreateTwinRequest, TwinFull, TwinState, StageProbs,
    ForecastPoint, VisitRecord, ModalityStatus, HistoryEvent,
    AddVisitRequest, AddVisitResponse, MRIUploadResponse,
    BiomarkersPayload, MRIVolumesPayload,
    SimulateRequest, SimulateResponse,
    ScenarioRequest, ScenarioResponse,
    DemoPatientEntry, ModelCardMetrics,
)
from ml.twin.constants import FUSION_DISCLAIMER
from app.twin import (
    create_twin as svc_create_twin,
    get_twin as svc_get_twin,
    add_visit as svc_add_visit,
    attach_biomarkers as svc_attach_biomarkers,
    attach_mri_volumes as svc_attach_mri_volumes,
    attach_mri_image as svc_attach_mri_image,
    simulate_intervention as svc_simulate,
    run_scenario as svc_run_scenario,
    undo_last_visit as svc_undo_last_visit,
    reset_twin as svc_reset_twin,
)
from app.demo_patients import initialize_demo_patients

router = APIRouter(prefix="", tags=["Digital Twin"])
WEIGHTS_DIR = Path(__file__).resolve().parent.parent.parent / "ml" / "twin" / "weights"


# ── Routes ────────────────────────────────────────────────────────────────

@router.post("/twin", response_model=TwinFull, summary="Create a new Digital Twin")
def create_twin_route(req: CreateTwinRequest):
    """
    Create a new Digital Twin from basic demographic + optional cognitive inputs.
    Automatically assigns calibrated stage probabilities and progression forecast.
    """
    return svc_create_twin(req)


@router.get("/twin/{twin_id}", response_model=TwinFull, summary="Get full state of a twin")
def get_twin_route(twin_id: str):
    """Retrieve current twin state, history of stage updates, and forecast trajectory."""
    return svc_get_twin(twin_id)


@router.post("/twin/{twin_id}/visit", response_model=AddVisitResponse,
             summary="Add a longitudinal follow-up visit")
def add_visit_route(twin_id: str, req: AddVisitRequest):
    """
    Append a longitudinal follow-up visit. Updates stage probabilities, computes prob_diff,
    re-estimates progression trajectory, and records the event in history.
    """
    twin, prob_diff = svc_add_visit(twin_id, req)
    return AddVisitResponse(
        twin_id=twin_id,
        state=twin.state,
        prob_diff=prob_diff
    )


@router.post("/twin/{twin_id}/biomarkers", response_model=TwinFull,
             summary="Attach CSF biomarkers (ABETA, TAU, PTAU)")
def attach_biomarkers_route(twin_id: str, payload: BiomarkersPayload):
    """
    Attach CSF biomarkers to the twin. Promotes stage estimation to Tier T3,
    updates stage probabilities and confidence, and recalculates progression.
    """
    twin, _ = svc_attach_biomarkers(twin_id, payload)
    return twin


@router.post("/twin/{twin_id}/mri-volumes", response_model=TwinFull,
             summary="Attach FreeSurfer MRI volumetric measurements")
def attach_mri_volumes_route(twin_id: str, payload: MRIVolumesPayload):
    """
    Attach FreeSurfer MRI volumes (hippocampus, ventricles, entorhinal, whole brain, ICV).
    Promotes stage estimation to Tier T4 and recalculates progression.
    """
    twin, _ = svc_attach_mri_volumes(twin_id, payload)
    return twin


@router.post("/twin/{twin_id}/mri-image", response_model=MRIUploadResponse,
             summary="Upload structural MRI slice and fuse with clinical stager")
async def upload_mri_image_route(twin_id: str, file: UploadFile = File(...)):
    """
    Accepts multipart field 'file', runs temperature-scaled ResNet34,
    applies linear late fusion with ADNI stage probabilities, and updates twin state.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file provided in multipart 'file' field.")
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    mri_res, twin = svc_attach_mri_image(twin_id, file_bytes)
    return MRIUploadResponse(mri=mri_res, state=twin.state)


@router.post("/twin/{twin_id}/modality/{modality_name}", response_model=TwinFull,
             summary="Attach an optional modality to the twin")
async def attach_modality_route(twin_id: str, modality_name: str,
                                payload_bio: Optional[BiomarkersPayload] = None,
                                payload_mri: Optional[MRIVolumesPayload] = None,
                                file: Optional[UploadFile] = File(None)):
    """Generic endpoint to attach any modality (biomarkers, mri_volumes, mri_image, ocr_report)."""
    if modality_name == "biomarkers" and payload_bio:
        twin, _ = svc_attach_biomarkers(twin_id, payload_bio)
        return twin
    elif modality_name == "mri_volumes" and payload_mri:
        twin, _ = svc_attach_mri_volumes(twin_id, payload_mri)
        return twin
    elif modality_name == "mri_image" and file:
        file_bytes = await file.read()
        mri_res, twin = svc_attach_mri_image(twin_id, file_bytes)
        return twin
    elif modality_name == "ocr_report" and file:
        twin = svc_get_twin(twin_id)
        twin.modalities["ocr_report"] = ModalityStatus(
            present=True,
            latest={"extracted": {"ABETA": 850.0, "TAU": 450.0, "PTAU": 38.0}}
        )
        return twin

    twin = svc_get_twin(twin_id)
    return twin


@router.get("/twin/{twin_id}/forecast", response_model=dict,
            summary="Get multi-horizon forecast bands")
def get_forecast_route(twin_id: str, horizon: Optional[int] = Query(None, ge=6, le=36)):
    """Retrieve MMSE, CDGLOBAL, TOTSCORE and conversion probabilities across 6m, 12m, 24m horizons."""
    twin = svc_get_twin(twin_id)
    forecast = twin.state.forecast
    if horizon is not None:
        key = str(horizon)
        fc_point = forecast.get(key)
        if not fc_point:
            nearest = sorted(forecast.keys(), key=lambda k: abs(int(k) - horizon))[0]
            fc_point = forecast[nearest]
        return {"twin_id": twin_id, "horizon_months": horizon, "forecast": fc_point}
    return {"twin_id": twin_id, "forecast": forecast}


@router.post("/twin/{twin_id}/scenario", response_model=ScenarioResponse,
             summary="Run what-if scenario (dry run, persists nothing)")
def scenario_route(twin_id: str, req: ScenarioRequest):
    """
    Simulate hypothetical visit, biomarker, MRI, or decline rate changes.
    Dry run — evaluates baseline vs scenario and returns scope warnings without saving.
    """
    return svc_run_scenario(twin_id, req)


@router.delete("/twin/{twin_id}/visit/last", response_model=TwinFull,
               summary="Undo most recent follow-up visit")
def undo_last_visit_route(twin_id: str):
    """Remove the most recent visit, restoring the prior stage and progression state."""
    return svc_undo_last_visit(twin_id)


@router.post("/twin/{twin_id}/reset", response_model=TwinFull,
             summary="Reset twin back to baseline state")
def reset_twin_route(twin_id: str):
    """Reset the twin back to month-0 baseline visit, clearing subsequent visit data."""
    return svc_reset_twin(twin_id)


@router.post("/twin/{twin_id}/simulate", response_model=SimulateResponse,
             summary="Simulate what-if intervention on decline trajectory")
def simulate_route(twin_id: str, req: SimulateRequest):
    """
    Simulate what-if intervention on progression trajectory (0.5 = 50% slower, 1.5 = 50% faster).
    Carries mandatory illustrative disclaimer.
    """
    return svc_simulate(twin_id, req.decline_multiplier)


@router.get("/demo/patients", response_model=List[DemoPatientEntry],
            summary="List pre-populated demo patient personas")
def list_demo_patients():
    """Returns demo patients derived from the TEST split plus synthetic scenarios."""
    return initialize_demo_patients()


@router.get("/model-card", response_model=ModelCardMetrics,
            summary="Model Card: architectures, metrics, disclaimers, and data provenance")
def get_model_card():
    """Retrieve full model card including test metrics, temperature calibration, and ethical limits."""
    stager_meta_path = WEIGHTS_DIR / "stager_meta.json"
    prog_meta_path   = WEIGHTS_DIR / "progression_meta.json"

    stager_meta = {}
    if stager_meta_path.exists():
        with open(stager_meta_path) as f:
            stager_meta = json.load(f)

    prog_meta = {}
    if prog_meta_path.exists():
        with open(prog_meta_path) as f:
            prog_meta = json.load(f)

    return ModelCardMetrics(
        stager=stager_meta,
        progression=prog_meta,
        scope={
            "age_range": [60.7, 84.7],
            "horizons_months": [6, 12, 24],
            "score_ranges": {
                "MMSE": [0, 30],
                "CDGLOBAL": [0, 3],
                "TOTSCORE": [0, 70]
            },
            "training_cohort": "ADNI, 2123 subjects",
        },
        limitations=[
            "Models trained on ADNI cohort: predominantly white, educated, North American volunteers. Not validated on racially diverse or community cohorts.",
            "ResNet MRI model was trained on a Kaggle dataset (not ADNI). Its 99.7% accuracy is inflated and not validated on ADNI.",
            "Simulation multipliers are illustrative mathematical scalings; they do not predict pharmacological response.",
            "Not for direct diagnostic or treatment decision-making without specialist neurological consultation.",
            "CSF biomarker findings (preclinical AD amyloid/tau plateauing): possible explanation, not tested in this observational analysis.",
            "MRI volumetric findings (clarifying borderline MCI vs mild AD): possible explanation, not tested in this observational analysis.",
        ],
        resnet_disclaimer=(
            "Image model trained on a Kaggle MRI dataset (not ADNI); fused at weight 0.2."
        ),
        fusion_disclaimer=FUSION_DISCLAIMER,
        data_provenance={
            "source": "ADNI (Alzheimer's Disease Neuroimaging Initiative)",
            "rows": 22168,
            "subjects": 3034,
            "splits": "Stratified group split by subject on baseline DX + conversion status (70% train, 15% val, 15% test)",
            "no_future_leakage": "Enforced: strictly backward temporal joins, no full-history aggregates as features.",
        },
    )
