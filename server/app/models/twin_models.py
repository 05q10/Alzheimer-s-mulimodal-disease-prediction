"""
server/app/models/twin_models.py
=================================
Pydantic models for all 8 new Digital Twin API routes.
These are the full production schemas — stubs return fixture data.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


# ── Shared leaf models ─────────────────────────────────────────────────────

class StageProbs(BaseModel):
    CN: float = Field(ge=0, le=1)
    MCI: float = Field(ge=0, le=1)
    AD: float = Field(ge=0, le=1)


class ForecastPoint(BaseModel):
    month: Optional[int] = None
    date: Optional[str] = None          # Calendar date YYYY-MM-DD
    MMSE_mean: Optional[float] = None
    MMSE_lo: Optional[float] = None     # 5th percentile
    MMSE_hi: Optional[float] = None     # 95th percentile
    CDGLOBAL_mean: Optional[float] = None
    CDGLOBAL_lo: Optional[float] = None
    CDGLOBAL_hi: Optional[float] = None
    TOTSCORE_mean: Optional[float] = None
    TOTSCORE_lo: Optional[float] = None
    TOTSCORE_hi: Optional[float] = None
    stage_probs: Optional[StageProbs] = None
    conversion_prob: Optional[float] = None   # P(stage worsens by horizon)


class VisitRecord(BaseModel):
    month: int = Field(ge=0, description="Months since baseline")
    exam_date: Optional[str] = Field(None, description="Observed date YYYY-MM-DD")
    age: Optional[float] = Field(None, description="Derived patient age at this visit")
    MMSE: Optional[float] = Field(None, ge=0, le=30)
    CDGLOBAL: Optional[float] = Field(None, ge=0, le=3)
    TOTSCORE: Optional[float] = Field(None, ge=0, le=70,
                                       description="ADAS11-Cog prorated (0-70, higher=worse)")
    DX_obs: Optional[int] = Field(None, ge=1, le=3,
                                   description="Observed DX at this visit (1=CN,2=MCI,3=AD); null if unknown")
    source: Literal["user", "adni", "synthetic"] = "user"


class ModalityStatus(BaseModel):
    present: bool = False
    latest: Optional[Dict[str, Any]] = None


class TwinState(BaseModel):
    stage_probs: StageProbs
    stage: Literal["CN", "MCI", "AD"]
    confidence: float = Field(ge=0, le=1)
    tier_used: Literal["T1", "T1a", "T1b", "T2a", "T2b", "T2", "T3", "T4"]
    forecast: Dict[str, ForecastPoint] = Field(
        default_factory=dict,
        description="Keys: '6', '12', '24' (months)"
    )
    conversion_risk: Dict[str, float] = Field(
        default_factory=lambda: {"12m": 0.0, "24m": 0.0},
        description="Probability of converting to a worse stage"
    )
    rate_of_decline: Optional[Dict[str, Any]] = None
    shap_drivers: Optional[List[Dict[str, Any]]] = None
    fusion: Optional[Dict[str, Any]] = None
    scope_warnings: Optional[List[str]] = Field(default_factory=list)
    forecast_reliability: Optional[Literal["ok", "low"]] = "ok"


class HistoryEvent(BaseModel):
    recorded_at: str
    timestamp: Optional[str] = None
    stage_probs: StageProbs
    event: str   # e.g. "visit_added", "modality_attached", "twin_created", "reset"
    prob_diff: Optional[Dict[str, float]] = None


class TwinFull(BaseModel):
    twin_id: str
    name: Optional[str] = None
    is_synthetic: bool = False
    persona_label: Optional[str] = None     # e.g. "CN→MCI Converter"
    baseline_date: Optional[str] = None
    baseline_snapshot: Optional[Dict[str, Any]] = None
    demographics: Dict[str, Any]
    lifestyle: Optional[Dict[str, Any]] = None
    modalities: Dict[str, ModalityStatus]
    visits: List[VisitRecord]
    scenarios: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    state: TwinState
    history: List[HistoryEvent]


# ── Request / Input models ──────────────────────────────────────────────────

class DemographicsInput(BaseModel):
    model_config = ConfigDict(extra="allow")
    age: float = Field(ge=18, le=110)
    sex: Literal[0, 1] = Field(description="0=Male, 1=Female")
    education_years: float = Field(ge=0, le=30)
    apoe4: Optional[int] = Field(None, ge=0, le=2, description="Alias for apoe4_count")
    apoe4_count: Optional[int] = Field(None, ge=0, le=2)


class ScoresInput(BaseModel):
    model_config = ConfigDict(extra="allow")
    MMSE: Optional[float] = Field(None, ge=0, le=30)
    CDGLOBAL: Optional[float] = Field(None, ge=0, le=3)
    TOTSCORE: Optional[float] = Field(None, ge=0, le=70)
    CDRSB: Optional[float] = Field(None, ge=0, le=18)
    FAQ: Optional[float] = Field(None, ge=0, le=30)
    MOCA: Optional[float] = Field(None, ge=0, le=30)


class BiomarkersPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    ABETA: Optional[float] = None
    TAU: Optional[float] = None
    PTAU: Optional[float] = None
    APVOLUME: Optional[float] = None
    GENOTYPE: Optional[str] = None
    FDG: Optional[float] = None


class MRIVolumesPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    hippocampus: Optional[float] = None
    hippocampus_left: Optional[float] = None
    hippocampus_right: Optional[float] = None
    entorhinal: Optional[float] = None
    entorhinal_left: Optional[float] = None
    entorhinal_right: Optional[float] = None
    ventricles: Optional[float] = None
    whole_brain: Optional[float] = None
    icv: Optional[float] = None


class CreateTwinRequest(BaseModel):
    """POST /twin — supports both patient-builder nested shape and flat legacy shape."""
    model_config = ConfigDict(extra="allow")
    name: Optional[str] = None
    baseline_date: Optional[str] = None   # YYYY-MM-DD

    # Patient-builder nested structures
    demographics: Optional[DemographicsInput] = None
    scores: Optional[ScoresInput] = None
    biomarkers: Optional[BiomarkersPayload] = None
    mri_volumes: Optional[MRIVolumesPayload] = None
    lifestyle: Optional[Dict[str, Any]] = None
    visits: Optional[List[VisitRecord]] = None

    # Flat fields (backward compatibility with Phase 1-3 callers)
    age: Optional[float] = Field(None, ge=18, le=110)
    sex: Optional[Literal[0, 1]] = None
    education_years: Optional[float] = Field(None, ge=0, le=30)
    apoe4_count: Optional[int] = Field(None, ge=0, le=2)
    MMSE: Optional[float] = Field(None, ge=0, le=30)
    CDGLOBAL: Optional[float] = Field(None, ge=0, le=3)
    TOTSCORE: Optional[float] = Field(None, ge=0, le=70)
    CDRSB: Optional[float] = Field(None, ge=0, le=18)
    MOCA: Optional[float] = Field(None, ge=0, le=30)
    FAQ: Optional[float] = Field(None, ge=0, le=30)
    ABETA: Optional[float] = None
    TAU: Optional[float] = None
    PTAU: Optional[float] = None
    initial_visits: Optional[List[VisitRecord]] = None
    is_synthetic: bool = False
    persona_label: Optional[str] = None


class AddVisitRequest(BaseModel):
    """POST /twin/{id}/visit"""
    month: Optional[int] = Field(None, ge=0)
    exam_date: Optional[str] = None   # YYYY-MM-DD
    MMSE: Optional[float] = Field(None, ge=0, le=30)
    CDGLOBAL: Optional[float] = Field(None, ge=0, le=3)
    TOTSCORE: Optional[float] = Field(None, ge=0, le=70)
    DX_obs: Optional[int] = Field(None, ge=1, le=3)
    source: Literal["user", "adni", "synthetic"] = "user"


class AddVisitResponse(BaseModel):
    twin_id: str
    state: TwinState
    prob_diff: Dict[str, float]   # new_prob - old_prob per class


class MRIUploadResponse(BaseModel):
    mri: Dict[str, Any]
    state: TwinState


class ScenarioRequest(BaseModel):
    """POST /twin/{id}/scenario"""
    horizon_months: int
    hypothetical_visit: Optional[Dict[str, Optional[float]]] = None
    biomarkers: Optional[BiomarkersPayload] = None
    mri_volumes: Optional[MRIVolumesPayload] = None
    lifestyle: Optional[Dict[str, Any]] = None
    decline_multiplier: Optional[float] = 1.0


class ScenarioResponse(BaseModel):
    twin_id: str
    horizon_months: int
    baseline: Dict[str, Any]
    scenario: Dict[str, Any]
    delta: Dict[str, Any]
    scope_warnings: List[str]
    forecast_reliability: Optional[Literal["ok", "low"]] = "ok"
    disclaimer: str = (
        "Hypothetical what-if scenario. Projections represent counterfactual simulations "
        "and are NOT clinical predictions. Model uncertainty and biological variability apply."
    )
    method: List[str] = Field(default_factory=list)
    drivers_delta: List[Dict[str, Any]] = Field(default_factory=list)


# ── Explainability Models ──────────────────────────────────────────────────

class InputUsageItem(BaseModel):
    field: str
    label: str
    category: str
    value: Any = None
    entered_value: Any = None
    status: Literal[
        "used_in_stage",
        "used_in_forecast",
        "used_in_lifestyle_modifier",
        "used_in_symptom_screen",
        "used_in_biomarker_ensemble",
        "used_in_mri_fusion",
        "recorded_only",
        "missing",
        "assumed_default"
    ]
    display_status: str
    used_by: List[str] = Field(default_factory=list)
    reason: str


class InputUsageResponse(BaseModel):
    twin_id: str
    items: List[InputUsageItem]
    field_count: int


class TwinExplanationResponse(BaseModel):
    twin_id: str
    stage: Dict[str, Any]
    forecast: Dict[str, Any]
    lifestyle_modifier: Dict[str, Any]
    symptom_screen: Dict[str, Any]
    reliability: Any
    forecast_reliability: Optional[str] = "ok"
    scope_warnings: List[str] = Field(default_factory=list)
    narrative: str


class SimulateRequest(BaseModel):
    """POST /twin/{id}/simulate"""
    decline_multiplier: float = Field(
        ge=0.25, le=3.0,
        description="1.0 = as predicted. 0.5 = half speed decline. 1.5 = 50% faster."
    )


class SimulateResponse(BaseModel):
    twin_id: str
    decline_multiplier: float
    simulated_forecast: Dict[str, ForecastPoint]
    disclaimer: str = (
        "Illustrative simulation only. Trajectory is scaled from the model's "
        "central prediction by the chosen multiplier. This is NOT a clinical "
        "prediction and does not account for individual biological variability, "
        "treatment effects, or model uncertainty beyond the shown bands."
    )


class DemoPatientEntry(BaseModel):
    twin_id: str
    persona_label: str
    is_synthetic: bool
    baseline_stage: Literal["CN", "MCI", "AD"]
    trajectory: str   # e.g. "Stable CN", "CN→MCI", "Fast MCI→AD"
    n_visits: int


class ModelCardMetrics(BaseModel):
    """GET /model-card"""
    stager: Dict[str, Any]
    progression: Dict[str, Any]
    scope: Dict[str, Any]
    limitations: List[str]
    resnet_disclaimer: str
    fusion_disclaimer: str
    data_provenance: Dict[str, Any]
    lifestyle_modifier: Optional[Dict[str, Any]] = None
