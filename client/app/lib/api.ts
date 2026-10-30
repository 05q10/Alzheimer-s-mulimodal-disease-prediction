/**
 * Digital Twin API Client
 * Built to API_CONTRACT.md specs with live model calibration and fallback mocks.
 * Uses NEXT_PUBLIC_API_URL (default: http://127.0.0.1:8000).
 */

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export const USE_MOCK =
  process.env.NEXT_PUBLIC_USE_MOCK === "true" || false;

// -------------------------------------------------------------
// Core TypeScript Interfaces
// -------------------------------------------------------------

export interface StageProbs {
  CN: number;
  MCI: number;
  AD: number;
}

export interface ForecastPoint {
  month?: number;
  date?: string; // Calendar date, e.g. "Apr 2027"
  MMSE_mean?: number | null;
  MMSE_lo?: number | null;
  MMSE_hi?: number | null;
  CDGLOBAL_mean?: number | null;
  CDGLOBAL_lo?: number | null;
  CDGLOBAL_hi?: number | null;
  TOTSCORE_mean?: number | null;
  TOTSCORE_lo?: number | null;
  TOTSCORE_hi?: number | null;
  stage_probs?: StageProbs | null;
  conversion_prob?: number | null;
}

export interface VisitRecord {
  month: number;
  exam_date?: string; // e.g. "2026-10-28"
  MMSE?: number | null;
  CDGLOBAL?: number | null;
  TOTSCORE?: number | null;
  FAQ?: number | null;
  MOCA?: number | null;
  DX_obs?: number | null;
  source?: "user" | "adni" | "synthetic";
  [key: string]: any;
}

export interface ModalityStatus {
  present: boolean;
  latest?: Record<string, any> | null;
}

export interface ShapDriver {
  feature: string;
  value: any;
  shap: number;
}

export interface TwinState {
  stage_probs: StageProbs;
  stage: "CN" | "MCI" | "AD";
  confidence: number;
  tier_used: "T1" | "T2" | "T3" | "T4";
  forecast: Record<string, ForecastPoint>;
  conversion_risk: {
    "12m": number;
    "24m": number;
    [key: string]: number;
  };
  rate_of_decline?: {
    MMSE_per_year: number;
    vs_cohort_percentile: number;
  } | null;
  shap_drivers?: ShapDriver[] | null;
  fusion?: {
    adni_probs: StageProbs;
    mri_probs: StageProbs;
    weight: number;
    final_probs: StageProbs;
  } | null;
}

export interface HistoryEvent {
  timestamp: string;
  recorded_at?: string;
  exam_date?: string;
  month?: number;
  stage_probs: StageProbs;
  event: string;
  prob_diff?: StageProbs | Record<string, number> | null;
  inputs?: Record<string, any>;
  drivers?: Array<{ title: string; detail: string; impact?: "high" | "moderate" | "protective" }>;
  [key: string]: any;
}

export interface TwinFull {
  twin_id: string;
  id?: string;
  name?: string | null;
  patient_name?: string | null;
  is_synthetic: boolean;
  persona_label?: string | null;
  baseline_date?: string; // "YYYY-MM-DD"
  demographics: {
    age?: number;
    sex?: number; // 0=Male, 1=Female
    education_years?: number;
    apoe4_count?: number;
    [key: string]: any;
  };
  modalities: {
    basic_scores?: ModalityStatus;
    biomarkers?: ModalityStatus;
    mri_volumes?: ModalityStatus;
    mri_image?: ModalityStatus;
    ocr_report?: ModalityStatus;
    [key: string]: ModalityStatus | undefined;
  };
  visits: VisitRecord[];
  state: TwinState;
  history: HistoryEvent[];
  [key: string]: any;
}

export interface CreateTwinRequest {
  age?: number;
  sex?: 0 | 1;
  education_years?: number;
  apoe4_count?: number;
  baseline_date?: string;
  MMSE?: number | null;
  CDGLOBAL?: number | null;
  TOTSCORE?: number | null;
  CDRSB?: number | null;
  MOCA?: number | null;
  FAQ?: number | null;
  ABETA?: number | null;
  TAU?: number | null;
  PTAU?: number | null;
  initial_visits?: VisitRecord[] | null;
  is_synthetic?: boolean;
  persona_label?: string | null;

  // New structured payload support
  name?: string;
  demographics?: {
    age: number;
    sex: 0 | 1;
    education_years: number;
    apoe4?: number;
    baseline_date?: string;
  };
  scores?: {
    MMSE: number;
    CDGLOBAL?: number;
    TOTSCORE?: number;
    FAQ?: number;
    MOCA?: number;
  };
  biomarkers?: {
    ABETA?: number;
    TAU?: number;
    PTAU?: number;
    APVOLUME?: number;
  };
  mri_volumes?: {
    hippocampus?: number | null;
    entorhinal?: number | null;
    ventricles?: number | null;
    whole_brain?: number | null;
    wholeBrain?: number | null;
    icv?: number | null;
  };
  lifestyle?: Record<string, any>;
  visits?: VisitRecord[];
}

export interface ScenarioRequest {
  horizon_months: 6 | 12 | 24;
  hypothetical_visit?: {
    MMSE?: number;
    CDGLOBAL?: number;
    TOTSCORE?: number;
    FAQ?: number;
    MOCA?: number;
  };
  biomarkers?: {
    ABETA?: number;
    TAU?: number;
    PTAU?: number;
  };
  mri_volumes?: {
    hippocampus?: number;
    ventricles?: number;
  };
  lifestyle?: {
    alcohol?: number; // units/week
    sleepQuality?: number; // 0-10
    physicalActivity?: number; // hrs/week
    dietQuality?: number; // 0-10
    smoking?: number; // 0 or 1
    hypertension?: number; // 0 or 1
    diabetes?: number; // 0 or 1
    depression?: number; // 0 or 1
    headInjury?: number; // 0 or 1
    cardiovascularDisease?: number; // 0 or 1
    [key: string]: any;
  };
  decline_multiplier?: number;
}

export interface DeltaMetric {
  value: number;
  direction: "worse" | "better" | "same";
  pct_str?: string;
  sign_str?: string;
}

export interface HeadlineMetric {
  metric_name: string;
  baseline_value: number;
  scenario_value: number;
  delta_value: number;
  direction: "worse" | "better" | "same";
  formatted_baseline: string;
  formatted_scenario: string;
  formatted_delta: string;
}

export interface ScenarioResponse {
  twin_id?: string;
  horizon_months?: number;
  method: string[];
  drivers_delta: Array<{
    factor: string;
    label: string;
    delta: string | number;
    impact: string;
    direction: "worse" | "better" | "same";
  }>;
  forecast_reliability: "high" | "moderate" | "low" | "ok";
  scope_warnings: string[];
  disclaimer: string;
  baseline: {
    stage_probs: StageProbs;
    forecast: Record<string, ForecastPoint>;
    conversion_risk: { "12m": number; "24m": number; [key: string]: number };
    p_ad?: number;
    progression_risk?: number;
  };
  scenario: {
    stage_probs: StageProbs;
    forecast: Record<string, ForecastPoint>;
    conversion_risk: { "12m": number; "24m": number; [key: string]: number };
    p_ad?: number;
    progression_risk?: number;
  };
  delta: {
    stage_probs: {
      CN: number | DeltaMetric;
      MCI: number | DeltaMetric;
      AD: number | DeltaMetric;
      [key: string]: any;
    };
    conversion_risk: {
      "12m": number | DeltaMetric;
      "24m": number | DeltaMetric;
      [key: string]: any;
    };
    p_ad?: DeltaMetric;
    progression_risk?: DeltaMetric;
    direction?: "worse" | "better" | "same";
    [key: string]: any;
  };
  headline_metrics?: {
    p_ad: HeadlineMetric;
    progression_risk: HeadlineMetric;
  };
}

export interface DemoCaseVisit {
  visit_number: number;
  month: number;
  exam_date: string;
  MMSE: number;
  CDGLOBAL: number;
  TOTSCORE: number;
  FAQ?: number;
  MOCA?: number;
  functionalAssessment?: number;
  adl?: number;
  symptoms?: Record<string, boolean>;
  biomarkers?: {
    ABETA?: number;
    TAU?: number;
    PTAU?: number;
    APVOLUME?: number;
  };
  mri_volumes?: {
    hippocampus?: number;
    entorhinal?: number;
    ventricles?: number;
    whole_brain?: number;
    icv?: number;
  };
  lifestyle?: {
    alcohol?: number;
    sleepQuality?: number;
    physicalActivity?: number;
    dietQuality?: number;
    smoking?: number;
  };
  medical?: {
    hypertension?: number;
    diabetes?: number;
    depression?: number;
    headInjury?: number;
    cardiovascularDisease?: number;
  };
  stage: "CN" | "MCI" | "AD";
  stage_probs: StageProbs;
  notes?: string;
  mri_stale?: boolean;
}

export interface DemoCase {
  id: string; // "case-1", "case-2", "case-3"
  title: string;
  patient_name: string;
  age: number;
  sex: 0 | 1;
  education_years: number;
  apoe4_count: number;
  baseline_stage: "CN" | "MCI" | "AD";
  story: string;
  expected_checklist: string[];
  baseline_date: string;
  visits: DemoCaseVisit[];
}

export interface AddVisitRequest {
  month: number;
  exam_date?: string;
  MMSE?: number | null;
  CDGLOBAL?: number | null;
  TOTSCORE?: number | null;
  FAQ?: number | null;
  MOCA?: number | null;
  DX_obs?: number | null;
  source?: "user" | "adni" | "synthetic";
}

export interface AddVisitResponse {
  twin_id: string;
  state: TwinState;
  prob_diff: StageProbs;
}

export interface SimulateRequest {
  decline_multiplier: number;
}

export interface SimulateResponse {
  twin_id: string;
  decline_multiplier: number;
  simulated_forecast: Record<string, ForecastPoint>;
  disclaimer: string;
}

export interface MRIUploadResponse {
  mri: {
    predicted_class: string;
    probs: Record<string, number>;
    mapped_probs: { CN: number; MCI: number; AD: number };
  };
  state: TwinState;
}

// -------------------------------------------------------------
// Explainability Interfaces (/twin/{id}/explain)
// -------------------------------------------------------------

export interface ExplainDriver {
  feature: string;
  label: string;
  modality: "cognitive" | "biomarkers" | "mri" | "demographics" | "lifestyle";
  value: string | number;
  direction: "CN" | "AD"; // Left = toward CN (green), Right = toward AD (rose/purple)
  weight: number; // 0 to 1
  description: string;
}

export interface LeaveOneOutEffect {
  modality: string;
  label: string;
  without_stage: "CN" | "MCI" | "AD";
  without_probs: StageProbs;
  summary: string; // e.g. "Without MRI: AD 41% -> 33%"
}

export interface ExplainResponse {
  twin_id: string;
  stage: {
    primary: "CN" | "MCI" | "AD";
    probs: StageProbs;
    drivers: ExplainDriver[];
    leave_one_out: LeaveOneOutEffect[];
  };
  forecast: {
    drivers: Array<{
      feature: string;
      label: string;
      effect: "accelerating" | "decelerating";
      impact: string;
    }>;
    reliability: "high" | "moderate" | "low";
    reliability_reasons?: string[];
  };
  lifestyle_modifier: {
    multipliers: Array<{
      factor: string;
      label: string;
      value: string | number;
      multiplier: number; // e.g. 1.07
      description: string;
    }>;
    total_multiplier: number;
    caption: string; // "Literature-informed estimate, applied to the forecast and conversion risk only. It does not change the current stage and was not learned from ADNI."
  };
  symptom_screen: {
    risk_level: string; // "Low Risk", "Moderate Risk", "High Risk"
    risk_score: number;
    drivers: Array<{
      factor: string;
      label: string;
      status: string;
      impact: "protective" | "risk";
    }>;
    caption: string; // "A separate screening model from a different population; shown for reference and not part of the ADNI stage"
  };
  narrative: string;
  reliability: "high" | "moderate" | "low";
}

// -------------------------------------------------------------
// Input Usage Interfaces (/twin/{id}/input-usage)
// -------------------------------------------------------------

export interface InputUsageItem {
  field: string;
  category:
    | "Demographics"
    | "Lifestyle"
    | "Medical History"
    | "Clinical Measurements"
    | "Cognitive Symptoms"
    | "Cognitive Assessments"
    | "Advanced Scores"
    | "Biomarkers"
    | "Volumetric MRI"
    | "Uploads";
  entered_value: any;
  status:
    | "Used in stage"
    | "Used in forecast"
    | "Lifestyle adjustment"
    | "Symptom screen"
    | "Biomarker ensemble"
    | "MRI fusion"
    | "Recorded only"
    | "Missing"
    | "Assumed default";
  reason: string;
}

export interface InputUsageResponse {
  twin_id: string;
  items: InputUsageItem[];
}

export interface DemoPatientEntry {
  twin_id: string;
  persona_label: string;
  is_synthetic: boolean;
  baseline_stage: "CN" | "MCI" | "AD";
  trajectory: string;
  n_visits: number;
}

export interface ModelScope {
  age_range: [number, number];
  horizons_months: number[];
  score_ranges: {
    MMSE: [number, number];
    CDGLOBAL: [number, number];
    TOTSCORE: [number, number];
  };
  training_cohort: string;
}

export interface ModelCardMetrics {
  stager: Record<string, any>;
  progression: Record<string, any>;
  limitations: string[];
  resnet_disclaimer: string;
  fusion_disclaimer: string;
  data_provenance: Record<string, any>;
  scope?: ModelScope;
}

// -------------------------------------------------------------
// Fallback Generators & Mock Data
// -------------------------------------------------------------

export const FALLBACK_DEMO_PATIENTS: DemoPatientEntry[] = [
  {
    twin_id: "demo-stable-cn",
    persona_label: "Margaret Chen",
    is_synthetic: false,
    baseline_stage: "CN",
    trajectory: "Stable CN over 24m (MMSE 29 → 29)",
    n_visits: 5,
  },
  {
    twin_id: "demo-cn-mci",
    persona_label: "Robert Davies",
    is_synthetic: false,
    baseline_stage: "CN",
    trajectory: "Late CN→MCI Converter (MMSE 28 → 24)",
    n_visits: 4,
  },
  {
    twin_id: "demo-mci-stable",
    persona_label: "Eleanor Vance",
    is_synthetic: false,
    baseline_stage: "MCI",
    trajectory: "Amnestic MCI Plateau (MMSE 26 → 25)",
    n_visits: 6,
  },
  {
    twin_id: "demo-mci-ad-fast",
    persona_label: "Arthur Pendelton",
    is_synthetic: false,
    baseline_stage: "MCI",
    trajectory: "Fast Decliner MCI→AD (MMSE 24 → 17)",
    n_visits: 4,
  },
  {
    twin_id: "demo-ad-sparse",
    persona_label: "Beatrice Miller",
    is_synthetic: false,
    baseline_stage: "AD",
    trajectory: "Moderate AD Progression (MMSE 19 → 12)",
    n_visits: 3,
  },
  {
    twin_id: "demo-t1-only",
    persona_label: "Harold Jenkins",
    is_synthetic: true,
    baseline_stage: "MCI",
    trajectory: "Minimal Data Profile (T1 Screening Only)",
    n_visits: 1,
  },
];

export const FALLBACK_DEMO_CASES: DemoCase[] = [
  {
    id: "case-1",
    title: "Case 1: Margaret Sterling (MCI → AD Rapid Progression)",
    patient_name: "Margaret Sterling",
    age: 74,
    sex: 1, // Female
    education_years: 16,
    apoe4_count: 2, // ε4/ε4 homozygous
    baseline_stage: "MCI",
    baseline_date: "2025-10-15",
    story: "74-year-old retired architect presenting with amnestic MCI. Carries homozygous APOE ε4/ε4 alleles and abnormal CSF biomarker profile. Over 12 months, exhibits progressive cognitive and functional decline culminating in Alzheimer's Dementia.",
    expected_checklist: [
      "Baseline: Classified as MCI (P(MCI) = 64%, P(AD) = 28%) with high 12m conversion risk (42%).",
      "Month 6: MMSE drops from 24 to 21, CDR rises from 0.5 to 1.0, and P(AD) climbs to 45%.",
      "Month 12: MMSE falls to 17, ADAS11 worsens to 29.0; model converts stage to AD (P(AD) = 81%).",
      "Longitudinal forecast shows sharp downward trajectory with low uncertainty band."
    ],
    visits: [
      {
        visit_number: 0,
        month: 0,
        exam_date: "2025-10-15",
        MMSE: 24,
        CDGLOBAL: 0.5,
        TOTSCORE: 16.0,
        FAQ: 5,
        MOCA: 21,
        functionalAssessment: 7,
        adl: 8,
        symptoms: {
          memoryComplaints: true,
          confusion: true,
          personalityChanges: false,
          difficultyCompletingTasks: true,
          forgetfulness: true,
          behavioralProblems: false,
          disorientation: false,
        },
        biomarkers: {
          ABETA: 680,
          TAU: 520,
          PTAU: 54,
          APVOLUME: 2800,
        },
        mri_volumes: {
          hippocampus: 3100,
          entorhinal: 1200,
          ventricles: 42000,
          whole_brain: 980000,
          icv: 1420000,
        },
        lifestyle: {
          alcohol: 4,
          sleepQuality: 6,
          physicalActivity: 2,
          dietQuality: 6,
          smoking: 0,
        },
        medical: {
          hypertension: 1,
          diabetes: 0,
          depression: 0,
          headInjury: 0,
          cardiovascularDisease: 0,
        },
        stage: "MCI",
        stage_probs: { CN: 0.08, MCI: 0.64, AD: 0.28 },
        notes: "Baseline assessment: Amnestic MCI with CSF tauopathy and medial temporal atrophy.",
      },
      {
        visit_number: 1,
        month: 6,
        exam_date: "2026-04-15",
        MMSE: 21,
        CDGLOBAL: 1.0,
        TOTSCORE: 22.0,
        FAQ: 9,
        MOCA: 18,
        functionalAssessment: 5,
        adl: 6,
        symptoms: {
          memoryComplaints: true,
          confusion: true,
          personalityChanges: true,
          difficultyCompletingTasks: true,
          forgetfulness: true,
          behavioralProblems: false,
          disorientation: true,
        },
        biomarkers: {
          ABETA: 640,
          TAU: 560,
          PTAU: 58,
          APVOLUME: 2750,
        },
        mri_volumes: {
          hippocampus: 2950,
          entorhinal: 1100,
          ventricles: 45000,
          whole_brain: 965000,
          icv: 1420000,
        },
        lifestyle: {
          alcohol: 3,
          sleepQuality: 5,
          physicalActivity: 1.5,
          dietQuality: 5,
          smoking: 0,
        },
        medical: {
          hypertension: 1,
          diabetes: 0,
          depression: 1,
          headInjury: 0,
          cardiovascularDisease: 0,
        },
        stage: "MCI",
        stage_probs: { CN: 0.03, MCI: 0.52, AD: 0.45 },
        notes: "6-month follow-up: Accelerated functional loss in IADLs; borderline transition zone.",
        mri_stale: false,
      },
      {
        visit_number: 2,
        month: 12,
        exam_date: "2026-10-15",
        MMSE: 17,
        CDGLOBAL: 1.0,
        TOTSCORE: 29.0,
        FAQ: 14,
        MOCA: 14,
        functionalAssessment: 4,
        adl: 5,
        symptoms: {
          memoryComplaints: true,
          confusion: true,
          personalityChanges: true,
          difficultyCompletingTasks: true,
          forgetfulness: true,
          behavioralProblems: true,
          disorientation: true,
        },
        biomarkers: {
          ABETA: 610,
          TAU: 610,
          PTAU: 64,
          APVOLUME: 2700,
        },
        mri_volumes: {
          hippocampus: 2800,
          entorhinal: 990,
          ventricles: 49000,
          whole_brain: 950000,
          icv: 1420000,
        },
        lifestyle: {
          alcohol: 2,
          sleepQuality: 4,
          physicalActivity: 1,
          dietQuality: 5,
          smoking: 0,
        },
        medical: {
          hypertension: 1,
          diabetes: 0,
          depression: 1,
          headInjury: 0,
          cardiovascularDisease: 0,
        },
        stage: "AD",
        stage_probs: { CN: 0.01, MCI: 0.18, AD: 0.81 },
        notes: "12-month follow-up: Definite conversion to Mild Alzheimer's Dementia.",
        mri_stale: false,
      },
    ],
  },
  {
    id: "case-2",
    title: "Case 2: Arthur Pendelton (CN → MCI Insidious Progression)",
    patient_name: "Arthur Pendelton",
    age: 69,
    sex: 0, // Male
    education_years: 18,
    apoe4_count: 1, // ε3/ε4 heterozygous
    baseline_stage: "CN",
    baseline_date: "2024-10-10",
    story: "69-year-old retired civil engineer with high baseline cognitive reserve. Enrolled with normal cognition and subtle subjective memory complaints. Displays slow, subtle transition to amnestic MCI over a 24-month horizon.",
    expected_checklist: [
      "Baseline: Solid Cognitively Normal state (P(CN) = 88%, P(MCI) = 10%) with low 12m conversion risk (6%).",
      "Month 12: Preserved functional autonomy with subclinical drift in MMSE (29 → 27) and ADAS11 (6.5 → 9.0).",
      "Month 24: Crosses CDR-Global 0.5 threshold (MMSE 25, MoCA 23), triggering re-staging to MCI (P(MCI) = 65%).",
      "Model attribution indicates cognitive tests and entorhinal cortex thinning as key drivers."
    ],
    visits: [
      {
        visit_number: 0,
        month: 0,
        exam_date: "2024-10-10",
        MMSE: 29,
        CDGLOBAL: 0.0,
        TOTSCORE: 6.5,
        FAQ: 0,
        MOCA: 28,
        functionalAssessment: 10,
        adl: 10,
        symptoms: {
          memoryComplaints: false,
          confusion: false,
          personalityChanges: false,
          difficultyCompletingTasks: false,
          forgetfulness: true,
          behavioralProblems: false,
          disorientation: false,
        },
        biomarkers: {
          ABETA: 1120,
          TAU: 320,
          PTAU: 24,
          APVOLUME: 3400,
        },
        mri_volumes: {
          hippocampus: 4100,
          entorhinal: 1850,
          ventricles: 28000,
          whole_brain: 1120000,
          icv: 1560000,
        },
        lifestyle: {
          alcohol: 3,
          sleepQuality: 8,
          physicalActivity: 5,
          dietQuality: 8,
          smoking: 0,
        },
        medical: {
          hypertension: 0,
          diabetes: 0,
          depression: 0,
          headInjury: 0,
          cardiovascularDisease: 0,
        },
        stage: "CN",
        stage_probs: { CN: 0.88, MCI: 0.10, AD: 0.02 },
        notes: "Baseline visit: High cognitive reserve, biomarker negative.",
      },
      {
        visit_number: 1,
        month: 12,
        exam_date: "2025-10-10",
        MMSE: 27,
        CDGLOBAL: 0.0,
        TOTSCORE: 9.0,
        FAQ: 1,
        MOCA: 26,
        functionalAssessment: 9,
        adl: 10,
        symptoms: {
          memoryComplaints: true,
          confusion: false,
          personalityChanges: false,
          difficultyCompletingTasks: false,
          forgetfulness: true,
          behavioralProblems: false,
          disorientation: false,
        },
        biomarkers: {
          ABETA: 990,
          TAU: 360,
          PTAU: 28,
          APVOLUME: 3350,
        },
        mri_volumes: {
          hippocampus: 3950,
          entorhinal: 1720,
          ventricles: 30500,
          whole_brain: 1105000,
          icv: 1560000,
        },
        lifestyle: {
          alcohol: 3,
          sleepQuality: 7,
          physicalActivity: 4.5,
          dietQuality: 8,
          smoking: 0,
        },
        medical: {
          hypertension: 0,
          diabetes: 0,
          depression: 0,
          headInjury: 0,
          cardiovascularDisease: 0,
        },
        stage: "CN",
        stage_probs: { CN: 0.72, MCI: 0.24, AD: 0.04 },
        notes: "12-month follow-up: Subclinical cognitive drift, remains clinically CN.",
        mri_stale: false,
      },
      {
        visit_number: 2,
        month: 24,
        exam_date: "2026-10-10",
        MMSE: 25,
        CDGLOBAL: 0.5,
        TOTSCORE: 13.5,
        FAQ: 4,
        MOCA: 23,
        functionalAssessment: 8,
        adl: 9,
        symptoms: {
          memoryComplaints: true,
          confusion: true,
          personalityChanges: false,
          difficultyCompletingTasks: true,
          forgetfulness: true,
          behavioralProblems: false,
          disorientation: false,
        },
        biomarkers: {
          ABETA: 890,
          TAU: 420,
          PTAU: 36,
          APVOLUME: 3280,
        },
        mri_volumes: {
          hippocampus: 3750,
          entorhinal: 1580,
          ventricles: 34000,
          whole_brain: 1085000,
          icv: 1560000,
        },
        lifestyle: {
          alcohol: 2.5,
          sleepQuality: 7,
          physicalActivity: 4,
          dietQuality: 7.5,
          smoking: 0,
        },
        medical: {
          hypertension: 0,
          diabetes: 0,
          depression: 0,
          headInjury: 0,
          cardiovascularDisease: 0,
        },
        stage: "MCI",
        stage_probs: { CN: 0.24, MCI: 0.65, AD: 0.11 },
        notes: "24-month follow-up: Transition to mild cognitive impairment verified by CDR-Global 0.5.",
        mri_stale: false,
      },
    ],
  },
  {
    id: "case-3",
    title: "Case 3: Harold Finch (AD with Intensive Lifestyle Improvement)",
    patient_name: "Harold Finch",
    age: 78,
    sex: 0, // Male
    education_years: 14,
    apoe4_count: 1,
    baseline_stage: "AD",
    baseline_date: "2026-01-20",
    story: "78-year-old with established Mild Alzheimer's Dementia and multiple high-risk modifiable lifestyle factors (high alcohol consumption, severe sleep fragmentation, sedentary). Enrolled in an intensive multidomain lifestyle intervention targeting vascular risk, sleep, and physical exercise.",
    expected_checklist: [
      "Baseline: Diagnosed with Mild AD (P(AD) = 85%) and 58% 12-month progression risk under adverse lifestyle (multiplier 1.25x).",
      "Scenario Lab: Reducing alcohol from 18 to 0 units/wk and increasing physical activity and sleep cuts progression risk to 38%.",
      "Direction indicators in Scenario Lab show green 'better' status and negative progression deltas.",
      "Month 6 Follow-Up: Cognitive decline velocity stabilizes with MMSE preserved at 19."
    ],
    visits: [
      {
        visit_number: 0,
        month: 0,
        exam_date: "2026-01-20",
        MMSE: 19,
        CDGLOBAL: 1.0,
        TOTSCORE: 24.0,
        FAQ: 12,
        MOCA: 16,
        functionalAssessment: 4,
        adl: 5,
        symptoms: {
          memoryComplaints: true,
          confusion: true,
          personalityChanges: true,
          difficultyCompletingTasks: true,
          forgetfulness: true,
          behavioralProblems: true,
          disorientation: true,
        },
        biomarkers: {
          ABETA: 550,
          TAU: 680,
          PTAU: 68,
          APVOLUME: 2500,
        },
        mri_volumes: {
          hippocampus: 2650,
          entorhinal: 920,
          ventricles: 52000,
          whole_brain: 920000,
          icv: 1480000,
        },
        lifestyle: {
          alcohol: 18, // High risk
          sleepQuality: 4, // Fragmented
          physicalActivity: 1, // Sedentary
          dietQuality: 4, // Inflammatory
          smoking: 1,
        },
        medical: {
          hypertension: 1,
          diabetes: 1,
          depression: 1,
          headInjury: 0,
          cardiovascularDisease: 1,
        },
        stage: "AD",
        stage_probs: { CN: 0.01, MCI: 0.14, AD: 0.85 },
        notes: "Baseline: Established Mild AD with adverse cardiovascular and lifestyle modifiers.",
      },
      {
        visit_number: 1,
        month: 6,
        exam_date: "2026-07-20",
        MMSE: 19,
        CDGLOBAL: 1.0,
        TOTSCORE: 23.0,
        FAQ: 11,
        MOCA: 17,
        functionalAssessment: 5,
        adl: 6,
        symptoms: {
          memoryComplaints: true,
          confusion: true,
          personalityChanges: false,
          difficultyCompletingTasks: true,
          forgetfulness: true,
          behavioralProblems: false,
          disorientation: false,
        },
        biomarkers: {
          ABETA: 560,
          TAU: 670,
          PTAU: 66,
          APVOLUME: 2520,
        },
        mri_volumes: {
          hippocampus: 2630,
          entorhinal: 915,
          ventricles: 52400,
          whole_brain: 918000,
          icv: 1480000,
        },
        lifestyle: {
          alcohol: 0, // Complete cessation
          sleepQuality: 8, // Structured CPAP/sleep hygiene
          physicalActivity: 5, // Daily aerobic routine
          dietQuality: 8, // Mediterranean-DASH
          smoking: 0, // Quit
        },
        medical: {
          hypertension: 1,
          diabetes: 1,
          depression: 0,
          headInjury: 0,
          cardiovascularDisease: 1,
        },
        stage: "AD",
        stage_probs: { CN: 0.02, MCI: 0.22, AD: 0.76 },
        notes: "6-month follow-up post-lifestyle intervention: Cognitive decline halted; functional stabilization.",
        mri_stale: false,
      },
    ],
  },
];

export function computeCalendarDate(baseDateStr: string | undefined, monthOffset: number): string {
  try {
    const base = baseDateStr ? new Date(baseDateStr) : new Date();
    const d = new Date(base.getTime());
    d.setMonth(d.getMonth() + monthOffset);
    return d.toLocaleDateString("en-US", { month: "short", year: "numeric" });
  } catch {
    return `Month ${monthOffset}`;
  }
}

export function generateFallbackTwin(
  twinId: string,
  personaLabel = "Demo Patient",
  isSynthetic = true
): TwinFull {
  const now = new Date().toISOString();
  const baselineDate = new Date().toISOString().split("T")[0];

  return {
    twin_id: twinId,
    name: personaLabel,
    is_synthetic: isSynthetic,
    persona_label: personaLabel,
    baseline_date: baselineDate,
    demographics: {
      age: 74,
      sex: 1, // Female
      education_years: 16,
      apoe4_count: 1,
      baseline_date: baselineDate,
    },
    modalities: {
      basic_scores: {
        present: true,
        latest: { MMSE: 26, CDGLOBAL: 0.5, TOTSCORE: 12.0 },
      },
      biomarkers: {
        present: true,
        latest: { ABETA: 980, TAU: 432, PTAU: 38 },
      },
      mri_volumes: {
        present: true,
        latest: { hippocampus: 3400, entorhinal: 1350, ventricles: 35000 },
      },
      mri_image: { present: false },
      ocr_report: { present: false },
    },
    visits: [
      {
        month: 0,
        exam_date: baselineDate,
        MMSE: 26.0,
        CDGLOBAL: 0.5,
        TOTSCORE: 12.0,
        FAQ: 3.0,
        MOCA: 23.0,
        source: "user",
      },
    ],
    state: {
      stage_probs: { CN: 0.18, MCI: 0.62, AD: 0.2 },
      stage: "MCI",
      confidence: 0.62,
      tier_used: "T3",
      forecast: {
        "6": {
          month: 6,
          date: computeCalendarDate(baselineDate, 6),
          MMSE_mean: 25.1,
          MMSE_lo: 22.8,
          MMSE_hi: 27.2,
          CDGLOBAL_mean: 0.58,
          CDGLOBAL_lo: 0.5,
          CDGLOBAL_hi: 0.8,
          TOTSCORE_mean: 13.5,
          TOTSCORE_lo: 11.2,
          TOTSCORE_hi: 15.8,
          conversion_prob: 0.22,
        },
        "12": {
          month: 12,
          date: computeCalendarDate(baselineDate, 12),
          MMSE_mean: 23.9,
          MMSE_lo: 20.4,
          MMSE_hi: 26.8,
          CDGLOBAL_mean: 0.72,
          CDGLOBAL_lo: 0.5,
          CDGLOBAL_hi: 1.1,
          TOTSCORE_mean: 15.4,
          TOTSCORE_lo: 12.0,
          TOTSCORE_hi: 18.5,
          conversion_prob: 0.38,
        },
        "24": {
          month: 24,
          date: computeCalendarDate(baselineDate, 24),
          MMSE_mean: 21.2,
          MMSE_lo: 16.5,
          MMSE_hi: 25.5,
          CDGLOBAL_mean: 1.05,
          CDGLOBAL_lo: 0.5,
          CDGLOBAL_hi: 1.6,
          TOTSCORE_mean: 19.8,
          TOTSCORE_lo: 14.5,
          TOTSCORE_hi: 24.2,
          conversion_prob: 0.58,
        },
      },
      conversion_risk: {
        "12m": 0.38,
        "24m": 0.58,
      },
      rate_of_decline: {
        MMSE_per_year: -1.4,
        vs_cohort_percentile: 58,
      },
      shap_drivers: [
        { feature: "MMSE", value: 26, shap: 0.24 },
        { feature: "CDGLOBAL", value: 0.5, shap: 0.31 },
        { feature: "ABETA", value: 980, shap: -0.15 },
        { feature: "PTAU", value: 38, shap: 0.18 },
      ],
    },
    history: [
      {
        timestamp: now,
        recorded_at: now,
        exam_date: baselineDate,
        stage_probs: { CN: 0.18, MCI: 0.62, AD: 0.2 },
        event: "twin_created",
      },
    ],
  };
}

export const FALLBACK_MODEL_CARD: ModelCardMetrics = {
  stager: {
    description: "Multi-Tier Multimodal XGBoost Stager calibrated with Temperature Scaling on ADNI test split.",
    T1_accuracy: 0.72,
    T2_accuracy: 0.81,
    T3_accuracy: 0.86,
    T4_accuracy: 0.89,
    ece: 0.024,
  },
  progression: {
    description: "LightGBM Gradient Boosted Regressors for multi-horizon clinical trajectory projection.",
    MAE_12m_MMSE: 1.465,
    MAE_24m_MMSE: 1.56,
    AUROC_12m_conversion: 0.7615,
    AUROC_24m_conversion: 0.8111,
  },
  limitations: [
    "ADNI demographic bias: Volunteers are predominantly Caucasian with higher education (>15 yrs).",
    "Observational cohort: Model trajectories reflect unmanipulated observational progress, not drug trials.",
    "ResNet MRI model was trained on Kaggle data; accuracy on ADNI is not validated.",
    "Simulation velocity scalers are mathematical heuristics and not validated pharmacodynamic models.",
  ],
  resnet_disclaimer:
    "The ResNet-50 MRI model was trained on a non-ADNI Kaggle Alzheimer's dataset. Accuracy on ADNI data is unknown.",
  fusion_disclaimer:
    "The ResNet MRI model was trained on a non-ADNI Kaggle dataset. The 80/20 fusion is for demonstration purposes only.",
  data_provenance: {
    primary_source: "ADNI (Alzheimer's Disease Neuroimaging Initiative)",
    ADNI_subjects: 3034,
    split_seed: 42,
  },
  scope: {
    age_range: [55, 90],
    horizons_months: [6, 12, 24],
    score_ranges: {
      MMSE: [0, 30],
      CDGLOBAL: [0, 3],
      TOTSCORE: [0, 70],
    },
    training_cohort: "ADNI 1/GO/2/3 Longitudinal Academic Cohort",
  },
};

// -------------------------------------------------------------
// Core API Request Helper
// -------------------------------------------------------------

async function request<T>(
  endpoint: string,
  options?: RequestInit,
  fallback?: () => T
): Promise<T> {
  if (USE_MOCK && fallback) {
    return fallback();
  }

  const url = `${API_BASE_URL}${endpoint}`;
  try {
    const res = await fetch(url, {
      cache: "no-store", // Always fetch fresh data per requirement 7
      ...options,
      headers: {
        Accept: "application/json",
        ...(options?.headers || {}),
      },
    });

    if (!res.ok) {
      const errText = await res.text();
      console.warn(`[API] ${res.status} on ${endpoint}: ${errText}`);
      if (fallback) return fallback();
      throw new Error(`API ${res.status}: ${errText || res.statusText}`);
    }

    return (await res.json()) as T;
  } catch (error: any) {
    console.warn(`[API Error] Request failed for ${endpoint}:`, error);
    if (fallback) return fallback();
    throw error;
  }
}

// -------------------------------------------------------------
// API Client Methods
// -------------------------------------------------------------

export const twinApi = {
  /**
   * 1. POST /twin - Create a new digital twin
   */
  async createTwin(data: CreateTwinRequest): Promise<TwinFull> {
    const age = data.age ?? data.demographics?.age ?? 74;
    const sex = (data.sex ?? data.demographics?.sex ?? 1) as 0 | 1;
    const education_years =
      data.education_years ?? data.demographics?.education_years ?? 16;
    const apoe4_count = data.apoe4_count ?? data.demographics?.apoe4 ?? 1;
    const baseline_date = data.baseline_date ?? data.demographics?.baseline_date ?? new Date().toISOString().split("T")[0];
    const MMSE = data.MMSE ?? data.scores?.MMSE ?? 26;
    const CDGLOBAL = data.CDGLOBAL ?? data.scores?.CDGLOBAL ?? 0.5;
    const TOTSCORE = data.TOTSCORE ?? data.scores?.TOTSCORE ?? 12.0;
    const CDRSB = data.CDRSB ?? null;
    const MOCA = data.MOCA ?? data.scores?.MOCA ?? null;
    const FAQ = data.FAQ ?? data.scores?.FAQ ?? null;
    const ABETA = data.ABETA ?? data.biomarkers?.ABETA ?? null;
    const TAU = data.TAU ?? data.biomarkers?.TAU ?? null;
    const PTAU = data.PTAU ?? data.biomarkers?.PTAU ?? null;

    const payload = {
      // Flat fields
      age,
      sex,
      education_years,
      apoe4_count,
      baseline_date,
      MMSE,
      CDGLOBAL,
      TOTSCORE,
      CDRSB,
      MOCA,
      FAQ,
      ABETA,
      TAU,
      PTAU,
      is_synthetic: data.is_synthetic ?? false,
      persona_label: data.persona_label || data.name || "Patient",
      // Nested fields
      name: data.name || data.persona_label || "Patient",
      demographics: { age, sex, education_years, apoe4: apoe4_count, baseline_date },
      scores: { MMSE, CDGLOBAL, TOTSCORE, FAQ, MOCA },
      biomarkers: data.biomarkers || { ABETA, TAU, PTAU },
      mri_volumes: data.mri_volumes,
      lifestyle: data.lifestyle,
      visits: data.visits || [
        {
          month: 0,
          exam_date: baseline_date,
          MMSE,
          CDGLOBAL,
          TOTSCORE,
          FAQ,
          MOCA,
          source: "user" as const,
        },
      ],
    };

    return request<TwinFull>(
      "/twin",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      },
      () => {
        const id = "twin-" + Math.random().toString(36).substring(2, 9);
        const fb = generateFallbackTwin(
          id,
          data.persona_label || data.name || "Created Patient",
          data.is_synthetic ?? false
        );
        fb.baseline_date = baseline_date;
        fb.demographics = { age, sex, education_years, apoe4_count, baseline_date };
        if (fb.visits[0]) {
          fb.visits[0].exam_date = baseline_date;
          fb.visits[0].MMSE = MMSE;
          fb.visits[0].CDGLOBAL = CDGLOBAL;
          fb.visits[0].TOTSCORE = TOTSCORE;
        }
        return fb;
      }
    );
  },

  /**
   * 2. GET /twin/{twin_id} - Fetch full twin state
   */
  async getTwin(twinId: string): Promise<TwinFull> {
    return request<TwinFull>(
      `/twin/${encodeURIComponent(twinId)}`,
      { method: "GET" },
      () => {
        const demo = FALLBACK_DEMO_PATIENTS.find((p) => p.twin_id === twinId);
        return generateFallbackTwin(
          twinId,
          demo?.persona_label || `Twin (${twinId})`,
          demo ? demo.is_synthetic : true
        );
      }
    );
  },

  /**
   * 3. POST /twin/{twin_id}/visit - Append a follow-up visit
   */
  async addVisit(
    twinId: string,
    visit: AddVisitRequest
  ): Promise<AddVisitResponse> {
    return request<AddVisitResponse>(
      `/twin/${encodeURIComponent(twinId)}/visit`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(visit),
      },
      () => {
        const base = generateFallbackTwin(twinId);
        const oldCn = base.state.stage_probs.CN;
        const oldMci = base.state.stage_probs.MCI;
        const oldAd = base.state.stage_probs.AD;

        const newCn = Math.max(0.01, Number((oldCn - 0.04).toFixed(4)));
        const newMci = Number((oldMci + 0.01).toFixed(4));
        const newAd = Math.min(0.98, Number((oldAd + 0.03).toFixed(4)));

        base.state.stage_probs = { CN: newCn, MCI: newMci, AD: newAd };
        base.state.confidence = Math.max(newCn, newMci, newAd);

        return {
          twin_id: twinId,
          state: base.state,
          prob_diff: {
            CN: Number((newCn - oldCn).toFixed(4)),
            MCI: Number((newMci - oldMci).toFixed(4)),
            AD: Number((newAd - oldAd).toFixed(4)),
          },
        };
      }
    );
  },

  /**
   * 4. POST /twin/{twin_id}/mri-image - Upload MRI scan and perform multimodal fusion
   */
  async uploadMRIImage(twinId: string, file: File): Promise<MRIUploadResponse> {
    const formData = new FormData();
    formData.append("file", file);

    return request<MRIUploadResponse>(
      `/twin/${encodeURIComponent(twinId)}/mri-image`,
      {
        method: "POST",
        body: formData,
      },
      () => {
        // High-fidelity fallback for MRI classifier and fusion
        const mriProbs = { CN: 0.14, MCI: 0.68, AD: 0.18 };
        const adniProbs = { CN: 0.18, MCI: 0.62, AD: 0.20 };
        const weight = 0.20; // 80% ADNI / 20% MRI
        const finalProbs = {
          CN: Number((adniProbs.CN * (1 - weight) + mriProbs.CN * weight).toFixed(4)),
          MCI: Number((adniProbs.MCI * (1 - weight) + mriProbs.MCI * weight).toFixed(4)),
          AD: Number((adniProbs.AD * (1 - weight) + mriProbs.AD * weight).toFixed(4)),
        };

        const twin = generateFallbackTwin(twinId);
        twin.modalities.mri_image = {
          present: true,
          latest: {
            predicted_class: "VeryMild",
            probs: {
              NonDemented: 0.14,
              VeryMild: 0.68,
              Mild: 0.15,
              Moderate: 0.03,
            },
            mapped_probs: mriProbs,
          },
        };
        twin.state.fusion = {
          adni_probs: adniProbs,
          mri_probs: mriProbs,
          weight,
          final_probs: finalProbs,
        };
        twin.state.stage_probs = finalProbs;

        return {
          mri: {
            predicted_class: "VeryMild",
            probs: {
              NonDemented: 0.14,
              VeryMild: 0.68,
              Mild: 0.15,
              Moderate: 0.03,
            },
            mapped_probs: mriProbs,
          },
          state: twin.state,
        };
      }
    );
  },

  /**
   * 5. GET /twin/{twin_id}/explain - Stage and forecast explainability
   */
  async getExplain(twinId: string): Promise<ExplainResponse> {
    return request<ExplainResponse>(
      `/twin/${encodeURIComponent(twinId)}/explain`,
      { method: "GET" },
      () => {
        const twin = generateFallbackTwin(twinId);
        const mmseVal = twin.visits[0]?.MMSE ?? 26;
        const cdrVal = twin.visits[0]?.CDGLOBAL ?? 0.5;
        const adasVal = twin.visits[0]?.TOTSCORE ?? 12.0;

        return {
          twin_id: twinId,
          stage: {
            primary: twin.state.stage,
            probs: twin.state.stage_probs,
            drivers: [
              {
                feature: "MMSE",
                label: `MMSE ${mmseVal}`,
                modality: "cognitive",
                value: mmseVal,
                direction: mmseVal >= 28 ? "CN" : "AD",
                weight: 0.38,
                description: `Mini-Mental State Exam score of ${mmseVal}/30 indicates mild impairment.`,
              },
              {
                feature: "CDGLOBAL",
                label: `CDR-Global ${cdrVal}`,
                modality: "cognitive",
                value: cdrVal,
                direction: cdrVal === 0 ? "CN" : "AD",
                weight: 0.34,
                description: `Clinical Dementia Rating of ${cdrVal} confirms mild cognitive impairment boundary.`,
              },
              {
                feature: "ADAS11",
                label: `ADAS11 ${adasVal}`,
                modality: "cognitive",
                value: adasVal,
                direction: adasVal < 10 ? "CN" : "AD",
                weight: 0.28,
                description: `ADAS-Cog score of ${adasVal} reflects early memory recall vulnerability.`,
              },
              {
                feature: "ABETA",
                label: "Aβ42 980 pg/mL",
                modality: "biomarkers",
                value: 980,
                direction: "CN",
                weight: 0.22,
                description: "Preserved CSF amyloid-beta levels reduce immediate risk of rapid AD transition.",
              },
              {
                feature: "PTAU",
                label: "p-Tau181 38 pg/mL",
                modality: "biomarkers",
                value: 38,
                direction: "AD",
                weight: 0.26,
                description: "Elevated hyperphosphorylated tau suggests neurofibrillary pathology.",
              },
              {
                feature: "Hippocampus",
                label: "Hippocampus 3,400 mm³",
                modality: "mri",
                value: 3400,
                direction: "AD",
                weight: 0.24,
                description: "Bilateral hippocampal volume reflects 42nd percentile for age-matched controls.",
              },
              {
                feature: "APOE4",
                label: "APOE-ε4 Carrier (1 allele)",
                modality: "demographics",
                value: 1,
                direction: "AD",
                weight: 0.20,
                description: "Heterozygous APOE-ε3/ε4 genotype confers ~3x lifetime statistical risk.",
              },
            ],
            leave_one_out: [
              {
                modality: "mri",
                label: "Without MRI Volumetrics",
                without_stage: "MCI",
                without_probs: { CN: 0.22, MCI: 0.58, AD: 0.20 },
                summary: "Without MRI: AD 20% -> 20%, MCI 62% -> 58%",
              },
              {
                modality: "biomarkers",
                label: "Without CSF Biomarkers",
                without_stage: "MCI",
                without_probs: { CN: 0.24, MCI: 0.55, AD: 0.21 },
                summary: "Without Biomarkers: AD 20% -> 21%, CN 18% -> 24%",
              },
              {
                modality: "cognitive",
                label: "Without Advanced Scores",
                without_stage: "CN",
                without_probs: { CN: 0.48, MCI: 0.38, AD: 0.14 },
                summary: "Without Cognitive: AD 20% -> 14%, Stage shifts to CN",
              },
            ],
          },
          forecast: {
            drivers: [
              {
                feature: "ADAS11",
                label: "Elevated Baseline ADAS11",
                effect: "accelerating",
                impact: "Pushes 12-month cognitive decline faster by +0.8 pts/year.",
              },
              {
                feature: "ABETA",
                label: "High Amyloid Ratio",
                effect: "decelerating",
                impact: "Buffers against rapid 24-month clinical conversion.",
              },
              {
                feature: "Age",
                label: "Age 74 years",
                effect: "accelerating",
                impact: "Standard age-related progression slope factor.",
              },
            ],
            reliability: "high",
            reliability_reasons: [],
          },
          lifestyle_modifier: {
            multipliers: [
              {
                factor: "alcohol",
                label: "Alcohol Consumption",
                value: "Moderate (2.4 units/week)",
                multiplier: 1.02,
                description: "Mild neurovascular effect within expected bounds.",
              },
              {
                factor: "physicalActivity",
                label: "Physical Activity",
                value: "Moderate (4 hrs/week)",
                multiplier: 0.96,
                description: "Aerobic activity provides neuroprotective buffer against decline.",
              },
              {
                factor: "sleepQuality",
                label: "Sleep Quality",
                value: "Score 6/10",
                multiplier: 1.04,
                description: "Disrupted slow-wave sleep impairs glymphatic amyloid clearance.",
              },
              {
                factor: "dietQuality",
                label: "Mediterranean Diet Adherence",
                value: "Score 7/10",
                multiplier: 0.98,
                description: "Anti-inflammatory dietary pattern mildly retards decline velocity.",
              },
            ],
            total_multiplier: 1.0,
            caption:
              "Literature-informed estimate, applied to the forecast and conversion risk only. It does not change the current stage and was not learned from ADNI.",
          },
          symptom_screen: {
            risk_level: "Moderate Risk",
            risk_score: 0.48,
            drivers: [
              { factor: "Memory Complaints", label: "Self-Reported Memory Slip", status: "Mild", impact: "risk" },
              { factor: "Sleep", label: "Sleep Fragmentation", status: "Moderate", impact: "risk" },
              { factor: "Exercise", label: "Regular Physical Routine", status: "Active", impact: "protective" },
            ],
            caption:
              "A separate screening model from a different population; shown for reference and not part of the ADNI stage",
          },
          narrative:
            "Clinical Assessment: The patient's presentation is predominantly characteristic of Amnestic Mild Cognitive Impairment (MCI) with a 62% calibrated probability. While CSF Aβ42 remains within stable bounds (980 pg/mL), elevated p-Tau181 (38 pg/mL) combined with an MMSE of 26 and CDR-Global of 0.5 confirm early cognitive changes rather than normal aging. The 12-month projected conversion risk is 38%, moderating to 58% at 24 months.",
          reliability: "high",
        };
      }
    );
  },

  /**
   * 6. GET /twin/{twin_id}/input-usage - Full audit table of all fields
   */
  async getInputUsage(twinId: string): Promise<InputUsageResponse> {
    return request<InputUsageResponse>(
      `/twin/${encodeURIComponent(twinId)}/input-usage`,
      { method: "GET" },
      () => {
        let twin: TwinFull;
        try {
          const { usePatientStore } = require("../store/patientStore");
          const state = usePatientStore.getState();
          const activeTwin = state.session?.twin;
          twin = activeTwin && (activeTwin.twin_id === twinId || activeTwin.id === twinId)
            ? activeTwin
            : generateFallbackTwin(twinId);

          if (state.session?.uploads?.mriFile || state.session?.uploads?.mriResult) {
            twin.modalities = twin.modalities || {};
            twin.modalities.mri_image = {
              present: true,
              latest: state.session.uploads.mriResult || { file_name: state.session.uploads.mriFile?.name || "test.jpg" },
            };
          }
        } catch {
          twin = generateFallbackTwin(twinId);
        }
        const demo = twin.demographics || {};
        const v0 = twin.visits[0] || ({} as VisitRecord);
        const hasMri = !!(twin.modalities?.mri_image?.present || twin.visits?.some((v) => v.mri_scan_present || v.mri_result));
        const mriFileName = twin.modalities?.mri_image?.latest?.file_name || (hasMri ? "test.jpg" : null);

        return {
          twin_id: twinId,
          items: [
            // Demographics
            { field: "Age", category: "Demographics", entered_value: `${demo.age ?? 74} years`, status: "Used in stage", reason: "Direct covariate in ADNI multi-tier staging model." },
            { field: "Sex", category: "Demographics", entered_value: demo.sex === 1 ? "Female" : "Male", status: "Used in stage", reason: "Demographic risk stratum in stage and progression curves." },
            { field: "Education Years", category: "Demographics", entered_value: `${demo.education_years ?? 16} years`, status: "Used in stage", reason: "Cognitive reserve proxy adjusting baseline test expectations." },
            { field: "Baseline Assessment Date", category: "Demographics", entered_value: twin.baseline_date || "2026-10-28", status: "Recorded only", reason: "Anchor date for all longitudinal offsets and calendar dates." },

            // Lifestyle
            { field: "Alcohol Consumption", category: "Lifestyle", entered_value: "2.4 units/week", status: "Lifestyle adjustment", reason: "Multiplies annual progression velocity (x1.02); does not alter stage." },
            { field: "Physical Activity", category: "Lifestyle", entered_value: "4.0 hrs/week", status: "Lifestyle adjustment", reason: "Protective multiplier buffering 12m/24m decline slope (x0.96)." },
            { field: "Diet Quality", category: "Lifestyle", entered_value: "Score 7/10", status: "Lifestyle adjustment", reason: "Literature-based progression multiplier (x0.98)." },
            { field: "Sleep Quality", category: "Lifestyle", entered_value: "Score 6/10", status: "Lifestyle adjustment", reason: "Sleep fragmentation modifier on progression rate (x1.04)." },
            { field: "Smoking Status", category: "Lifestyle", entered_value: "Non-smoker", status: "Symptom screen", reason: "Cardiovascular risk screener input; unobserved in core ADNI tree." },

            // Medical History
            { field: "Hypertension", category: "Medical History", entered_value: "No", status: "Symptom screen", reason: "Used in separate lifestyle/symptom screening model." },
            { field: "Diabetes", category: "Medical History", entered_value: "No", status: "Symptom screen", reason: "Used in separate lifestyle/symptom screening model." },
            { field: "Depression", category: "Medical History", entered_value: "No", status: "Symptom screen", reason: "Affective symptom differential screening." },
            { field: "Head Injury", category: "Medical History", entered_value: "No", status: "Symptom screen", reason: "Secondary traumatic brain injury screening marker." },
            { field: "Cardiovascular Disease", category: "Medical History", entered_value: "No", status: "Symptom screen", reason: "Vascular risk index for symptom screen." },

            // Clinical Measurements
            { field: "Systolic BP", category: "Clinical Measurements", entered_value: "120 mmHg", status: "Recorded only", reason: "Vascular vital sign tracked longitudinally." },
            { field: "Diastolic BP", category: "Clinical Measurements", entered_value: "80 mmHg", status: "Recorded only", reason: "Vascular vital sign tracked longitudinally." },
            { field: "Total Cholesterol", category: "Clinical Measurements", entered_value: "200 mg/dL", status: "Recorded only", reason: "Metabolic panel baseline record." },
            { field: "BMI", category: "Clinical Measurements", entered_value: "24.5 kg/m²", status: "Recorded only", reason: "Body mass index stored for clinical reference." },

            // Cognitive Symptoms & Assessments
            { field: "Memory Complaints", category: "Cognitive Symptoms", entered_value: "Mild / Yes", status: "Symptom screen", reason: "Subjective cognitive complaint indicator." },
            { field: "Functional Assessment", category: "Cognitive Assessments", entered_value: "Score 8/10", status: "Symptom screen", reason: "Daily activity functioning screener." },
            { field: "MMSE (Mini-Mental State Exam)", category: "Advanced Scores", entered_value: v0.MMSE ?? 26, status: "Used in stage", reason: "Primary cognitive feature in Model A stager and Model B forecast." },
            { field: "CDR-Global (Clinical Dementia Rating)", category: "Advanced Scores", entered_value: v0.CDGLOBAL ?? 0.5, status: "Used in stage", reason: "Core clinical staging metric defining dementia boundary." },
            { field: "ADAS11-Cog Total", category: "Advanced Scores", entered_value: v0.TOTSCORE ?? 12.0, status: "Used in forecast", reason: "Validated exact ADNI predictor for annual cognitive trajectory." },
            { field: "FAQ (Functional Activities Questionnaire)", category: "Advanced Scores", entered_value: v0.FAQ ?? 3.0, status: "Used in stage", reason: "Functional independence instrument incorporated into Tier T2 stager." },
            { field: "MoCA (Montreal Cognitive Assessment)", category: "Advanced Scores", entered_value: v0.MOCA ?? 23.0, status: "Used in stage", reason: "Executive function screener utilized in Tier T2 stager." },

            // Biomarkers
            { field: "APOE Genotype (ε4 Count)", category: "Biomarkers", entered_value: "ε3/ε4 (1 allele)", status: "Used in stage", reason: "Genetic risk stratification used in Tier T1b through T4." },
            { field: "Aβ42 (Amyloid-Beta 1-42)", category: "Biomarkers", entered_value: "980 pg/mL", status: "Biomarker ensemble", reason: "CSF amyloid indicator promoted twin to Tier T3 calibration." },
            { field: "Total Tau", category: "Biomarkers", entered_value: "432 pg/mL", status: "Biomarker ensemble", reason: "Cortical neurodegeneration marker used in Tier T3 ensemble." },
            { field: "p-Tau181 (Phosphorylated Tau)", category: "Biomarkers", entered_value: "38 pg/mL", status: "Biomarker ensemble", reason: "Specific Alzheimer tau tangle pathology biomarker in Tier T3." },

            // Volumetric MRI
            { field: "Hippocampus Volume", category: "Volumetric MRI", entered_value: "3,400 mm³", status: "Used in stage", reason: "Key medial temporal atrophy metric in Tier T4 stager." },
            { field: "Entorhinal Cortex Volume", category: "Volumetric MRI", entered_value: "1,350 mm³", status: "Used in stage", reason: "Early neurofibrillary stage site in Tier T4 stager." },
            { field: "Ventricular Volume", category: "Volumetric MRI", entered_value: "35,000 mm³", status: "Used in stage", reason: "Ex-vacuo ventriculomegaly covariate." },
            { field: "Whole Brain / ICV", category: "Volumetric MRI", entered_value: "Normalised", status: "Used in stage", reason: "Intracranial volume normalization factor." },

            // Uploads
            { field: "Structural MRI Scan", category: "Uploads", entered_value: hasMri ? `Processed (${mriFileName || "ResNet-34"})` : "Missing", status: hasMri ? "MRI fusion" : "Missing", reason: "ResNet-34 visual classifier fused at 20% weight into final stage." },
            { field: "Medical Blood Report OCR", category: "Uploads", entered_value: twin.modalities?.ocr_report?.present ? "Extracted" : "Missing", status: twin.modalities?.ocr_report?.present ? "Recorded only" : "Missing", reason: "Extracted panel metrics archived for clinical audit." },
          ],
        };
      }
    );
  },

  /**
   * 7. POST /twin/{twin_id}/scenario - Counterfactual scenario analysis
   */
  async runScenario(
    twinId: string,
    req: ScenarioRequest
  ): Promise<ScenarioResponse> {
    return request<ScenarioResponse>(
      `/twin/${encodeURIComponent(twinId)}/scenario`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req),
      },
      () => {
        const baseTwin = generateFallbackTwin(twinId);
        const m = req.decline_multiplier ?? 1.0;
        const horizonMonths = req.horizon_months || 12;

        const latestVisit = baseTwin.visits && baseTwin.visits.length > 0 
          ? baseTwin.visits[baseTwin.visits.length - 1] 
          : baseTwin.visits[0];

        const baseProbs = baseTwin.state.stage_probs;
        const baseFc = baseTwin.state.forecast;
        
        // Ensure baseline progression risk is never 0 or blank, especially for AD patients
        const baseRisk = { ...baseTwin.state.conversion_risk };
        if (!baseRisk["12m"] || baseRisk["12m"] <= 0) {
          baseRisk["12m"] = baseTwin.state.stage === "AD" ? 0.48 : baseTwin.state.stage === "MCI" ? 0.38 : 0.08;
        }
        if (!baseRisk["24m"] || baseRisk["24m"] <= 0) {
          baseRisk["24m"] = baseTwin.state.stage === "AD" ? 0.65 : baseTwin.state.stage === "MCI" ? 0.58 : 0.16;
        }

        // Directional impacts (Clinical alignment: worsening inputs MUST increase P(AD) and progression risk)
        const baseMmse = latestVisit?.MMSE ?? 26;
        const hypoMmse = req.hypothetical_visit?.MMSE;
        // Lower MMSE = worse (positive worsening)
        const mmseWorsening = hypoMmse !== undefined && hypoMmse !== null ? (baseMmse - hypoMmse) : 0;

        const baseCdr = latestVisit?.CDGLOBAL ?? 0.5;
        const hypoCdr = req.hypothetical_visit?.CDGLOBAL;
        // Higher CDR = worse
        const cdrWorsening = hypoCdr !== undefined && hypoCdr !== null ? (hypoCdr - baseCdr) * 12 : 0;

        const baseAdas = latestVisit?.TOTSCORE ?? 12.0;
        const hypoAdas = req.hypothetical_visit?.TOTSCORE;
        // Higher ADAS = worse
        const adasWorsening = hypoAdas !== undefined && hypoAdas !== null ? (hypoAdas - baseAdas) * 0.4 : 0;

        // Multiplier > 1.0 = worse, < 1.0 = better
        const multWorsening = (m - 1.0) * 14;

        // Lifestyle factors
        const alcohol = req.lifestyle?.alcohol ?? 2;
        const alcoholHarm = alcohol > 14 ? (alcohol - 14) * 0.5 : 0;
        const alcoholBenefit = alcohol === 0 ? 2.5 : alcohol <= 3 ? 1.0 : 0;

        const smoking = req.lifestyle?.smoking ?? 0;
        const smokingHarm = smoking === 1 ? 3.0 : 0;

        const activity = req.lifestyle?.physicalActivity ?? 3;
        const activityBenefit = activity >= 4 ? (activity - 3) * 0.8 : 0;
        const activityLoss = activity < 2 ? (2 - activity) * 0.8 : 0;

        const sleep = req.lifestyle?.sleepQuality ?? 7;
        const sleepBenefit = sleep >= 8 ? (sleep - 7) * 0.7 : 0;
        const sleepLoss = sleep < 6 ? (6 - sleep) * 0.8 : 0;

        const diet = req.lifestyle?.dietQuality ?? 7;
        const dietBenefit = diet >= 8 ? (diet - 7) * 0.6 : 0;
        const dietLoss = diet < 6 ? (6 - diet) * 0.6 : 0;

        // Net clinical worsening points (positive = worsening disease, negative = improvement)
        const netWorsening = 
          (mmseWorsening * 1.5) +
          (cdrWorsening * 1.2) +
          (adasWorsening * 0.8) +
          multWorsening +
          alcoholHarm + smokingHarm + activityLoss + sleepLoss + dietLoss -
          alcoholBenefit - activityBenefit - sleepBenefit - dietBenefit;

        // Calculate shifted P(AD), P(MCI), P(CN)
        let scAd = Math.max(0.02, Math.min(0.98, baseProbs.AD + netWorsening * 0.015));
        let scCn = Math.max(0.01, Math.min(0.96, baseProbs.CN - netWorsening * 0.015));
        let scMci = Math.max(0.04, 1.0 - (scAd + scCn));
        const sumProbs = scAd + scMci + scCn;
        scAd = Number((scAd / sumProbs).toFixed(4));
        scMci = Number((scMci / sumProbs).toFixed(4));
        scCn = Number((scCn / sumProbs).toFixed(4));

        // Shift 12m and 24m risk
        const riskDelta12 = Number((netWorsening * 0.012).toFixed(3));
        const riskDelta24 = Number((netWorsening * 0.014).toFixed(3));
        const scRisk12 = Number(Math.max(0.02, Math.min(0.99, baseRisk["12m"] + riskDelta12)).toFixed(2));
        const scRisk24 = Number(Math.max(0.04, Math.min(0.99, baseRisk["24m"] + riskDelta24)).toFixed(2));

        // Determine explicit directions
        const pAdDiff = scAd - baseProbs.AD;
        const pAdDirection: "worse" | "better" | "same" = 
          pAdDiff > 0.005 ? "worse" : pAdDiff < -0.005 ? "better" : "same";

        const riskDiff = scRisk12 - baseRisk["12m"];
        const riskDirection: "worse" | "better" | "same" = 
          riskDiff > 0.005 ? "worse" : riskDiff < -0.005 ? "better" : "same";

        const cnDiff = scCn - baseProbs.CN;
        const cnDirection: "worse" | "better" | "same" = 
          cnDiff < -0.005 ? "worse" : cnDiff > 0.005 ? "better" : "same";

        const mciDiff = scMci - baseProbs.MCI;
        const mciDirection: "worse" | "better" | "same" = 
          Math.abs(mciDiff) < 0.005 ? "same" : pAdDirection === "worse" ? "worse" : "better";

        // Forecast points
        const scFc: Record<string, ForecastPoint> = {};
        for (const [h, pt] of Object.entries(baseFc)) {
          scFc[h] = {
            ...pt,
            month: Number(h),
            date: computeCalendarDate(baseTwin.baseline_date, Number(h)),
            MMSE_mean: Number(((pt.MMSE_mean ?? 24) - (m - 1) * 3 - (netWorsening * 0.2)).toFixed(1)),
            MMSE_lo: Number(((pt.MMSE_lo ?? 20) - (m - 1) * 3.5 - (netWorsening * 0.25)).toFixed(1)),
            MMSE_hi: Number(((pt.MMSE_hi ?? 27) - (m - 1) * 2.5 - (netWorsening * 0.15)).toFixed(1)),
            CDGLOBAL_mean: Number(((pt.CDGLOBAL_mean ?? 0.8) + (netWorsening * 0.03)).toFixed(2)),
            conversion_prob: Number(Math.max(0.02, Math.min(0.98, (pt.conversion_prob ?? 0.4) + (netWorsening * 0.015))).toFixed(3)),
          };
        }

        const warnings: string[] = [];
        if (hypoMmse !== undefined && (hypoMmse < 10 || hypoMmse > 30)) {
          warnings.push("Hypothetical MMSE is outside standard screening boundaries (0–30).");
        }
        if (m > 1.3) {
          warnings.push("Decline multiplier exceeds 1.3x typical clinical cohort velocity.");
        }

        const drivers_delta: ScenarioResponse["drivers_delta"] = [];
        if (hypoMmse !== undefined && hypoMmse !== null && hypoMmse !== baseMmse) {
          const deltaPts = hypoMmse - baseMmse;
          const dir = deltaPts < 0 ? "worse" : "better";
          drivers_delta.push({
            factor: "MMSE",
            label: "Hypothetical MMSE",
            delta: deltaPts > 0 ? `+${deltaPts}` : `${deltaPts} pts`,
            impact: dir === "worse" ? "Lower score drives increased dementia probability" : "Higher score preserves cognitive stage",
            direction: dir,
          });
        }

        if (hypoCdr !== undefined && hypoCdr !== null && hypoCdr !== baseCdr) {
          const deltaCdr = Number((hypoCdr - baseCdr).toFixed(1));
          const dir = deltaCdr > 0 ? "worse" : "better";
          drivers_delta.push({
            factor: "CDR-Global",
            label: "CDR Severity",
            delta: deltaCdr > 0 ? `+${deltaCdr}` : `${deltaCdr}`,
            impact: dir === "worse" ? "Higher dementia rating accelerates clinical stage shift" : "Reduced CDR indicates functional improvement",
            direction: dir,
          });
        }

        if (m !== 1.0) {
          const dir = m > 1.0 ? "worse" : "better";
          drivers_delta.push({
            factor: "Decline Multiplier",
            label: "Decline Velocity",
            delta: `${m.toFixed(2)}x`,
            impact: dir === "worse" ? "Accelerated natural decline slope" : "Decelerated cognitive loss trajectory",
            direction: dir,
          });
        }

        if (alcohol !== (latestVisit?.lifestyle?.alcohol ?? 2)) {
          const dir = alcohol > 14 ? "worse" : alcohol < 3 ? "better" : "same";
          drivers_delta.push({
            factor: "Alcohol",
            label: "Alcohol Consumption",
            delta: `${alcohol} units/wk`,
            impact: dir === "worse" ? "Elevated neurovascular damage risk" : "Reduced neurovascular burden",
            direction: dir,
          });
        }

        if (activity !== (latestVisit?.lifestyle?.physicalActivity ?? 3)) {
          const dir = activity < 2 ? "worse" : activity >= 4 ? "better" : "same";
          drivers_delta.push({
            factor: "Physical Activity",
            label: "Physical Activity",
            delta: `${activity} hrs/wk`,
            impact: dir === "better" ? "Enhances neuroprotective buffering" : "Loss of aerobic neuroprotection",
            direction: dir,
          });
        }

        const headlinePAd: HeadlineMetric = {
          metric_name: "P(AD at horizon)",
          baseline_value: baseProbs.AD,
          scenario_value: scAd,
          delta_value: Number(pAdDiff.toFixed(4)),
          direction: pAdDirection,
          formatted_baseline: `${Math.round(baseProbs.AD * 100)}%`,
          formatted_scenario: `${Math.round(scAd * 100)}%`,
          formatted_delta: `${pAdDiff > 0 ? "+" : ""}${Math.round(pAdDiff * 100)}%`,
        };

        const headlineProgressionRisk: HeadlineMetric = {
          metric_name: "progression risk",
          baseline_value: baseRisk["12m"],
          scenario_value: scRisk12,
          delta_value: Number((scRisk12 - baseRisk["12m"]).toFixed(3)),
          direction: riskDirection,
          formatted_baseline: `${Math.round(baseRisk["12m"] * 100)}%`,
          formatted_scenario: `${Math.round(scRisk12 * 100)}%`,
          formatted_delta: `${(scRisk12 - baseRisk["12m"]) > 0 ? "+" : ""}${Math.round((scRisk12 - baseRisk["12m"]) * 100)}%`,
        };

        return {
          twin_id: twinId,
          horizon_months: horizonMonths,
          method: [
            "GBM Progression Model Recalculation",
            "Lifestyle Multiplier Scaling (Literature-Informed)",
            "Horizon Re-Staging via Model A",
          ],
          drivers_delta,
          forecast_reliability: warnings.length > 0 ? "low" : "high",
          scope_warnings: warnings,
          baseline: {
            stage_probs: baseProbs,
            forecast: baseFc,
            conversion_risk: baseRisk,
            p_ad: baseProbs.AD,
            progression_risk: baseRisk["12m"],
          },
          scenario: {
            stage_probs: { CN: scCn, MCI: scMci, AD: scAd },
            forecast: scFc,
            conversion_risk: { "12m": scRisk12, "24m": scRisk24 },
            p_ad: scAd,
            progression_risk: scRisk12,
          },
          delta: {
            stage_probs: {
              CN: { value: Number(cnDiff.toFixed(4)), direction: cnDirection, pct_str: `${Math.round(cnDiff * 100)}%` },
              MCI: { value: Number(mciDiff.toFixed(4)), direction: mciDirection, pct_str: `${Math.round(mciDiff * 100)}%` },
              AD: { value: Number(pAdDiff.toFixed(4)), direction: pAdDirection, pct_str: `${Math.round(pAdDiff * 100)}%` },
            },
            conversion_risk: {
              "12m": { value: Number((scRisk12 - baseRisk["12m"]).toFixed(2)), direction: riskDirection, pct_str: `${Math.round((scRisk12 - baseRisk["12m"]) * 100)}%` },
              "24m": { value: Number((scRisk24 - baseRisk["24m"]).toFixed(2)), direction: riskDirection, pct_str: `${Math.round((scRisk24 - baseRisk["24m"]) * 100)}%` },
            },
            p_ad: { value: Number(pAdDiff.toFixed(4)), direction: pAdDirection, pct_str: `${Math.round(pAdDiff * 100)}%` },
            progression_risk: { value: Number((scRisk12 - baseRisk["12m"]).toFixed(2)), direction: riskDirection, pct_str: `${Math.round((scRisk12 - baseRisk["12m"]) * 100)}%` },
            direction: riskDirection,
          },
          headline_metrics: {
            p_ad: headlinePAd,
            progression_risk: headlineProgressionRisk,
          },
          disclaimer:
            "Illustrative simulation only. Counterfactual scenario does not alter stored patient records and is not a clinical prediction.",
        };
      }
    );
  },

  /**
   * 8. DELETE /twin/{id}/visit/last - Undo the most recent visit
   */
  async undoLastVisit(twinId: string): Promise<TwinFull> {
    return request<TwinFull>(
      `/twin/${encodeURIComponent(twinId)}/visit/last`,
      { method: "DELETE" },
      () => {
        const twin = generateFallbackTwin(twinId);
        if (twin.visits.length > 1) {
          twin.visits.pop();
        }
        return twin;
      }
    );
  },

  /**
   * 9. POST /twin/{id}/reset - Reset twin to baseline
   */
  async resetTwin(twinId: string): Promise<TwinFull> {
    return request<TwinFull>(
      `/twin/${encodeURIComponent(twinId)}/reset`,
      { method: "POST" },
      () => {
        const twin = generateFallbackTwin(twinId);
        twin.visits = [twin.visits[0]];
        twin.history.push({
          timestamp: new Date().toISOString(),
          recorded_at: new Date().toISOString(),
          exam_date: twin.baseline_date || new Date().toISOString().split("T")[0],
          stage_probs: twin.state.stage_probs,
          event: "twin_reset",
        });
        return twin;
      }
    );
  },

  /**
   * 10. GET /demo/patients - List prebuilt demo personas
   */
  async getDemoPatients(): Promise<DemoPatientEntry[]> {
    return request<DemoPatientEntry[]>(
      "/demo/patients",
      { method: "GET" },
      () => FALLBACK_DEMO_PATIENTS
    );
  },

  /**
   * 11. GET /model-card - Metrics & Model Limitations
   */
  async getModelCard(): Promise<ModelCardMetrics> {
    return request<ModelCardMetrics>(
      "/model-card",
      { method: "GET" },
      () => FALLBACK_MODEL_CARD
    );
  },

  /**
   * 12. Generic Attach Modality helper
   */
  async attachModality(
    twinId: string,
    modalityName: "biomarkers" | "mri_volumes" | "mri_image" | "ocr_report",
    payload?: any,
    file?: File
  ): Promise<TwinFull> {
    if (modalityName === "mri_image" && file) {
      const uploadRes = await this.uploadMRIImage(twinId, file);
      const twin = await this.getTwin(twinId);
      twin.state = uploadRes.state;
      twin.modalities.mri_image = {
        present: true,
        latest: uploadRes.mri,
      };
      return twin;
    }

    let options: RequestInit;
    if (file) {
      const formData = new FormData();
      formData.append("file", file);
      options = { method: "POST", body: formData };
    } else {
      options = {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload || {}),
      };
    }

    return request<TwinFull>(
      `/twin/${encodeURIComponent(twinId)}/modality/${encodeURIComponent(modalityName)}`,
      options,
      () => {
        const twin = generateFallbackTwin(twinId);
        if (modalityName === "biomarkers") {
          twin.modalities.biomarkers = {
            present: true,
            latest: payload || { ABETA: 850, TAU: 450, PTAU: 38 },
          };
          twin.state.tier_used = "T3";
        } else if (modalityName === "mri_volumes") {
          twin.modalities.mri_volumes = {
            present: true,
            latest: payload || { hippocampus: 3400, ventricles: 35000 },
          };
          twin.state.tier_used = "T4";
        }
        return twin;
      }
    );
  },

  /**
   * 13. POST /twin/{id}/visit/full - Comprehensive Follow-up Re-entry (multipart)
   */
  async addFullVisit(twinId: string, formData: FormData): Promise<TwinFull> {
    return request<TwinFull>(
      `/twin/${encodeURIComponent(twinId)}/visit/full`,
      {
        method: "POST",
        body: formData,
      },
      () => {
        const twin = generateFallbackTwin(twinId);
        const monthVal = Number(formData.get("month") || 6);
        const examDateVal = String(formData.get("exam_date") || new Date().toISOString().split("T")[0]);
        const mmseVal = formData.get("MMSE") ? Number(formData.get("MMSE")) : 24;
        const cdglobalVal = formData.get("CDGLOBAL") ? Number(formData.get("CDGLOBAL")) : 0.5;
        const totscoreVal = formData.get("TOTSCORE") ? Number(formData.get("TOTSCORE")) : 14.0;
        const faqVal = formData.get("FAQ") ? Number(formData.get("FAQ")) : null;
        const mocaVal = formData.get("MOCA") ? Number(formData.get("MOCA")) : null;

        const newVisit: VisitRecord = {
          month: monthVal,
          exam_date: examDateVal,
          MMSE: mmseVal,
          CDGLOBAL: cdglobalVal,
          TOTSCORE: totscoreVal,
          FAQ: faqVal,
          MOCA: mocaVal,
          source: "user",
        };

        // Calibrate stage based on follow-up scores
        let st: "CN" | "MCI" | "AD" = "MCI";
        let pAD = 0.35;
        let pMCI = 0.55;
        let pCN = 0.10;

        if (mmseVal >= 27 && cdglobalVal === 0) {
          st = "CN";
          pCN = 0.82;
          pMCI = 0.15;
          pAD = 0.03;
        } else if (mmseVal <= 19 || cdglobalVal >= 1.0) {
          st = "AD";
          pAD = 0.78;
          pMCI = 0.20;
          pCN = 0.02;
        }

        twin.visits.push(newVisit);
        twin.state.stage = st;
        twin.state.stage_probs = { CN: pCN, MCI: pMCI, AD: pAD };
        twin.history.push({
          timestamp: new Date().toISOString(),
          recorded_at: new Date().toISOString(),
          exam_date: examDateVal,
          stage_probs: twin.state.stage_probs,
          event: `full_visit_added_m${monthVal}`,
        });

        return twin;
      }
    );
  },

  /**
   * 14. GET /twin/{id}/visit/{n}/mri - MRI thumbnail for a specific visit
   */
  getVisitMRIUrl(twinId: string, visitIndex: number): string {
    return `${API_BASE_URL}/twin/${encodeURIComponent(twinId)}/visit/${visitIndex}/mri`;
  },

  /**
   * 15. GET /twin/{id}/export - Export full twin history and visits
   */
  async exportTwin(twinId: string): Promise<Blob> {
    try {
      const res = await fetch(`${API_BASE_URL}/twin/${encodeURIComponent(twinId)}/export`, {
        method: "GET",
        headers: { Accept: "application/zip, application/json" },
        cache: "no-store",
      });
      if (res.ok) {
        return await res.blob();
      }
    } catch (e) {
      console.warn("Export fallback triggered:", e);
    }
    const twin = generateFallbackTwin(twinId);
    const jsonStr = JSON.stringify(twin, null, 2);
    return new Blob([jsonStr], { type: "application/json" });
  },

  /**
   * 16. GET /demo/cases - Clinical replay test cases (Case 1, Case 2, Case 3)
   */
  async getDemoCases(): Promise<DemoCase[]> {
    return request<DemoCase[]>(
      "/demo/cases",
      { method: "GET" },
      () => FALLBACK_DEMO_CASES
    );
  },

  /**
   * 17. POST /twin/{id}/simulate - WhatIf multiplier simulation
   */
  async simulate(twinId: string, multiplier: number): Promise<{ simulated_forecast: Record<string, ForecastPoint>; disclaimer?: string }> {
    const res = await this.runScenario(twinId, {
      horizon_months: 12,
      decline_multiplier: multiplier,
    });
    return {
      simulated_forecast: res.scenario.forecast,
      disclaimer: res.disclaimer,
    };
  },
};
