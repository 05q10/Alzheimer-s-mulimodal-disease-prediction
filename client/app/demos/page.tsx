"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  DemoPatientEntry,
  DemoCase,
  twinApi,
  FALLBACK_DEMO_PATIENTS,
  FALLBACK_DEMO_CASES,
  TwinFull,
  VisitRecord,
} from "../lib/api";
import { usePatientStore } from "../store/patientStore";
import {
  Users,
  ShieldAlert,
  Calendar,
  Edit3,
  Eye,
  ArrowLeft,
  Sparkles,
  Play,
  CheckCircle2,
  Clock,
  Activity,
  Brain,
  RotateCcw,
} from "lucide-react";

export default function DemosPage() {
  const router = useRouter();
  const { loadPersona, loadDemoCase, setTwin, setTwinId } = usePatientStore();

  const [demoPatients, setDemoPatients] = useState<DemoPatientEntry[]>(FALLBACK_DEMO_PATIENTS);
  const [demoCases, setDemoCases] = useState<DemoCase[]>(FALLBACK_DEMO_CASES);
  const [loading, setLoading] = useState(true);

  // Track playback index per case: 0 = baseline only, 1 = up to visit 2, 2 = up to visit 3
  const [caseStep, setCaseStep] = useState<Record<string, number>>({
    "case-1": 0,
    "case-2": 0,
    "case-3": 0,
  });

  const [playingCaseId, setPlayingCaseId] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      twinApi.getDemoPatients().catch(() => FALLBACK_DEMO_PATIENTS),
      twinApi.getDemoCases().catch(() => FALLBACK_DEMO_CASES),
    ])
      .then(([patients, cases]) => {
        if (patients && patients.length > 0) setDemoPatients(patients);
        if (cases && cases.length > 0) setDemoCases(cases);
      })
      .finally(() => setLoading(false));
  }, []);

  const getStageBadge = (stage: string) => {
    if (stage === "CN") {
      return "bg-emerald-500/15 text-emerald-400 border-emerald-500/30";
    }
    if (stage === "MCI") {
      return "bg-amber-500/15 text-amber-400 border-amber-500/30";
    }
    return "bg-rose-500/15 text-rose-400 border-rose-500/30";
  };

  const handleLoadBaselineIntoEditor = (c: DemoCase) => {
    loadDemoCase(c);
    router.push("/intake");
  };

  const handlePlayNextVisit = async (c: DemoCase) => {
    setPlayingCaseId(c.id);

    try {
      const currentIdx = caseStep[c.id] ?? 0;
      const nextIdx = Math.min(currentIdx + 1, c.visits.length - 1);

      // Create or build the Twin instance with visits up to nextIdx
      const twinId = `twin-${c.id}`;

      // Assemble visits up to nextIdx
      const builtVisits: VisitRecord[] = c.visits.slice(0, nextIdx + 1).map((v, i) => ({
        visit_number: i + 1,
        month: v.month,
        exam_date: v.exam_date,
        source: "synthetic",
        stage: v.stage,
        stage_probs: v.stage_probs,
        scores: {
          MMSE: v.MMSE,
          CDGLOBAL: v.CDGLOBAL,
          TOTSCORE: v.TOTSCORE,
          FAQ: v.FAQ,
          MOCA: v.MOCA,
          functionalAssessment: v.functionalAssessment,
          adl: v.adl,
        },
        biomarkers: v.biomarkers,
        mri_volumes: v.mri_volumes,
        lifestyle: v.lifestyle,
        symptoms: v.symptoms,
        mri_scan_present: i === 0 || !v.mri_stale,
        mri_stale: v.mri_stale ?? (i > 0),
        mri_result: i === 0 ? {
          predicted_class: c.baseline_stage === "CN" ? "NonDemented" : c.baseline_stage === "MCI" ? "VeryMild" : "Mild",
          confidence: 0.94,
          file_name: "test.jpg",
          slice_index: 24,
        } : undefined,
        fusion: {
          adni_probs: v.stage_probs,
          mri_probs: v.stage_probs,
          weight: 0.2,
          final_probs: v.stage_probs,
        },
        forecast: {
          "6m": {
            CN: Math.max(0.02, v.stage_probs.CN - (v.stage === "CN" ? 0.05 : 0.02)),
            MCI: v.stage_probs.MCI,
            AD: Math.min(0.95, v.stage_probs.AD + 0.05),
          },
          "12m": {
            CN: Math.max(0.01, v.stage_probs.CN - (v.stage === "CN" ? 0.1 : 0.04)),
            MCI: Math.max(0.05, v.stage_probs.MCI - 0.05),
            AD: Math.min(0.98, v.stage_probs.AD + 0.1),
          },
          "24m": {
            CN: Math.max(0.01, v.stage_probs.CN - 0.15),
            MCI: Math.max(0.05, v.stage_probs.MCI - 0.1),
            AD: Math.min(0.99, v.stage_probs.AD + 0.18),
          },
        },
        conversion_risk: {
          "12m": Math.round(v.stage_probs.AD * 70),
          "24m": Math.round(v.stage_probs.AD * 85),
        },
        lifestyle_multiplier: v.lifestyle?.smoking ? 1.25 : v.lifestyle?.physicalActivity && v.lifestyle.physicalActivity >= 7 ? 0.8 : 1.0,
      }));

      const latestVisit = builtVisits[builtVisits.length - 1];

      const fullTwin: TwinFull = {
        twin_id: twinId,
        id: twinId,
        patient_name: c.patient_name,
        created_at: new Date().toISOString(),
        demographics: {
          age: c.age,
          sex: c.sex,
          education_years: c.education_years,
          apoe4: c.apoe4_count,
          baseline_date: c.baseline_date,
        },
        scores: latestVisit.scores as any,
        biomarkers: latestVisit.biomarkers as any,
        mri_volumes: latestVisit.mri_volumes as any,
        lifestyle: latestVisit.lifestyle as any,
        stage: latestVisit.stage,
        stage_probs: latestVisit.stage_probs,
        progression_risk: latestVisit.stage === "AD" ? 82 : Math.round(latestVisit.stage_probs.AD * 100),
        conversion_risk: latestVisit.conversion_risk,
        forecast: latestVisit.forecast,
        modalities: {
          basic_scores: { present: true },
          biomarkers: { present: true },
          mri_volumes: { present: true },
          mri_image: { present: true },
          ocr_report: { present: false },
        },
        state: {
          stage: latestVisit.stage,
          stage_probs: latestVisit.stage_probs,
          confidence: latestVisit.stage_probs[latestVisit.stage],
          tier_used: "T3",
          forecast: latestVisit.forecast,
          conversion_risk: latestVisit.conversion_risk,
        },
        history: builtVisits.map((v) => ({
          timestamp: new Date().toISOString(),
          recorded_at: v.exam_date,
          exam_date: v.exam_date,
          stage_probs: v.stage_probs,
          event: `Visit M${v.month} evaluation recorded`,
        })),
        visits: builtVisits,
        timeline: builtVisits.map((v, i) => ({
          visit_number: i + 1,
          month: v.month,
          exam_date: v.exam_date,
          stage: v.stage,
          stage_probs: v.stage_probs,
          notes: `Recorded Month ${v.month} evaluation`,
        })),
        is_synthetic: true,
      };

      // Update store and navigate
      setTwinId(twinId);
      setTwin(fullTwin);
      setCaseStep((prev) => ({ ...prev, [c.id]: nextIdx }));

      // Small delay for smooth feedback
      setTimeout(() => {
        router.push(`/twin/${encodeURIComponent(twinId)}`);
      }, 350);
    } catch (err) {
      console.error("Failed to play next visit:", err);
    } finally {
      setPlayingCaseId(null);
    }
  };

  const handleResetCase = (caseId: string) => {
    setCaseStep((prev) => ({ ...prev, [caseId]: 0 }));
  };

  return (
    <div className="min-h-screen bg-transparent text-white relative selection:bg-purple-500 selection:text-white pb-24">
      {/* Background Ambience & Grid */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(168,85,247,0.15),rgba(255,255,255,0))]" />
        <div className="absolute top-1/4 left-10 w-96 h-96 bg-blue-600/10 rounded-full blur-[128px]" />
        <div
          className="absolute inset-0 
          bg-[linear-gradient(rgba(255,255,255,0.06)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.06)_1px,transparent_1px)] 
          bg-[size:60px_60px] opacity-20"
        />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-6 pt-12 space-y-12">
        {/* Header Bar */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-white/10">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="p-2.5 rounded-xl bg-white/5 hover:bg-white/10 text-neutral-400 hover:text-white border border-white/10 transition flex items-center gap-1.5 text-xs font-semibold"
            >
              <ArrowLeft className="w-4 h-4" />
              <span>Back to Home</span>
            </Link>

            <div>
              <div className="flex items-center gap-2">
                <Brain className="w-5 h-5 text-purple-400" />
                <h1 className="text-2xl font-black tracking-tight text-white">
                  Clinical Demonstration Cases & Personas
                </h1>
              </div>
              <p className="text-xs text-neutral-400 mt-0.5">
                Replay multi-visit longitudinal cases step-by-step or load individual patient profiles into the editor.
              </p>
            </div>
          </div>

          <Link
            href="/intake"
            className="px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-bold shadow-lg shadow-blue-600/30 transition flex items-center gap-1.5 self-start sm:self-auto"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Create Custom Twin</span>
          </Link>
        </div>

        {/* SECTION 1: 3 REPLAYABLE DEMO CASES */}
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-emerald-400" />
                <h2 className="text-lg font-bold text-white tracking-wide">
                  Step-by-Step Replay Cases
                </h2>
              </div>
              <p className="text-xs text-neutral-400">
                Interactive clinical trajectories: load the baseline into the intake form, or step through visits sequentially.
              </p>
            </div>
            <span className="text-xs font-mono text-purple-400 bg-purple-500/10 px-3 py-1 rounded-full border border-purple-500/20">
              3 Benchmark Trajectories
            </span>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {demoCases.map((c, idx) => {
              const currentStep = caseStep[c.id] ?? 0;
              const maxSteps = c.visits.length - 1;
              const hasMoreVisits = currentStep < maxSteps;
              const nextVisitNum = currentStep + 2;
              const nextVisitMonth = c.visits[currentStep + 1]?.month ?? 12;

              return (
                <motion.div
                  key={`case-card-${c.id}-${idx}`}
                  initial={{ opacity: 0, y: 16 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.1, duration: 0.4 }}
                  className="rounded-2xl p-6 bg-white/[0.04] backdrop-blur-xl border border-white/10 hover:border-purple-500/30 transition-all duration-300 flex flex-col justify-between shadow-xl"
                >
                  <div className="space-y-4">
                    {/* Top Row: Case tag & Stage */}
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[11px] font-mono font-bold px-2.5 py-1 rounded-lg bg-purple-500/15 text-purple-300 border border-purple-500/30">
                        {c.id.toUpperCase()}
                      </span>
                      <span
                        className={`text-xs font-bold px-2.5 py-0.5 rounded-full border ${getStageBadge(
                          c.baseline_stage
                        )}`}
                      >
                        Baseline: {c.baseline_stage}
                      </span>
                    </div>

                    {/* Title & Patient Name */}
                    <div>
                      <h3 className="text-base font-bold text-white group-hover:text-purple-300 transition">
                        {c.title}
                      </h3>
                      <div className="flex items-center gap-2 text-xs text-neutral-400 mt-1">
                        <span className="font-medium text-neutral-300">{c.patient_name}</span>
                        <span>•</span>
                        <span>Age {c.age}</span>
                        <span>•</span>
                        <span>APOE: {c.apoe4_count} allele{c.apoe4_count === 1 ? "" : "s"}</span>
                      </div>
                    </div>

                    {/* Narrative Story */}
                    <p className="text-xs text-neutral-300 leading-relaxed bg-black/20 p-3 rounded-xl border border-white/5">
                      {c.story}
                    </p>

                    {/* Longitudinal Progression Stepper */}
                    <div className="p-3 rounded-xl bg-white/[0.02] border border-white/5 space-y-2">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-neutral-400 font-medium">Replay Progress</span>
                        <span className="text-purple-400 font-mono font-semibold">
                          Visit {currentStep + 1} of {c.visits.length} ({c.visits[currentStep]?.month ?? 0}m)
                        </span>
                      </div>

                      {/* Progress bar dots */}
                      <div className="grid grid-cols-3 gap-2 pt-1">
                        {c.visits.map((v, vIdx) => {
                          const isDone = vIdx <= currentStep;
                          const isCurrent = vIdx === currentStep;

                          return (
                            <div
                              key={`progress-dot-${c.id}-${v.visit_number}-${vIdx}`}
                              className={`rounded-lg p-2 text-center transition-all ${
                                isCurrent
                                  ? "bg-purple-600/30 border border-purple-400/50 text-white shadow-md shadow-purple-500/20"
                                  : isDone
                                  ? "bg-emerald-500/10 border border-emerald-500/30 text-emerald-300"
                                  : "bg-white/5 border border-white/10 text-neutral-500"
                              }`}
                            >
                              <div className="text-[10px] font-bold">M{v.month}</div>
                              <div className="text-[11px] font-semibold">{v.stage}</div>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    {/* What to Expect Checklist */}
                    <div className="space-y-1.5 pt-1">
                      <div className="text-[11px] font-bold text-neutral-400 uppercase tracking-wider">
                        What to Expect
                      </div>
                      <div className="space-y-1.5">
                        {c.expected_checklist.map((item, cIdx) => (
                          <div
                            key={`checklist-${c.id}-${cIdx}`}
                            className="flex items-start gap-2 text-xs text-neutral-300 bg-white/[0.02] px-2.5 py-1.5 rounded-lg border border-white/5"
                          >
                            <CheckCircle2 className="w-3.5 h-3.5 text-purple-400 shrink-0 mt-0.5" />
                            <span className="leading-tight">{item}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="space-y-2 pt-5 mt-4 border-t border-white/10">
                    <div className="grid grid-cols-2 gap-2">
                      <button
                        onClick={() => handleLoadBaselineIntoEditor(c)}
                        className="py-2.5 px-3 rounded-xl bg-white/5 hover:bg-white/10 text-neutral-200 border border-white/10 hover:border-white/20 font-semibold text-xs flex items-center justify-center gap-1.5 transition cursor-pointer"
                        title="Prefill the multi-step intake form with this case's baseline measurements"
                      >
                        <Edit3 className="w-3.5 h-3.5 text-blue-400" />
                        <span>Load baseline into form</span>
                      </button>

                      <button
                        onClick={() => handlePlayNextVisit(c)}
                        disabled={playingCaseId === c.id}
                        className={`py-2.5 px-3 rounded-xl font-bold text-xs flex items-center justify-center gap-1.5 shadow-lg transition cursor-pointer ${
                          hasMoreVisits
                            ? "bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 text-white shadow-purple-600/30"
                            : "bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-600/30"
                        }`}
                        title={
                          hasMoreVisits
                            ? `Apply Visit ${nextVisitNum} (+${nextVisitMonth}m) and view twin dashboard`
                            : "Inspect final twin state"
                        }
                      >
                        {playingCaseId === c.id ? (
                          <Clock className="w-3.5 h-3.5 animate-spin" />
                        ) : (
                          <Play className="w-3.5 h-3.5 fill-current" />
                        )}
                        <span>
                          {hasMoreVisits ? `Play Visit ${nextVisitNum} (+${nextVisitMonth}m)` : "View Full Twin"}
                        </span>
                      </button>
                    </div>

                    {currentStep > 0 && (
                      <button
                        onClick={() => handleResetCase(c.id)}
                        className="w-full py-1 text-[11px] text-neutral-400 hover:text-neutral-200 flex items-center justify-center gap-1 transition"
                      >
                        <RotateCcw className="w-3 h-3" />
                        <span>Reset replay back to baseline</span>
                      </button>
                    )}
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>

        {/* SECTION 2: COHORT PERSONA TILES */}
        <div className="space-y-6 pt-6 border-t border-white/10">
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center gap-2">
                <Users className="w-4 h-4 text-purple-400" />
                <h2 className="text-lg font-bold text-white tracking-wide">
                  Standard Cohort Personas
                </h2>
              </div>
              <p className="text-xs text-neutral-400">
                Representative profiles from the ADNI cohort for quick exploration.
              </p>
            </div>
            <span className="text-xs font-mono text-neutral-400">
              {demoPatients.length} Profiles
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {demoPatients.map((p, idx) => (
              <motion.div
                key={`persona-${p.twin_id}-${idx}`}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.05, duration: 0.35 }}
                className="rounded-2xl p-6 bg-white/[0.03] backdrop-blur-xl border border-white/10 hover:border-purple-500/40 hover:bg-white/[0.05] hover:shadow-2xl hover:shadow-purple-500/10 transition-all duration-300 flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-center justify-between gap-2 mb-3">
                    <span
                      className={`text-xs font-bold px-2.5 py-0.5 rounded-full border ${getStageBadge(
                        p.baseline_stage
                      )}`}
                    >
                      {p.baseline_stage} Baseline
                    </span>

                    {p.is_synthetic ? (
                      <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/30 flex items-center gap-1">
                        <ShieldAlert className="w-3 h-3 text-amber-400" />
                        Demo data
                      </span>
                    ) : (
                      <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-300 border border-blue-500/20">
                        Real ADNI
                      </span>
                    )}
                  </div>

                  <h3 className="text-lg font-bold text-white mb-1.5">{p.persona_label}</h3>

                  <p className="text-xs text-neutral-400 leading-relaxed mb-4">
                    {p.trajectory}
                  </p>

                  <div className="flex items-center gap-1.5 text-xs text-neutral-500 mb-6">
                    <Calendar className="w-3.5 h-3.5" />
                    <span>{p.n_visits} Recorded Longitudinal Visits</span>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="grid grid-cols-2 gap-2.5 pt-4 border-t border-white/5">
                  <Link
                    href={`/twin/${encodeURIComponent(p.twin_id)}`}
                    className="py-2.5 px-3 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-semibold text-xs flex items-center justify-center gap-1.5 shadow-md shadow-purple-600/30 transition text-center cursor-pointer"
                  >
                    <Eye className="w-3.5 h-3.5" />
                    <span>Inspect Twin</span>
                  </Link>

                  <button
                    onClick={() => {
                      loadPersona(p);
                      router.push("/intake");
                    }}
                    className="py-2.5 px-3 rounded-xl bg-white/5 hover:bg-white/10 text-neutral-200 border border-white/10 hover:border-white/20 font-semibold text-xs flex items-center justify-center gap-1.5 transition cursor-pointer"
                  >
                    <Edit3 className="w-3.5 h-3.5" />
                    <span>Load into editor</span>
                  </button>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
