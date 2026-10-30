"use client";

import React, { useState } from "react";
import { HistoryEvent, TwinFull, computeCalendarDate } from "@/app/lib/api";
import {
  History,
  ArrowUpRight,
  ArrowDownRight,
  Clock,
  Calendar,
  Activity,
  Brain,
  Dna,
  Layers,
  Sparkles,
  ChevronDown,
  ChevronUp,
  FileText,
  ShieldCheck,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface TimelinePanelProps {
  twin: TwinFull;
}

interface ClinicalDriver {
  title: string;
  detail: string;
  impact: "high" | "moderate" | "protective";
}

function formatClinicalDate(dateStr?: string | null): string {
  if (!dateStr) return "";
  try {
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) return dateStr;
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
  } catch {
    return dateStr;
  }
}

function getVisitContributors(
  visit: any,
  prevVisit: any,
  twin: TwinFull
): ClinicalDriver[] {
  const drivers: ClinicalDriver[] = [];
  const s = visit?.scores || visit || {};
  const mmse = s.MMSE ?? s.MMSCORE;
  const cdg = s.CDGLOBAL ?? s.CDR;
  const adas = s.TOTSCORE ?? s.ADAS11;
  const faq = s.FAQ ?? s.FAQTOTAL;
  const prevS = prevVisit?.scores || prevVisit || {};
  const prevMmse = prevS.MMSE ?? prevS.MMSCORE;

  // 1. CDR-Global stage anchor
  if (cdg !== undefined && cdg !== null) {
    if (cdg >= 1.0) {
      drivers.push({
        title: `CDR-Global ${cdg} (Dementia Threshold)`,
        detail: "Clinical Dementia Rating ≥ 1.0 establishes loss of independent living autonomy, driving primary AD classification.",
        impact: "high",
      });
    } else if (cdg === 0.5) {
      drivers.push({
        title: `CDR-Global 0.5 (MCI Boundary)`,
        detail: "Indicates circumscribed memory deficit with preserved overall daily function; primary anchor for Mild Cognitive Impairment.",
        impact: "high",
      });
    } else if (cdg === 0.0) {
      drivers.push({
        title: `CDR-Global 0.0 (Preserved Autonomy)`,
        detail: "Complete preservation of cognitive independence and functional staging, providing heavy buffer toward Cognitively Normal.",
        impact: "protective",
      });
    }
  }

  // 2. MMSE Score & Longitudinal Velocity
  if (prevMmse !== undefined && mmse !== undefined && prevMmse !== null && mmse !== null) {
    const dMmse = mmse - prevMmse;
    if (dMmse <= -2.0) {
      drivers.push({
        title: `Rapid MMSE Decline (${dMmse.toFixed(1)} pts)`,
        detail: `MMSE dropped from ${prevMmse} to ${mmse} over ${visit.month - (prevVisit?.month ?? 0)} months, accelerating transition risk.`,
        impact: "high",
      });
    } else if (dMmse >= 1.0) {
      drivers.push({
        title: `Cognitive Stabilization (+${dMmse.toFixed(1)} pts)`,
        detail: `MMSE score stabilized from ${prevMmse} to ${mmse}, restraining conversion trajectory.`,
        impact: "protective",
      });
    }
  } else if (mmse !== undefined && mmse !== null) {
    if (mmse < 22) {
      drivers.push({
        title: `Severe Cognitive Loss (MMSE ${mmse}/30)`,
        detail: "Score falls below dementia boundary (< 22), heavily shifting diagnostic weighting toward AD.",
        impact: "high",
      });
    } else if (mmse < 27) {
      drivers.push({
        title: `Borderline Screening (MMSE ${mmse}/30)`,
        detail: "Score sits below normal cutoff (27), reflecting active episodic retrieval impairment.",
        impact: "moderate",
      });
    } else {
      drivers.push({
        title: `Preserved Mental State (MMSE ${mmse}/30)`,
        detail: "Score remains within healthy normative bounds (27-30).",
        impact: "protective",
      });
    }
  }

  // 3. ADAS11-Cog Error Burden
  if (adas !== undefined && adas !== null) {
    if (adas >= 18.0) {
      drivers.push({
        title: `Elevated ADAS11-Cog (${adas}/70)`,
        detail: "Pronounced memory recall and word comprehension errors place patient in typical AD range (18–32).",
        impact: "high",
      });
    } else if (adas >= 11.0) {
      drivers.push({
        title: `Subclinical ADAS11 Drift (${adas}/70)`,
        detail: "Intermediate error rate characteristic of early amnestic MCI pathology (8.5–17.5).",
        impact: "moderate",
      });
    }
  }

  // 4. Functional Activities (FAQ)
  if (faq !== undefined && faq !== null && faq >= 4) {
    drivers.push({
      title: `IADL Functional Loss (FAQ ${faq}/30)`,
      detail: "Impaired complex daily activities (finances, medication management) reinforces syndromic progression.",
      impact: "moderate",
    });
  }

  // 5. Biomarkers (CSF Amyloid & Tau)
  const bio = visit?.biomarkers || twin?.modalities?.biomarkers?.latest || {};
  if (bio.ABETA && bio.ABETA < 980) {
    drivers.push({
      title: `Amyloid-Beta Positivity (Aβ42 ${bio.ABETA} pg/mL)`,
      detail: "Sub-threshold CSF amyloid indicates cortical amyloid plaque aggregation.",
      impact: "high",
    });
  }
  if (bio.PTAU && bio.PTAU > 27) {
    drivers.push({
      title: `Elevated Phosphorylated Tau (${bio.PTAU} pg/mL)`,
      detail: "Reflects active neurofibrillary tangle pathology and synaptic neurodegeneration.",
      impact: "high",
    });
  }

  // 6. Neuroimaging & Hippocampal Volume
  const mri = visit?.mri_volumes || twin?.modalities?.mri_volumes?.latest || {};
  if (mri.hippocampus && mri.hippocampus < 3150) {
    drivers.push({
      title: `Medial Temporal Atrophy (Hippocampus ${mri.hippocampus} mm³)`,
      detail: "Volume loss in hippocampus indicates structural neuronal loss in CA1/subiculum.",
      impact: "high",
    });
  }

  // 7. ResNet-34 MRI Image Classifier Fusion
  if (twin?.modalities?.mri_image?.present || visit?.mri_scan_present) {
    drivers.push({
      title: "ResNet-34 2D MRI Visual Feature Fusion",
      detail: "Direct structural image classifier fused at 20% ensemble weight into final stage probability.",
      impact: "moderate",
    });
  }

  // 8. Genetics: APOE ε4 Risk Alleles
  const apoe = twin?.demographics?.apoe4 ?? twin?.demographics?.apoe4_count ?? 0;
  if (apoe > 0) {
    drivers.push({
      title: `Genetic Enrichment: ${apoe}× APOE ε4 Allele${apoe > 1 ? "s" : ""}`,
      detail: `Carrying ${apoe} copy of APOE ε4 accelerates lifelong amyloid deposition and steepens decline slope.`,
      impact: "high",
    });
  }

  return drivers.slice(0, 4);
}

export default function TimelinePanel({ twin }: TimelinePanelProps) {
  const history = [...(twin.history || [])].reverse(); // newest first
  const [expandedDetails, setExpandedDetails] = useState<Record<number, boolean>>({});

  const toggleExpand = (idx: number) => {
    setExpandedDetails((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  const formatEventName = (raw: string, month?: number) => {
    if (raw === "twin_created") return "Digital Twin Initialized (Baseline Consultation)";
    if (raw.startsWith("visit_added") || raw.startsWith("Visit M")) {
      return month !== undefined
        ? `Clinical Follow-up Visit (Month ${month})`
        : "Follow-up Visit Recorded";
    }
    if (raw === "twin_reset" || raw === "reset") return "Reset to Baseline";
    if (raw.startsWith("modality_attached:")) {
      const mod = raw.replace("modality_attached:", "").replace("_", " ");
      return `Modality Attached: ${mod.toUpperCase()}`;
    }
    return raw;
  };

  const formatRecordedAt = (item: HistoryEvent) => {
    const raw = item.recorded_at || item.timestamp;
    try {
      const d = new Date(raw);
      return d.toLocaleDateString([], {
        month: "short",
        day: "numeric",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return raw;
    }
  };

  return (
    <div className="rounded-2xl bg-neutral-900/60 backdrop-blur-xl border border-white/10 p-6 shadow-2xl">
      <div className="flex items-center justify-between mb-5">
        <div>
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <History className="w-4 h-4 text-blue-400" />
            Twin Evolution Timeline
          </h3>
          <p className="text-xs text-neutral-400 mt-0.5">
            Audit trail of clinical inputs, exam dates, and prediction contribution drivers.
          </p>
        </div>
        <span className="text-xs px-2.5 py-1 rounded-full bg-white/5 border border-white/10 text-neutral-400 font-mono">
          {history.length} Event{history.length === 1 ? "" : "s"}
        </span>
      </div>

      <div className="relative pl-6 space-y-6 before:absolute before:left-2.5 before:top-2 before:bottom-2 before:w-0.5 before:bg-gradient-to-b before:from-blue-500 before:via-purple-500/50 before:to-transparent">
        <AnimatePresence initial={false}>
          {history.map((item, idx) => {
            const hasDiff = item.prob_diff && Object.keys(item.prob_diff).length > 0;

            // Resolve visit month from item or event string
            const eventMonth = item.month !== undefined
              ? item.month
              : (item.event.match(/m(\d+)/i)?.[1]
                  ? parseInt(item.event.match(/m(\d+)/i)![1], 10)
                  : (item.event === "twin_created" ? 0 : undefined));

            // Find matching visit in twin.visits
            const visit = twin.visits?.find((v) => v.month === eventMonth)
              || (item.event === "twin_created" ? twin.visits?.[0] : undefined);

            // Find prior visit for delta velocity calculations
            const prevVisit = eventMonth !== undefined && eventMonth > 0
              ? twin.visits?.filter((v) => v.month < eventMonth).sort((a, b) => b.month - a.month)[0]
              : undefined;

            // Clinical exam date (patient consultation date) vs audit logged timestamp
            const rawExamDate = item.exam_date || visit?.exam_date || (
              twin.baseline_date && eventMonth !== undefined
                ? computeCalendarDate(twin.baseline_date, eventMonth)
                : null
            );
            const clinicalExamDate = formatClinicalDate(rawExamDate);

            // Extract clinical inputs for this visit
            const scoresObj = visit?.scores || visit || item.inputs || {};
            const mmse = scoresObj.MMSE ?? scoresObj.MMSCORE ?? item.inputs?.MMSE;
            const cdg = scoresObj.CDGLOBAL ?? scoresObj.CDR ?? item.inputs?.CDGLOBAL;
            const adas = scoresObj.TOTSCORE ?? scoresObj.ADAS11 ?? item.inputs?.TOTSCORE;
            const faq = scoresObj.FAQ ?? scoresObj.FAQTOTAL ?? item.inputs?.FAQ;
            const moca = scoresObj.MOCA ?? item.inputs?.MOCA;

            const bioObj = visit?.biomarkers || twin.modalities?.biomarkers?.latest || {};
            const mriObj = visit?.mri_volumes || twin.modalities?.mri_volumes?.latest || {};
            const hasMriScan = !!(twin.modalities?.mri_image?.present || visit?.mri_scan_present || visit?.mri_result);

            // Prediction drivers for this visit
            const contributors = getVisitContributors(visit || item.inputs, prevVisit, twin);

            const isExpanded = expandedDetails[idx] ?? true; // Default open for rich clinical insight

            return (
              <motion.div
                key={`timeline-${item.event}-${item.timestamp || item.recorded_at}-${idx}`}
                initial={{ opacity: 0, x: -15 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: idx * 0.05 }}
                className="relative group"
              >
                {/* Timeline node */}
                <span className="absolute -left-[27px] top-2.5 w-3 h-3 rounded-full bg-blue-500 border-2 border-neutral-900 group-hover:scale-125 transition" />

                <div className="p-4 rounded-xl bg-white/[0.03] border border-white/5 hover:border-white/15 transition-all space-y-3">
                  {/* Top Bar: Event Name + Clinical Exam Date + System Audit Timestamp */}
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-white/5">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-bold text-white">
                        {formatEventName(item.event, eventMonth)}
                      </span>
                      {eventMonth !== undefined && (
                        <span className="px-2 py-0.5 rounded bg-blue-500/15 border border-blue-500/30 text-blue-300 font-mono text-[10px] font-semibold">
                          M{eventMonth}
                        </span>
                      )}
                    </div>

                    <div className="flex flex-wrap items-center gap-2 text-[11px] font-mono">
                      {clinicalExamDate && (
                        <span className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 font-semibold">
                          <Calendar className="w-3 h-3 text-emerald-400" />
                          Exam: {clinicalExamDate}
                        </span>
                      )}
                      <span
                        className="flex items-center gap-1 text-neutral-400 text-[10px]"
                        title="Database audit timestamp when event was ledger-recorded into software"
                      >
                        <Clock className="w-2.5 h-2.5" />
                        (Audit: {formatRecordedAt(item)})
                      </span>
                    </div>
                  </div>

                  {/* Section 1: Values Input at this Visit */}
                  <div>
                    <div className="text-[10px] uppercase font-bold text-neutral-400 tracking-wider mb-1.5 flex items-center gap-1">
                      <FileText className="w-3 h-3 text-blue-400" />
                      Values Input at this Visit:
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {mmse !== undefined && mmse !== null && (
                        <span className="px-2 py-0.5 rounded bg-blue-500/10 border border-blue-500/20 text-blue-200 text-[11px] font-mono">
                          MMSE: <strong className="text-white">{mmse}/30</strong>
                        </span>
                      )}
                      {cdg !== undefined && cdg !== null && (
                        <span className="px-2 py-0.5 rounded bg-purple-500/10 border border-purple-500/20 text-purple-200 text-[11px] font-mono">
                          CDR: <strong className="text-white">{cdg}</strong>{" "}
                          <span className="text-[10px] text-purple-300">
                            ({cdg === 0 ? "Normal" : cdg === 0.5 ? "MCI" : "AD"})
                          </span>
                        </span>
                      )}
                      {adas !== undefined && adas !== null && (
                        <span className="px-2 py-0.5 rounded bg-amber-500/10 border border-amber-500/20 text-amber-200 text-[11px] font-mono">
                          ADAS11: <strong className="text-white">{adas}/70</strong>
                        </span>
                      )}
                      {faq !== undefined && faq !== null && (
                        <span className="px-2 py-0.5 rounded bg-neutral-800 border border-neutral-700 text-neutral-300 text-[11px] font-mono">
                          FAQ: <strong className="text-white">{faq}/30</strong>
                        </span>
                      )}
                      {moca !== undefined && moca !== null && (
                        <span className="px-2 py-0.5 rounded bg-neutral-800 border border-neutral-700 text-neutral-300 text-[11px] font-mono">
                          MoCA: <strong className="text-white">{moca}/30</strong>
                        </span>
                      )}
                      {bioObj.ABETA && (
                        <span className="px-2 py-0.5 rounded bg-cyan-950/40 border border-cyan-800/40 text-cyan-300 text-[11px] font-mono">
                          Aβ42: <strong>{bioObj.ABETA} pg/mL</strong>
                        </span>
                      )}
                      {bioObj.TAU && (
                        <span className="px-2 py-0.5 rounded bg-cyan-950/40 border border-cyan-800/40 text-cyan-300 text-[11px] font-mono">
                          Tau: <strong>{bioObj.TAU} pg/mL</strong>
                        </span>
                      )}
                      {mriObj.hippocampus && (
                        <span className="px-2 py-0.5 rounded bg-indigo-950/40 border border-indigo-800/40 text-indigo-300 text-[11px] font-mono">
                          Hippo: <strong>{mriObj.hippocampus} mm³</strong>
                        </span>
                      )}
                      {hasMriScan && (
                        <span className="px-2 py-0.5 rounded bg-emerald-950/40 border border-emerald-800/40 text-emerald-300 text-[11px] font-mono flex items-center gap-1">
                          <Layers className="w-2.5 h-2.5" />
                          MRI Slice: Processed
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Section 2: What Contributed to the Predictions (Prediction Drivers) */}
                  {contributors.length > 0 && (
                    <div className="pt-2 border-t border-white/5 space-y-1.5">
                      <div className="flex items-center justify-between text-[10px] uppercase font-bold text-neutral-400 tracking-wider">
                        <span className="flex items-center gap-1">
                          <Brain className="w-3 h-3 text-purple-400" />
                          What Contributed to the Prediction:
                        </span>
                        <button
                          onClick={() => toggleExpand(idx)}
                          className="text-[10px] text-blue-400 hover:text-blue-300 flex items-center gap-0.5 cursor-pointer lowercase"
                        >
                          {isExpanded ? (
                            <>
                              <span>hide</span>
                              <ChevronUp className="w-3 h-3" />
                            </>
                          ) : (
                            <>
                              <span>show ({contributors.length})</span>
                              <ChevronDown className="w-3 h-3" />
                            </>
                          )}
                        </button>
                      </div>

                      {isExpanded && (
                        <div className="grid grid-cols-1 gap-1.5 pt-1">
                          {contributors.map((driver, dIdx) => (
                            <div
                              key={`driver-${idx}-${dIdx}`}
                              className="p-2 rounded-lg bg-neutral-950/60 border border-white/5 flex items-start gap-2 text-xs"
                            >
                              <span
                                className={`w-1.5 h-1.5 rounded-full mt-1.5 shrink-0 ${
                                  driver.impact === "high"
                                    ? "bg-rose-400 shadow-[0_0_8px_rgba(251,113,133,0.6)]"
                                    : driver.impact === "moderate"
                                    ? "bg-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.5)]"
                                    : "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.5)]"
                                }`}
                              />
                              <div className="space-y-0.5">
                                <p className="font-semibold text-neutral-200 text-[11px]">
                                  {driver.title}
                                </p>
                                <p className="text-[10px] text-neutral-400 leading-relaxed">
                                  {driver.detail}
                                </p>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Section 3: Post-event Stage Distribution & Shift Delta */}
                  <div className="pt-2 border-t border-white/5 flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2 text-xs">
                      <span className="text-neutral-400 text-[11px]">Post-event Stage:</span>
                      <span className="font-mono text-neutral-200 text-xs font-semibold">
                        CN: {(item.stage_probs.CN * 100).toFixed(0)}% · MCI:{" "}
                        {(item.stage_probs.MCI * 100).toFixed(0)}% · AD:{" "}
                        {(item.stage_probs.AD * 100).toFixed(0)}%
                      </span>
                    </div>

                    {hasDiff && (
                      <div className="flex items-center gap-1.5">
                        <span className="text-[10px] uppercase font-semibold text-neutral-400">
                          Shift:
                        </span>
                        {(["CN", "MCI", "AD"] as const).map((cls) => {
                          const diff = Number((item.prob_diff as any)?.[cls] ?? 0);
                          if (Math.abs(diff) < 0.001) return null;
                          const isPos = diff > 0;
                          const isGoodForPatient = cls === "CN" ? isPos : !isPos;

                          return (
                            <span
                              key={`timeline-${idx}-diff-${cls}`}
                              className={`inline-flex items-center gap-0.5 px-2 py-0.5 rounded text-[10px] font-mono font-semibold border ${
                                isGoodForPatient
                                  ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                                  : "bg-rose-500/10 text-rose-400 border-rose-500/20"
                              }`}
                            >
                              {isPos ? (
                                <ArrowUpRight className="w-2.5 h-2.5" />
                              ) : (
                                <ArrowDownRight className="w-2.5 h-2.5" />
                              )}
                              {cls} {isPos ? `+${(diff * 100).toFixed(1)}%` : `${(diff * 100).toFixed(1)}%`}
                            </span>
                          );
                        })}
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </div>
  );
}
