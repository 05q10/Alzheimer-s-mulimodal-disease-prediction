"use client";

import React, { useEffect, useState, useMemo } from "react";
import { TwinFull, ExplainResponse, twinApi } from "@/app/lib/api";
import {
  HelpCircle,
  TrendingDown,
  TrendingUp,
  Brain,
  Info,
  Layers,
  Sparkles,
  ToggleLeft,
  ToggleRight,
  Activity,
  Dna,
  User,
} from "lucide-react";

interface WhyThisStagePanelProps {
  twin: TwinFull;
}

export default function WhyThisStagePanel({ twin }: WhyThisStagePanelProps) {
  const [explainData, setExplainData] = useState<ExplainResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeExcludedModality, setActiveExcludedModality] = useState<string | null>(null);

  // Refetch explain data when twin ID or twin state changes
  useEffect(() => {
    let active = true;
    async function loadExplain() {
      setLoading(true);
      try {
        const res = await twinApi.getExplain(twin.twin_id);
        if (active) {
          setExplainData(res);
        }
      } catch (err) {
        console.warn("Failed to load /explain:", err);
      } finally {
        if (active) setLoading(false);
      }
    }
    loadExplain();
    return () => {
      active = false;
    };
  }, [twin.twin_id, twin.state.stage_probs, twin.visits.length]);

  // Group drivers by modality
  const groupedDrivers = useMemo(() => {
    if (!explainData?.stage?.drivers) return {};
    const groups: Record<string, typeof explainData.stage.drivers> = {};
    for (const d of explainData.stage.drivers) {
      const mod = d.modality || "cognitive";
      if (!groups[mod]) groups[mod] = [];
      groups[mod].push(d);
    }
    return groups;
  }, [explainData]);

  const modalityIcons: Record<string, React.ElementType> = {
    cognitive: Activity,
    biomarkers: Dna,
    mri: Layers,
    demographics: User,
    lifestyle: Sparkles,
  };

  if (!explainData && loading) {
    return (
      <div className="rounded-2xl bg-neutral-900/60 backdrop-blur-xl border border-white/10 p-6 shadow-2xl flex items-center justify-center py-12">
        <div className="flex items-center gap-3 text-neutral-400 text-sm">
          <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
          <span>Synthesizing clinical explainability models...</span>
        </div>
      </div>
    );
  }

  if (!explainData) return null;

  const currentStage = explainData.stage.primary || twin.state.stage;
  const leaveOneOut = explainData.stage.leave_one_out || [];

  return (
    <div className="rounded-2xl bg-neutral-900/60 backdrop-blur-xl border border-white/10 p-5 sm:p-6 shadow-2xl space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-white/10">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-purple-500/20 text-purple-300 border border-purple-500/30">
            <Brain className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              Why This Stage?
              <span className="text-xs px-2.5 py-0.5 rounded-full bg-purple-500/20 text-purple-300 border border-purple-500/30 font-semibold">
                Classified: {currentStage}
              </span>
            </h3>
            <p className="text-xs text-neutral-400 mt-0.5">
              Explainable feature attribution (GBM + SHAP) & modality counterfactuals.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs text-neutral-400">
          <Info className="w-3.5 h-3.5 text-blue-400" />
          <span>Grouped by clinical modality</span>
        </div>
      </div>

      {/* Narrative Card */}
      {explainData.narrative && (
        <div className="p-4 rounded-xl bg-purple-950/20 border border-purple-500/25 relative overflow-hidden">
          <div className="flex items-start gap-3">
            <div className="p-1.5 rounded-lg bg-purple-500/20 text-purple-300 shrink-0 mt-0.5">
              <Sparkles className="w-4 h-4" />
            </div>
            <div>
              <h4 className="text-xs font-bold text-purple-200 uppercase tracking-wider mb-1">
                Clinical Staging Summary
              </h4>
              <p className="text-xs text-neutral-300 leading-relaxed font-sans">
                {explainData.narrative}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Diverging Bars Section */}
      <div>
        <div className="flex items-center justify-between text-xs mb-3 font-semibold px-2">
          <span className="text-emerald-400 flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            ← Favours Cognitively Normal (CN)
          </span>
          <span className="text-neutral-500 text-[11px]">Diverging Attribution</span>
          <span className="text-purple-400 flex items-center gap-1">
            Favours Alzheimer&apos;s Dementia (AD) →
            <span className="w-2 h-2 rounded-full bg-purple-400" />
          </span>
        </div>

        {/* Groups */}
        <div className="space-y-4">
          {Object.entries(groupedDrivers).map(([modality, drivers], groupIdx) => {
            const ModIcon = modalityIcons[modality] || Activity;
            return (
              <div
                key={`group-${modality}-${groupIdx}`}
                className="p-3.5 rounded-xl bg-white/[0.02] border border-white/5 space-y-2.5"
              >
                <div className="flex items-center gap-2 text-xs font-semibold text-neutral-300 uppercase tracking-wider">
                  <ModIcon className="w-3.5 h-3.5 text-blue-400" />
                  <span>{modality}</span>
                </div>

                <div className="space-y-2">
                  {drivers.map((d, dIdx) => {
                    const isTowardCn = d.direction === "CN";
                    const barWidth = Math.round(d.weight * 100);

                    return (
                      <div key={`driver-${modality}-${d.feature}-${dIdx}`} className="space-y-1">
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-neutral-300 font-medium">{d.label}</span>
                          <span className="text-[11px] text-neutral-400">
                            {d.description}
                          </span>
                        </div>

                        {/* Diverging bar track */}
                        <div className="h-3 w-full bg-white/5 rounded-full overflow-hidden flex relative">
                          {/* Left Half (CN) */}
                          <div className="w-1/2 h-full flex justify-end pr-0.5 border-r border-white/20">
                            {isTowardCn && (
                              <div
                                className="h-full bg-gradient-to-l from-emerald-400 to-emerald-600 rounded-l-full transition-all duration-500"
                                style={{ width: `${barWidth}%` }}
                              />
                            )}
                          </div>

                          {/* Right Half (AD) */}
                          <div className="w-1/2 h-full flex justify-start pl-0.5">
                            {!isTowardCn && (
                              <div
                                className="h-full bg-gradient-to-r from-purple-500 to-rose-500 rounded-r-full transition-all duration-500"
                                style={{ width: `${barWidth}%` }}
                              />
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Leave-One-Out Modality Toggles */}
      {leaveOneOut.length > 0 && (
        <div className="pt-2">
          <h4 className="text-xs font-bold text-neutral-300 uppercase tracking-wider mb-2.5 flex items-center gap-2">
            <HelpCircle className="w-3.5 h-3.5 text-indigo-400" />
            Without This Modality (Counterfactual Impact)
          </h4>
          <p className="text-xs text-neutral-400 mb-3">
            Examine how the stager&apos;s diagnosis re-evaluates if specific diagnostic layers are withheld:
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
            {leaveOneOut.map((loo, looIdx) => {
              const isSelected = activeExcludedModality === loo.modality;
              return (
                <button
                  key={`loo-${loo.modality}-${looIdx}`}
                  onClick={() =>
                    setActiveExcludedModality(isSelected ? null : loo.modality)
                  }
                  className={`p-3 rounded-xl border text-left transition cursor-pointer flex flex-col justify-between ${
                    isSelected
                      ? "bg-indigo-950/50 border-indigo-500/50 shadow-md shadow-indigo-500/10"
                      : "bg-white/[0.03] border-white/10 hover:border-white/20 hover:bg-white/[0.05]"
                  }`}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-xs font-bold text-white">{loo.label}</span>
                    {isSelected ? (
                      <ToggleRight className="w-4 h-4 text-indigo-400" />
                    ) : (
                      <ToggleLeft className="w-4 h-4 text-neutral-500" />
                    )}
                  </div>
                  <p className="text-[11px] text-neutral-300 font-mono">
                    {loo.summary}
                  </p>
                  <div className="mt-2 pt-1.5 border-t border-white/5 flex gap-2 text-[10px] text-neutral-400">
                    <span>CN {Math.round(loo.without_probs.CN * 100)}%</span>
                    <span>MCI {Math.round(loo.without_probs.MCI * 100)}%</span>
                    <span>AD {Math.round(loo.without_probs.AD * 100)}%</span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Forecast Drivers Block */}
      {explainData.forecast?.drivers && explainData.forecast.drivers.length > 0 && (
        <div className="pt-2 border-t border-white/10">
          <h4 className="text-xs font-bold text-neutral-300 uppercase tracking-wider mb-2.5 flex items-center gap-2">
            <TrendingUp className="w-3.5 h-3.5 text-blue-400" />
            Forecast Drivers (Longitudinal Trajectory Modifiers)
          </h4>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {explainData.forecast.drivers.map((fd, fdIdx) => {
              const isAccelerating = fd.effect === "accelerating";
              return (
                <div
                  key={`forecast-driver-${fd.feature}-${fdIdx}`}
                  className="p-3 rounded-xl bg-white/[0.02] border border-white/5 flex flex-col justify-between"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-semibold text-white">{fd.label}</span>
                    <span
                      className={`text-[10px] font-bold px-2 py-0.5 rounded flex items-center gap-1 ${
                        isAccelerating
                          ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                          : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                      }`}
                    >
                      {isAccelerating ? (
                        <TrendingUp className="w-3 h-3" />
                      ) : (
                        <TrendingDown className="w-3 h-3" />
                      )}
                      {isAccelerating ? "Faster Decline" : "Slower Decline"}
                    </span>
                  </div>
                  <p className="text-[11px] text-neutral-400 leading-relaxed mt-1">
                    {fd.impact}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
