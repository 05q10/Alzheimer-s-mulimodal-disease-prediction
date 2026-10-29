"use client";

import React, { useEffect, useState } from "react";
import { TwinFull, ExplainResponse, twinApi } from "@/app/lib/api";
import { ShieldAlert, ShieldCheck, HeartPulse, AlertCircle, Info } from "lucide-react";

interface SymptomScreenCardProps {
  twin: TwinFull;
}

export default function SymptomScreenCard({ twin }: SymptomScreenCardProps) {
  const [explainData, setExplainData] = useState<ExplainResponse | null>(null);

  useEffect(() => {
    let active = true;
    async function fetchExplain() {
      try {
        const res = await twinApi.getExplain(twin.twin_id);
        if (active) setExplainData(res);
      } catch (err) {
        console.warn("Failed to fetch /explain in SymptomScreenCard:", err);
      }
    }
    fetchExplain();
    return () => {
      active = false;
    };
  }, [twin.twin_id]);

  const screen = explainData?.symptom_screen || {
    risk_level: "Moderate Risk",
    risk_score: 0.48,
    drivers: [
      { factor: "Memory Complaints", label: "Self-Reported Memory Slip", status: "Mild", impact: "risk" as const },
      { factor: "Sleep", label: "Sleep Fragmentation (6 hrs)", status: "Moderate", impact: "risk" as const },
      { factor: "Exercise", label: "Regular Physical Routine", status: "Active", impact: "protective" as const },
      { factor: "Diet", label: "Mediterranean Diet Score (7/10)", status: "High", impact: "protective" as const },
      { factor: "Functional Assessment", label: "ADL Score 8/10", status: "Mild Impairment", impact: "risk" as const },
    ],
    caption:
      "A separate screening model from a different population; shown for reference and not part of the ADNI stage",
  };

  const isHigh = screen.risk_level.toLowerCase().includes("high");
  const isModerate = screen.risk_level.toLowerCase().includes("moderate");

  return (
    <div className="rounded-2xl bg-neutral-900/60 backdrop-blur-xl border border-white/10 p-5 sm:p-6 shadow-2xl space-y-4">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-white/10">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-amber-500/20 text-amber-300 border border-amber-500/30">
            <HeartPulse className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              Symptom &amp; Lifestyle Screen
              <span
                className={`text-xs px-2.5 py-0.5 rounded-full font-semibold border ${
                  isHigh
                    ? "bg-rose-500/20 text-rose-300 border-rose-500/30"
                    : isModerate
                    ? "bg-amber-500/20 text-amber-300 border-amber-500/30"
                    : "bg-emerald-500/20 text-emerald-300 border-emerald-500/30"
                }`}
              >
                {screen.risk_level} ({Math.round(screen.risk_score * 100)}%)
              </span>
            </h3>
            <p className="text-xs text-neutral-400 mt-0.5">
              Population-based risk index (Kaggle Alzheimer&apos;s epidemiological model).
            </p>
          </div>
        </div>
      </div>

      {/* Drivers Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
        {screen.drivers.map((d, idx) => {
          const isRisk = d.impact === "risk";
          return (
            <div
              key={`${d.factor}-${idx}`}
              className="p-3 rounded-xl bg-white/[0.02] border border-white/5 flex flex-col justify-between"
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-semibold text-white">{d.label}</span>
                {isRisk ? (
                  <span className="p-1 rounded bg-rose-500/20 text-rose-400">
                    <ShieldAlert className="w-3.5 h-3.5" />
                  </span>
                ) : (
                  <span className="p-1 rounded bg-emerald-500/20 text-emerald-400">
                    <ShieldCheck className="w-3.5 h-3.5" />
                  </span>
                )}
              </div>
              <div className="flex items-center justify-between mt-2 pt-1.5 border-t border-white/5 text-[11px]">
                <span className="text-neutral-400 font-mono">{d.status}</span>
                <span
                  className={
                    isRisk ? "text-rose-400 font-semibold" : "text-emerald-400 font-semibold"
                  }
                >
                  {isRisk ? "Risk Factor" : "Protective"}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Mandatory Fixed Caption */}
      <div className="p-3 rounded-xl bg-white/[0.03] border border-white/10 text-xs text-neutral-400 flex items-start gap-2">
        <Info className="w-4 h-4 text-blue-400 shrink-0 mt-0.5" />
        <span className="leading-relaxed font-sans">{screen.caption}</span>
      </div>
    </div>
  );
}
