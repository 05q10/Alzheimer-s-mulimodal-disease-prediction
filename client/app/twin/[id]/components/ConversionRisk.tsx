"use client";

import React from "react";
import { TwinFull } from "@/app/lib/api";
import { AlertCircle, Gauge, ArrowDownRight, Compass } from "lucide-react";
import { motion } from "framer-motion";

interface ConversionRiskProps {
  twin: TwinFull;
}

export default function ConversionRisk({ twin }: ConversionRiskProps) {
  const { conversion_risk, rate_of_decline, stage } = twin.state;

  const risk12m = conversion_risk?.["12m"] ?? 0;
  const risk24m = conversion_risk?.["24m"] ?? 0;

  const mmsePerYear = rate_of_decline?.MMSE_per_year ?? -1.8;
  const percentile = rate_of_decline?.vs_cohort_percentile ?? 65;

  const getRiskLevel = (prob: number) => {
    if (prob < 0.25) return { label: "Low Risk", color: "text-emerald-400", stroke: "#10b981", bg: "bg-emerald-500/10" };
    if (prob < 0.5) return { label: "Moderate Risk", color: "text-amber-400", stroke: "#f59e0b", bg: "bg-amber-500/10" };
    return { label: "High Progression Risk", color: "text-rose-400", stroke: "#f43f5e", bg: "bg-rose-500/10" };
  };

  const riskLevel12 = getRiskLevel(risk12m);
  const riskLevel24 = getRiskLevel(risk24m);

  return (
    <div className="rounded-2xl bg-neutral-900/60 backdrop-blur-xl border border-white/10 p-6 shadow-2xl">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <Gauge className="w-4 h-4 text-purple-400" />
            Progression & Conversion Risk
          </h3>
          <p className="text-xs text-neutral-400 mt-0.5">
            Model-estimated risk of cognitive worsening or diagnostic conversion.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* 12-Month Gauge */}
        <div className="rounded-xl bg-white/[0.03] border border-white/5 p-4 flex flex-col items-center text-center relative overflow-hidden">
          <span className="text-xs font-semibold text-neutral-300 uppercase tracking-wider mb-2">
            12-Month Conversion Risk
          </span>

          {/* Circular SVG Gauge */}
          <div className="relative w-32 h-32 flex items-center justify-center my-2">
            <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
              <circle
                cx="50"
                cy="50"
                r="40"
                fill="transparent"
                stroke="#262626"
                strokeWidth="9"
              />
              <motion.circle
                cx="50"
                cy="50"
                r="40"
                fill="transparent"
                stroke={riskLevel12.stroke}
                strokeWidth="9"
                strokeDasharray={251.2}
                initial={{ strokeDashoffset: 251.2 }}
                animate={{ strokeDashoffset: 251.2 - 251.2 * risk12m }}
                transition={{ duration: 1, ease: "easeOut" }}
                strokeLinecap="round"
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-2xl font-bold font-mono text-white">
                {(risk12m * 100).toFixed(0)}%
              </span>
              <span className={`text-[10px] font-semibold ${riskLevel12.color}`}>
                {riskLevel12.label}
              </span>
            </div>
          </div>

          <p className="text-[11px] text-neutral-400 mt-1 max-w-[200px]">
            {stage === "CN"
              ? "Probability of progression to MCI within 12 months."
              : stage === "MCI"
              ? "Probability of conversion to Alzheimer's Dementia."
              : "Probability of advanced functional decline."}
          </p>
        </div>

        {/* 24-Month Gauge */}
        <div className="rounded-xl bg-white/[0.03] border border-white/5 p-4 flex flex-col items-center text-center relative overflow-hidden">
          <span className="text-xs font-semibold text-neutral-300 uppercase tracking-wider mb-2">
            24-Month Cumulative Risk
          </span>

          {/* Circular SVG Gauge */}
          <div className="relative w-32 h-32 flex items-center justify-center my-2">
            <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
              <circle
                cx="50"
                cy="50"
                r="40"
                fill="transparent"
                stroke="#262626"
                strokeWidth="9"
              />
              <motion.circle
                cx="50"
                cy="50"
                r="40"
                fill="transparent"
                stroke={riskLevel24.stroke}
                strokeWidth="9"
                strokeDasharray={251.2}
                initial={{ strokeDashoffset: 251.2 }}
                animate={{ strokeDashoffset: 251.2 - 251.2 * risk24m }}
                transition={{ duration: 1.2, ease: "easeOut" }}
                strokeLinecap="round"
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-2xl font-bold font-mono text-white">
                {(risk24m * 100).toFixed(0)}%
              </span>
              <span className={`text-[10px] font-semibold ${riskLevel24.color}`}>
                {riskLevel24.label}
              </span>
            </div>
          </div>

          <p className="text-[11px] text-neutral-400 mt-1 max-w-[200px]">
            Two-year horizon incorporates expected cognitive score decay.
          </p>
        </div>
      </div>

      {/* Decline Velocity & Cohort Comparison */}
      <div className="mt-4 pt-4 border-t border-white/10 grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div className="flex items-center gap-3 p-3 rounded-xl bg-white/[0.02] border border-white/5">
          <div className="p-2 rounded-lg bg-purple-500/10 text-purple-400">
            <ArrowDownRight className="w-4 h-4" />
          </div>
          <div>
            <span className="text-[11px] text-neutral-400 uppercase tracking-wide block">
              Cognitive Velocity
            </span>
            <span className="text-sm font-bold font-mono text-white">
              {mmsePerYear > 0 ? `+${mmsePerYear}` : mmsePerYear} MMSE / yr
            </span>
          </div>
        </div>

        <div className="flex items-center gap-3 p-3 rounded-xl bg-white/[0.02] border border-white/5">
          <div className="p-2 rounded-lg bg-blue-500/10 text-blue-400">
            <Compass className="w-4 h-4" />
          </div>
          <div>
            <span className="text-[11px] text-neutral-400 uppercase tracking-wide block">
              Decline vs ADNI Cohort
            </span>
            <span className="text-sm font-bold font-mono text-white">
              {percentile}th Percentile (Faster decline)
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
