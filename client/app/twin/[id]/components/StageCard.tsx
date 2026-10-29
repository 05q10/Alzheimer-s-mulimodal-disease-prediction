"use client";

import React from "react";
import { motion } from "framer-motion";
import { StageProbs, TwinFull } from "@/app/lib/api";
import { AlertTriangle, ShieldCheck, Cpu, Sparkles } from "lucide-react";

interface StageCardProps {
  twin: TwinFull;
}

export default function StageCard({ twin }: StageCardProps) {
  const { state, is_synthetic, persona_label, modalities } = twin;
  const { stage, stage_probs, confidence, tier_used } = state;

  const stageColors: Record<string, { bg: string; text: string; border: string; glow: string }> = {
    CN: {
      bg: "bg-emerald-500/15",
      text: "text-emerald-400",
      border: "border-emerald-500/30",
      glow: "shadow-emerald-500/20",
    },
    MCI: {
      bg: "bg-amber-500/15",
      text: "text-amber-400",
      border: "border-amber-500/30",
      glow: "shadow-amber-500/20",
    },
    AD: {
      bg: "bg-rose-500/15",
      text: "text-rose-400",
      border: "border-rose-500/30",
      glow: "shadow-rose-500/20",
    },
  };

  const currentTheme = stageColors[stage] || stageColors.MCI;

  const stageDescriptions: Record<string, string> = {
    CN: "Cognitively Normal — Stable cognitive profile without measurable impairment.",
    MCI: "Mild Cognitive Impairment — Detectable deficit with preserved independence.",
    AD: "Alzheimer's Disease — Progressive neurodegenerative impairment requiring care.",
  };

  const tierLabels: Record<string, string> = {
    T1: "T1: Basic Demographics & Cognitive Scores",
    T2: "T2: Detailed Neuropsychological Batteries",
    T3: "T3: CSF & Plasma Fluid Biomarkers (Aβ/Tau)",
    T4: "T4: Structural MRI Volumetric Features",
  };

  const hasMriImage = !!(modalities?.mri_image?.present || twin.visits?.some((v) => v.mri_scan_present || v.mri_result));
  const mriImageLatest = modalities?.mri_image?.latest || twin.visits?.find((v) => v.mri_result)?.mri_result;

  return (
    <div className="relative rounded-2xl bg-[#0e1e3b]/70 backdrop-blur-xl border border-white/10 p-6 shadow-2xl overflow-hidden">
      {/* Background ambient lighting */}
      <div
        className={`absolute -top-24 -right-24 w-64 h-64 rounded-full blur-3xl opacity-20 pointer-events-none ${
          stage === "CN"
            ? "bg-emerald-500"
            : stage === "MCI"
            ? "bg-amber-500"
            : "bg-rose-500"
        }`}
      />

      {/* Header Badges */}
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div className="flex items-center gap-2">
          <span className="text-xs uppercase tracking-wider font-semibold text-neutral-400 flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-blue-400" />
            Digital Twin Staging
          </span>
          {persona_label && (
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-white/10 text-neutral-300 border border-white/10 font-mono">
              {persona_label}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* Tier badge */}
          <span
            className="text-xs font-medium px-2.5 py-1 rounded-full bg-blue-500/20 border border-blue-500/30 text-blue-300 flex items-center gap-1"
            title={tierLabels[tier_used] || tier_used}
          >
            <Sparkles className="w-3 h-3" />
            {tier_used}
          </span>

          {/* Synthetic Demo Data Badge */}
          {is_synthetic && (
            <span
              id="demo-data-badge"
              className="text-xs font-semibold px-2.5 py-1 rounded-full bg-amber-500/20 border border-amber-500/40 text-amber-300 flex items-center gap-1 shadow-sm shadow-amber-500/20"
            >
              <AlertTriangle className="w-3 h-3 text-amber-400" />
              Demo data
            </span>
          )}
        </div>
      </div>

      {/* Main Stage Display */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 py-2">
        <div>
          <div className="flex items-baseline gap-3">
            <h2
              className={`text-4xl md:text-5xl font-black tracking-tight ${currentTheme.text}`}
            >
              {stage === "CN"
                ? "Cognitively Normal (CN)"
                : stage === "MCI"
                ? "Mild Cognitive Impairment (MCI)"
                : "Alzheimer's Dementia (AD)"}
            </h2>
          </div>
          <p className="text-sm text-neutral-300 mt-2 max-w-xl">
            {stageDescriptions[stage] || ""}
          </p>
        </div>

        <div className="flex flex-col items-start md:items-end justify-center min-w-[170px]">
          <span
            className="text-xs text-neutral-400 uppercase tracking-wider cursor-help border-b border-dotted border-neutral-500"
            title="calibrated probability, not accuracy"
          >
            Probability of this stage
          </span>
          <div className="flex items-center gap-1.5 mt-1" title="calibrated probability, not accuracy">
            <ShieldCheck className="w-5 h-5 text-blue-400" />
            <span className="text-3xl font-mono font-bold text-white">
              {(confidence * 100).toFixed(1)}%
            </span>
          </div>
          <span className="text-[11px] text-neutral-500 mt-0.5">
            Calibrated posterior probability
          </span>
        </div>
      </div>

      {/* Probability Distribution Bars */}
      <div className="mt-6 pt-5 border-t border-white/10">
        <div className="flex justify-between items-center mb-2">
          <span className="text-xs font-semibold text-neutral-300 uppercase tracking-wider">
            Calibrated Class Probabilities
          </span>
          <span className="text-xs text-neutral-500 font-mono">Sum = 1.00</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {(["CN", "MCI", "AD"] as const).map((cls) => {
            const prob = stage_probs[cls] ?? 0;
            const pct = (prob * 100).toFixed(1);
            const isSelected = stage === cls;
            const barColors = {
              CN: "from-emerald-500 to-teal-400",
              MCI: "from-amber-500 to-orange-400",
              AD: "from-rose-500 to-pink-500",
            }[cls];

            return (
              <div
                key={cls}
                className={`p-3 rounded-xl border transition-all ${
                  isSelected
                    ? "bg-white/10 border-blue-400/40 shadow-md shadow-blue-500/10"
                    : "bg-white/5 border-white/5 opacity-80"
                }`}
              >
                <div className="flex justify-between items-center text-sm mb-1.5">
                  <span className="font-semibold text-neutral-200">{cls}</span>
                  <span className="font-mono font-bold text-white">{pct}%</span>
                </div>
                <div className="h-2 w-full bg-neutral-800 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min(100, Math.max(2, prob * 100))}%` }}
                    transition={{ duration: 0.8, ease: "easeOut" }}
                    className={`h-full rounded-full bg-gradient-to-r ${barColors}`}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
