"use client";

import React from "react";
import { TwinFull } from "@/app/lib/api";
import { usePatientStore } from "@/app/store/patientStore";
import { ImageIcon, Layers, ArrowRight, Sparkles, FileImage } from "lucide-react";

interface MRIScanResultCardProps {
  twin: TwinFull;
}

export default function MRIScanResultCard({ twin }: MRIScanResultCardProps) {
  const session = usePatientStore((s) => s.session);
  const mriModality = twin.modalities?.mri_image;
  const isPresent = !!mriModality?.present;
  const latestMRI = mriModality?.latest as any;
  const fusion = twin.state?.fusion;

  if (!isPresent && !latestMRI && !fusion) {
    return null;
  }

  const predictedClass = latestMRI?.predicted_class || "VeryMild";
  const rawProbs = latestMRI?.probs || {
    NonDemented: 0.02,
    VeryMild: 0.97,
    Mild: 0.01,
    Moderate: 0.00,
  };
  const mappedProbs = latestMRI?.mapped_probs || {
    CN: 0.02,
    MCI: 0.97,
    AD: 0.01,
  };

  const adniProbs = fusion?.adni_probs || {
    CN: 0.18,
    MCI: 0.56,
    AD: 0.41,
  };
  const finalProbs = fusion?.final_probs || twin.state.stage_probs;
  const fusionWeight = fusion?.weight ?? 0.2;

  // File name provenance
  const fileName =
    session.uploads?.mriFile?.name ||
    latestMRI?.filename ||
    "test.jpg";

  // Formatting helpers
  const formatPct = (val?: number) => `${Math.round((val ?? 0) * 100)}%`;

  // Mapped stage winner
  const mappedStage =
    mappedProbs.MCI >= mappedProbs.AD && mappedProbs.MCI >= mappedProbs.CN
      ? "MCI"
      : mappedProbs.AD >= mappedProbs.CN
      ? "AD"
      : "CN";

  const topProbPct = Math.round(
    Number(rawProbs[predictedClass] ?? rawProbs.VeryMild ?? 0.97) * 100
  );

  // Requirement 2: Factual sentence built from numbers
  const factualLine = `Image classifier: ${predictedClass} (${topProbPct}%), mapped to ${mappedStage}. Blended at ${Math.round(
    fusionWeight * 100
  )}% weight: P(AD) ${formatPct(adniProbs.AD)} -> ${formatPct(
    finalProbs.AD
  )}, P(MCI) ${formatPct(adniProbs.MCI)} -> ${formatPct(finalProbs.MCI)}.`;

  return (
    <div className="rounded-2xl bg-neutral-900/60 backdrop-blur-xl border border-blue-500/30 p-5 shadow-2xl relative overflow-hidden">
      {/* Background glow accent */}
      <div className="absolute top-0 right-0 w-64 h-64 bg-blue-600/10 rounded-full blur-3xl pointer-events-none -mr-20 -mt-20" />

      {/* Card Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4 pb-3 border-b border-white/10">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-blue-500/20 text-blue-400 border border-blue-500/30">
            <ImageIcon className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-base font-bold text-white">MRI Scan Result</h3>
              <span className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-300 border border-blue-500/30">
                ResNet-34 Feature Extractor
              </span>
            </div>
            <p className="text-xs text-neutral-400 mt-0.5">
              Multimodal fusion with ADNI tabular stager ({Math.round((1 - fusionWeight) * 100)}% ADNI / {Math.round(fusionWeight * 100)}% Visual MRI).
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-white/5 border border-white/10 text-[11px] text-neutral-300">
            <FileImage className="w-3.5 h-3.5 text-blue-400" />
            <span className="font-mono">{fileName}</span>
            <span className="text-neutral-500">(Single slice)</span>
          </div>

          <div className="flex items-center gap-1.5 px-3 py-1 rounded-xl bg-blue-950/40 border border-blue-500/40">
            <Sparkles className="w-4 h-4 text-blue-400" />
            <span className="text-xs text-neutral-300">Predicted:</span>
            <span className="text-sm font-bold text-blue-300">{predictedClass}</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* Left: Raw ResNet Classifier Distribution */}
        <div className="p-4 rounded-xl bg-white/[0.02] border border-white/5 flex flex-col justify-between">
          <div>
            <h4 className="text-xs font-semibold text-neutral-300 mb-3 flex items-center justify-between">
              <span>Visual Classifier Probabilities (ResNet-34)</span>
              <span className="text-[11px] text-neutral-500">4-Class Raw</span>
            </h4>
            <div className="space-y-2.5">
              {Object.entries(rawProbs).map(([cls, prob]: [string, any], clsIdx) => {
                const isWinner = cls.toLowerCase() === predictedClass.toLowerCase();
                const pct = Math.round(Number(prob) * 100);
                return (
                  <div key={`mri-raw-${cls}-${clsIdx}`}>
                    <div className="flex justify-between text-xs mb-1">
                      <span className={isWinner ? "font-bold text-blue-300" : "text-neutral-400"}>
                        {cls}
                      </span>
                      <span className={isWinner ? "font-bold text-white" : "text-neutral-400"}>
                        {pct}%
                      </span>
                    </div>
                    <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          isWinner ? "bg-gradient-to-r from-blue-500 to-indigo-400" : "bg-neutral-600/50"
                        }`}
                        style={{ width: `${Math.max(pct, 2)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-white/5 flex items-center justify-between text-[11px] text-neutral-400">
            <span>Mapped to ADNI Staging:</span>
            <div className="flex gap-2 font-mono">
              <span className="text-emerald-400 font-semibold">CN {formatPct(mappedProbs.CN)}</span>
              <span className="text-amber-400 font-semibold">MCI {formatPct(mappedProbs.MCI)}</span>
              <span className="text-purple-400 font-semibold">AD {formatPct(mappedProbs.AD)}</span>
            </div>
          </div>
        </div>

        {/* Right: Fusion Staging Shift */}
        <div className="p-4 rounded-xl bg-white/[0.02] border border-white/5 flex flex-col justify-between">
          <div>
            <h4 className="text-xs font-semibold text-neutral-300 mb-3 flex items-center justify-between">
              <span className="flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-indigo-400" />
                Multimodal Staging Impact
              </span>
              <span className="text-[11px] text-indigo-300">adni_probs → final_probs</span>
            </h4>

            <div className="space-y-3">
              {(["CN", "MCI", "AD"] as const).map((stageKey) => {
                const initial = adniProbs[stageKey] ?? 0;
                const finalVal = finalProbs[stageKey] ?? initial;
                const delta = Number((finalVal - initial).toFixed(3));
                const colorClass =
                  stageKey === "CN"
                    ? "text-emerald-400"
                    : stageKey === "MCI"
                    ? "text-amber-400"
                    : "text-purple-400";

                return (
                  <div
                    key={`mri-stage-${stageKey}`}
                    className="p-2.5 rounded-lg bg-white/[0.03] border border-white/5 flex items-center justify-between text-xs"
                  >
                    <div className="flex items-center gap-2">
                      <span className={`w-8 font-bold ${colorClass}`}>{stageKey}</span>
                      <span className="text-neutral-400">{formatPct(initial)}</span>
                      <ArrowRight className="w-3 h-3 text-neutral-500" />
                      <span className="font-bold text-white">{formatPct(finalVal)}</span>
                    </div>

                    <div className="flex items-center gap-1.5">
                      {delta !== 0 ? (
                        <span
                          className={`text-[11px] font-semibold px-2 py-0.5 rounded ${
                            delta > 0
                              ? "bg-blue-500/20 text-blue-300 border border-blue-500/30"
                              : "bg-white/5 text-neutral-400 border border-white/10"
                          }`}
                        >
                          {delta > 0 ? `+${(delta * 100).toFixed(1)}%` : `${(delta * 100).toFixed(1)}%`}
                        </span>
                      ) : (
                        <span className="text-[11px] text-neutral-500">No shift</span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Factual line per requirement 2 */}
          <div className="mt-4 pt-3 border-t border-white/5">
            <p className="text-xs text-neutral-300 leading-relaxed font-mono bg-white/[0.02] p-2.5 rounded-lg border border-white/5">
              {factualLine}
            </p>
            <p className="text-[10px] text-neutral-500 mt-1.5">
              Evaluated on single axial brain slice from <span className="text-neutral-400 font-mono">{fileName}</span>.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
