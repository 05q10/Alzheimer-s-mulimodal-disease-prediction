"use client";

import React, { useEffect, useState } from "react";
import { TwinFull, ExplainResponse, twinApi } from "@/app/lib/api";
import { Sparkles, Info, TrendingUp, TrendingDown, Check } from "lucide-react";

interface LifestyleModifierCardProps {
  twin: TwinFull;
}

export default function LifestyleModifierCard({ twin }: LifestyleModifierCardProps) {
  const [explainData, setExplainData] = useState<ExplainResponse | null>(null);

  useEffect(() => {
    let active = true;
    async function fetchExplain() {
      try {
        const res = await twinApi.getExplain(twin.twin_id);
        if (active) setExplainData(res);
      } catch (err) {
        console.warn("Failed to fetch /explain in LifestyleModifierCard:", err);
      }
    }
    fetchExplain();
    return () => {
      active = false;
    };
  }, [twin.twin_id]);

  const mod = explainData?.lifestyle_modifier || {
    multipliers: [
      {
        factor: "alcohol",
        label: "Alcohol Consumption",
        value: "2.4 units/week",
        multiplier: 1.02,
        description: "Mild neurovascular effect within expected bounds.",
      },
      {
        factor: "physicalActivity",
        label: "Physical Activity",
        value: "4.0 hrs/week",
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
  };

  const total = mod.total_multiplier ?? 1.0;

  return (
    <div className="rounded-2xl bg-neutral-900/60 backdrop-blur-xl border border-white/10 p-5 sm:p-6 shadow-2xl space-y-4">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-white/10">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-xl bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              Lifestyle &amp; Risk Factors
              <span className="text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-semibold font-mono">
                Total Multiplier: {total.toFixed(2)}x
              </span>
            </h3>
            <p className="text-xs text-neutral-400 mt-0.5">
              Adjusts progression velocity &amp; conversion risk curves (does not alter baseline stage).
            </p>
          </div>
        </div>
      </div>

      {/* Multiplier Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
        {mod.multipliers.map((m) => {
          const isAccelerating = m.multiplier > 1.0;
          const isNeutral = m.multiplier === 1.0;

          return (
            <div
              key={m.factor}
              className="p-3.5 rounded-xl bg-white/[0.02] border border-white/5 flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-semibold text-white">{m.label}</span>
                  <span
                    className={`text-xs font-bold font-mono px-2 py-0.5 rounded flex items-center gap-1 ${
                      isNeutral
                        ? "bg-white/10 text-neutral-300 border border-white/10"
                        : isAccelerating
                        ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                        : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                    }`}
                  >
                    {isAccelerating ? (
                      <TrendingUp className="w-3 h-3" />
                    ) : isNeutral ? (
                      <Check className="w-3 h-3" />
                    ) : (
                      <TrendingDown className="w-3 h-3" />
                    )}
                    {m.multiplier.toFixed(2)}x
                  </span>
                </div>
                <p className="text-[11px] text-neutral-300 font-mono mt-1">
                  Value: {String(m.value)}
                </p>
              </div>

              <p className="text-[11px] text-neutral-400 leading-relaxed mt-2 pt-2 border-t border-white/5">
                {m.description}
              </p>
            </div>
          );
        })}
      </div>

      {/* Mandatory Fixed Caption */}
      <div className="p-3 rounded-xl bg-white/[0.03] border border-white/10 text-xs text-neutral-400 flex items-start gap-2">
        <Info className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
        <span className="leading-relaxed font-sans">{mod.caption}</span>
      </div>
    </div>
  );
}
