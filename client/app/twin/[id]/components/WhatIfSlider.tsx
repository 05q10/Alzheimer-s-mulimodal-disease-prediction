"use client";

import React, { useState, useEffect } from "react";
import { TwinFull, twinApi, ForecastPoint } from "@/app/lib/api";
import { Sliders, RefreshCw, AlertTriangle, Sparkles, Info } from "lucide-react";

interface WhatIfSliderProps {
  twin: TwinFull;
  onSimulationUpdated: (
    simForecast: Record<string, ForecastPoint> | null,
    multiplier: number
  ) => void;
}

export default function WhatIfSlider({ twin, onSimulationUpdated }: WhatIfSliderProps) {
  const [multiplier, setMultiplier] = useState<number>(1.0);
  const [loading, setLoading] = useState(false);
  const [disclaimerText, setDisclaimerText] = useState<string>(
    "Illustrative simulation only. Trajectory is scaled from the model's central prediction by the chosen multiplier. This is NOT a clinical prediction and does not account for individual biological variability, treatment effects, or model uncertainty beyond the shown bands."
  );

  const handleSliderChange = async (newVal: number) => {
    setMultiplier(newVal);
    if (newVal === 1.0) {
      onSimulationUpdated(null, 1.0);
      return;
    }

    try {
      setLoading(true);
      const res = await twinApi.simulate(twin.twin_id, newVal);
      if (res.disclaimer) {
        setDisclaimerText(res.disclaimer);
      }
      onSimulationUpdated(res.simulated_forecast, newVal);
    } catch (err) {
      console.error("Simulation failed:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setMultiplier(1.0);
    onSimulationUpdated(null, 1.0);
  };

  const getMultiplierLabel = (val: number) => {
    if (val === 1.0) return "1.0x — As Predicted";
    if (val < 1.0) return `${val}x — Slowed Decline (-${Math.round((1 - val) * 100)}%)`;
    return `${val}x — Accelerated Decline (+${Math.round((val - 1) * 100)}%)`;
  };

  return (
    <div className="rounded-2xl bg-neutral-900/60 backdrop-blur-xl border border-white/10 p-6 shadow-2xl">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div>
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-purple-400" />
            <h3 className="text-base font-bold text-white">
              What-If Decline Velocity Simulator
            </h3>
          </div>
          <p className="text-xs text-purple-300/80 font-medium mt-0.5">
            Illustrative simulation, not a clinical prediction
          </p>
        </div>

        <div className="flex items-center gap-2">
          {multiplier !== 1.0 && (
            <button
              onClick={handleReset}
              className="text-xs px-2.5 py-1 rounded-lg bg-white/5 hover:bg-white/10 text-neutral-300 border border-white/10 flex items-center gap-1 transition cursor-pointer"
            >
              <RefreshCw className="w-3 h-3" />
              Reset to 1.0x
            </button>
          )}
          <span className="text-xs font-mono font-bold px-3 py-1 rounded-full bg-purple-500/20 text-purple-300 border border-purple-500/30">
            {getMultiplierLabel(multiplier)}
          </span>
        </div>
      </div>

      {/* Slider Controls */}
      <div className="py-2">
        <div className="relative flex items-center">
          <input
            type="range"
            min={0.5}
            max={1.5}
            step={0.05}
            value={multiplier}
            onChange={(e) => handleSliderChange(parseFloat(e.target.value))}
            className="w-full h-2 bg-neutral-800 rounded-lg appearance-none cursor-pointer accent-purple-500"
          />
        </div>

        <div className="flex justify-between items-center text-[11px] text-neutral-500 mt-2 font-mono">
          <span>0.5x (Half-Speed Decline)</span>
          <span>1.0x (Standard Model)</span>
          <span>1.5x (+50% Faster Decline)</span>
        </div>
      </div>

      {/* Mandatory Disclaimer Displayed Verbatim (Contract Rule) */}
      <div className="mt-4 p-3 rounded-xl bg-purple-950/30 border border-purple-500/20 flex items-start gap-2.5">
        <Info className="w-4 h-4 text-purple-400 shrink-0 mt-0.5" />
        <p className="text-[11px] text-purple-200/90 leading-relaxed font-sans">
          <strong className="text-purple-200">Mandatory Simulation Disclaimer: </strong>
          {disclaimerText}
        </p>
      </div>
    </div>
  );
}
