"use client";

import dynamic from "next/dynamic";
import React from "react";
import { ForecastPoint, TwinFull } from "@/app/lib/api";

interface TrajectoryChartProps {
  twin: TwinFull;
  simulatedForecast?: Record<string, ForecastPoint> | null;
  baselineForecast?: Record<string, ForecastPoint> | null;
  declineMultiplier?: number;
  forecastReliability?: "high" | "moderate" | "low" | "ok";
  scopeWarnings?: string[];
}

const TrajectoryChartInner = dynamic(
  () => import("./TrajectoryChartInner"),
  {
    ssr: false,
    loading: () => (
      <div className="rounded-2xl bg-neutral-900/60 backdrop-blur-xl border border-white/10 p-6 shadow-2xl h-96 flex flex-col items-center justify-center animate-pulse">
        <div className="w-8 h-8 border-2 border-purple-500/30 border-t-purple-400 rounded-full animate-spin mb-3" />
        <span className="text-xs text-neutral-400 font-medium">Loading Trajectory Forecast...</span>
      </div>
    ),
  }
);

export default function TrajectoryChart(props: TrajectoryChartProps) {
  return <TrajectoryChartInner {...props} />;
}
