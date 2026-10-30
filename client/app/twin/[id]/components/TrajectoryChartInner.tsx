"use client";

import React, { useState, useMemo } from "react";
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceArea,
} from "recharts";
import { ForecastPoint, TwinFull } from "@/app/lib/api";
import {
  TrendingDown,
  Activity,
  AlertTriangle,
  Calendar,
  Layers,
  Info,
} from "lucide-react";

interface TrajectoryChartInnerProps {
  twin: TwinFull;
  simulatedForecast?: Record<string, ForecastPoint> | null;
  baselineForecast?: Record<string, ForecastPoint> | null;
  declineMultiplier?: number;
  forecastReliability?: "high" | "moderate" | "low" | "ok";
  scopeWarnings?: string[];
}

type MetricType = "MMSE" | "CDGLOBAL" | "TOTSCORE";

interface ChartRow {
  month: number;
  monthLabel: string;
  calendarDate?: string;
  observed?: number | null;
  forecast?: number | null;
  predictedVal?: number | null;
  uncertaintyLo?: number | null;
  uncertaintyHi?: number | null;
  bandOffset?: number | null;
  bandSpan?: number | null;
  // Scenario comparison fields
  baselineForecastVal?: number | null;
  baselineBandOffset?: number | null;
  baselineBandSpan?: number | null;
  scenarioForecastVal?: number | null;
  scenarioBandOffset?: number | null;
  scenarioBandSpan?: number | null;
  isToday?: boolean;
}

// Compute calendar date e.g. "Oct 2026", "Apr 2027"
function computeCalendarDate(baseStr?: string, monthsOffset = 0): string {
  try {
    const d = new Date(baseStr || new Date().toISOString().split("T")[0]);
    d.setMonth(d.getMonth() + monthsOffset);
    return d.toLocaleDateString("en-US", { month: "short", year: "numeric" });
  } catch {
    return `+${monthsOffset}m`;
  }
}

export default function TrajectoryChartInner({
  twin,
  simulatedForecast,
  baselineForecast,
  declineMultiplier = 1.0,
  forecastReliability,
  scopeWarnings,
}: TrajectoryChartInnerProps) {
  const [metric, setMetric] = useState<MetricType>("MMSE");

  const metricConfig = {
    MMSE: {
      name: "MMSE (Mini-Mental State Exam)",
      axisTitle: "MMSE score, 0 to 30, higher = better",
      unit: "/ 30",
      description: "Primary cognitive screening metric (Higher = Better cognition)",
      key_mean: "MMSE_mean" as const,
      key_lo: "MMSE_lo" as const,
      key_hi: "MMSE_hi" as const,
      // Reference typical ranges (mean ± SD)
      adniRef: {
        CN: { y1: 27.8, y2: 30.0, label: "Typical ADNI CN (29.0 ± 1.2)" },
        MCI: { y1: 24.5, y2: 28.5, label: "Typical ADNI MCI (26.5 ± 2.0)" },
        AD: { y1: 17.5, y2: 24.5, label: "Typical ADNI AD (21.0 ± 3.5)" },
      },
    },
    CDGLOBAL: {
      name: "CDR-Global (Clinical Dementia Rating)",
      axisTitle: "CDR-Global, 0 to 3, higher = worse",
      unit: "/ 3.0",
      description: "Clinical Dementia Rating (0 = None, 0.5 = MCI, 1 = Mild, 2 = Mod, 3 = Severe)",
      key_mean: "CDGLOBAL_mean" as const,
      key_lo: "CDGLOBAL_lo" as const,
      key_hi: "CDGLOBAL_hi" as const,
      adniRef: {
        CN: { y1: 0.0, y2: 0.1, label: "Typical ADNI CN (0.0 ± 0.1)" },
        MCI: { y1: 0.3, y2: 0.7, label: "Typical ADNI MCI (0.5 ± 0.2)" },
        AD: { y1: 1.0, y2: 2.2, label: "Typical ADNI AD (1.5 ± 0.6)" },
      },
    },
    TOTSCORE: {
      name: "ADAS11-Cog / Total Score",
      axisTitle: "ADAS11 score, 0 to 70, higher = worse",
      unit: "/ 70",
      description: "Alzheimer's Disease Assessment Scale 11-item score (Higher = Worse cognitive loss)",
      key_mean: "TOTSCORE_mean" as const,
      key_lo: "TOTSCORE_lo" as const,
      key_hi: "TOTSCORE_hi" as const,
      adniRef: {
        CN: { y1: 3.1, y2: 9.3, label: "Typical ADNI CN (6.2 ± 3.1)" },
        MCI: { y1: 8.5, y2: 17.5, label: "Typical ADNI MCI (13.0 ± 4.5)" },
        AD: { y1: 18.0, y2: 32.0, label: "Typical ADNI AD (25.0 ± 7.0)" },
      },
    },
  }[metric];

  // Derive sorted timeline points and dynamic fitted domain
  const { chartData, todayMonth, yDomain, plainEnglishSummary } = useMemo(() => {
    const visits = [...(twin.visits || [])].sort((a, b) => a.month - b.month);
    const activeForecast = simulatedForecast || twin.state.forecast || {};

    const maxObsMonth = visits.length > 0 ? Math.max(...visits.map((v) => v.month)) : 0;
    const latestVisit = visits.find((v) => v.month === maxObsMonth);
    const latestVal = latestVisit ? (latestVisit[metric] as number | null) : null;

    // Collect all unique months
    const monthSet = new Set<number>();
    visits.forEach((v) => {
      if (typeof v.month === "number" && !isNaN(v.month)) {
        monthSet.add(v.month);
      }
    });
    Object.keys(activeForecast).forEach((k) => {
      const parsed = parseInt(k, 10);
      if (!isNaN(parsed)) monthSet.add(parsed);
    });
    if (baselineForecast) {
      Object.keys(baselineForecast).forEach((k) => {
        const parsed = parseInt(k, 10);
        if (!isNaN(parsed)) monthSet.add(parsed);
      });
    }

    const sortedMonths = Array.from(monthSet).sort((a, b) => a - b);
    let minObsOrFc = 999;
    let maxObsOrFc = -999;

    const rows: ChartRow[] = sortedMonths.map((m) => {
      const visit = visits.find((v) => v.month === m);
      const fc = activeForecast[String(m)] || activeForecast[`${m}m`];

      const isObserved = m <= maxObsMonth;
      const obsVal = visit ? (visit[metric] as number | null) : null;

      let meanVal: number | null = null;
      let loVal: number | null = null;
      let hiVal: number | null = null;

      if (fc) {
        meanVal = (fc[metricConfig.key_mean] as number | null) ?? null;
        loVal = (fc[metricConfig.key_lo] as number | null) ?? null;
        hiVal = (fc[metricConfig.key_hi] as number | null) ?? null;
      } else if (visit && (visit as any).predicted_scores?.[metric] !== undefined) {
        meanVal = (visit as any).predicted_scores[metric];
        const b = (visit as any).predicted_scores[`${metric}_band`];
        if (b) {
          loVal = b.lo;
          hiVal = b.hi;
        }
      } else if (m === 0 && obsVal !== null) {
        meanVal = obsVal;
        loVal = obsVal;
        hiVal = obsVal;
      }

      // Track domain bounds
      const valuesToCheck = [obsVal, meanVal, loVal, hiVal].filter(
        (v): v is number => v !== null && v !== undefined
      );
      valuesToCheck.forEach((v) => {
        if (v < minObsOrFc) minObsOrFc = v;
        if (v > maxObsOrFc) maxObsOrFc = v;
      });

      // Uncertainty band format
      const bandOffset = loVal;
      const bandSpan = hiVal !== null && loVal !== null ? Math.max(0, hiVal - loVal) : null;

      // Baseline comparison values
      let blMean: number | null = null;
      let blLo: number | null = null;
      let blHi: number | null = null;
      if (baselineForecast) {
        const blFc = baselineForecast[String(m)] || baselineForecast[`${m}m`];
        if (blFc) {
          blMean = (blFc[metricConfig.key_mean] as number | null) ?? null;
          blLo = (blFc[metricConfig.key_lo] as number | null) ?? null;
          blHi = (blFc[metricConfig.key_hi] as number | null) ?? null;
        } else if (m === maxObsMonth && obsVal !== null) {
          blMean = obsVal;
          blLo = obsVal;
          blHi = obsVal;
        }
      }
      const blSpan = blHi !== null && blLo !== null ? Math.max(0, blHi - blLo) : null;

      const calDate = computeCalendarDate(twin.baseline_date, m);

      return {
        month: m,
        monthLabel: `M${m} · ${calDate}`,
        calendarDate: calDate,
        observed: isObserved ? obsVal : null,
        forecast: meanVal,
        predictedVal: meanVal,
        uncertaintyLo: loVal,
        uncertaintyHi: hiVal,
        bandOffset,
        bandSpan,
        baselineForecastVal: m >= maxObsMonth ? blMean : null,
        baselineBandOffset: blLo,
        baselineBandSpan: blSpan,
        scenarioForecastVal: m >= maxObsMonth ? meanVal : null,
        scenarioBandOffset: bandOffset,
        scenarioBandSpan: bandSpan,
        isToday: m === maxObsMonth,
      };
    });

    // Compute padded dynamic domain
    let domain: [number, number];
    if (minObsOrFc > maxObsOrFc) {
      domain = metric === "MMSE" ? [10, 30] : metric === "CDGLOBAL" ? [0, 3] : [0, 50];
    } else {
      if (metric === "MMSE") {
        domain = [
          Math.max(0, Math.floor(minObsOrFc - 2)),
          Math.min(30, Math.ceil(maxObsOrFc + 2)),
        ];
      } else if (metric === "CDGLOBAL") {
        domain = [
          Math.max(0, Number((minObsOrFc - 0.25).toFixed(1))),
          Math.min(3.0, Number((maxObsOrFc + 0.25).toFixed(1))),
        ];
      } else {
        domain = [
          Math.max(0, Math.floor(minObsOrFc - 3)),
          Math.min(70, Math.ceil(maxObsOrFc + 4)),
        ];
      }
    }

    // Plain English Line: e.g. "Forecast: MMSE 26 -> 24.1 at 12 months, range 21.5 to 26.7"
    const baselineM0Val = visits[0] ? (visits[0][metric] as number | null) : null;
    const fc12 = activeForecast["12"];
    let plainText = "";
    if (baselineM0Val !== null && fc12) {
      const fc12Val = fc12[metricConfig.key_mean] as number | null;
      const lo12 = fc12[metricConfig.key_lo] as number | null;
      const hi12 = fc12[metricConfig.key_hi] as number | null;
      if (fc12Val !== null && lo12 !== null && hi12 !== null) {
        plainText = `Forecast: ${metric} ${baselineM0Val} → ${fc12Val} at 12 months, likely range ${lo12} to ${hi12}.`;
      }
    }

    return {
      chartData: rows,
      todayMonth: maxObsMonth,
      yDomain: domain,
      plainEnglishSummary: plainText,
    };
  }, [
    twin.visits,
    twin.baseline_date,
    twin.state.forecast,
    simulatedForecast,
    baselineForecast,
    metric,
    metricConfig,
  ]);

  // Check low reliability / scope warnings
  const isLowReliability =
    forecastReliability === "low" || (scopeWarnings && scopeWarnings.length > 0);

  return (
    <div className="rounded-2xl bg-neutral-900/60 backdrop-blur-xl border border-white/10 p-5 sm:p-6 shadow-2xl space-y-4">
      {/* 11. LOW RELIABILITY AMBER BANNER */}
      {isLowReliability && (
        <div className="p-3.5 rounded-xl bg-amber-500/15 border border-amber-500/30 text-amber-200 text-xs flex items-start gap-2.5">
          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <p className="font-bold text-white">
              Inputs outside the training range, treat the forecast as illustrative.
            </p>
            {scopeWarnings && scopeWarnings.length > 0 ? (
              <ul className="list-disc list-inside text-amber-300 space-y-0.5 text-[11px]">
                {scopeWarnings.map((w, idx) => (
                  <li key={idx}>{w}</li>
                ))}
              </ul>
            ) : (
              <p className="text-[11px] text-amber-300/80">
                Extrapolated follow-up timeline or clinical boundaries exceed ADNI cohort coverage.
              </p>
            )}
          </div>
        </div>
      )}

      {/* Header Controls */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-3 border-b border-white/10">
        <div>
          <div className="flex items-center gap-2">
            <TrendingDown className="w-5 h-5 text-purple-400" />
            <h3 className="text-base sm:text-lg font-bold text-white">
              Longitudinal Disease Trajectory
            </h3>
          </div>
          <p className="text-xs text-neutral-400 mt-0.5">{metricConfig.axisTitle}</p>
        </div>

        {/* Metric Switcher Tabs */}
        <div className="flex items-center gap-1.5 p-1 rounded-xl bg-neutral-800/80 border border-white/10 self-start md:self-auto">
          {(
            [
              { id: "MMSE", label: "MMSE (0–30)" },
              { id: "CDGLOBAL", label: "CDR-Global (0–3)" },
              { id: "TOTSCORE", label: "ADAS11 (0–70)" },
            ] as const
          ).map((tab) => (
            <button
              key={tab.id}
              onClick={() => setMetric(tab.id)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition cursor-pointer ${
                metric === tab.id
                  ? "bg-blue-600 text-white shadow-md shadow-blue-600/30"
                  : "text-neutral-400 hover:text-white"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Legend in words (Requirement 9) */}
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs px-1">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5">
            <span className="w-3.5 h-1.5 bg-blue-500 rounded-full" />
            <span className="text-neutral-300 font-medium">Observed visits</span>
          </div>

          {baselineForecast ? (
            <>
              <div className="flex items-center gap-1.5">
                <span className="w-3.5 h-1 border-t-2 border-dashed border-neutral-400" />
                <span className="text-neutral-400 font-medium">Baseline forecast</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-3.5 h-1 border-t-2 border-dashed border-purple-400" />
                <span className="text-purple-300 font-medium">Scenario forecast</span>
              </div>
            </>
          ) : (
            <>
              <div className="flex items-center gap-1.5">
                <span className="w-3.5 h-1 border-t-2 border-dashed border-purple-400" />
                <span className="text-neutral-300 font-medium">
                  Model forecast {declineMultiplier !== 1.0 && `(${declineMultiplier}x)`}
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-3.5 h-2.5 bg-purple-500/20 border border-purple-500/40 rounded-sm" />
                <span className="text-neutral-300 font-medium">Likely range (90%)</span>
              </div>
            </>
          )}
        </div>

        <div className="flex items-center gap-2">
          <span className="text-neutral-500 text-[11px]">Latest Visit:</span>
          <span className="px-2 py-0.5 rounded bg-blue-500/20 text-blue-300 font-mono font-bold border border-blue-500/30 text-[11px]">
            Month {todayMonth}
          </span>
        </div>
      </div>

      {/* Chart Canvas */}
      <div className="h-80 w-full relative">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={chartData}
            margin={{ top: 20, right: 30, left: 10, bottom: 20 }}
          >
            <defs>
              <linearGradient id="purpleBand" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#a855f7" stopOpacity={0.25} />
                <stop offset="100%" stopColor="#7e22ce" stopOpacity={0.08} />
              </linearGradient>
              <linearGradient id="greyBand" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#737373" stopOpacity={0.2} />
                <stop offset="100%" stopColor="#404040" stopOpacity={0.05} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="#262626" vertical={false} />

            {/* Reference Bands (typical ADNI ranges per stage) */}
            <ReferenceArea
              y1={metricConfig.adniRef.CN.y1}
              y2={metricConfig.adniRef.CN.y2}
              fill="#10b981"
              fillOpacity={0.03}
              stroke="#10b981"
              strokeOpacity={0.15}
            />
            <ReferenceArea
              y1={metricConfig.adniRef.MCI.y1}
              y2={metricConfig.adniRef.MCI.y2}
              fill="#f59e0b"
              fillOpacity={0.03}
              stroke="#f59e0b"
              strokeOpacity={0.15}
            />
            <ReferenceArea
              y1={metricConfig.adniRef.AD.y1}
              y2={metricConfig.adniRef.AD.y2}
              fill="#a855f7"
              fillOpacity={0.03}
              stroke="#a855f7"
              strokeOpacity={0.15}
            />

            <XAxis
              dataKey="monthLabel"
              stroke="#737373"
              tick={{ fill: "#a3a3a3", fontSize: 11 }}
              tickLine={{ stroke: "#404040" }}
            />

            <YAxis
              domain={yDomain}
              stroke="#737373"
              tick={{ fill: "#a3a3a3", fontSize: 11 }}
              tickLine={{ stroke: "#404040" }}
            />

            <Tooltip
              content={({ active, payload, label }) => {
                if (!active || !payload || !payload.length) return null;
                const row = payload[0].payload as ChartRow;
                const hasObs = row.observed !== null && row.observed !== undefined;
                const predVal = row.predictedVal ?? row.forecast;
                const hasPred = predVal !== null && predVal !== undefined;
                const diff = hasObs && hasPred && row.month > 0 ? Number((predVal! - row.observed!).toFixed(1)) : null;
                const inBand = hasObs && row.uncertaintyLo !== null && row.uncertaintyHi !== null
                  ? (row.observed! >= row.uncertaintyLo && row.observed! <= row.uncertaintyHi)
                  : null;

                return (
                  <div className="rounded-xl bg-neutral-950/95 border border-white/20 p-3.5 shadow-2xl backdrop-blur-md text-xs min-w-[230px]">
                    <div className="flex justify-between items-center mb-2 pb-1.5 border-b border-white/10">
                      <p className="font-bold text-white text-sm">
                        {label}
                      </p>
                      <span className="text-[10px] text-neutral-400 font-mono">
                        {row.month === 0 ? "Baseline" : `Month ${row.month}`}
                      </span>
                    </div>

                    {hasObs && (
                      <div className="flex justify-between gap-4 py-0.5 text-blue-300">
                        <span className="flex items-center gap-1.5">
                          <span className="w-2 h-2 rounded-full bg-blue-500 inline-block"></span>
                          Observed (Cohort):
                        </span>
                        <span className="font-mono font-bold text-white">
                          {row.observed} {metricConfig.unit}
                        </span>
                      </div>
                    )}

                    {baselineForecast ? (
                      <>
                        {row.baselineForecastVal !== null && row.baselineForecastVal !== undefined && (
                          <div className="flex justify-between gap-4 py-0.5 text-neutral-400">
                            <span>Baseline Forecast:</span>
                            <span className="font-mono font-bold">
                              {row.baselineForecastVal}
                            </span>
                          </div>
                        )}
                        {row.scenarioForecastVal !== null && row.scenarioForecastVal !== undefined && (
                          <div className="flex justify-between gap-4 py-0.5 text-purple-300">
                            <span>Scenario Forecast:</span>
                            <span className="font-mono font-bold">
                              {row.scenarioForecastVal}
                            </span>
                          </div>
                        )}
                      </>
                    ) : (
                      <>
                        {hasPred && (
                          <div className="flex justify-between gap-4 py-0.5 text-purple-300">
                            <span className="flex items-center gap-1.5">
                              <span className="w-2 h-2 rounded-full bg-purple-500 inline-block"></span>
                              Model Forecast:
                            </span>
                            <span className="font-mono font-bold text-purple-200">
                              {predVal} {metricConfig.unit}
                            </span>
                          </div>
                        )}

                        {diff !== null && (
                          <div className="flex justify-between gap-4 py-0.5 text-neutral-400">
                            <span>Residual Error (Δ):</span>
                            <span className={`font-mono font-semibold ${Math.abs(diff) <= 1.5 ? "text-emerald-400" : "text-amber-400"}`}>
                              {diff > 0 ? `+${diff}` : diff} pts
                            </span>
                          </div>
                        )}

                        {row.uncertaintyLo !== null && row.uncertaintyHi !== null && (
                          <div className="mt-1.5 pt-1.5 border-t border-white/10 flex justify-between items-center text-neutral-400">
                            <span>Likely range (90%):</span>
                            <span className="font-mono text-purple-200 font-semibold">
                              [{row.uncertaintyLo} – {row.uncertaintyHi}]
                              {inBand !== null && (
                                <span className={`ml-1.5 px-1 py-0.5 rounded text-[10px] font-bold ${inBand ? "bg-emerald-950/80 text-emerald-400 border border-emerald-800" : "bg-amber-950/80 text-amber-400 border border-amber-800"}`}>
                                  {inBand ? "✓ In Band" : "⚠ Out"}
                                </span>
                              )}
                            </span>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                );
              }}
            />

            {/* Reference Line for TODAY */}
            <ReferenceLine
              x={chartData.find((r) => r.isToday)?.monthLabel}
              stroke="#3b82f6"
              strokeWidth={2}
              strokeDasharray="4 4"
            />

            {baselineForecast ? (
              <>
                {/* Baseline Area */}
                <Area
                  dataKey="baselineBandOffset"
                  stackId="blband"
                  stroke="none"
                  fill="transparent"
                  legendType="none"
                  tooltipType="none"
                />
                <Area
                  name="Baseline Band"
                  dataKey="baselineBandSpan"
                  stackId="blband"
                  stroke="#737373"
                  strokeWidth={0.5}
                  strokeDasharray="2 2"
                  fill="url(#greyBand)"
                  isAnimationActive={true}
                />

                {/* Scenario Area */}
                <Area
                  dataKey="scenarioBandOffset"
                  stackId="scband"
                  stroke="none"
                  fill="transparent"
                  legendType="none"
                  tooltipType="none"
                />
                <Area
                  name="Scenario Band"
                  dataKey="scenarioBandSpan"
                  stackId="scband"
                  stroke="#a855f7"
                  strokeWidth={0.5}
                  strokeDasharray="3 3"
                  fill="url(#purpleBand)"
                  isAnimationActive={true}
                />

                {/* Observed visits (Solid Blue Line with larger markers) */}
                <Line
                  name="Observed"
                  type="monotone"
                  dataKey="observed"
                  stroke="#3b82f6"
                  strokeWidth={3}
                  dot={{ r: 6, fill: "#3b82f6", stroke: "#1d4ed8", strokeWidth: 2 }}
                  activeDot={{ r: 8, fill: "#60a5fa" }}
                  connectNulls={false}
                />

                {/* Baseline Forecast (Dashed Grey Line) */}
                <Line
                  name="Baseline Forecast"
                  type="monotone"
                  dataKey="baselineForecastVal"
                  stroke="#9ca3af"
                  strokeWidth={2}
                  strokeDasharray="4 4"
                  dot={{ r: 4, fill: "#737373", stroke: "#404040" }}
                  activeDot={{ r: 6, fill: "#d4d4d4" }}
                  connectNulls={true}
                />

                {/* Scenario Trajectory (Dashed Purple Line with labelled dots) */}
                <Line
                  name="Scenario Trajectory"
                  type="monotone"
                  dataKey="scenarioForecastVal"
                  stroke="#c084fc"
                  strokeWidth={2.5}
                  strokeDasharray="6 6"
                  dot={(props: any) => {
                    const { cx, cy, payload } = props;
                    if (
                      !payload ||
                      payload.scenarioForecastVal === null ||
                      cx == null ||
                      cy == null ||
                      typeof cx !== "number" ||
                      typeof cy !== "number" ||
                      isNaN(cx) ||
                      isNaN(cy)
                    ) {
                      return null;
                    }
                    const showLabel = [6, 12, 24].includes(payload.month);
                    return (
                      <g key={`dot-sc-${payload.month}`}>
                        <circle
                          cx={cx}
                          cy={cy}
                          r={5}
                          fill="#a855f7"
                          stroke="#581c87"
                          strokeWidth={2}
                        />
                        {showLabel && (
                          <text
                            x={cx}
                            y={cy - 10}
                            fill="#d8b4fe"
                            fontSize={10}
                            fontWeight="bold"
                            textAnchor="middle"
                          >
                            {payload.scenarioForecastVal}
                          </text>
                        )}
                      </g>
                    );
                  }}
                  activeDot={{ r: 7, fill: "#e879f9" }}
                  connectNulls={true}
                />
              </>
            ) : (
              <>
                {/* Uncertainty Band */}
                <Area
                  dataKey="bandOffset"
                  stackId="band"
                  stroke="none"
                  fill="transparent"
                  legendType="none"
                  tooltipType="none"
                />
                <Area
                  name="Uncertainty Band"
                  dataKey="bandSpan"
                  stackId="band"
                  stroke="#a855f7"
                  strokeWidth={0.5}
                  strokeDasharray="3 3"
                  fill="url(#purpleBand)"
                  isAnimationActive={true}
                />

                {/* Observed visits (Solid Blue Line with larger markers) */}
                <Line
                  name="Observed"
                  type="monotone"
                  dataKey="observed"
                  stroke="#3b82f6"
                  strokeWidth={3}
                  dot={{ r: 6, fill: "#3b82f6", stroke: "#1d4ed8", strokeWidth: 2 }}
                  activeDot={{ r: 8, fill: "#60a5fa" }}
                  connectNulls={false}
                />

                {/* Forecast Trajectory (Dashed Purple Line with labelled dots at 6, 12, 24m) */}
                <Line
                  name="Forecast"
                  type="monotone"
                  dataKey="forecast"
                  stroke="#c084fc"
                  strokeWidth={2.5}
                  strokeDasharray="6 6"
                  dot={(props: any) => {
                    const { cx, cy, payload } = props;
                    if (
                      !payload ||
                      payload.forecast === null ||
                      cx == null ||
                      cy == null ||
                      typeof cx !== "number" ||
                      typeof cy !== "number" ||
                      isNaN(cx) ||
                      isNaN(cy)
                    ) {
                      return null;
                    }
                    const showLabel = [6, 12, 24].includes(payload.month);
                    return (
                      <g key={`dot-fc-${payload.month}`}>
                        <circle
                          cx={cx}
                          cy={cy}
                          r={5}
                          fill="#a855f7"
                          stroke="#581c87"
                          strokeWidth={2}
                        />
                        {showLabel && (
                          <text
                            x={cx}
                            y={cy - 10}
                            fill="#d8b4fe"
                            fontSize={10}
                            fontWeight="bold"
                            textAnchor="middle"
                          >
                            {payload.forecast}
                          </text>
                        )}
                      </g>
                    );
                  }}
                  activeDot={{ r: 7, fill: "#e879f9" }}
                  connectNulls={true}
                />
              </>
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* 9. Plain-English summary line under the chart */}
      {plainEnglishSummary && (
        <div className="p-3 rounded-xl bg-purple-950/20 border border-purple-500/20 text-xs text-purple-200 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Info className="w-4 h-4 text-purple-400 shrink-0" />
            <span className="font-medium">{plainEnglishSummary}</span>
          </div>
          <span className="text-[10px] text-neutral-400 hidden sm:inline">
            Typical ADNI Ranges: CN (green), MCI (amber), AD (purple)
          </span>
        </div>
      )}

      {/* Small Data Table for Scenario Comparison */}
      {baselineForecast && (
        <div className="pt-2 overflow-x-auto">
          <table className="w-full text-left text-xs text-neutral-300 divide-y divide-white/10 rounded-xl border border-white/10">
            <thead className="bg-white/[0.03] text-neutral-400 font-semibold text-[11px]">
              <tr>
                <th className="py-2 px-3">Horizon</th>
                <th className="py-2 px-3">Date</th>
                <th className="py-2 px-3">Baseline Forecast</th>
                <th className="py-2 px-3">Scenario Forecast</th>
                <th className="py-2 px-3">Scenario Delta</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 font-mono text-[11px]">
              {chartData
                .filter((r) => [0, 6, 12, 24].includes(r.month))
                .map((r) => {
                  const bl = r.baselineForecastVal ?? r.observed ?? "-";
                  const sc = r.scenarioForecastVal ?? r.observed ?? "-";
                  const delta =
                    typeof bl === "number" && typeof sc === "number"
                      ? Number((sc - bl).toFixed(2))
                      : 0;

                  return (
                    <tr key={r.month} className="hover:bg-white/[0.02]">
                      <td className="py-2 px-3 font-bold text-white">M{r.month}</td>
                      <td className="py-2 px-3 text-neutral-400">{r.calendarDate}</td>
                      <td className="py-2 px-3 text-neutral-300">{bl}</td>
                      <td className="py-2 px-3 text-purple-300 font-bold">{sc}</td>
                      <td className="py-2 px-3">
                        {delta !== 0 ? (
                          <span
                            className={
                              delta > 0 ? "text-emerald-400 font-bold" : "text-rose-400 font-bold"
                            }
                          >
                            {delta > 0 ? `+${delta}` : delta}
                          </span>
                        ) : (
                          <span className="text-neutral-500">0.0</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
