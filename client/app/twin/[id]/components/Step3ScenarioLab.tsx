"use client";

import React, { useState, useEffect } from "react";
import { TwinFull, twinApi, ScenarioResponse, ScenarioRequest } from "@/app/lib/api";
import { usePatientStore } from "@/app/store/patientStore";
import TrajectoryChart from "./TrajectoryChart";
import {
  FlaskConical,
  Sparkles,
  AlertTriangle,
  RotateCcw,
  Check,
  TrendingDown,
  TrendingUp,
  Gauge,
  Sliders,
  Calendar,
  Layers,
  ArrowRight,
  HelpCircle,
  Save,
  ChevronDown,
  ChevronUp,
  Info,
  Heart,
  Activity,
  Dna,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface Step3ScenarioLabProps {
  twin: TwinFull;
  onTwinUpdated: (twin: TwinFull) => void;
}

export default function Step3ScenarioLab({ twin, onTwinUpdated }: Step3ScenarioLabProps) {
  const session = usePatientStore((s) => s.session);

  // 10a. "How this works" card open by default
  const [showHowItWorks, setShowHowItWorks] = useState(true);

  // Horizon selection: ONLY 6, 12, 24 months are supported by model scope
  const [horizonMonths, setHorizonMonths] = useState<6 | 12 | 24>(12);

  // Hypothetical visit inputs (initial defaults based on latest visit)
  const latestVisit = twin.visits && twin.visits.length > 0 ? twin.visits[twin.visits.length - 1] : null;
  const [hypoMmse, setHypoMmse] = useState<number>(latestVisit?.MMSE ?? 26);
  const [hypoCdglobal, setHypoCdglobal] = useState<number>(latestVisit?.CDGLOBAL ?? 0.5);
  const [hypoAdas11, setHypoAdas11] = useState<number>(latestVisit?.TOTSCORE ?? 12.0);
  const [hypoFaq, setHypoFaq] = useState<string>(latestVisit?.FAQ != null ? String(latestVisit.FAQ) : "");
  const [hypoMoca, setHypoMoca] = useState<string>(latestVisit?.MOCA != null ? String(latestVisit.MOCA) : "");

  // Decline velocity multiplier (0.5x to 1.5x)
  const [declineMultiplier, setDeclineMultiplier] = useState<number>(1.0);

  // 10b. Lifestyle controls prefilled from intake
  const [alcohol, setAlcohol] = useState<number>(session.demographics?.alcoholConsumption ?? 2);
  const [sleepQuality, setSleepQuality] = useState<number>(session.demographics?.sleepQuality ?? 7);
  const [physicalActivity, setPhysicalActivity] = useState<number>(session.demographics?.physicalActivity ?? 4);
  const [dietQuality, setDietQuality] = useState<number>(session.demographics?.dietQuality ?? 7);
  const [smoking, setSmoking] = useState<number>(session.demographics?.smoking ?? 0);

  // Medical conditions toggles
  const [hypertension, setHypertension] = useState<number>(session.medical?.hypertension ?? 0);
  const [diabetes, setDiabetes] = useState<number>(session.medical?.diabetes ?? 0);
  const [depression, setDepression] = useState<number>(session.medical?.depression ?? 0);
  const [headInjury, setHeadInjury] = useState<number>(session.medical?.headInjury ?? 0);
  const [cardiovascularDisease, setCardiovascularDisease] = useState<number>(session.medical?.cardiovascularDisease ?? 0);

  // Optional Biomarker & MRI volume overrides
  const [showOverrides, setShowOverrides] = useState(false);
  const [overrideAbeta, setOverrideAbeta] = useState<string>("");
  const [overrideTau, setOverrideTau] = useState<string>("");
  const [overridePtau, setOverridePtau] = useState<string>("");
  const [overrideHippocampus, setOverrideHippocampus] = useState<string>("");
  const [overrideVentricles, setOverrideVentricles] = useState<string>("");

  // Scenario execution state
  const [scenarioResult, setScenarioResult] = useState<ScenarioResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [savingVisit, setSavingVisit] = useState(false);
  const [actionNotice, setActionNotice] = useState<string | null>(null);

  // Auto-run scenario whenever inputs change (debounced)
  useEffect(() => {
    let isCancelled = false;

    const run = async () => {
      setLoading(true);
      try {
        const req: ScenarioRequest = {
          horizon_months: horizonMonths,
          hypothetical_visit: {
            MMSE: Number(hypoMmse),
            CDGLOBAL: Number(hypoCdglobal),
            TOTSCORE: Number(hypoAdas11),
            FAQ: hypoFaq !== "" ? Number(hypoFaq) : undefined,
            MOCA: hypoMoca !== "" ? Number(hypoMoca) : undefined,
          },
          lifestyle: {
            alcohol: Number(alcohol),
            sleepQuality: Number(sleepQuality),
            physicalActivity: Number(physicalActivity),
            dietQuality: Number(dietQuality),
            smoking: Number(smoking),
            hypertension: Number(hypertension),
            diabetes: Number(diabetes),
            depression: Number(depression),
            headInjury: Number(headInjury),
            cardiovascularDisease: Number(cardiovascularDisease),
          },
          decline_multiplier: declineMultiplier,
        };

        if (overrideAbeta || overrideTau || overridePtau) {
          req.biomarkers = {};
          if (overrideAbeta) req.biomarkers.ABETA = Number(overrideAbeta);
          if (overrideTau) req.biomarkers.TAU = Number(overrideTau);
          if (overridePtau) req.biomarkers.PTAU = Number(overridePtau);
        }

        if (overrideHippocampus || overrideVentricles) {
          req.mri_volumes = {};
          if (overrideHippocampus) req.mri_volumes.hippocampus = Number(overrideHippocampus);
          if (overrideVentricles) req.mri_volumes.ventricles = Number(overrideVentricles);
        }

        const res = await twinApi.runScenario(twin.twin_id, req);
        if (!isCancelled) {
          setScenarioResult(res);
        }
      } catch (err) {
        console.warn("Scenario call failed, fallback used:", err);
      } finally {
        if (!isCancelled) setLoading(false);
      }
    };

    const timer = setTimeout(run, 250);
    return () => {
      isCancelled = true;
      clearTimeout(timer);
    };
  }, [
    twin.twin_id,
    horizonMonths,
    hypoMmse,
    hypoCdglobal,
    hypoAdas11,
    hypoFaq,
    hypoMoca,
    alcohol,
    sleepQuality,
    physicalActivity,
    dietQuality,
    smoking,
    hypertension,
    diabetes,
    depression,
    headInjury,
    cardiovascularDisease,
    declineMultiplier,
    overrideAbeta,
    overrideTau,
    overridePtau,
    overrideHippocampus,
    overrideVentricles,
  ]);

  // Save scenario as real visit
  const handleSaveScenarioAsVisit = async () => {
    try {
      setSavingVisit(true);
      const visitPayload = {
        month: horizonMonths,
        MMSE: Number(hypoMmse),
        CDGLOBAL: Number(hypoCdglobal),
        TOTSCORE: Number(hypoAdas11),
        FAQ: hypoFaq !== "" ? Number(hypoFaq) : null,
        MOCA: hypoMoca !== "" ? Number(hypoMoca) : null,
        source: "user" as const,
      };

      const res = await twinApi.addVisit(twin.twin_id, visitPayload);
      const updatedTwin: TwinFull = {
        ...twin,
        visits: [...twin.visits, visitPayload],
        state: res.state,
        history: [
          ...twin.history,
          {
            timestamp: new Date().toISOString(),
            recorded_at: new Date().toISOString(),
            exam_date: new Date().toISOString().split("T")[0],
            stage_probs: res.state.stage_probs,
            event: `visit_added_m${horizonMonths}`,
            prob_diff: res.prob_diff,
          },
        ],
      };

      onTwinUpdated(updatedTwin);
      setActionNotice(`Scenario saved as real Month ${horizonMonths} visit! Twin re-calibrated.`);
      setTimeout(() => setActionNotice(null), 5000);
    } catch (err: any) {
      console.error("Save scenario failed:", err);
      setActionNotice(err?.message || "Failed to commit scenario to twin");
    } finally {
      setSavingVisit(false);
    }
  };

  // Reset scenario inputs back to baseline overrides (does NOT delete twin visits!)
  const handleResetScenario = () => {
    const v0 = twin.visits && twin.visits.length > 0 ? twin.visits[twin.visits.length - 1] : null;
    setHypoMmse(v0?.MMSE ?? 26);
    setHypoCdglobal(v0?.CDGLOBAL ?? 0.5);
    setHypoAdas11(v0?.TOTSCORE ?? 12.0);
    setHypoFaq(v0?.FAQ != null ? String(v0.FAQ) : "");
    setHypoMoca(v0?.MOCA != null ? String(v0.MOCA) : "");
    setDeclineMultiplier(1.0);
    setAlcohol(session.demographics?.alcoholConsumption ?? 2);
    setSleepQuality(session.demographics?.sleepQuality ?? 7);
    setPhysicalActivity(session.demographics?.physicalActivity ?? 4);
    setDietQuality(session.demographics?.dietQuality ?? 7);
    setSmoking(session.demographics?.smoking ?? 0);
    setHypertension(session.medical?.hypertension ?? 0);
    setDiabetes(session.medical?.diabetes ?? 0);
    setDepression(session.medical?.depression ?? 0);
    setHeadInjury(session.medical?.headInjury ?? 0);
    setCardiovascularDisease(session.medical?.cardiovascularDisease ?? 0);
    setOverrideAbeta("");
    setOverrideTau("");
    setOverridePtau("");
    setOverrideHippocampus("");
    setOverrideVentricles("");
    setActionNotice("Scenario parameters and overrides cleared.");
    setTimeout(() => setActionNotice(null), 4000);
  };

  const baselineProbs = scenarioResult?.baseline?.stage_probs || twin.state.stage_probs;
  const scenarioProbs = scenarioResult?.scenario?.stage_probs || twin.state.stage_probs;
  const deltaProbs: any = scenarioResult?.delta?.stage_probs || { CN: 0, MCI: 0, AD: 0 };

  const baselineRisk = scenarioResult?.baseline?.conversion_risk || twin.state.conversion_risk;
  const scenarioRisk = scenarioResult?.scenario?.conversion_risk || twin.state.conversion_risk;
  const deltaRisk: any = scenarioResult?.delta?.conversion_risk || { "12m": 0, "24m": 0 };

  // Headline metrics from API or fallback
  const headlinePAd = scenarioResult?.headline_metrics?.p_ad || {
    metric_name: "P(AD at horizon)",
    baseline_value: baselineProbs.AD || 0,
    scenario_value: scenarioProbs.AD || 0,
    delta_value: (scenarioProbs.AD || 0) - (baselineProbs.AD || 0),
    direction: (scenarioProbs.AD || 0) > (baselineProbs.AD || 0) + 0.005 ? "worse" : (scenarioProbs.AD || 0) < (baselineProbs.AD || 0) - 0.005 ? "better" : "same",
    formatted_baseline: `${Math.round((baselineProbs.AD || 0) * 100)}%`,
    formatted_scenario: `${Math.round((scenarioProbs.AD || 0) * 100)}%`,
    formatted_delta: `${((scenarioProbs.AD || 0) - (baselineProbs.AD || 0)) > 0 ? "+" : ""}${Math.round(((scenarioProbs.AD || 0) - (baselineProbs.AD || 0)) * 100)}%`,
  };

  const fallbackProgBase = baselineRisk["12m"] || (twin.state.stage === "AD" ? 0.48 : 0.38);
  const fallbackProgSc = scenarioRisk["12m"] || (twin.state.stage === "AD" ? 0.48 : 0.38);
  const headlineProgRisk = scenarioResult?.headline_metrics?.progression_risk || {
    metric_name: "progression risk",
    baseline_value: fallbackProgBase,
    scenario_value: fallbackProgSc,
    delta_value: fallbackProgSc - fallbackProgBase,
    direction: fallbackProgSc > fallbackProgBase + 0.005 ? "worse" : fallbackProgSc < fallbackProgBase - 0.005 ? "better" : "same",
    formatted_baseline: `${Math.round(fallbackProgBase * 100)}%`,
    formatted_scenario: `${Math.round(fallbackProgSc * 100)}%`,
    formatted_delta: `${(fallbackProgSc - fallbackProgBase) > 0 ? "+" : ""}${Math.round((fallbackProgSc - fallbackProgBase) * 100)}%`,
  };

  // Plain English Result text
  const baseAdPct = Math.round((baselineProbs.AD || 0) * 100);
  const simAdPct = Math.round((scenarioProbs.AD || 0) * 100);
  const plainEnglishResult = `Under this scenario, P(AD) at ${horizonMonths} months goes from ${baseAdPct}% to ${simAdPct}%.`;

  const renderDirectionBadge = (dir: "worse" | "better" | "same" | string, label?: string) => {
    if (dir === "worse") {
      return (
        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-rose-500/20 text-rose-300 border border-rose-500/30">
          <TrendingUp className="w-3 h-3 text-rose-400" />
          <span>{label || "Worsening Risk"}</span>
        </span>
      );
    }
    if (dir === "better") {
      return (
        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
          <TrendingDown className="w-3 h-3 text-emerald-400" />
          <span>{label || "Improving / Protective"}</span>
        </span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-neutral-800 text-neutral-400 border border-white/10">
        <span>— Unchanged</span>
      </span>
    );
  };

  return (
    <div className="space-y-6">
      {/* 10a. "HOW THIS WORKS" CARD (Open the first time) */}
      <div className="rounded-2xl bg-gradient-to-r from-purple-950/40 via-blue-950/30 to-purple-950/40 border border-purple-500/30 p-5 shadow-2xl relative overflow-hidden">
        <button
          type="button"
          onClick={() => setShowHowItWorks(!showHowItWorks)}
          className="w-full flex items-center justify-between text-left cursor-pointer"
        >
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-purple-500/20 text-purple-300 border border-purple-500/30">
              <FlaskConical className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                Scenario Lab: Counterfactual Simulator
                <span className="text-[10px] font-semibold uppercase px-2 py-0.5 rounded-full bg-purple-500/20 text-purple-300 border border-purple-500/30">
                  Interactive Sandbox
                </span>
              </h3>
              <p className="text-xs text-neutral-400 mt-0.5">
                Simulate hypothetical lifestyle interventions and disease velocities.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 text-xs text-purple-300 font-semibold">
            <span>{showHowItWorks ? "Collapse Guide" : "How this works"}</span>
            {showHowItWorks ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </div>
        </button>

        <AnimatePresence>
          {showHowItWorks && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-4 pt-4 border-t border-white/10 space-y-4 text-xs text-neutral-300"
            >
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-3.5 rounded-xl bg-white/[0.03] border border-white/5 space-y-1.5">
                  <span className="font-bold text-white flex items-center gap-1.5">
                    <Info className="w-4 h-4 text-blue-400" />
                    What the Lab Does
                  </span>
                  <p className="text-neutral-400 leading-relaxed text-[11px]">
                    Takes your hypothetical next-visit test scores, lifestyle changes, and decline velocity, re-runs the trained forecasting model, and re-stages the patient at that future horizon. <strong>Saves nothing</strong> until you explicitly choose to commit.
                  </p>
                </div>

                <div className="p-3.5 rounded-xl bg-white/[0.03] border border-white/5 space-y-1.5">
                  <span className="font-bold text-white flex items-center gap-1.5">
                    <Calendar className="w-4 h-4 text-purple-400" />
                    Why Only 6 / 12 / 24 Months
                  </span>
                  <p className="text-neutral-400 leading-relaxed text-[11px]">
                    The underlying models were trained on ADNI follow-ups at 6, 12, and 24-month clinical visits. Longer horizons (e.g. 36 months) would be unvalidated statistical extrapolation, so 36 months remains disabled with this explanation.
                  </p>
                </div>

                <div className="p-3.5 rounded-xl bg-white/[0.03] border border-white/5 space-y-1.5">
                  <span className="font-bold text-white flex items-center gap-1.5">
                    <Sliders className="w-4 h-4 text-emerald-400" />
                    What the Multiplier Means
                  </span>
                  <p className="text-neutral-400 leading-relaxed text-[11px]">
                    <strong>0.5x</strong> corresponds to half the predicted natural decline velocity (simulating successful disease-modifying therapy or intensive intervention), while <strong>1.5x</strong> represents 50% accelerated clinical deterioration.
                  </p>
                </div>
              </div>

              {/* API Method Steps */}
              {scenarioResult?.method && scenarioResult.method.length > 0 && (
                <div className="p-3 rounded-xl bg-white/[0.02] border border-white/5 flex items-center gap-2 flex-wrap text-[11px]">
                  <span className="font-bold text-neutral-300">Model Pipeline Steps:</span>
                  {scenarioResult.method.map((step, idx) => (
                    <span
                      key={`sc-method-${idx}`}
                      className="px-2 py-0.5 rounded bg-purple-500/10 border border-purple-500/20 text-purple-200"
                    >
                      {idx + 1}. {step}
                    </span>
                  ))}
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* HEADLINE METRICS: P(AD at horizon) and progression risk */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Metric 1: P(AD at horizon) */}
        <div className="p-4 rounded-2xl bg-neutral-900/60 backdrop-blur-xl border border-white/10 shadow-xl space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-neutral-300 uppercase tracking-wider flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-purple-400" />
              <span>{headlinePAd.metric_name}</span>
            </span>
            {renderDirectionBadge(headlinePAd.direction)}
          </div>
          <div className="flex items-baseline justify-between pt-1">
            <div className="flex items-baseline gap-2 font-mono">
              <span className="text-xs text-neutral-400">Baseline: {headlinePAd.formatted_baseline}</span>
              <ArrowRight className="w-3.5 h-3.5 text-neutral-500 inline" />
              <span className="text-xl font-bold text-white">Scenario: {headlinePAd.formatted_scenario}</span>
            </div>
            <span
              className={`text-xs font-bold font-mono px-2 py-0.5 rounded ${
                headlinePAd.direction === "worse"
                  ? "bg-rose-500/20 text-rose-300 border border-rose-500/30"
                  : headlinePAd.direction === "better"
                  ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                  : "bg-neutral-800 text-neutral-400"
              }`}
            >
              Δ {headlinePAd.formatted_delta}
            </span>
          </div>
          <p className="text-[10px] text-neutral-500">
            Calibrated stage probability at {horizonMonths}-month simulated horizon.
          </p>
        </div>

        {/* Metric 2: progression risk */}
        <div className="p-4 rounded-2xl bg-neutral-900/60 backdrop-blur-xl border border-white/10 shadow-xl space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-neutral-300 uppercase tracking-wider flex items-center gap-1.5">
              <Gauge className="w-3.5 h-3.5 text-blue-400" />
              <span>{headlineProgRisk.metric_name}</span>
            </span>
            {renderDirectionBadge(headlineProgRisk.direction)}
          </div>
          <div className="flex items-baseline justify-between pt-1">
            <div className="flex items-baseline gap-2 font-mono">
              <span className="text-xs text-neutral-400">Baseline: {headlineProgRisk.formatted_baseline}</span>
              <ArrowRight className="w-3.5 h-3.5 text-neutral-500 inline" />
              <span className="text-xl font-bold text-white">Scenario: {headlineProgRisk.formatted_scenario}</span>
            </div>
            <span
              className={`text-xs font-bold font-mono px-2 py-0.5 rounded ${
                headlineProgRisk.direction === "worse"
                  ? "bg-rose-500/20 text-rose-300 border border-rose-500/30"
                  : headlineProgRisk.direction === "better"
                  ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                  : "bg-neutral-800 text-neutral-400"
              }`}
            >
              Δ {headlineProgRisk.formatted_delta}
            </span>
          </div>
          <p className="text-[10px] text-neutral-500">
            Hazard of cognitive progression / stage worsening over natural history.
          </p>
        </div>
      </div>

      {/* PLAIN-ENGLISH RESULT & DRIVERS DELTA BANNER */}
      {scenarioResult && (
        <div className="p-4 rounded-2xl bg-neutral-900/60 backdrop-blur-xl border border-white/10 shadow-xl space-y-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-purple-400 shrink-0" />
              <p className="text-sm font-bold text-white font-sans">{plainEnglishResult}</p>
            </div>
            <div className="flex items-center gap-2 self-end sm:self-auto">
              <button
                onClick={handleSaveScenarioAsVisit}
                disabled={savingVisit || loading}
                className="px-3.5 py-1.5 rounded-xl bg-purple-600 hover:bg-purple-500 disabled:opacity-50 text-white font-semibold text-xs transition flex items-center gap-1.5 shadow-md shadow-purple-600/30 cursor-pointer"
              >
                <Save className="w-3.5 h-3.5" />
                <span>Save as Real Visit</span>
              </button>
              <button
                onClick={handleResetScenario}
                disabled={loading}
                className="px-3 py-1.5 rounded-xl bg-white/5 hover:bg-white/10 text-neutral-300 border border-white/10 text-xs font-semibold flex items-center gap-1 transition cursor-pointer"
              >
                <RotateCcw className="w-3 h-3 text-neutral-400" />
                <span>Reset Scenario Overrides</span>
              </button>
            </div>
          </div>

          {/* Drivers Delta (Strictly styled by d.direction from API) */}
          {scenarioResult.drivers_delta && scenarioResult.drivers_delta.length > 0 && (
            <div className="pt-2 border-t border-white/5 flex flex-wrap gap-2 items-center">
              <span className="text-[11px] text-neutral-400 font-semibold mr-1">
                Drivers of Difference:
              </span>
              {scenarioResult.drivers_delta.map((d, idx) => {
                const isWorse = d.direction === "worse";
                const isBetter = d.direction === "better";
                return (
                  <div
                    key={`sc-driver-${d.factor}-${idx}`}
                    className={`px-2.5 py-1 rounded-lg border text-xs flex items-center gap-1.5 ${
                      isWorse
                        ? "bg-rose-950/40 border-rose-500/30 text-rose-300"
                        : isBetter
                        ? "bg-emerald-950/40 border-emerald-500/30 text-emerald-300"
                        : "bg-white/[0.04] border-white/10 text-neutral-300"
                    }`}
                  >
                    <span className="font-bold">{d.label}:</span>
                    <span className="font-mono">{String(d.delta)}</span>
                    {isWorse ? (
                      <TrendingUp className="w-3 h-3 text-rose-400" />
                    ) : isBetter ? (
                      <TrendingDown className="w-3 h-3 text-emerald-400" />
                    ) : null}
                    <span className="text-[10px] opacity-80">({d.impact})</span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Notice toast */}
      <AnimatePresence>
        {actionNotice && (
          <motion.div
            initial={{ opacity: 0, y: -5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="p-3 rounded-xl bg-emerald-950/40 border border-emerald-500/30 text-emerald-300 text-xs flex items-center gap-2"
          >
            <Check className="w-4 h-4" />
            <span>{actionNotice}</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* CONTROLS SECTION */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 bg-neutral-900/60 backdrop-blur-xl border border-white/10 p-6 rounded-3xl shadow-xl">
        {/* Left Column: Horizon & Velocity Slider & Lifestyle Controls */}
        <div className="lg:col-span-6 space-y-5 pr-0 lg:pr-4 lg:border-r border-white/10">
          {/* Horizon Selector (Strictly 6, 12, 24 months ONLY) */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-bold text-neutral-300 uppercase tracking-wider flex items-center gap-1.5">
                <Calendar className="w-3.5 h-3.5 text-blue-400" />
                <span>Simulation Horizon (Strict Scope)</span>
              </label>
              <span className="text-[10px] text-neutral-500 font-mono">6, 12, 24m Only</span>
            </div>

            <div className="grid grid-cols-4 gap-2">
              {([6, 12, 24] as const).map((h) => (
                <button
                  key={h}
                  type="button"
                  onClick={() => setHorizonMonths(h)}
                  className={`py-2 px-3 rounded-xl text-xs font-bold transition flex flex-col items-center justify-center cursor-pointer ${
                    horizonMonths === h
                      ? "bg-purple-600 text-white shadow-lg shadow-purple-600/30 border border-purple-400"
                      : "bg-neutral-800 text-neutral-300 hover:bg-neutral-700 border border-white/5"
                  }`}
                >
                  <span>{h} mo</span>
                  <span className="text-[10px] font-normal opacity-70">
                    {h === 6 ? "Short" : h === 12 ? "1-Year" : "2-Year"}
                  </span>
                </button>
              ))}

              {/* Disabled 36m Option with explicit tooltip */}
              <div
                className="group relative py-2 px-3 rounded-xl text-xs font-semibold bg-neutral-900/80 text-neutral-600 border border-neutral-800/80 flex flex-col items-center justify-center cursor-not-allowed opacity-60"
                title="Horizons >24m disabled: ADNI progression model was trained strictly on 6, 12, 24m follow-up visits; longer horizons would be extrapolation."
              >
                <span>36 mo</span>
                <span className="text-[9px] text-rose-500/80">Extrapolation</span>
              </div>
            </div>
          </div>

          {/* Decline Velocity Multiplier (0.5x to 1.5x) */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-xs font-bold text-neutral-300 uppercase tracking-wider flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-purple-400" />
                <span>Decline Velocity Multiplier (↑ higher = worse / accelerated loss)</span>
              </label>
              <span className="text-xs font-mono font-bold text-purple-300 bg-purple-500/20 px-2 py-0.5 rounded-md border border-purple-500/30">
                {declineMultiplier.toFixed(2)}x Speed
              </span>
            </div>

            <input
              type="range"
              min={0.5}
              max={1.5}
              step={0.05}
              value={declineMultiplier}
              onChange={(e) => setDeclineMultiplier(parseFloat(e.target.value))}
              className="w-full accent-purple-500 cursor-pointer h-2 bg-neutral-800 rounded-lg appearance-none"
            />

            <div className="flex justify-between text-[11px] text-neutral-500 font-mono">
              <span className={declineMultiplier < 0.9 ? "text-emerald-400 font-semibold" : ""}>
                0.5x (Protected / Slower)
              </span>
              <span className={Math.abs(declineMultiplier - 1.0) < 0.05 ? "text-white font-semibold" : ""}>
                1.0x (Standard)
              </span>
              <span className={declineMultiplier > 1.1 ? "text-rose-400 font-semibold" : ""}>
                1.5x (Faster Decline)
              </span>
            </div>
          </div>

          {/* 10b. LIFESTYLE CONTROLS PREFILLED FROM INTAKE */}
          <div className="p-4 rounded-2xl bg-white/[0.02] border border-white/5 space-y-4">
            <h4 className="text-xs font-bold text-neutral-300 uppercase tracking-wider flex items-center gap-1.5">
              <Heart className="w-3.5 h-3.5 text-rose-400" />
              <span>Lifestyle Modifiers (Counterfactual Overrides)</span>
            </h4>

            {/* Sliders Grid */}
            <div className="space-y-3">
              {/* Alcohol Slider */}
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-neutral-400">Alcohol Consumption (↑ higher = harmful)</span>
                  <span className="font-mono text-purple-300 font-semibold">{alcohol} units/week</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={20}
                  step={1}
                  value={alcohol}
                  onChange={(e) => setAlcohol(Number(e.target.value))}
                  className="w-full accent-purple-500 cursor-pointer h-1.5 bg-neutral-800 rounded-lg appearance-none"
                />
              </div>

              {/* Physical Activity Slider */}
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-neutral-400">Physical Activity (↑ higher = protective)</span>
                  <span className="font-mono text-emerald-300 font-semibold">{physicalActivity} hrs/week</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={10}
                  step={0.5}
                  value={physicalActivity}
                  onChange={(e) => setPhysicalActivity(Number(e.target.value))}
                  className="w-full accent-emerald-500 cursor-pointer h-1.5 bg-neutral-800 rounded-lg appearance-none"
                />
              </div>

              {/* Sleep Quality Slider */}
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-neutral-400">Sleep Quality (↑ higher = protective)</span>
                  <span className="font-mono text-blue-300 font-semibold">{sleepQuality}/10</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={10}
                  step={1}
                  value={sleepQuality}
                  onChange={(e) => setSleepQuality(Number(e.target.value))}
                  className="w-full accent-blue-500 cursor-pointer h-1.5 bg-neutral-800 rounded-lg appearance-none"
                />
              </div>

              {/* Diet Quality Slider */}
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-neutral-400">Diet Quality (↑ higher = protective)</span>
                  <span className="font-mono text-amber-300 font-semibold">{dietQuality}/10</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={10}
                  step={1}
                  value={dietQuality}
                  onChange={(e) => setDietQuality(Number(e.target.value))}
                  className="w-full accent-amber-500 cursor-pointer h-1.5 bg-neutral-800 rounded-lg appearance-none"
                />
              </div>
            </div>

            {/* Smoking and Medical Condition Toggles */}
            <div className="pt-2 border-t border-white/5 space-y-2">
              <span className="text-[11px] text-neutral-400 font-semibold block">
                Smoking &amp; Health Condition Toggles:
              </span>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                {[
                  { label: "Smoking (harmful)", val: smoking, setVal: setSmoking },
                  { label: "Hypertension", val: hypertension, setVal: setHypertension },
                  { label: "Diabetes", val: diabetes, setVal: setDiabetes },
                  { label: "Depression", val: depression, setVal: setDepression },
                  { label: "Head Injury", val: headInjury, setVal: setHeadInjury },
                  { label: "CVD", val: cardiovascularDisease, setVal: setCardiovascularDisease },
                ].map((item, itemIdx) => (
                  <button
                    key={`health-toggle-${item.label}-${itemIdx}`}
                    type="button"
                    onClick={() => item.setVal(item.val === 1 ? 0 : 1)}
                    className={`py-1.5 px-2.5 rounded-lg text-xs font-semibold border transition cursor-pointer flex items-center justify-between ${
                      item.val === 1
                        ? "bg-purple-950/50 border-purple-500/50 text-purple-300"
                        : "bg-white/5 border-white/10 text-neutral-400 hover:text-white"
                    }`}
                  >
                    <span>{item.label}</span>
                    <span
                      className={`w-2 h-2 rounded-full ${
                        item.val === 1 ? "bg-purple-400" : "bg-neutral-600"
                      }`}
                    />
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Hypothetical Scores & Overrides */}
        <div className="lg:col-span-6 space-y-4">
          <div className="flex items-center justify-between">
            <label className="text-xs font-bold text-neutral-300 uppercase tracking-wider flex items-center gap-1.5">
              <Activity className="w-3.5 h-3.5 text-blue-400" />
              <span>Hypothetical Neuropsych Scores (at Month {horizonMonths})</span>
            </label>
            <span className="text-[11px] text-neutral-400">Edit scores to simulate</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {/* MMSE */}
            <div className="p-3 rounded-2xl bg-neutral-800/50 border border-white/5 space-y-1">
              <div className="flex justify-between text-xs">
                <span className="font-semibold text-neutral-300">MMSE (↑ higher = better)</span>
                <span className="text-neutral-500 font-mono">0–30</span>
              </div>
              <input
                type="number"
                min={0}
                max={30}
                step={0.5}
                value={hypoMmse}
                onChange={(e) => setHypoMmse(Number(e.target.value))}
                className="w-full bg-neutral-900 border border-white/10 rounded-xl px-3 py-1.5 text-sm text-white font-mono focus:border-purple-500 focus:outline-none"
              />
              <span className="text-[10px] text-neutral-500 block">Baseline: {twin.visits[0]?.MMSE ?? 26}</span>
            </div>

            {/* CDR-Global */}
            <div className="p-3 rounded-2xl bg-neutral-800/50 border border-white/5 space-y-1">
              <div className="flex justify-between text-xs">
                <span className="font-semibold text-neutral-300">CDR-Global (↑ higher = worse)</span>
                <span className="text-neutral-500 font-mono">0–3</span>
              </div>
              <select
                value={hypoCdglobal}
                onChange={(e) => setHypoCdglobal(Number(e.target.value))}
                className="w-full bg-neutral-900 border border-white/10 rounded-xl px-3 py-1.5 text-sm text-white font-mono focus:border-purple-500 focus:outline-none"
              >
                <option value={0}>0.0 (None)</option>
                <option value={0.5}>0.5 (Very Mild / MCI)</option>
                <option value={1}>1.0 (Mild)</option>
                <option value={2}>2.0 (Moderate)</option>
                <option value={3}>3.0 (Severe)</option>
              </select>
              <span className="text-[10px] text-neutral-500 block">Baseline: {twin.visits[0]?.CDGLOBAL ?? 0.5}</span>
            </div>

            {/* ADAS11 */}
            <div className="p-3 rounded-2xl bg-neutral-800/50 border border-white/5 space-y-1">
              <div className="flex justify-between text-xs">
                <span className="font-semibold text-neutral-300">ADAS11 (↑ higher = worse)</span>
                <span className="text-neutral-500 font-mono">0–70</span>
              </div>
              <input
                type="number"
                min={0}
                max={70}
                step={0.5}
                value={hypoAdas11}
                onChange={(e) => setHypoAdas11(Number(e.target.value))}
                className="w-full bg-neutral-900 border border-white/10 rounded-xl px-3 py-1.5 text-sm text-white font-mono focus:border-purple-500 focus:outline-none"
              />
              <span className="text-[10px] text-neutral-500 block">Baseline: {twin.visits[0]?.TOTSCORE ?? 12.0}</span>
            </div>
          </div>

          {/* Optional FAQ & MOCA inputs */}
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 rounded-2xl bg-neutral-800/50 border border-white/5 space-y-1">
              <div className="flex justify-between text-xs">
                <span className="font-semibold text-neutral-300">FAQ (Optional)</span>
                <span className="text-neutral-500 font-mono">0–30</span>
              </div>
              <input
                type="number"
                min={0}
                max={30}
                value={hypoFaq}
                onChange={(e) => setHypoFaq(e.target.value)}
                placeholder="e.g. 4"
                className="w-full bg-neutral-900 border border-white/10 rounded-xl px-3 py-1.5 text-xs text-white font-mono focus:border-purple-500 focus:outline-none"
              />
            </div>
            <div className="p-3 rounded-2xl bg-neutral-800/50 border border-white/5 space-y-1">
              <div className="flex justify-between text-xs">
                <span className="font-semibold text-neutral-300">MoCA (Optional)</span>
                <span className="text-neutral-500 font-mono">0–30</span>
              </div>
              <input
                type="number"
                min={0}
                max={30}
                value={hypoMoca}
                onChange={(e) => setHypoMoca(e.target.value)}
                placeholder="e.g. 23"
                className="w-full bg-neutral-900 border border-white/10 rounded-xl px-3 py-1.5 text-xs text-white font-mono focus:border-purple-500 focus:outline-none"
              />
            </div>
          </div>

          {/* Optional Biomarker & MRI Volume Overrides */}
          <div className="pt-2">
            <button
              type="button"
              onClick={() => setShowOverrides(!showOverrides)}
              className="text-xs font-semibold text-purple-400 hover:text-purple-300 flex items-center gap-1.5 transition"
            >
              <Dna className="w-3.5 h-3.5" />
              <span>{showOverrides ? "▼ Hide" : "▶ Show"} Optional Biomarker &amp; MRI Volume Overrides</span>
            </button>

            {showOverrides && (
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-2.5 mt-3 p-3.5 rounded-2xl bg-neutral-800/30 border border-white/5">
                <div>
                  <label className="text-[11px] text-neutral-400 block mb-1">Aβ42 (pg/mL)</label>
                  <input
                    type="number"
                    value={overrideAbeta}
                    onChange={(e) => setOverrideAbeta(e.target.value)}
                    placeholder="e.g. 850"
                    className="w-full bg-neutral-900 border border-white/10 rounded-xl px-2.5 py-1 text-xs text-white font-mono focus:border-purple-500 focus:outline-none placeholder:text-neutral-700"
                  />
                </div>
                <div>
                  <label className="text-[11px] text-neutral-400 block mb-1">Tau (pg/mL)</label>
                  <input
                    type="number"
                    value={overrideTau}
                    onChange={(e) => setOverrideTau(e.target.value)}
                    placeholder="e.g. 450"
                    className="w-full bg-neutral-900 border border-white/10 rounded-xl px-2.5 py-1 text-xs text-white font-mono focus:border-purple-500 focus:outline-none placeholder:text-neutral-700"
                  />
                </div>
                <div>
                  <label className="text-[11px] text-neutral-400 block mb-1">p-Tau (pg/mL)</label>
                  <input
                    type="number"
                    value={overridePtau}
                    onChange={(e) => setOverridePtau(e.target.value)}
                    placeholder="e.g. 42"
                    className="w-full bg-neutral-900 border border-white/10 rounded-xl px-2.5 py-1 text-xs text-white font-mono focus:border-purple-500 focus:outline-none placeholder:text-neutral-700"
                  />
                </div>
                <div>
                  <label className="text-[11px] text-neutral-400 block mb-1">Hippocampus</label>
                  <input
                    type="number"
                    value={overrideHippocampus}
                    onChange={(e) => setOverrideHippocampus(e.target.value)}
                    placeholder="mm³"
                    className="w-full bg-neutral-900 border border-white/10 rounded-xl px-2.5 py-1 text-xs text-white font-mono focus:border-purple-500 focus:outline-none placeholder:text-neutral-700"
                  />
                </div>
                <div>
                  <label className="text-[11px] text-neutral-400 block mb-1">Ventricles</label>
                  <input
                    type="number"
                    value={overrideVentricles}
                    onChange={(e) => setOverrideVentricles(e.target.value)}
                    placeholder="mm³"
                    className="w-full bg-neutral-900 border border-white/10 rounded-xl px-2.5 py-1 text-xs text-white font-mono focus:border-purple-500 focus:outline-none placeholder:text-neutral-700"
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* SIDE-BY-SIDE COMPARISON: Baseline (Grey) vs Scenario (Colour) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Stage Distribution Shift */}
        <div className="lg:col-span-5 space-y-6">
          <div className="bg-neutral-900/60 backdrop-blur-xl border border-white/10 p-6 rounded-3xl shadow-xl space-y-6">
            <h3 className="text-base font-bold text-white flex items-center justify-between">
              <span>Stage Distribution Shift</span>
              <span className="text-xs text-neutral-400 font-normal">Baseline vs Scenario</span>
            </h3>

            {/* Stage Probs Comparison Table / Bars */}
            <div className="space-y-4">
              {(["CN", "MCI", "AD"] as const).map((stage) => {
                const bVal = baselineProbs[stage] ?? 0;
                const sVal = scenarioProbs[stage] ?? 0;
                const dVal = (deltaProbs[stage] ?? (sVal - bVal)) * 100;
                const color =
                  stage === "CN" ? "text-emerald-400" : stage === "MCI" ? "text-amber-400" : "text-rose-400";
                const bgBar =
                  stage === "CN" ? "bg-emerald-500" : stage === "MCI" ? "bg-amber-500" : "bg-rose-500";

                return (
                  <div key={stage} className="p-3.5 rounded-2xl bg-white/[0.02] border border-white/5 space-y-2">
                    <div className="flex items-center justify-between text-xs">
                      <span className={`font-bold ${color}`}>{stage}</span>
                      <div className="flex items-center gap-3 font-mono">
                        <span className="text-neutral-500">
                          Base: {(bVal * 100).toFixed(1)}%
                        </span>
                        <span className="text-white font-bold">
                          Sim: {(sVal * 100).toFixed(1)}%
                        </span>
                        <span
                          className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${
                            Math.abs(dVal) < 0.1
                              ? "bg-neutral-800 text-neutral-400"
                              : dVal > 0
                              ? stage === "CN"
                                ? "bg-emerald-500/20 text-emerald-300"
                                : "bg-rose-500/20 text-rose-300"
                              : stage === "CN"
                              ? "bg-rose-500/20 text-rose-300"
                              : "bg-emerald-500/20 text-emerald-300"
                          }`}
                        >
                          {dVal > 0 ? `+${dVal.toFixed(1)}%` : `${dVal.toFixed(1)}%`}
                        </span>
                      </div>
                    </div>

                    {/* Dual comparison bar */}
                    <div className="space-y-1">
                      {/* Baseline bar (Grey) */}
                      <div className="w-full h-1.5 rounded-full bg-neutral-800 overflow-hidden">
                        <div
                          className="h-full bg-neutral-500 rounded-full transition-all duration-500"
                          style={{ width: `${Math.max(2, bVal * 100)}%` }}
                        />
                      </div>
                      {/* Scenario bar (Coloured) */}
                      <div className="w-full h-2 rounded-full bg-neutral-800 overflow-hidden">
                        <div
                          className={`h-full ${bgBar} rounded-full transition-all duration-500`}
                          style={{ width: `${Math.max(2, sVal * 100)}%` }}
                        />
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Conversion Risk Side-by-Side (12m & 24m) */}
            <div className="pt-4 border-t border-white/10 space-y-3">
              <h4 className="text-xs font-bold text-neutral-300 uppercase tracking-wider flex items-center gap-2">
                <Gauge className="w-4 h-4 text-purple-400" />
                <span>Conversion Risk Comparison</span>
              </h4>

              <div className="grid grid-cols-2 gap-3">
                {/* 12m Risk */}
                <div className="p-3.5 rounded-2xl bg-white/[0.02] border border-white/5 space-y-1 text-center">
                  <span className="text-[11px] font-semibold text-neutral-400">12-Month Risk</span>
                  <div className="flex items-center justify-center gap-2 font-mono">
                    <span className="text-sm text-neutral-500">
                      {((baselineRisk["12m"] || 0) * 100).toFixed(0)}%
                    </span>
                    <ArrowRight className="w-3.5 h-3.5 text-neutral-600" />
                    <span className="text-base font-bold text-purple-300">
                      {((scenarioRisk["12m"] || 0) * 100).toFixed(0)}%
                    </span>
                  </div>
                  <span className="text-[10px] text-neutral-500 block font-mono">
                    Δ {((deltaRisk["12m"] || 0) * 100).toFixed(0)}%
                  </span>
                </div>

                {/* 24m Risk */}
                <div className="p-3.5 rounded-2xl bg-white/[0.02] border border-white/5 space-y-1 text-center">
                  <span className="text-[11px] font-semibold text-neutral-400">24-Month Risk</span>
                  <div className="flex items-center justify-center gap-2 font-mono">
                    <span className="text-sm text-neutral-500">
                      {((baselineRisk["24m"] || 0) * 100).toFixed(0)}%
                    </span>
                    <ArrowRight className="w-3.5 h-3.5 text-neutral-600" />
                    <span className="text-base font-bold text-purple-300">
                      {((scenarioRisk["24m"] || 0) * 100).toFixed(0)}%
                    </span>
                  </div>
                  <span className="text-[10px] text-neutral-500 block font-mono">
                    Δ {((deltaRisk["24m"] || 0) * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Trajectory Chart with Two Lines & Bands + Data Table */}
        <div className="lg:col-span-7">
          <TrajectoryChart
            twin={twin}
            baselineForecast={scenarioResult?.baseline?.forecast || twin.state.forecast}
            simulatedForecast={scenarioResult?.scenario?.forecast || twin.state.forecast}
            declineMultiplier={declineMultiplier}
            forecastReliability={scenarioResult?.forecast_reliability}
            scopeWarnings={scenarioResult?.scope_warnings}
          />
        </div>
      </div>
    </div>
  );
}
