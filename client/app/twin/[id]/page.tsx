"use client";

import React, { useEffect, useState, use } from "react";
import Link from "next/link";
import { TwinFull, twinApi, ForecastPoint, AddVisitResponse } from "@/app/lib/api";
import { usePatientStore } from "@/app/store/patientStore";

// Stepper Components
import Step1Baseline from "./components/Step1Baseline";
import Step2Visits from "./components/Step2Visits";
import Step3ScenarioLab from "./components/Step3ScenarioLab";
import Step4Profile from "./components/Step4Profile";

import ModelCardDrawer from "./components/ModelCardDrawer";

import {
  ArrowLeft,
  FileCode,
  Sparkles,
  Wifi,
  WifiOff,
  User,
  AlertCircle,
  RefreshCw,
  Layers,
  Calendar,
  FlaskConical,
  FileText,
  ShieldCheck,
} from "lucide-react";

interface PageProps {
  params: Promise<{ id: string }>;
}

export default function DigitalTwinDashboard({ params }: PageProps) {
  const resolvedParams = use(params);
  const twinId = resolvedParams.id;

  const { session, setTwin: setStoreTwin } = usePatientStore();

  const [twin, setTwin] = useState<TwinFull | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isOffline, setIsOffline] = useState(false);

  // Stepper state: 1 Baseline, 2 Visits, 3 Scenario, 4 Profile
  const [activeStep, setActiveStep] = useState<1 | 2 | 3 | 4>(1);

  // Model Card Drawer state
  const [isModelCardOpen, setIsModelCardOpen] = useState(false);

  // Fetch twin data on load
  const loadTwinData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Check if store already has this twin
      if (session?.twin && session.twin.twin_id === twinId) {
        setTwin(session.twin);
        setLoading(false);
        return;
      }

      const res = await twinApi.getTwin(twinId);
      setTwin(res);
      setStoreTwin(res);
    } catch (err: any) {
      console.warn("Failed to fetch twin:", err);
      setIsOffline(true);
      setError(err?.message || "Unable to reach Digital Twin API server");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadTwinData();
  }, [twinId]);

  // Update handler
  const handleTwinUpdated = (updated: TwinFull) => {
    setTwin(updated);
    setStoreTwin(updated);
  };

  const stepsConfig = [
    {
      num: 1,
      title: "Baseline & Stage",
      desc: "Brain visual, modality cards & tiers",
      icon: Layers,
    },
    {
      num: 2,
      title: "Follow-Up Visits",
      desc: "Longitudinal inputs & trajectory updates",
      icon: Calendar,
    },
    {
      num: 3,
      title: "Scenario Lab",
      desc: "Counterfactual dry run & delta shift",
      icon: FlaskConical,
    },
    {
      num: 4,
      title: "Patient Profile",
      desc: "One-page summary, PDF & JSON",
      icon: FileText,
    },
  ] as const;

  return (
    <div className="min-h-screen bg-transparent text-white relative selection:bg-purple-500 selection:text-white pb-20">
      {/* Background Ambience & Subtle Grid */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(59,130,246,0.15),rgba(255,255,255,0))]" />
        <div className="absolute top-1/4 right-0 w-96 h-96 bg-purple-600/10 rounded-full blur-[128px]" />
        <div className="absolute bottom-10 left-10 w-96 h-96 bg-blue-600/10 rounded-full blur-[128px]" />
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#1f293715_1px,transparent_1px),linear-gradient(to_bottom,#1f293715_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)]" />
      </div>

      {/* Main Container */}
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
        {/* Top Header Bar */}
        <header className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-6 border-b border-white/10 print:hidden">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="p-2.5 rounded-xl bg-white/5 hover:bg-white/10 text-neutral-400 hover:text-white border border-white/10 transition flex items-center gap-1.5 text-xs font-semibold"
            >
              <ArrowLeft className="w-4 h-4" />
              <span>Back to Hub</span>
            </Link>

            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl sm:text-2xl font-black tracking-tight text-white flex items-center gap-2">
                  <User className="w-5 h-5 text-blue-400" />
                  {twin?.persona_label || `Digital Twin (${twinId.substring(0, 12)})`}
                </h1>
                {twin?.is_synthetic && (
                  <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/30">
                    Synthetic Persona
                  </span>
                )}
                <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-purple-500/20 text-purple-300 border border-purple-500/30 font-mono">
                  Tier {twin?.state?.tier_used || "T1"}
                </span>
              </div>
              <p className="text-xs text-neutral-400 mt-0.5 font-mono">
                ID: {twinId} · Baseline Age: {twin?.demographics?.age ?? 74} · Sex:{" "}
                {twin?.demographics?.sex === 1 ? "Female" : "Male"} · APOE-ε4:{" "}
                {twin?.demographics?.apoe4_count ?? 1} allele(s)
              </p>
            </div>
          </div>

          {/* Action buttons & Persistent Model Scope Chip */}
          <div className="flex flex-wrap items-center gap-2.5">
            {/* Persistent Model Scope Chip */}
            <button
              onClick={() => setIsModelCardOpen(true)}
              className="px-3 py-1.5 rounded-xl bg-purple-950/40 hover:bg-purple-900/50 text-purple-300 border border-purple-500/30 text-xs font-semibold flex items-center gap-1.5 shadow-md transition cursor-pointer"
              title="Click to view full ADNI cohort validation, calibrated score bounds, and horizons"
            >
              <ShieldCheck className="w-4 h-4 text-purple-400" />
              <span>Model Scope: ADNI · Horizons ≤ 24m</span>
            </button>

            {/* Offline / Online indicator */}
            <div
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl border text-xs font-medium ${
                isOffline
                  ? "bg-amber-950/40 border-amber-500/30 text-amber-300"
                  : "bg-emerald-950/40 border-emerald-500/30 text-emerald-300"
              }`}
            >
              {isOffline ? (
                <>
                  <WifiOff className="w-3.5 h-3.5" />
                  <span>Offline / Fallback Data</span>
                </>
              ) : (
                <>
                  <Wifi className="w-3.5 h-3.5" />
                  <span>API Connected</span>
                </>
              )}
            </div>

            {/* Model Card Full Details */}
            <button
              onClick={() => setIsModelCardOpen(true)}
              className="px-3.5 py-1.5 rounded-xl bg-neutral-900 hover:bg-neutral-800 text-neutral-200 border border-white/15 hover:border-blue-400/50 shadow-md text-xs font-semibold flex items-center gap-1.5 transition cursor-pointer"
            >
              <FileCode className="w-4 h-4 text-blue-400" />
              <span>Model Card</span>
            </button>
          </div>
        </header>

        {/* GUIDED STEPPER TABS (Presenter can click any step freely) */}
        <nav aria-label="Digital Twin Workflow Steps" className="print:hidden">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 bg-neutral-900/60 backdrop-blur-xl p-2 rounded-2xl border border-white/10 shadow-lg">
            {stepsConfig.map((s) => {
              const Icon = s.icon;
              const isActive = activeStep === s.num;
              return (
                <button
                  key={s.num}
                  type="button"
                  onClick={() => setActiveStep(s.num as 1 | 2 | 3 | 4)}
                  className={`p-3.5 rounded-xl text-left transition flex items-center gap-3 cursor-pointer ${
                    isActive
                      ? "bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg shadow-purple-600/20 border border-white/20"
                      : "bg-white/[0.02] hover:bg-white/[0.05] text-neutral-400 hover:text-white border border-transparent"
                  }`}
                >
                  <div
                    className={`w-9 h-9 rounded-xl flex items-center justify-center font-bold text-sm shrink-0 ${
                      isActive
                        ? "bg-white/20 text-white"
                        : "bg-white/5 text-neutral-400 border border-white/10"
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                  </div>
                  <div className="min-w-0">
                    <span className="text-[10px] uppercase tracking-wider font-bold block opacity-80">
                      Step {s.num}
                    </span>
                    <span className="text-xs sm:text-sm font-bold truncate block text-white">
                      {s.title}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        </nav>

        {/* Loading State Skeleton */}
        {loading && !twin && (
          <div className="space-y-6">
            <div className="h-64 rounded-2xl bg-white/5 animate-pulse border border-white/10" />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="h-80 rounded-2xl bg-white/5 animate-pulse border border-white/10" />
              <div className="h-80 rounded-2xl bg-white/5 animate-pulse border border-white/10" />
            </div>
          </div>
        )}

        {/* Error State with retry */}
        {error && !twin && (
          <div className="p-8 rounded-2xl bg-rose-950/30 border border-rose-500/30 text-center space-y-4 max-w-xl mx-auto">
            <AlertCircle className="w-12 h-12 text-rose-400 mx-auto" />
            <h3 className="text-lg font-bold text-white">Could Not Load Digital Twin</h3>
            <p className="text-xs text-rose-200/80">{error}</p>
            <button
              onClick={loadTwinData}
              className="px-5 py-2.5 rounded-xl bg-rose-600 hover:bg-rose-500 text-white font-semibold text-xs transition flex items-center gap-2 mx-auto cursor-pointer"
            >
              <RefreshCw className="w-4 h-4" />
              <span>Retry Connection</span>
            </button>
          </div>
        )}

        {/* Active Step Content */}
        {twin && (
          <main className="transition-all">
            {activeStep === 1 && (
              <Step1Baseline twin={twin} onTwinUpdated={handleTwinUpdated} />
            )}
            {activeStep === 2 && (
              <Step2Visits twin={twin} onTwinUpdated={handleTwinUpdated} />
            )}
            {activeStep === 3 && (
              <Step3ScenarioLab twin={twin} onTwinUpdated={handleTwinUpdated} />
            )}
            {activeStep === 4 && (
              <Step4Profile twin={twin} />
            )}
          </main>
        )}
      </div>

      {/* Model Card Drawer Modal */}
      <ModelCardDrawer
        isOpen={isModelCardOpen}
        onClose={() => setIsModelCardOpen(false)}
      />
    </div>
  );
}
