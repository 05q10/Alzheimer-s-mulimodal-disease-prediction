"use client";

import React from "react";
import { TwinFull } from "@/app/lib/api";
import {
  Printer,
  Download,
  User,
  Shield,
  Activity,
  Calendar,
  Layers,
  Dna,
  FileText,
  Clock,
  TrendingDown,
  Gauge,
  CheckCircle2,
  XCircle,
  AlertTriangle,
} from "lucide-react";
import LifestyleModifierCard from "./LifestyleModifierCard";
import VisitHistorySection from "./VisitHistorySection";

interface Step4ProfileProps {
  twin: TwinFull;
}

export default function Step4Profile({ twin }: Step4ProfileProps) {
  const { demographics, state, modalities, visits, history } = twin;

  // Handle Download JSON
  const handleDownloadJson = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(twin, null, 2));
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `digital-twin-${twin.twin_id}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  // Handle Print / Save as PDF
  const handlePrint = () => {
    if (typeof window !== "undefined") {
      window.print();
    }
  };

  const stageColor =
    state.stage === "CN" ? "text-emerald-400" : state.stage === "MCI" ? "text-amber-400" : "text-rose-400";
  const stageBadgeBg =
    state.stage === "CN" ? "bg-emerald-500/20 border-emerald-500/30" : state.stage === "MCI" ? "bg-amber-500/20 border-amber-500/30" : "bg-rose-500/20 border-rose-500/30";

  return (
    <div className="space-y-8">
      {/* Action Header (Hidden in Print) */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 bg-neutral-900/60 backdrop-blur-xl border border-white/10 p-5 rounded-2xl shadow-xl print:hidden">
        <div>
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <User className="w-4 h-4 text-blue-400" />
            <span>Comprehensive Digital Twin Patient Profile</span>
          </h2>
          <p className="text-xs text-neutral-400 mt-0.5">
            One-page structured clinical report summarizing demographics, modalities, forecast, and longitudinal history.
          </p>
        </div>

        <div className="flex items-center gap-3 self-end sm:self-auto">
          <button
            onClick={handlePrint}
            className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs transition flex items-center gap-2 shadow-lg shadow-blue-600/30 cursor-pointer"
          >
            <Printer className="w-4 h-4" />
            <span>Print / Save as PDF</span>
          </button>

          <button
            onClick={handleDownloadJson}
            className="px-4 py-2 rounded-xl bg-neutral-800 hover:bg-neutral-700 text-neutral-200 border border-white/10 font-semibold text-xs transition flex items-center gap-2 cursor-pointer"
          >
            <Download className="w-4 h-4 text-purple-400" />
            <span>Download twin JSON</span>
          </button>
        </div>
      </div>

      {/* PRINT-OPTIMIZED ONE-PAGE CLINICAL REPORT CONTAINER */}
      <div className="bg-neutral-900/80 print:bg-white print:text-black backdrop-blur-2xl border border-white/10 print:border-neutral-300 p-8 sm:p-10 rounded-3xl shadow-2xl space-y-8 print:p-0 print:space-y-6">
        {/* Report Top Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-6 border-b border-white/10 print:border-neutral-300 gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-bold tracking-widest text-blue-400 uppercase">
                Alzheimer's Multi-Modal Digital Twin Dossier
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-purple-500/20 text-purple-300 border border-purple-500/30 font-semibold">
                Tier {state.tier_used}
              </span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-black text-white print:text-black tracking-tight mt-1">
              {twin.persona_label || "Patient-Specific Twin"}
            </h1>
            <p className="text-xs text-neutral-400 print:text-neutral-600 font-mono mt-0.5">
              Twin ID: {twin.twin_id} · Generated: {new Date().toLocaleDateString()}
            </p>
          </div>

          <div className="text-right">
            <span className={`inline-block px-3 py-1.5 rounded-xl border text-sm font-bold ${stageBadgeBg} ${stageColor} print:border-neutral-400 print:text-black`}>
              Current Stage: {state.stage}
            </span>
            <p className="text-[11px] text-neutral-400 print:text-neutral-600 mt-1">
              Model Confidence: {(state.confidence * 100).toFixed(1)}%
            </p>
          </div>
        </div>

        {/* Section 1: Demographics & Clinical Attributes */}
        <div>
          <h3 className="text-xs font-bold text-neutral-300 print:text-neutral-800 uppercase tracking-wider mb-3 flex items-center gap-2">
            <User className="w-3.5 h-3.5 text-blue-400 print:text-blue-700" />
            <span>1. Baseline Demographics & Genetic Risk Profile</span>
          </h3>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 p-4 rounded-2xl bg-white/[0.02] print:bg-neutral-50 border border-white/5 print:border-neutral-200">
            <div>
              <span className="text-[11px] text-neutral-400 print:text-neutral-600 block">Baseline Age</span>
              <span className="text-sm font-bold text-white print:text-black">{demographics?.age ?? 74} years</span>
            </div>
            <div>
              <span className="text-[11px] text-neutral-400 print:text-neutral-600 block">Biological Sex</span>
              <span className="text-sm font-bold text-white print:text-black">
                {demographics?.sex === 1 ? "Female" : demographics?.sex === 0 ? "Male" : "Not specified"}
              </span>
            </div>
            <div>
              <span className="text-[11px] text-neutral-400 print:text-neutral-600 block">Education</span>
              <span className="text-sm font-bold text-white print:text-black">
                {demographics?.education_years ?? 16} years
              </span>
            </div>
            <div>
              <span className="text-[11px] text-neutral-400 print:text-neutral-600 block">APOE-ε4 Allele Count</span>
              <span className="text-sm font-bold text-purple-300 print:text-purple-700 font-mono">
                {demographics?.apoe4_count ?? 1} allele(s) (Elevated Risk)
              </span>
            </div>
          </div>
        </div>

        {/* Section 2: Multimodal Cascades Status (Present vs Missing) */}
        <div>
          <h3 className="text-xs font-bold text-neutral-300 print:text-neutral-800 uppercase tracking-wider mb-3 flex items-center gap-2">
            <Layers className="w-3.5 h-3.5 text-purple-400 print:text-purple-700" />
            <span>2. Multimodal Data Ingestion (Cascades Present vs Missing)</span>
          </h3>

          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            {[
              {
                id: "basic_scores",
                label: "Cognitive Scores",
                detail: `MMSE ${visits[0]?.MMSE ?? 26}, CDR ${visits[0]?.CDGLOBAL ?? 0.5}`,
                present: modalities.basic_scores?.present,
              },
              {
                id: "biomarkers",
                label: "Fluid Biomarkers",
                detail: modalities.biomarkers?.present ? "Aβ42, t-tau, p-tau calibrated" : "Missing / Unobserved",
                present: modalities.biomarkers?.present,
              },
              {
                id: "mri_volumes",
                label: "Volumetric MRI",
                detail: modalities.mri_volumes?.present ? "Hippocampal/ventricular volumes" : "Missing / Unobserved",
                present: modalities.mri_volumes?.present,
              },
              {
                id: "mri_image",
                label: "MRI Scan (ResNet)",
                detail: modalities.mri_image?.present ? "demo-only, not validated on ADNI" : "Not uploaded",
                present: modalities.mri_image?.present,
              },
              {
                id: "ocr_report",
                label: "Extracted Lab OCR",
                detail: modalities.ocr_report?.present ? "Blood panel metrics ingested" : "Not uploaded",
                present: modalities.ocr_report?.present,
              },
            ].map((m) => (
              <div
                key={m.id}
                className="p-3 rounded-xl bg-white/[0.02] print:bg-neutral-50 border border-white/5 print:border-neutral-200"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-bold text-white print:text-black">{m.label}</span>
                  {m.present ? (
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                  ) : (
                    <XCircle className="w-3.5 h-3.5 text-neutral-500" />
                  )}
                </div>
                <p className="text-[10px] text-neutral-400 print:text-neutral-600 line-clamp-2">{m.detail}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Section 3: Diagnostic Stage Probabilities & Conversion Risk */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          <div className="p-4 rounded-2xl bg-white/[0.02] print:bg-neutral-50 border border-white/5 print:border-neutral-200 space-y-3">
            <h4 className="text-xs font-bold text-neutral-300 print:text-neutral-800 uppercase tracking-wider flex items-center gap-2">
              <Activity className="w-3.5 h-3.5 text-blue-400 print:text-blue-700" />
              <span>Current Stage Probability Distribution</span>
            </h4>
            <div className="grid grid-cols-3 gap-2 text-center">
              <div className="p-2 rounded-xl bg-neutral-800/40 print:bg-neutral-100">
                <span className="text-[10px] text-neutral-400 print:text-neutral-600 block">CN</span>
                <span className="text-base font-bold text-emerald-400 print:text-emerald-700 font-mono">
                  {((state.stage_probs.CN || 0) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="p-2 rounded-xl bg-neutral-800/40 print:bg-neutral-100">
                <span className="text-[10px] text-neutral-400 print:text-neutral-600 block">MCI</span>
                <span className="text-base font-bold text-amber-400 print:text-amber-700 font-mono">
                  {((state.stage_probs.MCI || 0) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="p-2 rounded-xl bg-neutral-800/40 print:bg-neutral-100">
                <span className="text-[10px] text-neutral-400 print:text-neutral-600 block">AD</span>
                <span className="text-base font-bold text-rose-400 print:text-rose-700 font-mono">
                  {((state.stage_probs.AD || 0) * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>

          <div className="p-4 rounded-2xl bg-white/[0.02] print:bg-neutral-50 border border-white/5 print:border-neutral-200 space-y-3">
            <h4 className="text-xs font-bold text-neutral-300 print:text-neutral-800 uppercase tracking-wider flex items-center gap-2">
              <Gauge className="w-3.5 h-3.5 text-purple-400 print:text-purple-700" />
              <span>Progression & Conversion Probability</span>
            </h4>
            <div className="grid grid-cols-2 gap-2 text-center">
              <div className="p-2 rounded-xl bg-neutral-800/40 print:bg-neutral-100">
                <span className="text-[10px] text-neutral-400 print:text-neutral-600 block">12-Month Conversion</span>
                <span className="text-base font-bold text-purple-300 print:text-purple-800 font-mono">
                  {((state.conversion_risk?.["12m"] ?? 0) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="p-2 rounded-xl bg-neutral-800/40 print:bg-neutral-100">
                <span className="text-[10px] text-neutral-400 print:text-neutral-600 block">24-Month Conversion</span>
                <span className="text-base font-bold text-purple-300 print:text-purple-800 font-mono">
                  {((state.conversion_risk?.["24m"] ?? 0) * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Lifestyle & Risk Factors Card (Requirement 6) */}
        <div>
          <LifestyleModifierCard twin={twin} />
        </div>

        {/* Section 4: Longitudinal Trajectory Forecast Projections */}
        <div>
          <h3 className="text-xs font-bold text-neutral-300 print:text-neutral-800 uppercase tracking-wider mb-3 flex items-center gap-2">
            <TrendingDown className="w-3.5 h-3.5 text-blue-400 print:text-blue-700" />
            <span>3. Model Trajectory Forecast (Mean & 90% Bootstrap Confidence Interval)</span>
          </h3>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-white/10 print:border-neutral-300 text-neutral-400 print:text-neutral-600">
                  <th className="py-2.5 px-3">Horizon</th>
                  <th className="py-2.5 px-3">Projected MMSE (90% CI)</th>
                  <th className="py-2.5 px-3">Projected CDR-Global</th>
                  <th className="py-2.5 px-3">Projected ADAS11-Cog</th>
                  <th className="py-2.5 px-3">Conversion Probability</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5 print:divide-neutral-200 font-mono">
                {Object.entries(state.forecast || {}).map(([horizon, pt]) => (
                  <tr key={horizon} className="hover:bg-white/[0.01]">
                    <td className="py-2.5 px-3 font-bold text-white print:text-black">Month {horizon}</td>
                    <td className="py-2.5 px-3 text-purple-300 print:text-purple-900">
                      {pt.MMSE_mean?.toFixed(1)} [{pt.MMSE_lo?.toFixed(1)} – {pt.MMSE_hi?.toFixed(1)}]
                    </td>
                    <td className="py-2.5 px-3 text-neutral-300 print:text-neutral-800">
                      {pt.CDGLOBAL_mean?.toFixed(2)}
                    </td>
                    <td className="py-2.5 px-3 text-neutral-300 print:text-neutral-800">
                      {pt.TOTSCORE_mean?.toFixed(1)}
                    </td>
                    <td className="py-2.5 px-3 font-bold text-rose-300 print:text-rose-800">
                      {((pt.conversion_prob || 0) * 100).toFixed(0)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Section 5: Recorded Longitudinal Visits */}
        <div>
          <h3 className="text-xs font-bold text-neutral-300 print:text-neutral-800 uppercase tracking-wider mb-3 flex items-center gap-2">
            <Calendar className="w-3.5 h-3.5 text-blue-400 print:text-blue-700" />
            <span>4. Longitudinal Observation Records</span>
          </h3>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-white/10 print:border-neutral-300 text-neutral-400 print:text-neutral-600">
                  <th className="py-2 px-3">Visit Month</th>
                  <th className="py-2 px-3">MMSE</th>
                  <th className="py-2 px-3">CDR-Global</th>
                  <th className="py-2 px-3">ADAS-Cog</th>
                  <th className="py-2 px-3">FAQ</th>
                  <th className="py-2 px-3">MoCA</th>
                  <th className="py-2 px-3">Source</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5 print:divide-neutral-200 font-mono">
                {(visits || []).map((v, i) => (
                  <tr key={`profile-visit-${v.month}-${i}`}>
                    <td className="py-2 px-3 font-bold text-white print:text-black">
                      Month {v.month} {v.month === 0 && "(Baseline)"}
                    </td>
                    <td className="py-2 px-3 text-neutral-200 print:text-black">{v.MMSE ?? "—"}</td>
                    <td className="py-2 px-3 text-neutral-200 print:text-black">{v.CDGLOBAL ?? "—"}</td>
                    <td className="py-2 px-3 text-neutral-200 print:text-black">{v.TOTSCORE ?? "—"}</td>
                    <td className="py-2 px-3 text-neutral-200 print:text-black">{v.FAQ ?? "—"}</td>
                    <td className="py-2 px-3 text-neutral-200 print:text-black">{v.MOCA ?? "—"}</td>
                    <td className="py-2 px-3 text-neutral-400 print:text-neutral-600 capitalize">{v.source || "user"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Visit History Section with Visit Cards, Comparison and Export */}
        <div className="pt-4 border-t border-white/10 print:hidden">
          <VisitHistorySection twin={twin} />
        </div>

        {/* Regulatory Footer */}
        <div className="pt-6 border-t border-white/10 print:border-neutral-300 text-[10px] text-neutral-500 print:text-neutral-600 space-y-1">
          <p>
            <strong>Research & Educational Use Only:</strong> This Digital Twin report was generated using longitudinal statistical progression models trained on the ADNI cohort.
            Image-based ResNet classification and multimodal fusion are designated as demo-only and are not validated on external clinical cohorts.
          </p>
          <p>
            Model Version: 2.4.0 · Training Cohort: ADNI 1/GO/2/3 Longitudinal Academic Cohort · Valid horizons: 6, 12, 24 months.
          </p>
        </div>
      </div>
    </div>
  );
}
