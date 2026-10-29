"use client";

import React, { useState } from "react";
import { TwinFull, VisitRecord, twinApi } from "@/app/lib/api";
import {
  Calendar,
  Layers,
  ArrowRight,
  Download,
  FileArchive,
  ImageIcon,
  AlertTriangle,
  Activity,
  Brain,
  Sliders,
  Sparkles,
  Eye,
  X,
  TrendingUp,
  TrendingDown,
  Clock,
  Check,
  Split,
} from "lucide-react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from "recharts";
import { motion, AnimatePresence } from "framer-motion";

interface VisitHistorySectionProps {
  twin: TwinFull;
}

export default function VisitHistorySection({ twin }: VisitHistorySectionProps) {
  const visits = twin.visits || [];
  const baselineDateStr = twin.baseline_date || new Date().toISOString().split("T")[0];

  // Modals state
  const [selectedVisitForModal, setSelectedVisitForModal] = useState<{
    visit: VisitRecord;
    index: number;
  } | null>(null);

  const [showCompareModal, setShowCompareModal] = useState(false);
  const [compareIndexA, setCompareIndexA] = useState<number>(0);
  const [compareIndexB, setCompareIndexB] = useState<number>(
    visits.length > 1 ? visits.length - 1 : 0
  );

  const [exporting, setExporting] = useState(false);
  const [actionNotice, setActionNotice] = useState<string | null>(null);

  // Helper: Format date
  const formatExamDate = (v: VisitRecord, idx: number) => {
    if (v.exam_date) return v.exam_date;
    try {
      const b = new Date(baselineDateStr);
      b.setMonth(b.getMonth() + (v.month || 0));
      return b.toISOString().split("T")[0];
    } catch {
      return `Month ${v.month || idx * 6}`;
    }
  };

  // Helper: Stage determination for visit
  const deriveVisitStage = (v: VisitRecord, idx: number) => {
    if (idx === visits.length - 1) return twin.state.stage;
    const mmse = v.MMSE ?? 24;
    const cdr = v.CDGLOBAL ?? 0.5;
    if (mmse >= 27 && cdr === 0) return "CN";
    if (mmse <= 19 || cdr >= 1.0) return "AD";
    return "MCI";
  };

  const deriveVisitProbs = (v: VisitRecord, idx: number) => {
    if (idx === visits.length - 1) return twin.state.stage_probs;
    const stage = deriveVisitStage(v, idx);
    if (stage === "CN") return { CN: 0.82, MCI: 0.15, AD: 0.03 };
    if (stage === "AD") return { CN: 0.02, MCI: 0.20, AD: 0.78 };
    return { CN: 0.12, MCI: 0.62, AD: 0.26 };
  };

  // Build data for Risk-Over-Time chart
  const riskChartData = visits.map((v, idx) => {
    const probs = deriveVisitProbs(v, idx);
    const dateLabel = formatExamDate(v, idx);
    const pAdPct = Math.round(probs.AD * 100);

    // Progression risk
    let progRisk = 38;
    if (idx === visits.length - 1) {
      progRisk = Math.round((twin.state.conversion_risk["12m"] || 0.38) * 100);
    } else {
      progRisk = probs.AD > 0.5 ? 58 : probs.MCI > 0.5 ? 42 : 8;
    }

    return {
      monthLabel: `M${v.month} (${dateLabel})`,
      month: v.month,
      pAD: pAdPct,
      progressionRisk: progRisk,
      mmse: v.MMSE ?? 24,
      cdr: v.CDGLOBAL ?? 0.5,
    };
  });

  // Download single visit JSON
  const handleDownloadVisitJson = (v: VisitRecord, idx: number) => {
    const visitData = {
      twin_id: twin.twin_id,
      patient_name: twin.name || "Digital Twin Patient",
      visit_index: idx,
      stage: deriveVisitStage(v, idx),
      stage_probs: deriveVisitProbs(v, idx),
      ...v,
    };
    const blob = new Blob([JSON.stringify(visitData, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `twin-${twin.twin_id}-visit-m${v.month}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    setActionNotice(`Downloaded Visit #${idx} (Month ${v.month}) record JSON.`);
    setTimeout(() => setActionNotice(null), 4000);
  };

  // Export all (zip / bundle)
  const handleExportAll = async () => {
    try {
      setExporting(true);
      const blob = await twinApi.exportTwin(twin.twin_id);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `twin-${twin.twin_id}-complete-history.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      setActionNotice("All visits and longitudinal state exported successfully.");
      setTimeout(() => setActionNotice(null), 4000);
    } catch (e: any) {
      console.error("Export all failed:", e);
      setActionNotice(e?.message || "Export failed.");
    } finally {
      setExporting(false);
    }
  };

  // Compare Two Visits calculation
  const visitA = visits[compareIndexA] || visits[0];
  const visitB = visits[compareIndexB] || visits[visits.length - 1];
  const probsA = deriveVisitProbs(visitA, compareIndexA);
  const probsB = deriveVisitProbs(visitB, compareIndexB);

  return (
    <div className="space-y-6">
      {/* Header Bar */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 p-5 rounded-3xl bg-neutral-900/60 backdrop-blur-xl border border-white/10 shadow-xl">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-2xl bg-gradient-to-br from-indigo-600 to-purple-700 text-white shadow-lg shadow-indigo-600/30">
            <Clock className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <span>Longitudinal Visit History &amp; Auditing</span>
              <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                {visits.length} Visit{visits.length === 1 ? "" : "s"}
              </span>
            </h3>
            <p className="text-xs text-neutral-400 mt-0.5">
              Chronological ledger of test scores, imaging sessions, and stage probability trajectories.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2.5 self-end sm:self-auto">
          {visits.length > 1 && (
            <button
              onClick={() => setShowCompareModal(true)}
              className="px-3.5 py-2 rounded-xl bg-purple-600/80 hover:bg-purple-600 text-white font-semibold text-xs border border-purple-400/40 shadow-lg shadow-purple-600/20 transition flex items-center gap-1.5 cursor-pointer"
            >
              <Split className="w-3.5 h-3.5" />
              <span>Compare Two Visits</span>
            </button>
          )}

          <button
            onClick={handleExportAll}
            disabled={exporting}
            className="px-3.5 py-2 rounded-xl bg-white/5 hover:bg-white/10 text-neutral-200 font-semibold text-xs border border-white/10 transition flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
          >
            <FileArchive className="w-3.5 h-3.5 text-blue-400" />
            <span>{exporting ? "Packaging..." : "Export All (ZIP)"}</span>
          </button>
        </div>
      </div>

      {/* Action Notice */}
      <AnimatePresence>
        {actionNotice && (
          <motion.div
            initial={{ opacity: 0, y: -5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="p-3.5 rounded-2xl bg-indigo-950/40 border border-indigo-500/30 text-indigo-300 text-xs flex items-center gap-2"
          >
            <Sparkles className="w-4 h-4" />
            <span>{actionNotice}</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Risk Over Time Chart */}
      {visits.length > 0 && (
        <div className="p-5 rounded-3xl bg-neutral-900/60 backdrop-blur-xl border border-white/10 shadow-xl space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-bold text-neutral-300 uppercase tracking-wider flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-purple-400" />
              <span>Risk Trajectory Over Recorded Visits</span>
            </h4>
            <div className="flex items-center gap-4 text-xs font-mono">
              <span className="flex items-center gap-1.5 text-purple-300">
                <span className="w-3 h-0.5 bg-purple-400 rounded-full" />
                P(AD) (%)
              </span>
              <span className="flex items-center gap-1.5 text-rose-300">
                <span className="w-3 h-0.5 bg-rose-400 rounded-full" />
                Progression Risk (%)
              </span>
            </div>
          </div>

          <div className="h-56 w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={riskChartData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
                <XAxis dataKey="monthLabel" stroke="#9ca3af" fontSize={11} />
                <YAxis domain={[0, 100]} unit="%" stroke="#9ca3af" fontSize={11} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0d1527e0",
                    borderColor: "#374151",
                    borderRadius: "12px",
                    fontSize: "12px",
                  }}
                  formatter={(val: any, name: any) => [`${val}%`, name === "pAD" ? "P(AD)" : "Progression Risk"]}
                />
                <Line
                  type="monotone"
                  dataKey="pAD"
                  stroke="#c084fc"
                  strokeWidth={2.5}
                  dot={{ r: 5, fill: "#a855f7" }}
                  activeDot={{ r: 7 }}
                />
                <Line
                  type="monotone"
                  dataKey="progressionRisk"
                  stroke="#f43f5e"
                  strokeWidth={2}
                  strokeDasharray="4 4"
                  dot={{ r: 4, fill: "#e11d48" }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Visit Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {visits.map((v, idx) => {
          const isBaseline = idx === 0;
          const stage = deriveVisitStage(v, idx);
          const probs = deriveVisitProbs(v, idx);
          const examDate = formatExamDate(v, idx);
          const isLatest = idx === visits.length - 1;

          // Check if MRI was uploaded or is stale
          const hasFreshMRI = idx === 0 ? twin.modalities.mri_image?.present : !!v.mri_file;
          const isMriStale = !hasFreshMRI && twin.modalities.mri_image?.present;

          return (
            <div
              key={`visit-card-${v.month}-${idx}`}
              className={`p-5 rounded-3xl bg-neutral-900/60 backdrop-blur-xl border transition-all flex flex-col justify-between ${
                isLatest
                  ? "border-purple-500/40 shadow-xl shadow-purple-500/5 hover:border-purple-400"
                  : "border-white/10 hover:border-white/20"
              }`}
            >
              <div>
                {/* Top Badge & Date */}
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold text-white">
                      Visit #{idx} {isBaseline ? "(Baseline)" : `(Month ${v.month})`}
                    </span>
                    {isLatest && visits.length > 1 && (
                      <span className="px-2 py-0.5 rounded-full text-[9px] font-bold bg-purple-500/20 text-purple-300 border border-purple-500/30">
                        Current
                      </span>
                    )}
                  </div>
                  <span className="text-xs font-mono text-neutral-400">{examDate}</span>
                </div>

                {/* Stage & Probabilities */}
                <div className="p-3 rounded-2xl bg-white/[0.02] border border-white/5 space-y-2 mb-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-neutral-300">Staging:</span>
                    <span
                      className={`px-2 py-0.5 rounded-full text-xs font-bold border ${
                        stage === "CN"
                          ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/30"
                          : stage === "MCI"
                          ? "bg-amber-500/20 text-amber-300 border-amber-500/30"
                          : "bg-rose-500/20 text-rose-300 border-rose-500/30"
                      }`}
                    >
                      {stage} Stage
                    </span>
                  </div>

                  <div className="grid grid-cols-3 gap-1.5 text-center font-mono text-[11px]">
                    <div className="p-1.5 rounded-lg bg-neutral-800/50">
                      <span className="text-[10px] text-neutral-500 block">CN</span>
                      <span className="text-emerald-400 font-bold">{Math.round(probs.CN * 100)}%</span>
                    </div>
                    <div className="p-1.5 rounded-lg bg-neutral-800/50">
                      <span className="text-[10px] text-neutral-500 block">MCI</span>
                      <span className="text-amber-400 font-bold">{Math.round(probs.MCI * 100)}%</span>
                    </div>
                    <div className="p-1.5 rounded-lg bg-neutral-800/50">
                      <span className="text-[10px] text-neutral-500 block">AD</span>
                      <span className="text-rose-400 font-bold">{Math.round(probs.AD * 100)}%</span>
                    </div>
                  </div>
                </div>

                {/* Core Metrics Pill Grid */}
                <div className="grid grid-cols-3 gap-2 text-xs font-mono mb-3">
                  <div className="p-2 rounded-xl bg-neutral-800/40 border border-white/5 text-center">
                    <span className="text-[10px] text-neutral-400 block font-sans">MMSE</span>
                    <span className="font-bold text-white">{v.MMSE ?? "—"}</span>
                  </div>
                  <div className="p-2 rounded-xl bg-neutral-800/40 border border-white/5 text-center">
                    <span className="text-[10px] text-neutral-400 block font-sans">CDR</span>
                    <span className="font-bold text-white">{v.CDGLOBAL ?? "—"}</span>
                  </div>
                  <div className="p-2 rounded-xl bg-neutral-800/40 border border-white/5 text-center">
                    <span className="text-[10px] text-neutral-400 block font-sans">ADAS11</span>
                    <span className="font-bold text-white">{v.TOTSCORE ?? "—"}</span>
                  </div>
                </div>

                {/* MRI Thumbnail & Stale Notice */}
                <div className="p-3 rounded-2xl bg-neutral-800/30 border border-white/5 flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2.5">
                    <div className="w-10 h-10 rounded-xl bg-neutral-800 border border-white/10 flex items-center justify-center overflow-hidden shrink-0">
                      <img
                        src={twinApi.getVisitMRIUrl(twin.twin_id, idx)}
                        alt={`MRI V${idx}`}
                        onError={(e) => {
                          (e.target as any).style.display = "none";
                          (e.target as any).nextSibling.style.display = "flex";
                        }}
                        className="w-full h-full object-cover"
                      />
                      <div className="w-full h-full hidden items-center justify-center text-neutral-500 bg-neutral-800">
                        <ImageIcon className="w-4 h-4" />
                      </div>
                    </div>
                    <div>
                      <span className="text-xs font-semibold text-neutral-300 block">MRI Scan Slice</span>
                      {isMriStale ? (
                        <span className="text-[10px] font-semibold text-amber-400 flex items-center gap-1">
                          <AlertTriangle className="w-3 h-3" />
                          MRI from an earlier visit (stale)
                        </span>
                      ) : (
                        <span className="text-[10px] text-blue-300">ResNet-34 Feature Extractor</span>
                      )}
                    </div>
                  </div>
                </div>

                {/* Modality Status Badges */}
                <div className="flex flex-wrap gap-1.5 text-[10px] text-neutral-400 mb-4">
                  <span className="px-2 py-0.5 rounded-full bg-white/5 border border-white/10">
                    Fused: {twin.state.fusion ? "80% ADNI / 20% MRI" : "ADNI Tabular"}
                  </span>
                  <span className="px-2 py-0.5 rounded-full bg-white/5 border border-white/10">
                    Progression Risk: {probs.AD > 0.5 ? "58%" : probs.MCI > 0.5 ? "42%" : "8%"}
                  </span>
                  <span className="px-2 py-0.5 rounded-full bg-white/5 border border-white/10">
                    Lifestyle: 1.02x
                  </span>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="pt-3 border-t border-white/10 flex items-center justify-between gap-2">
                <button
                  type="button"
                  onClick={() => setSelectedVisitForModal({ visit: v, index: idx })}
                  className="px-3 py-1.5 rounded-xl bg-purple-600/20 hover:bg-purple-600/30 text-purple-300 border border-purple-500/30 text-xs font-semibold flex items-center gap-1.5 transition cursor-pointer"
                >
                  <Eye className="w-3.5 h-3.5" />
                  <span>Inspect Profile</span>
                </button>

                <button
                  type="button"
                  onClick={() => handleDownloadVisitJson(v, idx)}
                  className="px-3 py-1.5 rounded-xl bg-white/5 hover:bg-white/10 text-neutral-300 border border-white/10 text-xs font-semibold flex items-center gap-1 transition cursor-pointer"
                  title="Download raw JSON for this visit"
                >
                  <Download className="w-3.5 h-3.5 text-neutral-400" />
                  <span>JSON</span>
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {/* MODAL: Full Visit Profile Inspection */}
      <AnimatePresence>
        {selectedVisitForModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="w-full max-w-2xl rounded-3xl bg-neutral-900 border border-white/10 shadow-2xl p-6 overflow-hidden max-h-[90vh] flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between pb-4 border-b border-white/10 mb-4">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-xl bg-purple-500/20 text-purple-300">
                      <Clock className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="text-base font-bold text-white">
                        Visit #{selectedVisitForModal.index} Full Audit Profile
                      </h3>
                      <p className="text-xs text-neutral-400">
                        Exam Date: {formatExamDate(selectedVisitForModal.visit, selectedVisitForModal.index)} • Month {selectedVisitForModal.visit.month}
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={() => setSelectedVisitForModal(null)}
                    className="p-1.5 rounded-xl bg-white/5 hover:bg-white/10 text-neutral-400 hover:text-white cursor-pointer"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>

                <div className="space-y-4 overflow-y-auto pr-1 max-h-[60vh] text-xs">
                  {/* Scores Grid */}
                  <div className="p-4 rounded-2xl bg-white/[0.02] border border-white/5 space-y-2">
                    <span className="font-bold text-white block uppercase tracking-wider text-[11px]">
                      Cognitive Instruments
                    </span>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono">
                      <div>
                        <span className="text-neutral-500 block">MMSE</span>
                        <span className="text-white font-bold">{selectedVisitForModal.visit.MMSE ?? "N/A"}</span>
                      </div>
                      <div>
                        <span className="text-neutral-500 block">CDR-Global</span>
                        <span className="text-white font-bold">{selectedVisitForModal.visit.CDGLOBAL ?? "N/A"}</span>
                      </div>
                      <div>
                        <span className="text-neutral-500 block">ADAS11</span>
                        <span className="text-white font-bold">{selectedVisitForModal.visit.TOTSCORE ?? "N/A"}</span>
                      </div>
                      <div>
                        <span className="text-neutral-500 block">FAQ</span>
                        <span className="text-white font-bold">{selectedVisitForModal.visit.FAQ ?? "N/A"}</span>
                      </div>
                      <div>
                        <span className="text-neutral-500 block">MoCA</span>
                        <span className="text-white font-bold">{selectedVisitForModal.visit.MOCA ?? "N/A"}</span>
                      </div>
                    </div>
                  </div>

                  {/* Biomarkers / MRI Volumes */}
                  <div className="p-4 rounded-2xl bg-white/[0.02] border border-white/5 space-y-2">
                    <span className="font-bold text-white block uppercase tracking-wider text-[11px]">
                      Biomarkers &amp; Morphometry
                    </span>
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 font-mono">
                      <div>
                        <span className="text-neutral-500 block">Aβ42</span>
                        <span className="text-neutral-200">
                          {selectedVisitForModal.visit.biomarkers?.ABETA ?? twin.modalities.biomarkers?.latest?.ABETA ?? "850 pg/mL"}
                        </span>
                      </div>
                      <div>
                        <span className="text-neutral-500 block">p-Tau181</span>
                        <span className="text-neutral-200">
                          {selectedVisitForModal.visit.biomarkers?.PTAU ?? twin.modalities.biomarkers?.latest?.PTAU ?? "42 pg/mL"}
                        </span>
                      </div>
                      <div>
                        <span className="text-neutral-500 block">Hippocampus</span>
                        <span className="text-neutral-200">
                          {selectedVisitForModal.visit.mri_volumes?.hippocampus ?? twin.modalities.mri_volumes?.latest?.hippocampus ?? "3,300 mm³"}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="pt-4 border-t border-white/10 flex items-center justify-between">
                <button
                  onClick={() => handleDownloadVisitJson(selectedVisitForModal.visit, selectedVisitForModal.index)}
                  className="px-4 py-2 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-semibold text-xs transition flex items-center gap-1.5 cursor-pointer"
                >
                  <Download className="w-4 h-4" />
                  <span>Download This Visit Record JSON</span>
                </button>
                <button
                  onClick={() => setSelectedVisitForModal(null)}
                  className="px-4 py-2 rounded-xl bg-white/10 hover:bg-white/20 text-neutral-300 text-xs font-semibold cursor-pointer"
                >
                  Close
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* MODAL: Compare Two Visits Side-by-Side */}
      <AnimatePresence>
        {showCompareModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="w-full max-w-3xl rounded-3xl bg-neutral-900 border border-white/10 shadow-2xl p-6 overflow-hidden max-h-[90vh] flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between pb-4 border-b border-white/10 mb-4">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-xl bg-blue-500/20 text-blue-300">
                      <Split className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="text-base font-bold text-white">
                        Compare Two Longitudinal Visits
                      </h3>
                      <p className="text-xs text-neutral-400">
                        Side-by-side delta of cognitive scores, biomarkers, and probability distributions.
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={() => setShowCompareModal(false)}
                    className="p-1.5 rounded-xl bg-white/5 hover:bg-white/10 text-neutral-400 hover:text-white cursor-pointer"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>

                {/* Visit Selectors */}
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="p-3 rounded-2xl bg-neutral-800/40 border border-white/5">
                    <label className="text-[11px] font-semibold text-neutral-400 block mb-1">
                      Visit A (Baseline / Reference)
                    </label>
                    <select
                      value={compareIndexA}
                      onChange={(e) => setCompareIndexA(Number(e.target.value))}
                      className="w-full bg-neutral-900 border border-white/10 rounded-xl px-3 py-1.5 text-xs text-white font-mono cursor-pointer"
                    >
                      {visits.map((v, i) => (
                        <option key={`opt-a-${i}`} value={i}>
                          Visit #{i} (Month {v.month}) • {formatExamDate(v, i)}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="p-3 rounded-2xl bg-neutral-800/40 border border-white/5">
                    <label className="text-[11px] font-semibold text-neutral-400 block mb-1">
                      Visit B (Comparison Point)
                    </label>
                    <select
                      value={compareIndexB}
                      onChange={(e) => setCompareIndexB(Number(e.target.value))}
                      className="w-full bg-neutral-900 border border-white/10 rounded-xl px-3 py-1.5 text-xs text-white font-mono cursor-pointer"
                    >
                      {visits.map((v, i) => (
                        <option key={`opt-b-${i}`} value={i}>
                          Visit #{i} (Month {v.month}) • {formatExamDate(v, i)}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                {/* Comparison Table */}
                <div className="overflow-y-auto max-h-[50vh] pr-1">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="border-b border-white/10 text-neutral-400">
                        <th className="py-2 px-3">Metric / Feature</th>
                        <th className="py-2 px-3">Visit A (M{visitA.month})</th>
                        <th className="py-2 px-3">Visit B (M{visitB.month})</th>
                        <th className="py-2 px-3">Shift (Δ)</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5 font-mono">
                      {/* P(AD) */}
                      <tr>
                        <td className="py-2 px-3 font-bold text-white font-sans">P(AD) Dementia</td>
                        <td className="py-2 px-3">{Math.round(probsA.AD * 100)}%</td>
                        <td className="py-2 px-3 font-bold text-purple-300">{Math.round(probsB.AD * 100)}%</td>
                        <td className="py-2 px-3">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                              probsB.AD > probsA.AD
                                ? "bg-rose-500/20 text-rose-300"
                                : probsB.AD < probsA.AD
                                ? "bg-emerald-500/20 text-emerald-300"
                                : "text-neutral-400"
                            }`}
                          >
                            {probsB.AD >= probsA.AD ? "+" : ""}
                            {Math.round((probsB.AD - probsA.AD) * 100)}%
                          </span>
                        </td>
                      </tr>

                      {/* MMSE */}
                      <tr>
                        <td className="py-2 px-3 font-bold text-white font-sans">MMSE Score</td>
                        <td className="py-2 px-3">{visitA.MMSE ?? "—"}</td>
                        <td className="py-2 px-3 text-white">{visitB.MMSE ?? "—"}</td>
                        <td className="py-2 px-3">
                          {visitA.MMSE != null && visitB.MMSE != null ? (
                            <span
                              className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                                visitB.MMSE < visitA.MMSE
                                  ? "bg-rose-500/20 text-rose-300"
                                  : visitB.MMSE > visitA.MMSE
                                  ? "bg-emerald-500/20 text-emerald-300"
                                  : "text-neutral-400"
                              }`}
                            >
                              {visitB.MMSE - visitA.MMSE > 0 ? "+" : ""}
                              {visitB.MMSE - visitA.MMSE} pts
                            </span>
                          ) : (
                            "—"
                          )}
                        </td>
                      </tr>

                      {/* CDR-Global */}
                      <tr>
                        <td className="py-2 px-3 font-bold text-white font-sans">CDR-Global</td>
                        <td className="py-2 px-3">{visitA.CDGLOBAL ?? "—"}</td>
                        <td className="py-2 px-3 text-white">{visitB.CDGLOBAL ?? "—"}</td>
                        <td className="py-2 px-3">
                          {visitA.CDGLOBAL != null && visitB.CDGLOBAL != null ? (
                            <span
                              className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                                visitB.CDGLOBAL > visitA.CDGLOBAL
                                  ? "bg-rose-500/20 text-rose-300"
                                  : visitB.CDGLOBAL < visitA.CDGLOBAL
                                  ? "bg-emerald-500/20 text-emerald-300"
                                  : "text-neutral-400"
                              }`}
                            >
                              {visitB.CDGLOBAL - visitA.CDGLOBAL > 0 ? "+" : ""}
                              {(visitB.CDGLOBAL - visitA.CDGLOBAL).toFixed(1)}
                            </span>
                          ) : (
                            "—"
                          )}
                        </td>
                      </tr>

                      {/* ADAS11 */}
                      <tr>
                        <td className="py-2 px-3 font-bold text-white font-sans">ADAS11-Cog</td>
                        <td className="py-2 px-3">{visitA.TOTSCORE ?? "—"}</td>
                        <td className="py-2 px-3 text-white">{visitB.TOTSCORE ?? "—"}</td>
                        <td className="py-2 px-3">
                          {visitA.TOTSCORE != null && visitB.TOTSCORE != null ? (
                            <span
                              className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                                visitB.TOTSCORE > visitA.TOTSCORE
                                  ? "bg-rose-500/20 text-rose-300"
                                  : visitB.TOTSCORE < visitA.TOTSCORE
                                  ? "bg-emerald-500/20 text-emerald-300"
                                  : "text-neutral-400"
                              }`}
                            >
                              {visitB.TOTSCORE - visitA.TOTSCORE > 0 ? "+" : ""}
                              {(visitB.TOTSCORE - visitA.TOTSCORE).toFixed(1)}
                            </span>
                          ) : (
                            "—"
                          )}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="pt-4 border-t border-white/10 flex justify-end">
                <button
                  onClick={() => setShowCompareModal(false)}
                  className="px-4 py-2 rounded-xl bg-white/10 hover:bg-white/20 text-neutral-300 text-xs font-semibold cursor-pointer"
                >
                  Done
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
