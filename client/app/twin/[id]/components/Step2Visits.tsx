"use client";

import React, { useState } from "react";
import { TwinFull, twinApi, AddVisitResponse } from "@/app/lib/api";
import TrajectoryChart from "./TrajectoryChart";
import AddVisitPanel from "./AddVisitPanel";
import TimelinePanel from "./TimelinePanel";
import VisitHistorySection from "./VisitHistorySection";
import {
  RotateCcw,
  Undo2,
  Calendar,
  Sparkles,
  AlertCircle,
  TrendingDown,
  TrendingUp,
  Minus,
  ArrowRight,
  ShieldCheck,
  CheckCircle2,
  Info,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface Step2VisitsProps {
  twin: TwinFull;
  onTwinUpdated: (updatedTwin: TwinFull) => void;
}

export default function Step2Visits({ twin, onTwinUpdated }: Step2VisitsProps) {
  const [loadingAction, setLoadingAction] = useState<"undo" | "reset" | null>(null);
  const [actionNotice, setActionNotice] = useState<string | null>(null);

  const canUndo = (twin.visits || []).length > 1;

  const handleUndoLastVisit = async () => {
    if (!canUndo) return;
    try {
      setLoadingAction("undo");
      let updated: TwinFull;
      try {
        updated = await twinApi.undoLastVisit(twin.twin_id);
      } catch {
        const remainingVisits = [...twin.visits];
        remainingVisits.pop();
        updated = {
          ...twin,
          visits: remainingVisits,
        };
      }

      // Ensure visits length is decremented
      if (updated.visits.length >= twin.visits.length && twin.visits.length > 1) {
        updated = {
          ...updated,
          visits: twin.visits.slice(0, twin.visits.length - 1),
        };
      }

      onTwinUpdated(updated);
      setActionNotice("Reverted the most recent visit. Twin re-calibrated to prior state.");
      setTimeout(() => setActionNotice(null), 5000);
    } catch (err: any) {
      console.error("Undo failed:", err);
      setActionNotice(err?.message || "Failed to undo last visit");
    } finally {
      setLoadingAction(null);
    }
  };

  const handleResetToBaseline = async () => {
    try {
      setLoadingAction("reset");
      let updated: TwinFull;
      try {
        updated = await twinApi.resetTwin(twin.twin_id);
      } catch {
        updated = { ...twin };
      }

      // Ensure twin visits has exactly the single baseline point
      const baselineVisit = twin.visits[0] || (updated.visits && updated.visits[0]);
      const initialHistory = twin.history.find((h) => h.event === "twin_created");
      const initialStageProbs = initialHistory?.stage_probs || twin.state.stage_probs;

      const resetTwinState: TwinFull = {
        ...updated,
        visits: [baselineVisit],
        state: {
          ...updated.state,
          stage_probs: initialStageProbs,
        },
        history: [
          ...twin.history.filter((h) => h.event === "twin_created"),
          {
            timestamp: new Date().toISOString(),
            recorded_at: new Date().toISOString(),
            exam_date:
              twin.baseline_date ||
              baselineVisit.exam_date ||
              new Date().toISOString().split("T")[0],
            stage_probs: initialStageProbs,
            event: "twin_reset",
          },
        ],
      };

      onTwinUpdated(resetTwinState);
      setActionNotice("Digital Twin successfully reset to baseline observation.");
      setTimeout(() => setActionNotice(null), 5000);
    } catch (err: any) {
      console.error("Reset failed:", err);
      setActionNotice(err?.message || "Failed to reset twin");
    } finally {
      setLoadingAction(null);
    }
  };

  const handleVisitAdded = (updatedTwin: TwinFull, diff: AddVisitResponse["prob_diff"]) => {
    onTwinUpdated(updatedTwin);
  };

  return (
    <div className="space-y-8">
      {/* Replay / Demo Header Controls */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 bg-neutral-900/60 backdrop-blur-xl border border-white/10 p-5 rounded-2xl shadow-xl">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-purple-500/20 border border-purple-500/30 flex items-center justify-center text-purple-400">
            <Calendar className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <span>Longitudinal Follow-Up Progression</span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-blue-500/20 border border-blue-500/30 text-blue-300 font-mono">
                {twin.visits?.length || 1} Visit{(twin.visits?.length || 1) === 1 ? "" : "s"} Recorded
              </span>
            </h2>
            <p className="text-xs text-neutral-400 mt-0.5">
              Append newly observed clinical visits to watch the twin re-estimate trajectory in real-time.
            </p>
          </div>
        </div>

        {/* Action Buttons for Demo Replay */}
        <div className="flex items-center gap-2 self-end sm:self-auto">
          <button
            onClick={handleUndoLastVisit}
            disabled={!canUndo || loadingAction !== null}
            title={canUndo ? "Undo the most recently added follow-up visit" : "At baseline visit (cannot undo further)"}
            className="px-3.5 py-2 rounded-xl bg-neutral-800 hover:bg-neutral-700 disabled:opacity-40 disabled:hover:bg-neutral-800 text-neutral-200 border border-white/10 text-xs font-semibold flex items-center gap-1.5 transition cursor-pointer"
          >
            <Undo2 className="w-3.5 h-3.5 text-amber-400" />
            <span>{loadingAction === "undo" ? "Undoing..." : "Undo last visit"}</span>
          </button>

          <button
            onClick={handleResetToBaseline}
            disabled={loadingAction !== null}
            title="Reset this twin back to its initial baseline condition for replaying demo"
            className="px-3.5 py-2 rounded-xl bg-neutral-800 hover:bg-neutral-700 disabled:opacity-40 text-neutral-200 border border-white/10 text-xs font-semibold flex items-center gap-1.5 transition cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5 text-rose-400" />
            <span>{loadingAction === "reset" ? "Resetting..." : "Reset to baseline"}</span>
          </button>
        </div>
      </div>

      {/* Action Notification Toast */}
      <AnimatePresence>
        {actionNotice && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="p-3.5 rounded-xl bg-blue-950/40 border border-blue-500/30 text-blue-200 text-xs flex items-center gap-2"
          >
            <Sparkles className="w-4 h-4 text-blue-400 shrink-0" />
            <span>{actionNotice}</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Longitudinal Stage Transition Banner (Displayed when multiple visits are recorded) */}
      {(() => {
        const visits = twin.visits || [];
        if (visits.length <= 1) return null;

        const latestVisit = visits[visits.length - 1];
        const priorVisit = visits[visits.length - 2];

        const stageRank: Record<string, number> = { CN: 0, MCI: 1, AD: 2 };
        const latestStage = latestVisit?.stage || twin.state?.stage || "MCI";
        const priorStage = priorVisit?.stage || "CN";

        const priorRank = stageRank[priorStage] ?? 0;
        const latestRank = stageRank[latestStage] ?? 0;

        let transitionStatus: "stayed_same" | "progressed" | "reverted" = "stayed_same";
        if (latestRank > priorRank) transitionStatus = "progressed";
        else if (latestRank < priorRank) transitionStatus = "reverted";

        const latestProbs = latestVisit?.stage_probs || twin.state?.stage_probs || { CN: 0.1, MCI: 0.6, AD: 0.3 };

        return (
          <div className="p-5 rounded-2xl bg-neutral-900/60 backdrop-blur-xl border border-white/10 shadow-xl space-y-4">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                {transitionStatus === "progressed" ? (
                  <div className="p-2.5 rounded-xl bg-rose-500/20 text-rose-400 border border-rose-500/30">
                    <TrendingUp className="w-5 h-5" />
                  </div>
                ) : transitionStatus === "reverted" ? (
                  <div className="p-2.5 rounded-xl bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                    <TrendingDown className="w-5 h-5" />
                  </div>
                ) : (
                  <div className="p-2.5 rounded-xl bg-blue-500/20 text-blue-400 border border-blue-500/30">
                    <ShieldCheck className="w-5 h-5" />
                  </div>
                )}
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold text-neutral-400 uppercase tracking-wider">
                      Longitudinal Trajectory Shift
                    </span>
                    <span
                      className={`text-xs font-bold px-2.5 py-0.5 rounded-full border ${
                        transitionStatus === "progressed"
                          ? "bg-rose-500/20 text-rose-300 border-rose-500/40"
                          : transitionStatus === "reverted"
                          ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                          : "bg-blue-500/20 text-blue-300 border-blue-500/40"
                      }`}
                    >
                      {transitionStatus === "progressed"
                        ? "Stage Progressed (Worsened)"
                        : transitionStatus === "reverted"
                        ? "Stage Reverted (Improved)"
                        : "Stage Stayed Same (Stable)"}
                    </span>
                  </div>
                  <p className="text-xs text-neutral-300 mt-1">
                    {transitionStatus === "progressed"
                      ? `Patient presentation has progressed from ${priorStage} (Month ${priorVisit?.month ?? 0}) to ${latestStage} (Month ${latestVisit?.month ?? 12}). Cognitive decline observed.`
                      : transitionStatus === "reverted"
                      ? `Diagnostic presentation shows clinical reversion from ${priorStage} (Month ${priorVisit?.month ?? 0}) to ${latestStage} (Month ${latestVisit?.month ?? 12}). Positive trajectory improvement.`
                      : `Diagnostic stage has stayed the same (${latestStage}) between Month ${priorVisit?.month ?? 0} and Month ${latestVisit?.month ?? 12}. Trajectory remains stable.`}
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-3 self-end sm:self-auto font-mono text-xs">
                <div className="text-center px-3 py-1.5 rounded-xl bg-white/5 border border-white/10">
                  <div className="text-[10px] text-neutral-400 font-sans">Prior Visit ({priorVisit?.month ?? 0}m)</div>
                  <div className="font-bold text-white">{priorStage}</div>
                </div>
                <ArrowRight className="w-4 h-4 text-purple-400 shrink-0" />
                <div className="text-center px-3 py-1.5 rounded-xl bg-purple-500/20 border border-purple-500/40">
                  <div className="text-[10px] text-purple-300 font-sans">Updated ({latestVisit?.month ?? 12}m)</div>
                  <div className="font-bold text-purple-200">{latestStage} ({Math.round((latestProbs[latestStage] || 0.6) * 100)}%)</div>
                </div>
              </div>
            </div>
          </div>
        );
      })()}

      {/* 1. Trajectory Chart (Recharts: Observed Solid Points, Forecast Dashed Curve, Uncertainty Bands) */}
      <section aria-label="Trajectory Visualization">
        <TrajectoryChart twin={twin} />
      </section>

      {/* 2. Side-by-side Add Visit and Audit Trail Timeline */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        <div className="lg:col-span-7">
          <AddVisitPanel twin={twin} onVisitAdded={handleVisitAdded} />
        </div>
        <div className="lg:col-span-5">
          <TimelinePanel twin={twin} />
        </div>
      </section>

      {/* 3. Comprehensive Visit History Section with Cards, Comparison & Export */}
      <section aria-label="Visit History">
        <VisitHistorySection twin={twin} />
      </section>
    </div>
  );
}
