"use client";

import React from "react";
import { HistoryEvent, TwinFull } from "@/app/lib/api";
import { History, ArrowUpRight, ArrowDownRight, Minus, Clock, CheckCircle } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface TimelinePanelProps {
  twin: TwinFull;
}

export default function TimelinePanel({ twin }: TimelinePanelProps) {
  const history = [...(twin.history || [])].reverse(); // newest first

  const formatEventName = (raw: string) => {
    if (raw === "twin_created") return "Digital Twin Initialized";
    if (raw.startsWith("visit_added")) return "Follow-up Visit Recorded";
    if (raw === "twin_reset" || raw === "reset") return "Reset to Baseline";
    if (raw.startsWith("modality_attached:")) {
      const mod = raw.replace("modality_attached:", "").replace("_", " ");
      return `Modality Attached: ${mod.toUpperCase()}`;
    }
    return raw;
  };

  const formatRecordedAt = (item: HistoryEvent) => {
    const raw = item.recorded_at || item.timestamp;
    try {
      const d = new Date(raw);
      return d.toLocaleDateString([], {
        month: "short",
        day: "numeric",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return raw;
    }
  };

  return (
    <div className="rounded-2xl bg-neutral-900/60 backdrop-blur-xl border border-white/10 p-6 shadow-2xl">
      <div className="flex items-center justify-between mb-5">
        <div>
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <History className="w-4 h-4 text-blue-400" />
            Twin Evolution Timeline
          </h3>
          <p className="text-xs text-neutral-400 mt-0.5">
            Audit trail of clinical inputs, exam dates, and probability shifts.
          </p>
        </div>
        <span className="text-xs px-2.5 py-1 rounded-full bg-white/5 border border-white/10 text-neutral-400 font-mono">
          {history.length} Event{history.length === 1 ? "" : "s"}
        </span>
      </div>

      <div className="relative pl-6 space-y-5 before:absolute before:left-2.5 before:top-2 before:bottom-2 before:w-0.5 before:bg-gradient-to-b before:from-blue-500 before:via-purple-500/50 before:to-transparent">
        <AnimatePresence initial={false}>
          {history.map((item, idx) => {
            const hasDiff = item.prob_diff && Object.keys(item.prob_diff).length > 0;

            return (
              <motion.div
                key={`timeline-${item.event}-${item.timestamp || item.recorded_at}-${idx}`}
                initial={{ opacity: 0, x: -15 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: idx * 0.05 }}
                className="relative group"
              >
                {/* Timeline node */}
                <span className="absolute -left-[27px] top-1.5 w-3 h-3 rounded-full bg-blue-500 border-2 border-neutral-900 group-hover:scale-125 transition" />

                <div className="p-3.5 rounded-xl bg-white/[0.03] border border-white/5 group-hover:border-white/15 transition-all">
                  <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
                    <span className="text-xs font-bold text-white">
                      {formatEventName(item.event)}
                    </span>
                    <div className="flex items-center gap-2 text-[11px] text-neutral-400 font-mono">
                      {item.exam_date && (
                        <span className="px-2 py-0.5 rounded bg-white/5 border border-white/10 text-blue-300">
                          Exam: {item.exam_date}
                        </span>
                      )}
                      <span className="flex items-center gap-1 text-neutral-500">
                        <Clock className="w-3 h-3" />
                        {formatRecordedAt(item)}
                      </span>
                    </div>
                  </div>

                  {/* Stage snapshot */}
                  <div className="flex items-center gap-2 text-xs mb-2">
                    <span className="text-neutral-400">Post-event Stage:</span>
                    <span className="font-mono text-neutral-200">
                      CN: {(item.stage_probs.CN * 100).toFixed(0)}% · MCI:{" "}
                      {(item.stage_probs.MCI * 100).toFixed(0)}% · AD:{" "}
                      {(item.stage_probs.AD * 100).toFixed(0)}%
                    </span>
                  </div>

                  {/* Animated Probability Diff (Before / After delta) */}
                  {hasDiff && (
                    <div className="mt-2 pt-2 border-t border-white/5 flex flex-wrap items-center gap-2">
                      <span className="text-[10px] uppercase font-semibold text-neutral-400">
                        Shift:
                      </span>
                      {(["CN", "MCI", "AD"] as const).map((cls) => {
                        const diff = Number((item.prob_diff as any)?.[cls] ?? 0);
                        if (Math.abs(diff) < 0.001) return null;
                        const isPos = diff > 0;
                        const isGoodForPatient = cls === "CN" ? isPos : !isPos;

                        return (
                          <span
                            key={`timeline-${idx}-diff-${cls}`}
                            className={`inline-flex items-center gap-0.5 px-2 py-0.5 rounded text-[11px] font-mono font-semibold border ${
                              isGoodForPatient
                                ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                                : "bg-rose-500/10 text-rose-400 border-rose-500/20"
                            }`}
                          >
                            {isPos ? (
                              <ArrowUpRight className="w-3 h-3" />
                            ) : (
                              <ArrowDownRight className="w-3 h-3" />
                            )}
                            {cls} {isPos ? `+${(diff * 100).toFixed(1)}%` : `${(diff * 100).toFixed(1)}%`}
                          </span>
                        );
                      })}
                    </div>
                  )}
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </div>
  );
}
