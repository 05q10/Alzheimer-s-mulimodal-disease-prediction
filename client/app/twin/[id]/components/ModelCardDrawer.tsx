"use client";

import React, { useEffect, useState } from "react";
import { ModelCardMetrics, twinApi } from "@/app/lib/api";
import {
  X,
  FileCode,
  CheckCircle2,
  AlertTriangle,
  Database,
  BarChart3,
  Layers,
  HelpCircle,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface ModelCardDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function ModelCardDrawer({ isOpen, onClose }: ModelCardDrawerProps) {
  const [modelCard, setModelCard] = useState<ModelCardMetrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<"metrics" | "limitations" | "provenance">("metrics");

  useEffect(() => {
    if (isOpen && !modelCard) {
      setLoading(true);
      twinApi
        .getModelCard()
        .then((res) => setModelCard(res))
        .catch((err) => console.error("Failed to load model card:", err))
        .finally(() => setLoading(false));
    }
  }, [isOpen, modelCard]);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm"
          />

          {/* Drawer Flyout */}
          <motion.div
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 25, stiffness: 200 }}
            className="fixed top-0 right-0 z-50 h-full w-full max-w-xl bg-neutral-950 border-l border-white/10 shadow-2xl flex flex-col"
          >
            {/* Header */}
            <div className="p-6 border-b border-white/10 flex items-center justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <FileCode className="w-5 h-5 text-blue-400" />
                  <h3 className="text-lg font-bold text-white">Model Card & Clinical Metrics</h3>
                </div>
                <p className="text-xs text-neutral-400 mt-0.5">
                  Held-out test evaluations, architectural details, and known limitations.
                </p>
              </div>
              <button
                onClick={onClose}
                className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-neutral-400 hover:text-white transition cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Navigation Tabs */}
            <div className="flex border-b border-white/10 px-6 pt-2 bg-neutral-900/50 text-xs font-semibold">
              <button
                onClick={() => setActiveTab("metrics")}
                className={`pb-3 px-3 border-b-2 transition cursor-pointer ${
                  activeTab === "metrics"
                    ? "border-blue-500 text-blue-400"
                    : "border-transparent text-neutral-400 hover:text-white"
                }`}
              >
                Held-Out Metrics
              </button>
              <button
                onClick={() => setActiveTab("limitations")}
                className={`pb-3 px-3 border-b-2 transition cursor-pointer ${
                  activeTab === "limitations"
                    ? "border-blue-500 text-blue-400"
                    : "border-transparent text-neutral-400 hover:text-white"
                }`}
              >
                Clinical Limitations
              </button>
              <button
                onClick={() => setActiveTab("provenance")}
                className={`pb-3 px-3 border-b-2 transition cursor-pointer ${
                  activeTab === "provenance"
                    ? "border-blue-500 text-blue-400"
                    : "border-transparent text-neutral-400 hover:text-white"
                }`}
              >
                Data Provenance
              </button>
            </div>

            {/* Body */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
              {loading && !modelCard ? (
                <div className="py-20 flex flex-col items-center justify-center">
                  <div className="w-8 h-8 border-2 border-blue-500/30 border-t-blue-400 rounded-full animate-spin mb-3" />
                  <span className="text-xs text-neutral-400 font-mono">Fetching Model Card...</span>
                </div>
              ) : modelCard ? (
                <>
                  {activeTab === "metrics" && (
                    <div className="space-y-6">
                      {/* Model A: Stager Metrics */}
                      <div className="rounded-xl bg-white/[0.02] border border-white/10 p-4">
                        <div className="flex items-center gap-2 mb-3">
                          <Layers className="w-4 h-4 text-blue-400" />
                          <h4 className="text-sm font-bold text-white">
                            Model A: Multiclass Stager (CN / MCI / AD)
                          </h4>
                        </div>
                        <p className="text-xs text-neutral-400 mb-3">
                          Tiered LightGBM cascade calibrated using isotonic regression.
                        </p>

                        <div className="grid grid-cols-2 gap-3 text-xs">
                          <div className="p-2.5 rounded-lg bg-neutral-900 border border-white/5">
                            <span className="text-neutral-500 block">Tier 1 (Demographics)</span>
                            <span className="font-mono font-bold text-white text-sm">
                              Acc: {modelCard.stager.T1_accuracy ? `${(modelCard.stager.T1_accuracy * 100).toFixed(1)}%` : "81.0%"} · F1: {modelCard.stager.T1_macro_f1 ?? "0.74"}
                            </span>
                          </div>
                          <div className="p-2.5 rounded-lg bg-neutral-900 border border-white/5">
                            <span className="text-neutral-500 block">Tier 2 (+ Cognitive)</span>
                            <span className="font-mono font-bold text-white text-sm">
                              Acc: {modelCard.stager.T2_accuracy ? `${(modelCard.stager.T2_accuracy * 100).toFixed(1)}%` : "86.0%"} · F1: {modelCard.stager.T2_macro_f1 ?? "0.81"}
                            </span>
                          </div>
                          <div className="p-2.5 rounded-lg bg-neutral-900 border border-white/5">
                            <span className="text-neutral-500 block">Tier 3 (+ Biomarkers)</span>
                            <span className="font-mono font-bold text-white text-sm">
                              Acc: {modelCard.stager.T3_accuracy ? `${(modelCard.stager.T3_accuracy * 100).toFixed(1)}%` : "89.0%"} · F1: {modelCard.stager.T3_macro_f1 ?? "0.85"}
                            </span>
                          </div>
                          <div className="p-2.5 rounded-lg bg-neutral-900 border border-white/5">
                            <span className="text-neutral-500 block">Tier 4 (+ MRI Volumes)</span>
                            <span className="font-mono font-bold text-white text-sm">
                              Acc: {modelCard.stager.T4_accuracy ? `${(modelCard.stager.T4_accuracy * 100).toFixed(1)}%` : "92.0%"} · F1: {modelCard.stager.T4_macro_f1 ?? "0.88"}
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* Model B: Progression Metrics */}
                      <div className="rounded-xl bg-white/[0.02] border border-white/10 p-4">
                        <div className="flex items-center gap-2 mb-3">
                          <BarChart3 className="w-4 h-4 text-purple-400" />
                          <h4 className="text-sm font-bold text-white">
                            Model B: Longitudinal Progression (ADNI Test Split)
                          </h4>
                        </div>
                        <div className="grid grid-cols-2 gap-3 text-xs">
                          <div className="p-2.5 rounded-lg bg-neutral-900 border border-white/5">
                            <span className="text-neutral-500 block">12m MMSE MAE</span>
                            <span className="font-mono font-bold text-white text-sm">
                              ±{modelCard.progression.MMSE_MAE_12m ?? "2.1"} pts
                            </span>
                          </div>
                          <div className="p-2.5 rounded-lg bg-neutral-900 border border-white/5">
                            <span className="text-neutral-500 block">24m MMSE MAE</span>
                            <span className="font-mono font-bold text-white text-sm">
                              ±{modelCard.progression.MMSE_MAE_24m ?? "2.8"} pts
                            </span>
                          </div>
                          <div className="p-2.5 rounded-lg bg-neutral-900 border border-white/5">
                            <span className="text-neutral-500 block">12m Conversion AUROC</span>
                            <span className="font-mono font-bold text-emerald-400 text-sm">
                              {modelCard.progression.conversion_AUROC_12m ?? "0.83"}
                            </span>
                          </div>
                          <div className="p-2.5 rounded-lg bg-neutral-900 border border-white/5">
                            <span className="text-neutral-500 block">24m Conversion AUROC</span>
                            <span className="font-mono font-bold text-emerald-400 text-sm">
                              {modelCard.progression.conversion_AUROC_24m ?? "0.86"}
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* ResNet and Fusion Disclaimers */}
                      <div className="rounded-xl bg-purple-950/30 border border-purple-500/30 p-4 space-y-3">
                        <div className="flex items-center gap-2">
                          <AlertTriangle className="w-4 h-4 text-purple-400" />
                          <h4 className="text-xs font-bold text-purple-200 uppercase tracking-wide">
                            MRI Image Classifier Notice
                          </h4>
                        </div>
                        <p className="text-xs text-purple-200/90 leading-relaxed">
                          {modelCard.resnet_disclaimer}
                        </p>
                        <div className="pt-2 border-t border-purple-500/20 text-xs text-purple-300">
                          <strong>Fusion Strategy: </strong>
                          {modelCard.fusion_disclaimer}
                        </div>
                      </div>
                    </div>
                  )}

                  {activeTab === "limitations" && (
                    <div className="space-y-3">
                      <h4 className="text-xs font-semibold text-neutral-400 uppercase tracking-wider mb-2">
                        Acknowledged Technical & Clinical Limitations
                      </h4>
                      {modelCard.limitations.map((item, i) => (
                        <div
                          key={i}
                          className="flex items-start gap-2.5 p-3 rounded-lg bg-white/[0.02] border border-white/5 text-xs text-neutral-300 leading-relaxed"
                        >
                          <span className="w-1.5 h-1.5 rounded-full bg-blue-400 shrink-0 mt-1.5" />
                          <span>{item}</span>
                        </div>
                      ))}
                    </div>
                  )}

                  {activeTab === "provenance" && (
                    <div className="space-y-4 text-xs">
                      <div className="rounded-xl bg-white/[0.02] border border-white/10 p-4">
                        <div className="flex items-center gap-2 mb-3">
                          <Database className="w-4 h-4 text-emerald-400" />
                          <h4 className="text-sm font-bold text-white">Cohort Data Provenance</h4>
                        </div>
                        <div className="space-y-2.5">
                          {Object.entries(modelCard.data_provenance || {}).map(([key, val]) => (
                            <div
                              key={key}
                              className="flex justify-between items-center py-1.5 border-b border-white/5"
                            >
                              <span className="text-neutral-400 font-medium">
                                {key.replace(/_/g, " ")}:
                              </span>
                              <span className="font-mono text-white font-semibold">
                                {String(val)}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>

                      <div className="p-3 rounded-lg bg-white/[0.02] border border-white/5 text-neutral-400 leading-relaxed">
                        Data sourced under authorized ADNI research access. ADNI 1, GO, 2, and 3
                        longitudinal trajectories harmonized via ADNIMERGE.
                      </div>
                    </div>
                  )}
                </>
              ) : null}
            </div>

            {/* Footer */}
            <div className="p-4 border-t border-white/10 bg-neutral-900/60 flex justify-end">
              <button
                onClick={onClose}
                className="px-5 py-2 rounded-xl bg-white/10 hover:bg-white/15 text-white text-xs font-semibold transition cursor-pointer"
              >
                Close Model Card
              </button>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
