/* temporal-prediction/page.tsx — Main page for LSTM Temporal Cognitive Forecasting */
"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import Link from "next/link";
import WorkflowVisualization from "./WorkflowVisualization";
import PredictionTool from "./PredictionTool";

type Tab = "workflow" | "predict";

export default function TemporalPredictionPage() {
  const [tab, setTab] = useState<Tab>("workflow");

  return (
    <div className="relative min-h-screen overflow-hidden bg-transparent text-white">
      {/* Background */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(72,85,255,0.12),transparent_70%)]" />
        <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.04)_1px,transparent_1px)] bg-[size:60px_60px] opacity-20" />
        {/* Subtle neural-network mesh */}
        <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-[radial-gradient(circle,rgba(139,92,246,0.1),transparent_60%)]" />
        <div className="absolute bottom-0 left-0 w-[400px] h-[400px] bg-[radial-gradient(circle,rgba(16,185,129,0.08),transparent_60%)]" />
      </div>

      {/* Navigation */}
      <nav className="relative z-20 flex items-center justify-between px-6 py-4 border-b border-white/5">
        <Link
          href="/"
          className="text-sm text-neutral-400 hover:text-white transition-colors flex items-center gap-2"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Back to Home
        </Link>
        <div className="text-xs text-neutral-600 font-mono tracking-wider">LSTM · BiLSTM-64</div>
      </nav>

      {/* Hero */}
      <section className="relative z-10 text-center pt-12 pb-8 px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-purple-500/10 border border-purple-500/30 text-purple-300 text-xs font-medium mb-5">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-purple-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-purple-500" />
            </span>
            LSTM Time-Series Analysis
          </div>
          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight mb-3">
            Temporal Cognitive{" "}
            <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">
              Forecasting
            </span>
          </h1>
          <p className="max-w-2xl mx-auto text-neutral-400 text-sm sm:text-base leading-relaxed">
            Leverage longitudinal patient data to predict Alzheimer&#39;s progression using a
            Bidirectional LSTM trained on the ADNI clinical dataset. Track cognitive trajectories
            across multiple clinical visits.
          </p>
        </motion.div>
      </section>

      {/* Tab Switcher */}
      <section className="relative z-10 flex justify-center px-6 pb-6">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2, duration: 0.4 }}
          className="inline-flex rounded-2xl bg-white/[0.04] border border-white/10 p-1"
        >
          <button
            onClick={() => setTab("workflow")}
            className={`px-6 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ${
              tab === "workflow"
                ? "bg-blue-600/30 text-white shadow-lg shadow-blue-600/20"
                : "text-neutral-400 hover:text-neutral-200"
            }`}
          >
            <span className="mr-1.5">🔬</span> Visualize Workflow
          </button>
          <button
            onClick={() => setTab("predict")}
            className={`px-6 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ${
              tab === "predict"
                ? "bg-blue-600/30 text-white shadow-lg shadow-blue-600/20"
                : "text-neutral-400 hover:text-neutral-200"
            }`}
          >
            <span className="mr-1.5">🎯</span> Predict Next Visit
          </button>
        </motion.div>
      </section>

      {/* Content */}
      <section className="relative z-10 max-w-5xl mx-auto px-6 pb-20">
        <motion.div
          key={tab}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          {tab === "workflow" ? <WorkflowVisualization /> : <PredictionTool />}
        </motion.div>
      </section>
    </div>
  );
}
