/* WorkflowVisualization.tsx — Step-by-step animated LSTM pipeline walkthrough */
"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import ArchitectureDiagram from "./ArchitectureDiagram";

const STAGES = [
  {
    id: "raw",
    title: "1 — Raw Longitudinal Data",
    icon: "📋",
    description:
      "The ADNI dataset contains 22,168 rows for 3,034 patients across multiple visits. Each row captures: subject_id, visit type (bl, sc, m06, m12 …), entry_age, CDGLOBAL (Clinical Dementia Rating), MMSCORE (Mini-Mental State), DIAGNOSIS, and TOTSCORE.",
    detail: (
      <div className="overflow-x-auto mt-4">
        <table className="text-xs w-full border-collapse">
          <thead>
            <tr className="text-neutral-400">
              {["subject_id","visit","age","CDGLOBAL","MMSCORE","DIAG","TOTSCORE"].map(h => (
                <th key={h} className="px-3 py-2 border border-white/10 text-left font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="text-neutral-300">
            {[
              ["941_S_10002","bl","72.95","NaN","NaN","2","9.00"],
              ["941_S_10002","m12","72.95","0.5","24","2","2.33"],
              ["941_S_10002","m24","72.95","0.5","24","2","5.00"],
              ["941_S_10002","sc","72.95","0.5","27","2","NaN"],
            ].map((r,i) => (
              <tr key={i} className={i%2===0 ? "bg-white/[0.02]" : ""}>
                {r.map((c,j) => (
                  <td key={j} className={`px-3 py-1.5 border border-white/10 ${c==="NaN" ? "text-red-400 italic" : ""}`}>{c}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    ),
  },
  {
    id: "preprocess",
    title: "2 — Preprocessing & Visit Merging",
    icon: "⚙️",
    description:
      "Baseline (bl) and screening (sc) visits are merged into a single 'bl_sc' row per patient — filling NaN from one into the other. Non-standard visits (scmri, m03, m30, etc.) are filtered out, keeping only 10 clean visit types. Visits are converted to month numbers (bl_sc → 0, m06 → 6, m12 → 12, …).",
    detail: (
      <div className="mt-4 space-y-3">
        <div className="flex flex-wrap gap-2">
          {["bl_sc → 0","m06 → 6","m12 → 12","m24 → 24","m36 → 36","m48 → 48","m60 → 60","m72 → 72","m78 → 78","m84 → 84"].map(v => (
            <span key={v} className="px-3 py-1 text-xs rounded-full bg-blue-500/15 border border-blue-500/30 text-blue-300">{v}</span>
          ))}
        </div>
        <p className="text-xs text-neutral-500">Diagnosis labels converted: 1,2,3 → 0 (CN), 1 (MCI), 2 (AD)</p>
      </div>
    ),
  },
  {
    id: "windows",
    title: "3 — Sliding Window Construction",
    icon: "🔲",
    description:
      "For each patient, a sliding window of size 3 (stride 1) groups consecutive visits. The last 3 visits become the input; the next visit's diagnosis is the target. Shorter sequences are pre-padded with zeros.",
    detail: (
      <div className="mt-4 space-y-2">
        <div className="overflow-x-auto">
          <table className="text-xs w-full border-collapse">
            <thead>
              <tr className="text-neutral-400">
                <th className="px-3 py-2 border border-white/10 text-left">Seq</th>
                <th className="px-3 py-2 border border-white/10 text-left">Input Visits (months)</th>
                <th className="px-3 py-2 border border-white/10 text-left">Target Visit</th>
                <th className="px-3 py-2 border border-white/10 text-left">Target</th>
              </tr>
            </thead>
            <tbody className="text-neutral-300">
              {[
                ["0","[0, 6]  (padded)","12","CN (0)"],
                ["1","[0, 6, 12]","24","CN (0)"],
                ["2","[6, 12, 24]","36","CN (0)"],
                ["3","[12, 24, 36]","48","CN (0)"],
              ].map((r,i) => (
                <tr key={i} className={i%2===0 ? "bg-white/[0.02]" : ""}>
                  {r.map((c,j) => (
                    <td key={j} className="px-3 py-1.5 border border-white/10">{c}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-neutral-500">Each input: [entry_age, CDGLOBAL, MMSCORE, TOTSCORE, visit_month] × 3 timesteps</p>
      </div>
    ),
  },
  {
    id: "scale",
    title: "4 — Feature Scaling",
    icon: "📐",
    description:
      "NaN values are replaced with 0.0, then all features are normalized using MinMaxScaler (fitted on training data). This ensures the LSTM receives values between 0 and 1, improving convergence and stability.",
    detail: (
      <div className="mt-4 grid grid-cols-2 sm:grid-cols-5 gap-2">
        {["entry_age","CDGLOBAL","MMSCORE","TOTSCORE","visit_month"].map(f => (
          <div key={f} className="text-center px-2 py-3 rounded-lg bg-white/[0.04] border border-white/10">
            <div className="text-xs text-neutral-400 mb-1">{f}</div>
            <div className="text-sm text-green-400 font-mono">0 → 1</div>
          </div>
        ))}
      </div>
    ),
  },
  {
    id: "model",
    title: "5 — Model Architecture",
    icon: "🧠",
    description:
      "A Bidirectional LSTM processes the sequence in both directions (64 units each direction), followed by BatchNormalization, Dense(32, ReLU) with L2 regularization, Dropout(0.3), and a Softmax output layer for 3-class classification.",
    detail: <ArchitectureDiagram />,
  },
  {
    id: "result",
    title: "6 — Prediction Output",
    icon: "🎯",
    description:
      "The model outputs a probability distribution over 3 classes. The highest probability determines the predicted diagnosis. Trained on ~5,560 sequences with class-balanced weights, achieving ~82% accuracy.",
    detail: (
      <div className="mt-4 space-y-3">
        <div className="grid grid-cols-3 gap-3">
          {[
            { label: "CN", prob: "0.87", color: "bg-emerald-500/20 border-emerald-500/40 text-emerald-300", p: "87", f1: "0.89" },
            { label: "MCI", prob: "0.08", color: "bg-amber-500/20 border-amber-500/40 text-amber-300", p: "82", f1: "0.78" },
            { label: "AD", prob: "0.05", color: "bg-red-500/20 border-red-500/40 text-red-300", p: "77", f1: "0.81" },
          ].map(c => (
            <div key={c.label} className={`rounded-lg p-3 border text-center ${c.color}`}>
              <div className="text-lg font-bold">{c.label}</div>
              <div className="text-xs opacity-70 mt-1">Precision: {c.p}%</div>
              <div className="text-xs opacity-70">F1: {c.f1}</div>
            </div>
          ))}
        </div>
        <p className="text-xs text-neutral-500 text-center">Overall test accuracy: ~83% across 1,340 test sequences</p>
      </div>
    ),
  },
];

const cardVariant = {
  hidden: { opacity: 0, y: 30, scale: 0.97 },
  visible: { opacity: 1, y: 0, scale: 1, transition: { duration: 0.45, ease: "easeOut" as const } },
  exit: { opacity: 0, y: -20, scale: 0.97, transition: { duration: 0.25 } },
};

export default function WorkflowVisualization() {
  const [activeStage, setActiveStage] = useState(0);

  return (
    <div className="space-y-6">
      {/* Stage selector pills */}
      <div className="flex flex-wrap gap-2 justify-center">
        {STAGES.map((s, i) => (
          <button
            key={s.id}
            onClick={() => setActiveStage(i)}
            className={`px-4 py-2 rounded-full text-sm font-medium transition-all duration-300 border ${
              activeStage === i
                ? "bg-blue-600/30 border-blue-400/60 text-white shadow-lg shadow-blue-500/20"
                : "bg-white/5 border-white/10 text-neutral-400 hover:border-white/25 hover:text-neutral-200"
            }`}
          >
            <span className="mr-1.5">{s.icon}</span>
            Step {i + 1}
          </button>
        ))}
      </div>

      {/* Active stage card */}
      <AnimatePresence mode="wait">
        <motion.div
          key={STAGES[activeStage].id}
          variants={cardVariant}
          initial="hidden"
          animate="visible"
          exit="exit"
          className="rounded-2xl p-6 md:p-8 bg-white/[0.04] backdrop-blur-xl border border-white/10"
        >
          <h3 className="text-xl font-semibold text-white mb-3">
            {STAGES[activeStage].icon} {STAGES[activeStage].title}
          </h3>
          <p className="text-neutral-300 text-sm leading-relaxed">{STAGES[activeStage].description}</p>
          {STAGES[activeStage].detail}
        </motion.div>
      </AnimatePresence>

      {/* Nav buttons */}
      <div className="flex justify-between">
        <button
          onClick={() => setActiveStage(p => Math.max(0, p - 1))}
          disabled={activeStage === 0}
          className="px-5 py-2 rounded-xl text-sm bg-white/5 border border-white/10 text-neutral-400 hover:text-white hover:border-white/25 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
        >
          ← Previous
        </button>
        <button
          onClick={() => setActiveStage(p => Math.min(STAGES.length - 1, p + 1))}
          disabled={activeStage === STAGES.length - 1}
          className="px-5 py-2 rounded-xl text-sm bg-blue-600/30 border border-blue-400/50 text-blue-200 hover:bg-blue-600/40 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
        >
          Next Step →
        </button>
      </div>
    </div>
  );
}
