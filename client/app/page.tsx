// app/page.tsx
"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import {
  Users,
  Activity,
  ArrowRight,
  Sparkles,
  Layers,
  Cpu,
} from "lucide-react";

export default function Home() {
  return (
    <div className="relative min-h-screen overflow-hidden bg-transparent text-white">
      {/* --- Background gradient grid --- */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(72,85,255,0.18),transparent_70%)]" />
        <div className="absolute top-1/3 -right-20 w-[500px] h-[500px] bg-purple-600/10 rounded-full blur-[140px]" />
        <div
          className="absolute inset-0 
          bg-[linear-gradient(rgba(255,255,255,0.06)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.06)_1px,transparent_1px)] 
          bg-[size:60px_60px] opacity-20"
        />
      </div>

      {/* --- Hero --- */}
      <section className="relative z-10 flex flex-col items-center justify-center text-center pt-28 px-6">
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-blue-500/10 border border-blue-500/30 text-blue-400 text-xs font-semibold mb-6 shadow-sm shadow-blue-500/20">
          <Sparkles className="w-3.5 h-3.5" />
          <span>Patient-Specific Alzheimer's Digital Twin Platform</span>
        </div>

        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-4xl sm:text-6xl font-black tracking-tight text-white mb-4 max-w-4xl leading-tight"
        >
          Alzheimer’s Digital Twin & Multimodal Staging
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.8 }}
          className="max-w-2xl text-neutral-300 text-base sm:text-lg leading-relaxed"
        >
          Build an individualized neurological digital twin combining clinical demographics,
          cognitive testing, fluid biomarkers, and structural MRI with 24-month progression forecasting.
        </motion.p>

        {/* CTA Buttons */}
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.4, duration: 0.6 }}
          className="mt-10 flex flex-wrap items-center justify-center gap-4"
        >
          <Link
            href="/intake"
            className="px-8 py-4 rounded-full bg-blue-600 hover:bg-blue-500 
            text-white font-bold text-base shadow-xl shadow-blue-600/40 
            transition-all duration-300 hover:scale-105 cursor-pointer flex items-center gap-2"
          >
            <span>Create a Digital Twin</span>
            <ArrowRight className="w-4 h-4" />
          </Link>

          <Link
            href="/demos"
            className="px-8 py-4 rounded-full bg-neutral-900 hover:bg-neutral-800 border border-white/20 
            text-neutral-200 font-semibold text-base shadow-lg hover:border-purple-500/40 
            transition-all duration-300 hover:scale-105 cursor-pointer flex items-center gap-2"
          >
            <Users className="w-4 h-4 text-purple-400" />
            <span>Explore Demo Patients</span>
          </Link>
        </motion.div>
      </section>

      {/* --- Feature Cards Section --- */}
      <section className="relative z-10 mt-28 px-6 pb-24 max-w-6xl mx-auto">
        <h2 className="text-sm font-bold text-neutral-400 uppercase tracking-wider mb-6 text-center sm:text-left">
          Exploration & Diagnostics
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Card 1: Demo Patients */}
          <Link href="/demos" className="block group">
            <motion.div
              whileInView={{ opacity: 1, y: 0 }}
              initial={{ opacity: 0, y: 30 }}
              transition={{ duration: 0.6 }}
              className="rounded-2xl p-6 bg-gradient-to-b from-purple-950/30 to-blue-950/20 backdrop-blur-lg border border-purple-500/30 hover:border-purple-400 hover:shadow-xl hover:shadow-purple-500/20 transition-all duration-300 cursor-pointer h-full"
            >
              <div className="flex items-center justify-between mb-4">
                <div className="w-10 h-10 rounded-xl bg-purple-500/20 text-purple-400 flex items-center justify-center">
                  <Users className="w-5 h-5" />
                </div>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-purple-500/20 border border-purple-500/30 text-purple-300 font-semibold uppercase">
                  6 Personas
                </span>
              </div>
              <h3 className="text-lg font-bold text-white mb-2 group-hover:text-purple-300 transition">
                Demo Patients
              </h3>
              <p className="text-xs text-neutral-300 leading-relaxed mb-4">
                Inspect 6 pre-configured ADNI cohort profiles (Stable CN, MCI Converters, Fast Decliners) or load them directly into the intake editor.
              </p>
              <span className="text-xs text-purple-400 font-bold flex items-center gap-1">
                Browse Personas <ArrowRight className="w-3.5 h-3.5" />
              </span>
            </motion.div>
          </Link>

          {/* Card 2: Temporal Prediction */}
          <Link href="/temporal-prediction" className="block group">
            <motion.div
              whileInView={{ opacity: 1, y: 0 }}
              initial={{ opacity: 0, y: 30 }}
              transition={{ delay: 0.1, duration: 0.6 }}
              className="rounded-2xl p-6 bg-white/[0.03] backdrop-blur-lg border border-white/10 hover:border-blue-400/40 hover:shadow-lg hover:shadow-blue-500/20 transition-all duration-300 cursor-pointer h-full"
            >
              <div className="flex items-center justify-between mb-4">
                <div className="w-10 h-10 rounded-xl bg-blue-500/20 text-blue-400 flex items-center justify-center">
                  <Activity className="w-5 h-5" />
                </div>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-blue-500/20 border border-blue-500/30 text-blue-300 font-medium">
                  LSTM Model
                </span>
              </div>
              <h3 className="text-lg font-bold text-white mb-2 group-hover:text-blue-300 transition">
                Temporal Prediction
              </h3>
              <p className="text-xs text-neutral-400 leading-relaxed mb-4">
                Track disease progression using LSTM time-series analysis on longitudinal clinical scores to forecast future states.
              </p>
              <span className="text-xs text-blue-400 font-bold flex items-center gap-1">
                Explore Tool <ArrowRight className="w-3.5 h-3.5" />
              </span>
            </motion.div>
          </Link>

          {/* Card 3: Multimodal Staging */}
          <Link href="/intake" className="block group">
            <motion.div
              whileInView={{ opacity: 1, y: 0 }}
              initial={{ opacity: 0, y: 30 }}
              transition={{ delay: 0.2, duration: 0.6 }}
              className="rounded-2xl p-6 bg-white/[0.03] backdrop-blur-lg border border-white/10 hover:border-blue-400/40 hover:shadow-lg hover:shadow-blue-500/20 transition-all duration-300 cursor-pointer h-full"
            >
              <div className="flex items-center justify-between mb-4">
                <div className="w-10 h-10 rounded-xl bg-cyan-500/20 text-cyan-400 flex items-center justify-center">
                  <Layers className="w-5 h-5" />
                </div>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-500/20 border border-cyan-500/30 text-cyan-300 font-medium">
                  T1–T4 Cascades
                </span>
              </div>
              <h3 className="text-lg font-bold text-white mb-2 group-hover:text-cyan-300 transition">
                Multimodal Staging
              </h3>
              <p className="text-xs text-neutral-400 leading-relaxed mb-4">
                Upload MRI scans, blood reports, and lifestyle indicators. Dynamic tiering refines uncertainty as more modalities are attached.
              </p>
              <span className="text-xs text-cyan-400 font-bold flex items-center gap-1">
                Start Intake <ArrowRight className="w-3.5 h-3.5" />
              </span>
            </motion.div>
          </Link>
        </div>
      </section>
    </div>
  );
}
