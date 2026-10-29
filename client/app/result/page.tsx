"use client";

import React, { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { usePatientStore } from "../store/patientStore";
import { useRouter } from "next/navigation";

export default function ResultPage() {
  const router = useRouter();
  const session = usePatientStore((s) => s.session);

  React.useEffect(() => {
    const twinId = (session as any)?.twinId || (session as any)?.twin?.twin_id;
    if (twinId) {
      router.push(`/twin/${twinId}`);
    }
  }, [session, router]);

  // Correct store references
  const baseResult = session?.baseModel ?? { prediction: "N/A", input_used: {} };
  const alleleResult = session?.alleleResult ?? {
  predictedClass: null,
  probability: null,
  riskCategory: null,
};

  const mriResult = session?.uploads?.mriResult ?? null;
  const ocrResult = session?.uploads?.ocrResult ?? null;
  const alleleInput = session?.alleleInput ?? {};
  const demographics = session?.demographics ?? {};
  const cognitive = session?.cognitive ?? {};

  const patientName = demographics?.name ?? "Patient";

  // Floating metric chips
  const floatingStats = useMemo(() => {
    const items = [
      { label: "MMSE", value: cognitive?.mmse },
      { label: "ADL", value: cognitive?.adl },
      { label: "Functional", value: cognitive?.functionalAssessment },
      { label: "Age", value: demographics?.age },
      { label: "Alcohol", value: demographics?.alcoholConsumption },
      { label: "Diet", value: demographics?.dietQuality },
      { label: "Sleep", value: demographics?.sleepQuality },
      { label: "ABETA", value: alleleInput?.ABETA },
      { label: "TAU", value: alleleInput?.TAU },
      { label: "APVOL", value: alleleInput?.APVOLUME },
      { label: "Genotype", value: alleleInput?.GENOTYPE },
    ];

    return items.map((it) => ({ ...it, value: it.value ?? "N/A" })).slice(0, 9);
  }, [demographics, cognitive, alleleInput]);

  const [expanded, setExpanded] = useState(false);

  /* ------------------------ JSON DOWNLOAD ------------------------ */
  const downloadJSON = () => {
    const full = {
      patient: {
        name: patientName,
        demographics,
      },
      cognitive,
      allele_input: alleleInput,
      results: {
        base: baseResult,
        allele: alleleResult,
        mri: mriResult,
        ocr: ocrResult,
      },
      sessionMeta: {
        id: session?.id ?? null,
        createdAt: session?.timestamp ?? null,
      },
    };

    const blob = new Blob([JSON.stringify(full, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${(patientName || "patient").replace(/\s+/g, "_")}_digital_twin.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  /* ----------------------- CHIP POSITIONS ----------------------- */
  const chipPositions = [
    { left: "6%", top: "8%" },
    { left: "72%", top: "6%" },
    { left: "86%", top: "36%" },
    { left: "70%", top: "70%" },
    { left: "42%", top: "82%" },
    { left: "8%", top: "72%" },
    { left: "18%", top: "42%" },
    { left: "50%", top: "2%" },
    { left: "92%", top: "8%" },
  ];

  return (
    <div className="min-h-screen bg-black text-white relative overflow-hidden">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-slate-950 via-blue-900/20 to-slate-950 -z-10" />

      <main className="relative z-10 max-w-7xl mx-auto px-6 pt-20 pb-28">
        {/* Title */}
        <div className="mb-8 text-center">
          <h1 className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
            Digital Twin — {patientName}
          </h1>
          <p className="text-sm text-cyan-200/70 mt-2">
            Interactive multimodal diagnostic view — hover the brain to explore values.
          </p>
        </div>

        {/* MAIN LAYOUT */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-start">

          {/* ---------------------- LEFT SIDE: BRAIN ---------------------- */}
          <div className="relative flex justify-center items-start">
            <div className="relative w-[520px] h-[520px] md:w-[560px] md:h-[560px]">

              {/* Glow */}
              <div
                className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full blur-[80px]"
                style={{
                  width: 520,
                  height: 520,
                  background: "rgba(6,182,212,0.06)",
                }}
              />

              {/* Brain */}
              <motion.div
                whileHover={{ rotateY: 8, rotateX: -6, scale: 1.03 }}
                className="relative w-[500px] h-[500px] mx-auto"
                style={{ perspective: 1200 }}
              >
                <motion.img
                  src="/brain.png"
                  alt="Digital Brain"
                  draggable={false}
                  className="w-full h-full object-contain rounded-2xl drop-shadow-[0_30px_80px_rgba(6,182,212,0.12)]"
                  initial={{ opacity: 0, scale: 0.96 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.45, ease: "easeOut" }}
                />

                {/* Additional glow */}
                <div
                  className="absolute inset-0 pointer-events-none rounded-2xl"
                  style={{
                    boxShadow:
                      "inset 0 0 60px rgba(6,182,212,0.06), 0 40px 120px rgba(6,182,212,0.06)",
                  }}
                />
              </motion.div>

              {/* Floating chips */}
              {floatingStats.map((s, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 8, scale: 0.9 }}
                  animate={{
                    opacity: [0, 1],
                    y: [8, 0, 5],
                    scale: [0.95, 1, 0.98],
                  }}
                  transition={{
                    duration: 2.4,
                    repeat: Infinity,
                    repeatType: "mirror",
                    delay: i * 0.15,
                    ease: "easeInOut",
                  }}
                  style={{
                    position: "absolute",
                    ...chipPositions[i % chipPositions.length],
                    transform: "translate(-50%, -50%)",
                    zIndex: 30,
                    maxWidth: 160,
                  }}
                >
                  <div className="px-3 py-1.5 rounded-xl bg-gradient-to-r from-cyan-600/20 to-blue-600/10 border border-cyan-400/15 backdrop-blur-md text-xs text-cyan-100">
                    <div className="font-semibold text-xs text-white/90">{s.label}</div>
                    <div className="text-[12px] text-cyan-200 mt-0.5">{String(s.value)}</div>
                  </div>
                </motion.div>
              ))}

              {/* Connector lines */}
              {floatingStats.map((_, i) => {
                const pos = chipPositions[i % chipPositions.length];
                const pxLeft = (parseFloat(String(pos.left)) / 100) * 520;
                const pxTop = (parseFloat(String(pos.top)) / 100) * 520;

                return (
                  <svg
                    key={`conn-${i}`}
                    className="absolute pointer-events-none"
                    style={{
                      left: 0,
                      top: 0,
                      width: 520,
                      height: 520,
                      transform: "translate(-50%,-50%)",
                    }}
                    viewBox={`0 0 520 520`}
                  >
                    <line
                      x1={260}
                      y1={260}
                      x2={Math.max(10, Math.min(510, pxLeft))}
                      y2={Math.max(10, Math.min(510, pxTop))}
                      stroke="rgba(6,182,212,0.08)"
                      strokeWidth={1}
                      strokeLinecap="round"
                    />
                    <circle
                      cx={Math.max(10, Math.min(510, pxLeft))}
                      cy={Math.max(10, Math.min(510, pxTop))}
                      r={1.6}
                      fill="rgba(6,182,212,0.12)"
                    />
                  </svg>
                );
              })}
            </div>
          </div>

          {/* ---------------------- RIGHT SIDE: PANELS ---------------------- */}
          <div className="space-y-6">
            <div className="bg-slate-900/80 backdrop-blur-md border border-cyan-500/20 rounded-3xl p-6 shadow-sm">

              {/* Primary Diagnosis */}
              <h3 className="text-lg font-semibold text-cyan-300 mb-3">Primary Diagnosis</h3>
              <div
                className={`p-4 rounded-xl text-lg font-semibold ${
                  String(baseResult?.prediction)
                    .toLowerCase()
                    .includes("positive")
                    ? "bg-rose-600/10 text-rose-300 border border-rose-600/20"
                    : "bg-emerald-600/8 text-emerald-300 border border-emerald-600/12"
                }`}
              >
                {baseResult?.prediction ?? "N/A"}
              </div>

              {/* Allele */}
              <div className="mt-6">
              <h4 className="text-sm text-cyan-200/80 mb-2">Allele Risk Assessment</h4>

              <div className="p-3 rounded-lg bg-black/30 border border-cyan-500/12">

                {/* Risk Category */}
                <div className="flex items-center justify-between">
                  <div className="text-sm text-cyan-200">Risk Category</div>
                  <div className="text-sm font-medium text-white">
                    {alleleResult?.riskCategory ?? "N/A"}
                  </div>
                </div>

                {/* Predicted Class */}
                <div className="mt-2 flex items-center justify-between text-xs text-cyan-300/70">
                  <div>Predicted Class</div>
                  <div className="font-semibold">
                    {alleleResult?.predictedClass ?? "N/A"}
                  </div>
                </div>

                {/* Probability */}
                <div className="mt-2 flex items-center justify-between text-xs text-cyan-300/70">
                  <div>Probability</div>
                  <div className="font-semibold">
                    {alleleResult?.probability ?? "N/A"}
                  </div>
                </div>

              </div>
            </div>


              {/* MRI */}
              <div className="mt-6">
                <h4 className="text-sm text-cyan-200/80 mb-2">MRI Result</h4>
                <div className="p-3 rounded-lg bg-black/30 border border-cyan-500/12 text-sm text-cyan-200 max-h-36 overflow-auto">
                  <pre className="whitespace-pre-wrap text-xs">
                    {mriResult ? JSON.stringify(mriResult, null, 2) : "N/A"}
                  </pre>
                </div>
              </div>

              {/* OCR */}
              <div className="mt-6">
                <h4 className="text-sm text-cyan-200/80 mb-2">Extracted Lab Values</h4>
                <div className="p-3 rounded-lg bg-black/30 border border-cyan-500/12 text-sm text-cyan-200 max-h-36 overflow-auto">
                  {ocrResult && Array.isArray(ocrResult?.test_results) ? (
                    <>
                      {ocrResult.test_results.slice(0, 6).map((t: any, idx: number) => (
                        <div key={idx} className="text-xs">
                          <span className="font-medium">{t.test_name}</span>: {t.result}{" "}
                          {t.unit}
                        </div>
                      ))}
                      {ocrResult.test_results.length > 6 && (
                        <div className="text-xs text-cyan-300/50">
                          +{ocrResult.test_results.length - 6} more
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="text-xs text-cyan-300/50">No lab report data</div>
                  )}
                </div>
              </div>

              {/* Actions */}
              <div className="mt-6 flex flex-col sm:flex-row gap-3">
                <button
                  onClick={() => router.push(`/twin/${session?.twinId || "demo-cn-mci"}`)}
                  className="w-full py-3 rounded-xl bg-gradient-to-r from-purple-600 to-blue-600 hover:brightness-110 transition font-bold shadow-lg shadow-purple-600/30 flex items-center justify-center gap-2 cursor-pointer"
                >
                  <span>Open Trajectory Dashboard →</span>
                </button>
              </div>

              <div className="mt-3 flex gap-3">
                <button
                  onClick={downloadJSON}
                  className="flex-1 py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-500 hover:brightness-110 transition font-semibold text-xs cursor-pointer"
                >
                  Download Report JSON
                </button>
                <button
                  onClick={() => router.push("/intake")}
                  className="flex-1 py-3 rounded-xl bg-slate-800 border border-cyan-500/12 hover:bg-slate-700 transition font-semibold text-xs cursor-pointer"
                >
                  New Patient
                </button>
              </div>
            </div>

            {/* Small meta card */}
            <div className="bg-slate-900/60 rounded-2xl p-4 border border-cyan-500/10 text-sm text-cyan-200">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs text-cyan-300/80">Patient</div>
                  <div className="font-medium text-white">{patientName}</div>
                </div>
                <div className="text-right">
                  <div className="text-xs text-cyan-300/80">Age</div>
                  <div className="font-medium text-white">{demographics?.age ?? "N/A"}</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* DETAIL SECTION */}
        <div className="mt-10">
          <div className="bg-slate-900/70 border border-cyan-500/10 rounded-2xl p-4">
            <div
              className="flex items-center justify-between cursor-pointer"
              onClick={() => setExpanded((v) => !v)}
            >
              <div className="text-sm font-semibold text-cyan-300">
                Detailed Patient Data
              </div>
              <div className="text-xs text-cyan-200">
                {expanded ? "Collapse" : "Expand"}
              </div>
            </div>

            {expanded && (
              <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-cyan-200">

                {/* Left */}
                <div className="space-y-3">
                  <div>
                    <div className="text-xs text-cyan-300/80">Demographics</div>
                    <pre className="text-xs mt-1 bg-black/30 p-3 rounded-md overflow-auto">
                      {JSON.stringify(demographics, null, 2)}
                    </pre>
                  </div>

                  <div>
                    <div className="text-xs text-cyan-300/80">Cognitive Inputs</div>
                    <pre className="text-xs mt-1 bg-black/30 p-3 rounded-md overflow-auto">
                      {JSON.stringify(cognitive, null, 2)}
                    </pre>
                  </div>
                </div>

                {/* Right */}
                <div className="space-y-3">

                  <div>
                    <div className="text-xs text-cyan-300/80">Allele Input</div>
                    <pre className="text-xs mt-1 bg-black/30 p-3 rounded-md overflow-auto">
                      {JSON.stringify(alleleInput, null, 2)}
                    </pre>
                  </div>

                  <div>
                    <div className="text-xs text-cyan-300/80">Model Results</div>
                    <pre className="text-xs mt-1 bg-black/30 p-3 rounded-md overflow-auto">
                      {JSON.stringify(
                        {
                          baseResult,
                          alleleResult,
                          mriResult,
                          ocrResult,
                        },
                        null,
                        2
                      )}
                    </pre>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

      </main>

      {/* Responsive tweak */}
      <style jsx>{`
        @media (max-width: 768px) {
          .w-\\[520px\\] {
            width: 420px !important;
          }
          .h-\\[520px\\] {
            height: 420px !important;
          }
        }
      `}</style>
    </div>
  );
}
