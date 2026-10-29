// client/app/processing/page.tsx
"use client";

import React, { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { useRouter } from "next/navigation";
import { usePatientStore } from "../store/patientStore";
import { API_BASE_URL, twinApi } from "../lib/api";

export default function ProcessingPage() {
  const router = useRouter();

  // Zustand actions
  const session = usePatientStore((s) => s.session);
  const setOCRResult = usePatientStore((s) => s.setOCRResult);
  const setMRIResult = usePatientStore((s) => s.setMRIResult);
  const setAlleleResult = usePatientStore((s) => s.setAlleleResult);
  const setBaseModelResult = usePatientStore((s) => s.setBaseModelResult);
  const setTwin = usePatientStore((s) => s.setTwin);
  const setTwinId = usePatientStore((s) => s.setTwinId);

  const BASE_URL = API_BASE_URL;

  const [logs, setLogs] = useState<string[]>([]);
  const [stepIndex, setStepIndex] = useState(0);
  const [busy, setBusy] = useState(true);

  const startedRef = useRef(false);

  const pushLog = (text: string) =>
    setLogs((l) => [...l, `${new Date().toLocaleTimeString()} — ${text}`]);

  useEffect(() => {
    if (startedRef.current) return; // prevent double-run in StrictMode / re-mounts
    startedRef.current = true;

    let mounted = true;

    async function runPipeline() {
      pushLog("Initializing cognitive analysis engine...");
      await delay(600);

      // ---------- 1) OCR ----------
      try {
        if (session.uploads.reportFile) {
          pushLog("Uploading blood report for OCR extraction...");
          setStepIndex(1);
          const fd = new FormData();
          fd.append("file", session.uploads.reportFile as File);

          const res = await fetch(`${BASE_URL}/extract-report`, {
            method: "POST",
            body: fd,
          });

          if (!res.ok) throw new Error(`OCR status ${res.status}`);
          const ocrJson = await res.json();
          setOCRResult(ocrJson);
          pushLog("OCR extraction complete — values parsed.");
        } else {
          pushLog("No blood report uploaded — skipping OCR.");
        }
      } catch (err: any) {
        pushLog(`OCR extraction failed: ${err?.message ?? String(err)}`);
      }

      await delay(500);

      // ---------- 2) MRI ----------
      try {
        if (session.uploads.mriFile) {
          pushLog("Uploading MRI scan for image analysis...");
          setStepIndex(2);
          const fd = new FormData();
          fd.append("file", session.uploads.mriFile as File);

          const res = await fetch(`${BASE_URL}/image`, {
            method: "POST",
            body: fd,
          });

          if (!res.ok) throw new Error(`MRI status ${res.status}`);
          const mriJson = await res.json();
          setMRIResult(mriJson);
          pushLog("MRI analysis complete — image model returned results.");
        } else {
          pushLog("No MRI uploaded — skipping MRI analysis.");
        }
      } catch (err: any) {
        pushLog(`MRI analysis failed: ${err?.message ?? String(err)}`);
      }

      await delay(500);

      // ---------- 3) Allele ----------
      try {
        const alleleInput = session.alleleInput;
        if (
          alleleInput &&
          (alleleInput.ABETA != null ||
            alleleInput.TAU != null ||
            alleleInput.APVOLUME != null)
        ) {
          pushLog("Running allele risk ensemble model...");
          setStepIndex(3);

          const alleleBody = {
            ABETA: alleleInput.ABETA ?? null,
            TAU: alleleInput.TAU ?? null,
            MMSE: alleleInput.MMSE ?? session.cognitive.mmse ?? 0,
            APVOLUME: alleleInput.APVOLUME ?? null,
            GENOTYPE: alleleInput.GENOTYPE ?? "N/A",
          };

          const res = await fetch(`${BASE_URL}/alele`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(alleleBody),
          });

          if (!res.ok) throw new Error(`Allele status ${res.status}`);
          const alleleJson = await res.json();

          setAlleleResult({
            predictedClass: alleleJson.predicted_class ?? "Unknown",
            probability: alleleJson.probability ?? null,
            riskCategory: alleleJson.risk_category ?? "Unknown",
          });


          pushLog("Allele ensemble finished — risk estimated.");
        } else {
          pushLog("Insufficient allele inputs — skipping allele model.");
        }
      } catch (err: any) {
        pushLog(`Allele model failed: ${err?.message ?? String(err)}`);
      }

      await delay(600);

      // ---------- 4) Base model (send full expected payload) ----------
      try {
        pushLog("Running primary Alzheimer classifier (base model)...");
        setStepIndex(4);

        // Build a complete payload covering all expected keys from the backend model spec.
        // Use sensible defaults (0 or null) where data is missing.
        const payload: Record<string, any> = {
  PatientID: Number(session.id.replace(/\D/g, "").slice(0, 6)) || 999999,

  Age: session.demographics.age ?? 0,
  Gender: session.demographics.gender ?? 0,
  Ethnicity: session.demographics.ethnicity ?? 3,
  EducationLevel: session.demographics.educationLevel ?? 0,

  BMI: session.demographics.bmi ?? 0,
  Smoking: session.demographics.smoking ?? 0,
  AlcoholConsumption: session.demographics.alcoholConsumption ?? 0,
  PhysicalActivity: session.demographics.physicalActivity ?? 0,
  DietQuality: session.demographics.dietQuality ?? 0,
  SleepQuality: session.demographics.sleepQuality ?? 0,

  FamilyHistoryAlzheimers: session.demographics.familyHistoryAlzheimers ?? 0,

  CardiovascularDisease: session.medical.cardiovascularDisease ?? 0,
  Diabetes: session.medical.diabetes ?? 0,
  Depression: session.medical.depression ?? 0,
  HeadInjury: session.medical.headInjury ?? 0,
  Hypertension: session.medical.hypertension ?? 0,

  SystolicBP: session.medical.systolicBP ?? 0,
  DiastolicBP: session.medical.diastolicBP ?? 0,
  CholesterolTotal: session.medical.cholesterolTotal ?? 0,
  CholesterolLDL: session.medical.cholesterolLDL ?? 0,
  CholesterolHDL: session.medical.cholesterolHDL ?? 0,
  CholesterolTriglycerides: session.medical.cholesterolTriglycerides ?? 0,

  MMSE: session.cognitive.mmse ?? 0,
  FunctionalAssessment: session.cognitive.functionalAssessment ?? 0,
  MemoryComplaints: session.cognitive.memoryComplaints ?? 0,
  BehavioralProblems: session.cognitive.behavioralProblems ?? 0,
  ADL: session.cognitive.adl ?? 0,

  Confusion: session.cognitive.confusion ?? 0,
  Disorientation: session.cognitive.disorientation ?? 0,
  PersonalityChanges: session.cognitive.personalityChanges ?? 0,
  DifficultyCompletingTasks: session.cognitive.difficultyCompletingTasks ?? 0,
  Forgetfulness: session.cognitive.forgetfulness ?? 0,

  DoctorInCharge: 5 // backend example is always a number
};


        // POST payload to backend
        const res = await fetch(`${BASE_URL}/base`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!res.ok) throw new Error(`Base status ${res.status}`);
        const baseJson = await res.json();

        // Save results to store (use payload as fallback if backend didn't return input_used)
        setBaseModelResult(baseJson.input_used ?? payload, baseJson.prediction ?? "N/A");
        pushLog("Base model completed — primary prediction ready.");
      } catch (err: any) {
        pushLog(`Base model failed: ${err?.message ?? String(err)}`);
      }

      await delay(600);

      pushLog("Synthesizing patient-specific Digital Twin...");
      setStepIndex(5);
      await delay(600);

      let createdTwinId = session.twinId;
      try {
        const educationMap: Record<number, number> = { 0: 8, 1: 12, 2: 16, 3: 18 };
        const eduYears =
          session.demographics.educationLevel !== null && session.demographics.educationLevel !== undefined
            ? educationMap[session.demographics.educationLevel] ?? 12
            : 12;

        const genotype = session.alleleInput.GENOTYPE || "N/A";
        let apoe4 = 0;
        if (genotype.includes("4_4") || genotype.includes("4/4")) apoe4 = 2;
        else if (genotype.includes("4")) apoe4 = 1;
        else if (session.demographics.familyHistoryAlzheimers === 1) apoe4 = 1;

        const twin = await twinApi.createTwin({
          name: session.demographics.name || "Patient",
          baseline_date: session.demographics.baselineDate || new Date().toISOString().split("T")[0],
          demographics: {
            age: session.demographics.age || 70,
            sex: session.demographics.gender === 1 ? 1 : 0,
            education_years: eduYears,
            apoe4,
            baseline_date: session.demographics.baselineDate || new Date().toISOString().split("T")[0],
          },
          scores: {
            MMSE: session.cognitive.mmse ?? 26,
            CDGLOBAL: session.advancedScores?.CDGLOBAL ?? undefined,
            TOTSCORE: session.advancedScores?.ADAS11 ?? undefined,
            FAQ: session.advancedScores?.FAQ ?? undefined,
            MOCA: session.advancedScores?.MOCA ?? undefined,
          },
          biomarkers: {
            ABETA: session.alleleInput.ABETA ?? undefined,
            TAU: session.alleleInput.TAU ?? undefined,
            PTAU: session.advancedScores?.PTAU ?? undefined,
            APVOLUME: session.alleleInput.APVOLUME ?? undefined,
          },
          mri_volumes: session.advancedScores?.mriVolumes
            ? {
                hippocampus: session.advancedScores.mriVolumes.hippocampus ?? null,
                entorhinal: session.advancedScores.mriVolumes.entorhinal ?? null,
                ventricles: session.advancedScores.mriVolumes.ventricles ?? null,
                whole_brain: (session.advancedScores.mriVolumes as any).whole_brain ?? session.advancedScores.mriVolumes.wholeBrain ?? null,
                icv: session.advancedScores.mriVolumes.icv ?? null,
              }
            : undefined,
          lifestyle: {
            smoking: session.demographics.smoking,
            alcoholConsumption: session.demographics.alcoholConsumption,
            physicalActivity: session.demographics.physicalActivity,
            dietQuality: session.demographics.dietQuality,
            sleepQuality: session.demographics.sleepQuality,
            hypertension: session.medical?.hypertension,
            diabetes: session.medical?.diabetes,
            depression: session.medical?.depression,
            headInjury: session.medical?.headInjury,
            cardiovascularDisease: session.medical?.cardiovascularDisease,
          },
          is_synthetic: false,
          persona_label: session.demographics.name
            ? `Patient ${session.demographics.name}`
            : "Clinical Twin",
        });

        // Attach modalities if files were uploaded
        if (session.uploads.mriFile) {
          try {
            pushLog("Uploading MRI scan to /twin/{id}/mri-image and running multimodal fusion...");
            const mriRes = await twinApi.uploadMRIImage(
              twin.twin_id,
              session.uploads.mriFile
            );
            pushLog(`MRI scan classified: ${mriRes.mri.predicted_class}. Multimodal fusion complete.`);
            twin.state = mriRes.state;
            twin.modalities.mri_image = {
              present: true,
              latest: mriRes.mri,
            };
            setMRIResult(mriRes.mri);
            usePatientStore.getState().setMRIUploadResult(mriRes);
          } catch (e: any) {
            pushLog(`MRI upload error: ${e?.message ?? String(e)}`);
            console.warn("Failed to attach MRI file to twin:", e);
          }
        }
        if (session.uploads.reportFile) {
          try {
            await twinApi.attachModality(
              twin.twin_id,
              "ocr_report",
              undefined,
              session.uploads.reportFile
            );
          } catch (e) {
            console.warn("Failed to attach report file to twin:", e);
          }
        }

        setTwin(twin);
        setTwinId(twin.twin_id);
        createdTwinId = twin.twin_id;
        pushLog("Digital twin synthesized and calibrated successfully.");
      } catch (err: any) {
        pushLog(`Digital twin synthesis notice: ${err?.message ?? String(err)}`);
      }

      pushLog("Done. Directing to Digital Twin workspace...");
      setBusy(false);
      await delay(600);

      router.push(`/twin/${createdTwinId || session.twinId || "demo-cn-mci"}`);
    }

    runPipeline();

    return () => {
      mounted = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // it's safe to include session, but run guarded by startedRef


  return (
    <div className="min-h-screen bg-transparent text-white relative overflow-hidden">
      {/* Neural Background - Fixed z-index */}
      <NeuralMesh />

      {/* Content Layer */}
      <div className="relative z-10 max-w-4xl mx-auto pt-24 px-6 pb-20">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-[#0b172e]/85 backdrop-blur-xl border border-cyan-500/30 rounded-3xl p-8 shadow-2xl shadow-cyan-500/20"
        >
          {/* Header */}
          <div className="mb-6">
            <div className="flex items-center gap-3 mb-2">
              <div className="w-3 h-3 bg-cyan-400 rounded-full animate-pulse shadow-lg shadow-cyan-400/50"></div>
              <h2 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
                Neural Processing Engine
              </h2>
            </div>
            <p className="text-cyan-200/70 text-sm ml-6">
              Analyzing multimodal data through AI pipeline
            </p>
          </div>

          {/* Terminal Display */}
          <div className="bg-[#070e1d]/90 border border-cyan-500/30 rounded-2xl p-6 font-mono text-sm shadow-inner">
            <div className="flex items-center gap-2 mb-4 pb-3 border-b border-cyan-500/20">
              <div className="flex gap-1.5">
                <div className="w-3 h-3 rounded-full bg-red-500/80"></div>
                <div className="w-3 h-3 rounded-full bg-yellow-500/80"></div>
                <div className="w-3 h-3 rounded-full bg-green-500/80"></div>
              </div>
              <span className="text-cyan-400/60 text-xs ml-2">system/neural-analyzer</span>
            </div>

            <div className="h-64 overflow-y-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-cyan-500/30">
              {logs.length === 0 ? (
                <div className="text-cyan-400/60 flex items-center gap-2">
                  <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse"></div>
                  Initializing neural pathways...
                </div>
              ) : (
                logs.map((l, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3 }}
                    className="mb-2 flex items-start gap-2"
                  >
                    <span className="text-cyan-400 mt-0.5">▸</span>
                    <span className="text-green-300 flex-1">{l}</span>
                  </motion.div>
                ))
              )}
              {busy && (
                <motion.div
                  animate={{ opacity: [0.5, 1, 0.5] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                  className="text-cyan-400/60 flex items-center gap-2 mt-2"
                >
                  <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full"></div>
                  <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full animation-delay-150"></div>
                  <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full animation-delay-300"></div>
                </motion.div>
              )}
            </div>
          </div>

          {/* Progress Bar */}
          <div className="mt-6">
            <div className="flex items-center justify-between mb-3">
              <div className="text-sm text-cyan-200">
                {busy ? "Processing..." : "Analysis Complete"}
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-cyan-400/60">Step</span>
                <span className="text-cyan-400 font-bold text-lg">{stepIndex}</span>
                <span className="text-cyan-400/40">/</span>
                <span className="text-cyan-400/60">5</span>
              </div>
            </div>

            <div className="relative h-3 bg-slate-800 rounded-full overflow-hidden border border-cyan-500/20">
              <motion.div
                className="absolute inset-y-0 left-0 bg-gradient-to-r from-cyan-500 via-blue-500 to-purple-500 shadow-lg shadow-cyan-500/50"
                initial={{ width: "0%" }}
                animate={{ width: `${(stepIndex / 5) * 100}%` }}
                transition={{ duration: 0.5, ease: "easeOut" }}
              />
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer"></div>
            </div>

            {/* Step Labels */}
            <div className="flex justify-between mt-2 px-1">
              {["OCR", "MRI", "Allele", "Base", "Twin"].map((label, i) => (
                <div
                  key={i}
                  className={`text-xs transition-colors ${
                    stepIndex > i
                      ? "text-cyan-400 font-semibold"
                      : stepIndex === i
                      ? "text-cyan-300"
                      : "text-cyan-400/30"
                  }`}
                >
                  {label}
                </div>
              ))}
            </div>
          </div>

          {busy && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="mt-6 text-center text-cyan-400/60 text-sm flex items-center justify-center gap-2"
            >
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                  fill="none"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              Please wait while we process your data
            </motion.div>
          )}
        </motion.div>
      </div>

      <style jsx>{`
        @keyframes shimmer {
          0% {
            transform: translateX(-100%);
          }
          100% {
            transform: translateX(100%);
          }
        }
        .animate-shimmer {
          animation: shimmer 2s infinite;
        }
      `}</style>
    </div>
  );
}

function delay(ms: number) {
  return new Promise((res) => setTimeout(res, ms));
}

/* Enhanced Neural Mesh Background */
function NeuralMesh() {
  return (
    <div className="fixed inset-0 z-0">
      {/* Base Gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950" />

      {/* Animated Grid */}
      <div className="absolute inset-0 opacity-30">
        <div 
          className="absolute inset-0 animate-gridMove"
          style={{
            backgroundImage: `
              linear-gradient(rgba(6, 182, 212, 0.3) 1px, transparent 1px),
              linear-gradient(90deg, rgba(6, 182, 212, 0.3) 1px, transparent 1px)
            `,
            backgroundSize: '60px 60px'
          }}
        />
      </div>

      {/* Central Glow */}
      <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
        <div className="w-[800px] h-[800px] rounded-full bg-cyan-500/20 blur-[200px] animate-pulse-slow" />
      </div>

      {/* Neural Nodes */}
      <svg className="absolute inset-0 w-full h-full opacity-40">
        <defs>
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Connecting Lines */}
        {generateConnections().map((line, i) => (
          <motion.line
            key={`line-${i}`}
            x1={line.x1}
            y1={line.y1}
            x2={line.x2}
            y2={line.y2}
            stroke="rgba(6, 182, 212, 0.4)"
            strokeWidth="1"
            filter="url(#glow)"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ 
              pathLength: [0, 1, 0],
              opacity: [0, 0.6, 0]
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              delay: i * 0.2,
              ease: "easeInOut"
            }}
          />
        ))}

        {/* Neural Nodes */}
        {generateNodes().map((node, i) => (
          <motion.circle
            key={`node-${i}`}
            cx={node.x}
            cy={node.y}
            r="4"
            fill="rgba(6, 182, 212, 0.8)"
            filter="url(#glow)"
            animate={{
              r: [3, 6, 3],
              opacity: [0.5, 1, 0.5]
            }}
            transition={{
              duration: 2 + Math.random() * 2,
              repeat: Infinity,
              delay: Math.random() * 2,
              ease: "easeInOut"
            }}
          />
        ))}
      </svg>

      {/* Floating Particles */}
      <div className="absolute inset-0">
        {Array.from({ length: 30 }).map((_, i) => (
          <motion.div
            key={`particle-${i}`}
            className="absolute w-1 h-1 rounded-full bg-cyan-400 shadow-lg shadow-cyan-400/50"
            animate={{
              x: [
                random(-300, 300),
                random(-200, 200),
                random(-250, 250),
                random(-300, 300),
              ],
              y: [
                random(-300, 300),
                random(-150, 150),
                random(-250, 250),
                random(-300, 300),
              ],
              opacity: [0.2, 0.8, 0.2],
              scale: [0.5, 1.2, 0.5]
            }}
            transition={{
              duration: random(8, 15),
              repeat: Infinity,
              ease: "easeInOut",
              delay: Math.random() * 5
            }}
            style={{
              top: '50%',
              left: '50%',
            }}
          />
        ))}
      </div>

      {/* Scanning Line Effect */}
      <motion.div
        className="absolute inset-x-0 h-px bg-gradient-to-r from-transparent via-cyan-400 to-transparent opacity-40"
        animate={{
          top: ['0%', '100%']
        }}
        transition={{
          duration: 4,
          repeat: Infinity,
          ease: "linear"
        }}
      />

      <style jsx>{`
        @keyframes gridMove {
          0% {
            background-position: 0px 0px, 0px 0px;
          }
          100% {
            background-position: 60px 60px, 60px 60px;
          }
        }
        .animate-gridMove {
          animation: gridMove 10s linear infinite;
        }
        @keyframes pulse-slow {
          0%, 100% {
            opacity: 0.2;
          }
          50% {
            opacity: 0.4;
          }
        }
        .animate-pulse-slow {
          animation: pulse-slow 4s ease-in-out infinite;
        }
      `}</style>
    </div>
  );
}

function random(min: number, max: number) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

// Generate neural network nodes
function generateNodes() {
  const nodes = [];
  const width = typeof window !== 'undefined' ? window.innerWidth : 1920;
  const height = typeof window !== 'undefined' ? window.innerHeight : 1080;
  
  for (let i = 0; i < 40; i++) {
    nodes.push({
      x: Math.random() * width,
      y: Math.random() * height
    });
  }
  return nodes;
}

// Generate connections between nodes
function generateConnections() {
  const connections = [];
  const width = typeof window !== 'undefined' ? window.innerWidth : 1920;
  const height = typeof window !== 'undefined' ? window.innerHeight : 1080;
  
  for (let i = 0; i < 30; i++) {
    connections.push({
      x1: Math.random() * width,
      y1: Math.random() * height,
      x2: Math.random() * width,
      y2: Math.random() * height
    });
  }
  return connections;
}