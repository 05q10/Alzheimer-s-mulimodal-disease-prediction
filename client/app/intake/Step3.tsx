"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useRouter } from "next/navigation";
import { usePatientStore, AdvancedScores } from "../store/patientStore";
import { Sparkles, AlertTriangle, ChevronDown, ChevronUp, Layers, Activity, Dna } from "lucide-react";

interface Props {
  back: () => void;
}

export default function Step3({ back }: Props) {
  const router = useRouter();
  const {
    session,
    updateCognitive,
    updateAdvancedScores,
    updateMriVolumes,
    setAlleleInput,
    autofillSamplePatient,
  } = usePatientStore();

  const cognitive = session.cognitive;
  const storedAllele = session.alleleInput;
  const adv = session.advancedScores || {};

  /* ---------------------- Biomarker Local State ---------------------- */
  const [ABETA, setABETA] = useState<number | null>(storedAllele.ABETA ?? null);
  const [TAU, setTAU] = useState<number | null>(storedAllele.TAU ?? null);
  const [APVOLUME, setAPVOLUME] = useState<number | null>(storedAllele.APVOLUME ?? null);
  const [GENOTYPE, setGENOTYPE] = useState<string>(storedAllele.GENOTYPE ?? "e3/e4");

  /* ---------------------- Advanced Scores State ---------------------- */
  const [showAdvanced, setShowAdvanced] = useState<boolean>(true);
  const [cdglobal, setCdglobal] = useState<number | null>(adv.CDGLOBAL ?? null);
  const [adas11, setAdas11] = useState<number | null>(adv.ADAS11 ?? null);
  const [faq, setFaq] = useState<number | null>(adv.FAQ ?? null);
  const [moca, setMoca] = useState<number | null>(adv.MOCA ?? null);
  const [ptau, setPtau] = useState<number | null>(adv.PTAU ?? null);

  const [hippocampus, setHippocampus] = useState<number | null>(
    adv.mriVolumes?.hippocampus ?? null
  );
  const [entorhinal, setEntorhinal] = useState<number | null>(
    adv.mriVolumes?.entorhinal ?? null
  );
  const [ventricles, setVentricles] = useState<number | null>(
    adv.mriVolumes?.ventricles ?? null
  );
  const [wholeBrain, setWholeBrain] = useState<number | null>(
    adv.mriVolumes?.wholeBrain ?? null
  );
  const [icv, setIcv] = useState<number | null>(adv.mriVolumes?.icv ?? null);

  const [error, setError] = useState<string | null>(null);

  const validateAllele = () => {
    // Missing biomarkers are permitted as missing fields for T1/T2 staging
    return true;
  };

  const proceedToProcessing = () => {
    setError(null);
    if (!validateAllele()) return;

    // Save Allele inputs
    setAlleleInput({
      ABETA,
      TAU,
      APVOLUME,
      MMSE: cognitive.mmse ?? null,
      GENOTYPE,
    });

    // Save Advanced Scores (empty is kept as null / MISSING, never defaulted to 0)
    updateAdvancedScores({
      CDGLOBAL: cdglobal,
      ADAS11: adas11,
      FAQ: faq,
      MOCA: moca,
      PTAU: ptau,
    });

    updateMriVolumes({
      hippocampus,
      entorhinal,
      ventricles,
      wholeBrain,
      icv,
    });

    router.push("/processing");
  };

  // Scope check: Soft non-blocking warnings
  const mmseWarning =
    cognitive.mmse !== null && (cognitive.mmse < 0 || cognitive.mmse > 30);
  const cdrWarning = cdglobal !== null && (cdglobal < 0 || cdglobal > 3);
  const adasWarning = adas11 !== null && (adas11 < 0 || adas11 > 70);

  return (
    <div className="min-h-screen bg-transparent text-white p-4 md:p-8 rounded-xl">
      <div className="max-w-6xl mx-auto space-y-10">
        {/* HEADER */}
        <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
              Cognitive & Biomarker Evaluation
            </h1>
            <p className="text-blue-200 mt-1">
              Enter cognitive assessments, fluid biomarkers, and optional volumetric MRI metrics.
            </p>
          </div>

          <button
            type="button"
            onClick={() => {
              autofillSamplePatient();
              // Update local state to reflect autofill
              setABETA(980);
              setTAU(432);
              setAPVOLUME(3400);
              setGENOTYPE("e3/e4");
              setCdglobal(0.5);
              setAdas11(12.0);
              setFaq(4.0);
              setMoca(23.0);
              setPtau(38.0);
              setHippocampus(3400);
              setEntorhinal(1350);
              setVentricles(35000);
              setWholeBrain(1050000);
              setIcv(1450000);
            }}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white text-xs font-bold shadow-lg shadow-purple-600/30 transition hover:scale-105 cursor-pointer self-start sm:self-auto shrink-0"
          >
            <Sparkles className="w-4 h-4 text-purple-200" />
            <span>Autofill Sample Patient</span>
          </button>
        </motion.div>

        {/* GRID */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
          {/* LEFT SECTION */}
          <div className="lg:col-span-2 space-y-8">
            {/* ------------------ Core Cognitive Section ------------------ */}
            <motion.div className="bg-white/5 p-6 rounded-2xl border border-white/10 backdrop-blur-md">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5 text-blue-400" />
                <span>Primary Cognitive Screening</span>
              </h2>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <RangeBlock
                  label="MMSE (0–30)"
                  min={0}
                  max={30}
                  value={cognitive.mmse ?? 26}
                  onChange={(v) => updateCognitive({ mmse: v })}
                />

                <RangeBlock
                  label="Functional Assessment (0–10)"
                  min={0}
                  max={10}
                  value={cognitive.functionalAssessment ?? 6}
                  onChange={(v) => updateCognitive({ functionalAssessment: v })}
                />

                <RangeBlock
                  label="ADL (0–10)"
                  min={0}
                  max={10}
                  value={cognitive.adl ?? 4}
                  onChange={(v) => updateCognitive({ adl: v })}
                />
              </div>

              {mmseWarning && (
                <div className="mt-3 text-xs text-amber-300 bg-amber-500/10 border border-amber-500/30 p-2.5 rounded-xl flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
                  <span>MMSE is outside the model's training range (0–30). Treat outputs as illustrative.</span>
                </div>
              )}

              {/* ------------------ Symptoms ------------------ */}
              <h3 className="text-lg font-semibold mt-6 mb-2">Neurological Symptoms (Yes/No)</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <ToggleBlock
                  label="Memory Complaints"
                  value={cognitive.memoryComplaints ?? 0}
                  onChange={(v) => updateCognitive({ memoryComplaints: v })}
                />
                <ToggleBlock
                  label="Behavioral Problems"
                  value={cognitive.behavioralProblems ?? 0}
                  onChange={(v) => updateCognitive({ behavioralProblems: v })}
                />
                <ToggleBlock
                  label="Confusion"
                  value={cognitive.confusion ?? 0}
                  onChange={(v) => updateCognitive({ confusion: v })}
                />
                <ToggleBlock
                  label="Disorientation"
                  value={cognitive.disorientation ?? 0}
                  onChange={(v) => updateCognitive({ disorientation: v })}
                />
                <ToggleBlock
                  label="Personality Changes"
                  value={cognitive.personalityChanges ?? 0}
                  onChange={(v) => updateCognitive({ personalityChanges: v })}
                />
                <ToggleBlock
                  label="Difficulty Completing Tasks"
                  value={cognitive.difficultyCompletingTasks ?? 0}
                  onChange={(v) => updateCognitive({ difficultyCompletingTasks: v })}
                />
                <ToggleBlock
                  label="Forgetfulness"
                  value={cognitive.forgetfulness ?? 0}
                  onChange={(v) => updateCognitive({ forgetfulness: v })}
                />
              </div>
            </motion.div>

            {/* ------------------ Optional Advanced Scores & MRI Volumes ------------------ */}
            <motion.div className="bg-white/5 p-6 rounded-2xl border border-white/10 backdrop-blur-md">
              <button
                type="button"
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="w-full flex items-center justify-between text-left cursor-pointer"
              >
                <div className="flex items-center gap-2.5">
                  <Layers className="w-5 h-5 text-purple-400" />
                  <div>
                    <h2 className="text-xl font-semibold text-white">
                      Advanced Scores & Volumetric MRI (Optional)
                    </h2>
                    <p className="text-xs text-neutral-400 mt-0.5">
                      Empty fields remain unobserved (MISSING). Never defaulted to 0.
                    </p>
                  </div>
                </div>
                {showAdvanced ? (
                  <ChevronUp className="w-5 h-5 text-neutral-400" />
                ) : (
                  <ChevronDown className="w-5 h-5 text-neutral-400" />
                )}
              </button>

              <AnimatePresence>
                {showAdvanced && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mt-6 pt-6 border-t border-white/10 space-y-6"
                  >
                    <div>
                      <h3 className="text-sm font-semibold text-purple-300 uppercase tracking-wide mb-3">
                        Neuropsychological Batteries (T2 Escalation)
                      </h3>
                      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
                        <NumberBlock
                          label="CDR-Global (0–3)"
                          value={cdglobal}
                          onChange={setCdglobal}
                          placeholder="e.g. 0.5"
                          step={0.5}
                        />
                        <NumberBlock
                          label="ADAS11-Cog (0–70)"
                          value={adas11}
                          onChange={setAdas11}
                          placeholder="e.g. 12"
                          step={0.5}
                        />
                        <NumberBlock
                          label="FAQ (0–30)"
                          value={faq}
                          onChange={setFaq}
                          placeholder="e.g. 4"
                        />
                        <NumberBlock
                          label="MOCA (0–30)"
                          value={moca}
                          onChange={setMoca}
                          placeholder="e.g. 23"
                        />
                      </div>

                      {(cdrWarning || adasWarning) && (
                        <div className="mt-3 text-xs text-amber-300 bg-amber-500/10 border border-amber-500/30 p-2.5 rounded-xl flex items-center gap-2">
                          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
                          <span>Outside the model's training range, treat results as illustrative.</span>
                        </div>
                      )}
                    </div>

                    <div>
                      <h3 className="text-sm font-semibold text-cyan-300 uppercase tracking-wide mb-3">
                        Structural MRI Volumetrics (T4 Escalation)
                      </h3>
                      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                        <NumberBlock
                          label="Hippocampus (mm³)"
                          value={hippocampus}
                          onChange={setHippocampus}
                          placeholder="e.g. 3400"
                        />
                        <NumberBlock
                          label="Entorhinal (mm³)"
                          value={entorhinal}
                          onChange={setEntorhinal}
                          placeholder="e.g. 1350"
                        />
                        <NumberBlock
                          label="Ventricles (mm³)"
                          value={ventricles}
                          onChange={setVentricles}
                          placeholder="e.g. 35000"
                        />
                        <NumberBlock
                          label="Whole Brain (mm³)"
                          value={wholeBrain}
                          onChange={setWholeBrain}
                          placeholder="e.g. 1050000"
                        />
                        <NumberBlock
                          label="ICV (mm³)"
                          value={icv}
                          onChange={setIcv}
                          placeholder="e.g. 1450000"
                        />
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

            {/* ------------------ Biomarkers Section ------------------ */}
            <motion.div className="bg-white/5 p-6 rounded-2xl border border-white/10 backdrop-blur-md">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Dna className="w-5 h-5 text-emerald-400" />
                <span>Fluid Biomarkers (T3 Cascade)</span>
              </h2>

              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <NumberBlock
                  label="Aβ42 (pg/mL)"
                  value={ABETA}
                  onChange={setABETA}
                  placeholder="e.g. 980"
                />
                <NumberBlock
                  label="Total Tau (pg/mL)"
                  value={TAU}
                  onChange={setTAU}
                  placeholder="e.g. 432"
                />
                <NumberBlock
                  label="p-Tau181 (pg/mL)"
                  value={ptau}
                  onChange={setPtau}
                  placeholder="e.g. 38"
                />
                <NumberBlock
                  label="APVOLUME (mm³)"
                  value={APVOLUME}
                  onChange={setAPVOLUME}
                  placeholder="e.g. 3400"
                />
              </div>

              <div className="mt-4">
                <label className="text-sm text-blue-200">APOE Genotype (Allele Risk)</label>
                <select
                  value={GENOTYPE}
                  onChange={(e) => setGENOTYPE(e.target.value)}
                  className="w-full mt-2 px-3 py-2.5 rounded-xl bg-slate-800 border border-white/10 text-white font-mono text-sm"
                >
                  <option value="e3/e3">e3/e3 (0 APOE-ε4 alleles — Neutral risk)</option>
                  <option value="e2/e3">e2/e3 (0 APOE-ε4 alleles — Protective)</option>
                  <option value="e2/e2">e2/e2 (0 APOE-ε4 alleles — Protective)</option>
                  <option value="e3/e4">e3/e4 (1 APOE-ε4 allele — Moderate risk)</option>
                  <option value="e2/e4">e2/e4 (1 APOE-ε4 allele — Moderate risk)</option>
                  <option value="e4/e4">e4/e4 (2 APOE-ε4 alleles — High risk)</option>
                  <option value="N/A">N/A (Missing / Not sequenced)</option>
                </select>
              </div>

              {error && <div className="text-rose-400 text-sm mt-3">{error}</div>}
            </motion.div>
          </div>

          {/* ------------------ Right Sidebar ------------------ */}
          <div className="space-y-6">
            <motion.div className="bg-gradient-to-br from-blue-500/10 to-cyan-500/10 p-6 rounded-3xl border border-blue-400/20 backdrop-blur-xl sticky top-8">
              <h3 className="text-lg font-bold mb-4 text-white">Review Summary</h3>

              <SummaryRow label="Patient" value={session.demographics.name ?? "—"} />
              <SummaryRow label="Age" value={session.demographics.age ? `${session.demographics.age} y` : "—"} />
              <SummaryRow label="Sex" value={session.demographics.gender === 1 ? "Female" : session.demographics.gender === 0 ? "Male" : "—"} />
              <SummaryRow label="MMSE" value={cognitive.mmse ?? "—"} />
              <SummaryRow label="CDR-Global" value={cdglobal !== null ? cdglobal : "—"} />
              <SummaryRow label="ADAS11" value={adas11 !== null ? adas11 : "—"} />

              <div className="h-px bg-white/10 my-3"></div>

              <div className="text-xs uppercase tracking-wide text-neutral-400 mb-1">Biomarkers & Genes</div>
              <div className="text-xs text-white/90 font-mono">
                Aβ: {ABETA ?? "—"} · Tau: {TAU ?? "—"} · p-Tau: {ptau ?? "—"}
              </div>
              <div className="text-xs text-white/90 font-mono mt-1">
                APOE: {GENOTYPE} · APVOL: {APVOLUME ?? "—"}
              </div>

              <div className="flex gap-3 mt-6">
                <button
                  type="button"
                  onClick={back}
                  className="px-4 py-3 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-xs font-semibold transition cursor-pointer"
                >
                  ← Back
                </button>

                <button
                  type="button"
                  onClick={proceedToProcessing}
                  className="flex-1 px-4 py-3 rounded-xl bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white font-bold text-xs shadow-lg shadow-blue-600/30 transition hover:scale-[1.02] cursor-pointer text-center"
                >
                  Run Analysis →
                </button>
              </div>
            </motion.div>
          </div>
        </div>
      </div>

      {/* STYLES */}
      <style jsx>{`
        .slider {
          height: 6px;
          background: linear-gradient(to right, #3b82f6 0%, #3b82f6 100%);
          border-radius: 10px;
          appearance: none;
        }
        .slider::-webkit-slider-thumb {
          appearance: none;
          width: 18px;
          height: 18px;
          border-radius: 50%;
          background: linear-gradient(135deg, #3b82f6, #06b6d4);
          border: 3px solid white;
          cursor: pointer;
        }
      `}</style>
    </div>
  );
}

/* ------------------------------------------------------- */
/* INPUT HELPER COMPONENTS */
/* ------------------------------------------------------- */

function RangeBlock({
  label,
  min,
  max,
  value,
  onChange,
}: {
  label: string;
  min: number;
  max: number;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="space-y-2">
      <label className="text-sm text-blue-200">{label}</label>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="slider w-full"
      />
      <div className="flex justify-between text-xs text-blue-300">
        <span>{min}</span>
        <span className="font-semibold">{value}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}

function NumberBlock({
  label,
  value,
  onChange,
  placeholder,
  step,
}: {
  label: string;
  value: number | null;
  onChange: (v: number | null) => void;
  placeholder?: string;
  step?: number;
}) {
  return (
    <div>
      <label className="text-xs text-blue-200 block mb-1">{label}</label>
      <input
        type="number"
        step={step}
        value={value !== null && value !== undefined ? value : ""}
        onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))}
        className="w-full px-3 py-2 rounded-xl bg-white/5 border border-white/10 text-white font-mono text-sm placeholder:text-neutral-500 focus:outline-none focus:border-blue-400"
        placeholder={placeholder}
      />
    </div>
  );
}

function ToggleBlock({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="p-3.5 bg-white/5 rounded-xl border border-white/10">
      <div className="flex justify-between items-center mb-2">
        <span className="text-xs text-blue-200">{label}</span>
        <span className="font-semibold text-xs">{value === 1 ? "Yes" : "No"}</span>
      </div>

      <div className="flex gap-2">
        {[0, 1].map((opt) => (
          <button
            key={opt}
            type="button"
            onClick={() => onChange(opt)}
            className={`flex-1 py-1.5 rounded-lg text-xs font-medium cursor-pointer transition ${
              value === opt
                ? "bg-gradient-to-r from-blue-600 to-cyan-600 text-white font-bold shadow-md shadow-blue-600/30"
                : "bg-white/5 text-neutral-400 hover:text-white"
            }`}
          >
            {opt === 1 ? "Yes" : "No"}
          </button>
        ))}
      </div>
    </div>
  );
}

function SummaryRow({ label, value }: { label: string; value: any }) {
  return (
    <div className="flex justify-between text-xs mb-1.5">
      <span className="text-neutral-400">{label}</span>
      <span className="font-medium text-white font-mono">{value}</span>
    </div>
  );
}
