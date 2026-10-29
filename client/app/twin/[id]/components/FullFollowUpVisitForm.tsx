"use client";

import React, { useState } from "react";
import { TwinFull, twinApi } from "@/app/lib/api";
import { usePatientStore } from "@/app/store/patientStore";
import {
  CalendarPlus,
  Sparkles,
  Upload,
  FileText,
  FileImage,
  ImageIcon,
  Check,
  AlertCircle,
  Activity,
  Brain,
  Dna,
  Heart,
  ChevronDown,
  ChevronUp,
  X,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface FullFollowUpVisitFormProps {
  twin: TwinFull;
  onVisitAdded: (updatedTwin: TwinFull) => void;
}

export default function FullFollowUpVisitForm({
  twin,
  onVisitAdded,
}: FullFollowUpVisitFormProps) {
  const session = usePatientStore((s) => s.session);
  const todayStr = new Date().toISOString().split("T")[0];
  const baselineDateStr = twin.baseline_date || todayStr;

  // Previous visit record
  const lastVisit =
    twin.visits && twin.visits.length > 0
      ? twin.visits[twin.visits.length - 1]
      : ({} as any);

  // Derive initial next month
  const nextMonthDefault =
    lastVisit.month != null ? lastVisit.month + 6 : 6;

  // Derive initial exam date (capped at today)
  const computeInitialDate = (offsetMonths: number): string => {
    try {
      const b = new Date(baselineDateStr);
      b.setMonth(b.getMonth() + offsetMonths);
      const targetStr = b.toISOString().split("T")[0];
      return targetStr <= todayStr ? targetStr : todayStr;
    } catch {
      return todayStr;
    }
  };

  // 1. Visit Metadata
  const [examDate, setExamDate] = useState<string>(computeInitialDate(nextMonthDefault));
  const [month, setMonth] = useState<number>(nextMonthDefault);

  // Track touched fields for "carried forward" badge
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const mark = (key: string) =>
    setTouched((prev) => (prev[key] ? prev : { ...prev, [key]: true }));

  // 2. Cognitive & Neuropsych Scores
  const [mmse, setMmse] = useState<number>(lastVisit.MMSE ?? 24);
  const [cdglobal, setCdglobal] = useState<number>(lastVisit.CDGLOBAL ?? 0.5);
  const [totscore, setTotscore] = useState<number>(lastVisit.TOTSCORE ?? 14.0);
  const [faq, setFaq] = useState<string>(lastVisit.FAQ != null ? String(lastVisit.FAQ) : "4");
  const [moca, setMoca] = useState<string>(lastVisit.MOCA != null ? String(lastVisit.MOCA) : "22");
  const [functionalAssessment, setFunctionalAssessment] = useState<number>(
    lastVisit.functionalAssessment ?? (session.cognitive?.functionalAssessment != null ? session.cognitive.functionalAssessment : 7)
  );
  const [adl, setAdl] = useState<number>(
    lastVisit.adl ?? (session.cognitive?.adl != null ? session.cognitive.adl : 8)
  );

  // 3. 7 Symptoms
  const defaultSymptoms = lastVisit.symptoms || {
    memoryComplaints: session.cognitive?.memoryComplaints === 1,
    behavioralProblems: session.cognitive?.behavioralProblems === 1,
    confusion: session.cognitive?.confusion === 1,
    disorientation: session.cognitive?.disorientation === 1,
    personalityChanges: session.cognitive?.personalityChanges === 1,
    difficultyCompletingTasks: session.cognitive?.difficultyCompletingTasks === 1,
    forgetfulness: session.cognitive?.forgetfulness === 1,
  };
  const [symptoms, setSymptoms] = useState<Record<string, boolean>>(defaultSymptoms);

  // 4. Fluid Biomarkers
  const defaultBio = lastVisit.biomarkers || twin.modalities.biomarkers?.latest || {};
  const [abeta, setAbeta] = useState<string>(
    defaultBio.ABETA != null ? String(defaultBio.ABETA) : "850"
  );
  const [tau, setTau] = useState<string>(
    defaultBio.TAU != null ? String(defaultBio.TAU) : "460"
  );
  const [ptau, setPtau] = useState<string>(
    defaultBio.PTAU != null ? String(defaultBio.PTAU) : "42"
  );
  const [apvolume, setApvolume] = useState<string>(
    defaultBio.APVOLUME != null ? String(defaultBio.APVOLUME) : "2900"
  );

  // 5. Volumetric MRI
  const defaultMri = lastVisit.mri_volumes || twin.modalities.mri_volumes?.latest || {};
  const [hippocampus, setHippocampus] = useState<string>(
    defaultMri.hippocampus != null ? String(defaultMri.hippocampus) : "3300"
  );
  const [entorhinal, setEntorhinal] = useState<string>(
    defaultMri.entorhinal != null ? String(defaultMri.entorhinal) : "1300"
  );
  const [ventricles, setVentricles] = useState<string>(
    defaultMri.ventricles != null ? String(defaultMri.ventricles) : "38000"
  );
  const [wholeBrain, setWholeBrain] = useState<string>(
    defaultMri.whole_brain != null ? String(defaultMri.whole_brain) : "1020000"
  );
  const [icv, setIcv] = useState<string>(
    defaultMri.icv != null ? String(defaultMri.icv) : "1450000"
  );

  // 6. Lifestyle Factors
  const defaultLife = lastVisit.lifestyle || session.demographics || {};
  const [alcohol, setAlcohol] = useState<number>(defaultLife.alcoholConsumption ?? defaultLife.alcohol ?? 3);
  const [sleepQuality, setSleepQuality] = useState<number>(defaultLife.sleepQuality ?? 6);
  const [physicalActivity, setPhysicalActivity] = useState<number>(defaultLife.physicalActivity ?? 3);
  const [dietQuality, setDietQuality] = useState<number>(defaultLife.dietQuality ?? 6);
  const [smoking, setSmoking] = useState<number>(defaultLife.smoking ?? 0);

  // 7. Medical History
  const defaultMed = lastVisit.medical || session.medical || {};
  const [hypertension, setHypertension] = useState<number>(defaultMed.hypertension ?? 0);
  const [diabetes, setDiabetes] = useState<number>(defaultMed.diabetes ?? 0);
  const [depression, setDepression] = useState<number>(defaultMed.depression ?? 0);
  const [headInjury, setHeadInjury] = useState<number>(defaultMed.headInjury ?? 0);
  const [cardiovascularDisease, setCardiovascularDisease] = useState<number>(defaultMed.cardiovascularDisease ?? 0);

  // 8. Uploads
  const [mriFile, setMriFile] = useState<File | null>(null);
  const [mriPreview, setMriPreview] = useState<string | null>(null);
  const [reportFile, setReportFile] = useState<File | null>(null);

  // Form submission state
  const [submitting, setSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  // Accordion section toggles
  const [openSections, setOpenSections] = useState({
    cognitive: true,
    symptoms: true,
    biomarkers: false,
    mri: false,
    lifestyle: false,
    uploads: false,
  });

  const toggleSection = (key: keyof typeof openSections) =>
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }));

  // Quick button handler
  const handleQuickMonth = (offset: number) => {
    mark("month");
    mark("exam_date");
    setMonth(offset);
    const dateVal = computeInitialDate(offset);
    setExamDate(dateVal);
  };

  const handleMRIChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setMriFile(file);
      mark("mri_file");
      const reader = new FileReader();
      reader.onload = (ev) => {
        setMriPreview(ev.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleReportChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setReportFile(e.target.files[0]);
      mark("report_file");
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setErrorMessage(null);
    setSuccessMessage(null);

    try {
      const formData = new FormData();
      formData.append("month", String(month));
      formData.append("exam_date", examDate);
      formData.append("MMSE", String(mmse));
      formData.append("CDGLOBAL", String(cdglobal));
      formData.append("TOTSCORE", String(totscore));
      if (faq) formData.append("FAQ", String(faq));
      if (moca) formData.append("MOCA", String(moca));
      formData.append("functionalAssessment", String(functionalAssessment));
      formData.append("adl", String(adl));

      formData.append("symptoms", JSON.stringify(symptoms));

      formData.append(
        "biomarkers",
        JSON.stringify({
          ABETA: abeta ? Number(abeta) : undefined,
          TAU: tau ? Number(tau) : undefined,
          PTAU: ptau ? Number(ptau) : undefined,
          APVOLUME: apvolume ? Number(apvolume) : undefined,
        })
      );

      formData.append(
        "mri_volumes",
        JSON.stringify({
          hippocampus: hippocampus ? Number(hippocampus) : undefined,
          entorhinal: entorhinal ? Number(entorhinal) : undefined,
          ventricles: ventricles ? Number(ventricles) : undefined,
          whole_brain: wholeBrain ? Number(wholeBrain) : undefined,
          icv: icv ? Number(icv) : undefined,
        })
      );

      formData.append(
        "lifestyle",
        JSON.stringify({
          alcohol,
          sleepQuality,
          physicalActivity,
          dietQuality,
          smoking,
        })
      );

      formData.append(
        "medical",
        JSON.stringify({
          hypertension,
          diabetes,
          depression,
          headInjury,
          cardiovascularDisease,
        })
      );

      if (mriFile) {
        formData.append("mri_file", mriFile);
      }
      if (reportFile) {
        formData.append("report_file", reportFile);
      }

      const updatedTwin = await twinApi.addFullVisit(twin.twin_id, formData);
      setSuccessMessage(`New follow-up visit (Month ${month}) recorded and twin recalibrated!`);
      onVisitAdded(updatedTwin);
    } catch (err: any) {
      console.error("Add full visit failed:", err);
      setErrorMessage(err?.message || "Failed to submit follow-up visit");
    } finally {
      setSubmitting(false);
    }
  };

  const renderBadge = (fieldKey: string) => {
    const isTouched = !!touched[fieldKey];
    if (isTouched) {
      return (
        <span className="px-1.5 py-0.5 rounded text-[9px] font-bold bg-purple-500/20 text-purple-300 border border-purple-500/30">
          Modified
        </span>
      );
    }
    return (
      <span className="px-1.5 py-0.5 rounded text-[9px] font-medium bg-blue-500/10 text-blue-300 border border-blue-500/20">
        Carried forward
      </span>
    );
  };

  return (
    <div className="rounded-3xl bg-neutral-900/60 backdrop-blur-xl border border-white/10 p-6 shadow-2xl relative overflow-hidden">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-6 pb-4 border-b border-white/10">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-2xl bg-gradient-to-br from-blue-600 to-indigo-700 text-white shadow-lg shadow-blue-600/30">
            <CalendarPlus className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <span>New Follow-Up Visit Re-Entry</span>
              <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-300 border border-blue-500/30">
                Full Intake Re-Evaluation
              </span>
            </h3>
            <p className="text-xs text-neutral-400 mt-0.5">
              Comprehensive re-entry prefilled from Visit #{twin.visits.length - 1} (Month {lastVisit.month ?? 0}).
            </p>
          </div>
        </div>

        {/* Quick Horizon Buttons */}
        <div className="flex items-center gap-1.5 self-end sm:self-auto">
          <span className="text-[11px] text-neutral-500 font-semibold mr-1">Quick:</span>
          {[6, 12, 24].map((mo) => (
            <button
              key={`quick-mo-${mo}`}
              type="button"
              onClick={() => handleQuickMonth(mo)}
              className={`px-2.5 py-1 rounded-lg text-xs font-bold transition cursor-pointer border ${
                month === mo
                  ? "bg-blue-600 text-white border-blue-400 shadow-md shadow-blue-600/30"
                  : "bg-white/5 text-neutral-300 hover:bg-white/10 border-white/10"
              }`}
            >
              +{mo}m
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Visit Timing Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 p-4 rounded-2xl bg-white/[0.02] border border-white/5">
          <div>
            <label className="text-xs font-semibold text-neutral-300 mb-1 flex items-center justify-between">
              <span>Exam Date (No Future Dates)</span>
              {renderBadge("exam_date")}
            </label>
            <input
              type="date"
              max={todayStr}
              value={examDate}
              onChange={(e) => {
                mark("exam_date");
                setExamDate(e.target.value);
              }}
              required
              className="w-full bg-neutral-900 border border-white/10 rounded-xl px-3 py-2 text-sm text-white font-mono focus:border-blue-500 focus:outline-none"
            />
          </div>

          <div>
            <label className="text-xs font-semibold text-neutral-300 mb-1 flex items-center justify-between">
              <span>Visit Month (from Baseline)</span>
              {renderBadge("month")}
            </label>
            <input
              type="number"
              min={1}
              max={60}
              value={month}
              onChange={(e) => {
                mark("month");
                setMonth(Number(e.target.value));
              }}
              required
              className="w-full bg-neutral-900 border border-white/10 rounded-xl px-3 py-2 text-sm text-white font-mono focus:border-blue-500 focus:outline-none"
            />
          </div>
        </div>

        {/* Section 1: Cognitive Scores */}
        <div className="rounded-2xl bg-white/[0.02] border border-white/5 overflow-hidden">
          <button
            type="button"
            onClick={() => toggleSection("cognitive")}
            className="w-full p-4 flex items-center justify-between text-left hover:bg-white/[0.02] transition cursor-pointer"
          >
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-blue-400" />
              <span className="text-xs font-bold text-white uppercase tracking-wider">
                1. Cognitive &amp; Functional Scores
              </span>
            </div>
            {openSections.cognitive ? <ChevronUp className="w-4 h-4 text-neutral-400" /> : <ChevronDown className="w-4 h-4 text-neutral-400" />}
          </button>

          {openSections.cognitive && (
            <div className="p-4 pt-0 border-t border-white/5 grid grid-cols-2 sm:grid-cols-4 gap-3 mt-3">
              <div>
                <div className="flex items-center justify-between text-[11px] mb-1">
                  <span className="text-neutral-400">MMSE (0–30)</span>
                  {renderBadge("mmse")}
                </div>
                <input
                  type="number"
                  min={0}
                  max={30}
                  step={0.5}
                  value={mmse}
                  onChange={(e) => {
                    mark("mmse");
                    setMmse(Number(e.target.value));
                  }}
                  className="w-full bg-neutral-900 border border-white/10 rounded-xl px-2.5 py-1.5 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
                />
              </div>

              <div>
                <div className="flex items-center justify-between text-[11px] mb-1">
                  <span className="text-neutral-400">CDR-Global</span>
                  {renderBadge("cdglobal")}
                </div>
                <select
                  value={cdglobal}
                  onChange={(e) => {
                    mark("cdglobal");
                    setCdglobal(Number(e.target.value));
                  }}
                  className="w-full bg-neutral-900 border border-white/10 rounded-xl px-2 py-1.5 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
                >
                  <option value={0}>0.0 (CN)</option>
                  <option value={0.5}>0.5 (MCI)</option>
                  <option value={1}>1.0 (Mild AD)</option>
                  <option value={2}>2.0 (Moderate)</option>
                  <option value={3}>3.0 (Severe)</option>
                </select>
              </div>

              <div>
                <div className="flex items-center justify-between text-[11px] mb-1">
                  <span className="text-neutral-400">ADAS11 (0–70)</span>
                  {renderBadge("totscore")}
                </div>
                <input
                  type="number"
                  min={0}
                  max={70}
                  step={0.5}
                  value={totscore}
                  onChange={(e) => {
                    mark("totscore");
                    setTotscore(Number(e.target.value));
                  }}
                  className="w-full bg-neutral-900 border border-white/10 rounded-xl px-2.5 py-1.5 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
                />
              </div>

              <div>
                <div className="flex items-center justify-between text-[11px] mb-1">
                  <span className="text-neutral-400">FAQ (0–30)</span>
                  {renderBadge("faq")}
                </div>
                <input
                  type="number"
                  min={0}
                  max={30}
                  value={faq}
                  onChange={(e) => {
                    mark("faq");
                    setFaq(e.target.value);
                  }}
                  className="w-full bg-neutral-900 border border-white/10 rounded-xl px-2.5 py-1.5 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
                />
              </div>

              <div>
                <div className="flex items-center justify-between text-[11px] mb-1">
                  <span className="text-neutral-400">MoCA (0–30)</span>
                  {renderBadge("moca")}
                </div>
                <input
                  type="number"
                  min={0}
                  max={30}
                  value={moca}
                  onChange={(e) => {
                    mark("moca");
                    setMoca(e.target.value);
                  }}
                  className="w-full bg-neutral-900 border border-white/10 rounded-xl px-2.5 py-1.5 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
                />
              </div>

              <div>
                <div className="flex items-center justify-between text-[11px] mb-1">
                  <span className="text-neutral-400">Functional (0–10)</span>
                  {renderBadge("functional")}
                </div>
                <input
                  type="number"
                  min={0}
                  max={10}
                  value={functionalAssessment}
                  onChange={(e) => {
                    mark("functional");
                    setFunctionalAssessment(Number(e.target.value));
                  }}
                  className="w-full bg-neutral-900 border border-white/10 rounded-xl px-2.5 py-1.5 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
                />
              </div>

              <div>
                <div className="flex items-center justify-between text-[11px] mb-1">
                  <span className="text-neutral-400">ADL (0–10)</span>
                  {renderBadge("adl")}
                </div>
                <input
                  type="number"
                  min={0}
                  max={10}
                  value={adl}
                  onChange={(e) => {
                    mark("adl");
                    setAdl(Number(e.target.value));
                  }}
                  className="w-full bg-neutral-900 border border-white/10 rounded-xl px-2.5 py-1.5 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
                />
              </div>
            </div>
          )}
        </div>

        {/* Section 2: 7 Symptoms */}
        <div className="rounded-2xl bg-white/[0.02] border border-white/5 overflow-hidden">
          <button
            type="button"
            onClick={() => toggleSection("symptoms")}
            className="w-full p-4 flex items-center justify-between text-left hover:bg-white/[0.02] transition cursor-pointer"
          >
            <div className="flex items-center gap-2">
              <Brain className="w-4 h-4 text-purple-400" />
              <span className="text-xs font-bold text-white uppercase tracking-wider">
                2. Cognitive Symptoms Screen (7 Domains)
              </span>
            </div>
            {openSections.symptoms ? <ChevronUp className="w-4 h-4 text-neutral-400" /> : <ChevronDown className="w-4 h-4 text-neutral-400" />}
          </button>

          {openSections.symptoms && (
            <div className="p-4 pt-0 border-t border-white/5 grid grid-cols-2 sm:grid-cols-4 gap-2.5 mt-3">
              {[
                { key: "memoryComplaints", label: "Memory Complaints" },
                { key: "forgetfulness", label: "Forgetfulness" },
                { key: "confusion", label: "Confusion" },
                { key: "disorientation", label: "Disorientation" },
                { key: "personalityChanges", label: "Personality Changes" },
                { key: "difficultyCompletingTasks", label: "Task Difficulties" },
                { key: "behavioralProblems", label: "Behavioral Problems" },
              ].map((sym) => {
                const isChecked = !!symptoms[sym.key];
                return (
                  <button
                    key={`sym-${sym.key}`}
                    type="button"
                    onClick={() => {
                      mark("symptoms");
                      setSymptoms((prev) => ({ ...prev, [sym.key]: !prev[sym.key] }));
                    }}
                    className={`py-2 px-3 rounded-xl text-xs font-medium border flex items-center justify-between transition cursor-pointer ${
                      isChecked
                        ? "bg-purple-950/40 border-purple-500/40 text-purple-200"
                        : "bg-white/[0.02] border-white/5 text-neutral-400 hover:text-white"
                    }`}
                  >
                    <span>{sym.label}</span>
                    <span
                      className={`w-3.5 h-3.5 rounded flex items-center justify-center text-[10px] ${
                        isChecked ? "bg-purple-500 text-white" : "border border-neutral-600"
                      }`}
                    >
                      {isChecked && <Check className="w-2.5 h-2.5" />}
                    </span>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Section 3: Biomarkers & MRI Volumes (Collapsible) */}
        <div className="rounded-2xl bg-white/[0.02] border border-white/5 overflow-hidden">
          <button
            type="button"
            onClick={() => toggleSection("biomarkers")}
            className="w-full p-4 flex items-center justify-between text-left hover:bg-white/[0.02] transition cursor-pointer"
          >
            <div className="flex items-center gap-2">
              <Dna className="w-4 h-4 text-emerald-400" />
              <span className="text-xs font-bold text-white uppercase tracking-wider">
                3. Biomarkers &amp; Volumetric MRI
              </span>
            </div>
            {openSections.biomarkers ? <ChevronUp className="w-4 h-4 text-neutral-400" /> : <ChevronDown className="w-4 h-4 text-neutral-400" />}
          </button>

          {openSections.biomarkers && (
            <div className="p-4 pt-0 border-t border-white/5 space-y-4 mt-3">
              <div>
                <span className="text-[11px] font-bold text-neutral-400 block mb-2">CSF Biomarkers</span>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div>
                    <label className="text-[10px] text-neutral-400 block mb-1">Aβ42 (pg/mL)</label>
                    <input
                      type="number"
                      value={abeta}
                      onChange={(e) => { mark("abeta"); setAbeta(e.target.value); }}
                      className="w-full bg-neutral-900 border border-white/10 rounded-xl px-2.5 py-1.5 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="text-[10px] text-neutral-400 block mb-1">Total Tau (pg/mL)</label>
                    <input
                      type="number"
                      value={tau}
                      onChange={(e) => { mark("tau"); setTau(e.target.value); }}
                      className="w-full bg-neutral-900 border border-white/10 rounded-xl px-2.5 py-1.5 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="text-[10px] text-neutral-400 block mb-1">p-Tau181 (pg/mL)</label>
                    <input
                      type="number"
                      value={ptau}
                      onChange={(e) => { mark("ptau"); setPtau(e.target.value); }}
                      className="w-full bg-neutral-900 border border-white/10 rounded-xl px-2.5 py-1.5 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="text-[10px] text-neutral-400 block mb-1">APVOLUME (mm³)</label>
                    <input
                      type="number"
                      value={apvolume}
                      onChange={(e) => { mark("apvolume"); setApvolume(e.target.value); }}
                      className="w-full bg-neutral-900 border border-white/10 rounded-xl px-2.5 py-1.5 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                </div>
              </div>

              <div>
                <span className="text-[11px] font-bold text-neutral-400 block mb-2">FreeSurfer Volumetric MRI</span>
                <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
                  <div>
                    <label className="text-[10px] text-neutral-400 block mb-1">Hippocampus</label>
                    <input
                      type="number"
                      value={hippocampus}
                      onChange={(e) => { mark("hippocampus"); setHippocampus(e.target.value); }}
                      className="w-full bg-neutral-900 border border-white/10 rounded-xl px-2.5 py-1.5 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="text-[10px] text-neutral-400 block mb-1">Entorhinal</label>
                    <input
                      type="number"
                      value={entorhinal}
                      onChange={(e) => { mark("entorhinal"); setEntorhinal(e.target.value); }}
                      className="w-full bg-neutral-900 border border-white/10 rounded-xl px-2.5 py-1.5 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="text-[10px] text-neutral-400 block mb-1">Ventricles</label>
                    <input
                      type="number"
                      value={ventricles}
                      onChange={(e) => { mark("ventricles"); setVentricles(e.target.value); }}
                      className="w-full bg-neutral-900 border border-white/10 rounded-xl px-2.5 py-1.5 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="text-[10px] text-neutral-400 block mb-1">Whole Brain</label>
                    <input
                      type="number"
                      value={wholeBrain}
                      onChange={(e) => { mark("wholeBrain"); setWholeBrain(e.target.value); }}
                      className="w-full bg-neutral-900 border border-white/10 rounded-xl px-2.5 py-1.5 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="text-[10px] text-neutral-400 block mb-1">ICV</label>
                    <input
                      type="number"
                      value={icv}
                      onChange={(e) => { mark("icv"); setIcv(e.target.value); }}
                      className="w-full bg-neutral-900 border border-white/10 rounded-xl px-2.5 py-1.5 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Section 4: Lifestyle & Medical Background */}
        <div className="rounded-2xl bg-white/[0.02] border border-white/5 overflow-hidden">
          <button
            type="button"
            onClick={() => toggleSection("lifestyle")}
            className="w-full p-4 flex items-center justify-between text-left hover:bg-white/[0.02] transition cursor-pointer"
          >
            <div className="flex items-center gap-2">
              <Heart className="w-4 h-4 text-rose-400" />
              <span className="text-xs font-bold text-white uppercase tracking-wider">
                4. Lifestyle &amp; Medical History
              </span>
            </div>
            {openSections.lifestyle ? <ChevronUp className="w-4 h-4 text-neutral-400" /> : <ChevronDown className="w-4 h-4 text-neutral-400" />}
          </button>

          {openSections.lifestyle && (
            <div className="p-4 pt-0 border-t border-white/5 space-y-4 mt-3">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div>
                  <label className="text-[10px] text-neutral-400 block mb-1">Alcohol (units/wk)</label>
                  <input
                    type="number"
                    value={alcohol}
                    onChange={(e) => { mark("alcohol"); setAlcohol(Number(e.target.value)); }}
                    className="w-full bg-neutral-900 border border-white/10 rounded-xl px-2.5 py-1.5 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-neutral-400 block mb-1">Sleep Quality (0–10)</label>
                  <input
                    type="number"
                    min={0}
                    max={10}
                    value={sleepQuality}
                    onChange={(e) => { mark("sleepQuality"); setSleepQuality(Number(e.target.value)); }}
                    className="w-full bg-neutral-900 border border-white/10 rounded-xl px-2.5 py-1.5 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-neutral-400 block mb-1">Activity (hrs/wk)</label>
                  <input
                    type="number"
                    min={0}
                    value={physicalActivity}
                    onChange={(e) => { mark("physicalActivity"); setPhysicalActivity(Number(e.target.value)); }}
                    className="w-full bg-neutral-900 border border-white/10 rounded-xl px-2.5 py-1.5 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-neutral-400 block mb-1">Diet Quality (0–10)</label>
                  <input
                    type="number"
                    min={0}
                    max={10}
                    value={dietQuality}
                    onChange={(e) => { mark("dietQuality"); setDietQuality(Number(e.target.value)); }}
                    className="w-full bg-neutral-900 border border-white/10 rounded-xl px-2.5 py-1.5 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
                  />
                </div>
              </div>

              <div>
                <span className="text-[11px] font-bold text-neutral-400 block mb-2">Clinical Conditions</span>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                  {[
                    { label: "Hypertension", val: hypertension, setVal: setHypertension, k: "hyp" },
                    { label: "Diabetes", val: diabetes, setVal: setDiabetes, k: "diab" },
                    { label: "Depression", val: depression, setVal: setDepression, k: "dep" },
                    { label: "Head Injury", val: headInjury, setVal: setHeadInjury, k: "hi" },
                    { label: "CVD", val: cardiovascularDisease, setVal: setCardiovascularDisease, k: "cvd" },
                    { label: "Smoker", val: smoking, setVal: setSmoking, k: "smk" },
                  ].map((cond) => (
                    <button
                      key={`cond-${cond.k}`}
                      type="button"
                      onClick={() => {
                        mark(cond.k);
                        cond.setVal(cond.val === 1 ? 0 : 1);
                      }}
                      className={`py-1.5 px-3 rounded-lg text-xs font-semibold border flex items-center justify-between transition cursor-pointer ${
                        cond.val === 1
                          ? "bg-rose-950/40 border-rose-500/40 text-rose-300"
                          : "bg-white/[0.02] border-white/5 text-neutral-400 hover:text-white"
                      }`}
                    >
                      <span>{cond.label}</span>
                      <span className={`w-2 h-2 rounded-full ${cond.val === 1 ? "bg-rose-400" : "bg-neutral-600"}`} />
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Section 5: MRI & Blood Report Uploads with Preview */}
        <div className="rounded-2xl bg-white/[0.02] border border-white/5 overflow-hidden">
          <button
            type="button"
            onClick={() => toggleSection("uploads")}
            className="w-full p-4 flex items-center justify-between text-left hover:bg-white/[0.02] transition cursor-pointer"
          >
            <div className="flex items-center gap-2">
              <Upload className="w-4 h-4 text-blue-400" />
              <span className="text-xs font-bold text-white uppercase tracking-wider">
                5. New Visit Uploads (MRI Slice &amp; Blood Report)
              </span>
            </div>
            {openSections.uploads ? <ChevronUp className="w-4 h-4 text-neutral-400" /> : <ChevronDown className="w-4 h-4 text-neutral-400" />}
          </button>

          {openSections.uploads && (
            <div className="p-4 pt-0 border-t border-white/5 grid grid-cols-1 sm:grid-cols-2 gap-4 mt-3">
              {/* MRI Upload with Preview */}
              <div className="space-y-2">
                <label className="text-xs font-semibold text-neutral-300 block">
                  Structural Brain MRI Slice (.jpg, .png)
                </label>
                {mriPreview ? (
                  <div className="relative rounded-2xl overflow-hidden border border-blue-500/40 bg-neutral-900 p-2 flex items-center gap-3">
                    <img
                      src={mriPreview}
                      alt="MRI Slice Preview"
                      className="w-16 h-16 object-cover rounded-xl border border-white/10"
                    />
                    <div className="overflow-hidden flex-1">
                      <p className="text-xs font-mono text-white truncate">{mriFile?.name}</p>
                      <p className="text-[10px] text-blue-300">Ready for ResNet-34 late fusion</p>
                    </div>
                    <button
                      type="button"
                      onClick={() => {
                        setMriFile(null);
                        setMriPreview(null);
                      }}
                      className="p-1 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-neutral-400 hover:text-white"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                ) : (
                  <label className="border-2 border-dashed border-white/10 hover:border-blue-500/50 rounded-2xl p-4 flex flex-col items-center justify-center gap-2 cursor-pointer transition hover:bg-white/[0.02]">
                    <ImageIcon className="w-6 h-6 text-neutral-500" />
                    <span className="text-xs text-neutral-400 font-semibold">Upload visit MRI slice</span>
                    <span className="text-[10px] text-neutral-600">Single 2D axial slice</span>
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleMRIChange}
                      className="hidden"
                    />
                  </label>
                )}
              </div>

              {/* Blood Report Upload */}
              <div className="space-y-2">
                <label className="text-xs font-semibold text-neutral-300 block">
                  Laboratory Blood Report (.pdf, .jpg)
                </label>
                {reportFile ? (
                  <div className="relative rounded-2xl border border-purple-500/40 bg-neutral-900 p-3 flex items-center justify-between">
                    <div className="flex items-center gap-2.5 overflow-hidden">
                      <FileText className="w-5 h-5 text-purple-400 shrink-0" />
                      <span className="text-xs font-mono text-white truncate">{reportFile.name}</span>
                    </div>
                    <button
                      type="button"
                      onClick={() => setReportFile(null)}
                      className="p-1 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-neutral-400 hover:text-white"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                ) : (
                  <label className="border-2 border-dashed border-white/10 hover:border-purple-500/50 rounded-2xl p-4 flex flex-col items-center justify-center gap-2 cursor-pointer transition hover:bg-white/[0.02]">
                    <FileText className="w-6 h-6 text-neutral-500" />
                    <span className="text-xs text-neutral-400 font-semibold">Upload blood panel report</span>
                    <span className="text-[10px] text-neutral-600">OCR biomarker extraction</span>
                    <input
                      type="file"
                      accept=".pdf,image/*"
                      onChange={handleReportChange}
                      className="hidden"
                    />
                  </label>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Status Alerts */}
        <AnimatePresence>
          {errorMessage && (
            <motion.div
              initial={{ opacity: 0, y: -5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="p-3.5 rounded-2xl bg-rose-950/40 border border-rose-500/30 text-rose-300 text-xs flex items-center gap-2"
            >
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{errorMessage}</span>
            </motion.div>
          )}

          {successMessage && (
            <motion.div
              initial={{ opacity: 0, y: -5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="p-3.5 rounded-2xl bg-emerald-950/40 border border-emerald-500/30 text-emerald-300 text-xs flex items-center gap-2"
            >
              <Check className="w-4 h-4 shrink-0" />
              <span>{successMessage}</span>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={submitting}
          className="w-full py-3 px-5 rounded-2xl bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white font-bold text-xs shadow-xl shadow-blue-600/30 transition flex items-center justify-center gap-2 disabled:opacity-50 cursor-pointer"
        >
          {submitting ? (
            <>
              <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
              <span>Uploading &amp; Calibrating Follow-Up Visit...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4 text-purple-300" />
              <span>Commit Follow-Up Visit to Digital Twin</span>
            </>
          )}
        </button>
      </form>
    </div>
  );
}
