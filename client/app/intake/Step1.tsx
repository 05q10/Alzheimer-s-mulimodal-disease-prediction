"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { usePatientStore, Demographics } from "@/app/store/patientStore";
import { twinApi } from "@/app/lib/api";
import { Sparkles, AlertTriangle } from "lucide-react";

interface Props {
  next: () => void;
}

export default function Step1({ next }: Props) {
  const {
    session,
    updateDemographics,
    setTwin,
    setTwinId,
    autofillSamplePatient,
  } = usePatientStore();
  const d = session.demographics;

  const [isCreatingTwin, setIsCreatingTwin] = useState(false);
  const [hoveredSlider, setHoveredSlider] = useState<string | null>(null);

  /* ------------------------------- */
  /* EDUCATION OPTIONS */
  /* ------------------------------- */
  const educationLevels = [
    { id: 0, label: "No Formal Education" },
    { id: 1, label: "Schooling (10th/12th)" },
    { id: 2, label: "Undergraduate" },
    { id: 3, label: "Postgraduate / Higher" },
  ];

  /* ------------------------------- */
  /* Lifestyle sliders */
  /* ------------------------------- */
  const lifestyleMetrics: {
    id: keyof Demographics;
    title: string;
    icon: string;
    labels: string[];
    min: number;
    max: number;
    description: string;
    gradient: string;
  }[] = [
    {
      id: "alcoholConsumption",
      title: "Alcohol Consumption",
      icon: "🍷",
      labels: ["0", "5", "10", "15", "20"],
      min: 0,
      max: 20,
      description: "Weekly alcohol consumption (units)",
      gradient: "from-amber-500 to-orange-600",
    },
    {
      id: "physicalActivity",
      title: "Physical Activity",
      icon: "🏃",
      labels: ["0", "2", "4", "6", "8", "10"],
      min: 0,
      max: 10,
      description: "Weekly activity (hours)",
      gradient: "from-green-500 to-emerald-600",
    },
    {
      id: "dietQuality",
      title: "Diet Quality",
      icon: "🥗",
      labels: ["0", "2", "4", "6", "8", "10"],
      min: 0,
      max: 10,
      description: "Nutritional quality of diet",
      gradient: "from-lime-500 to-green-600",
    },
    {
      id: "sleepQuality",
      title: "Sleep Quality",
      icon: "😴",
      labels: ["4", "5", "6", "7", "8", "9", "10"],
      min: 4,
      max: 10,
      description: "Quality of your sleep",
      gradient: "from-indigo-500 to-purple-600",
    },
  ];

  function getLabel(metric: (typeof lifestyleMetrics)[0], value: number) {
    const idx = Math.floor(
      ((value - metric.min) / (metric.max - metric.min)) *
        (metric.labels.length - 1)
    );
    return metric.labels[Math.max(0, Math.min(idx, metric.labels.length - 1))];
  }

  return (
    <div className="min-h-screen bg-transparent text-white p-4 md:p-8 rounded-xl">
      <div className="max-w-7xl mx-auto">
        {/* HEADER */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <div className="inline-block mb-4">
            <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-2xl flex items-center justify-center text-3xl shadow-lg shadow-blue-500/50">
              🧠
            </div>
          </div>

          <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent mb-3">
            Patient Health Profile
          </h1>

          <p className="text-blue-200 text-lg mb-4">
            Provide key details to personalize clinical predictions
          </p>

          {/* Presenter Aid: Autofill Sample Patient */}
          <button
            type="button"
            onClick={autofillSamplePatient}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white text-xs font-bold shadow-lg shadow-purple-600/30 transition hover:scale-105 cursor-pointer"
          >
            <Sparkles className="w-4 h-4 text-purple-200" />
            <span>Autofill Sample Patient (74y, F, MMSE 26, APOE e3/e4)</span>
          </button>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* LEFT FORM */}
          <div className="lg:col-span-2 space-y-8">
            {/* BASIC INFO */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-white/5 backdrop-blur-xl rounded-3xl p-6 md:p-8 border border-white/10 shadow-2xl"
            >
              <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
                <span className="text-2xl">👤</span> Basic Information
              </h2>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* NAME */}
                <div className="space-y-3">
                  <label className="text-lg font-medium text-blue-100">
                    Full Name
                  </label>
                  <input
                    type="text"
                    placeholder="Enter patient's name"
                    value={d.name ?? ""}
                    onChange={(e) =>
                      updateDemographics({ name: e.target.value })
                    }
                    className="w-full px-4 py-3 rounded-xl bg-white/10 border border-white/20 text-white placeholder-blue-200/40"
                  />
                </div>

                {/* AGE */}
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <label className="text-lg font-medium text-blue-100">
                      Age
                    </label>
                    <input
                      type="number"
                      min={18}
                      max={100}
                      value={d.age ?? 18}
                      onChange={(e) =>
                        updateDemographics({ age: Number(e.target.value) })
                      }
                      className="w-20 px-3 py-2 rounded-xl text-center bg-blue-500/20 border border-blue-400 text-blue-200 font-bold"
                    />
                  </div>

                  <input
                    type="range"
                    min="18"
                    max="100"
                    value={d.age ?? 18}
                    onChange={(e) =>
                      updateDemographics({ age: Number(e.target.value) })
                    }
                    className="w-full slider"
                  />

                  <div className="flex justify-between px-1 text-blue-300/50 text-xs">
                    {[18, 30, 40, 50, 60, 70, 80, 90, 100].map((v) => (
                      <span key={v}>{v}</span>
                    ))}
                  </div>

                  {d.age !== null && d.age !== undefined && (d.age < 55 || d.age > 90) && (
                    <div className="mt-2 text-xs text-amber-300 bg-amber-500/10 border border-amber-500/30 px-3 py-1.5 rounded-xl flex items-center gap-1.5">
                      <AlertTriangle className="w-3.5 h-3.5 shrink-0 text-amber-400" />
                      <span>Outside the model's training range (55–90), treat results as illustrative.</span>
                    </div>
                  )}
                </div>

                {/* GENDER */}
                <div className="space-y-3">
                  <label className="text-lg font-medium text-blue-100">
                    Gender
                  </label>

                  <div className="flex gap-3">
                    {[
                      { id: 0, label: "Male", icon: "👨" },
                      { id: 1, label: "Female", icon: "👩" },
                    ].map((g) => (
                      <motion.button
                        key={g.id}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={() =>
                          updateDemographics({ gender: g.id })
                        }
                        className={`flex-1 px-6 py-4 rounded-2xl border-2 transition-all
                          ${
                            d.gender === g.id
                              ? "bg-gradient-to-br from-blue-500 to-cyan-500 border-blue-400 shadow-lg"
                              : "bg-white/5 border-white/20 hover:border-blue-400/50 hover:bg-white/10"
                          }`}
                      >
                        <div className="text-3xl">{g.icon}</div>
                        <div>{g.label}</div>
                      </motion.button>
                    ))}
                  </div>
                </div>

                {/* EDUCATION */}
                <div className="space-y-3">
                  <label className="text-lg font-medium text-blue-100">
                    Education Level
                  </label>

                  <select
                    value={d.educationLevel ?? ""}
                    onChange={(e) =>
                      updateDemographics({
                        educationLevel: Number(e.target.value),
                      })
                    }
                    className="w-full px-4 py-3 rounded-xl bg-white/10 border border-white/20 text-white"
                  >
                    <option value="" disabled>
                      Select level
                    </option>
                    {educationLevels.map((e) => (
                      <option key={e.id} value={e.id} className="text-black">
                        {e.label}
                      </option>
                    ))}
                  </select>
                </div>

                {/* BASELINE ASSESSMENT DATE */}
                <div className="space-y-3">
                  <label className="text-lg font-medium text-blue-100 flex items-center justify-between">
                    <span>Baseline Assessment Date</span>
                    <span className="text-xs text-blue-300/70 font-normal">Anchor date for visits</span>
                  </label>
                  <input
                    type="date"
                    value={d.baselineDate ?? new Date().toISOString().split("T")[0]}
                    onChange={(e) =>
                      updateDemographics({ baselineDate: e.target.value })
                    }
                    className="w-full px-4 py-3 rounded-xl bg-white/10 border border-white/20 text-white font-mono"
                  />
                </div>

                {/* ETHNICITY */}
                <div className="space-y-3">
                  <label className="text-lg font-medium text-blue-100">
                    Ethnicity
                  </label>

                  <select
                    value={d.ethnicity ?? ""}
                    onChange={(e) =>
                      updateDemographics({
                        ethnicity: Number(e.target.value),
                      })
                    }
                    className="w-full px-4 py-3 rounded-xl bg-white/10 border border-white/20 text-white"
                  >
                    <option value="" disabled>
                      Select ethnicity
                    </option>
                    <option className="text-black" value={0}>
                      Caucasian
                    </option>
                    <option className="text-black" value={1}>
                      African American
                    </option>
                    <option className="text-black" value={2}>
                      Asian
                    </option>
                    <option className="text-black" value={3}>
                      Other
                    </option>
                  </select>
                </div>

                {/* DOCTOR-IN-CHARGE */}
                <div className="space-y-3">
                  <label className="text-lg font-medium text-blue-100">
                    Doctor In Charge
                  </label>

                  <select
                    value={d.doctorInCharge ?? ""}
                    onChange={(e) =>
                      updateDemographics({
                        doctorInCharge: e.target.value,
                      })
                    }
                    className="w-full px-4 py-3 rounded-xl bg-white/10 border border-white/20 text-white"
                  >
                    <option value="" disabled>
                      Select doctor
                    </option>
                    {["Dr. A", "Dr. B", "Dr. C", "Dr. D"].map((doc) => (
                      <option key={doc} value={doc} className="text-black">
                        {doc}
                      </option>
                    ))}
                  </select>
                </div>

                {/* HEIGHT */}
                <div className="space-y-3">
                  <label className="text-lg font-medium">Height (cm)</label>
                  <input
                    type="number"
                    min={80}
                    max={250}
                    value={d.height ?? ""}
                    onChange={(e) =>
                      updateDemographics({
                        height: Number(e.target.value),
                      })
                    }
                    className="w-full px-4 py-3 rounded-xl bg-white/10 border border-white/20 text-white"
                  />
                </div>

                {/* WEIGHT */}
                <div className="space-y-3">
                  <label className="text-lg font-medium">Weight (kg)</label>
                  <input
                    type="number"
                    min={20}
                    max={200}
                    value={d.weight ?? ""}
                    onChange={(e) =>
                      updateDemographics({
                        weight: Number(e.target.value),
                      })
                    }
                    className="w-full px-4 py-3 rounded-xl bg-white/10 border border-white/20 text-white"
                  />
                </div>

                {/* BMI */}
                <div className="space-y-3">
                  <label className="text-lg font-medium">BMI</label>
                  <div className="px-4 py-3 rounded-xl bg-blue-500/10 border border-blue-400/30 text-blue-300 font-bold">
                    {d.bmi ? `${d.bmi}` : "Auto-calculated"}
                  </div>
                </div>

                {/* SMOKING */}
                <div className="space-y-3">
                  <label className="text-lg font-medium">Smoking</label>

                  <div className="flex gap-3">
                    {[0, 1].map((value) => (
                      <button
                        key={value}
                        onClick={() =>
                          updateDemographics({ smoking: value })
                        }
                        className={`flex-1 py-3 rounded-xl border ${
                          d.smoking === value
                            ? "border-green-400 bg-green-600/30"
                            : "border-white/20"
                        }`}
                      >
                        {value === 1 ? "Yes" : "No"}
                      </button>
                    ))}
                  </div>
                </div>

                {/* FAMILY HISTORY */}
                <div className="space-y-3">
                  <label className="text-lg font-medium">
                    Family History of Alzheimer's
                  </label>

                  <div className="flex gap-3">
                    {[0, 1].map((value) => (
                      <button
                        key={value}
                        onClick={() =>
                          updateDemographics({
                            familyHistoryAlzheimers: value,
                          })
                        }
                        className={`flex-1 py-3 rounded-xl border ${
                          d.familyHistoryAlzheimers === value
                            ? "border-purple-400 bg-purple-600/30"
                            : "border-white/20"
                        }`}
                      >
                        {value === 1 ? "Yes" : "No"}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </motion.div>

            {/* LIFESTYLE */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-white/5 backdrop-blur-xl rounded-3xl p-6 md:p-8 border border-white/10 shadow-2xl"
            >
              <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3">
                📊 Lifestyle Assessment
              </h2>

              <div className="space-y-8">
                {lifestyleMetrics.map((metric) => (
                  <div key={metric.id} className="space-y-3">
                    {/* LABEL */}
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <span className="text-2xl">{metric.icon}</span>
                        <div>
                          <div className="font-semibold">
                            {metric.title}
                          </div>
                          <div className="text-sm text-blue-200/60">
                            {metric.description}
                          </div>
                        </div>
                      </div>

                      <div className="px-4 py-2 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-500 font-bold shadow-lg">
                        {getLabel(
                          metric,
                          Number(d[metric.id as keyof Demographics] ?? metric.min)
                        )}
                      </div>
                    </div>

                    {/* SLIDER */}
                    <input
                      type="range"
                      min={metric.min}
                      max={metric.max}
                      value={
                        Number(d[metric.id as keyof Demographics] ?? metric.min)
                      }
                      onChange={(e) =>
                        updateDemographics(
                          {
                            [metric.id]: Number(e.target.value),
                          } as Partial<Demographics>
                        )
                      }
                      className="w-full slider"
                    />

                    <div className="flex justify-between px-1 text-xs text-blue-300/50">
                      {metric.labels.map((l) => (
                        <span key={l}>{l}</span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>

            {/* NEXT BUTTON */}
            <button
              onClick={async () => {
                try {
                  setIsCreatingTwin(true);
                  const educationMap: Record<number, number> = { 0: 8, 1: 12, 2: 16, 3: 18 };
                  const eduYears =
                    d.educationLevel !== null && d.educationLevel !== undefined
                      ? educationMap[d.educationLevel] ?? 12
                      : 12;

                  const twin = await twinApi.createTwin({
                    age: d.age || 70,
                    sex: d.gender === 1 ? 1 : 0,
                    education_years: eduYears,
                    apoe4_count: d.familyHistoryAlzheimers === 1 ? 1 : 0,
                    MMSE: session.cognitive?.mmse ?? undefined,
                    is_synthetic: false,
                    persona_label: d.name ? `Patient ${d.name}` : "Clinical Intake",
                  });

                  setTwin(twin);
                  setTwinId(twin.twin_id);
                } catch (err) {
                  console.error("Twin creation failed, proceeding with intake:", err);
                } finally {
                  setIsCreatingTwin(false);
                  next();
                }
              }}
              disabled={isCreatingTwin}
              className="w-full py-5 bg-gradient-to-r from-blue-600 to-cyan-600 rounded-2xl text-lg font-bold shadow-xl hover:scale-[1.02] active:scale-[0.98] transition flex items-center justify-center gap-2 cursor-pointer disabled:opacity-70"
            >
              {isCreatingTwin ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Synthesizing Digital Twin...</span>
                </>
              ) : (
                <span>Continue →</span>
              )}
            </button>
          </div>

          {/* SUMMARY PANEL */}
          <div className="lg:sticky lg:top-8 h-fit">
            <div className="bg-gradient-to-br from-blue-500/10 to-cyan-500/10 backdrop-blur-xl rounded-3xl p-6 border border-blue-400/30 shadow-2xl">
              <h3 className="text-xl font-semibold mb-6 flex items-center gap-2">
                📋 Summary
              </h3>

              <div className="space-y-4 text-blue-200 text-sm">
                <Summary label="Name" value={d.name} />
                <Summary label="Age" value={d.age?.toString()} />
                <Summary
                  label="Gender"
                  value={
                    d.gender === 0
                      ? "Male"
                      : d.gender === 1
                      ? "Female"
                      : ""
                  }
                />
                <Summary
                  label="Education"
                  value={
                    educationLevels.find(
                      (x) => x.id === d.educationLevel
                    )?.label
                  }
                />
                <Summary
                  label="Height"
                  value={d.height ? `${d.height} cm` : ""}
                />
                <Summary
                  label="Weight"
                  value={d.weight ? `${d.weight} kg` : ""}
                />
                <Summary label="BMI" value={d.bmi?.toString()} />

                <div className="h-px bg-white/20 my-4" />

                <Summary
                  label="Smoking"
                  value={
                    d.smoking === 1
                      ? "Yes"
                      : d.smoking === 0
                      ? "No"
                      : ""
                  }
                />

                <Summary
                  label="Family History AD"
                  value={
                    d.familyHistoryAlzheimers === 1
                      ? "Yes"
                      : d.familyHistoryAlzheimers === 0
                      ? "No"
                      : ""
                  }
                />

                <Summary
                  label="Alcohol"
                  value={d.alcoholConsumption?.toString()}
                />
                <Summary
                  label="Physical Activity"
                  value={d.physicalActivity?.toString()}
                />
                <Summary
                  label="Diet Quality"
                  value={d.dietQuality?.toString()}
                />
                <Summary
                  label="Sleep Quality"
                  value={d.sleepQuality?.toString()}
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* SLIDER STYLE */}
      <style jsx>{`
        .slider {
          height: 6px;
          background: linear-gradient(to right, #3b82f6, #06b6d4);
          border-radius: 10px;
          appearance: none;
        }
        .slider::-webkit-slider-thumb {
          appearance: none;
          width: 22px;
          height: 22px;
          border-radius: 50%;
          background: white;
          border: 3px solid #3b82f6;
          cursor: pointer;
          transition: 0.2s;
        }
        .slider::-webkit-slider-thumb:hover {
          transform: scale(1.2);
        }
      `}</style>
    </div>
  );
}

function Summary({ label, value }: { label: string; value?: string | null }) {
  return (
    <div className="flex justify-between">
      <span>{label}</span>
      <span className="font-semibold text-white">{value ?? "—"}</span>
    </div>
  );
}
