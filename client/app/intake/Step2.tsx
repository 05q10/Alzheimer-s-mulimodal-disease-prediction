"use client";

import { useRef, useState } from "react";
import { motion } from "framer-motion";
import { usePatientStore } from "../store/patientStore";

interface Props {
  next: () => void;
  back: () => void;
}

export default function Step2({ next, back }: Props) {
  const { session, updateUploads, updateMedical } = usePatientStore();

  const mriInputRef = useRef<HTMLInputElement>(null);
  const reportInputRef = useRef<HTMLInputElement>(null);

  /* ---------------------- FILE UPLOAD HANDLERS ---------------------- */

  const handleMRIUpload = (file: File) => {
    updateUploads({ mriFile: file });
  };

  const handleReportUpload = (file: File) => {
    updateUploads({ reportFile: file });
  };

  /* ---------------------- FILE DROPZONE ---------------------- */

  const FileDropZone = ({
    label,
    icon,
    file,
    onClick,
    onDrop,
  }: {
    label: string;
    icon: string;
    file: File | null;
    onClick: () => void;
    onDrop: (f: File) => void;
  }) => (
    <motion.div
      whileHover={{ scale: 1.02 }}
      className="p-8 border-2 border-dashed rounded-3xl text-center cursor-pointer bg-white/5 border-white/20 hover:border-blue-400/40 hover:bg-white/10 transition-all"
      onClick={onClick}
      onDragOver={(e: React.DragEvent) => e.preventDefault()}
      onDrop={(e: React.DragEvent) => {
        e.preventDefault();
        if (e.dataTransfer.files[0]) onDrop(e.dataTransfer.files[0]);
      }}
    >
      {file ? (
        <div className="flex flex-col items-center">
          {file.type.startsWith("image/") || file.name.match(/\.(jpg|jpeg|png|webp)$/i) ? (
            /* eslint-disable-next-line @next/next/no-img-element */
            <img
              src={URL.createObjectURL(file)}
              alt="Uploaded preview"
              className="w-32 h-32 object-cover rounded-xl border border-white/20 mb-3 shadow-lg"
            />
          ) : (
            <div className="text-5xl mb-3">{icon}</div>
          )}
          <p className="text-blue-200 text-sm font-semibold">{file.name}</p>
          <p className="text-[11px] text-emerald-400 mt-1">✓ File Attached ({(file.size / 1024).toFixed(1)} KB)</p>
        </div>
      ) : (
        <div>
          <div className="text-5xl mb-3">{icon}</div>
          <p className="text-blue-200">{label}</p>
          <p className="text-xs text-blue-300/50 mt-1">Click or drag a file here</p>
        </div>
      )}
    </motion.div>
  );

  /* ---------------------- YES/NO MEDICAL TOGGLE ---------------------- */

  const MedicalToggle = ({
    label,
    keyName,
    value,
  }: {
    label: string;
    keyName: keyof typeof session.medical;
    value: number | null;
  }) => (
    <div className="flex items-center justify-between p-4 bg-white/5 rounded-2xl border border-white/10 hover:border-blue-400/20 transition">
      <span className="text-blue-100">{label}</span>

      <div className="flex gap-3">
        {[0, 1].map((v) => (
          <button
            key={v}
            onClick={() => updateMedical({ [keyName]: v })}
            className={`px-5 py-2 rounded-xl border transition ${
              value === v
                ? "border-cyan-400 bg-cyan-500/20 text-cyan-300"
                : "border-white/20 bg-white/5 text-blue-200 hover:bg-white/10"
            }`}
          >
            {v === 1 ? "Yes" : "No"}
          </button>
        ))}
      </div>
    </div>
  );

  /* ---------------------- FIXED NUMBER INPUT (NO CURSOR JUMP) ---------------------- */

  const NumberInput = ({
  label,
  keyName,
  value,
  min,
  max,
}: {
  label: string;
  keyName: keyof typeof session.medical;
  value: number | null;
  min?: number;
  max?: number;
}) => {
  const [local, setLocal] = useState<string>(value?.toString() ?? "");

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setLocal(val); // controls UI 100% smoothly — FIXES CURSOR

    if (val === "") {
      updateMedical({ [keyName]: null });
    } else {
      updateMedical({ [keyName]: Number(val) });
    }
  };

  return (
    <div className="space-y-2">
      <label className="text-blue-100">{label}</label>
      <input
        type="number"
        value={local}
        min={min}
        max={max}
        onChange={handleChange}
        className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white"
      />
    </div>
  );
};


  /* ---------------------- MEDICAL HISTORY ITEMS ---------------------- */

  const medicalItems = [
    { key: "cardiovascularDisease", label: "Cardiovascular Disease" },
    { key: "diabetes", label: "Diabetes" },
    { key: "depression", label: "Depression" },
    { key: "headInjury", label: "History of Head Injury" },
    { key: "hypertension", label: "Hypertension" },
  ] as const;

  /* ---------------------- UI ---------------------- */

  return (
    <div className="min-h-screen bg-transparent text-white p-4 md:p-8 rounded-xl">
      <div className="max-w-6xl mx-auto space-y-12">

        {/* HEADER */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center"
        >
          <div className="inline-block mb-4">
            <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-blue-500 rounded-2xl flex items-center justify-center text-3xl shadow-lg shadow-purple-500/50">
              🧬
            </div>
          </div>

          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
            Medical History & Uploads
          </h1>

          <p className="text-blue-200 mt-2">
            Add your clinical background and upload your MRI/report.
          </p>
        </motion.div>

        {/* GRID */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-10">

          {/* LEFT — UPLOADS */}
          <div>
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">🧠 MRI Scan</h2>

            <FileDropZone
              label="Upload MRI Image"
              icon="🖼️"
              file={session.uploads.mriFile}
              onClick={() => mriInputRef.current?.click()}
              onDrop={handleMRIUpload}
            />

            <input
              type="file"
              accept="image/*"
              className="hidden"
              ref={mriInputRef}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                e.target.files && handleMRIUpload(e.target.files[0])
              }
            />

            <h2 className="text-xl font-semibold mt-10 mb-4 flex items-center gap-2">📄 Blood Report</h2>

            <FileDropZone
              label="Upload Medical Report"
              icon="📄"
              file={session.uploads.reportFile}
              onClick={() => reportInputRef.current?.click()}
              onDrop={handleReportUpload}
            />

            <input
              type="file"
              accept="image/*,application/pdf"
              className="hidden"
              ref={reportInputRef}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                e.target.files && handleReportUpload(e.target.files[0])
              }
            />
          </div>

          {/* RIGHT — MEDICAL HISTORY & CLINICAL DATA */}
          <div className="space-y-8">

            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">❤️ Medical Background</h2>

            <div className="space-y-4">
              {medicalItems.map((item) => (
                <MedicalToggle
                  key={item.key}
                  keyName={item.key}
                  label={item.label}
                  value={session.medical[item.key]}
                />
              ))}
            </div>

            {/* CLINICAL MEASUREMENTS */}
            <h2 className="text-xl font-semibold mt-8 mb-4 flex items-center gap-2">📈 Clinical Measurements</h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <NumberInput
                label="Systolic BP (90–180)"
                keyName="systolicBP"
                value={session.medical.systolicBP}
                min={90}
                max={180}
              />

              <NumberInput
                label="Diastolic BP (60–120)"
                keyName="diastolicBP"
                value={session.medical.diastolicBP}
                min={60}
                max={120}
              />

              <NumberInput
                label="Total Cholesterol (150–300)"
                keyName="cholesterolTotal"
                value={session.medical.cholesterolTotal}
                min={150}
                max={300}
              />

              <NumberInput
                label="LDL (50–200)"
                keyName="cholesterolLDL"
                value={session.medical.cholesterolLDL}
                min={50}
                max={200}
              />

              <NumberInput
                label="HDL (20–100)"
                keyName="cholesterolHDL"
                value={session.medical.cholesterolHDL}
                min={20}
                max={100}
              />

              <NumberInput
                label="Triglycerides (50–400)"
                keyName="cholesterolTriglycerides"
                value={session.medical.cholesterolTriglycerides}
                min={50}
                max={400}
              />
            </div>
          </div>
        </div>

        {/* BUTTONS */}
        <div className="flex justify-between mt-10">
          <motion.button
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.95 }}
            onClick={back}
            className="px-8 py-4 bg-white/10 border border-white/20 hover:bg-white/20 rounded-xl"
          >
            ← Back
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.95 }}
            onClick={next}
            className="px-8 py-4 bg-gradient-to-r from-blue-600 to-cyan-600 rounded-xl text-lg font-semibold shadow-xl"
          >
            Continue →
          </motion.button>
        </div>
      </div>
    </div>
  );
}
