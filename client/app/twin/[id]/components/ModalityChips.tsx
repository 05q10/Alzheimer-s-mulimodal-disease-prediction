"use client";

import React, { useState } from "react";
import { ModalityStatus, TwinFull, twinApi } from "@/app/lib/api";
import {
  FileText,
  Activity,
  Dna,
  Layers,
  Image as ImageIcon,
  Plus,
  CheckCircle2,
  XCircle,
  AlertCircle,
  X,
  Upload,
} from "lucide-react";

interface ModalityChipsProps {
  twin: TwinFull;
  onTwinUpdated: (twin: TwinFull) => void;
}

type ModalityKey = "basic_scores" | "biomarkers" | "mri_volumes" | "mri_image" | "ocr_report";

interface ModalityMeta {
  key: ModalityKey;
  label: string;
  icon: React.ElementType;
  description: string;
  inputType: "form" | "file";
  fileAccept?: string;
  disclaimer?: string;
}

const MODALITIES: ModalityMeta[] = [
  {
    key: "basic_scores",
    label: "Basic Scores",
    icon: Activity,
    description: "Cognitive screening metrics (MMSE, CDR-Global, ADAS-Cog/TOTSCORE)",
    inputType: "form",
  },
  {
    key: "biomarkers",
    label: "Fluid Biomarkers",
    icon: Dna,
    description: "CSF or plasma markers (Aβ42, t-tau, p-tau181)",
    inputType: "form",
  },
  {
    key: "mri_volumes",
    label: "MRI Volumes",
    icon: Layers,
    description: "Volumetric parcellations (Hippocampus, Ventricles, Whole Brain)",
    inputType: "form",
  },
  {
    key: "mri_image",
    label: "MRI Image Scan",
    icon: ImageIcon,
    description: "Structural T1-weighted axial brain scan (PNG/JPEG)",
    inputType: "file",
    fileAccept: "image/png,image/jpeg,image/webp",
  },
  {
    key: "ocr_report",
    label: "OCR Lab Report",
    icon: FileText,
    description: "Clinical PDF or image scan of lab tests with automated regex extraction",
    inputType: "file",
    fileAccept: ".pdf,image/*",
  },
];

export default function ModalityChips({ twin, onTwinUpdated }: ModalityChipsProps) {
  const [activeModal, setActiveModal] = useState<ModalityMeta | null>(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Form states for modals
  const [basicForm, setBasicForm] = useState({ MMSE: 26, CDGLOBAL: 0.5, TOTSCORE: 12 });
  const [biomarkersForm, setBiomarkersForm] = useState({ ABETA: 850, TAU: 420, PTAU: 35 });
  const [mriVolForm, setMriVolForm] = useState({
    hippocampus_left: 3250,
    hippocampus_right: 3300,
    ventricles: 36000,
  });
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [uploadSuccessResult, setUploadSuccessResult] = useState<any | null>(null);

  const handleOpenModal = (modality: ModalityMeta) => {
    setActiveModal(modality);
    setSelectedFile(null);
    setPreviewUrl(null);
    setUploadSuccessResult(null);
    setErrorMsg(null);
  };

  const handleCloseModal = () => {
    setActiveModal(null);
    setSelectedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setUploadSuccessResult(null);
    setErrorMsg(null);
    setLoading(false);
  };

  const handleFileChange = (file: File | null) => {
    setSelectedFile(file);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    if (file && (file.type.startsWith("image/") || file.name.match(/\.(jpg|jpeg|png|webp)$/i))) {
      setPreviewUrl(URL.createObjectURL(file));
    } else {
      setPreviewUrl(null);
    }
  };

  const handleSubmitModal = async () => {
    if (!activeModal) return;
    setLoading(true);
    setErrorMsg(null);

    try {
      let updatedTwin: TwinFull;
      if (activeModal.key === "biomarkers") {
        updatedTwin = await twinApi.attachModality(twin.twin_id, "biomarkers", biomarkersForm);
      } else if (activeModal.key === "mri_volumes") {
        updatedTwin = await twinApi.attachModality(twin.twin_id, "mri_volumes", mriVolForm);
      } else if (activeModal.key === "basic_scores") {
        // Basic scores are added via visit or directly
        updatedTwin = await twinApi.attachModality(twin.twin_id, "basic_scores" as any, basicForm);
      } else if (activeModal.key === "mri_image") {
        let fileToUpload = selectedFile;
        if (!fileToUpload) {
          const blob = new Blob(["mock-image-data"], { type: "image/png" });
          fileToUpload = new File([blob], "sample_mri_t1.png", { type: "image/png" });
        }
        const mriRes = await twinApi.uploadMRIImage(twin.twin_id, fileToUpload);
        setUploadSuccessResult(mriRes.mri);
        try {
          // Update patientStore if present
          const store = (window as any).__patientStore;
          if (store?.getState?.()?.setMRIUploadResult) {
            store.getState().setMRIUploadResult(mriRes);
          }
        } catch (_) {}

        updatedTwin = {
          ...twin,
          state: mriRes.state,
          modalities: {
            ...twin.modalities,
            mri_image: {
              present: true,
              latest: mriRes.mri,
            },
          },
        };
      } else if (activeModal.key === "ocr_report") {
        if (!selectedFile) {
          const blob = new Blob(["mock-report-pdf"], { type: "application/pdf" });
          const dummyFile = new File([blob], "lab_report.pdf", { type: "application/pdf" });
          updatedTwin = await twinApi.attachModality(twin.twin_id, "ocr_report", undefined, dummyFile);
        } else {
          updatedTwin = await twinApi.attachModality(twin.twin_id, "ocr_report", undefined, selectedFile);
        }
      } else {
        throw new Error("Unknown modality");
      }

      onTwinUpdated(updatedTwin);
      setTimeout(() => {
        handleCloseModal();
      }, 500);
    } catch (err: any) {
      console.error("Attach modality failed:", err);
      setErrorMsg(err?.message || "Failed to attach modality");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="rounded-2xl bg-neutral-900/60 backdrop-blur-xl border border-white/10 p-5 shadow-2xl">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <Layers className="w-4 h-4 text-blue-400" />
            Clinical Modalities
          </h3>
          <p className="text-xs text-neutral-400 mt-0.5">
            Attach multi-modal data streams to refine diagnostic certainty (T1 → T4).
          </p>
        </div>
      </div>

      {/* Chips list */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
        {MODALITIES.map((mod) => {
          const status = twin.modalities?.[mod.key] as ModalityStatus | undefined;
          const isPresent = !!(
            status?.present ||
            (mod.key === "mri_image" && twin.visits?.some((v) => v.mri_scan_present || v.mri_result))
          );
          const Icon = mod.icon;

          return (
            <div
              key={mod.key}
              className={`relative rounded-xl p-3.5 border transition-all flex flex-col justify-between ${
                isPresent
                  ? "bg-blue-950/30 border-blue-500/40 shadow-sm shadow-blue-500/10"
                  : "bg-white/[0.03] border-white/10 hover:border-white/20"
              }`}
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <div
                      className={`p-1.5 rounded-lg ${
                        isPresent ? "bg-blue-500/20 text-blue-400" : "bg-white/5 text-neutral-400"
                      }`}
                    >
                      <Icon className="w-4 h-4" />
                    </div>
                    <span className="text-xs font-semibold text-white">{mod.label}</span>
                  </div>

                  {isPresent ? (
                    <span className="flex items-center gap-1 text-[11px] font-medium text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20">
                      <CheckCircle2 className="w-3 h-3" />
                      Present
                    </span>
                  ) : (
                    <span className="flex items-center gap-1 text-[11px] font-medium text-neutral-400 bg-white/5 px-2 py-0.5 rounded-full border border-white/10">
                      <XCircle className="w-3 h-3 text-neutral-500" />
                      Missing
                    </span>
                  )}
                </div>

                <p className="text-[11px] text-neutral-400 line-clamp-2 leading-relaxed">
                  {mod.description}
                </p>
              </div>

              {/* Action Button */}
              <div className="mt-3 pt-2 border-t border-white/5">
                <button
                  onClick={() => handleOpenModal(mod)}
                  className={`w-full py-1.5 px-3 rounded-lg text-xs font-medium flex items-center justify-center gap-1.5 transition cursor-pointer ${
                    isPresent
                      ? "bg-white/5 hover:bg-white/10 text-neutral-300 border border-white/10"
                      : "bg-blue-600 hover:bg-blue-500 text-white shadow-sm shadow-blue-600/30"
                  }`}
                >
                  <Plus className="w-3.5 h-3.5" />
                  {isPresent ? "Update" : "Add Modality"}
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {/* Modal Dialog */}
      {activeModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md">
          <div className="relative w-full max-w-lg rounded-2xl bg-neutral-900 border border-white/20 p-6 shadow-2xl">
            {/* Modal Header */}
            <div className="flex items-center justify-between pb-4 border-b border-white/10">
              <div className="flex items-center gap-2.5">
                <activeModal.icon className="w-5 h-5 text-blue-400" />
                <h4 className="text-lg font-bold text-white">Attach {activeModal.label}</h4>
              </div>
              <button
                onClick={handleCloseModal}
                className="p-1 rounded-lg hover:bg-white/10 text-neutral-400 hover:text-white transition"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Disclaimer if any */}
            {activeModal.disclaimer && (
              <div className="mt-4 p-3 rounded-xl bg-purple-950/40 border border-purple-500/30 text-xs text-purple-200 flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-purple-400 shrink-0 mt-0.5" />
                <span>
                  <strong>Notice:</strong> {activeModal.disclaimer}
                </span>
              </div>
            )}

            {/* Form Inputs based on modality */}
            <div className="mt-4 space-y-4">
              {activeModal.key === "basic_scores" && (
                <div className="grid grid-cols-3 gap-3">
                  <div>
                    <label className="text-xs text-neutral-400 block mb-1">MMSE (0–30)</label>
                    <input
                      type="number"
                      min={0}
                      max={30}
                      value={basicForm.MMSE}
                      onChange={(e) => setBasicForm({ ...basicForm, MMSE: Number(e.target.value) })}
                      className="w-full bg-neutral-800 border border-white/10 rounded-lg p-2 text-sm text-white"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-neutral-400 block mb-1">CDR (0–3)</label>
                    <input
                      type="number"
                      step={0.5}
                      min={0}
                      max={3}
                      value={basicForm.CDGLOBAL}
                      onChange={(e) =>
                        setBasicForm({ ...basicForm, CDGLOBAL: Number(e.target.value) })
                      }
                      className="w-full bg-neutral-800 border border-white/10 rounded-lg p-2 text-sm text-white"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-neutral-400 block mb-1">TOTSCORE (0–70)</label>
                    <input
                      type="number"
                      min={0}
                      max={70}
                      value={basicForm.TOTSCORE}
                      onChange={(e) =>
                        setBasicForm({ ...basicForm, TOTSCORE: Number(e.target.value) })
                      }
                      className="w-full bg-neutral-800 border border-white/10 rounded-lg p-2 text-sm text-white"
                    />
                  </div>
                </div>
              )}

              {activeModal.key === "biomarkers" && (
                <div className="grid grid-cols-3 gap-3">
                  <div>
                    <label className="text-xs text-neutral-400 block mb-1">Aβ42 (pg/mL)</label>
                    <input
                      type="number"
                      value={biomarkersForm.ABETA}
                      onChange={(e) =>
                        setBiomarkersForm({ ...biomarkersForm, ABETA: Number(e.target.value) })
                      }
                      className="w-full bg-neutral-800 border border-white/10 rounded-lg p-2 text-sm text-white"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-neutral-400 block mb-1">t-Tau (pg/mL)</label>
                    <input
                      type="number"
                      value={biomarkersForm.TAU}
                      onChange={(e) =>
                        setBiomarkersForm({ ...biomarkersForm, TAU: Number(e.target.value) })
                      }
                      className="w-full bg-neutral-800 border border-white/10 rounded-lg p-2 text-sm text-white"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-neutral-400 block mb-1">p-Tau181 (pg/mL)</label>
                    <input
                      type="number"
                      value={biomarkersForm.PTAU}
                      onChange={(e) =>
                        setBiomarkersForm({ ...biomarkersForm, PTAU: Number(e.target.value) })
                      }
                      className="w-full bg-neutral-800 border border-white/10 rounded-lg p-2 text-sm text-white"
                    />
                  </div>
                </div>
              )}

              {activeModal.key === "mri_volumes" && (
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-xs text-neutral-400 block mb-1">Hippocampus L (mm³)</label>
                    <input
                      type="number"
                      value={mriVolForm.hippocampus_left}
                      onChange={(e) =>
                        setMriVolForm({
                          ...mriVolForm,
                          hippocampus_left: Number(e.target.value),
                        })
                      }
                      className="w-full bg-neutral-800 border border-white/10 rounded-lg p-2 text-sm text-white"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-neutral-400 block mb-1">Ventricles (mm³)</label>
                    <input
                      type="number"
                      value={mriVolForm.ventricles}
                      onChange={(e) =>
                        setMriVolForm({
                          ...mriVolForm,
                          ventricles: Number(e.target.value),
                        })
                      }
                      className="w-full bg-neutral-800 border border-white/10 rounded-lg p-2 text-sm text-white"
                    />
                  </div>
                </div>
              )}

              {(activeModal.key === "mri_image" || activeModal.key === "ocr_report") && (
                <div className="space-y-3">
                  <div className="border-2 border-dashed border-white/15 rounded-xl p-5 text-center hover:border-blue-400/50 transition bg-white/[0.02]">
                    {previewUrl ? (
                      <div className="flex flex-col items-center">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={previewUrl}
                          alt="MRI preview"
                          className="w-40 h-40 object-cover rounded-lg border border-white/20 shadow-lg mb-3"
                        />
                        <p className="text-xs font-semibold text-emerald-400 mb-1 flex items-center gap-1.5">
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          {selectedFile?.name} ({(selectedFile ? (selectedFile.size / 1024).toFixed(1) : "0")} KB)
                        </p>
                      </div>
                    ) : (
                      <>
                        <Upload className="w-8 h-8 text-blue-400 mx-auto mb-2" />
                        <p className="text-sm font-medium text-white mb-1">
                          {selectedFile ? selectedFile.name : `Select or drop ${activeModal.label} file`}
                        </p>
                        <p className="text-xs text-neutral-400 mb-3">
                          Accepts {activeModal.fileAccept || "standard documents"}
                        </p>
                      </>
                    )}

                    <label className="inline-block px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg text-xs font-semibold text-white cursor-pointer transition">
                      {selectedFile ? "Change File" : "Browse File"}
                      <input
                        type="file"
                        accept={activeModal.fileAccept}
                        onChange={(e) => handleFileChange(e.target.files?.[0] || null)}
                        className="hidden"
                      />
                    </label>
                    {!selectedFile && (
                      <p className="text-[11px] text-neutral-400 mt-2">
                        (Or click Submit to load a pre-configured sample scan)
                      </p>
                    )}
                  </div>

                  {uploadSuccessResult && (
                    <div className="p-3.5 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-xs flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                        <div>
                          <p className="font-bold text-white">MRI Classification Succeeded</p>
                          <p className="text-[11px] text-neutral-300">
                            Predicted: <span className="font-semibold text-emerald-300">{uploadSuccessResult.predicted_class}</span>
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {errorMsg && (
                <div className="p-3 rounded-lg bg-rose-500/20 border border-rose-500/30 text-rose-300 text-xs">
                  {errorMsg}
                </div>
              )}
            </div>

            {/* Modal Actions */}
            <div className="mt-6 flex items-center justify-end gap-3 pt-4 border-t border-white/10">
              <button
                onClick={handleCloseModal}
                disabled={loading}
                className="px-4 py-2 rounded-lg text-xs font-medium text-neutral-400 hover:text-white transition"
              >
                Cancel
              </button>
              <button
                onClick={handleSubmitModal}
                disabled={loading}
                className="px-5 py-2 rounded-lg text-xs font-bold bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-600/30 transition flex items-center gap-2 cursor-pointer disabled:opacity-60"
              >
                {loading && (
                  <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                )}
                <span>{loading ? "Processing..." : "Submit Modality"}</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
