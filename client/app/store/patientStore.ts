// app/store/patientStore.ts
"use client";

import { create } from "zustand";
import { nanoid } from "nanoid";

/* ----------------------------------------------------------
   INTERFACES
---------------------------------------------------------- */

export interface Demographics {
  name: string | null;
  age: number | null;
  gender: number | null; // 0=Male, 1=Female

  ethnicity: number | null; // 0=Caucasian,1=African American,2=Asian,3=Other
  educationLevel: number | null; // 0=None,1=High School,2=Bachelor's,3=Higher

  height: number | null;
  weight: number | null;
  bmi: number | null;

  smoking: number | null; // 0/1
  alcoholConsumption: number | null; // 0–20
  physicalActivity: number | null; // 0–10
  dietQuality: number | null; // 0–10
  sleepQuality: number | null; // 4–10

  familyHistoryAlzheimers: number | null; // 0/1
  baselineDate?: string | null; // "YYYY-MM-DD"
  doctorInCharge: string | null; // For completeness ("XXXConfid")
}

export interface Medical {
  cardiovascularDisease: number | null;
  diabetes: number | null;
  depression: number | null;
  headInjury: number | null;
  hypertension: number | null;

  // Clinical Measurements
  systolicBP: number | null;
  diastolicBP: number | null;
  cholesterolTotal: number | null;
  cholesterolLDL: number | null;
  cholesterolHDL: number | null;
  cholesterolTriglycerides: number | null;
}

export interface Cognitive {
  mmse: number | null;
  functionalAssessment: number | null;
  adl: number | null;

  memoryComplaints: number | null;
  behavioralProblems: number | null;
  confusion: number | null;

  disorientation: number | null;
  personalityChanges: number | null;
  difficultyCompletingTasks: number | null;
  forgetfulness: number | null;
}

export interface AlleleInput {
  ABETA: number | null;
  TAU: number | null;
  MMSE: number | null;
  APVOLUME: number | null;
  GENOTYPE: string | null;
}

export interface AdvancedScores {
  CDGLOBAL: number | null;
  ADAS11: number | null;
  FAQ: number | null;
  MOCA: number | null;
  PTAU: number | null;
  mriVolumes: {
    hippocampus?: number | null;
    entorhinal?: number | null;
    ventricles?: number | null;
    wholeBrain?: number | null;
    icv?: number | null;
  };
}

export interface PatientSession {
  id: string;
  timestamp: string;

  demographics: Demographics;
  medical: Medical;

  uploads: {
    mriFile: File | null;
    reportFile: File | null;
    mriResult: any | null;
    ocrResult: any | null;
  };

  cognitive: Cognitive;
  advancedScores: AdvancedScores;

  baseModel: {
    input_used: any | null;
    prediction: string | null;
  };

  alleleInput: AlleleInput;

  alleleResult: {
    predictedClass: string | null;
    probability: number | null;
    riskCategory: string | null;
  };

  twinId?: string | null;
  twin?: import("../lib/api").TwinFull | null;
  explain?: import("../lib/api").ExplainResponse | null;
  inputUsage?: import("../lib/api").InputUsageResponse | null;
  mriUploadResult?: import("../lib/api").MRIUploadResponse | null;
}

interface StoreState {
  session: PatientSession;

  updateDemographics: (data: Partial<Demographics>) => void;
  updateMedical: (data: Partial<Medical>) => void;
  updateCognitive: (data: Partial<Cognitive>) => void;
  updateAdvancedScores: (data: Partial<AdvancedScores>) => void;
  updateMriVolumes: (data: Partial<AdvancedScores["mriVolumes"]>) => void;
  updateUploads: (data: Partial<PatientSession["uploads"]>) => void;

  setAlleleInput: (data: Partial<AlleleInput>) => void;
  setAlleleResult: (res: Partial<{
    predictedClass: string;
    probability: number | null;
    riskCategory: string;
  }>) => void;

  setMRIResult: (res: any) => void;
  setOCRResult: (res: any) => void;

  setBaseModelResult: (input: any, prediction: string) => void;

  setTwinId: (twinId: string | null) => void;
  setTwin: (twin: import("../lib/api").TwinFull | null) => void;
  updateTwinState: (updater: (prev: import("../lib/api").TwinFull | null) => import("../lib/api").TwinFull | null) => void;

  setExplain: (explain: import("../lib/api").ExplainResponse | null) => void;
  setInputUsage: (inputUsage: import("../lib/api").InputUsageResponse | null) => void;
  setMRIUploadResult: (result: import("../lib/api").MRIUploadResponse | null) => void;

  autofillSamplePatient: () => void;
  loadPersona: (persona: import("../lib/api").DemoPatientEntry) => void;
  loadDemoCase: (demoCase: import("../lib/api").DemoCase) => void;

  resetSession: () => void;
}

/* ----------------------------------------------------------
   DEFAULT SESSION OBJECT
---------------------------------------------------------- */

const defaultCognitive: Cognitive = {
  mmse: null,
  functionalAssessment: null,
  adl: null,

  memoryComplaints: null,
  behavioralProblems: null,
  confusion: null,

  disorientation: null,
  personalityChanges: null,
  difficultyCompletingTasks: null,
  forgetfulness: null,
};

const defaultDemographics: Demographics = {
  name: null,
  age: null,
  gender: null,
  ethnicity: null,
  educationLevel: null,

  height: null,
  weight: null,
  bmi: null,

  smoking: null,
  alcoholConsumption: null,
  physicalActivity: null,
  dietQuality: null,
  sleepQuality: null,

  familyHistoryAlzheimers: null,
  doctorInCharge: "XXXConfid",
};

const defaultMedical: Medical = {
  cardiovascularDisease: null,
  diabetes: null,
  depression: null,
  headInjury: null,
  hypertension: null,

  systolicBP: null,
  diastolicBP: null,
  cholesterolTotal: null,
  cholesterolLDL: null,
  cholesterolHDL: null,
  cholesterolTriglycerides: null,
};

const defaultAllele: AlleleInput = {
  ABETA: null,
  TAU: null,
  MMSE: null,
  APVOLUME: null,
  GENOTYPE: null,
};

const defaultAdvancedScores: AdvancedScores = {
  CDGLOBAL: null,
  ADAS11: null,
  FAQ: null,
  MOCA: null,
  PTAU: null,
  mriVolumes: {
    hippocampus: null,
    entorhinal: null,
    ventricles: null,
    wholeBrain: null,
    icv: null,
  },
};

/* ----------------------------------------------------------
   ZUSTAND STORE
---------------------------------------------------------- */

export const usePatientStore = create<StoreState>((set) => ({
  session: {
    id: nanoid(),
    timestamp: new Date().toISOString(),

    demographics: { ...defaultDemographics },
    medical: { ...defaultMedical },
    cognitive: { ...defaultCognitive },
    advancedScores: { ...defaultAdvancedScores },
    alleleInput: { ...defaultAllele },

    uploads: {
      mriFile: null,
      reportFile: null,
      mriResult: null,
      ocrResult: null,
    },

    baseModel: {
      input_used: null,
      prediction: null,
    },

    alleleResult: {
      predictedClass: null,
      probability: null,
      riskCategory: null,
    },

    twinId: null,
    twin: null,
  },

  /* ------------------ UPDATE DEMOGRAPHICS ------------------ */
  updateDemographics: (data: Partial<Demographics>) =>
    set((state) => {
      const newDemo = { ...state.session.demographics, ...data };

      if (newDemo.height && newDemo.weight) {
        newDemo.bmi = Number(
          (newDemo.weight / (newDemo.height / 100) ** 2).toFixed(1)
        );
      }

      return {
        session: {
          ...state.session,
          demographics: newDemo,
        },
      };
    }),

  /* ------------------ UPDATE MEDICAL ------------------ */
  updateMedical: (data: Partial<Medical>) =>
    set((state) => ({
      session: {
        ...state.session,
        medical: { ...state.session.medical, ...data },
      },
    })),

  /* ------------------ UPDATE COGNITIVE ------------------ */
  updateCognitive: (data: Partial<Cognitive>) =>
    set((state) => ({
      session: {
        ...state.session,
        cognitive: { ...state.session.cognitive, ...data },
      },
    })),

  /* ------------------ UPLOADS ------------------ */
  updateUploads: (data: Partial<PatientSession["uploads"]>) =>
    set((state) => ({
      session: {
        ...state.session,
        uploads: { ...state.session.uploads, ...data },
      },
    })),

  setMRIResult: (res: any) =>
    set((state) => ({
      session: {
        ...state.session,
        uploads: { ...state.session.uploads, mriResult: res },
      },
    })),

  setOCRResult: (res: any) =>
    set((state) => ({
      session: {
        ...state.session,
        uploads: { ...state.session.uploads, ocrResult: res },
      },
    })),

  /* ------------------ ALLELE ------------------ */
  setAlleleInput: (data: Partial<AlleleInput>) =>
    set((state) => ({
      session: {
        ...state.session,
        alleleInput: { ...state.session.alleleInput, ...data },
      },
    })),

  setAlleleResult: (res) =>
  set((state) => ({
    session: {
      ...state.session,
      alleleResult: {
        ...state.session.alleleResult,
        ...res,
      },
    },
  })),


  /* ------------------ BASE MODEL ------------------ */
  setBaseModelResult: (input, prediction) =>
    set((state) => ({
      session: {
        ...state.session,
        baseModel: { input_used: input, prediction },
      },
    })),

  /* ------------------ DIGITAL TWIN ------------------ */
  setTwinId: (twinId) =>
    set((state) => ({
      session: {
        ...state.session,
        twinId,
      },
    })),

  setTwin: (twin) =>
    set((state) => ({
      session: {
        ...state.session,
        twin,
        twinId: twin ? twin.twin_id : state.session.twinId,
      },
    })),

  updateTwinState: (updater) =>
    set((state) => {
      const updated = updater(state.session.twin ?? null);
      return {
        session: {
          ...state.session,
          twin: updated,
          twinId: updated ? updated.twin_id : state.session.twinId,
        },
      };
    }),

  setExplain: (explain) =>
    set((state) => ({
      session: {
        ...state.session,
        explain,
      },
    })),

  setInputUsage: (inputUsage) =>
    set((state) => ({
      session: {
        ...state.session,
        inputUsage,
      },
    })),

  setMRIUploadResult: (mriUploadResult) =>
    set((state) => ({
      session: {
        ...state.session,
        mriUploadResult,
      },
    })),

  /* ------------------ ADVANCED SCORES ------------------ */
  updateAdvancedScores: (data: Partial<AdvancedScores>) =>
    set((state) => ({
      session: {
        ...state.session,
        advancedScores: { ...state.session.advancedScores, ...data },
      },
    })),

  updateMriVolumes: (data: Partial<AdvancedScores["mriVolumes"]>) =>
    set((state) => ({
      session: {
        ...state.session,
        advancedScores: {
          ...state.session.advancedScores,
          mriVolumes: {
            ...state.session.advancedScores.mriVolumes,
            ...data,
          },
        },
      },
    })),

  /* ------------------ AUTOFILL SAMPLE PATIENT ------------------ */
  autofillSamplePatient: () =>
    set((state) => ({
      session: {
        ...state.session,
        demographics: {
          ...state.session.demographics,
          name: "Eleanor Vance",
          age: 74,
          gender: 1, // Female
          ethnicity: 0, // Caucasian
          educationLevel: 2, // Undergraduate (16 years)
          height: 162,
          weight: 62,
          bmi: 23.6,
          smoking: 0,
          alcoholConsumption: 2,
          physicalActivity: 6,
          dietQuality: 7,
          sleepQuality: 8,
          familyHistoryAlzheimers: 1,
          baselineDate: new Date().toISOString().split("T")[0],
          doctorInCharge: "Dr. A. Sterling",
        },
        cognitive: {
          ...state.session.cognitive,
          mmse: 26,
          functionalAssessment: 15,
          adl: 18,
          memoryComplaints: 1,
          confusion: 0,
          behavioralProblems: 0,
          forgetfulness: 1,
          difficultyCompletingTasks: 0,
          disorientation: 0,
          personalityChanges: 0,
        },
        advancedScores: {
          CDGLOBAL: 0.5,
          ADAS11: 12.0,
          FAQ: 4.0,
          MOCA: 23.0,
          PTAU: 38.0,
          mriVolumes: {
            hippocampus: 3400,
            entorhinal: 1350,
            ventricles: 35000,
            wholeBrain: 1050000,
            icv: 1450000,
          },
        },
        alleleInput: {
          ABETA: 980,
          TAU: 432,
          MMSE: 26,
          APVOLUME: 3400,
          GENOTYPE: "e3/e4",
        },
      },
    })),

  /* ------------------ LOAD PERSONA INTO EDITOR ------------------ */
  loadPersona: (persona) =>
    set((state) => {
      // Map persona baseline traits to realistic intake values
      const isAd = persona.baseline_stage === "AD";
      const isMci = persona.baseline_stage === "MCI";
      const isCn = persona.baseline_stage === "CN";

      const mmse = isCn ? 29 : isMci ? 25 : 18;
      const cdr = isCn ? 0.0 : isMci ? 0.5 : 1.5;
      const adas = isCn ? 6.0 : isMci ? 14.0 : 28.0;
      const abeta = isCn ? 1200 : isMci ? 850 : 540;
      const tau = isCn ? 280 : isMci ? 440 : 680;
      const ptau = isCn ? 22 : isMci ? 38 : 65;

      return {
        session: {
          ...state.session,
          demographics: {
            ...state.session.demographics,
            name: persona.persona_label,
            age: 73,
            gender: 0,
            educationLevel: 2,
            familyHistoryAlzheimers: isCn ? 0 : 1,
          },
          cognitive: {
            ...state.session.cognitive,
            mmse,
            memoryComplaints: isCn ? 0 : 1,
            forgetfulness: isCn ? 0 : 1,
          },
          advancedScores: {
            CDGLOBAL: cdr,
            ADAS11: adas,
            FAQ: isCn ? 0 : isMci ? 4 : 12,
            MOCA: isCn ? 28 : isMci ? 22 : 15,
            PTAU: ptau,
            mriVolumes: {
              hippocampus: isCn ? 3800 : isMci ? 3300 : 2600,
              ventricles: isCn ? 26000 : isMci ? 38000 : 56000,
              wholeBrain: 1100000,
              entorhinal: isCn ? 1600 : isMci ? 1300 : 950,
              icv: 1480000,
            },
          },
          alleleInput: {
            ABETA: abeta,
            TAU: tau,
            MMSE: mmse,
            APVOLUME: 3400,
            GENOTYPE: isCn ? "e3/e3" : isMci ? "e3/e4" : "e4/e4",
          },
        },
      };
    }),

  /* ------------------ LOAD DEMO CASE BASELINE ------------------ */
  loadDemoCase: (demoCase) =>
    set((state) => {
      const v0 = demoCase.visits[0] || {};
      const isCn = demoCase.baseline_stage === "CN";
      const isMci = demoCase.baseline_stage === "MCI";
      const genotype = demoCase.apoe4_count === 2 ? "e4/e4" : demoCase.apoe4_count === 1 ? "e3/e4" : "e3/e3";

      return {
        session: {
          ...state.session,
          demographics: {
            ...state.session.demographics,
            name: demoCase.patient_name,
            age: demoCase.age,
            gender: demoCase.sex,
            educationLevel: demoCase.education_years >= 16 ? 2 : demoCase.education_years >= 12 ? 1 : 0,
            baselineDate: demoCase.baseline_date || v0.exam_date || "2024-01-15",
            familyHistoryAlzheimers: demoCase.apoe4_count > 0 ? 1 : 0,
            smoking: v0.lifestyle?.smoking ?? 0,
            alcoholConsumption: v0.lifestyle?.alcohol ?? 1,
            physicalActivity: v0.lifestyle?.physicalActivity ?? 5,
            dietQuality: v0.lifestyle?.dietQuality ?? 6,
            sleepQuality: v0.lifestyle?.sleepQuality ?? 7,
          },
          medical: {
            ...state.session.medical,
            hypertension: v0.medical?.hypertension ?? 0,
            diabetes: v0.medical?.diabetes ?? 0,
            depression: v0.medical?.depression ?? 0,
            headInjury: v0.medical?.headInjury ?? 0,
            cardiovascularDisease: v0.medical?.cardiovascularDisease ?? 0,
          },
          cognitive: {
            ...state.session.cognitive,
            mmse: v0.MMSE,
            functionalAssessment: v0.functionalAssessment ?? (v0.MMSE >= 26 ? 8 : v0.MMSE >= 21 ? 5 : 3),
            adl: v0.adl ?? (v0.MMSE >= 26 ? 9 : v0.MMSE >= 21 ? 6 : 4),
            memoryComplaints: v0.symptoms?.memoryComplaints ? 1 : isCn ? 0 : 1,
            behavioralProblems: v0.symptoms?.behavioralProblems ? 1 : 0,
            confusion: v0.symptoms?.confusion ? 1 : isCn ? 0 : 1,
            disorientation: v0.symptoms?.disorientation ? 1 : 0,
            personalityChanges: v0.symptoms?.personalityChanges ? 1 : 0,
            difficultyCompletingTasks: v0.symptoms?.difficultyCompletingTasks ? 1 : isCn ? 0 : 1,
            forgetfulness: v0.symptoms?.forgetfulness ? 1 : isCn ? 0 : 1,
          },
          advancedScores: {
            CDGLOBAL: v0.CDGLOBAL,
            ADAS11: v0.TOTSCORE,
            FAQ: v0.FAQ ?? (v0.CDGLOBAL === 0 ? 0 : v0.CDGLOBAL === 0.5 ? 4 : 12),
            MOCA: v0.MOCA ?? Math.max(10, v0.MMSE - 2),
            PTAU: v0.biomarkers?.PTAU ?? (v0.MMSE < 20 ? 65 : v0.MMSE < 26 ? 38 : 22),
            mriVolumes: {
              hippocampus: v0.mri_volumes?.hippocampus ?? (isCn ? 3800 : isMci ? 3300 : 2600),
              ventricles: v0.mri_volumes?.ventricles ?? (isCn ? 26000 : isMci ? 38000 : 56000),
              wholeBrain: v0.mri_volumes?.whole_brain ?? 1100000,
              entorhinal: v0.mri_volumes?.entorhinal ?? (isCn ? 1600 : isMci ? 1300 : 950),
              icv: v0.mri_volumes?.icv ?? 1480000,
            },
          },
          alleleInput: {
            ABETA: v0.biomarkers?.ABETA ?? (isCn ? 1200 : isMci ? 850 : 540),
            TAU: v0.biomarkers?.TAU ?? (isCn ? 280 : isMci ? 440 : 680),
            MMSE: v0.MMSE,
            APVOLUME: v0.biomarkers?.APVOLUME ?? 3400,
            GENOTYPE: genotype,
          },
        },
      };
    }),

  /* ------------------ RESET EVERYTHING ------------------ */
  resetSession: () =>
    set({
      session: {
        id: nanoid(),
        timestamp: new Date().toISOString(),
        demographics: { ...defaultDemographics },
        medical: { ...defaultMedical },
        cognitive: { ...defaultCognitive },
        advancedScores: { ...defaultAdvancedScores },
        uploads: {
          mriFile: null,
          reportFile: null,
          mriResult: null,
          ocrResult: null,
        },
        baseModel: { input_used: null, prediction: null },
        alleleInput: { ...defaultAllele },
        alleleResult: {
          predictedClass: null,
          probability: null,
          riskCategory: null,
        },
        twinId: null,
        twin: null,
      },
    }),
}));
