<div align="center">

<br/>

<img src="https://img.shields.io/badge/Research-Alzheimer's%20AI-8B5CF6?style=for-the-badge&logo=brain&logoColor=white" />
<img src="https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge" />
<img src="https://img.shields.io/badge/Python-FastAPI-3B82F6?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/Next.js-React-black?style=for-the-badge&logo=next.js&logoColor=white" />
<img src="https://img.shields.io/badge/PyTorch-FastAI-EF4444?style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/Keras-Bi--LSTM-FFCA28?style=for-the-badge&logo=keras&logoColor=black" />

<br/><br/>

# 🧠 Multimodal Digital Twin for Alzheimer's Disease
### Current Diagnosis & Longitudinal Progression Forecasting

**A unified AI framework combining clinical data, MRI scans, biomarkers, and longitudinal temporal modeling to predict both current disease state and future disease trajectories.**

<br/>

[![Watch Full Project Demo](https://img.youtube.com/vi/J1eA6TE1r8I/maxresdefault.jpg)](https://youtu.be/J1eA6TE1r8I?si=zeuuYOkh5rjfdig2)

<br/>

</div>

---

## ⚡ 30-Second Project Overview

Traditional Alzheimer's diagnosis typically assesses a patient at a single point in time, answering: *"Does this patient currently have Alzheimer's?"* 

Our project moves beyond this by asking: **"How will this patient's condition evolve in the future?"**

We developed a **multimodal Digital Twin platform** for Alzheimer's Disease. 
- **Phase 1:** Developed independent prediction pipelines using clinical data, MRI scans, biomarker-genetic data, and an OCR medical report ingestion pipeline.
- **Phase 2:** Extended the system with **longitudinal temporal modeling** using a Bidirectional LSTM network to predict future disease progression from historical patient visits.

The ultimate goal is to create a patient-specific digital twin that continuously updates and forecasts the disease trajectory over time.

---

## 🏗️ Full System Architecture

```text
                      Patient Data
                            │
                            │
      ┌─────────────────────┼─────────────────────┐
      │                     │                     │                     │
┌─────────────┐       ┌────────────┐       ┌──────────────┐      ┌────────────┐
│ Clinical    │       │ MRI Scans  │       │ Biomarkers & │      │ Medical    │
│ Records     │       │            │       │ Genetics     │      │ Reports    │
└──────┬──────┘       └──────┬─────┘       └──────┬───────┘      └──────┬─────┘
       │                     │                    │                     │
┌─────────────┐       ┌────────────┐       ┌──────────────┐      ┌────────────┐
│ CatBoost    │       │ ResNet34   │       │ Ensemble     │      │ Llama 4    │
│ Classifier  │       │ (FastAI)   │       │ (XGB+RF+MLP) │      │ Scout OCR  │
└──────┬──────┘       └──────┬─────┘       └──────┬───────┘      └──────┬─────┘
       │                     │                    │                     │
       └─────────────────────┴────────────────────┴─────────────────────┘
                                     │
                                     ▼
                            ┌────────────────┐
                            │  Digital Twin  │ (Current Snapshot)
                            └────────┬───────┘
                                     │
                                     ▼
                            ┌────────────────┐
                            │ Longitudinal AI│ (Bi-LSTM)
                            │ Temporal Model │
                            └────────┬───────┘
                                     │
                                     ▼
                        ┌────────────────────────┐
                        │ Future Disease         │
                        │ Progression Forecast   │
                        └────────────────────────┘
```

---

## 🧬 Phase 1: Multimodal Snapshot Prediction

In Phase 1, we built an ensemble of distinct models targeting specific data modalities. These individual predictions are fused into a **unified JSON-based Digital Twin**.

### 1️⃣ Clinical Prediction Model
*   **Dataset:** 2,149 patient records (Age, Lifestyle, Cognitive scores, Functional assessments).
*   **Model:** `CatBoost`
*   **Output:** Alzheimer's diagnosis prediction (Positive / Negative).
*   **Performance:** ~96% Accuracy | F1 = 0.94

### 2️⃣ MRI Analysis Pipeline
*   **Dataset:** 44,000 MRI brain scans.
*   **Model:** `ResNet34 CNN` (Implemented via `FastAI`).
*   **Output:** MRI-based dementia severity classification (Non Demented, Very Mild, Mild, Moderate).
*   **Performance:** ~99.7% Validation Accuracy.

### 3️⃣ Biomarker + Genetics Pipeline
*   **Dataset:** ADNI (ABETA, TAU, APOE genotype, MMSE, Brain volume).
*   **Model:** `Ensemble Model` (Weighted blend of XGBoost, Random Forest, and a Neural Network).
*   **Output:** Risk progression categories (e.g., Very Low Risk, Moderate MCI, Very High Risk).
*   **Performance:** ~63.3% Accuracy (Realistic benchmark for biomarker-only predictions).

### 4️⃣ OCR Medical Report Ingestion
*   **Problem:** Dealing with unstructured real-world hospital data (PDFs, scans, images).
*   **Solution:** A **Vision LLM OCR Pipeline** utilizing `meta-llama/llama-4-scout-17b-16e-instruct` (via Groq API) to accurately extract lab values, patient info, and test results into structured JSON format.

---

## ⏳ Phase 2: Longitudinal Disease Progression Modeling

Phase 1 provided a single-snapshot prediction. It could not answer critical questions like: *Will CN become MCI? Will MCI become AD? What is the trajectory over the years?*

Phase 2 shifts the paradigm from **Diagnosis** to **Disease Progression Prediction** using longitudinal patient history.

### Data Engineering & Temporal Sequencing
*   **Dataset:** ADNI Longitudinal Dataset (~22,000 records, 3,034 patients with multiple visits).
*   **Timeline Conversion:** Cleaned and converted visits (e.g., `bl`, `m06`, `m12`, `m24`) into a continuous month-based time representation.
*   **Sliding Window Sequences:** Grouped visits chronologically by `subject_id` to build sequential training samples with a sliding window. 
    *   *Input: `[Visit 1, Visit 2, Visit 3]` → Output: `Visit 4 Diagnosis`*

### Deep Learning Temporal Models
We evaluated multiple time-series networks (including GRU models) and finalized a **Bidirectional LSTM** to capture long-term temporal dependencies from past clinical scores (MMSCORE, CDGLOBAL, TOTSCORE, etc.).

**Final Deployed Architecture:**
`Bidirectional LSTM → Batch Normalization → Dense Layer (ReLU) → Dropout → Softmax Output (CN=0, MCI=1, AD=2)`

> **Research Contribution:** Moving beyond typical current-state diagnosis modeling, our system maps **patient-specific disease trajectory**, aligning with state-of-the-art Digital Twin research and Healthcare AI.

---

## 🚀 How to Run Locally

This project consists of a **Next.js** frontend (`client`) and a **FastAPI** backend (`server`).

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Alzheimer-s-mulimodal-disease-prediction.git
cd Alzheimer-s-mulimodal-disease-prediction
```

### 2. Run the FastAPI Server (Backend)
Open a terminal and navigate to the `server` directory:

```bash
cd server

# (Optional but recommended) Create and activate a Python virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On macOS/Linux

# Install requirements
pip install fastapi uvicorn pydantic python-multipart python-dotenv pandas numpy scikit-learn catboost xgboost fastai tensorflow joblib groq pillow

# Run the FastAPI server
uvicorn app.main:app --reload
```
*Note: Make sure to set up your `.env` file in the `server` folder with your `GROQ_API_KEY` for the OCR pipeline.*

### 3. Run the Next.js Client (Frontend)
Open another terminal and navigate to the `client` directory:

```bash
cd client

# Install Node dependencies
npm install

# Start the frontend development server
npm run dev
```

The web interface will start automatically at `http://localhost:3000`, connecting to your local AI models hosted by FastAPI on port 8000.

---

## 🛠️ Tech Stack

<div align="center">

| Category | Technologies |
|----------|-------------|
| **Frontend UI** | Next.js 16 · React 19 · Tailwind CSS v4 · Zustand · Framer Motion |
| **Backend API** | FastAPI (Python) |
| **Deep Learning** | FastAI / PyTorch (ResNet34) · TensorFlow / Keras (Bidirectional LSTM) |
| **Machine Learning** | CatBoost · XGBoost · Random Forest |
| **OCR / Vision Model** | Llama 4 Scout (via Groq API) |
| **Data Processing** | Pandas · NumPy · Scikit-learn · Joblib |

</div>

---

## 👥 Authors

> 🎓 **Sardar Patel Institute of Technology** — Major Project

| Name | Role |
|------|------|
| **Ria Talsania** | Co-Author |
| **Mihit Singasane** | Co-Author |
| **Ved Thakker** | Co-Author |
| **Prof. Swapnali Kurhade** | Project Guide |

---

## ⚠️ Disclaimer

> This project is for **research and educational purposes only** and is **not a certified medical diagnostic tool**.  
> All clinical decisions must be made by qualified healthcare professionals.
