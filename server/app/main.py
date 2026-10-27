from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Body
from .models.input import PatientInput
from PIL import Image
import io
from typing import List
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Legacy imports (safely guarded)
try:
    from ml.image_model.final_run import predict_image
except ImportError:
    predict_image = None

try:
    from ml.alele_model.final_run import predict_patient_risk 
except ImportError:
    predict_patient_risk = None

try:
    from ml.base_model.using_model import predict_alzheimer, sample_patient
except ImportError:
    predict_alzheimer = None
    sample_patient = {}

try:
    from ml.ocr.final_run import MedicalBloodReportExtractor
    extractor = MedicalBloodReportExtractor()
except ImportError:
    extractor = None

try:
    from ml.lstm_model.predict import predict_diagnosis
except ImportError:
    predict_diagnosis = None

app = FastAPI(title="Alzheimer's Digital Twin API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # or put "http://localhost:3000" if you want restricted access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Digital Twin routes (new in Phase 1; stubs until Phase 4) ─────────────
from app.routers.twin_router import router as twin_router  # noqa: E402
app.include_router(twin_router)


@app.post("/alele")
def predict(data: PatientInput):
    if predict_patient_risk is None:
        raise HTTPException(status_code=503, detail="Allele risk model is unavailable.")
    result = predict_patient_risk(
        ABETA=data.ABETA,
        TAU=data.TAU,
        MMSE=data.MMSE,
        APVOLUME=data.APVOLUME,
        GENOTYPE=data.GENOTYPE
    )
    return result

@app.post("/image")
async def predict_image_route(file: UploadFile = File(...)):
    if predict_image is None:
        raise HTTPException(status_code=503, detail="FastAI image model is unavailable.")
    img_bytes = await file.read()
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img = img.convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file uploaded.")
    result = predict_image(img)
    return result

@app.post("/extract-report")
async def extract_report(file: UploadFile = File(...)):
    if extractor is None:
        raise HTTPException(status_code=503, detail="OCR report extractor is unavailable.")
    bytes_data = await file.read()
    try:
        result = extractor.extract_from_file(bytes_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"OCR report service unavailable: {e}")

@app.post("/base")
async def predict(patient_data: dict = Body(...)):
    if predict_alzheimer is None:
        raise HTTPException(status_code=503, detail="Base model is unavailable.")
    patient = sample_patient.copy()
    for key, value in patient_data.items():
        if key in patient:
            try:
                patient[key] = float(value) if '.' in str(value) else int(value)
            except:
                patient[key] = value

    prediction = predict_alzheimer(patient)
    return {
        "input_used": patient,
        "prediction": prediction
    }

# ── Pydantic model for LSTM visits ────────────────────────────────────────
class VisitRecord(BaseModel):
    visit_month: int
    entry_age: float
    CDGLOBAL: float
    MMSCORE: float
    TOTSCORE: float

class LSTMPredictRequest(BaseModel):
    visits: List[VisitRecord]

@app.post("/lstm-predict")
async def lstm_predict(req: LSTMPredictRequest):
    if predict_diagnosis is None:
        raise HTTPException(status_code=503, detail="LSTM model is unavailable.")
    if len(req.visits) < 1 or len(req.visits) > 3:
        raise HTTPException(
            status_code=400,
            detail="Please provide between 1 and 3 visit records."
        )

    visits_data = [v.model_dump() for v in req.visits]
    result = predict_diagnosis(visits_data)
    return result