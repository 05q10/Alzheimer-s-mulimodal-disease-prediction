import sys
import time
from pathlib import Path

server_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(server_dir))

def test_mod(name, code):
    t0 = time.time()
    print(f"Testing {name}...", end=" ", flush=True)
    try:
        exec(code, globals())
        print(f"OK ({time.time()-t0:.2f}s)")
    except Exception as e:
        print(f"FAILED ({time.time()-t0:.2f}s): {e}")

test_mod("image_model", "from ml.image_model.final_run import predict_image")
test_mod("alele_model", "from ml.alele_model.final_run import predict_patient_risk")
test_mod("base_model", "from ml.base_model.using_model import predict_alzheimer, sample_patient")
test_mod("ocr", "from ml.ocr.final_run import MedicalBloodReportExtractor; extractor = MedicalBloodReportExtractor()")
test_mod("lstm", "from ml.lstm_model.predict import predict_diagnosis")

