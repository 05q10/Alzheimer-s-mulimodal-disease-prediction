"""
server/tests/test_legacy_routes.py
==================================
Tests ensuring all legacy endpoints (/alele, /image, /extract-report, /base, /lstm-predict)
respond cleanly without unhandled crashes. Status 200 or 503 (if underlying standalone model
file is absent) is acceptable; 404 or unhandled 500 is a failure.
"""

import io
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))

from app.main import app

client = TestClient(app, raise_server_exceptions=False)


def test_legacy_alele_endpoint():
    payload = {
        "ABETA": 800.0,
        "TAU": 200.0,
        "MMSE": 26.0,
        "APVOLUME": 0.5,
        "GENOTYPE": "3/4"
    }
    resp = client.post("/alele", json=payload)
    assert resp.status_code in [200, 503], f"Unexpected status {resp.status_code}: {resp.text}"


def test_legacy_image_endpoint():
    # 1x1 dummy PNG
    png_bytes = (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
        b'\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff'
        b'?\x00\x05\xfe\x02\xfe\xdc\xccY\xe7\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    files = {"file": ("test.png", png_bytes, "image/png")}
    resp = client.post("/image", files=files)
    assert resp.status_code in [200, 503], f"Unexpected status {resp.status_code}: {resp.text}"


def test_legacy_extract_report_endpoint():
    dummy_file = io.BytesIO(b"Dummy blood report content: WBC 5.4, RBC 4.5, Platelets 220")
    files = {"file": ("report.txt", dummy_file, "text/plain")}
    resp = client.post("/extract-report", files=files)
    assert resp.status_code in [200, 503], f"Unexpected status {resp.status_code}: {resp.text}"


def test_legacy_base_endpoint():
    payload = {
        "AGE": 74.0,
        "PTGENDER": 1,
        "MMSE": 26.0
    }
    resp = client.post("/base", json=payload)
    assert resp.status_code in [200, 503], f"Unexpected status {resp.status_code}: {resp.text}"


def test_legacy_lstm_predict_endpoint():
    payload = {
        "visits": [
            {
                "visit_month": 0,
                "entry_age": 74.0,
                "CDGLOBAL": 0.5,
                "MMSCORE": 26.0,
                "TOTSCORE": 12.0
            }
        ]
    }
    resp = client.post("/lstm-predict", json=payload)
    assert resp.status_code in [200, 503], f"Unexpected status {resp.status_code}: {resp.text}"
