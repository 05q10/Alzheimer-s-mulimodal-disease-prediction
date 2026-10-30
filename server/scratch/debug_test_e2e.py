import os
import sys
import time

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["MPLCONFIGDIR"] = os.path.abspath("server/scratch/mpl_cache")
os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "dummy_key_for_test")

from pathlib import Path
server_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(server_dir))

print("[1] Importing TestClient and app...", flush=True)
t0 = time.time()
from fastapi.testclient import TestClient
from app.main import app
print(f"[1] Done in {time.time()-t0:.2f}s", flush=True)

client = TestClient(app)

print("[2] Calling GET /demo/cases...", flush=True)
t0 = time.time()
res_cases = client.get("/demo/cases")
print(f"[2] GET /demo/cases: {res_cases.status_code} in {time.time()-t0:.2f}s", flush=True)

cases = ["case_a_stable_cn", "case_b_cn_to_mci", "case_c_mci_to_ad", "case_d_stable_mci"]
for case_id in cases:
    print(f"[3] Starting {case_id}...", flush=True)
    t0 = time.time()
    res_start = client.post(f"/demo/replay/{case_id}/start?include_mri=false", json={})
    print(f"    Start status: {res_start.status_code} in {time.time()-t0:.2f}s", flush=True)
    twin_id = res_start.json()["twin"]["twin_id"]

    for v_idx in [1, 2, 3]:
        t0 = time.time()
        res_next = client.post(f"/demo/replay/{case_id}/next?twin_id={twin_id}", json={})
        print(f"    Next v{v_idx}: {res_next.status_code} in {time.time()-t0:.2f}s", flush=True)

    t0 = time.time()
    res_rep = client.get(f"/demo/replay/{case_id}/report?twin_id={twin_id}")
    print(f"    Report status: {res_rep.status_code} in {time.time()-t0:.2f}s", flush=True)

print("[4] Calling GET /demo/replay/summary...", flush=True)
t0 = time.time()
res_sum = client.get("/demo/replay/summary")
print(f"[4] GET /demo/replay/summary: {res_sum.status_code} in {time.time()-t0:.2f}s", flush=True)
print("ALL DONE!")

