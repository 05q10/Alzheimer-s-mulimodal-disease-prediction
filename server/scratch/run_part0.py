import os
import sys
from pathlib import Path

server_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(server_dir))

cache_dir = server_dir / "scratch" / "mpl_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(cache_dir)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["DEMO_MRI_ROOT"] = str(server_dir / "data" / "archive (1)" / "combined_images")

print("=== REPRODUCING PART 0 ===", flush=True)

from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)
print("Client ready.", flush=True)

# 1. GET /demo/patients
print("\n[1] Testing GET /demo/patients:", flush=True)
res_patients = client.get("/demo/patients")
print(f"Status: {res_patients.status_code}", flush=True)
if res_patients.status_code == 200:
    data = res_patients.json()
    print(f"Returned {len(data)} patients. Sample twin_id: {data[0]['twin_id']}", flush=True)
    twin_id = data[0]['twin_id']
else:
    print(f"Error: {res_patients.text}", flush=True)
    twin_id = None

# 2. GET /demo/cases
print("\n[2] Testing GET /demo/cases:", flush=True)
res_cases = client.get("/demo/cases")
print(f"Status: {res_cases.status_code}", flush=True)
print(f"Body: {res_cases.text}", flush=True)

# 3. Create twin from demo persona
print("\n[3] Testing create twin from persona:", flush=True)
create_res = client.post("/twin", json={
    "demographics": {"age": 71.0, "sex": 1, "education_years": 16.0, "apoe4_count": 0},
    "scores": {"MMSE": 29.0, "CDGLOBAL": 0.0, "TOTSCORE": 7.0},
    "persona_label": "Eleanor Vance"
})
print(f"Status: {create_res.status_code}", flush=True)
if create_res.status_code == 200:
    twin_id = create_res.json()["twin_id"]
    print(f"Created twin_id: {twin_id}", flush=True)

# 4. Open it: GET /twin/{twin_id}
print(f"\n[4] Testing GET /twin/{twin_id}:", flush=True)
res_get = client.get(f"/twin/{twin_id}")
print(f"Status: {res_get.status_code}", flush=True)
if res_get.status_code == 200:
    t_data = res_get.json()
    print(f"Twin keys: {list(t_data.keys())}", flush=True)
    print(f"Twin state keys: {list(t_data['state'].keys())}", flush=True)
    print(f"Twin state: stage={t_data['state'].get('stage_probs')}, tier={t_data['state'].get('tier_used')}", flush=True)

# 5. Upload MRI from archive
print("\n[5] Testing Upload MRI:", flush=True)
demo_mri_root = Path(os.environ["DEMO_MRI_ROOT"])
sample_img = None
for folder in ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]:
    sub = demo_mri_root / folder
    if sub.exists():
        imgs = list(sub.glob("*.jpg"))
        if imgs:
            sample_img = imgs[0]
            print(f"Found sample image: {sample_img}", flush=True)
            break

if sample_img:
    with open(sample_img, "rb") as f:
        mri_bytes = f.read()
    print(f"Sending POST /twin/{twin_id}/mri-image with {len(mri_bytes)} bytes...", flush=True)
    try:
        upload_res = client.post(f"/twin/{twin_id}/mri-image", files={"file": (sample_img.name, mri_bytes, "image/jpeg")})
        print(f"Status: {upload_res.status_code}", flush=True)
        print(f"Body: {upload_res.text[:300]}", flush=True)
    except Exception as e:
        print(f"Exception during upload: {type(e).__name__}: {e}", flush=True)

print("\n=== REPRODUCTION COMPLETE ===", flush=True)
