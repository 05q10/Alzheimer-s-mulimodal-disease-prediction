import os
import sys
from pathlib import Path

# Set cache and environment variables before any imports
server_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(server_dir))

cache_dir = server_dir / "scratch" / "mpl_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(cache_dir)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["DEMO_MRI_ROOT"] = str(server_dir / "data" / "archive (1)" / "combined_images")

print("=== PART 0: REPRODUCE FIRST ===")
import time
t_start = time.time()

print("\nImporting FastAPI app...")
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
print(f"App initialized in {time.time() - t_start:.2f}s")

# 1. Call GET /demo/patients
print("\n--- Step 1: Call GET /demo/patients ---")
t0 = time.time()
res_patients = client.get("/demo/patients")
print(f"GET /demo/patients -> Status: {res_patients.status_code} in {time.time()-t0:.2f}s")
if res_patients.status_code == 200:
    patients_data = res_patients.json()
    print(f"Returned {len(patients_data)} patients. First 2 aliases/labels:")
    for p in patients_data[:2]:
        print(f"  - twin_id: {p.get('twin_id')} | persona: {p.get('persona_label')}")
else:
    print(f"Error: {res_patients.text}")

# 2. Call GET /demo/cases
print("\n--- Step 2: Call GET /demo/cases ---")
t0 = time.time()
res_cases = client.get("/demo/cases")
print(f"GET /demo/cases -> Status: {res_cases.status_code} in {time.time()-t0:.2f}s")
print(f"Response: {res_cases.text}")

# 3. Create a twin from a demo persona
print("\n--- Step 3: Create a twin from a demo persona ---")
twin_id = None
if res_patients.status_code == 200 and patients_data:
    twin_id = patients_data[0].get("twin_id")
    print(f"Using pre-initialized demo twin_id: {twin_id}")
else:
    print("Could not get twin_id from /demo/patients")

# 4. Open it: GET /twin/{twin_id}
print("\n--- Step 4: Open it: GET /twin/{twin_id} ---")
if twin_id:
    t0 = time.time()
    res_open = client.get(f"/twin/{twin_id}")
    print(f"GET /twin/{twin_id} -> Status: {res_open.status_code} in {time.time()-t0:.2f}s")
    if res_open.status_code == 200:
        twin_data = res_open.json()
        print(f"Twin open successful:")
        print(f"  - twin_id: {twin_data.get('twin_id')}")
        print(f"  - baseline_stage: {twin_data.get('baseline_stage')}")
        print(f"  - current stage_probs: {twin_data.get('state', {}).get('stage_probs')}")
        print(f"  - tier_used: {twin_data.get('state', {}).get('tier_used')}")
        print(f"  - visits count: {len(twin_data.get('visits', []))}")
    else:
        print(f"Error opening twin: {res_open.text}")

# 5. Upload an MRI from the archive folder
print("\n--- Step 5: Upload an MRI from archive folder ---")
demo_mri_root = Path(os.environ["DEMO_MRI_ROOT"])
print(f"DEMO_MRI_ROOT: {demo_mri_root}")
print(f"Exists: {demo_mri_root.exists()}")

mri_file = None
for folder in ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]:
    sub = demo_mri_root / folder
    if sub.exists():
        imgs = list(sub.glob("*.jpg")) + list(sub.glob("*.png"))
        if imgs:
            mri_file = imgs[0]
            print(f"Found MRI image in {folder}: {mri_file.name} (size: {mri_file.stat().st_size} bytes)")
            break

if mri_file and twin_id:
    t0 = time.time()
    with open(mri_file, "rb") as f:
        file_bytes = f.read()
    files = {"file": (mri_file.name, file_bytes, "image/jpeg")}
    upload_res = client.post(f"/twin/{twin_id}/mri-image", files=files)
    print(f"POST /twin/{twin_id}/mri-image -> Status: {upload_res.status_code} in {time.time()-t0:.2f}s")
    print(f"Response: {upload_res.text[:400]}")
else:
    print(f"Cannot upload MRI: mri_file={mri_file}, twin_id={twin_id}")

print(f"\nTotal reproduction runtime: {time.time() - t_start:.2f}s")
