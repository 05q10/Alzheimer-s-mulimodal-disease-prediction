import os
import sys
import asyncio
import httpx

# Set environment before any imports
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["MPLCONFIGDIR"] = os.path.abspath("server/scratch/mpl_cache")
os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "dummy_key_for_test")

from pathlib import Path
server_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(server_dir))

print("[STEP 1] Importing app...", flush=True)
from app.main import app
print("[STEP 2] Imported app. Starting httpx test...", flush=True)

async def run_tests():
    print("Testing API routes via ASGI Transport...")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. GET /demo/cases
        res = await client.get("/demo/cases")
        print("GET /demo/cases:", res.status_code, "Count:", len(res.json()))
        assert res.status_code == 200
        assert len(res.json()) >= 4

        # 2. POST /demo/replay/case_a_stable_cn/start
        res = await client.post("/demo/replay/case_a_stable_cn/start", params={"include_mri": True})
        print("POST /demo/replay/case_a_stable_cn/start:", res.status_code)
        assert res.status_code == 200
        data = res.json()
        twin_id = data["twin"]["twin_id"]
        assert twin_id is not None
        assert data["replay"]["current_visit_index"] == 0
        assert data["replay"]["pending_pre_visit_prediction"]["target_month"] == 6

        # 3. POST /demo/replay/case_a_stable_cn/next
        res = await client.post("/demo/replay/case_a_stable_cn/next", params={"twin_id": twin_id})
        print("POST /demo/replay/case_a_stable_cn/next (visit 1):", res.status_code)
        assert res.status_code == 200
        eval1 = res.json()["evaluation"]
        assert eval1["visit_index"] == 1
        assert eval1["actual_scores"]["MMSE"] == 29.0

        # Advance visit 2 and 3
        res = await client.post("/demo/replay/case_a_stable_cn/next", params={"twin_id": twin_id})
        assert res.status_code == 200
        res = await client.post("/demo/replay/case_a_stable_cn/next", params={"twin_id": twin_id})
        assert res.status_code == 200

        # 4. GET /demo/replay/case_a_stable_cn/report
        res = await client.get("/demo/replay/case_a_stable_cn/report", params={"twin_id": twin_id})
        print("GET /demo/replay/case_a_stable_cn/report:", res.status_code)
        assert res.status_code == 200
        rep = res.json()
        assert rep["is_complete"] is True
        assert len(rep["visits"]) == 4

        # 5. POST /demo/replay/case_a_stable_cn/reset
        res = await client.post("/demo/replay/case_a_stable_cn/reset", params={"twin_id": twin_id})
        print("POST /demo/replay/case_a_stable_cn/reset:", res.status_code)
        assert res.status_code == 200
        rst = res.json()
        assert rst["replay"]["current_visit_index"] == 0

        # 6. GET /demo/replay/summary
        res = await client.get("/demo/replay/summary")
        print("GET /demo/replay/summary:", res.status_code)
        assert res.status_code == 200
        summ = res.json()
        assert "evidence_statement" in summ
        assert len(summ["replay_patients"]) >= 4

        # 7. Old routes check: GET /demo/patients
        res = await client.get("/demo/patients")
        print("GET /demo/patients:", res.status_code, "Count:", len(res.json()))
        assert res.status_code == 200

        # 8. Old routes check: GET /model-card
        res = await client.get("/model-card")
        print("GET /model-card:", res.status_code)
        assert res.status_code == 200

    print("ALL API ROUTE CHECKS PASSED!")

if __name__ == "__main__":
    asyncio.run(run_tests())
