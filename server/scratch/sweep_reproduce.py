import json, sys
from pathlib import Path

# Add server root to sys.path
server_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(server_root))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def main():
    print("=== SWEEP REPRODUCE ===")
    # Create baseline MCI patient
    create_payload = {
        "demographics": {"age": 72.0, "sex": 1, "education_years": 16.0, "apoe4_count": 1},
        "scores": {"MMSE": 24.0, "CDGLOBAL": 0.5, "TOTSCORE": 18.0, "FAQ": 6.0},
        "lifestyle": {
            "alcohol": 2,
            "physical_activity": 4,
            "diet_quality": 7,
            "sleep_quality": 7,
            "smoking": 0,
            "hypertension": 0,
            "diabetes": 0,
            "depression": 0,
            "head_injury": 0,
            "cardiovascular_disease": 0
        }
    }
    r = client.post("/twin", json=create_payload)
    if r.status_code != 200:
        print("Create failed:", r.status_code, r.text)
        return
    twin_id = r.json()["twin_id"]
    print("Created twin:", twin_id)

    # 1. Sweep cognitive scores (worsening: MMSE from 28 down to 18, CDR 0 to 1, ADAS 10 to 35)
    print("\n--- SWEEP COGNITIVE SCORES ---")
    for mmse, cdr, adas in [(28.0, 0.0, 10.0), (24.0, 0.5, 18.0), (20.0, 1.0, 26.0), (16.0, 1.0, 35.0)]:
        scen = client.post(f"/twin/{twin_id}/scenario", json={
            "horizon_months": 12,
            "hypothetical_visit": {"MMSE": mmse, "CDGLOBAL": cdr, "TOTSCORE": adas}
        }).json()
        print(f"MMSE={mmse}, CDR={cdr}, ADAS={adas} ->")
        print(f"   baseline stage_probs: {scen['baseline']['stage_probs']}")
        print(f"   scenario stage_probs: {scen['scenario']['stage_probs']}")
        print(f"   baseline conv_risk: {scen['baseline']['conversion_risk']}")
        print(f"   scenario conv_risk: {scen['scenario']['conversion_risk']}")
        if "12" in scen['scenario']['forecast']:
            fc = scen['scenario']['forecast']['12']
            print(f"   forecast 12m: MMSE={fc.get('MMSE_mean')}, CDR={fc.get('CDGLOBAL_mean')}, ADAS={fc.get('TOTSCORE_mean')}, conv_prob={fc.get('conversion_prob')}, stage_probs={fc.get('stage_probs')}")

    # 2. Sweep decline multiplier
    print("\n--- SWEEP DECLINE MULTIPLIER ---")
    for mult in [0.5, 0.8, 1.0, 1.2, 1.5]:
        scen = client.post(f"/twin/{twin_id}/scenario", json={
            "horizon_months": 12,
            "decline_multiplier": mult
        }).json()
        fc = scen['scenario']['forecast'].get('12', {})
        print(f"Mult={mult} -> scenario stage_probs: {scen['scenario']['stage_probs']}, conv_risk: {scen['scenario']['conversion_risk']}, fc MMSE={fc.get('MMSE_mean')}, fc conv_prob={fc.get('conversion_prob')}, fc stage_probs={fc.get('stage_probs')}")

    # 3. Check Mild AD (MMSE 20) forecast at 12m
    print("\n--- CHECK MILD AD (MMSE 20) ---")
    ad_payload = {
        "demographics": {"age": 75.0, "sex": 0, "education_years": 14.0, "apoe4_count": 1},
        "scores": {"MMSE": 20.0, "CDGLOBAL": 1.0, "TOTSCORE": 25.0, "FAQ": 15.0},
    }
    r_ad = client.post("/twin", json=ad_payload)
    ad_id = r_ad.json()["twin_id"]
    scen_ad = client.post(f"/twin/{ad_id}/scenario", json={"horizon_months": 12}).json()
    fc_ad = scen_ad['scenario']['forecast']['12']
    print(f"Mild AD MMSE 20 forecast at 12m: {fc_ad.get('MMSE_mean')}, drop = {20.0 - fc_ad.get('MMSE_mean', 0)}")

if __name__ == "__main__":
    main()
