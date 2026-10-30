import os, sys
sys.path.insert(0, os.path.abspath("server"))
from app.replay_engine import get_replay_summary

print("Testing get_replay_summary()...")
summary = get_replay_summary()
print("Evidence statement:", summary["evidence_statement"])
print("Population test metrics keys:", list(summary["population_test_metrics"].keys()))
print("Replay patients count:", len(summary["replay_patients"]))
for p in summary["replay_patients"]:
    print(f"{p['case_id']} ({p['alias']}): MAE MMSE={p['mae_MMSE']}, Stage agree={p['stage_agreement_rate']}, Band cov={p['band_coverage_rate']}")

