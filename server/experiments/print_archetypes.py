import sys
from pathlib import Path
SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_DIR))

from ml.twin.stager import get_stager
from ml.twin.progression import get_progression

st = get_stager()
pr = get_progression()

archetypes = [
    ("CN", {"entry_age": 72, "MMSE": 29.0, "CDGLOBAL": 0.0, "TOTSCORE": 5.0, "FAQTOTAL": 0.0}),
    ("MCI", {"MMSE": 27.0, "CDGLOBAL": 0.5, "TOTSCORE": 10.0, "FAQTOTAL": 3.0}),
    ("Mild AD", {"MMSE": 20.0, "CDGLOBAL": 1.0, "TOTSCORE": 25.0, "FAQTOTAL": 15.0}),
    ("Moderate AD", {"MMSE": 16.0, "CDGLOBAL": 1.0, "TOTSCORE": 35.0}),
]

print("| Archetype | Top Stage | P(CN) | P(MCI) | P(AD) | 6m MMSE (lo-hi) | 12m MMSE (lo-hi) | 24m MMSE (lo-hi) | Reliability | Scope Warnings |")
print("|---|---|---|---|---|---|---|---|---|---|")

for name, inp in archetypes:
    s = st.predict_one(inp)
    fc = pr.predict_forecast(inp, s["probs"])
    p = s["probs"]
    fc6 = f"{fc['6']['MMSE_mean']} [{fc['6']['MMSE_lo']}-{fc['6']['MMSE_hi']}]"
    fc12 = f"{fc['12']['MMSE_mean']} [{fc['12']['MMSE_lo']}-{fc['12']['MMSE_hi']}]"
    fc24 = f"{fc['24']['MMSE_mean']} [{fc['24']['MMSE_lo']}-{fc['24']['MMSE_hi']}]"
    rel = fc["forecast_reliability"]
    warns = "; ".join(fc["scope_warnings"]) if fc["scope_warnings"] else "None"
    print(f"| {name} | {s['stage']} | {p['CN']:.3f} | {p['MCI']:.3f} | {p['AD']:.3f} | {fc6} | {fc12} | {fc24} | {rel} | {warns} |")
