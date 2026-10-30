import json
import random
import numpy as np

# Load the eligible subjects from previous script logic
from check_eligibility import trajectories, eligible_subjects

print("=== SEEDED RANDOM DRAW (SEED 42) ===")

# Sort lists by subject_id before draw for deterministic ordering across systems
for k in trajectories:
    trajectories[k].sort(key=lambda x: x[0])

rng = random.Random(42)

# Type A: stable CN
cand_A = [x[0] for x in trajectories["A_stable_CN"]]
chosen_A = rng.choice(cand_A)

# Type B: CN converting to MCI
cand_B = [x[0] for x in trajectories["B_CN_to_MCI"]]
chosen_B = rng.choice(cand_B)

# Type C: MCI converting to AD
cand_C = [x[0] for x in trajectories["C_MCI_to_AD"]]
chosen_C = rng.choice(cand_C)

# Type D: stable MCI (let's use stable MCI, which is the classic stable amnestic MCI phenotype)
cand_D = [x[0] for x in trajectories["D_stable_MCI"]]
chosen_D = rng.choice(cand_D)

chosen = {
    "A (Stable CN)": (chosen_A, "A_stable_CN", len(cand_A)),
    "B (CN -> MCI)": (chosen_B, "B_CN_to_MCI", len(cand_B)),
    "C (MCI -> AD)": (chosen_C, "C_MCI_to_AD", len(cand_C)),
    "D (Stable MCI)": (chosen_D, "D_stable_MCI", len(cand_D)),
}

for label, (sid, traj_key, pool_size) in chosen.items():
    print(f"\n{label}:")
    print(f"  Selected subject_id: {sid} (drawn from pool of {pool_size} subjects)")
    # Print the 4 visits
    # find subject in eligible_subjects
    v_dict = next(v for s, v in eligible_subjects if s == sid)
    for m in [0, 6, 12, 24]:
        row = v_dict[m]
        print(f"    Month {m:02d} (code {row.get('visit')}): DX={int(row['DIAGNOSIS'])}, MMSE={row['MMSE']}, CDR={row['CDGLOBAL']}, ADAS={row['TOTSCORE']}, Age={row.get('age_at_visit') or row.get('AGE')}")

