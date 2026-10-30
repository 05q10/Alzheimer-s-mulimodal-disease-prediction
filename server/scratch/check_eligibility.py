import json
import pandas as pd
import numpy as np

with open('server/data/splits.json') as f:
    splits = json.load(f)

test_subjects = set(splits['test'])
train_subjects = set(splits['train'])
val_subjects = set(splits['val'])
print(f"Train: {len(train_subjects)}, Val: {len(val_subjects)}, Test: {len(test_subjects)}")

# Verify splits are mutually exclusive
assert len(test_subjects.intersection(train_subjects)) == 0, "Test and Train overlap!"
assert len(test_subjects.intersection(val_subjects)) == 0, "Test and Val overlap!"
print("Assertion verified: Splits are strictly mutually exclusive.")

df = pd.read_parquet('server/data/unified_visits.parquet')
print("Total rows in unified_visits.parquet:", len(df))

test_df = df[df['subject_id'].isin(test_subjects)].copy()
print("Test rows:", len(test_df), "Unique test subjects:", test_df['subject_id'].nunique())

# Check columns for: MMSE, CDR-global, ADAS-Cog-11, sex, education, APOE
# In unified_visits:
# MMSE: 'MMSE' or 'MMSCORE'
# CDR-global: 'CDGLOBAL'
# ADAS-Cog-11: 'TOTSCORE'
# sex: 'PTGENDER' or 'sex'
# education: 'PTEDUCAT' or 'education_years'
# APOE: 'APOE4'

print("\nNull checks in test_df for key fields:")
for col in ['MMSE', 'MMSCORE', 'CDGLOBAL', 'TOTSCORE', 'PTGENDER', 'sex', 'PTEDUCAT', 'education_years', 'APOE4', 'DIAGNOSIS']:
    if col in test_df.columns:
        print(f"  {col}: {test_df[col].notna().sum()} / {len(test_df)} not-null")

# Map months 0, 6, 12, 24 with +/- 1 month tolerance
target_months = [0, 6, 12, 24]

# Let's find subjects who have a visit in [m-1, m+1] for all m in [0, 6, 12, 24]
# with observed MMSE, CDGLOBAL, TOTSCORE, and valid sex, education, APOE
eligible_subjects = []

for sid, grp in test_df.groupby('subject_id'):
    # Check demographics
    row0 = grp.iloc[0]
    sex_val = row0.get('sex') if pd.notna(row0.get('sex')) else row0.get('PTGENDER')
    edu_val = row0.get('education_years') if pd.notna(row0.get('education_years')) else row0.get('PTEDUCAT')
    apoe_val = row0.get('APOE4')
    
    if pd.isna(sex_val) or pd.isna(edu_val) or pd.isna(apoe_val):
        continue
        
    # Check visits at months 0, 6, 12, 24
    has_all_visits = True
    v_rows = {}
    for tm in target_months:
        candidates = grp[grp['visit_month'].between(tm - 1, tm + 1)]
        valid_cands = candidates[
            candidates['MMSE'].notna() & 
            candidates['CDGLOBAL'].notna() & 
            candidates['TOTSCORE'].notna() &
            candidates['DIAGNOSIS'].notna()
        ]
        if len(valid_cands) == 0:
            has_all_visits = False
            break
        # Pick the one closest to tm
        idx_closest = (valid_cands['visit_month'] - tm).abs().idxmin()
        v_rows[tm] = valid_cands.loc[idx_closest]
        
    if has_all_visits:
        eligible_subjects.append((sid, v_rows))

print(f"\nTotal eligible test subjects with complete data at months 0, 6, 12, 24: {len(eligible_subjects)}")

# Categorize into the 4 trajectory types:
# (A) stable CN (CN at all four visits: DIAGNOSIS == 1)
# (B) CN converting to MCI (starts CN at m0, becomes MCI at m6/12/24, stays MCI or later)
# (C) MCI converting to AD (starts MCI at m0, becomes AD at m6/12/24)
# (D) stable MCI (starts MCI at m0, stays MCI at all visits) OR slowly declining AD (starts AD at m0)

trajectories = {
    "A_stable_CN": [],
    "B_CN_to_MCI": [],
    "C_MCI_to_AD": [],
    "D_stable_MCI": [],
    "D_declining_AD": [],
}

for sid, v_dict in eligible_subjects:
    dxs = [int(v_dict[m]['DIAGNOSIS']) for m in target_months]
    # DIAGNOSIS: 1 = CN, 2 = MCI, 3 = AD
    if dxs == [1, 1, 1, 1]:
        trajectories["A_stable_CN"].append((sid, dxs))
    elif dxs[0] == 1 and any(d == 2 for d in dxs[1:]) and 3 not in dxs:
        # starts 1, converts to 2
        trajectories["B_CN_to_MCI"].append((sid, dxs))
    elif dxs[0] == 2 and any(d == 3 for d in dxs[1:]):
        # starts 2, converts to 3
        trajectories["C_MCI_to_AD"].append((sid, dxs))
    elif dxs == [2, 2, 2, 2]:
        trajectories["D_stable_MCI"].append((sid, dxs))
    elif dxs[0] == 3 and all(d == 3 for d in dxs[1:]):
        trajectories["D_declining_AD"].append((sid, dxs))
    else:
        # e.g. other patterns like CN directly to AD, or fluctuating
        pass

print("\nEligible counts by trajectory type:")
for k, v in trajectories.items():
    print(f"  {k}: {len(v)} subjects")

