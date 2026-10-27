import pandas as pd
import numpy as np
import os

def load_merged_data(data_dir):
    print("Loading datasets...")
    adni = pd.read_csv(os.path.join(data_dir, "ADNIMERGE_13Aug2025.csv"), low_memory=False)
    adsp = pd.read_csv(os.path.join(data_dir, "ADSP_PHC_T1_FS_15Aug2025.csv"), low_memory=False)
    apoe = pd.read_csv(os.path.join(data_dir, "All_Subjects_APOERES_fixed_simple.csv"), low_memory=False)

    print("Merging on RID...")
    # Clean up column names just in case
    adni.columns = adni.columns.str.strip()
    adsp.columns = adsp.columns.str.strip()
    apoe.columns = apoe.columns.str.strip()

    # Determine diagnosis column in ADNI
    dx_col = 'DX' if 'DX' in adni.columns else 'DX_bl'
    print(f"Using {dx_col} as target.")

    # Drop duplicates to avoid explosion
    adni = adni.drop_duplicates(subset=['RID'])
    adsp = adsp.drop_duplicates(subset=['RID'])
    apoe = apoe.drop_duplicates(subset=['RID'])

    merged = adni.merge(apoe, on='RID', how='inner', suffixes=('', '_apoe'))
    merged = merged.merge(adsp, on='RID', how='inner', suffixes=('', '_adsp'))
    
    print(f"Merged dataset shape: {merged.shape}")
    
    # Feature engineering for base features
    if 'ABETA' in merged.columns and 'TAU' in merged.columns:
        # handle > or < in ABETA/TAU strings (e.g. ">1700")
        for col in ['ABETA', 'TAU']:
            merged[col] = pd.to_numeric(merged[col].astype(str).str.replace(r'[<>]', '', regex=True), errors='coerce')
        merged['ABETA_TAU_RATIO'] = merged['ABETA'] / merged['TAU']
    
    if 'MMSE' in merged.columns and 'TAU' in merged.columns:
        merged['MMSE_TAU_INTERACT'] = merged['MMSE'] * merged['TAU']
    
    return merged, dx_col
