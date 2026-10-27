import pandas as pd

try:
    print("Loading data...")
    adni = pd.read_csv("D:/family/MiniProject/Allele Prediction/server/data/ADNIMERGE_13Aug2025.csv", low_memory=False)
    adsp = pd.read_csv("D:/family/MiniProject/Allele Prediction/server/data/ADSP_PHC_T1_FS_15Aug2025.csv", low_memory=False)
    apoe = pd.read_csv("D:/family/MiniProject/Allele Prediction/server/data/All_Subjects_APOERES_fixed_simple.csv", low_memory=False)

    print("\n--- ADNIMERGE_13Aug2025.csv ---")
    print("Shape:", adni.shape)
    # print("Columns:", adni.columns.tolist())
    print("Potential merge keys:", [c for c in adni.columns if c in ['RID', 'PTID', 'Subject']])

    print("\n--- ADSP_PHC_T1_FS_15Aug2025.csv ---")
    print("Shape:", adsp.shape)
    # print("Columns:", adsp.columns.tolist())
    print("Potential merge keys:", [c for c in adsp.columns if c in ['RID', 'PTID', 'Subject', 'SUBJID']])

    print("\n--- All_Subjects_APOERES_fixed_simple.csv ---")
    print("Shape:", apoe.shape)
    # print("Columns:", apoe.columns.tolist())
    print("Potential merge keys:", [c for c in apoe.columns if c in ['RID', 'PTID', 'Subject']])

    print("\n--- Join Coverage Check ---")
    if 'RID' in adni.columns and 'RID' in apoe.columns:
        apoe_match = adni['RID'].isin(apoe['RID']).mean()
        print(f"ADNIMERGE matching APOE on RID: {apoe_match:.1%}")
    if 'RID' in adni.columns and 'RID' in adsp.columns:
        adsp_match = adni['RID'].isin(adsp['RID']).mean()
        print(f"ADNIMERGE matching ADSP on RID: {adsp_match:.1%}")

except Exception as e:
    print("Error:", e)
