import os
import hashlib
import pandas as pd

def hash_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

data_dir = '../data'
files = [
    'ADNIMERGE_13Aug2025.csv',
    'ADSP_PHC_T1_FS_15Aug2025.csv',
    'All_Subjects_APOERES_fixed_simple.csv',
    'alzheimers_disease_data.csv'
]

results = []

print("Performing data provenance audit...")

for fname in files:
    path = os.path.join(data_dir, fname)
    exists = os.path.exists(path)
    if exists:
        try:
            df = pd.read_csv(path, low_memory=False)
            rows = len(df)
            file_hash = hash_file(path)
        except Exception as e:
            rows = "Error"
            file_hash = str(e)
    else:
        rows = "N/A"
        file_hash = "N/A"
        
    results.append({
        "File": fname,
        "Exists": exists,
        "Rows": rows,
        "SHA256": file_hash
    })

# The user requested table per experiment, but data files are shared across them.
# The table structure requested:
# | Experiment | Data source claimed | Data source actually used | Real/Synthetic (per Task 0) | File exists? | Row count | SHA256 |

experiments = [
    {"exp": "Exp 1", "claimed": "ADNIMERGE, ADSP, APOE", "used": "ADNIMERGE, ADSP, APOE", "real": "Real (No synthetic markers)"},
    {"exp": "Exp 2", "claimed": "ADNIMERGE, ADSP, APOE", "used": "ADNIMERGE, ADSP, APOE", "real": "Real (No synthetic markers)"},
    {"exp": "Exp 3", "claimed": "alzheimers_disease_data.csv, ADNIMERGE, ADSP, APOE", "used": "alzheimers_disease_data.csv, ADNIMERGE, ADSP, APOE", "real": "Real (small fallback in nb was fake)"},
    {"exp": "Exp 4", "claimed": "ADNIMERGE, ADSP, APOE", "used": "ADNIMERGE, ADSP, APOE", "real": "Real (synthetic just in title)"},
    {"exp": "Exp 5", "claimed": "alzheimers_disease_data.csv (expected), ADNIMERGE, ADSP, APOE", "used": "np.random.randn, ADNIMERGE, ADSP, APOE", "real": "Synthetic background for clinical model SHAP"},
    {"exp": "Exp 6", "claimed": "alzheimers_disease_data.csv, ADNIMERGE, ADSP, APOE", "used": "alzheimers_disease_data.csv, ADNIMERGE, ADSP, APOE", "real": "Real (np.random seed used harmlessly)"},
    {"exp": "Exp 7", "claimed": "07_ocr_evaluation.csv", "used": "07_ocr_evaluation.csv", "real": "Real per-field data from CSV (originally synthetic dict)"},
    {"exp": "Exp 8", "claimed": "ADNIMERGE, ADSP, APOE", "used": "ADNIMERGE, ADSP, APOE, np.random.uniform", "real": "Synthetic-tainted (clinical risk simulated)"},
]

with open('results/_audit/data_provenance.md', 'w') as f:
    f.write("# Data Provenance Audit\n\n")
    f.write("| Experiment | Data source claimed | Data source actually used | Real/Synthetic (per Task 0) | File exists? | Row count | SHA256 |\n")
    f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
    
    for exp in experiments:
        # Build existence, rows, hashes based on 'used'
        used_files = [res for res in results if res['File'] in exp['used']]
        if not used_files and '07_ocr' in exp['used']:
             exists_str = str(os.path.exists('results/07_ocr_evaluation.csv'))
             try:
                 df_ocr = pd.read_csv('results/07_ocr_evaluation.csv')
                 rows_str = str(len(df_ocr))
                 hash_str = hash_file('results/07_ocr_evaluation.csv')
             except:
                 rows_str, hash_str = "Error", "Error"
        else:
             exists_str = ", ".join([str(r['Exists']) for r in used_files])
             rows_str = ", ".join([str(r['Rows']) for r in used_files])
             hash_str = "<br>".join([r['SHA256'][:16]+"..." for r in used_files])
        
        f.write(f"| {exp['exp']} | {exp['claimed']} | {exp['used']} | {exp['real']} | {exists_str} | {rows_str} | {hash_str} |\n")

print("Done. Wrote results/_audit/data_provenance.md")
