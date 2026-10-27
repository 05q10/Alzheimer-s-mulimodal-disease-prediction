import pandas as pd
import json
import os

df = pd.read_csv('results/07_ocr_evaluation.csv')
confusion = df['Status'].value_counts().to_dict()
total = len(df)

output = {}
for k, v in confusion.items():
    output[k] = {
        "count": int(v),
        "percentage": round((v / total) * 100, 1)
    }

with open('results/07_ocr_confusion.json', 'w') as f:
    json.dump(output, f, indent=4)

print("--- OCR Confusion Matrix ---")
for k, v in output.items():
    print(f"{k}: {v['count']} ({v['percentage']}%)")
