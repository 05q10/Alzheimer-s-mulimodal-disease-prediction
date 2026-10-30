import sys
import time
from pathlib import Path

server_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(server_dir))

print("Importing initialize_demo_patients...")
t0 = time.time()
from app.demo_patients import initialize_demo_patients, DEMO_PERSONAS
print(f"Imported in {time.time() - t0:.2f}s")

print(f"Total personas: {len(DEMO_PERSONAS)}")
t1 = time.time()
res = initialize_demo_patients()
print(f"initialize_demo_patients finished in {time.time() - t1:.2f}s, returned {len(res)} entries")
for r in res:
    print(" -", r.twin_id, r.persona_label)

