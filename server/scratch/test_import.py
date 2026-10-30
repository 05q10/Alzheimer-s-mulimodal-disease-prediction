import time
print("Starting import test...")
t0 = time.time()
print("Importing app.main...")
from app.main import app
print(f"Imported app.main in {time.time() - t0:.2f}s")

