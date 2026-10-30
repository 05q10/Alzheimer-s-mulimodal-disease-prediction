# Re-exporting scripts/build_demo_replay.py
from pathlib import Path
import sys

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))
from scripts.build_demo_replay import build_replay_dataset

if __name__ == "__main__":
    build_replay_dataset()

