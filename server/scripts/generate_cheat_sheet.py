"""
server/scripts/generate_cheat_sheet.py
======================================
Mirror of scripts/generate_cheat_sheet.py.
"""

import sys
from pathlib import Path
root_script = Path(__file__).resolve().parent.parent.parent / "scripts" / "generate_cheat_sheet.py"
if root_script.exists():
    sys.path.insert(0, str(root_script.parent))
    from generate_cheat_sheet import run_and_generate
    if __name__ == "__main__":
        run_and_generate()

