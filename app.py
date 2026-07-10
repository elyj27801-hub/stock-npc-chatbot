import os
import runpy
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ["NPC_APP_MODE"] = "stock"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

runpy.run_path(str(ROOT_DIR / "app.py"), run_name="__main__")
