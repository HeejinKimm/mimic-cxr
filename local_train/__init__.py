from pathlib import Path
import sys

PROJ_ROOT = Path(__file__).resolve().parents[1]  # mimic-cxr\mimic-cxr
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))