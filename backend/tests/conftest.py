import os
from pathlib import Path
import sys

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("SIGNVISION_DATA_DIR", str(BACKEND / ".test-data"))
os.environ.setdefault("SIGNVISION_MODEL_DIR", str(BACKEND / ".test-models"))

