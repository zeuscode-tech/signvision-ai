from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("SIGNVISION_DATA_DIR", ROOT / "data"))
MODEL_DIR = Path(os.getenv("SIGNVISION_MODEL_DIR", ROOT / "models"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_URL = os.getenv("SIGNVISION_DATABASE_URL", f"sqlite:///{DATA_DIR / 'signvision.db'}")
MODEL_PATH = MODEL_DIR / "static_gestures.joblib"
BUKVA_MODEL_PATH = MODEL_DIR / "bukva_mobilenetv2_tsm.onnx"

