"""Shared paths and constants. All paths are relative to the repo root."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"

MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
MODEL_PATH = MODELS_DIR / "bean_model.keras"
H5_PATH = MODELS_DIR / "bean_model.h5"
META_PATH = MODELS_DIR / "bean_model_meta.json"

CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]
IMG_SIZE = (224, 224)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SEED = 42
