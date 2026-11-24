"""Point the app at throwaway data and model dirs *before* src.config is imported.

Retraining rewrites models/head.npz, so tests run against a temporary copy of the serving
artifacts and an empty data dir (with its own SQLite database).
"""
import os
import shutil
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_TMP = Path(tempfile.mkdtemp(prefix="bean_test_"))
_MODELS = _TMP / "models"
_MODELS.mkdir()
for name in ("backbone.tflite", "head.npz", "features_train.npz", "features_test.npz", "serving_meta.json"):
    shutil.copy2(ROOT / "models" / name, _MODELS / name)

os.environ["DATA_DIR"] = str(_TMP / "data")
os.environ["MODELS_DIR"] = str(_MODELS)
os.environ["RETRAIN_THRESHOLD"] = "1000"  # keep auto-retrain out of the upload tests
os.environ["ENABLE_TRAINING"] = "true"


@pytest.fixture(scope="session", autouse=True)
def _cleanup_tmp():
    yield
    shutil.rmtree(_TMP, ignore_errors=True)
