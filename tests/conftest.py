"""Point the app at a throwaway data dir *before* src.config is imported."""
import os
import shutil
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="bean_test_data_"))
os.environ["DATA_DIR"] = str(_TMP)
os.environ["RETRAIN_THRESHOLD"] = "1000"  # keep auto-retrain out of the upload tests
os.environ["ENABLE_TRAINING"] = "true"


@pytest.fixture(scope="session", autouse=True)
def _cleanup_tmp():
    yield
    shutil.rmtree(_TMP, ignore_errors=True)
