"""Inference with a cached model that hot-reloads when bean_model.keras changes.

Several API replicas share the models/ volume; when the trainer writes a new
bean_model.keras, each replica notices the new mtime on its next request and reloads.
"""
import threading

import numpy as np

from src.config import CLASSES, MODEL_PATH
from src.preprocessing import load_image

_lock = threading.Lock()
_model = None
_model_mtime = None


def get_model():
    global _model, _model_mtime
    try:
        mtime = MODEL_PATH.stat().st_mtime
    except FileNotFoundError:
        raise RuntimeError(f"No trained model at {MODEL_PATH}. Run `python -m src.model` first.")
    if _model is None or mtime != _model_mtime:
        with _lock:
            if _model is None or mtime != _model_mtime:
                import tensorflow as tf

                _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                _model_mtime = mtime
    return _model


def predict_array(arr):
    """arr: (224,224,3) or (N,224,224,3) raw 0..255 pixels. Returns list of result dicts."""
    batch = arr[None] if arr.ndim == 3 else arr
    probs = get_model()(batch, training=False).numpy()
    return [_to_result(p) for p in probs]


def _to_result(p):
    idx = int(np.argmax(p))
    return {
        "label": CLASSES[idx],
        "confidence": float(p[idx]),
        "probabilities": {c: float(v) for c, v in zip(CLASSES, p)},
    }


def predict(source):
    """Predict a single image given a path, bytes or file-like object."""
    return predict_array(load_image(source))[0]


if __name__ == "__main__":
    import sys

    from src.config import TEST_DIR

    path = sys.argv[1] if len(sys.argv) > 1 else next((TEST_DIR / CLASSES[1]).glob("*.jpg"))
    print(path, predict(path))
