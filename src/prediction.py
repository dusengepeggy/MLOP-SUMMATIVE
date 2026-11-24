"""Inference with a cached model that hot-reloads when bean_model.keras changes.

Several API replicas share the models/ volume; when the trainer writes a new
bean_model.keras, each replica notices the new mtime on its next request and reloads.

Inference is serialised per process. Under load, FastAPI's threadpool would otherwise run
dozens of TensorFlow calls at once, each spawning threads for every host core, which
thrashes a CPU-limited container and blows its memory limit (seen as OOM kills during
load tests). Requests queue cheaply instead, and throughput scales by adding replicas.
Set INFERENCE_THREADS to the container's CPU allocation to size TF's thread pools.
"""
import os
import threading

import numpy as np

from src.config import CLASSES, MODEL_PATH
from src.preprocessing import load_image

_lock = threading.Lock()  # guards (re)loading
_infer_lock = threading.Lock()  # one inference at a time per process
_model = None
_model_mtime = None
_threads_configured = False


def _configure_threads(tf):
    global _threads_configured
    if _threads_configured:
        return
    n = int(os.getenv("INFERENCE_THREADS", "0"))
    if n > 0:
        try:
            tf.config.threading.set_intra_op_parallelism_threads(n)
            tf.config.threading.set_inter_op_parallelism_threads(n)
        except RuntimeError:  # TF runtime already initialised; keep defaults
            pass
    _threads_configured = True


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

                _configure_threads(tf)
                _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                _model_mtime = mtime
    return _model


def predict_array(arr):
    """arr: (224,224,3) or (N,224,224,3) raw 0..255 pixels. Returns list of result dicts."""
    batch = arr[None] if arr.ndim == 3 else arr
    model = get_model()
    with _infer_lock:
        probs = model(batch, training=False).numpy()
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
