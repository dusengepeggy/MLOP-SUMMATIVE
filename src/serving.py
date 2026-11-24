"""TensorFlow-free serving: LiteRT feature extractor + numpy classifier head.

The trained Keras model (models/bean_model.h5) is split by scripts/export_serving_model.py into
  * models/backbone.tflite : rescale -> fine-tuned MobileNetV2 -> global average pooling (1280 features)
  * models/head.npz        : the trained Dense(3) softmax layer (W, b)
Prediction = softmax(features @ W + b), numerically the same as the Keras model, but it runs in
~150 MB of RAM without TensorFlow, so the API fits small free hosts (e.g. Render, 512 MB).

Retraining (src/retrain_light.py) fine-tunes the head starting from its current trained weights
(warm start), using cached backbone features of the training set plus the new uploads.
"""
import json
import os
import threading

import numpy as np

from src.config import CLASSES, MODELS_DIR
from src.preprocessing import load_image

BACKBONE_PATH = MODELS_DIR / "backbone.tflite"
HEAD_PATH = MODELS_DIR / "head.npz"
SERVING_META_PATH = MODELS_DIR / "serving_meta.json"
FEATURES_TRAIN_PATH = MODELS_DIR / "features_train.npz"
FEATURES_TEST_PATH = MODELS_DIR / "features_test.npz"


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- backbone (LiteRT)
class Backbone:
    def __init__(self, path=BACKBONE_PATH, threads=None):
        from ai_edge_litert.interpreter import Interpreter

        threads = threads or int(os.getenv("INFERENCE_THREADS", "0")) or None
        self._interp = Interpreter(model_path=str(path), num_threads=threads)
        self._interp.allocate_tensors()
        self._in = self._interp.get_input_details()[0]["index"]
        self._out = self._interp.get_output_details()[0]["index"]
        self._lock = threading.Lock()  # a LiteRT interpreter is not thread-safe

    def features(self, image):
        """image: (224,224,3) float32 raw 0..255 pixels -> (1280,) features."""
        with self._lock:
            self._interp.set_tensor(self._in, image[None].astype(np.float32))
            self._interp.invoke()
            return self._interp.get_tensor(self._out)[0].copy()


_backbone = None
_backbone_lock = threading.Lock()


def get_backbone():
    global _backbone
    if _backbone is None:
        with _backbone_lock:
            if _backbone is None:
                if not BACKBONE_PATH.exists():
                    raise RuntimeError(f"No serving model at {BACKBONE_PATH}. Run scripts/export_serving_model.py.")
                _backbone = Backbone()
    return _backbone


def extract_features(source):
    """Preprocess one image (path, bytes or file object) and return its 1280-d feature vector."""
    return get_backbone().features(load_image(source))


# ---------------------------------------------------------------- head (numpy)
class Head:
    def __init__(self, W, b):
        self.W = np.asarray(W, dtype=np.float32)
        self.b = np.asarray(b, dtype=np.float32)

    @classmethod
    def load(cls, path=HEAD_PATH):
        with np.load(path) as d:
            return cls(d["W"], d["b"])

    def save(self, path=HEAD_PATH):
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, W=self.W, b=self.b)
        os.replace(tmp, path)  # atomic: other replicas reload on mtime change

    def predict_proba(self, X):
        return softmax(np.asarray(X, dtype=np.float32) @ self.W + self.b)

    def fit(self, X, y, X_val=None, y_val=None, epochs=300, lr=1e-3, l2=1e-4, class_weight=None,
            patience=20, batch_size=64, seed=42, progress_cb=None):
        """Warm-start fine-tuning of the softmax layer (Adam, L2, class weights, early stopping)."""
        rng = np.random.default_rng(seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        k = len(CLASSES)
        Y = np.eye(k, dtype=np.float32)[y]
        sw = np.ones(len(y), dtype=np.float32) if class_weight is None else np.array([class_weight[c] for c in y], np.float32)
        W, b = self.W.copy(), self.b.copy()
        mW, vW, mb, vb = (np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b))
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        best = (self._val_loss(W, b, X_val, y_val), W.copy(), b.copy())
        bad, history = 0, []
        for epoch in range(epochs):
            order = rng.permutation(len(y))
            for start in range(0, len(y), batch_size):
                idx = order[start:start + batch_size]
                P = softmax(X[idx] @ W + b)
                G = (P - Y[idx]) * sw[idx, None] / sw[idx].sum()
                gW = X[idx].T @ G + l2 * W
                gb = G.sum(axis=0)
                t += 1
                mW = b1 * mW + (1 - b1) * gW
                vW = b2 * vW + (1 - b2) * gW**2
                mb = b1 * mb + (1 - b1) * gb
                vb = b2 * vb + (1 - b2) * gb**2
                W -= lr * (mW / (1 - b1**t)) / (np.sqrt(vW / (1 - b2**t)) + eps)
                b -= lr * (mb / (1 - b1**t)) / (np.sqrt(vb / (1 - b2**t)) + eps)
            val = self._val_loss(W, b, X_val, y_val)
            history.append(val)
            if progress_cb:
                progress_cb(epoch + 1, epochs, {"val_loss": val})
            if val < best[0] - 1e-5:
                best, bad = (val, W.copy(), b.copy()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
        self.W, self.b = best[1], best[2]
        return {"epochs_run": len(history), "best_val_loss": float(best[0]), "val_loss_history": [float(v) for v in history]}

    @staticmethod
    def _val_loss(W, b, X_val, y_val):
        if X_val is None or len(X_val) == 0:
            return 0.0
        P = softmax(np.asarray(X_val, np.float32) @ W + b)
        return float(-np.log(np.clip(P[np.arange(len(y_val)), y_val], 1e-7, 1)).mean())


_head = None
_head_mtime = None
_head_lock = threading.Lock()


def get_head():
    """Current head; reloaded when head.npz changes on disk (e.g. after a retrain on the trainer)."""
    global _head, _head_mtime
    try:
        mtime = HEAD_PATH.stat().st_mtime
    except FileNotFoundError:
        raise RuntimeError(f"No classifier head at {HEAD_PATH}. Run scripts/export_serving_model.py.")
    if _head is None or mtime != _head_mtime:
        with _head_lock:
            if _head is None or mtime != _head_mtime:
                _head, _head_mtime = Head.load(), mtime
    return _head


# ---------------------------------------------------------------- prediction / metadata
def _to_result(p):
    idx = int(np.argmax(p))
    return {"label": CLASSES[idx], "confidence": float(p[idx]),
            "probabilities": {c: float(v) for c, v in zip(CLASSES, p)}}


def predict(source):
    """Predict one image given a path, bytes or file-like object."""
    feats = extract_features(source)
    return _to_result(get_head().predict_proba(feats[None])[0])


def load_features(path):
    with np.load(path, allow_pickle=False) as d:
        return d["X"], d["y"], [str(p) for p in d["paths"]]


def load_meta():
    if SERVING_META_PATH.exists():
        return json.loads(SERVING_META_PATH.read_text())
    return {}


def save_meta(meta):
    tmp = SERVING_META_PATH.with_name("serving_meta.tmp.json")
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, SERVING_META_PATH)
