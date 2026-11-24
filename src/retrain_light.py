"""Lightweight retraining pipeline (TF-free), used by the API.

1. Upload  : images are saved to data/uploads/<label>/ and recorded in SQLite (src/db.py)
2. Preprocess: each upload is validated, converted to RGB, resized to 224x224 and passed through
             the trained MobileNetV2 backbone; the 1280-d feature vector is stored in the DB
3. Retrain : the trained classifier head is fine-tuned starting from its CURRENT weights
             (our own model is the pre-trained model) on the original training features plus all
             uploaded features, with Adam, L2 regularisation, class weights and early stopping
4. Evaluate: the held-out test set (data/test, never trained on) is scored before and after,
             so the effect of the new data is measured on the same images
5. Save    : new head version (previous one backed up), metadata and the DB records are updated
"""
import hashlib
import io
import shutil

import numpy as np
from PIL import Image

from src import db, serving
from src import preprocessing as pp
from src.config import CLASSES, MODELS_DIR, SEED
from src.metrics import compute_metrics


# ---------------------------------------------------------------- upload + preprocessing
def ingest_upload(data: bytes, label: str, filename: str):
    """Save an uploaded image, record it in the DB and preprocess it. Returns the DB row id."""
    path = pp.save_upload(data, label, filename)
    with Image.open(io.BytesIO(data)) as img:
        width, height = img.size
    upload_id = db.add_upload(path.name, label, path, hashlib.sha256(data).hexdigest(), width, height)
    preprocess(upload_id, path)
    return upload_id


def preprocess(upload_id, path):
    """RGB -> 224x224 -> backbone features, stored in the uploads table (status 'preprocessed')."""
    db.set_features(upload_id, serving.extract_features(path))


def preprocess_pending():
    for row in db.uploads_with_status("pending"):
        preprocess(row["id"], row["path"])


# ---------------------------------------------------------------- retraining
def _stratified_split(y, val_frac=0.15, seed=SEED):
    rng = np.random.default_rng(seed)
    val = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        val.extend(idx[: max(1, int(round(len(idx) * val_frac)))])
    mask = np.zeros(len(y), bool)
    mask[val] = True
    return ~mask, mask


def _class_weights(y):
    counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
    counts[counts == 0] = 1
    w = counts.sum() / (len(CLASSES) * counts)
    return {i: float(v) for i, v in enumerate(w)}


def retrain(epochs=30, progress_cb=None, trigger="manual"):
    db.init()
    preprocess_pending()
    new_rows = db.uploads_with_status("preprocessed")
    old_rows = db.uploads_with_status("used")

    X_base, y_base, _ = serving.load_features(serving.FEATURES_TRAIN_PATH)
    X_test, y_test, _ = serving.load_features(serving.FEATURES_TEST_PATH)
    X_up, labels_up = db.features_of(old_rows + new_rows)
    y_up = np.array([CLASSES.index(lbl) for lbl in labels_up], dtype=int)
    X = np.concatenate([X_base, X_up]) if len(X_up) else X_base
    y = np.concatenate([y_base, y_up]) if len(y_up) else y_base

    head = serving.Head.load()
    meta = serving.load_meta()
    version = int(meta.get("version", 1))
    before = compute_metrics(y_test, head.predict_proba(X_test))
    run_id = db.start_retrain(trigger, version, before)
    try:
        tr, va = _stratified_split(y)
        fit = head.fit(X[tr], y[tr], X[va], y[va], epochs=epochs, class_weight=_class_weights(y[tr]),
                       progress_cb=progress_cb)
        after = compute_metrics(y_test, head.predict_proba(X_test))

        shutil.copy2(serving.HEAD_PATH, MODELS_DIR / f"head_v{version}.npz")  # keep the previous version
        head.save()
        new_meta = {
            **{k: meta[k] for k in ("classes", "source_model") if k in meta},
            "version": version + 1,
            "kind": "retrain",
            "trained_at": db._now(),
            "trigger": trigger,
            "retrain_id": run_id,
            "n_train": int(tr.sum()),
            "n_uploads_total": int(len(y_up)),
            "n_new_uploads": len(new_rows),
            "epochs_run": fit["epochs_run"],
            "metrics": after,
            "previous_metrics": before,
        }
        serving.save_meta(new_meta)
        db.mark_used([r["id"] for r in new_rows], run_id)
        db.finish_retrain(run_id, "done", n_new=len(new_rows), n_train=int(tr.sum()), epochs_run=fit["epochs_run"],
                          version_after=version + 1, metrics_after=after)
        return new_meta
    except Exception as exc:
        db.finish_retrain(run_id, "failed", error=f"{type(exc).__name__}: {exc}")
        raise
