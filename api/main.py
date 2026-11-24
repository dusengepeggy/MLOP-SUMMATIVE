"""FastAPI service: prediction, data upload (to SQLite), retraining and monitoring.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000

Serving is TensorFlow-free (src/serving.py: LiteRT backbone + numpy head), so it runs in ~150 MB.

Environment variables
  RETRAIN_THRESHOLD  pending uploads that auto-trigger a retrain (default 30)
  RETRAIN_EPOCHS     head fine-tuning epochs per retrain (default 30, early stopping applies)
  ENABLE_TRAINING    "true" on the trainer, "false" on scaled prediction-only replicas
  AUTO_RETRAIN       "true" to auto-trigger after uploads (default true)
  MAX_UPLOAD_MB      per-file size limit (default 5)
"""
import os
import socket
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from src import db, retrain_light, serving
from src import preprocessing as pp
from src.config import CLASSES, IMAGE_EXTENSIONS, TRAIN_DIR


def _env_bool(name, default):
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "30"))
RETRAIN_EPOCHS = int(os.getenv("RETRAIN_EPOCHS", "30"))
ENABLE_TRAINING = _env_bool("ENABLE_TRAINING", True)
AUTO_RETRAIN = _env_bool("AUTO_RETRAIN", True)
MAX_UPLOAD_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024)

START_TIME = time.time()
HOSTNAME = socket.gethostname()


@asynccontextmanager
async def lifespan(_app):
    db.init()
    try:  # warm up so the first /predict is not slow; a missing model is reported by /predict
        serving.get_backbone()
        serving.get_head()
    except RuntimeError as exc:
        print(f"[startup] {exc}")
    yield


app = FastAPI(title="Bean Leaf Disease API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------- state
_counter_lock = threading.Lock()
_prediction_count = 0
_request_count = 0

_retrain_lock = threading.Lock()  # held for the whole duration of a retrain
_job = {"job_id": None, "status": "idle"}


@app.middleware("http")
async def _count_requests(request, call_next):
    global _request_count
    with _counter_lock:
        _request_count += 1
    return await call_next(request)


def run_retrain(epochs, progress_cb, trigger):
    """Indirection so tests can replace the training call."""
    return retrain_light.retrain(epochs=epochs, progress_cb=progress_cb, trigger=trigger)


def _now():
    return datetime.now(timezone.utc).isoformat()


def start_retrain(trigger, epochs=None):
    """Start a background retrain. Returns the job id, or None if one is already running."""
    if not _retrain_lock.acquire(blocking=False):
        return None
    epochs = epochs or RETRAIN_EPOCHS
    job_id = uuid.uuid4().hex[:12]
    _job.clear()
    _job.update(
        job_id=job_id,
        status="running",
        trigger=trigger,
        started_at=_now(),
        progress={"epoch": 0, "total": epochs},
        pending_at_start=db.pending_count(),
        metrics_before=serving.load_meta().get("metrics"),
    )

    def progress(epoch, total, logs):
        _job["progress"] = {"epoch": epoch, "total": total, **logs}

    def worker():
        try:
            meta = run_retrain(epochs, progress, trigger)
            _job.update(status="done", version=meta.get("version"), metrics=meta.get("metrics"),
                        metrics_before=meta.get("previous_metrics", _job.get("metrics_before")),
                        n_new_uploads=meta.get("n_new_uploads"), epochs_run=meta.get("epochs_run"))
        except Exception as exc:  # report the failure through /retrain/status
            _job.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            _job["finished_at"] = _now()
            _retrain_lock.release()

    threading.Thread(target=worker, name=f"retrain-{job_id}", daemon=True).start()
    return job_id


def _model_info():
    meta = serving.load_meta()
    return {"model_version": meta.get("version"), "trained_at": meta.get("trained_at")}


async def _read_validated(file: UploadFile) -> bytes:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        raise ValueError(f"{file.filename}: unsupported extension {ext or '(none)'}")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"{file.filename}: larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    if not pp.is_valid_image(data):
        raise ValueError(f"{file.filename}: not a valid image")
    return data


# ---------------------------------------------------------------- endpoints
@app.get("/")
def root():
    return {"service": "bean-disease-api", "docs": "/docs", "classes": CLASSES}


@app.get("/status")
def status():
    return {
        "status": "ok",
        "hostname": HOSTNAME,
        "start_time": datetime.fromtimestamp(START_TIME, timezone.utc).isoformat(),
        "uptime_seconds": round(time.time() - START_TIME, 1),
        **_model_info(),
        "model_loaded": serving._backbone is not None and serving._head is not None,
        "pending_uploads": db.pending_count(),
        "retrain_threshold": RETRAIN_THRESHOLD,
        "training_enabled": ENABLE_TRAINING,
        "retrain_status": _job["status"],
        "prediction_count": _prediction_count,
        "request_count": _request_count,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global _prediction_count
    try:
        data = await _read_validated(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    try:
        result = await run_in_threadpool(serving.predict, data)
    except RuntimeError as exc:  # no model yet
        raise HTTPException(status_code=503, detail=str(exc))
    with _counter_lock:
        _prediction_count += 1
    return result


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...), label: str = Form(...)):
    """Save labelled images, record them in the database and preprocess them for retraining."""
    if not ENABLE_TRAINING:
        raise HTTPException(status_code=403, detail="Uploads are handled by the training service")
    if label not in CLASSES:
        raise HTTPException(status_code=400, detail=f"label must be one of {CLASSES}")
    saved, rejected = [], []
    for f in files:
        try:
            data = await _read_validated(f)
            upload_id = await run_in_threadpool(retrain_light.ingest_upload, data, label, f.filename)
            saved.append({"id": upload_id, "filename": f.filename, "status": "preprocessed"})
        except ValueError as exc:
            rejected.append(str(exc))
    if not saved:
        raise HTTPException(status_code=400, detail={"message": "no valid images", "rejected": rejected})

    pending = db.pending_count()
    job_id = None
    if AUTO_RETRAIN and pending >= RETRAIN_THRESHOLD:
        job_id = start_retrain(trigger="auto")
    return {
        "saved": len(saved),
        "uploads": saved,
        "rejected": rejected,
        "label": label,
        "pending_uploads": pending,
        "retrain_threshold": RETRAIN_THRESHOLD,
        "auto_retrain_job_id": job_id,
    }


@app.get("/uploads")
def uploads(limit: int = Query(200, ge=1, le=1000)):
    """Database records of uploaded training images (newest first) and a per-class summary."""
    return {"pending_uploads": db.pending_count(), "summary": db.upload_summary(), "uploads": db.list_uploads(limit)}


@app.post("/retrain")
def retrain(epochs: int = Query(None, ge=1, le=300)):
    if not ENABLE_TRAINING:
        raise HTTPException(status_code=403, detail="Training is disabled on this replica")
    job_id = start_retrain(trigger="manual", epochs=epochs)
    if job_id is None:
        raise HTTPException(status_code=409, detail={"message": "Retrain already running", "job_id": _job["job_id"]})
    return {"job_id": job_id, "status": "running"}


@app.get("/retrain/status")
def retrain_status():
    latest = serving.load_meta()
    return {**_job, "latest_version": latest.get("version"), "latest_metrics": latest.get("metrics")}


@app.get("/retrain/history")
def retrain_history(limit: int = Query(20, ge=1, le=200)):
    return {"retrains": db.list_retrains(limit)}


def _base_distribution():
    """Original training-set class counts (from the cached features, so the image folders are optional)."""
    if TRAIN_DIR.exists() and any(TRAIN_DIR.glob("*/*.jpg")):
        return pp.class_distribution(TRAIN_DIR)
    _, y, _ = serving.load_features(serving.FEATURES_TRAIN_PATH)
    return {c: int((y == i).sum()) for i, c in enumerate(CLASSES)}


@app.get("/metrics")
def metrics():
    meta = serving.load_meta()
    used = {c: 0 for c in CLASSES}
    for row in db.upload_summary():
        if row["status"] == "used":
            used[row["label"]] += row["n"]
    base = _base_distribution()
    return {
        **_model_info(),
        "metrics": meta.get("metrics"),
        "previous_metrics": meta.get("previous_metrics"),
        "class_distribution": {c: base[c] + used[c] for c in CLASSES},
        "base_distribution": base,
        "uploaded_used_distribution": used,
        "pending_uploads": db.pending_count(),
        "prediction_count": _prediction_count,
        "hostname": HOSTNAME,
    }
