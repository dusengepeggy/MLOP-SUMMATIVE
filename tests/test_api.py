import io
import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

import api.main as main
from src import db

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "docs" / "samples"

client = TestClient(main.app)
db.init()


def _jpeg(color=(40, 160, 40), size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def _wait_for_job(timeout=60):
    for _ in range(int(timeout / 0.1)):
        body = client.get("/retrain/status").json()
        if body["status"] != "running":
            return body
        time.sleep(0.1)
    raise AssertionError("retrain did not finish")


def test_status():
    body = client.get("/status").json()
    for key in ("uptime_seconds", "model_version", "trained_at", "pending_uploads", "start_time"):
        assert key in body


def test_predict_sample_images_are_correct():
    for cls in main.CLASSES:
        r = client.post("/predict", files={"file": (f"{cls}.jpg", (SAMPLES / f"{cls}.jpg").read_bytes(), "image/jpeg")})
        assert r.status_code == 200
        body = r.json()
        assert body["label"] == cls
        assert abs(sum(body["probabilities"].values()) - 1) < 1e-4


def test_predict_rejects_non_image():
    assert client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")}).status_code == 400
    assert client.post("/predict", files={"file": ("fake.jpg", b"not a jpeg", "image/jpeg")}).status_code == 400


def test_upload_saves_to_database_and_preprocesses():
    before = client.get("/status").json()["pending_uploads"]
    files = [("files", (f"leaf{i}.jpg", _jpeg(), "image/jpeg")) for i in range(3)]
    r = client.post("/upload", files=files, data={"label": "healthy"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["saved"] == 3 and body["pending_uploads"] == before + 3
    rows = client.get("/uploads").json()["uploads"]
    assert {row["status"] for row in rows[:3]} == {"preprocessed"}
    assert all(len(row["sha256"]) == 64 for row in rows[:3])
    stored = db.uploads_with_status("preprocessed")
    assert len(stored[-1]["features"]) == 1280 * 4  # float32 feature vector stored in the DB


def test_upload_rejects_unknown_label_and_bad_files(monkeypatch):
    assert client.post("/upload", files=[("files", ("a.jpg", _jpeg(), "image/jpeg"))],
                       data={"label": "banana"}).status_code == 400
    mixed = [("files", ("ok.jpg", _jpeg(), "image/jpeg")), ("files", ("bad.txt", b"x", "text/plain"))]
    r = client.post("/upload", files=mixed, data={"label": "bean_rust"})
    assert r.json()["saved"] == 1 and len(r.json()["rejected"]) == 1
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 100)
    big = [("files", ("big.jpg", _jpeg(size=(256, 256)), "image/jpeg"))]
    assert client.post("/upload", files=big, data={"label": "healthy"}).status_code == 400


def test_real_retrain_uses_uploads_and_reports_before_after():
    client.post("/upload", files=[("files", ("rust.jpg", (SAMPLES / "bean_rust.jpg").read_bytes(), "image/jpeg"))],
                data={"label": "bean_rust"})
    version = client.get("/status").json()["model_version"]
    r = client.post("/retrain", params={"epochs": 5})
    assert r.status_code == 200
    done = _wait_for_job()
    assert done["status"] == "done", done
    assert done["version"] == version + 1
    for m in (done["metrics_before"], done["metrics"]):
        for key in ("accuracy", "precision", "recall", "f1", "loss", "roc_auc"):
            assert key in m
    assert done["metrics"]["accuracy"] > 0.85
    assert client.get("/status").json()["pending_uploads"] == 0
    assert {row["status"] for row in client.get("/uploads").json()["uploads"]} == {"used"}
    history = client.get("/retrain/history").json()["retrains"]
    assert history[0]["status"] == "done" and history[0]["n_new"] >= 1
    # replicas pick up the new head; prediction still works
    r = client.post("/predict", files={"file": ("h.jpg", (SAMPLES / "healthy.jpg").read_bytes(), "image/jpeg")})
    assert r.json()["label"] == "healthy"


def test_concurrent_retrain_returns_409(monkeypatch):
    release = threading.Event()

    def slow_retrain(epochs, progress_cb, trigger):
        progress_cb(1, epochs, {"val_loss": 0.2})
        release.wait(5)
        return {"version": 99, "metrics": {"accuracy": 0.9}}

    monkeypatch.setattr(main, "run_retrain", slow_retrain)
    assert client.post("/retrain").status_code == 200
    assert client.post("/retrain").status_code == 409
    release.set()
    assert _wait_for_job()["version"] == 99


def test_metrics_endpoint():
    body = client.get("/metrics").json()
    assert set(body["class_distribution"]) == set(main.CLASSES)
    assert "loss" in body["metrics"]
