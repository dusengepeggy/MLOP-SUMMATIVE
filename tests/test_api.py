import io
import threading
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api.main as main
from src.config import MODEL_PATH

client = TestClient(main.app)


def _jpeg(color=(40, 160, 40), size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def test_status():
    r = client.get("/status")
    assert r.status_code == 200
    body = r.json()
    for key in ("uptime_seconds", "model_version", "trained_at", "pending_uploads", "start_time"):
        assert key in body


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="no trained model")
def test_predict_valid_image():
    r = client.post("/predict", files={"file": ("leaf.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in main.CLASSES
    assert set(body["probabilities"]) == set(main.CLASSES)
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-4
    assert 0 <= body["confidence"] <= 1


def test_predict_rejects_non_image():
    r = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_predict_rejects_corrupt_jpeg():
    r = client.post("/predict", files={"file": ("fake.jpg", b"not really a jpeg", "image/jpeg")})
    assert r.status_code == 400


def test_upload_saves_files_and_counts_pending():
    before = client.get("/status").json()["pending_uploads"]
    files = [("files", (f"leaf{i}.jpg", _jpeg(), "image/jpeg")) for i in range(3)]
    r = client.post("/upload", files=files, data={"label": "healthy"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["saved"] == 3
    assert body["pending_uploads"] == before + 3
    assert body["auto_retrain_job_id"] is None


def test_upload_rejects_unknown_label():
    r = client.post("/upload", files=[("files", ("a.jpg", _jpeg(), "image/jpeg"))], data={"label": "banana"})
    assert r.status_code == 400


def test_upload_mixed_batch_reports_rejected():
    files = [
        ("files", ("ok.jpg", _jpeg(), "image/jpeg")),
        ("files", ("bad.txt", b"x", "text/plain")),
    ]
    r = client.post("/upload", files=files, data={"label": "bean_rust"})
    assert r.status_code == 200
    assert r.json()["saved"] == 1
    assert len(r.json()["rejected"]) == 1


def test_upload_rejects_oversized_file(monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 100)
    r = client.post("/upload", files=[("files", ("big.jpg", _jpeg(size=(256, 256)), "image/jpeg"))], data={"label": "healthy"})
    assert r.status_code == 400


def test_retrain_runs_in_background_and_blocks_concurrent_runs(monkeypatch):
    release = threading.Event()

    def fake_retrain(epochs, progress_cb):
        progress_cb(1, epochs, {"accuracy": 0.9})
        release.wait(5)
        return {"version": 99, "metrics": {"accuracy": 0.9}}

    monkeypatch.setattr(main, "run_retrain", fake_retrain)

    r1 = client.post("/retrain", params={"epochs": 2})
    assert r1.status_code == 200
    job_id = r1.json()["job_id"]

    r2 = client.post("/retrain")
    assert r2.status_code == 409

    running = client.get("/retrain/status").json()
    assert running["status"] == "running" and running["job_id"] == job_id

    release.set()
    for _ in range(50):
        if client.get("/retrain/status").json()["status"] != "running":
            break
        time.sleep(0.05)
    done = client.get("/retrain/status").json()
    assert done["status"] == "done"
    assert done["version"] == 99


def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert set(body["class_distribution"]) == set(main.CLASSES)
    assert "prediction_count" in body
