"""SQLite records for uploaded training data and retraining runs.

uploads   one row per uploaded image: file, label, checksum, size, preprocessing status and the
          preprocessed 1280-d feature vector (BLOB) that retraining consumes
retrains  one row per retraining run: trigger, number of new images, metrics before/after on the
          held-out test set, resulting model version

Status flow of an upload: pending -> preprocessed -> used (consumed by retrain #retrain_id).
"""
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config import DATA_DIR

DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "bean.db"))
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    label TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    width INTEGER, height INTEGER,
    status TEXT NOT NULL DEFAULT 'pending',
    features BLOB,
    uploaded_at TEXT NOT NULL,
    preprocessed_at TEXT,
    retrain_id INTEGER REFERENCES retrains(id)
);
CREATE TABLE IF NOT EXISTS retrains (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    n_new INTEGER,
    n_train INTEGER,
    epochs_run INTEGER,
    version_before INTEGER,
    version_after INTEGER,
    metrics_before TEXT,
    metrics_after TEXT,
    error TEXT
);
"""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    return con


def init():
    with _lock, _connect() as con:
        con.executescript(SCHEMA)


# ---------------------------------------------------------------- uploads
def add_upload(filename, label, path, sha256, width, height):
    with _lock, _connect() as con:
        cur = con.execute(
            "INSERT INTO uploads (filename, label, path, sha256, width, height, uploaded_at) VALUES (?,?,?,?,?,?,?)",
            (filename, label, str(path), sha256, width, height, _now()),
        )
        return cur.lastrowid


def set_features(upload_id, features):
    with _lock, _connect() as con:
        con.execute(
            "UPDATE uploads SET features=?, status='preprocessed', preprocessed_at=? WHERE id=?",
            (np.asarray(features, np.float32).tobytes(), _now(), upload_id),
        )


def uploads_with_status(*statuses):
    q = ",".join("?" * len(statuses))
    with _connect() as con:
        return [dict(r) for r in con.execute(f"SELECT * FROM uploads WHERE status IN ({q}) ORDER BY id", statuses)]


def features_of(rows):
    """(X, labels) for upload rows that have features."""
    rows = [r for r in rows if r["features"] is not None]
    if not rows:
        return np.zeros((0, 1280), np.float32), []
    return np.stack([np.frombuffer(r["features"], np.float32) for r in rows]), [r["label"] for r in rows]


def mark_used(ids, retrain_id):
    if not ids:
        return
    q = ",".join("?" * len(ids))
    with _lock, _connect() as con:
        con.execute(f"UPDATE uploads SET status='used', retrain_id=? WHERE id IN ({q})", (retrain_id, *ids))


def pending_count():
    with _connect() as con:
        return con.execute("SELECT COUNT(*) FROM uploads WHERE status IN ('pending','preprocessed')").fetchone()[0]


def list_uploads(limit=200):
    with _connect() as con:
        rows = con.execute(
            "SELECT id, filename, label, status, width, height, sha256, uploaded_at, preprocessed_at, retrain_id "
            "FROM uploads ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]


def upload_summary():
    with _connect() as con:
        rows = con.execute("SELECT label, status, COUNT(*) AS n FROM uploads GROUP BY label, status")
        return [dict(r) for r in rows]


# ---------------------------------------------------------------- retrains
def start_retrain(trigger, version_before, metrics_before):
    with _lock, _connect() as con:
        cur = con.execute(
            "INSERT INTO retrains (trigger, status, started_at, version_before, metrics_before) VALUES (?,?,?,?,?)",
            (trigger, "running", _now(), version_before, json.dumps(metrics_before)),
        )
        return cur.lastrowid


def finish_retrain(retrain_id, status, **fields):
    fields = {k: json.dumps(v) if k.startswith("metrics") else v for k, v in fields.items()}
    sets = ", ".join(f"{k}=?" for k in fields)
    with _lock, _connect() as con:
        con.execute(
            f"UPDATE retrains SET status=?, finished_at=?{', ' + sets if sets else ''} WHERE id=?",
            (status, _now(), *fields.values(), retrain_id),
        )


def list_retrains(limit=50):
    with _connect() as con:
        out = []
        for r in con.execute("SELECT * FROM retrains ORDER BY id DESC LIMIT ?", (limit,)):
            r = dict(r)
            for k in ("metrics_before", "metrics_after"):
                r[k] = json.loads(r[k]) if r[k] else None
            out.append(r)
        return out
