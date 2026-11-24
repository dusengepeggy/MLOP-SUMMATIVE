"""Load test for the prediction API behind nginx.

Headless example (same profile used for 1, 2 and 3 replicas):
  locust -f locustfile.py --host http://localhost --headless -u 50 -r 5 -t 2m \
         --csv docs/locust/api1 --html docs/locust_api1.html
"""
import os
import random
from pathlib import Path

from locust import HttpUser, between, task

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = Path(os.getenv("LOCUST_IMAGE_DIR", ROOT / "docs" / "samples"))
IMAGES = [(p.name, p.read_bytes()) for p in sorted(SAMPLE_DIR.glob("*.jpg"))]
if not IMAGES:
    raise SystemExit(f"No sample .jpg images found in {SAMPLE_DIR}")


class BeanUser(HttpUser):
    wait_time = between(0.5, 2)

    @task(5)
    def predict(self):
        name, data = random.choice(IMAGES)
        with self.client.post("/predict", files={"file": (name, data, "image/jpeg")}, name="/predict", catch_response=True) as r:
            if r.status_code != 200 or "label" not in r.text:
                r.failure(f"bad response {r.status_code}")

    @task(2)
    def status(self):
        self.client.get("/status", name="/status")

    @task(1)
    def metrics(self):
        self.client.get("/metrics", name="/metrics")
