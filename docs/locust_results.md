# Locust load-test results

`locustfile.py` simulates users who call `/predict` (weight 5, real held-out leaf photos), `/status` (2) and
`/metrics` (1), waiting 0.5–2 s between requests.

## 1. Cloud deployment (Render, free tier)

Target: `https://bean-leaf-api-pd.onrender.com`, one free instance (0.1 CPU, 512 MB, Frankfurt region).
Profile: 10 users, spawn rate 2/s, 2 minutes, run from Kigali (so latency includes the ~250 ms internet round trip).
Report: [locust_render.html](locust_render.html).

| Endpoint | Requests | Median (ms) | p95 (ms) | p99 (ms) | Avg (ms) | Requests/s | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| `/predict` | 459 | 360 | 700 | 930 | 411 | 3.84 | 0 (0 %) |
| `/status` | 152 | 260 | 410 | 620 | 289 | 1.27 | 0 (0 %) |
| `/metrics` | 91 | 490 | 1700 | 2400 | 603 | 0.76 | 0 (0 %) |
| **All** | **702** | **350** | **720** | **1100** | **409** | **5.87** | **0 (0 %)** |

**Uptime:** the instance served every request. Its `start_time` was unchanged and its `uptime_seconds` grew from 11 s to
134 s across the test (737 requests counted), so there were no restarts and no errors: **100 % availability** during the
test.

**Interpretation:**
- `/status` does no ML work, and its median of 260 ms is essentially network latency. A prediction therefore adds only
  about **100 ms** of server time, even on 0.1 CPU, because the API runs the TFLite backbone instead of full TensorFlow.
- The p95 of `/metrics` is higher because it queries the database and reads the cached features.
- One caveat of the free tier: after 15 minutes without traffic the instance sleeps. The first request then takes about
  **18 s** (cold start, measured), and the in-memory and on-disk state (uploads, retrained head) is reset. An uptime
  pinger (UptimeRobot, every 5 min) prevents this.

## 2. Scaling: 1, 2 and 3 API containers (local Docker + nginx)

The same profile was run for each scale: 50 users, spawn rate 5/s, 2 minutes, headless, through nginx round-robin.
Each API container is limited to **1 CPU and 512 MB** (the same memory budget as a Render free instance).
Reproduce with `bash scripts/run_load_tests.sh`, then `python scripts/summarize_locust.py`.

| Containers | Endpoint | Requests | Median (ms) | p95 (ms) | Avg (ms) | Requests/s | Failures % |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | all | 4235 | 54 | 370 | 103 | 35.6 | 0.00 |
| 1 | predict | 2660 | 59 | 790 | 131 | 22.4 | 0.00 |
| 2 | all | 4432 | 55 | 110 | 53 | 37.3 | 0.00 |
| 2 | predict | 2803 | 61 | 100 | 61 | 23.6 | 0.00 |
| 3 | all | 4384 | 51 | 94 | 73 | 36.4 | 0.00 |
| 3 | predict | 2747 | 59 | 94 | 89 | 22.8 | 0.00 |

![Latency and throughput vs number of API containers](locust_scaling.png)

HTML reports: [1 container](locust_api1.html) · [2 containers](locust_api2.html) · [3 containers](locust_api3.html)

**Interpretation:**
- **One container already sustains the whole load** (35.6 req/s, median 54 ms, 0 failures). The ceiling is the offered
  load: 50 users who pause 1.25 s on average can generate at most about 50 / 1.3 s ≈ 38 req/s.
- **Extra containers mainly cut tail latency.** With one container, bursts queue behind single inferences (p95 370 ms,
  `/predict` p95 790 ms). Two containers bring the p95 down to 110 ms. A third adds little (94 ms), so for this traffic
  **scaling stops paying off after 2 containers**. More users would be needed to justify a third.
- **History:** the first version of the API loaded full TensorFlow (about 750 MB per container).
  - Under the same test, one container managed only 15 req/s with a 1.8 s median.
  - Without a per-replica inference lock it was OOM-killed.
  - Moving inference to the TFLite backbone (about 70 MB per container) raised single-container throughput 2.3x, cut the
    median latency about 30x, and is what made the free cloud deployment possible.
