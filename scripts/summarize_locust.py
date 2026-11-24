"""Summarise docs/locust/api{N}_stats.csv into a markdown table and a scaling chart.

Usage: python scripts/summarize_locust.py   -> prints the table, writes docs/locust_scaling.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def load():
    rows = []
    for csv in sorted((DOCS / "locust").glob("api*_stats.csv")):
        n = int(csv.stem.split("_")[0][3:])
        df = pd.read_csv(csv).set_index("Name")
        for name, label in (("Aggregated", "all"), ("/predict", "predict")):
            r = df.loc[name]
            rows.append({
                "containers": n,
                "endpoint": label,
                "requests": int(r["Request Count"]),
                "median_ms": float(r["Median Response Time"]),
                "p95_ms": float(r["95%"]),
                "avg_ms": float(r["Average Response Time"]),
                "rps": float(r["Requests/s"]),
                "fail_pct": 100 * float(r["Failure Count"]) / max(float(r["Request Count"]), 1),
            })
    return pd.DataFrame(rows).sort_values(["endpoint", "containers"])


def markdown(df):
    lines = [
        "| Containers | Endpoint | Requests | Median (ms) | p95 (ms) | Avg (ms) | Requests/s | Failures % |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.sort_values(["containers", "endpoint"]).iterrows():
        lines.append(f"| {r.containers} | {r.endpoint} | {r.requests} | {r.median_ms:.0f} | {r.p95_ms:.0f} | "
                     f"{r.avg_ms:.0f} | {r.rps:.1f} | {r.fail_pct:.2f} |")
    return "\n".join(lines)


def chart(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    p = df[df.endpoint == "predict"]
    a = df[df.endpoint == "all"]
    axes[0].plot(p.containers, p.median_ms, "o-", label="/predict median")
    axes[0].plot(p.containers, p.p95_ms, "s--", label="/predict p95")
    axes[0].set(title="Latency vs API containers", xlabel="API containers", ylabel="ms", xticks=[1, 2, 3])
    axes[0].legend()
    axes[1].bar(a.containers, a.rps, color="#3b9a57")
    for x, y in zip(a.containers, a.rps):
        axes[1].text(x, y, f"{y:.1f}", ha="center", va="bottom")
    axes[1].set(title="Throughput (all endpoints)", xlabel="API containers", ylabel="requests/s", xticks=[1, 2, 3])
    fig.suptitle("Locust: 50 users, spawn 5/s, 2 min per run")
    fig.tight_layout()
    out = DOCS / "locust_scaling.png"
    fig.savefig(out, dpi=120)
    return out


if __name__ == "__main__":
    df = load()
    print(markdown(df))
    print("chart:", chart(df))
