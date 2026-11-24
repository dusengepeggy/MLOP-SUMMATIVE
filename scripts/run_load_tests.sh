#!/usr/bin/env bash
# Run the identical Locust profile against 1, 2 and 3 API replicas behind nginx.
#   bash scripts/run_load_tests.sh                 # local
#   HOST=http://<vm-ip> bash scripts/run_load_tests.sh   # against a remote deployment
# Env: USERS (50), SPAWN (5), DURATION (2m), REPLICAS ("1 2 3"), LOCUST (locust executable)
set -euo pipefail
cd "$(dirname "$0")/.."

HOST="${HOST:-http://localhost}"
USERS="${USERS:-50}"
SPAWN="${SPAWN:-5}"
DURATION="${DURATION:-2m}"
REPLICAS="${REPLICAS:-1 2 3}"
LOCUST="${LOCUST:-locust}"
mkdir -p docs/locust

for n in $REPLICAS; do
    echo "=== $n API replica(s) ==="
    docker compose up -d --no-build --scale api="$n" >/dev/null
    # wait until every replica is healthy, then give nginx time to re-resolve (valid=5s)
    until [ "$(docker compose ps api --format '{{.Health}}' | grep -c healthy)" -ge "$n" ]; do sleep 3; done
    sleep 10
    seen=$(for _ in $(seq 1 $((n * 6))); do curl -fsS "$HOST/status" | grep -o '"hostname":"[^"]*"'; done | sort -u | wc -l)
    echo "nginx is routing to $seen distinct replica(s)"
    for _ in $(seq 1 $((n * 3))); do curl -fsS -o /dev/null -X POST "$HOST/predict" -F "file=@docs/samples/healthy.jpg"; done  # warm-up

    "$LOCUST" -f locustfile.py --host "$HOST" --headless -u "$USERS" -r "$SPAWN" -t "$DURATION" \
        --csv "docs/locust/api$n" --html "docs/locust_api$n.html" --only-summary 2>&1 | tail -n 25
done
