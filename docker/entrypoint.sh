#!/bin/sh
# The trainer (DOWNLOAD_DATA=true) fetches the dataset into the shared data volume
# on first start, so retraining has the original images to learn from.
set -e
if [ "${DOWNLOAD_DATA:-false}" = "true" ]; then
    python -m src.data_acquisition || echo "WARNING: dataset download failed; retrain will only see uploads"
fi
exec "$@"
