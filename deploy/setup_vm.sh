#!/usr/bin/env bash
# One-shot setup for a fresh Ubuntu 22.04/24.04 VM (2 vCPU / 4 GB RAM or more).
#   curl -fsSL https://raw.githubusercontent.com/dusengepeggy/MLOP-SUMMATIVE/main/deploy/setup_vm.sh | bash
# Open inbound TCP 80 (API via nginx) and 8501 (Streamlit UI) in the cloud firewall / security group.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/dusengepeggy/MLOP-SUMMATIVE.git}"
APP_DIR="${APP_DIR:-$HOME/MLOP-SUMMATIVE}"
API_REPLICAS="${API_REPLICAS:-2}"

if ! command -v docker >/dev/null 2>&1; then
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker "$USER"
fi

# TensorFlow image builds need headroom on 4 GB machines.
if [ ! -f /swapfile ]; then
    sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile
    sudo mkswap /swapfile && sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

if [ -d "$APP_DIR/.git" ]; then
    git -C "$APP_DIR" pull --ff-only
else
    git clone "$REPO_URL" "$APP_DIR"
fi
cd "$APP_DIR"

sudo docker compose up -d --build --scale api="$API_REPLICAS"

echo "Waiting for the API to become healthy..."
for _ in $(seq 1 60); do
    if curl -fsS http://localhost/status >/dev/null 2>&1; then break; fi
    sleep 5
done
curl -fsS http://localhost/status && echo
IP=$(curl -fsS https://checkip.amazonaws.com 2>/dev/null || hostname -I | awk '{print $1}')
echo "API: http://$IP/docs"
echo "UI : http://$IP:8501"
