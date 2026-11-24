#!/usr/bin/env bash
# One-shot setup for a fresh Ubuntu 22.04/24.04 VM (2 vCPU / 4 GB RAM or more).
#
# Over SSH:
#   curl -fsSL https://raw.githubusercontent.com/dusengepeggy/MLOP-SUMMATIVE/main/deploy/setup_vm.sh | bash
# Without SSH (Azure portal > VM > Run command > RunShellScript, runs as root):
#   curl -fsSL https://raw.githubusercontent.com/dusengepeggy/MLOP-SUMMATIVE/main/deploy/setup_vm.sh -o /root/setup_vm.sh
#   nohup bash /root/setup_vm.sh > /var/log/bean-setup.log 2>&1 &
#   (check progress later with: tail -n 30 /var/log/bean-setup.log)
#
# Open inbound TCP 80 (API via nginx) and 8501 (Streamlit UI) in the cloud firewall / security group.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/dusengepeggy/MLOP-SUMMATIVE.git}"
APP_DIR="${APP_DIR:-${HOME:-/root}/MLOP-SUMMATIVE}"
API_REPLICAS="${API_REPLICAS:-2}"
SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

echo "[setup] $(date -u) starting"
if ! command -v docker >/dev/null 2>&1; then
    curl -fsSL https://get.docker.com | $SUDO sh
fi
# let the login user(s) run docker without sudo
for u in ${USER:-} $(getent group sudo | cut -d: -f4 | tr ',' ' '); do
    [ -n "$u" ] && [ "$u" != "root" ] && $SUDO usermod -aG docker "$u" || true
done

# TensorFlow image builds need headroom on 4 GB machines.
if [ ! -f /swapfile ]; then
    $SUDO fallocate -l 4G /swapfile && $SUDO chmod 600 /swapfile
    $SUDO mkswap /swapfile && $SUDO swapon /swapfile
    echo '/swapfile none swap sw 0 0' | $SUDO tee -a /etc/fstab
fi

if [ -d "$APP_DIR/.git" ]; then
    git -C "$APP_DIR" pull --ff-only
else
    git clone "$REPO_URL" "$APP_DIR"
fi
cd "$APP_DIR"

echo "[setup] $(date -u) building and starting containers"
# The first start downloads the dataset inside the trainer, which can outlast its
# healthcheck grace period; retrying `up` lets the dependent services start once it is healthy.
for attempt in 1 2 3; do
    if $SUDO docker compose up -d --build --scale api="$API_REPLICAS"; then break; fi
    echo "[setup] compose up attempt $attempt failed, retrying in 30 s"
    sleep 30
done

echo "[setup] waiting for the API to become healthy..."
for _ in $(seq 1 60); do
    if curl -fsS http://localhost/status >/dev/null 2>&1; then break; fi
    sleep 5
done
curl -fsS http://localhost/status && echo
IP=$(curl -fsS https://checkip.amazonaws.com 2>/dev/null || hostname -I | awk '{print $1}')
echo "API: http://$IP/docs"
echo "UI : http://$IP:8501"
echo "[setup] $(date -u) done"
