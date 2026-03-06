#!/bin/bash
# =============================================================================
# Traot — VPS Deployment Script (Hostinger)
# =============================================================================
# Deploys latest code + transfers models to your VPS.
#
# Usage:
#   ./scripts/deploy_vps.sh <vps-ip>
#
# Examples:
#   ./scripts/deploy_vps.sh 45.67.89.10
#
# What this does:
#   1. Pushes local git changes to GitHub
#   2. Transfers models/ to VPS via rsync
#   3. Transfers data/trading.db to VPS via rsync (if < 500MB)
#   4. SSHes in, pulls latest code, installs deps, restarts service
# =============================================================================

set -e  # Exit on any error

VPS_USER="root"
VPS_IP=${1:-""}
REMOTE_DIR="/root/ai-trade-bot"

if [ -z "$VPS_IP" ]; then
    echo "ERROR: VPS IP required."
    echo "Usage: ./scripts/deploy_vps.sh <ip>"
    echo "Example: ./scripts/deploy_vps.sh 45.67.89.10"
    exit 1
fi

echo ""
echo "=== Traot — Deploy to Hostinger VPS ==="
echo "Target: ${VPS_USER}@${VPS_IP}:${REMOTE_DIR}"
echo ""

# Step 1: Push local changes to GitHub
echo "[1/4] Pushing local changes to GitHub..."
git push origin master 2>&1 || echo "      (nothing new to push)"
echo "      Done."

# Step 2: Transfer models (only changed files via rsync)
echo "[2/4] Syncing models/ to VPS..."
rsync -avz --progress \
    models/ \
    ${VPS_USER}@${VPS_IP}:${REMOTE_DIR}/models/
echo "      Done."

# Step 3: Transfer database (skip if too large)
if [ -f "data/trading.db" ]; then
    DB_SIZE=$(du -sm data/trading.db 2>/dev/null | cut -f1)
    if [ "$DB_SIZE" -lt 500 ]; then
        echo "[3/4] Syncing data/trading.db to VPS (${DB_SIZE}MB)..."
        rsync -avz --progress \
            data/trading.db \
            ${VPS_USER}@${VPS_IP}:${REMOTE_DIR}/data/
        echo "      Done."
    else
        echo "[3/4] DB is ${DB_SIZE}MB — skipping (do manually if first deploy)."
    fi
else
    echo "[3/4] No local database — skipping."
fi

# Step 4: Pull code on VPS and restart service
echo "[4/4] Pulling code on VPS and restarting service..."
ssh ${VPS_USER}@${VPS_IP} "
    set -e
    cd ${REMOTE_DIR}
    git pull origin master
    source venv/bin/activate
    pip install -r requirements.txt --quiet
    systemctl restart ai-trade-bot
    echo ''
    systemctl status ai-trade-bot --no-pager
"

echo ""
echo "=== Deployment complete ==="
echo ""
echo "Watch live logs:"
echo "  ssh ${VPS_USER}@${VPS_IP} 'journalctl -u ai-trade-bot -f'"
echo ""
echo "Health check:"
echo "  ssh ${VPS_USER}@${VPS_IP} 'cd ${REMOTE_DIR} && bash scripts/health_check.sh'"
