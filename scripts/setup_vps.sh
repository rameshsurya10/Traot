#!/bin/bash
# =============================================================================
# Traot — First-Time VPS Setup (Hostinger Ubuntu)
# =============================================================================
# Run this ONCE on a fresh Hostinger VPS to set up everything.
#
# Usage (on the VPS):
#   curl -sSL https://raw.githubusercontent.com/rameshsurya10/Traot/master/scripts/setup_vps.sh | bash
#
# Or manually:
#   bash scripts/setup_vps.sh
# =============================================================================

set -e  # Exit on any error

REPO_URL="https://github.com/rameshsurya10/Traot.git"
APP_DIR="/root/ai-trade-bot"
PYTHON_VERSION="3.12"

echo ""
echo "=========================================="
echo "  Traot — VPS Setup (Hostinger Ubuntu)"
echo "=========================================="
echo ""

# ------------------------------------------------------------------
# Step 1: System packages
# ------------------------------------------------------------------
echo "[1/7] Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
    software-properties-common \
    git curl wget unzip \
    build-essential \
    libffi-dev libssl-dev \
    sqlite3 \
    > /dev/null 2>&1
echo "      Done."

# ------------------------------------------------------------------
# Step 2: Install Python 3.12
# ------------------------------------------------------------------
echo "[2/7] Installing Python ${PYTHON_VERSION}..."
if ! command -v python${PYTHON_VERSION} &> /dev/null; then
    add-apt-repository -y ppa:deadsnakes/ppa > /dev/null 2>&1
    apt-get update -qq
    apt-get install -y -qq \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-dev \
        > /dev/null 2>&1
    echo "      Python ${PYTHON_VERSION} installed."
else
    echo "      Python ${PYTHON_VERSION} already installed."
fi

# ------------------------------------------------------------------
# Step 3: Clone repository
# ------------------------------------------------------------------
echo "[3/7] Cloning repository..."
if [ -d "$APP_DIR" ]; then
    echo "      Directory exists — pulling latest..."
    cd "$APP_DIR"
    git pull origin master
else
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
fi
echo "      Done."

# ------------------------------------------------------------------
# Step 4: Create virtual environment + install dependencies
# ------------------------------------------------------------------
echo "[4/7] Setting up Python virtual environment..."
cd "$APP_DIR"
python${PYTHON_VERSION} -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel -q
echo "      Installing dependencies (this takes 2-5 minutes)..."
pip install -r requirements.txt -q
echo "      Done."

# ------------------------------------------------------------------
# Step 5: Create data directory
# ------------------------------------------------------------------
echo "[5/7] Creating data directories..."
mkdir -p "$APP_DIR/data"
mkdir -p "$APP_DIR/models"
echo "      Done."

# ------------------------------------------------------------------
# Step 6: Create .env from template
# ------------------------------------------------------------------
echo "[6/7] Setting up environment file..."
if [ ! -f "$APP_DIR/.env" ]; then
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    echo "      Created .env from template."
    echo ""
    echo "  >>> IMPORTANT: Edit your API keys now:"
    echo "      nano /root/ai-trade-bot/.env"
    echo ""
    echo "  Required keys for full functionality:"
    echo "    - BINANCE_API_KEY (read-only for market data)"
    echo "    - BINANCE_SECRET_KEY"
    echo "    - TWELVE_DATA_API_KEY (for forex/metals)"
    echo "    - NEWSAPI_KEY (for news sentiment)"
    echo ""
else
    echo "      .env already exists — skipping."
fi

# ------------------------------------------------------------------
# Step 7: Install systemd service
# ------------------------------------------------------------------
echo "[7/7] Installing systemd service..."
cp "$APP_DIR/scripts/ai-trade-bot.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable ai-trade-bot
echo "      Service installed and enabled (auto-starts on boot)."

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "  Next steps:"
echo ""
echo "  1. Edit API keys:"
echo "     nano /root/ai-trade-bot/.env"
echo ""
echo "  2. (Optional) Copy trained models from local machine:"
echo "     # Run this FROM YOUR LOCAL MACHINE:"
echo "     rsync -avz models/ root@YOUR_VPS_IP:/root/ai-trade-bot/models/"
echo ""
echo "  3. Start the bot:"
echo "     systemctl start ai-trade-bot"
echo ""
echo "  4. Watch logs:"
echo "     journalctl -u ai-trade-bot -f"
echo ""
echo "  5. Check health:"
echo "     cd /root/ai-trade-bot && bash scripts/health_check.sh"
echo ""
