# Oracle Cloud Always Free — Deployment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deploy the AI Trade Bot to an Oracle Cloud ARM VM so it runs 24/7 as a systemd service, accumulating paper trade outcomes continuously.

**Architecture:** Oracle Always Free ARM A1 Flex VM (4 OCPU / 24GB RAM / Ubuntu 22.04) runs `run_trading.py` as a systemd service with `Restart=always`. Code is pulled from GitHub; trained models and SQLite DB are transferred once via rsync then grow in place on the Oracle disk.

**Tech Stack:** Oracle Cloud Infrastructure, Ubuntu 22.04 LTS, Python 3.12, systemd, rsync, SSH

---

## Prerequisites (human steps — cannot be automated)

These must be done manually by the user before any task below:

**P1. Create Oracle Cloud account**
- Go to: https://cloud.oracle.com/
- Sign up with: email + phone number + home address (no credit card needed)
- Select "Always Free" resources only during signup
- Choose home region: closest to you (e.g., ap-mumbai-1 for India)
- Verification takes 5–15 minutes

**P2. Create ARM Instance**
- Dashboard → Compute → Instances → Create Instance
- Name: `ai-trade-bot`
- Image: Canonical Ubuntu 22.04 (Minimal)
- Shape: Ampere → VM.Standard.A1.Flex
- OCPUs: 4, Memory: 24 GB (maximum free allocation)
- Networking: Default VCN, public subnet, assign public IP
- SSH keys: Add your public key (`cat ~/.ssh/id_rsa.pub` or generate one)
- Click Create — instance will be `RUNNING` in 2–3 minutes

**P3. Open SSH port in Security List**
- Networking → Virtual Cloud Networks → your VCN → Security Lists → Default
- Add Ingress Rule: Source CIDR `0.0.0.0/0`, Protocol TCP, Port 22
- (Port 22 may already be open — check before adding)

**P4. Note the instance's public IP**
- Compute → Instances → ai-trade-bot → Public IP address
- Save this: `ORACLE_IP=<your.ip.here>`

---

## Task 1: Verify SSH Access

**Goal:** Confirm you can SSH into the Oracle VM before doing anything else.

**Files:** None (infrastructure verification)

**Step 1: Test SSH connection**
```bash
# From your LOCAL machine
ssh ubuntu@<ORACLE_IP> "echo 'SSH works' && uname -m"
```
Expected output:
```
SSH works
aarch64
```
`aarch64` confirms you're on ARM.

**Step 2: If SSH fails — check these**
```bash
# 1. Make sure your private key permissions are correct
chmod 600 ~/.ssh/id_rsa

# 2. Try with explicit key
ssh -i ~/.ssh/id_rsa ubuntu@<ORACLE_IP> "echo ok"

# 3. If still failing: check Oracle Security List has port 22 open (Prerequisite P3)
```

**Step 3: No commit needed** — this is infrastructure verification only.

---

## Task 2: Install Python and System Dependencies on VM

**Goal:** Get Python 3.12, pip, venv, and git installed on the Oracle VM.

**Step 1: SSH into VM and update package list**
```bash
ssh ubuntu@<ORACLE_IP>
sudo apt update && sudo apt upgrade -y
```
Expected: Package list updates, upgrades apply. Takes 2–5 minutes.

**Step 2: Install Python 3.12 and tools**
```bash
sudo apt install -y python3.12 python3.12-venv python3.12-dev python3-pip git build-essential
```
Expected: Packages install without errors.

**Step 3: Verify Python version**
```bash
python3.12 --version
python3.12 -m venv --help | head -1
```
Expected:
```
Python 3.12.x
usage: venv [-h] ...
```

**Step 4: No commit needed** — VM setup, not code change.

---

## Task 3: Clone Repository and Create Virtual Environment

**Goal:** Get the bot's code onto the Oracle VM and install all Python dependencies.

**Step 1: Clone the repository (on VM)**
```bash
# Still SSH'd into the Oracle VM
cd ~
git clone https://github.com/<your-username>/Ai-Trade-Bot.git ai-trade-bot
cd ai-trade-bot
git log --oneline -3
```
Expected: Repository clones, last 3 commits shown.

**Step 2: Create virtual environment**
```bash
python3.12 -m venv venv
source venv/bin/activate
python --version
```
Expected: `Python 3.12.x`

**Step 3: Install dependencies**
```bash
# This takes 5–15 minutes on first run (PyTorch + scipy + lightgbm + ccxt etc.)
pip install --upgrade pip
pip install -r requirements.txt
```
Expected: All packages install without errors. PyTorch will install the ARM wheel automatically.

**Step 4: Verify critical imports**
```bash
python -c "import torch, xgboost, lightgbm, ccxt, scipy; print('All imports OK')"
python -c "import torch; print('PyTorch:', torch.__version__)"
```
Expected:
```
All imports OK
PyTorch: 2.x.x+cpu
```

**Step 5: No commit needed** — VM setup only.

---

## Task 4: Transfer Models, Database, and Environment File

**Goal:** Move trained models and SQLite DB from local machine to Oracle VM so the bot doesn't retrain from scratch.

**Step 1: Transfer from LOCAL machine (open a NEW terminal on your local machine)**
```bash
# Run these from your LOCAL machine, not the Oracle VM

# Transfer trained LSTM and boosted models (6.2MB)
rsync -avz --progress \
  /home/development1/Desktop/Ai-Trade-Bot/models/ \
  ubuntu@<ORACLE_IP>:/home/ubuntu/ai-trade-bot/models/

# Transfer SQLite database with historical candles (18MB)
rsync -avz --progress \
  /home/development1/Desktop/Ai-Trade-Bot/data/ \
  ubuntu@<ORACLE_IP>:/home/ubuntu/ai-trade-bot/data/
```
Expected: Files transfer, rsync shows progress. Takes < 1 minute on decent connection.

**Step 2: Verify transfer on VM**
```bash
# Back in SSH session on Oracle VM
ls -la ~/ai-trade-bot/models/
ls -la ~/ai-trade-bot/data/
```
Expected:
```
models/model_BTC_USDT_15m.pt   (875 KB)
models/model_BTC_USDT_1h.pt    (875 KB)
models/model_ETH_USDT_15m.pt   (875 KB)
models/model_ETH_USDT_1h.pt    (875 KB)
models/boosted_BTC_USDT.joblib  (89 KB)
models/boosted_ETH_USDT.joblib  (87 KB)
data/trading.db
```

**Step 3: Create .env file on VM**
```bash
# On Oracle VM — copy your real API keys
nano ~/ai-trade-bot/.env
```
Add all keys from your local `.env` file:
```ini
BINANCE_API_KEY=your_actual_key
BINANCE_SECRET_KEY=your_actual_secret
TWELVE_DATA_API_KEY=your_actual_key
GEMINI_API_KEY=your_actual_key_if_using_llm
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```
Save with Ctrl+O, exit with Ctrl+X.

**Step 4: Verify .env is readable**
```bash
cat ~/ai-trade-bot/.env | head -3
# Should show your first few keys (not blank)
```

**Step 5: Quick smoke test — can the bot load?**
```bash
cd ~/ai-trade-bot
source venv/bin/activate
python -c "
from src.core.config import Config
from src.core.database import Database
cfg = Config.load()
db = Database('data/trading.db')
print('Config loaded:', cfg.exchange)
print('DB candles:', db.connection().__enter__().execute('SELECT COUNT(*) FROM candles').fetchone()[0])
"
```
Expected:
```
Config loaded: binance
DB candles: 87600  (or similar large number)
```

**Step 6: No commit needed** — transferring data, not code.

---

## Task 5: Create systemd Service

**Goal:** Create the systemd service that runs the bot permanently with auto-restart.

**Step 1: Create the service file (on Oracle VM)**
```bash
sudo nano /etc/systemd/system/ai-trade-bot.service
```
Paste exactly:
```ini
[Unit]
Description=AI Trade Bot — Continuous Learning
Documentation=https://github.com/<your-username>/Ai-Trade-Bot
After=network-online.target
Wants=network-online.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/ai-trade-bot
EnvironmentFile=/home/ubuntu/ai-trade-bot/.env
ExecStart=/home/ubuntu/ai-trade-bot/venv/bin/python3 run_trading.py
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal
SyslogIdentifier=ai-trade-bot

[Install]
WantedBy=multi-user.target
```
Save with Ctrl+O, exit with Ctrl+X.

**Step 2: Reload systemd and enable service**
```bash
sudo systemctl daemon-reload
sudo systemctl enable ai-trade-bot
```
Expected: `Created symlink /etc/systemd/system/multi-user.target.wants/ai-trade-bot.service`

**Step 3: Start the service**
```bash
sudo systemctl start ai-trade-bot
```

**Step 4: Verify it started**
```bash
sudo systemctl status ai-trade-bot
```
Expected:
```
● ai-trade-bot.service - AI Trade Bot — Continuous Learning
     Loaded: loaded (/etc/systemd/system/ai-trade-bot.service; enabled)
     Active: active (running) since ...
   Main PID: XXXX (python3)
```

**Step 5: Watch live logs for 60 seconds**
```bash
journalctl -u ai-trade-bot -f --no-pager
# Watch for 60 seconds — you should see startup logs, model loading, candle processing
# Press Ctrl+C to stop watching (service keeps running)
```
Expected to see:
- `Loading models...`
- `Starting LiveTradingRunner`
- `Backfilling data...`
- `Starting streams...`

**Step 6: No commit needed** — systemd configuration is infrastructure.

---

## Task 6: Create Oracle Idle Prevention Service

**Goal:** Prevent Oracle from reclaiming the VM by ensuring visible CPU activity every 4 hours.

**Background:** Oracle monitors CPU at the hypervisor level. If average CPU < ~10% for 7 consecutive days, they send a warning then reclaim. The trading bot's candle processing naturally generates CPU spikes, but to be safe, a minimal background job ensures activity is always measurable.

**Step 1: Create the idle prevention script on VM**
```bash
sudo nano /usr/local/bin/oracle-keepalive.sh
```
Paste:
```bash
#!/bin/bash
# Oracle Cloud idle prevention — runs lightweight CPU work every 4 hours
# Prevents Oracle from reclaiming "idle" Always Free instances
# CPU impact: < 0.1% average

while true; do
    # Compress /dev/urandom output for ~5 seconds — generates measurable CPU
    dd if=/dev/urandom bs=1M count=50 2>/dev/null | gzip > /dev/null
    # Sleep 4 hours
    sleep 14400
done
```
Save with Ctrl+O, exit with Ctrl+X.

**Step 2: Make it executable**
```bash
sudo chmod +x /usr/local/bin/oracle-keepalive.sh
```

**Step 3: Create systemd service for it**
```bash
sudo nano /etc/systemd/system/oracle-keepalive.service
```
Paste:
```ini
[Unit]
Description=Oracle Cloud Idle Prevention
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/oracle-keepalive.sh
Restart=always
RestartSec=60
Nice=19
IOSchedulingClass=idle

[Install]
WantedBy=multi-user.target
```
`Nice=19` and `IOSchedulingClass=idle` ensure this never competes with the trading bot.

**Step 4: Enable and start**
```bash
sudo systemctl daemon-reload
sudo systemctl enable oracle-keepalive
sudo systemctl start oracle-keepalive
sudo systemctl status oracle-keepalive
```
Expected: `active (running)`

---

## Task 7: Verify Full System and Run Health Check

**Goal:** Confirm the entire system is working — bot running, DB growing, signals being generated.

**Step 1: Check both services are running**
```bash
sudo systemctl status ai-trade-bot oracle-keepalive
```
Expected: Both show `active (running)`.

**Step 2: Check DB is being written to (wait 15+ minutes after start)**
```bash
cd ~/ai-trade-bot
source venv/bin/activate
python3 -c "
from src.core.database import Database
import sqlite3
db = Database('data/trading.db')
with db.connection() as conn:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) as total FROM signals')
    total = cur.fetchone()['total']
    cur.execute('SELECT COUNT(*) as total FROM trade_outcomes WHERE was_correct IS NOT NULL')
    outcomes = cur.fetchone()['total']
    cur.execute('SELECT MAX(confidence), AVG(confidence) FROM signals')
    row = cur.fetchone()
    print(f'Total signals: {total}')
    print(f'Closed outcomes: {outcomes}')
    if row[0]:
        print(f'Max confidence: {row[0]:.1%}')
        print(f'Avg confidence: {row[1]:.1%}')
"
```
Expected after 1 hour of runtime:
```
Total signals: 2+
Closed outcomes: 0+
Max confidence: depends on market
```

**Step 3: Check log for errors**
```bash
journalctl -u ai-trade-bot --since "1 hour ago" | grep -i "error\|exception\|failed" | head -20
```
Expected: Empty (no errors). If you see errors, check the error message — common causes:
- `BINANCE_API_KEY not set` → re-check your .env file
- `Connection refused` → network issue, wait for retry
- `ModuleNotFoundError` → re-run `pip install -r requirements.txt`

**Step 4: Confirm service survives if you disconnect**
```bash
# Disconnect SSH entirely
exit

# Wait 5 minutes, then reconnect
ssh ubuntu@<ORACLE_IP>
sudo systemctl status ai-trade-bot
```
Expected: Still `active (running)` — proves it doesn't depend on your SSH session.

---

## Task 8: Update Code Deployment Workflow (Document for Future Use)

**Goal:** Document the process for deploying code changes so you have a repeatable workflow.

**Step 1: Create a deployment helper script on LOCAL machine**
```bash
nano /home/development1/Desktop/Ai-Trade-Bot/scripts/deploy.sh
```
Paste:
```bash
#!/bin/bash
# Deploy latest code to Oracle VM
# Usage: ./scripts/deploy.sh <oracle-ip>

ORACLE_IP=${1:-"<YOUR_ORACLE_IP>"}
REMOTE_PATH="/home/ubuntu/ai-trade-bot"

echo "Deploying to Oracle VM at $ORACLE_IP..."

# Push local changes to GitHub first
git push origin master

# Pull on Oracle and restart
ssh ubuntu@$ORACLE_IP "
  cd $REMOTE_PATH &&
  git pull origin master &&
  source venv/bin/activate &&
  pip install -r requirements.txt --quiet &&
  sudo systemctl restart ai-trade-bot &&
  sudo systemctl status ai-trade-bot --no-pager
"

echo "Deployment complete."
```

**Step 2: Make it executable**
```bash
chmod +x /home/development1/Desktop/Ai-Trade-Bot/scripts/deploy.sh
```

**Step 3: Test the deploy script**
```bash
cd /home/development1/Desktop/Ai-Trade-Bot
./scripts/deploy.sh <ORACLE_IP>
```
Expected: Git push, git pull on Oracle, pip check, systemctl restart, status shows running.

**Step 4: Commit the deploy script**
```bash
git add scripts/deploy.sh
git commit -m "chore(deploy): add Oracle Cloud deployment script"
git push origin master
```

---

## Success Criteria Checklist

Run this after 24 hours of the bot running on Oracle:

```bash
ssh ubuntu@<ORACLE_IP>
cd ~/ai-trade-bot
source venv/bin/activate

python3 -c "
from src.core.database import Database
import sqlite3
db = Database('data/trading.db')
with db.connection() as conn:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    checks = [
        ('systemd service active', None),  # checked externally
        ('signals in DB', 'SELECT COUNT(*) FROM signals'),
        ('candles being added', 'SELECT COUNT(*) FROM candles'),
        ('trade_outcomes growing', 'SELECT COUNT(*) FROM trade_outcomes'),
    ]
    for label, query in checks:
        if query:
            cur.execute(query)
            count = cur.fetchone()[0]
            status = '✓' if count > 0 else '✗'
            print(f'{status} {label}: {count}')
"

# Check uptime
systemctl show ai-trade-bot --property=ActiveEnterTimestamp
```

**Target by Monday 2026-03-09:**
- [ ] `active (running)` for 24+ hours without restart
- [ ] 10+ signals recorded
- [ ] 0 unhandled exceptions in logs
- [ ] Oracle idle prevention service active

**Target by end of week:**
- [ ] 50+ closed trade outcomes
- [ ] CalibrationAnalyzer shows real data: `python3 -m src.analysis.calibration_analyzer --min-trades 10`

---

## Troubleshooting Reference

| Problem | Command | Fix |
|---------|---------|-----|
| Bot crashed | `journalctl -u ai-trade-bot -n 50` | Check error, likely API key or network |
| API key missing | `systemctl cat ai-trade-bot` | Verify EnvironmentFile path |
| Port blocked | `sudo ufw status` | `sudo ufw allow 22` |
| Out of memory | `free -h` | Should not happen on 24GB — check for memory leak |
| Oracle reclamation warning | Email from Oracle | Keepalive service + verify bot is running |
| Can't SSH | Oracle Console → Cloud Shell | Use browser SSH as backup |
