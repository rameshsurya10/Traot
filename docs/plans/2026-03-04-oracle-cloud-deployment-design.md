# Oracle Cloud Always Free — Continuous Deployment Design

**Date:** 2026-03-04
**Goal:** Run the AI Trade Bot 24/7 on Oracle Cloud free tier, collecting paper trade data continuously
**Budget:** $0 (Oracle Always Free — no card, phone + address verification only)

---

## Context

The bot currently runs manually on a local development machine, generating 1 signal in 5 days.
The goal is continuous runtime to accumulate paper trade outcomes for the CalibrationAnalyzer
(needs 50+ closed outcomes before calibration data is meaningful).

---

## Chosen Platform: Oracle Cloud Always Free ARM

| Property | Value |
|----------|-------|
| Instance type | Ampere A1 Flex (ARM) |
| OCPUs | 4 |
| RAM | 24 GB |
| Storage | 50 GB boot volume |
| OS | Ubuntu 22.04 LTS |
| Cost | $0 — permanent Always Free, not a trial |
| Sleep behavior | None — full VM, no idle timeout |
| Signup requirement | Email + phone + home address (no card) |

Oracle ARM instances are significantly more powerful than the bot requires.
PyTorch CPU-only with SQLite needs ~1-2GB RAM under load — well within the 24GB free allocation.

---

## Architecture

```
Local machine (development)
    │
    │  git push
    ▼
GitHub repository
    │
    │  git pull (manual, SSH)
    ▼
Oracle Cloud VM (Ubuntu 22.04)
    ├── /home/ubuntu/ai-trade-bot/          ← application code
    │       ├── run_trading.py
    │       ├── config.yaml
    │       ├── .env                        ← API keys (never in git)
    │       ├── models/                     ← LSTM + boosted models (rsync once)
    │       └── data/trading.db             ← SQLite DB (rsync once, grows in place)
    │
    └── systemd: ai-trade-bot.service
            Restart=always
            RestartSec=30
```

---

## Deployment Workflow

### Initial Setup (one-time)
1. Create Oracle account (cloud.oracle.com) — email, phone, address, no card
2. Create ARM A1 Flex instance (Ubuntu 22.04, 4 OCPU / 24GB RAM)
3. Open inbound ports: 22 (SSH) via Oracle Security List
4. SSH into VM, install Python 3.12 + pip + venv
5. Clone repo from GitHub onto VM
6. `rsync` existing models/ and data/ from local to VM (preserves trained models and DB)
7. Create `.env` file on VM with API keys
8. Install Python dependencies in venv
9. Create and enable systemd service

### Code Update Workflow (ongoing)
```bash
# Local machine: push changes
git push origin master

# Oracle VM: pull and restart
ssh ubuntu@<oracle-ip>
cd /home/ubuntu/ai-trade-bot
git pull origin master
sudo systemctl restart ai-trade-bot
sudo systemctl status ai-trade-bot
```

---

## systemd Service

File: `/etc/systemd/system/ai-trade-bot.service`

```ini
[Unit]
Description=AI Trade Bot — Continuous Learning
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

[Install]
WantedBy=multi-user.target
```

Key properties:
- `Restart=always` — restarts on any exit (crash, OOM, exception)
- `RestartSec=30` — 30-second delay prevents rapid restart loops
- `EnvironmentFile` — loads `.env` into the process environment
- `After=network-online.target` — waits for network before starting (critical for API connections)

---

## Oracle Idle Prevention

Oracle reclaims ARM instances with CPU < ~10% for 7 consecutive days.

The trading bot's continuous candle processing loop (every 15 minutes for crypto,
every 1 hour for forex) naturally generates CPU activity. As a safeguard, a lightweight
idle prevention script runs as a separate low-priority systemd service:

```bash
# Runs a harmless computation every 4 hours to ensure CPU activity is measurable
# Uses at most 0.1% CPU on average — no impact on bot performance
```

---

## Persistence Strategy

### First Deploy (one-time rsync)
```bash
# Transfer trained models from local to Oracle
rsync -avz models/ ubuntu@<oracle-ip>:/home/ubuntu/ai-trade-bot/models/

# Transfer existing SQLite DB (keeps historical candles and any recorded outcomes)
rsync -avz data/ ubuntu@<oracle-ip>:/home/ubuntu/ai-trade-bot/data/
```

### Ongoing
- SQLite DB grows in place on Oracle's 50GB disk
- Models are updated by the retraining engine in-place
- No backup strategy required for the research phase (DB can be rebuilt from exchange data)

---

## Monitoring

Telegram notifications are already configured in the bot (`config.yaml → notifications`).
The bot sends alerts on:
- Trading signal generated
- Trade outcome recorded
- System errors

No additional monitoring infrastructure needed for research phase.

---

## What Changes in the Bot

Nothing. The bot runs identically on Oracle ARM as on the local machine:
- `python3 run_trading.py` is the entry point
- `live_trading.mode: paper` in config.yaml remains — no live capital at risk
- All API keys passed via `.env`
- SQLite path unchanged

---

## Success Criteria

After deployment, within 72 hours:
- [ ] `systemctl status ai-trade-bot` shows `active (running)`
- [ ] `journalctl -u ai-trade-bot -f` shows live candle processing
- [ ] At least 5 new signals appear in `signals` table
- [ ] At least 1 signal exceeds 80% confidence (visible in confidence distribution)

After 7 days:
- [ ] `trade_outcomes` table has 10+ closed outcomes with `was_correct IS NOT NULL`
- [ ] `python3 -m src.analysis.calibration_analyzer --min-trades 10` runs without "no data" error

---

## Migration Path from Local (if bot is currently running locally)

1. Stop local bot (if running)
2. Final rsync of DB and models to Oracle
3. Start Oracle systemd service
4. Verify Oracle is running before closing local terminal
5. Local machine is now just development — Oracle is production
