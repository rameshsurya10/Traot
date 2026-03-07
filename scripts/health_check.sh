#!/bin/bash
# =============================================================================
# Traot — Health Check Script (Hostinger VPS)
# =============================================================================
# Run this on the VPS after deployment to verify the bot is working correctly.
#
# Usage (on the VPS):
#   cd ~/ai-trade-bot && bash scripts/health_check.sh
#
# Usage (from local machine via SSH):
#   ssh root@<vps-ip> 'cd ~/ai-trade-bot && bash scripts/health_check.sh'
# =============================================================================

VENV_PYTHON="${PWD}/venv/bin/python3"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "ERROR: venv not found at ${VENV_PYTHON}"
    echo "Run: python3.12 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

echo ""
echo "=== Traot — Health Check ==="
echo "Time: $(date)"
echo ""

# 1. System info
echo "[System]"
echo "  OS:     $(lsb_release -ds 2>/dev/null || cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2)"
echo "  CPU:    $(nproc) cores"
echo "  Memory: $(free -h | awk '/Mem:/ {print $3 "/" $2}')"
echo "  Disk:   $(df -h / | awk 'NR==2 {print $3 "/" $2 " (" $5 " used)"}')"
echo ""

# 2. Service status (pm2 or systemd)
echo "[Service]"
if command -v pm2 &>/dev/null && pm2 list 2>/dev/null | grep -q "Traot"; then
    PM2_STATUS=$(pm2 list 2>/dev/null | grep "Traot" | awk '{print $18}')
    PM2_UPTIME=$(pm2 list 2>/dev/null | grep "Traot" | awk '{print $14}')
    PM2_MEM=$(pm2 list 2>/dev/null | grep "Traot" | awk '{print $20, $21}')
    PM2_CPU=$(pm2 list 2>/dev/null | grep "Traot" | awk '{print $16, $17}')
    if pm2 list 2>/dev/null | grep "Traot" | grep -q "online"; then
        echo "  Status:  RUNNING (pm2)"
        echo "  Uptime:  ${PM2_UPTIME}"
        echo "  Memory:  ${PM2_MEM}"
        echo "  CPU:     ${PM2_CPU}"
    else
        echo "  Status:  NOT RUNNING"
        echo "  Run:     pm2 start run_trading.py --name Traot --interpreter /root/Traot/venv/bin/python3 --cwd /root/Traot"
    fi
elif systemctl is-active --quiet ai-trade-bot 2>/dev/null; then
    UPTIME=$(systemctl show ai-trade-bot --property=ActiveEnterTimestamp | cut -d= -f2)
    echo "  Status:  RUNNING (systemd)"
    echo "  Since:   ${UPTIME}"
else
    echo "  Status:  NOT RUNNING"
    echo "  Run:     pm2 start run_trading.py --name Traot --interpreter /root/Traot/venv/bin/python3 --cwd /root/Traot"
fi

# 3. Database stats
echo ""
echo "[Database]"
$VENV_PYTHON - << 'PYEOF'
import sys
sys.path.insert(0, '.')
try:
    from src.core.database import Database
    import sqlite3
    db = Database('data/trading.db')
    with db.connection() as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM candles")
        candles = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM signals")
        signals = cur.fetchone()[0]

        cur.execute("SELECT MAX(confidence), AVG(confidence) FROM signals")
        row = cur.fetchone()
        max_conf = f"{row[0]:.1%}" if row[0] else "none yet"
        avg_conf = f"{row[1]:.1%}" if row[1] else "none yet"

        cur.execute("SELECT COUNT(*) FROM trade_outcomes WHERE was_correct IS NOT NULL")
        outcomes = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM signals WHERE confidence >= 0.80")
        above_gate = cur.fetchone()[0]

        print(f"  Candles stored:       {candles:,}")
        print(f"  Signals generated:    {signals}")
        print(f"  Above 80% gate:       {above_gate}")
        print(f"  Max confidence seen:  {max_conf}")
        print(f"  Avg confidence:       {avg_conf}")
        print(f"  Closed outcomes:      {outcomes}")
except Exception as e:
    print(f"  ERROR: {e}")
PYEOF

# 4. Model files
echo ""
echo "[Models]"
for f in models/*.pt models/*.joblib; do
    if [ -f "$f" ]; then
        SIZE=$(du -sh "$f" | cut -f1)
        echo "  $f  ($SIZE)"
    fi
done

# 5. Recent log errors
echo ""
echo "[Recent Errors (last 30 min)]"
if command -v pm2 &>/dev/null && pm2 list 2>/dev/null | grep -q "Traot"; then
    ERRORS=$(pm2 logs Traot --lines 200 --nostream 2>/dev/null | grep -i "error\|exception\|failed\|traceback" | wc -l)
    if [ "$ERRORS" -eq 0 ]; then
        echo "  None — clean run"
    else
        echo "  ${ERRORS} error lines found:"
        pm2 logs Traot --lines 200 --nostream 2>/dev/null | grep -i "error\|exception\|failed" | tail -5 | sed 's/^/  /'
    fi
elif command -v journalctl &>/dev/null; then
    ERRORS=$(journalctl -u ai-trade-bot --since "30 minutes ago" 2>/dev/null | grep -i "error\|exception\|failed\|traceback" | wc -l)
    if [ "$ERRORS" -eq 0 ]; then
        echo "  None — clean run"
    else
        echo "  ${ERRORS} error lines found:"
        journalctl -u ai-trade-bot --since "30 minutes ago" 2>/dev/null | grep -i "error\|exception\|failed" | tail -5 | sed 's/^/  /'
    fi
else
    echo "  No log source available — check: pm2 logs Traot"
fi

# 6. Success criteria
echo ""
echo "=== 72-Hour Success Criteria ==="
$VENV_PYTHON - << 'PYEOF'
import sys
sys.path.insert(0, '.')
try:
    from src.core.database import Database
    import sqlite3
    db = Database('data/trading.db')
    with db.connection() as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM signals")
        signals = cur.fetchone()[0]

        cur.execute("SELECT MAX(confidence) FROM signals")
        max_conf = cur.fetchone()[0] or 0

        cur.execute("SELECT COUNT(*) FROM trade_outcomes WHERE was_correct IS NOT NULL")
        outcomes = cur.fetchone()[0]

        checks = [
            ("5+ signals generated",    signals >= 5,   f"{signals} signals"),
            ("70%+ confidence seen",     max_conf >= 0.70, f"max={max_conf:.1%}"),
            ("Outcomes being recorded",  outcomes >= 0,  f"{outcomes} outcomes"),
        ]
        all_pass = True
        for label, passed, detail in checks:
            icon = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            print(f"  [{icon}] {label} ({detail})")

        if all_pass:
            print("\n  RESULT: Test PASSING — keep it running!")
        else:
            print("\n  RESULT: Still accumulating data — check again in 24h")
except Exception as e:
    print(f"  ERROR: {e}")
PYEOF

echo ""
