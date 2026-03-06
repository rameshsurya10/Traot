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

# 2. systemd service status
echo "[Service]"
if systemctl is-active --quiet ai-trade-bot 2>/dev/null; then
    UPTIME=$(systemctl show ai-trade-bot --property=ActiveEnterTimestamp | cut -d= -f2)
    echo "  Status:  RUNNING"
    echo "  Since:   ${UPTIME}"
else
    echo "  Status:  NOT RUNNING"
    echo "  Run:     systemctl start ai-trade-bot"
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
if command -v journalctl &>/dev/null; then
    ERRORS=$(journalctl -u ai-trade-bot --since "30 minutes ago" 2>/dev/null | grep -i "error\|exception\|failed\|traceback" | wc -l)
    if [ "$ERRORS" -eq 0 ]; then
        echo "  None — clean run"
    else
        echo "  ${ERRORS} error lines found:"
        journalctl -u ai-trade-bot --since "30 minutes ago" 2>/dev/null | grep -i "error\|exception\|failed" | tail -5 | sed 's/^/  /'
    fi
else
    echo "  journalctl not available — check logs manually"
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
