#!/bin/bash
# Traot — Start/Restart Script
# Ensures only ONE instance runs at a time.

PIDFILE="/root/Traot/data/traot.pid"
LOGFILE="/root/Traot/data/trading.log"
PYTHON="/root/Traot/venv/bin/python3"
SCRIPT="/root/Traot/run_trading.py"

# Kill any existing instances
if [ -f "$PIDFILE" ]; then
    OLD_PID=$(cat "$PIDFILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Stopping existing instance (PID $OLD_PID)..."
        kill -9 "$OLD_PID" 2>/dev/null
        sleep 2
    fi
    rm -f "$PIDFILE"
fi

# Also kill any stray processes
pkill -9 -f "python3.*run_trading.py" 2>/dev/null
sleep 3

# Verify nothing is running
RUNNING=$(pgrep -f "run_trading.py" | wc -l)
if [ "$RUNNING" -gt 0 ]; then
    echo "ERROR: Could not kill all instances. PIDs:"
    pgrep -f "run_trading.py"
    exit 1
fi

# Start new instance
echo "Starting Traot..."
nohup $PYTHON $SCRIPT >> "$LOGFILE" 2>&1 &
NEW_PID=$!
echo $NEW_PID > "$PIDFILE"

sleep 2

# Verify it started
if kill -0 "$NEW_PID" 2>/dev/null; then
    echo "Traot started (PID $NEW_PID)"
    echo "Logs: tail -f $LOGFILE"
else
    echo "ERROR: Traot failed to start"
    tail -20 "$LOGFILE"
    exit 1
fi
