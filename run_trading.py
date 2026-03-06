#!/usr/bin/env python3
"""
Start Traot with Continuous Learning
============================================

This script starts the LiveTradingRunner with full continuous learning enabled.

Features:
- Automatic model training on startup (using 1-year data from database)
- Continuous learning from every candle
- Automatic retraining when performance drops
- Multi-timeframe analysis (15m, 1h, 4h, 1d)
- Strategy discovery (run analyze_strategies.py to see results)

Usage:
    python run_trading.py
"""

import logging
import os
import signal
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

from src.live_trading.runner import LiveTradingRunner, TradingMode
from src.core.config import Config

# Resolve trading mode: env var TRADING_MODE overrides default
# Valid values: paper (default), live, shadow
_MODE_MAP = {
    'paper': TradingMode.PAPER,
    'shadow': TradingMode.SHADOW,
}

# SAFETY: Live trading is disabled. Remove this block when ready to go live.
_LIVE_TRADING_ENABLED = False

def _resolve_mode() -> TradingMode:
    env_mode = os.environ.get('TRADING_MODE', 'paper').lower().strip()
    if env_mode == 'live':
        if not _LIVE_TRADING_ENABLED:
            print("BLOCKED: Live trading is disabled. Set _LIVE_TRADING_ENABLED = True in run_trading.py to enable.")
            print("Falling back to PAPER mode.")
            return TradingMode.PAPER
        return TradingMode.LIVE
    mode = _MODE_MAP.get(env_mode)
    if mode is None:
        print(f"WARNING: Unknown TRADING_MODE='{env_mode}', falling back to PAPER")
        return TradingMode.PAPER
    return mode

def main():
    mode = _resolve_mode()

    print("="*70)
    print(f"AI TRADE BOT - CONTINUOUS LEARNING MODE ({mode.value.upper()})")
    print("="*70)
    print()
    print("Features:")
    print("  - Automatic training on 1-year historical data")
    print("  - Continuous learning from every trade")
    print("  - Automatic retraining when accuracy drops")
    print("  - Multi-timeframe analysis (15m, 1h, 4h, 1d)")
    print("  - Strategy discovery and comparison")
    print("  - Crypto (Binance) + Forex/Metals (Twelve Data)")
    print()
    print(f"  Mode: {mode.value.upper()}")
    print(f"  (Set TRADING_MODE=paper|live|shadow in .env to change)")
    print()

    # Initialize runner
    print("Initializing LiveTradingRunner...")
    runner = LiveTradingRunner(
        config_path="config.yaml",
        mode=mode,
    )

    config = Config.load("config.yaml")

    # Add crypto symbols (Binance)
    print("Adding crypto symbols (Binance)...")
    runner.add_symbol("BTC/USDT", exchange="binance", interval="1h")
    runner.add_symbol("ETH/USDT", exchange="binance", interval="1h")

    # Add forex/metals symbols from Twelve Data config
    twelvedata_config = config.raw.get('twelvedata', {})
    if twelvedata_config.get('enabled', False):
        print("Adding forex/metals symbols (Twelve Data)...")
        for pair in twelvedata_config.get('pairs', ['EUR/USD', 'XAU/USD']):
            runner.add_symbol(pair, exchange="twelvedata", interval="1h")
            print(f"  + {pair}")
    else:
        print("Twelve Data disabled in config")

    print()
    print("="*70)
    print("✅ Configuration complete!")
    print("="*70)
    print()
    print("What happens now:")
    print("  1. Loads/trains models for all symbols and timeframes")
    print("  2. Connects to Binance WebSocket (crypto) + Twelve Data REST (forex)")
    print("  3. Makes predictions on every candle close")
    print("  4. Executes paper trades when confidence is high")
    print("  5. Records outcomes and retrains when needed")
    print()
    print("To analyze discovered strategies later, run:")
    print("  python scripts/analyze_strategies.py")
    print()
    print("="*70)
    print("Starting trading... (Press Ctrl+C to stop)")
    print("="*70)
    print()

    # Register signal handlers for graceful shutdown (systemd sends SIGTERM)
    _shutdown_requested = False

    def _shutdown_handler(signum, frame):
        nonlocal _shutdown_requested
        sig_name = signal.Signals(signum).name
        if _shutdown_requested:
            logger.warning(f"Second {sig_name} received — forcing exit")
            sys.exit(1)
        _shutdown_requested = True
        logger.info(f"Received {sig_name} — shutting down gracefully...")
        runner.stop()

    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    # Start trading (blocking - runs until signal or error)
    try:
        runner.start(blocking=True)
    except KeyboardInterrupt:
        pass  # Handled by signal handler above

    if not _shutdown_requested:
        # start() returned without a signal — likely an error exit
        logger.info("Runner exited — stopping...")
        runner.stop()

    logger.info("Stopped gracefully")
    print()
    print("To see what strategies were discovered, run:")
    print("  python scripts/analyze_strategies.py")

if __name__ == "__main__":
    main()
