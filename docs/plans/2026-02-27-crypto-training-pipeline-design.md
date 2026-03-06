# Crypto Training Pipeline & Live Data Provider

**Date:** 2026-02-27
**Status:** Approved

## Problem

Database has 0 crypto candles. The training pipeline reads from DB, blocking the entire chain:
No data → No training → LSTM stuck at 0.5 → Ensemble can't reach 0.565 → Never enters TRADING mode.

Additionally, `src/data/provider.py` (UnifiedDataProvider for crypto WebSocket) was never built,
and existing `.fisher` model files are EWC matrices that can't load as LSTM checkpoints.

## Design

### Component 1: `_backfill_crypto_data()` in runner.py

- Fetch 365 days of historical candles from Binance via CCXT
- Both 15m and 1h intervals for BTC/USDT and ETH/USDT
- Uses `fetch_ohlcv()` with pagination (1000 candles/request)
- Saves directly to database via `db.save_candles()`
- Runs BEFORE `_ensure_models_ready()` in startup sequence

### Component 2: `src/data/provider.py` — UnifiedDataProvider

- Singleton using CCXT WebSocket for real-time crypto candle streaming
- Interface matches TwelveDataProvider: subscribe/on_candle_closed/set_database/start/stop
- Falls back to REST polling if WebSocket unavailable

### Component 3: Model Training Fix

- Delete stale `.fisher` files
- Fresh `.pt` models trained from backfilled data
- Both LSTM and gradient boosting train on same data

## Files

| File | Action |
|------|--------|
| `src/data/provider.py` | Create (UnifiedDataProvider) |
| `src/live_trading/runner.py` | Modify (add `_backfill_crypto_data()`) |
| `models/*.fisher` | Delete (stale EWC matrices) |

## Expected Outcomes

- DB candles: 0 → ~87,600
- LSTM accuracy: N/A → 55-62%
- Ensemble prob: ~0.50 → 0.55-0.60
- Confidence: 25% → 50-75%
- Mode: Stuck LEARNING → Enters TRADING
