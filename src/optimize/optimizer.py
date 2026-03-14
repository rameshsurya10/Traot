"""
Optimizer — Optuna Objective Factory
=====================================
Creates the Optuna objective function that ties together:
  - Parameter suggestion (parameter_space.py)
  - Backtest execution (backtesting/engine.py)
  - Loss computation (loss_functions.py)

Usage:
    objective = create_objective(config_path='config.yaml', loss_name='traot')
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
"""

import copy
import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.core.config import load_config
from src.optimize.loss_functions import MAX_LOSS, compute_loss
from src.optimize.parameter_space import (
    apply_params_to_config,
    normalize_ensemble_weights,
    suggest_params,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_objective(
    config_path: str = "config.yaml",
    loss_name: str = "traot",
    spaces: Optional[list] = None,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    data_df: Optional[pd.DataFrame] = None,
    max_dd_param_active: bool = False,
) -> Callable:
    """Create an Optuna objective function.

    Loads config and data **once**, returns a closure that runs one backtest
    trial per call.

    Args:
        config_path: Path to the YAML configuration file.
        loss_name: Loss function name (see ``get_available_loss_functions``).
        spaces: Parameter spaces to search (default: ``['trading']``).
        symbol: Trading pair override (e.g. ``'BTC/USDT'``).
        timeframe: Candle timeframe override (e.g. ``'15m'``).
        data_df: Pre-loaded OHLCV DataFrame.  When ``None`` the function
            attempts to load from the trading database, falling back to
            synthetic data for testing.
        max_dd_param_active: Whether the ``max_drawdown_percent`` trial
            parameter should feed into the loss function's hard-kill ceiling.

    Returns:
        A callable ``objective(trial) -> float`` suitable for
        ``study.optimize(objective, ...)``.
    """
    if spaces is None:
        spaces = ["trading"]

    raw_config = load_config(config_path)

    # Load data once (expensive) -----------------------------------------
    if data_df is None:
        data_df = _load_historical_data(raw_config, symbol, timeframe)

    # Estimate backtest period in days ------------------------------------
    backtest_days = _estimate_backtest_days(data_df)

    # Closure -------------------------------------------------------------
    def objective(trial) -> float:
        try:
            # 1. Suggest parameters for this trial
            params = suggest_params(trial, spaces=spaces)

            # 2. Deep-copy config and apply suggested values
            trial_config = copy.deepcopy(raw_config)
            apply_params_to_config(trial_config, params)

            # 3. Normalize ensemble weights when the ensemble space is active
            if "ensemble" in spaces:
                _apply_normalized_ensemble(trial_config, params)

            # 4. Run backtest
            from src.backtesting.engine import BacktestEngine

            engine = BacktestEngine(
                config_path=config_path, config_dict=trial_config
            )
            engine.load_data(df=data_df, copy=False)
            metrics = engine.run_with_model()

            # 5. Compute loss
            max_dd = (
                params.get("max_drawdown_percent")
                if max_dd_param_active
                else None
            )
            loss = compute_loss(
                metrics,
                loss_name,
                backtest_days=backtest_days,
                max_dd_param=max_dd,
            )

            logger.info(
                "Trial %d: trades=%d WR=%.0f%% P&L=%+.2f%% "
                "DD=%.1f%% loss=%.4f",
                trial.number,
                metrics.total_trades,
                metrics.win_rate * 100,
                metrics.total_pnl_percent,
                metrics.max_drawdown,
                loss,
            )

            return loss

        except Exception as exc:
            logger.warning("Trial %d failed: %s", trial.number, exc)
            return MAX_LOSS

    return objective


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _estimate_backtest_days(df: pd.DataFrame) -> float:
    """Estimate the number of calendar days covered by *df*.

    Falls back to a heuristic (rows / 96 for 15-min candles) when the
    ``datetime`` column is missing.
    """
    if df is None or len(df) == 0:
        return 30.0

    if "datetime" in df.columns:
        time_range = df["datetime"].max() - df["datetime"].min()
        return max(1.0, time_range.total_seconds() / 86_400)

    # Heuristic: assume 15-minute candles (96 per day)
    return max(1.0, len(df) / 96)


def _load_historical_data(
    raw_config: dict,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> pd.DataFrame:
    """Load historical OHLCV data.

    Resolution order:
      1. Trading database (``data/trading.db`` or config override).
      2. Synthetic random-walk data (2 000 candles) as a test fallback.
    """
    import sqlite3

    db_path = raw_config.get("database", {}).get("path", "data/trading.db")
    sym = symbol or raw_config.get("data", {}).get("symbol", "BTC/USDT")

    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            "SELECT datetime, open, high, low, close, volume "
            "FROM candles WHERE symbol = ? ORDER BY datetime ASC",
            conn,
            params=(sym,),
        )
        conn.close()

        if len(df) > 200:
            df["datetime"] = pd.to_datetime(df["datetime"])
            logger.info(
                "Loaded %d candles for %s from %s", len(df), sym, db_path
            )
            return df
    except Exception as exc:
        logger.warning("Could not load from database: %s", exc)

    # Synthetic fallback ---------------------------------------------------
    logger.warning("No historical data found — using synthetic data")
    return _generate_synthetic_data()


def _generate_synthetic_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate minimal synthetic OHLCV data for testing.

    The random walk starts at 100 with a floor of 10.
    """
    rng = np.random.default_rng(seed)
    prices = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    prices = np.maximum(prices, 10.0)

    return pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=n, freq="15min"),
            "open": prices,
            "high": prices * (1 + np.abs(rng.standard_normal(n)) * 0.005),
            "low": prices * (1 - np.abs(rng.standard_normal(n)) * 0.005),
            "close": prices + rng.standard_normal(n) * 0.3,
            "volume": rng.integers(100, 10_000, size=n).astype(float),
        }
    )


def _apply_normalized_ensemble(
    trial_config: dict, params: dict
) -> None:
    """Normalize ensemble weights and write them back into *trial_config*.

    Strips the ``weight_`` prefix before normalisation so the LSTM floor
    logic works correctly.
    """
    raw_weights = {
        k.replace("weight_", ""): v
        for k, v in params.items()
        if k.startswith("weight_")
    }
    if not raw_weights:
        return

    normalized = normalize_ensemble_weights(raw_weights)

    weights_node = (
        trial_config.setdefault("prediction", {})
        .setdefault("ensemble", {})
        .setdefault("weights", {})
    )
    for model_name, weight_val in normalized.items():
        weights_node[model_name] = weight_val


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------


def split_train_val(
    df: pd.DataFrame, train_ratio: float = 0.7
) -> tuple:
    """Split a DataFrame into train and validation sets (time-series safe).

    Args:
        df: Input DataFrame ordered chronologically.
        train_ratio: Fraction of rows to use for training.

    Returns:
        ``(train_df, val_df)`` — both with reset indexes.
    """
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].reset_index(drop=True)
    val = df.iloc[split_idx:].reset_index(drop=True)
    return train, val


# ---------------------------------------------------------------------------
# Data download (optional, requires ccxt)
# ---------------------------------------------------------------------------


def download_historical_data(
    symbol: str,
    timeframe: str,
    days: int,
    exchange: str = "binance",
) -> pd.DataFrame:
    """Download historical OHLCV data via CCXT with parquet caching.

    Args:
        symbol: Trading pair, e.g. ``'BTC/USDT'``.
        timeframe: Candle interval, e.g. ``'15m'``, ``'1h'``.
        days: Number of calendar days to fetch.
        exchange: CCXT exchange id.

    Returns:
        DataFrame with columns ``[datetime, open, high, low, close, volume]``.
    """
    import ccxt
    from datetime import datetime, timedelta
    from pathlib import Path

    # Check cache first
    cache_dir = Path("data/hyperopt_candles")
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.replace("/", "_")
    cache_path = cache_dir / f"{safe_symbol}_{timeframe}_{days}d.parquet"

    if cache_path.exists():
        logger.info("Loading cached data from %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info(
        "Downloading %d days of %s %s from %s...",
        days, symbol, timeframe, exchange,
    )

    ex = getattr(ccxt, exchange)({"enableRateLimit": True})

    since = int(
        (datetime.utcnow() - timedelta(days=days)).timestamp() * 1000
    )

    all_candles: list = []
    while True:
        candles = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not candles:
            break
        all_candles.extend(candles)
        since = candles[-1][0] + 1
        if len(candles) < 1000:
            break

    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop(columns=["timestamp"])
    df = df.sort_values("datetime").reset_index(drop=True)

    df.to_parquet(cache_path)
    logger.info("Downloaded %d candles, cached to %s", len(df), cache_path)

    return df
