"""
Loss Functions for Hyperparameter Optimization
================================================
Objective functions that convert backtest metrics into a single scalar
that Optuna minimises.  Lower (more negative) = better strategy.

Available losses:
  - traot   — Custom multi-factor score (default)
  - sharpe  — Sharpe ratio
  - sortino — Sortino ratio (downside-deviation adjusted)
  - calmar  — Calmar ratio (return / max drawdown)
  - profit  — Raw total PnL
"""

from __future__ import annotations

import math
from typing import List, Optional

from src.backtesting.metrics import BacktestMetrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_LOSS: float = 100_000


# ---------------------------------------------------------------------------
# Hard-kill guardrails
# ---------------------------------------------------------------------------

def _hard_kill(metrics: BacktestMetrics, max_dd_ceiling: float) -> bool:
    """Return ``True`` if the trial should be killed immediately.

    Conditions (any triggers a kill):
        - total_trades < 10
        - win_rate < 0.50
        - max_drawdown > *max_dd_ceiling*
        - largest_loser < -5.0  (i.e. a single trade lost more than 5%)
        - profit_factor < 1.0
    """
    if metrics.total_trades < 10:
        return True
    if metrics.win_rate < 0.50:
        return True
    if metrics.max_drawdown > max_dd_ceiling:
        return True
    if metrics.largest_loser < -5.0:
        return True
    if metrics.profit_factor < 1.0:
        return True
    return False


# ---------------------------------------------------------------------------
# Individual loss functions
# ---------------------------------------------------------------------------

def _traot_loss(metrics: BacktestMetrics) -> float:
    """Custom multi-factor score emphasising consistency."""
    if metrics.avg_loser_pnl != 0:
        trade_quality = min(10.0, metrics.avg_winner_pnl / abs(metrics.avg_loser_pnl))
    else:
        trade_quality = 10.0

    score = (
        (metrics.win_rate ** 2)
        * ((1 - metrics.max_drawdown / 100) ** 3)
        * metrics.profit_factor
        * trade_quality
    )
    return -1 * score


def _sharpe_loss(metrics: BacktestMetrics) -> float:
    """Negative Sharpe ratio (already pre-computed on BacktestMetrics)."""
    return -1 * metrics.sharpe_ratio


def _sortino_loss(metrics: BacktestMetrics, backtest_days: float) -> float:
    """Negative Sortino ratio.

    Sortino = (avg_pnl / downside_std) * sqrt(N)
    where N = annualised trade count.

    Falls back to ``-sharpe_ratio`` when there are not enough losing trades
    to compute a meaningful downside deviation.
    """
    if metrics.total_trades < 2 or metrics.losers < 2:
        return -1 * metrics.sharpe_ratio

    # Reconstruct per-trade downside deviations from available metrics.
    # We don't have the raw trade list, so we approximate:
    #   downside_std ≈ |avg_loser_pnl| (reasonable when losses are roughly
    #   normally distributed around the loser mean).
    downside_std = abs(metrics.avg_loser_pnl) if metrics.avg_loser_pnl != 0 else 0.0
    if downside_std == 0:
        return -1 * metrics.sharpe_ratio

    trades_per_year = (metrics.total_trades / max(backtest_days, 1)) * 365
    sortino = (metrics.avg_pnl_percent / downside_std) * math.sqrt(max(trades_per_year, 1))
    return -1 * sortino


def _calmar_loss(metrics: BacktestMetrics) -> float:
    """Negative Calmar ratio (total_pnl / max_drawdown)."""
    if metrics.max_drawdown > 0:
        return -1 * (metrics.total_pnl_percent / metrics.max_drawdown)
    # No drawdown at all
    if metrics.total_pnl_percent <= 0:
        return MAX_LOSS
    # Positive PnL with zero drawdown — best possible outcome
    return -1 * metrics.total_pnl_percent


def _profit_loss(metrics: BacktestMetrics) -> float:
    """Simply negate total PnL percent."""
    return -1 * metrics.total_pnl_percent


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_LOSS_FUNCTIONS = {
    "traot": lambda m, d, _dd: _traot_loss(m),
    "sharpe": lambda m, d, _dd: _sharpe_loss(m),
    "sortino": lambda m, d, _dd: _sortino_loss(m, d),
    "calmar": lambda m, d, _dd: _calmar_loss(m),
    "profit": lambda m, d, _dd: _profit_loss(m),
}


def get_available_loss_functions() -> List[str]:
    """Return the names of all registered loss functions."""
    return list(_LOSS_FUNCTIONS.keys())


def compute_loss(
    metrics: BacktestMetrics,
    loss_name: str,
    backtest_days: float,
    max_dd_param: Optional[int] = None,
) -> float:
    """Compute the loss value for a completed backtest.

    Args:
        metrics: Backtest result metrics.
        loss_name: One of :func:`get_available_loss_functions`.
        backtest_days: Duration of the backtest period in days.
        max_dd_param: Optional max-drawdown parameter from the trial.
            Used to derive the hard-kill ceiling.

    Returns:
        A float where **lower is better**.  ``MAX_LOSS`` indicates the trial
        should be pruned.
    """
    if loss_name not in _LOSS_FUNCTIONS:
        raise ValueError(
            f"Unknown loss function '{loss_name}'. "
            f"Available: {get_available_loss_functions()}"
        )

    # Determine drawdown ceiling for hard-kill
    if max_dd_param is None:
        max_dd_ceiling = 15.0
    else:
        max_dd_ceiling = min(max_dd_param, 20)

    if _hard_kill(metrics, max_dd_ceiling):
        return MAX_LOSS

    return _LOSS_FUNCTIONS[loss_name](metrics, backtest_days, max_dd_param)
