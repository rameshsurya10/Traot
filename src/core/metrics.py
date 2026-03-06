"""
Performance Metrics Module
==========================
Comprehensive trading performance metrics calculation.

Metrics:
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Maximum Drawdown, Win Rate, Profit Factor
- Risk-adjusted returns, Alpha, Beta
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Constants
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.05  # 5% annual risk-free rate


@dataclass
class PerformanceMetrics:
    """Complete performance metrics for a trading strategy."""

    # Return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    daily_return_mean: float = 0.0
    daily_return_std: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    current_drawdown: float = 0.0
    volatility: float = 0.0
    downside_volatility: float = 0.0

    # Risk-adjusted metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Trading metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration: float = 0.0

    # Exposure metrics
    time_in_market: float = 0.0
    avg_position_size: float = 0.0

    # Timestamps
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    calculated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'total_return': f"{self.total_return:.2%}",
            'annualized_return': f"{self.annualized_return:.2%}",
            'max_drawdown': f"{self.max_drawdown:.2%}",
            'sharpe_ratio': f"{self.sharpe_ratio:.2f}",
            'sortino_ratio': f"{self.sortino_ratio:.2f}",
            'calmar_ratio': f"{self.calmar_ratio:.2f}",
            'win_rate': f"{self.win_rate:.1%}",
            'profit_factor': f"{self.profit_factor:.2f}",
            'total_trades': self.total_trades,
            'volatility': f"{self.volatility:.2%}",
        }


class MetricsCalculator:
    """
    Calculate comprehensive performance metrics.

    Usage:
        calculator = MetricsCalculator()
        metrics = calculator.calculate(equity_curve, trades)
    """

    def __init__(self, risk_free_rate: float = RISK_FREE_RATE):
        self.risk_free_rate = risk_free_rate
        self.daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR

    def calculate(
        self,
        equity_curve: pd.Series,
        trades: Optional[List[Dict]] = None
    ) -> PerformanceMetrics:
        """
        Calculate all performance metrics.

        Args:
            equity_curve: Series of portfolio values indexed by datetime
            trades: Optional list of trade dictionaries

        Returns:
            PerformanceMetrics dataclass
        """
        metrics = PerformanceMetrics()

        if equity_curve is None or len(equity_curve) < 2:
            return metrics

        # Ensure datetime index
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            equity_curve.index = pd.to_datetime(equity_curve.index)

        # Calculate returns
        returns = equity_curve.pct_change().dropna()

        if len(returns) < 1:
            return metrics

        # Basic return metrics
        metrics.total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        metrics.daily_return_mean = returns.mean()
        metrics.daily_return_std = returns.std()

        # Annualized return
        n_days = (equity_curve.index[-1] - equity_curve.index[0]).days
        if n_days > 0:
            years = n_days / 365
            metrics.annualized_return = (1 + metrics.total_return) ** (1 / years) - 1

        # Volatility
        metrics.volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        # Downside volatility (for Sortino)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            metrics.downside_volatility = negative_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        # Drawdown analysis
        dd_metrics = self._calculate_drawdown(equity_curve)
        metrics.max_drawdown = dd_metrics['max_drawdown']
        metrics.max_drawdown_duration_days = dd_metrics['max_duration_days']
        metrics.current_drawdown = dd_metrics['current_drawdown']

        # Risk-adjusted ratios
        metrics.sharpe_ratio = self._calculate_sharpe(returns)
        metrics.sortino_ratio = self._calculate_sortino(returns)
        metrics.calmar_ratio = self._calculate_calmar(
            metrics.annualized_return,
            metrics.max_drawdown
        )

        # Trade metrics
        if trades:
            trade_metrics = self._calculate_trade_metrics(trades)
            metrics.total_trades = trade_metrics['total_trades']
            metrics.winning_trades = trade_metrics['winning_trades']
            metrics.losing_trades = trade_metrics['losing_trades']
            metrics.win_rate = trade_metrics['win_rate']
            metrics.profit_factor = trade_metrics['profit_factor']
            metrics.avg_win = trade_metrics['avg_win']
            metrics.avg_loss = trade_metrics['avg_loss']
            metrics.largest_win = trade_metrics['largest_win']
            metrics.largest_loss = trade_metrics['largest_loss']
            metrics.avg_trade_duration = trade_metrics['avg_duration']

        # Timestamps
        metrics.start_date = equity_curve.index[0]
        metrics.end_date = equity_curve.index[-1]
        metrics.calculated_at = datetime.now()

        return metrics

    def _calculate_drawdown(self, equity: pd.Series) -> Dict:
        """Calculate drawdown metrics."""
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak

        max_drawdown = drawdown.min()
        current_drawdown = drawdown.iloc[-1]

        # Calculate max drawdown duration
        in_drawdown = drawdown < 0

        max_duration = 0
        current_duration = 0

        for i in range(len(in_drawdown)):
            if in_drawdown.iloc[i]:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return {
            'max_drawdown': abs(max_drawdown),
            'current_drawdown': abs(current_drawdown),
            'max_duration_days': max_duration
        }

    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if returns.std() < 1e-10:
            return 0.0

        excess_returns = returns - self.daily_rf
        sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)

        return float(sharpe)

    def _calculate_sortino(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (uses downside deviation)."""
        negative_returns = returns[returns < 0]

        if len(negative_returns) < 1:
            return float('inf') if returns.mean() > 0 else 0.0

        downside_std = negative_returns.std()
        if downside_std < 1e-10:
            return 0.0

        excess_return = returns.mean() - self.daily_rf
        sortino = (excess_return / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR)

        return float(sortino)

    def _calculate_calmar(
        self,
        annualized_return: float,
        max_drawdown: float
    ) -> float:
        """Calculate Calmar ratio (return / max drawdown)."""
        if max_drawdown < 1e-10:
            return 0.0

        return annualized_return / max_drawdown

    def _calculate_trade_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate trade-specific metrics."""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'avg_duration': 0.0
            }

        pnls = [t.get('pnl', t.get('pnl_percent', 0)) for t in trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]

        total_trades = len(trades)
        winning_trades = len(winners)
        losing_trades = len(losers)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        gross_profit = sum(winners)
        gross_loss = abs(sum(losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_win = np.mean(winners) if winners else 0
        avg_loss = np.mean(losers) if losers else 0
        largest_win = max(winners) if winners else 0
        largest_loss = min(losers) if losers else 0

        # Calculate average duration
        durations = []
        for t in trades:
            if 'duration_minutes' in t:
                durations.append(t['duration_minutes'])
            elif 'entry_time' in t and 'exit_time' in t:
                try:
                    entry = pd.to_datetime(t['entry_time'])
                    exit_ = pd.to_datetime(t['exit_time'])
                    durations.append((exit_ - entry).total_seconds() / 60)
                except (ValueError, TypeError, KeyError):
                    pass

        avg_duration = np.mean(durations) if durations else 0

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'avg_duration': avg_duration
        }


class SignalQualityScorer:
    """
    Score signal quality based on multiple factors.

    Factors:
    - Multi-timeframe alignment
    - Volume confirmation
    - Regime alignment
    - Historical accuracy
    - Ensemble agreement
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            'multi_timeframe': 0.25,
            'volume_confirmation': 0.20,
            'regime_alignment': 0.20,
            'historical_accuracy': 0.20,
            'ensemble_agreement': 0.15
        }

    def score(
        self,
        signal_direction: str,
        confidence: float,
        current_price: float,
        volume: float,
        avg_volume: float,
        regime: str,
        higher_tf_trend: Optional[str] = None,
        base_model_predictions: Optional[Dict[str, float]] = None,
        historical_win_rate: float = 0.5
    ) -> Dict:
        """
        Calculate signal quality score.

        Returns:
            Dict with score, grade, and factor breakdown
        """
        scores = {}

        # Multi-timeframe alignment
        if higher_tf_trend:
            if (signal_direction == 'BUY' and higher_tf_trend == 'UP') or \
               (signal_direction == 'SELL' and higher_tf_trend == 'DOWN'):
                scores['multi_timeframe'] = 1.0
            elif higher_tf_trend == 'SIDEWAYS':
                scores['multi_timeframe'] = 0.5
            else:
                scores['multi_timeframe'] = 0.2
        else:
            scores['multi_timeframe'] = 0.5  # Neutral if unknown

        # Volume confirmation
        if avg_volume > 0:
            volume_ratio = volume / avg_volume
            if volume_ratio >= 1.5:
                scores['volume_confirmation'] = 1.0
            elif volume_ratio >= 1.0:
                scores['volume_confirmation'] = 0.7
            elif volume_ratio >= 0.5:
                scores['volume_confirmation'] = 0.4
            else:
                scores['volume_confirmation'] = 0.2
        else:
            scores['volume_confirmation'] = 0.5

        # Regime alignment
        if regime == 'BULL' and signal_direction == 'BUY':
            scores['regime_alignment'] = 1.0
        elif regime == 'BEAR' and signal_direction == 'SELL':
            scores['regime_alignment'] = 1.0
        elif regime == 'SIDEWAYS':
            scores['regime_alignment'] = 0.5
        else:
            scores['regime_alignment'] = 0.3

        # Historical accuracy
        scores['historical_accuracy'] = min(1.0, historical_win_rate * 1.5)

        # Ensemble agreement
        if base_model_predictions:
            if signal_direction == 'BUY':
                agreement = sum(1 for p in base_model_predictions.values() if p > 0.5)
            else:
                agreement = sum(1 for p in base_model_predictions.values() if p < 0.5)
            scores['ensemble_agreement'] = agreement / len(base_model_predictions)
        else:
            scores['ensemble_agreement'] = confidence

        # Calculate weighted total
        total_score = sum(
            scores[factor] * self.weights[factor]
            for factor in scores
        )

        # Determine grade
        if total_score >= 0.8:
            grade = 'A'
        elif total_score >= 0.6:
            grade = 'B'
        elif total_score >= 0.4:
            grade = 'C'
        else:
            grade = 'D'

        return {
            'score': total_score,
            'grade': grade,
            'factors': scores,
            'recommendation': self._get_recommendation(grade, signal_direction)
        }

    def _get_recommendation(self, grade: str, direction: str) -> str:
        """Get trading recommendation based on grade."""
        if grade == 'A':
            return f"Strong {direction} signal - Full position size"
        elif grade == 'B':
            return f"Good {direction} signal - Standard position size"
        elif grade == 'C':
            return f"Weak {direction} signal - Reduced position size"
        else:
            return f"Poor {direction} signal - Consider skipping"


# Export
__all__ = [
    'PerformanceMetrics',
    'MetricsCalculator',
    'SignalQualityScorer',
    'TRADING_DAYS_PER_YEAR',
    'RISK_FREE_RATE'
]
