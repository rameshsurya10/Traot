"""
Backtest Metrics
================
Performance metrics calculation for backtesting results.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np


@dataclass
class BacktestMetrics:
    """
    Trading strategy performance metrics.

    All the key metrics needed to evaluate a trading strategy.
    """
    # Basic counts
    total_signals: int = 0
    total_trades: int = 0
    winners: int = 0
    losers: int = 0

    # Win rate
    win_rate: float = 0.0

    # PnL metrics
    total_pnl_percent: float = 0.0
    avg_pnl_percent: float = 0.0
    avg_winner_pnl: float = 0.0
    avg_loser_pnl: float = 0.0
    largest_winner: float = 0.0
    largest_loser: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0

    # Time metrics
    avg_trade_duration_hours: float = 0.0
    avg_winner_duration_hours: float = 0.0
    avg_loser_duration_hours: float = 0.0

    # Signal quality
    accuracy_by_confidence: Dict[str, float] = field(default_factory=dict)
    strong_signal_win_rate: float = 0.0
    medium_signal_win_rate: float = 0.0

    # Streak analysis
    max_consecutive_winners: int = 0
    max_consecutive_losers: int = 0
    current_streak: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_signals': self.total_signals,
            'total_trades': self.total_trades,
            'winners': self.winners,
            'losers': self.losers,
            'win_rate': self.win_rate,
            'total_pnl_percent': self.total_pnl_percent,
            'avg_pnl_percent': self.avg_pnl_percent,
            'avg_winner_pnl': self.avg_winner_pnl,
            'avg_loser_pnl': self.avg_loser_pnl,
            'largest_winner': self.largest_winner,
            'largest_loser': self.largest_loser,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'profit_factor': self.profit_factor,
            'expectancy': self.expectancy,
            'avg_trade_duration_hours': self.avg_trade_duration_hours,
            'strong_signal_win_rate': self.strong_signal_win_rate,
            'medium_signal_win_rate': self.medium_signal_win_rate,
            'max_consecutive_winners': self.max_consecutive_winners,
            'max_consecutive_losers': self.max_consecutive_losers,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "BACKTEST RESULTS",
            "=" * 60,
            "",
            f"Total Signals Generated: {self.total_signals}",
            f"Total Trades Executed:   {self.total_trades}",
            f"Winners: {self.winners} | Losers: {self.losers}",
            "",
            "-" * 60,
            "PERFORMANCE",
            "-" * 60,
            f"Win Rate:         {self.win_rate:.1%}",
            f"Total PnL:        {self.total_pnl_percent:+.2f}%",
            f"Avg Trade PnL:    {self.avg_pnl_percent:+.2f}%",
            f"Avg Winner:       {self.avg_winner_pnl:+.2f}%",
            f"Avg Loser:        {self.avg_loser_pnl:+.2f}%",
            f"Largest Winner:   {self.largest_winner:+.2f}%",
            f"Largest Loser:    {self.largest_loser:+.2f}%",
            "",
            "-" * 60,
            "RISK METRICS",
            "-" * 60,
            f"Max Drawdown:     {self.max_drawdown:.2f}%",
            f"Sharpe Ratio:     {self.sharpe_ratio:.2f}",
            f"Profit Factor:    {self.profit_factor:.2f}",
            f"Expectancy:       {self.expectancy:+.2f}%",
            "",
            "-" * 60,
            "SIGNAL QUALITY",
            "-" * 60,
            f"Strong Signal Win Rate: {self.strong_signal_win_rate:.1%}",
            f"Medium Signal Win Rate: {self.medium_signal_win_rate:.1%}",
            "",
            "-" * 60,
            "STREAKS",
            "-" * 60,
            f"Max Consecutive Wins:   {self.max_consecutive_winners}",
            f"Max Consecutive Losses: {self.max_consecutive_losers}",
            "",
            "=" * 60,
        ]

        # Assessment
        if self.total_trades >= 30:
            if self.win_rate >= 0.55 and self.profit_factor >= 1.5:
                lines.append("ASSESSMENT: STRONG - Strategy shows promise")
            elif self.win_rate >= 0.50 and self.profit_factor >= 1.2:
                lines.append("ASSESSMENT: VIABLE - Strategy may be profitable")
            elif self.win_rate >= 0.45:
                lines.append("ASSESSMENT: MARGINAL - Needs improvement")
            else:
                lines.append("ASSESSMENT: POOR - Do not use this strategy")
        else:
            lines.append(f"ASSESSMENT: INSUFFICIENT DATA - Need 30+ trades, have {self.total_trades}")

        lines.append("=" * 60)

        return "\n".join(lines)


def calculate_metrics(trades: List[Dict[str, Any]]) -> BacktestMetrics:
    """
    Calculate backtest metrics from trade results.

    Args:
        trades: List of trade dictionaries with keys:
            - pnl_percent: float
            - is_winner: bool
            - duration_minutes: float
            - strength: str ('STRONG' or 'MEDIUM')

    Returns:
        BacktestMetrics object
    """
    metrics = BacktestMetrics()

    if not trades:
        return metrics

    metrics.total_trades = len(trades)

    # Separate winners and losers
    pnls = [t['pnl_percent'] for t in trades]
    winner_trades = [t for t in trades if t.get('is_winner', t['pnl_percent'] > 0)]
    loser_trades = [t for t in trades if not t.get('is_winner', t['pnl_percent'] > 0)]

    metrics.winners = len(winner_trades)
    metrics.losers = len(loser_trades)

    # Win rate
    metrics.win_rate = metrics.winners / metrics.total_trades if metrics.total_trades > 0 else 0

    # PnL metrics
    metrics.total_pnl_percent = sum(pnls)
    metrics.avg_pnl_percent = np.mean(pnls) if pnls else 0

    winner_pnls = [t['pnl_percent'] for t in winner_trades]
    loser_pnls = [t['pnl_percent'] for t in loser_trades]

    metrics.avg_winner_pnl = np.mean(winner_pnls) if winner_pnls else 0
    metrics.avg_loser_pnl = np.mean(loser_pnls) if loser_pnls else 0
    metrics.largest_winner = max(winner_pnls) if winner_pnls else 0
    metrics.largest_loser = min(loser_pnls) if loser_pnls else 0

    # Max drawdown
    equity_curve = []
    running_pnl = 0
    peak = 0
    max_dd = 0

    for pnl in pnls:
        running_pnl += pnl
        equity_curve.append(running_pnl)
        peak = max(peak, running_pnl)
        drawdown = peak - running_pnl
        max_dd = max(max_dd, drawdown)

    metrics.max_drawdown = max_dd

    # Sharpe ratio (simplified - assumes daily returns)
    if len(pnls) > 1:
        std = np.std(pnls)
        if std > 0:
            metrics.sharpe_ratio = (np.mean(pnls) / std) * np.sqrt(252)

    # Profit factor
    gross_profit = sum(winner_pnls) if winner_pnls else 0
    gross_loss = abs(sum(loser_pnls)) if loser_pnls else 0
    if gross_loss > 0:
        metrics.profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        metrics.profit_factor = float('inf')

    # Expectancy (expected value per trade)
    if metrics.total_trades > 0:
        metrics.expectancy = (
            (metrics.win_rate * metrics.avg_winner_pnl) +
            ((1 - metrics.win_rate) * metrics.avg_loser_pnl)
        )

    # Duration metrics
    durations = [t.get('duration_minutes', 0) for t in trades]
    metrics.avg_trade_duration_hours = np.mean(durations) / 60 if durations else 0

    winner_durations = [t.get('duration_minutes', 0) for t in winner_trades]
    loser_durations = [t.get('duration_minutes', 0) for t in loser_trades]
    metrics.avg_winner_duration_hours = np.mean(winner_durations) / 60 if winner_durations else 0
    metrics.avg_loser_duration_hours = np.mean(loser_durations) / 60 if loser_durations else 0

    # Signal quality by strength
    strong_trades = [t for t in trades if t.get('strength') == 'STRONG']
    medium_trades = [t for t in trades if t.get('strength') == 'MEDIUM']

    if strong_trades:
        strong_winners = sum(1 for t in strong_trades if t.get('is_winner', t['pnl_percent'] > 0))
        metrics.strong_signal_win_rate = strong_winners / len(strong_trades)

    if medium_trades:
        medium_winners = sum(1 for t in medium_trades if t.get('is_winner', t['pnl_percent'] > 0))
        metrics.medium_signal_win_rate = medium_winners / len(medium_trades)

    # Streak analysis
    max_win_streak = 0
    max_lose_streak = 0
    win_streak = 0
    lose_streak = 0

    for trade in trades:
        is_winner = trade.get('is_winner', trade['pnl_percent'] > 0)
        if is_winner:
            win_streak += 1
            lose_streak = 0
            max_win_streak = max(max_win_streak, win_streak)
        else:
            lose_streak += 1
            win_streak = 0
            max_lose_streak = max(max_lose_streak, lose_streak)

    metrics.max_consecutive_winners = max_win_streak
    metrics.max_consecutive_losers = max_lose_streak

    # Current streak (from last trade)
    if trades:
        last_winner = trades[-1].get('is_winner', trades[-1]['pnl_percent'] > 0)
        streak_count = 1
        for i in range(len(trades) - 2, -1, -1):
            this_winner = trades[i].get('is_winner', trades[i]['pnl_percent'] > 0)
            if this_winner == last_winner:
                streak_count += 1
            else:
                break
        metrics.current_streak = streak_count if last_winner else -streak_count

    return metrics
