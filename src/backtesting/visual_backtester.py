"""
Visual Backtesting Interface
=============================
Interactive backtesting with real-time charts and comprehensive analytics.
"""

import logging
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass
import pandas as pd
import numpy as np

from .engine import BacktestEngine
from .metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Complete backtesting results with all metrics and trade history."""

    # Basic info
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float

    # Performance metrics
    total_return: float  # Percentage
    total_pnl: float  # Absolute
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # days

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: float  # hours

    # Equity curve data
    equity_curve: pd.DataFrame  # timestamp, equity, drawdown

    # Trade history
    trades: List[Dict]  # All executed trades

    # Monthly breakdown
    monthly_returns: pd.DataFrame  # month, return, pnl, trades

    # Risk metrics
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional VaR
    calmar_ratio: float
    sortino_ratio: float

    # Additional metrics
    expectancy: float
    consecutive_wins: int
    consecutive_losses: int
    recovery_factor: float


class VisualBacktester:
    """
    Advanced backtesting engine with visualization support.

    Features:
    - Run backtests with any strategy
    - Generate equity curves
    - Calculate comprehensive metrics
    - Support for multiple currencies
    - Parameter optimization
    """

    def __init__(self, initial_capital: float = 10000):
        """
        Initialize visual backtester.

        Args:
            initial_capital: Starting capital for backtest
        """
        self.initial_capital = initial_capital
        self.engine = BacktestEngine(initial_capital=initial_capital)

    def run_backtest(
        self,
        df: pd.DataFrame,
        signals: pd.DataFrame,
        symbol: str = "BTC/USDT",
        risk_per_trade: float = 0.02,
        commission: float = 0.001,
        slippage: float = 0.0005
    ) -> BacktestResult:
        """
        Run comprehensive backtest with full analytics.

        Args:
            df: Price data (OHLCV)
            signals: Trading signals with entry/exit/stop/target
            symbol: Trading symbol
            risk_per_trade: Risk percentage per trade
            commission: Trading commission (0.001 = 0.1%)
            slippage: Slippage percentage

        Returns:
            BacktestResult with all metrics and trade history
        """
        logger.info(f"Running backtest for {symbol}: {len(df)} candles, {len(signals)} signals")

        # Run backtest engine
        results = self.engine.run(
            price_data=df,
            signals=signals,
            risk_per_trade=risk_per_trade,
            commission=commission,
            slippage=slippage
        )

        # Calculate comprehensive metrics
        metrics = PerformanceMetrics(results['trades'])

        # Build equity curve
        equity_curve = self._build_equity_curve(results['trades'], df)

        # Calculate monthly returns
        monthly_returns = self._calculate_monthly_returns(results['trades'])

        # Calculate advanced risk metrics
        risk_metrics = self._calculate_risk_metrics(equity_curve, results['trades'])

        # Analyze trade sequences
        sequences = self._analyze_sequences(results['trades'])

        # Build comprehensive result
        final_capital = results['final_equity']
        total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100

        backtest_result = BacktestResult(
            start_date=df['datetime'].iloc[0] if 'datetime' in df else datetime.fromtimestamp(df['timestamp'].iloc[0]/1000),
            end_date=df['datetime'].iloc[-1] if 'datetime' in df else datetime.fromtimestamp(df['timestamp'].iloc[-1]/1000),
            initial_capital=self.initial_capital,
            final_capital=final_capital,

            total_return=total_return,
            total_pnl=final_capital - self.initial_capital,
            win_rate=metrics.win_rate,
            profit_factor=metrics.profit_factor,
            sharpe_ratio=metrics.sharpe_ratio,
            max_drawdown=metrics.max_drawdown,
            max_drawdown_duration=risk_metrics['max_dd_duration'],

            total_trades=len(results['trades']),
            winning_trades=metrics.winning_trades,
            losing_trades=metrics.losing_trades,
            avg_win=metrics.avg_winning_trade,
            avg_loss=metrics.avg_losing_trade,
            largest_win=metrics.largest_win,
            largest_loss=metrics.largest_loss,
            avg_trade_duration=self._avg_trade_duration(results['trades']),

            equity_curve=equity_curve,
            trades=results['trades'],
            monthly_returns=monthly_returns,

            var_95=risk_metrics['var_95'],
            cvar_95=risk_metrics['cvar_95'],
            calmar_ratio=risk_metrics['calmar_ratio'],
            sortino_ratio=risk_metrics['sortino_ratio'],

            expectancy=metrics.expectancy,
            consecutive_wins=sequences['max_consecutive_wins'],
            consecutive_losses=sequences['max_consecutive_losses'],
            recovery_factor=risk_metrics['recovery_factor']
        )

        logger.info(f"Backtest complete: {total_return:.2f}% return, {metrics.win_rate:.1f}% win rate")

        return backtest_result

    def _build_equity_curve(self, trades: List[Dict], price_data: pd.DataFrame) -> pd.DataFrame:
        """Build equity curve from trade history."""
        if not trades:
            return pd.DataFrame(columns=['timestamp', 'equity', 'drawdown', 'pnl'])

        # Create timeline from price data
        equity_data = []
        current_equity = self.initial_capital
        peak_equity = self.initial_capital

        # Sort trades by exit time
        sorted_trades = sorted(trades, key=lambda x: x['exit_time'])
        trade_idx = 0

        for idx, row in price_data.iterrows():
            timestamp = row['datetime'] if 'datetime' in row else datetime.fromtimestamp(row['timestamp']/1000)

            # Apply trades that exited at this timestamp
            while trade_idx < len(sorted_trades):
                trade = sorted_trades[trade_idx]
                trade_exit = pd.to_datetime(trade['exit_time'])

                if trade_exit <= timestamp:
                    # Apply trade P&L
                    current_equity += trade['pnl_absolute']
                    peak_equity = max(peak_equity, current_equity)
                    trade_idx += 1
                else:
                    break

            # Calculate drawdown
            drawdown = ((current_equity - peak_equity) / peak_equity) * 100 if peak_equity > 0 else 0

            equity_data.append({
                'timestamp': timestamp,
                'equity': current_equity,
                'drawdown': drawdown,
                'pnl': current_equity - self.initial_capital
            })

        return pd.DataFrame(equity_data)

    def _calculate_monthly_returns(self, trades: List[Dict]) -> pd.DataFrame:
        """Calculate monthly performance breakdown."""
        if not trades:
            return pd.DataFrame(columns=['month', 'return', 'pnl', 'trades', 'win_rate'])

        # Group trades by month
        df_trades = pd.DataFrame(trades)
        df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])
        df_trades['month'] = df_trades['exit_time'].dt.to_period('M')

        monthly_stats = []
        for month, group in df_trades.groupby('month'):
            total_pnl = group['pnl_absolute'].sum()
            return_pct = (total_pnl / self.initial_capital) * 100
            wins = len(group[group['pnl_absolute'] > 0])
            total = len(group)
            win_rate = (wins / total * 100) if total > 0 else 0

            monthly_stats.append({
                'month': str(month),
                'return': return_pct,
                'pnl': total_pnl,
                'trades': total,
                'win_rate': win_rate
            })

        return pd.DataFrame(monthly_stats)

    def _calculate_risk_metrics(self, equity_curve: pd.DataFrame, trades: List[Dict]) -> Dict:
        """Calculate advanced risk metrics."""
        if equity_curve.empty:
            return {
                'var_95': 0,
                'cvar_95': 0,
                'calmar_ratio': 0,
                'sortino_ratio': 0,
                'max_dd_duration': 0,
                'recovery_factor': 0
            }

        # Daily returns
        equity_curve['daily_return'] = equity_curve['equity'].pct_change()
        returns = equity_curve['daily_return'].dropna()

        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0

        # Conditional VaR (average of worst 5%)
        worst_returns = returns[returns <= var_95]
        cvar_95 = worst_returns.mean() if len(worst_returns) > 0 else 0

        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0001
        sortino_ratio = (returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0

        # Calmar Ratio (return / max drawdown)
        max_dd = abs(equity_curve['drawdown'].min())
        total_return = ((equity_curve['equity'].iloc[-1] - equity_curve['equity'].iloc[0]) /
                       equity_curve['equity'].iloc[0])
        calmar_ratio = (total_return / (max_dd / 100)) if max_dd > 0 else 0

        # Max drawdown duration
        in_drawdown = equity_curve['drawdown'] < 0
        dd_groups = (in_drawdown != in_drawdown.shift()).cumsum()
        max_dd_duration = in_drawdown.groupby(dd_groups).sum().max() if in_drawdown.any() else 0

        # Recovery factor
        total_pnl = equity_curve['pnl'].iloc[-1] if len(equity_curve) > 0 else 0
        recovery_factor = abs(total_pnl / (max_dd / 100)) if max_dd > 0 else 0

        return {
            'var_95': var_95,
            'cvar_95': cvar_95,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'max_dd_duration': int(max_dd_duration),
            'recovery_factor': recovery_factor
        }

    def _analyze_sequences(self, trades: List[Dict]) -> Dict:
        """Analyze win/loss sequences."""
        if not trades:
            return {'max_consecutive_wins': 0, 'max_consecutive_losses': 0}

        # Sort by exit time
        sorted_trades = sorted(trades, key=lambda x: x['exit_time'])

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in sorted_trades:
            if trade['pnl_absolute'] > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return {
            'max_consecutive_wins': max_wins,
            'max_consecutive_losses': max_losses
        }

    def _avg_trade_duration(self, trades: List[Dict]) -> float:
        """Calculate average trade duration in hours."""
        if not trades:
            return 0

        durations = []
        for trade in trades:
            entry = pd.to_datetime(trade['entry_time'])
            exit = pd.to_datetime(trade['exit_time'])
            duration = (exit - entry).total_seconds() / 3600  # hours
            durations.append(duration)

        return np.mean(durations) if durations else 0

    def compare_strategies(
        self,
        df: pd.DataFrame,
        strategies: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Compare multiple strategies side-by-side.

        Args:
            df: Price data
            strategies: Dict of {strategy_name: signals_df}

        Returns:
            DataFrame comparing all strategies
        """
        results = []

        for name, signals in strategies.items():
            logger.info(f"Backtesting strategy: {name}")
            result = self.run_backtest(df, signals)

            results.append({
                'Strategy': name,
                'Total Return %': result.total_return,
                'Win Rate %': result.win_rate,
                'Profit Factor': result.profit_factor,
                'Sharpe Ratio': result.sharpe_ratio,
                'Max Drawdown %': result.max_drawdown,
                'Total Trades': result.total_trades,
                'Expectancy': result.expectancy
            })

        return pd.DataFrame(results).sort_values('Total Return %', ascending=False)
