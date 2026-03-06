"""
Performance Tracker
===================
Tracks actual outcomes of trading signals in real-time.

Monitors each signal and updates its outcome when:
- Take profit is hit (WIN)
- Stop loss is hit (LOSS)
- Max hold time exceeded (TIMEOUT)
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import threading
import time

import pandas as pd

from src.core.types import Signal, SignalType
from src.core.config import Config
from src.core.database import Database

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Real-time performance tracker for trading signals.

    Monitors pending signals and updates their outcomes
    when price targets or stops are hit.

    Usage:
        tracker = PerformanceTracker(config)
        tracker.start()  # Start background monitoring

        # Later...
        stats = tracker.get_stats()
        print(f"Win rate: {stats['win_rate']:.1%}")
    """

    def __init__(self, config: Config, db: Optional[Database] = None):
        """
        Initialize performance tracker.

        Args:
            config: Trading configuration
            db: Database instance (creates new if None)
        """
        self.config = config
        self.db = db or Database(config.database.path)

        # Settings
        self.max_hold_hours = 24 * 7  # Max 7 days to hold
        self.check_interval = 60  # Check every 60 seconds

        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def check_signal_outcome(
        self,
        signal: Signal,
        current_price: float,
        high_since_entry: float,
        low_since_entry: float
    ) -> Optional[str]:
        """
        Check if a signal has reached its target or stop.

        Args:
            signal: Signal to check
            current_price: Current market price
            high_since_entry: Highest price since signal
            low_since_entry: Lowest price since signal

        Returns:
            'WIN', 'LOSS', 'TIMEOUT', or None if still pending
        """
        if not signal.stop_loss or not signal.take_profit:
            return None

        # Check BUY signals
        if signal.signal_type == SignalType.BUY:
            if low_since_entry <= signal.stop_loss:
                return 'LOSS'
            if high_since_entry >= signal.take_profit:
                return 'WIN'

        # Check SELL signals
        elif signal.signal_type == SignalType.SELL:
            if high_since_entry >= signal.stop_loss:
                return 'LOSS'
            if low_since_entry <= signal.take_profit:
                return 'WIN'

        # Check timeout
        hours_since = (datetime.utcnow() - signal.timestamp).total_seconds() / 3600
        if hours_since >= self.max_hold_hours:
            # Determine win/loss based on current price
            if signal.signal_type == SignalType.BUY:
                return 'WIN' if current_price > signal.price else 'LOSS'
            else:
                return 'WIN' if current_price < signal.price else 'LOSS'

        return None

    def update_pending_signals(self, df: pd.DataFrame):
        """
        Update all pending signals with current market data.

        Args:
            df: DataFrame with recent candles
        """
        if df.empty:
            return

        pending = self.db.get_pending_signals()

        if not pending:
            return

        current_price = df['close'].iloc[-1]

        for signal in pending:
            # Get price data since signal
            signal_time = signal.timestamp
            df_since = df[pd.to_datetime(df['datetime']) >= signal_time]

            if df_since.empty:
                continue

            high_since = df_since['high'].max()
            low_since = df_since['low'].min()

            outcome = self.check_signal_outcome(
                signal,
                current_price,
                high_since,
                low_since
            )

            if outcome:
                # Calculate PnL
                if signal.signal_type == SignalType.BUY:
                    if outcome == 'WIN':
                        exit_price = signal.take_profit
                    else:
                        exit_price = signal.stop_loss
                    pnl = ((exit_price - signal.price) / signal.price) * 100
                else:
                    if outcome == 'WIN':
                        exit_price = signal.take_profit
                    else:
                        exit_price = signal.stop_loss
                    pnl = ((signal.price - exit_price) / signal.price) * 100

                # Update database
                self.db.update_signal_outcome(
                    signal.id,
                    outcome,
                    exit_price,
                    pnl
                )

                logger.info(
                    f"Signal #{signal.id} resolved: {outcome} | "
                    f"PnL: {pnl:+.2f}%"
                )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current performance statistics.

        Returns:
            Dict with performance metrics
        """
        return self.db.get_performance_stats()

    def get_recent_performance(self, days: int = 7) -> Dict[str, Any]:
        """
        Get performance for recent period.

        Args:
            days: Number of days to look back

        Returns:
            Dict with recent performance metrics
        """
        signals = self.db.get_signals(limit=1000)

        # Filter to recent
        cutoff = datetime.utcnow() - timedelta(days=days)
        recent = [s for s in signals if s.timestamp >= cutoff]

        if not recent:
            return {
                'period_days': days,
                'total_signals': 0,
                'resolved': 0,
                'win_rate': 0,
                'total_pnl': 0,
            }

        resolved = [s for s in recent if s.actual_outcome in ['WIN', 'LOSS']]
        winners = sum(1 for s in resolved if s.actual_outcome == 'WIN')
        pnls = [s.pnl_percent for s in resolved if s.pnl_percent is not None]

        return {
            'period_days': days,
            'total_signals': len(recent),
            'resolved': len(resolved),
            'pending': len(recent) - len(resolved),
            'winners': winners,
            'losers': len(resolved) - winners,
            'win_rate': winners / len(resolved) if resolved else 0,
            'total_pnl': sum(pnls) if pnls else 0,
            'avg_pnl': sum(pnls) / len(pnls) if pnls else 0,
        }

    def generate_report(self) -> str:
        """
        Generate a performance report.

        Returns:
            Formatted report string
        """
        stats = self.get_stats()
        recent_7d = self.get_recent_performance(7)
        recent_30d = self.get_recent_performance(30)

        lines = [
            "=" * 60,
            "SIGNAL PERFORMANCE REPORT",
            "=" * 60,
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            "",
            "-" * 60,
            "ALL TIME",
            "-" * 60,
            f"Total Signals:     {stats['total_signals']}",
            f"Resolved Trades:   {stats['resolved_trades']}",
            f"Win Rate:          {stats['win_rate']:.1%}",
            f"Total PnL:         {stats['total_pnl']:+.2f}%",
            f"Average PnL:       {stats['avg_pnl']:+.2f}%",
            f"Winners:           {stats['winners']}",
            f"Losers:            {stats['losers']}",
            "",
            "-" * 60,
            "LAST 7 DAYS",
            "-" * 60,
            f"Signals:           {recent_7d['total_signals']}",
            f"Resolved:          {recent_7d['resolved']}",
            f"Pending:           {recent_7d.get('pending', 0)}",
            f"Win Rate:          {recent_7d['win_rate']:.1%}",
            f"Total PnL:         {recent_7d['total_pnl']:+.2f}%",
            "",
            "-" * 60,
            "LAST 30 DAYS",
            "-" * 60,
            f"Signals:           {recent_30d['total_signals']}",
            f"Resolved:          {recent_30d['resolved']}",
            f"Win Rate:          {recent_30d['win_rate']:.1%}",
            f"Total PnL:         {recent_30d['total_pnl']:+.2f}%",
            "",
            "=" * 60,
        ]

        # Assessment
        if stats['resolved_trades'] >= 20:
            if stats['win_rate'] >= 0.55:
                lines.append("STATUS: PROFITABLE - Strategy performing well")
            elif stats['win_rate'] >= 0.48:
                lines.append("STATUS: MARGINAL - Strategy needs monitoring")
            else:
                lines.append("STATUS: LOSING - Consider stopping or adjusting")
        else:
            lines.append("STATUS: INSUFFICIENT DATA - Need 20+ trades")

        lines.append("=" * 60)

        return "\n".join(lines)

    def _monitoring_loop(self, get_candles_func):
        """Background monitoring loop."""
        while self._running:
            try:
                df = get_candles_func()
                if df is not None and not df.empty:
                    self.update_pending_signals(df)
            except Exception as e:
                logger.error(f"Error in performance tracking: {e}")

            # Sleep with interrupt check
            for _ in range(self.check_interval):
                if not self._running:
                    break
                time.sleep(1)

    def start(self, get_candles_func):
        """
        Start background performance monitoring.

        Args:
            get_candles_func: Function that returns current candles DataFrame
        """
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._monitoring_loop,
            args=(get_candles_func,),
            daemon=True,
            name="PerformanceTracker"
        )
        self._thread.start()
        logger.info("Performance tracker started")

    def stop(self):
        """Stop background monitoring."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("Performance tracker stopped")


# CLI entry point for generating reports
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Performance Tracker')
    parser.add_argument('--config', default='config.yaml', help='Config file')
    args = parser.parse_args()

    config = Config.load(args.config)
    tracker = PerformanceTracker(config)
    print(tracker.generate_report())
