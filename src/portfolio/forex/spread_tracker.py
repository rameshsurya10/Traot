"""
Forex Spread Tracker
====================

Track and analyze Forex bid/ask spreads in real-time.

Spreads impact trading costs:
- Entry cost = spread/2 (you buy at ask, mid-price is lower)
- Total round-trip cost = full spread
- Spreads widen during news events and low liquidity

Key Features:
- Real-time spread monitoring
- Historical spread statistics
- Spread widening detection
- Trade delay recommendations

Usage:
    from src.portfolio.forex.spread_tracker import SpreadTracker

    tracker = SpreadTracker()

    # Update with tick data
    tracker.update("EUR/USD", bid=1.0998, ask=1.1002)

    # Check spread status
    stats = tracker.get_spread_stats("EUR/USD")
    if stats.is_widened:
        print("Spread widened, may want to wait")

    # Adjust stop loss for spread
    adjusted_sl = tracker.adjust_stop_for_spread("EUR/USD", 1.0950, "BUY")
"""

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .pip_calculator import PipCalculator
from .constants import get_pair_config

logger = logging.getLogger(__name__)


@dataclass
class SpreadSnapshot:
    """Point-in-time spread data."""
    symbol: str
    bid: float
    ask: float
    spread_pips: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def mid_price(self) -> float:
        """Calculate mid-market price."""
        return (self.bid + self.ask) / 2

    @property
    def spread_percent(self) -> float:
        """Spread as percentage of mid price."""
        mid = self.mid_price
        return ((self.ask - self.bid) / mid * 100) if mid > 0 else 0


@dataclass
class SpreadStats:
    """Spread statistics for a currency pair."""
    symbol: str
    current_spread: float       # Current spread in pips
    average_spread: float       # Rolling average spread
    min_spread: float           # Minimum observed
    max_spread: float           # Maximum observed
    std_dev: float              # Standard deviation
    is_widened: bool            # True if above threshold
    sample_count: int           # Number of observations

    @property
    def widening_ratio(self) -> float:
        """How much current spread exceeds average."""
        return self.current_spread / self.average_spread if self.average_spread > 0 else 1.0


class SpreadTracker:
    """
    Track and analyze Forex spreads in real-time.

    Thread-safe for concurrent updates from data streams.

    Attributes:
        pip_calculator: PipCalculator for pip conversions
        history_size: Number of snapshots to keep per pair
        widening_threshold: Multiplier to detect spread widening

    Example:
        tracker = SpreadTracker()

        # Update from tick data
        tracker.update("EUR/USD", bid=1.0998, ask=1.1002)

        # Check if entry should be delayed
        should_wait, reason, spread = tracker.should_delay_entry("EUR/USD")
        if should_wait:
            print(f"Wait: {reason}")
    """

    def __init__(
        self,
        pip_calculator: Optional[PipCalculator] = None,
        history_size: int = 1000,
        widening_threshold: float = 1.5  # 50% above average = widened
    ):
        """
        Initialize spread tracker.

        Args:
            pip_calculator: PipCalculator instance
            history_size: Number of snapshots to keep per pair
            widening_threshold: Multiplier to detect spread widening
        """
        self.pip_calc = pip_calculator or PipCalculator()
        self.history_size = history_size
        self.widening_threshold = widening_threshold

        # Spread history per symbol
        self._history: Dict[str, deque] = {}
        self._current: Dict[str, SpreadSnapshot] = {}
        self._lock = threading.Lock()

        logger.info(
            f"SpreadTracker initialized: history_size={history_size}, "
            f"widening_threshold={widening_threshold}"
        )

    def update(self, symbol: str, bid: float, ask: float) -> SpreadSnapshot:
        """
        Update spread data for a currency pair.

        Args:
            symbol: Currency pair (e.g., "EUR/USD")
            bid: Current bid price
            ask: Current ask price

        Returns:
            SpreadSnapshot with current data
        """
        # Normalize symbol
        normalized = symbol.upper().replace("_", "/")
        if "/" not in normalized and len(normalized) == 6:
            normalized = f"{normalized[:3]}/{normalized[3:]}"

        # Calculate spread in pips
        try:
            pip_size = self.pip_calc.get_pip_size(normalized)
        except ValueError:
            # Unknown pair, estimate pip size
            pip_size = 0.0001 if "JPY" not in normalized else 0.01

        spread_pips = (ask - bid) / pip_size

        snapshot = SpreadSnapshot(
            symbol=normalized,
            bid=bid,
            ask=ask,
            spread_pips=spread_pips
        )

        with self._lock:
            # Update current
            self._current[normalized] = snapshot

            # Add to history
            if normalized not in self._history:
                self._history[normalized] = deque(maxlen=self.history_size)
            self._history[normalized].append(snapshot)

        return snapshot

    def get_current_spread(self, symbol: str) -> Optional[float]:
        """
        Get current spread in pips.

        Args:
            symbol: Currency pair

        Returns:
            Spread in pips or None if no data
        """
        normalized = self._normalize_symbol(symbol)
        with self._lock:
            if normalized in self._current:
                return self._current[normalized].spread_pips
        return None

    def get_current_tick(self, symbol: str) -> Optional[SpreadSnapshot]:
        """
        Get current tick data.

        Args:
            symbol: Currency pair

        Returns:
            SpreadSnapshot or None if no data
        """
        normalized = self._normalize_symbol(symbol)
        with self._lock:
            return self._current.get(normalized)

    def get_spread_stats(self, symbol: str) -> Optional[SpreadStats]:
        """
        Get spread statistics for a currency pair.

        Args:
            symbol: Currency pair

        Returns:
            SpreadStats or None if no data
        """
        normalized = self._normalize_symbol(symbol)

        with self._lock:
            if normalized not in self._history or len(self._history[normalized]) == 0:
                return None

            history = list(self._history[normalized])
            spreads = [s.spread_pips for s in history]

            current = self._current.get(normalized)
            current_spread = current.spread_pips if current else spreads[-1]

            # Calculate statistics
            avg_spread = sum(spreads) / len(spreads)
            min_spread = min(spreads)
            max_spread = max(spreads)

            # Standard deviation
            variance = sum((s - avg_spread) ** 2 for s in spreads) / len(spreads)
            std_dev = variance ** 0.5

            # Detect widening
            is_widened = current_spread > (avg_spread * self.widening_threshold)

            return SpreadStats(
                symbol=normalized,
                current_spread=current_spread,
                average_spread=avg_spread,
                min_spread=min_spread,
                max_spread=max_spread,
                std_dev=std_dev,
                is_widened=is_widened,
                sample_count=len(spreads)
            )

    def get_spread_cost(
        self,
        symbol: str,
        lot_size: float,
        current_price: float
    ) -> float:
        """
        Calculate spread cost in account currency.

        Args:
            symbol: Currency pair
            lot_size: Position size in units
            current_price: Current mid price

        Returns:
            Spread cost in account currency

        Example:
            # Cost of 1 standard lot EUR/USD with 1 pip spread
            cost = tracker.get_spread_cost("EUR/USD", 100000, 1.1000)
            # Returns: 10.0 (USD)
        """
        normalized = self._normalize_symbol(symbol)
        current_spread = self.get_current_spread(normalized)

        if current_spread is None:
            # Use typical spread from config
            try:
                config = get_pair_config(normalized)
                current_spread = config.typical_spread
            except ValueError:
                current_spread = 1.0  # Default 1 pip

        pip_value = self.pip_calc.get_pip_value(normalized, current_price, lot_size)
        return current_spread * pip_value

    def adjust_stop_for_spread(
        self,
        symbol: str,
        stop_price: float,
        side: str,
        spread_buffer_pips: float = 0.5
    ) -> float:
        """
        Adjust stop loss price to account for spread.

        For BUY positions: Stop triggers on bid, so subtract spread
        For SELL positions: Stop triggers on ask, so add spread

        Args:
            symbol: Currency pair
            stop_price: Intended stop price
            side: Position side ("BUY" or "SELL")
            spread_buffer_pips: Additional buffer in pips

        Returns:
            Adjusted stop price

        Example:
            # Long EUR/USD with stop at 1.0950
            adjusted = tracker.adjust_stop_for_spread("EUR/USD", 1.0950, "BUY")
            # Returns: 1.0948 (lower to account for spread)
        """
        normalized = self._normalize_symbol(symbol)
        stats = self.get_spread_stats(normalized)

        if stats is None:
            return stop_price

        # Use average spread + buffer for safety
        total_pips = stats.average_spread + spread_buffer_pips
        pip_size = self.pip_calc.get_pip_size(normalized)
        adjustment = total_pips * pip_size

        if side.upper() == "BUY":
            # Long position closes on bid, which is lower than mid
            # So stop should be further away (lower)
            return stop_price - adjustment
        else:
            # Short position closes on ask, which is higher than mid
            # So stop should be further away (higher)
            return stop_price + adjustment

    def adjust_take_profit_for_spread(
        self,
        symbol: str,
        tp_price: float,
        side: str,
        spread_buffer_pips: float = 0.5
    ) -> float:
        """
        Adjust take profit price to account for spread.

        Args:
            symbol: Currency pair
            tp_price: Intended take profit price
            side: Position side ("BUY" or "SELL")
            spread_buffer_pips: Additional buffer in pips

        Returns:
            Adjusted take profit price
        """
        normalized = self._normalize_symbol(symbol)
        stats = self.get_spread_stats(normalized)

        if stats is None:
            return tp_price

        total_pips = stats.average_spread + spread_buffer_pips
        pip_size = self.pip_calc.get_pip_size(normalized)
        adjustment = total_pips * pip_size

        if side.upper() == "BUY":
            # Long closes on bid, so TP should be higher
            return tp_price + adjustment
        else:
            # Short closes on ask, so TP should be lower
            return tp_price - adjustment

    def should_delay_entry(
        self,
        symbol: str,
        max_spread_pips: Optional[float] = None
    ) -> Tuple[bool, str, float]:
        """
        Check if entry should be delayed due to wide spread.

        Args:
            symbol: Currency pair
            max_spread_pips: Maximum acceptable spread

        Returns:
            Tuple of (should_delay, reason, current_spread)

        Example:
            delay, reason, spread = tracker.should_delay_entry("EUR/USD")
            if delay:
                print(f"Entry delayed: {reason}")
        """
        normalized = self._normalize_symbol(symbol)
        stats = self.get_spread_stats(normalized)

        if stats is None:
            return (False, "No spread data", 0)

        # Use configured max or 2x average
        if max_spread_pips is None:
            max_spread_pips = stats.average_spread * 2

        if stats.current_spread > max_spread_pips:
            return (
                True,
                f"Spread {stats.current_spread:.1f} pips > max {max_spread_pips:.1f}",
                stats.current_spread
            )

        if stats.is_widened:
            return (
                True,
                f"Spread widened: {stats.current_spread:.1f} > avg {stats.average_spread:.1f}",
                stats.current_spread
            )

        return (False, "OK", stats.current_spread)

    def get_all_spreads(self) -> Dict[str, float]:
        """
        Get current spreads for all tracked pairs.

        Returns:
            Dict of symbol -> spread in pips
        """
        with self._lock:
            return {
                symbol: snapshot.spread_pips
                for symbol, snapshot in self._current.items()
            }

    def get_widened_pairs(self) -> List[str]:
        """
        Get list of pairs with widened spreads.

        Returns:
            List of symbol names with widened spreads
        """
        widened = []
        with self._lock:
            for symbol in self._current.keys():
                stats = self.get_spread_stats(symbol)
                if stats and stats.is_widened:
                    widened.append(symbol)
        return widened

    def clear_history(self, symbol: Optional[str] = None) -> None:
        """
        Clear spread history.

        Args:
            symbol: Specific symbol to clear, or None for all
        """
        with self._lock:
            if symbol:
                normalized = self._normalize_symbol(symbol)
                if normalized in self._history:
                    self._history[normalized].clear()
                if normalized in self._current:
                    del self._current[normalized]
            else:
                self._history.clear()
                self._current.clear()

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to standard format."""
        normalized = symbol.upper().replace("_", "/").replace("-", "/")
        if "/" not in normalized and len(normalized) == 6:
            normalized = f"{normalized[:3]}/{normalized[3:]}"
        return normalized
