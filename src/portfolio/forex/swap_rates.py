"""
Forex Swap/Rollover Rate Manager
================================

Manage overnight swap/rollover costs for Forex positions.

Key Concepts:
- Swap = Interest paid/received for holding positions overnight
- Rollover time: 5 PM New York (22:00 UTC winter, 21:00 summer)
- Wednesday triple swap: Compensates for weekend (3x normal rate)
- Positive swap: You receive interest (high-yield currency long)
- Negative swap: You pay interest (low-yield currency long)

Formula:
    Daily Swap = (Lot Size * Swap Rate * Point Value) / 10

Usage:
    from src.portfolio.forex.swap_rates import SwapRateManager

    manager = SwapRateManager()

    # Set rates (from broker API)
    manager.set_rate("EUR/USD", long=-0.5, short=0.1)

    # Calculate overnight cost for 1 lot long
    cost = manager.calculate_swap("EUR/USD", "BUY", 1.0, 1.1000)
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

from .constants import ROLLOVER_HOUR_UTC

logger = logging.getLogger(__name__)


@dataclass
class SwapRate:
    """
    Swap rate for a currency pair.

    Attributes:
        symbol: Currency pair (e.g., "EUR/USD")
        long_rate: Swap points for long positions (can be negative)
        short_rate: Swap points for short positions (can be negative)
        updated: Timestamp of last update
    """
    symbol: str
    long_rate: float   # Points per lot for long
    short_rate: float  # Points per lot for short
    updated: datetime = None

    def __post_init__(self):
        if self.updated is None:
            self.updated = datetime.utcnow()


# Default swap rates (in pips, approximate values)
# Positive = you receive, Negative = you pay
# These are examples and should be updated from broker API
DEFAULT_SWAP_RATES: Dict[str, Tuple[float, float]] = {
    # Majors (long, short)
    "EUR/USD": (-0.60, 0.10),   # Pay to be long EUR
    "GBP/USD": (-0.50, 0.05),   # Pay to be long GBP
    "USD/JPY": (0.50, -0.80),   # Receive to be long USD
    "USD/CHF": (0.30, -0.60),   # Receive to be long USD
    "AUD/USD": (-0.30, 0.05),   # Pay to be long AUD
    "USD/CAD": (0.20, -0.50),   # Receive to be long USD
    "NZD/USD": (-0.20, 0.00),   # Pay to be long NZD
    # Crosses
    "EUR/GBP": (-0.40, 0.10),
    "EUR/JPY": (0.20, -0.50),
    "GBP/JPY": (0.30, -0.60),
    "EUR/CHF": (-0.10, -0.20),
    "EUR/AUD": (-0.50, 0.20),
    "AUD/JPY": (0.40, -0.70),
    "CHF/JPY": (0.60, -0.90),
}


class SwapRateManager:
    """
    Manage Forex swap/rollover rates.

    Tracks overnight financing costs for holding positions.

    Key Features:
    - Track swap rates per currency pair
    - Calculate daily and multi-day swap costs
    - Detect rollover time
    - Account for triple swap Wednesdays

    Example:
        manager = SwapRateManager()

        # Calculate swap for 1 lot long EUR/USD
        swap = manager.calculate_swap("EUR/USD", "BUY", 1.0, 1.1000)
        print(f"Daily swap: ${swap:.2f}")

        # Check if holding overnight is beneficial
        if manager.is_positive_swap("EUR/USD", "BUY"):
            print("Positive carry trade")
    """

    # Rollover happens at 5 PM New York
    ROLLOVER_HOUR_UTC = ROLLOVER_HOUR_UTC
    TRIPLE_SWAP_DAY = 2  # Wednesday (0=Monday)

    def __init__(self, load_defaults: bool = True):
        """
        Initialize swap rate manager.

        Args:
            load_defaults: Load default swap rates
        """
        self._rates: Dict[str, SwapRate] = {}

        if load_defaults:
            self._load_default_rates()

        logger.info("SwapRateManager initialized")

    def _load_default_rates(self) -> None:
        """Load default swap rates."""
        for symbol, (long_rate, short_rate) in DEFAULT_SWAP_RATES.items():
            self._rates[symbol] = SwapRate(
                symbol=symbol,
                long_rate=long_rate,
                short_rate=short_rate
            )

    def set_rate(
        self,
        symbol: str,
        long_rate: float,
        short_rate: float
    ) -> None:
        """
        Set swap rates for a currency pair.

        Args:
            symbol: Currency pair
            long_rate: Swap points for long positions
            short_rate: Swap points for short positions
        """
        normalized = self._normalize_symbol(symbol)
        self._rates[normalized] = SwapRate(
            symbol=normalized,
            long_rate=long_rate,
            short_rate=short_rate
        )
        logger.debug(f"Swap rates set for {normalized}: long={long_rate}, short={short_rate}")

    def get_rate(self, symbol: str) -> Optional[SwapRate]:
        """
        Get swap rate for a currency pair.

        Args:
            symbol: Currency pair

        Returns:
            SwapRate or None if not found
        """
        normalized = self._normalize_symbol(symbol)
        return self._rates.get(normalized)

    def calculate_swap(
        self,
        symbol: str,
        side: str,
        lots: float,
        current_price: float,
        point_value: float = 10.0
    ) -> float:
        """
        Calculate swap cost/credit for one night.

        Formula: Swap = (Lots * Swap Points * Point Value) / 10

        Args:
            symbol: Currency pair
            side: "BUY" or "SELL"
            lots: Number of lots
            current_price: Current exchange rate (for JPY pairs)
            point_value: Value of 1 point (usually $10 for standard lot)

        Returns:
            Swap amount (negative = cost, positive = credit)

        Example:
            # 1 lot long EUR/USD with -0.6 swap rate
            swap = manager.calculate_swap("EUR/USD", "BUY", 1.0, 1.1000)
            # Returns: -0.60 (cost of $0.60)
        """
        normalized = self._normalize_symbol(symbol)
        rate = self._rates.get(normalized)

        if rate is None:
            logger.warning(f"No swap rate for {normalized}, assuming zero")
            return 0.0

        swap_points = rate.long_rate if side.upper() == "BUY" else rate.short_rate

        # Formula: Swap = (Lots * Swap Points * Point Value) / 10
        swap = (lots * swap_points * point_value) / 10

        return swap

    def calculate_holding_cost(
        self,
        symbol: str,
        side: str,
        lots: float,
        current_price: float,
        days: int
    ) -> float:
        """
        Calculate total swap cost for holding position for multiple days.

        Accounts for triple swap on Wednesday.

        Args:
            symbol: Currency pair
            side: Position side
            lots: Number of lots
            current_price: Current price
            days: Number of days holding

        Returns:
            Total swap cost/credit

        Example:
            # Hold 1 lot EUR/USD long for 7 days
            cost = manager.calculate_holding_cost("EUR/USD", "BUY", 1.0, 1.1000, 7)
        """
        daily_swap = self.calculate_swap(symbol, side, lots, current_price)

        # Count triple swap days (Wednesdays)
        today_weekday = datetime.utcnow().weekday()
        triple_count = 0

        for i in range(days):
            if (today_weekday + i) % 7 == self.TRIPLE_SWAP_DAY:
                triple_count += 1

        # Regular days = total days - triple days (they count as 3)
        regular_days = days - triple_count

        # Total swap = regular + 3x triple
        total = (regular_days * daily_swap) + (triple_count * 3 * daily_swap)

        return total

    def is_rollover_time(self, check_time: Optional[datetime] = None) -> bool:
        """
        Check if it's close to rollover time (within 30 minutes).

        Args:
            check_time: Time to check (default: now)

        Returns:
            True if within 30 minutes of rollover
        """
        check_time = check_time or datetime.utcnow()
        hour = check_time.hour
        minute = check_time.minute

        # Within 30 minutes before or after rollover
        if hour == self.ROLLOVER_HOUR_UTC - 1 and minute >= 30:
            return True
        if hour == self.ROLLOVER_HOUR_UTC:
            return True
        if hour == (self.ROLLOVER_HOUR_UTC + 1) % 24 and minute < 30:
            return True

        return False

    def is_triple_swap_day(self, check_date: Optional[datetime] = None) -> bool:
        """
        Check if today is triple swap day (Wednesday).

        Args:
            check_date: Date to check (default: today)

        Returns:
            True if Wednesday
        """
        check_date = check_date or datetime.utcnow()
        return check_date.weekday() == self.TRIPLE_SWAP_DAY

    def is_positive_swap(self, symbol: str, side: str) -> bool:
        """
        Check if position has positive swap (carry trade).

        Args:
            symbol: Currency pair
            side: "BUY" or "SELL"

        Returns:
            True if swap is positive (you receive interest)
        """
        normalized = self._normalize_symbol(symbol)
        rate = self._rates.get(normalized)

        if rate is None:
            return False

        swap_points = rate.long_rate if side.upper() == "BUY" else rate.short_rate
        return swap_points > 0

    def get_holding_impact_per_day(
        self,
        symbol: str,
        side: str,
        lots: float,
        account_balance: float
    ) -> float:
        """
        Get daily swap impact as percentage of account.

        Args:
            symbol: Currency pair
            side: Position side
            lots: Number of lots
            account_balance: Current account balance

        Returns:
            Daily impact as percentage (can be negative)
        """
        if account_balance == 0:
            return 0.0

        daily_swap = self.calculate_swap(symbol, side, lots, 1.0)
        return (daily_swap / account_balance) * 100

    def get_annual_swap_rate(
        self,
        symbol: str,
        side: str,
        lots: float,
        position_value: float
    ) -> float:
        """
        Calculate annualized swap rate as percentage of position.

        Args:
            symbol: Currency pair
            side: Position side
            lots: Number of lots
            position_value: Notional position value

        Returns:
            Annual swap rate as percentage
        """
        if position_value == 0:
            return 0.0

        daily_swap = self.calculate_swap(symbol, side, lots, 1.0)
        annual_swap = daily_swap * 365

        return (annual_swap / position_value) * 100

    def should_close_before_rollover(
        self,
        symbol: str,
        side: str,
        lots: float,
        unrealized_pnl: float,
        threshold_ratio: float = 0.5
    ) -> Tuple[bool, str]:
        """
        Check if position should be closed before rollover.

        Args:
            symbol: Currency pair
            side: Position side
            lots: Number of lots
            unrealized_pnl: Current unrealized P&L
            threshold_ratio: Close if swap exceeds this ratio of profit

        Returns:
            Tuple of (should_close, reason)
        """
        daily_swap = self.calculate_swap(symbol, side, lots, 1.0)

        # Adjust for triple swap day
        if self.is_triple_swap_day():
            daily_swap *= 3

        # If swap is positive, no need to close
        if daily_swap >= 0:
            return (False, f"Positive swap: ${daily_swap:.2f}")

        # Check if swap would eat too much of profit
        if unrealized_pnl > 0:
            swap_ratio = abs(daily_swap) / unrealized_pnl
            if swap_ratio > threshold_ratio:
                return (
                    True,
                    f"Swap ${abs(daily_swap):.2f} > {threshold_ratio*100}% of profit ${unrealized_pnl:.2f}"
                )

        # Check if in loss and swap makes it worse
        if unrealized_pnl < 0 and daily_swap < 0:
            return (
                True,
                f"Position in loss ${unrealized_pnl:.2f}, swap adds ${abs(daily_swap):.2f} cost"
            )

        return (False, "OK")

    def get_all_rates(self) -> Dict[str, SwapRate]:
        """Get all configured swap rates."""
        return self._rates.copy()

    def get_best_carry_trades(
        self,
        side: str,
        min_swap: float = 0.1
    ) -> list:
        """
        Get pairs with best positive swap for a given side.

        Args:
            side: "BUY" or "SELL"
            min_swap: Minimum swap rate to include

        Returns:
            List of (symbol, swap_rate) sorted by swap rate descending
        """
        result = []
        for symbol, rate in self._rates.items():
            swap = rate.long_rate if side.upper() == "BUY" else rate.short_rate
            if swap >= min_swap:
                result.append((symbol, swap))

        return sorted(result, key=lambda x: x[1], reverse=True)

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to standard format."""
        normalized = symbol.upper().replace("_", "/").replace("-", "/")
        if "/" not in normalized and len(normalized) == 6:
            normalized = f"{normalized[:3]}/{normalized[3:]}"
        return normalized
