"""
Forex Leverage Manager
======================

Track and enforce leverage limits for Forex trading.

Key Concepts:
- Leverage: Ratio of position size to account equity (e.g., 50:1)
- Margin: Collateral required to open a position
- Free Margin: Available margin for new positions
- Margin Level: Equity / Used Margin (triggers margin call/stop out)

Regulatory Limits:
- US (NFA/CFTC): 50:1 major pairs, 20:1 minor
- EU (ESMA): 30:1 major, 20:1 minor
- UK (FCA): 30:1 major, 20:1 minor
- Offshore: Up to 500:1

Usage:
    from src.portfolio.forex.leverage_manager import LeverageManager

    manager = LeverageManager(max_leverage=50.0)

    # Check if position can be opened
    can_trade, margin_req, reason = manager.can_open_position(
        account_equity=10000,
        position_value=100000,  # 1 standard lot
        current_exposure=0
    )
    # Returns: (True, 2000, "OK")
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .constants import (
    US_MAX_LEVERAGE,
    MARGIN_CALL_LEVEL,
    STOP_OUT_LEVEL,
    FOREX_PAIRS,
)

logger = logging.getLogger(__name__)


@dataclass
class LeverageState:
    """
    Current leverage state for an account.

    Attributes:
        total_exposure: Total notional value of all positions
        account_equity: Account equity (cash + unrealized P&L)
        used_leverage: Current leverage ratio (exposure / equity)
        available_leverage: Remaining leverage capacity
        margin_used: Total margin allocated to positions
        free_margin: Available margin for new positions
        margin_level: Equity / Margin ratio (for margin call detection)
    """
    total_exposure: float
    account_equity: float
    used_leverage: float
    available_leverage: float
    margin_used: float
    free_margin: float
    margin_level: float = 0.0

    @property
    def is_margin_call(self) -> bool:
        """Check if account is in margin call."""
        return self.margin_level < MARGIN_CALL_LEVEL and self.margin_level > 0

    @property
    def is_stop_out(self) -> bool:
        """Check if account should be stopped out."""
        return self.margin_level < STOP_OUT_LEVEL and self.margin_level > 0

    @property
    def utilization_percent(self) -> float:
        """Leverage utilization as percentage of max."""
        return (self.used_leverage / US_MAX_LEVERAGE) * 100 if US_MAX_LEVERAGE > 0 else 0


class LeverageManager:
    """
    Manage leverage for Forex trading.

    Enforces regulatory leverage limits and tracks margin usage.

    Attributes:
        max_leverage: Maximum allowed leverage ratio
        margin_call_level: Margin level triggering margin call (default 50%)
        stop_out_level: Margin level triggering forced liquidation (default 30%)

    Example:
        manager = LeverageManager(max_leverage=50.0)

        # Check if trade is allowed
        can_trade, margin, reason = manager.can_open_position(
            account_equity=10000,
            position_value=100000,
            current_exposure=0
        )

        if can_trade:
            print(f"Trade allowed, margin required: ${margin}")
        else:
            print(f"Trade blocked: {reason}")
    """

    def __init__(
        self,
        max_leverage: float = US_MAX_LEVERAGE,
        margin_call_level: float = MARGIN_CALL_LEVEL,
        stop_out_level: float = STOP_OUT_LEVEL
    ):
        """
        Initialize leverage manager.

        Args:
            max_leverage: Maximum allowed leverage (default 50:1)
            margin_call_level: Margin level for margin call (0.5 = 50%)
            stop_out_level: Margin level for forced liquidation (0.3 = 30%)
        """
        self.max_leverage = max_leverage
        self.margin_call_level = margin_call_level
        self.stop_out_level = stop_out_level

        # Track positions for margin calculation
        self._positions: Dict[str, float] = {}  # symbol -> notional value

        logger.info(
            f"LeverageManager initialized: max_leverage={max_leverage}:1, "
            f"margin_call={margin_call_level*100}%, stop_out={stop_out_level*100}%"
        )

    def calculate_required_margin(
        self,
        position_value: float,
        leverage: Optional[float] = None
    ) -> float:
        """
        Calculate margin required for a position.

        Formula: Margin = Position Value / Leverage

        Args:
            position_value: Notional value of position
            leverage: Leverage to use (default: max_leverage)

        Returns:
            Required margin in account currency

        Example:
            # 1 standard lot EUR/USD at 1.1000 = $110,000
            # At 50:1 leverage
            margin = manager.calculate_required_margin(110000)
            # Returns: 2200.0 (need $2,200 margin)
        """
        leverage = leverage or self.max_leverage
        if leverage == 0:
            return position_value  # No leverage = 100% margin
        return position_value / leverage

    def calculate_position_value(
        self,
        quantity: float,
        price: float,
        lot_size: float = 100000
    ) -> float:
        """
        Calculate notional value of a Forex position.

        Args:
            quantity: Number of lots
            price: Current exchange rate
            lot_size: Units per lot (default: standard lot)

        Returns:
            Notional value in quote currency

        Example:
            # 1 standard lot EUR/USD at 1.1000
            value = manager.calculate_position_value(1.0, 1.1000)
            # Returns: 110000.0
        """
        return quantity * lot_size * price

    def calculate_max_position(
        self,
        account_equity: float,
        current_exposure: float = 0,
        leverage: Optional[float] = None
    ) -> float:
        """
        Calculate maximum position value allowed.

        Args:
            account_equity: Current account equity
            current_exposure: Existing position exposure
            leverage: Leverage to use

        Returns:
            Maximum additional position value allowed

        Example:
            max_pos = manager.calculate_max_position(10000, 100000)
            # At 50:1 leverage, max exposure = $500,000
            # With $100,000 existing, can add $400,000 more
        """
        leverage = leverage or self.max_leverage
        max_exposure = account_equity * leverage
        return max(0, max_exposure - current_exposure)

    def calculate_max_lots(
        self,
        account_equity: float,
        current_exposure: float,
        price: float,
        lot_size: float = 100000,
        leverage: Optional[float] = None
    ) -> float:
        """
        Calculate maximum lots that can be opened.

        Args:
            account_equity: Current account equity
            current_exposure: Existing position exposure
            price: Current exchange rate
            lot_size: Units per lot
            leverage: Leverage to use

        Returns:
            Maximum lots that can be opened
        """
        max_value = self.calculate_max_position(account_equity, current_exposure, leverage)
        if price == 0 or lot_size == 0:
            return 0.0
        return max_value / (lot_size * price)

    def can_open_position(
        self,
        account_equity: float,
        position_value: float,
        current_exposure: float = 0,
        free_margin: Optional[float] = None
    ) -> Tuple[bool, float, str]:
        """
        Check if a new position can be opened.

        Args:
            account_equity: Current account equity
            position_value: Value of proposed position
            current_exposure: Current total exposure
            free_margin: Available margin (calculated if not provided)

        Returns:
            Tuple of (can_open, margin_required, reason)

        Example:
            can, margin, reason = manager.can_open_position(
                account_equity=10000,
                position_value=600000,  # 6 lots
                current_exposure=0
            )
            # Returns: (False, 12000, "Exceeds max leverage: 60.0x > 50x")
        """
        required_margin = self.calculate_required_margin(position_value)

        # Check leverage limit
        new_exposure = current_exposure + position_value
        new_leverage = new_exposure / account_equity if account_equity > 0 else float('inf')

        if new_leverage > self.max_leverage:
            return (
                False,
                required_margin,
                f"Exceeds max leverage: {new_leverage:.1f}x > {self.max_leverage:.0f}x"
            )

        # Check margin availability
        if free_margin is not None:
            if required_margin > free_margin:
                return (
                    False,
                    required_margin,
                    f"Insufficient margin: need ${required_margin:,.2f}, "
                    f"have ${free_margin:,.2f}"
                )

        return (True, required_margin, "OK")

    def get_leverage_state(
        self,
        account_equity: float,
        positions: Dict[str, float]
    ) -> LeverageState:
        """
        Get comprehensive leverage state.

        Args:
            account_equity: Current account equity
            positions: Dict of symbol -> notional value

        Returns:
            LeverageState with all metrics

        Example:
            state = manager.get_leverage_state(
                account_equity=10000,
                positions={"EUR/USD": 110000, "GBP/USD": 50000}
            )
            print(f"Used leverage: {state.used_leverage:.1f}x")
        """
        total_exposure = sum(abs(v) for v in positions.values())
        used_leverage = total_exposure / account_equity if account_equity > 0 else 0
        available_leverage = max(0, self.max_leverage - used_leverage)
        margin_used = self.calculate_required_margin(total_exposure)
        free_margin = max(0, account_equity - margin_used)

        # Calculate margin level
        margin_level = account_equity / margin_used if margin_used > 0 else float('inf')

        return LeverageState(
            total_exposure=total_exposure,
            account_equity=account_equity,
            used_leverage=used_leverage,
            available_leverage=available_leverage,
            margin_used=margin_used,
            free_margin=free_margin,
            margin_level=margin_level
        )

    def check_margin_call(
        self,
        account_equity: float,
        margin_used: float
    ) -> Tuple[bool, bool, float]:
        """
        Check if account is in margin call or stop out.

        Args:
            account_equity: Current equity
            margin_used: Total margin in use

        Returns:
            Tuple of (is_margin_call, is_stop_out, margin_level)

        Example:
            is_mc, is_so, level = manager.check_margin_call(800, 2000)
            # Returns: (True, False, 0.40) - 40% margin level = margin call
        """
        if margin_used == 0:
            return (False, False, float('inf'))

        margin_level = account_equity / margin_used
        is_margin_call = margin_level < self.margin_call_level
        is_stop_out = margin_level < self.stop_out_level

        return (is_margin_call, is_stop_out, margin_level)

    def calculate_liquidation_price(
        self,
        symbol: str,
        entry_price: float,
        position_units: float,
        side: str,
        account_equity: float,
        other_margin_used: float = 0
    ) -> float:
        """
        Calculate price at which position would be liquidated (stop out).

        Args:
            symbol: Currency pair
            entry_price: Entry price
            position_units: Position size in units
            side: "BUY" or "SELL"
            account_equity: Current account equity
            other_margin_used: Margin used by other positions

        Returns:
            Liquidation price

        Example:
            liq_price = manager.calculate_liquidation_price(
                "EUR/USD", 1.1000, 100000, "BUY", 10000
            )
        """
        # This position's margin
        position_value = position_units * entry_price
        position_margin = self.calculate_required_margin(position_value)

        # At stop out, equity = stop_out_level * total_margin
        # Loss that triggers stop out = equity - (stop_out_level * total_margin)
        total_margin = position_margin + other_margin_used
        max_loss = account_equity - (self.stop_out_level * total_margin)

        # Price change that causes this loss
        if position_units == 0:
            return entry_price

        price_change = max_loss / position_units

        if side.upper() == "BUY":
            # Long position loses when price goes down
            return entry_price - price_change
        else:
            # Short position loses when price goes up
            return entry_price + price_change

    def get_position_leverage(
        self,
        position_value: float,
        account_equity: float
    ) -> float:
        """
        Calculate leverage for a single position.

        Args:
            position_value: Notional value of position
            account_equity: Account equity

        Returns:
            Leverage ratio for this position
        """
        if account_equity == 0:
            return float('inf')
        return position_value / account_equity

    def adjust_position_for_leverage(
        self,
        desired_value: float,
        account_equity: float,
        current_exposure: float
    ) -> float:
        """
        Adjust position value to fit within leverage limits.

        Args:
            desired_value: Desired position value
            account_equity: Account equity
            current_exposure: Current total exposure

        Returns:
            Adjusted position value (may be reduced)
        """
        max_additional = self.calculate_max_position(account_equity, current_exposure)
        return min(desired_value, max_additional)

    def is_forex_symbol(self, symbol: str) -> bool:
        """Check if symbol is a configured Forex pair."""
        normalized = symbol.upper().replace("_", "/").replace("-", "/")
        if "/" not in normalized and len(normalized) == 6:
            normalized = f"{normalized[:3]}/{normalized[3:]}"
        return normalized in FOREX_PAIRS
