"""
Forex Position Sizer
====================

Calculate optimal position sizes for Forex trades using pip-based risk management.

Key Formula:
    Lots = (Account Risk $) / (Stop Loss Pips * Pip Value per Lot)

Example:
    Account: $10,000
    Risk per trade: 1% = $100
    Stop loss: 50 pips
    EUR/USD pip value: $10/pip (standard lot)

    Lots = $100 / (50 * $10) = 0.20 lots = 20,000 units

Features:
- Risk-based position sizing (% of account)
- Maximum position limits
- Leverage-aware sizing
- Spread cost adjustment
- Multi-currency support

Usage:
    from src.portfolio.forex.position_sizer import ForexPositionSizer

    sizer = ForexPositionSizer(
        max_risk_percent=1.0,
        max_position_percent=10.0
    )

    size = sizer.calculate_position_size(
        symbol="EUR/USD",
        account_equity=10000,
        stop_pips=50,
        current_price=1.1000
    )
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

from .pip_calculator import PipCalculator
from .leverage_manager import LeverageManager
from .spread_tracker import SpreadTracker
from .constants import (
    LotType,
    US_MAX_LEVERAGE,
)

logger = logging.getLogger(__name__)


@dataclass
class PositionSizeResult:
    """
    Result of position size calculation.

    Attributes:
        symbol: Currency pair
        lots: Position size in lots
        units: Position size in units
        notional_value: Total notional value
        margin_required: Margin needed
        risk_amount: Risk in account currency
        stop_pips: Stop loss in pips
        pip_value: Value per pip
        max_lots: Maximum allowed lots
        was_reduced: True if size was reduced due to constraints
        reduction_reason: Reason for reduction if applicable
    """
    symbol: str
    lots: float
    units: float
    notional_value: float
    margin_required: float
    risk_amount: float
    stop_pips: float
    pip_value: float
    max_lots: float
    was_reduced: bool = False
    reduction_reason: str = ""

    @property
    def lot_type(self) -> str:
        """Describe the lot size type."""
        if self.units >= 100000:
            return "Standard"
        elif self.units >= 10000:
            return "Mini"
        elif self.units >= 1000:
            return "Micro"
        else:
            return "Nano"

    @property
    def position_risk_ratio(self) -> float:
        """Risk as percentage of notional value."""
        if self.notional_value == 0:
            return 0.0
        return (self.risk_amount / self.notional_value) * 100


class ForexPositionSizer:
    """
    Calculate optimal Forex position sizes.

    Uses pip-based risk management to determine position sizes while
    respecting leverage and exposure limits.

    Attributes:
        max_risk_percent: Maximum risk per trade (default 1%)
        max_position_percent: Maximum position as % of account (default 10%)
        pip_calculator: PipCalculator instance
        leverage_manager: LeverageManager instance
        spread_tracker: Optional SpreadTracker for spread adjustments

    Example:
        sizer = ForexPositionSizer(max_risk_percent=1.0)

        result = sizer.calculate_position_size(
            symbol="EUR/USD",
            account_equity=10000,
            stop_pips=50,
            current_price=1.1000
        )

        print(f"Position: {result.lots:.2f} lots ({result.units:,.0f} units)")
        print(f"Risk: ${result.risk_amount:.2f}")
    """

    def __init__(
        self,
        max_risk_percent: float = 1.0,
        max_position_percent: float = 10.0,
        pip_calculator: Optional[PipCalculator] = None,
        leverage_manager: Optional[LeverageManager] = None,
        spread_tracker: Optional[SpreadTracker] = None,
        max_leverage: float = US_MAX_LEVERAGE
    ):
        """
        Initialize Forex position sizer.

        Args:
            max_risk_percent: Max risk per trade (1.0 = 1%)
            max_position_percent: Max position as % of equity
            pip_calculator: PipCalculator instance
            leverage_manager: LeverageManager instance
            spread_tracker: SpreadTracker for spread adjustments
            max_leverage: Maximum leverage allowed (default 50:1)
        """
        self.max_risk_percent = max_risk_percent
        self.max_position_percent = max_position_percent
        self.pip_calc = pip_calculator or PipCalculator()
        self.leverage_mgr = leverage_manager or LeverageManager(max_leverage)
        self.spread_tracker = spread_tracker
        self.max_leverage = max_leverage

        logger.info(
            f"ForexPositionSizer initialized: max_risk={max_risk_percent}%, "
            f"max_position={max_position_percent}%"
        )

    def calculate_position_size(
        self,
        symbol: str,
        account_equity: float,
        stop_pips: float,
        current_price: float,
        risk_percent: Optional[float] = None,
        current_exposure: float = 0,
        fx_rates: Optional[Dict[str, float]] = None,
        include_spread: bool = True
    ) -> PositionSizeResult:
        """
        Calculate optimal position size based on risk parameters.

        Formula: Lots = Risk Amount / (Stop Pips * Pip Value per Lot)

        Args:
            symbol: Currency pair (e.g., "EUR/USD")
            account_equity: Account equity in base currency
            stop_pips: Stop loss distance in pips
            current_price: Current exchange rate
            risk_percent: Risk % (overrides max_risk_percent if provided)
            current_exposure: Current total Forex exposure
            fx_rates: FX rates for cross pair calculations
            include_spread: Adjust stop loss for spread

        Returns:
            PositionSizeResult with calculated position details

        Example:
            result = sizer.calculate_position_size(
                "EUR/USD", 10000, 50, 1.1000
            )
            # For 1% risk on $10k with 50 pip stop:
            # Lots = $100 / (50 * $10) = 0.20 lots
        """
        # Validate inputs
        if account_equity <= 0:
            logger.warning("Account equity <= 0, returning zero position")
            return self._zero_result(symbol, "No account equity")

        if stop_pips <= 0:
            logger.warning("Stop pips <= 0, returning zero position")
            return self._zero_result(symbol, "Invalid stop loss")

        if current_price <= 0:
            logger.warning("Price <= 0, returning zero position")
            return self._zero_result(symbol, "Invalid price")

        # Use provided risk or default
        risk_pct = risk_percent if risk_percent is not None else self.max_risk_percent
        risk_amount = account_equity * (risk_pct / 100)

        # Adjust stop for spread if tracker available and requested
        effective_stop_pips = stop_pips
        if include_spread and self.spread_tracker:
            spread = self.spread_tracker.get_current_spread(symbol)
            if spread is not None and spread > 0:
                # Add half spread to stop (entry slippage)
                effective_stop_pips = stop_pips + (spread / 2)

        # Get pip value for standard lot
        pip_value = self.pip_calc.get_pip_value(
            symbol, current_price, lot_size=LotType.STANDARD.value, fx_rates=fx_rates
        )

        if pip_value <= 0:
            logger.warning(f"Pip value <= 0 for {symbol}, returning zero position")
            return self._zero_result(symbol, "Cannot calculate pip value")

        # Calculate position size
        # Formula: Lots = Risk Amount / (Stop Pips * Pip Value per Lot)
        lots = risk_amount / (effective_stop_pips * pip_value)
        units = lots * LotType.STANDARD.value
        notional_value = units * current_price

        # Calculate margin required
        margin_required = self.leverage_mgr.calculate_required_margin(notional_value)

        # Check constraints and reduce if necessary
        was_reduced = False
        reduction_reason = ""

        # 1. Check max position percent
        max_position_value = account_equity * (self.max_position_percent / 100)
        if notional_value > max_position_value:
            old_lots = lots
            lots = max_position_value / (LotType.STANDARD.value * current_price)
            units = lots * LotType.STANDARD.value
            notional_value = units * current_price
            margin_required = self.leverage_mgr.calculate_required_margin(notional_value)
            was_reduced = True
            reduction_reason = f"Max position limit ({self.max_position_percent}%)"
            logger.debug(f"Position reduced from {old_lots:.2f} to {lots:.2f} lots: {reduction_reason}")

        # 2. Check leverage limits
        can_open, _, leverage_reason = self.leverage_mgr.can_open_position(
            account_equity=account_equity,
            position_value=notional_value,
            current_exposure=current_exposure
        )

        if not can_open:
            # Reduce to fit leverage
            max_additional = self.leverage_mgr.calculate_max_position(
                account_equity, current_exposure
            )
            if max_additional > 0:
                old_lots = lots
                lots = max_additional / (LotType.STANDARD.value * current_price)
                units = lots * LotType.STANDARD.value
                notional_value = units * current_price
                margin_required = self.leverage_mgr.calculate_required_margin(notional_value)
                was_reduced = True
                reduction_reason = leverage_reason
                logger.debug(f"Position reduced for leverage: {old_lots:.2f} -> {lots:.2f}")
            else:
                return self._zero_result(symbol, leverage_reason)

        # 3. Round to valid lot size
        lots = self._round_to_lot_precision(lots)
        units = lots * LotType.STANDARD.value
        notional_value = units * current_price
        margin_required = self.leverage_mgr.calculate_required_margin(notional_value)

        # Calculate actual risk amount after rounding
        actual_risk = lots * effective_stop_pips * pip_value

        # Calculate max lots based on constraints
        max_lots = self._calculate_max_lots(
            account_equity, current_price, current_exposure
        )

        return PositionSizeResult(
            symbol=symbol,
            lots=lots,
            units=units,
            notional_value=notional_value,
            margin_required=margin_required,
            risk_amount=actual_risk,
            stop_pips=effective_stop_pips,
            pip_value=pip_value,
            max_lots=max_lots,
            was_reduced=was_reduced,
            reduction_reason=reduction_reason
        )

    def calculate_lots_for_fixed_risk(
        self,
        symbol: str,
        risk_amount: float,
        stop_pips: float,
        current_price: float,
        fx_rates: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate lots for a fixed dollar risk amount.

        Args:
            symbol: Currency pair
            risk_amount: Fixed amount to risk
            stop_pips: Stop loss in pips
            current_price: Current price
            fx_rates: FX rates for cross pairs

        Returns:
            Number of lots

        Example:
            lots = sizer.calculate_lots_for_fixed_risk(
                "EUR/USD", 100, 50, 1.1000
            )
            # Returns: 0.20 lots for $100 risk with 50 pip stop
        """
        if stop_pips <= 0 or risk_amount <= 0:
            return 0.0

        pip_value = self.pip_calc.get_pip_value(
            symbol, current_price, LotType.STANDARD.value, fx_rates
        )

        if pip_value <= 0:
            return 0.0

        lots = risk_amount / (stop_pips * pip_value)
        return self._round_to_lot_precision(lots)

    def calculate_risk_for_position(
        self,
        symbol: str,
        lots: float,
        stop_pips: float,
        current_price: float,
        fx_rates: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate dollar risk for a given position size.

        Args:
            symbol: Currency pair
            lots: Position size in lots
            stop_pips: Stop loss in pips
            current_price: Current price
            fx_rates: FX rates for cross pairs

        Returns:
            Risk amount in account currency

        Example:
            risk = sizer.calculate_risk_for_position(
                "EUR/USD", 0.20, 50, 1.1000
            )
            # Returns: $100.00
        """
        pip_value = self.pip_calc.get_pip_value(
            symbol, current_price, LotType.STANDARD.value, fx_rates
        )
        return lots * stop_pips * pip_value

    def calculate_stop_pips_for_risk(
        self,
        symbol: str,
        lots: float,
        risk_amount: float,
        current_price: float,
        fx_rates: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate stop loss distance in pips for a fixed risk.

        Args:
            symbol: Currency pair
            lots: Position size in lots
            risk_amount: Amount to risk
            current_price: Current price
            fx_rates: FX rates for cross pairs

        Returns:
            Stop loss distance in pips

        Example:
            stop = sizer.calculate_stop_pips_for_risk(
                "EUR/USD", 0.20, 100, 1.1000
            )
            # Returns: 50.0 pips
        """
        if lots <= 0 or risk_amount <= 0:
            return 0.0

        pip_value = self.pip_calc.get_pip_value(
            symbol, current_price, LotType.STANDARD.value, fx_rates
        )

        if pip_value <= 0:
            return 0.0

        return risk_amount / (lots * pip_value)

    def get_position_metrics(
        self,
        symbol: str,
        lots: float,
        entry_price: float,
        current_price: float,
        stop_price: float,
        take_profit_price: Optional[float] = None,
        side: str = "BUY",
        fx_rates: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Get comprehensive metrics for a position.

        Args:
            symbol: Currency pair
            lots: Position size in lots
            entry_price: Entry price
            current_price: Current price
            stop_price: Stop loss price
            take_profit_price: Take profit price (optional)
            side: "BUY" or "SELL"
            fx_rates: FX rates for cross pairs

        Returns:
            Dict with position metrics

        Example:
            metrics = sizer.get_position_metrics(
                "EUR/USD", 0.20, 1.1000, 1.1050, 1.0950
            )
        """
        units = lots * LotType.STANDARD.value
        pip_value = self.pip_calc.get_pip_value(symbol, current_price, units, fx_rates)

        # Calculate pips to stop
        stop_pips = abs(self.pip_calc.price_to_pips(symbol, entry_price, stop_price))

        # Calculate risk
        risk_amount = stop_pips * pip_value

        # Calculate unrealized P&L
        if side.upper() == "BUY":
            pnl_pips = self.pip_calc.price_to_pips(symbol, current_price, entry_price)
        else:
            pnl_pips = self.pip_calc.price_to_pips(symbol, entry_price, current_price)

        unrealized_pnl = pnl_pips * pip_value

        # Calculate reward if TP provided
        reward_amount = 0.0
        reward_pips = 0.0
        risk_reward_ratio = 0.0

        if take_profit_price:
            reward_pips = abs(self.pip_calc.price_to_pips(symbol, entry_price, take_profit_price))
            reward_amount = reward_pips * pip_value
            if stop_pips > 0:
                risk_reward_ratio = reward_pips / stop_pips

        return {
            "symbol": symbol,
            "lots": lots,
            "units": units,
            "entry_price": entry_price,
            "current_price": current_price,
            "side": side,
            "pip_value": pip_value,
            "stop_price": stop_price,
            "stop_pips": stop_pips,
            "risk_amount": risk_amount,
            "take_profit_price": take_profit_price,
            "reward_pips": reward_pips,
            "reward_amount": reward_amount,
            "risk_reward_ratio": risk_reward_ratio,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pips": pnl_pips,
            "notional_value": units * current_price,
        }

    def scale_position(
        self,
        current_lots: float,
        scale_factor: float,
        direction: str = "in"
    ) -> Tuple[float, float]:
        """
        Calculate scaled position size.

        Args:
            current_lots: Current position size
            scale_factor: Factor to scale by (0.5 = 50%)
            direction: "in" to add, "out" to reduce

        Returns:
            Tuple of (new_total_lots, change_lots)

        Example:
            # Scale in 50% more
            new_lots, add_lots = sizer.scale_position(0.20, 0.5, "in")
            # Returns: (0.30, 0.10)

            # Scale out 50%
            new_lots, remove_lots = sizer.scale_position(0.20, 0.5, "out")
            # Returns: (0.10, 0.10)
        """
        change = current_lots * scale_factor

        if direction.lower() == "in":
            new_total = current_lots + change
        else:
            new_total = max(0, current_lots - change)
            change = current_lots - new_total

        return (
            self._round_to_lot_precision(new_total),
            self._round_to_lot_precision(change)
        )

    def _calculate_max_lots(
        self,
        account_equity: float,
        current_price: float,
        current_exposure: float = 0
    ) -> float:
        """Calculate maximum lots based on constraints."""
        # Max by position percent
        max_position_value = account_equity * (self.max_position_percent / 100)
        max_by_position = max_position_value / (LotType.STANDARD.value * current_price)

        # Max by leverage
        max_by_leverage = self.leverage_mgr.calculate_max_lots(
            account_equity=account_equity,
            current_exposure=current_exposure,
            price=current_price,
            lot_size=LotType.STANDARD.value
        )

        return min(max_by_position, max_by_leverage)

    def _round_to_lot_precision(self, lots: float, precision: float = 0.01) -> float:
        """Round lots to valid precision (default: 0.01 = micro lot)."""
        return round(lots / precision) * precision

    def _zero_result(self, symbol: str, reason: str) -> PositionSizeResult:
        """Create a zero-size result."""
        return PositionSizeResult(
            symbol=symbol,
            lots=0.0,
            units=0.0,
            notional_value=0.0,
            margin_required=0.0,
            risk_amount=0.0,
            stop_pips=0.0,
            pip_value=0.0,
            max_lots=0.0,
            was_reduced=True,
            reduction_reason=reason
        )


class ForexKellyPosition:
    """
    Kelly Criterion position sizing for Forex.

    Uses win rate and profit factor to determine optimal bet size.

    Kelly % = W - [(1-W) / R]
    Where:
        W = Win rate (probability of winning)
        R = Win/Loss ratio (average win / average loss)

    Example:
        kelly = ForexKellyPosition()
        kelly_pct = kelly.calculate(
            win_rate=0.55,
            avg_win_pips=30,
            avg_loss_pips=20
        )
        # Returns optimal risk % (often halved for safety)
    """

    def __init__(self, max_kelly_fraction: float = 0.5):
        """
        Initialize Kelly position sizer.

        Args:
            max_kelly_fraction: Fraction of Kelly to use (0.5 = half Kelly)
        """
        self.max_fraction = max_kelly_fraction

    def calculate(
        self,
        win_rate: float,
        avg_win_pips: float,
        avg_loss_pips: float
    ) -> float:
        """
        Calculate Kelly optimal bet percentage.

        Args:
            win_rate: Historical win rate (0-1)
            avg_win_pips: Average winning trade in pips
            avg_loss_pips: Average losing trade in pips

        Returns:
            Optimal risk percentage (already fractional Kelly)

        Example:
            kelly_pct = kelly.calculate(0.55, 30, 20)
            # With 55% win rate, 1.5 R/R
            # Kelly = 0.55 - (0.45/1.5) = 0.25 = 25%
            # Half Kelly = 12.5%
        """
        if avg_loss_pips <= 0:
            return 0.0

        if win_rate <= 0 or win_rate >= 1:
            return 0.0

        # R = Win/Loss ratio
        r = avg_win_pips / avg_loss_pips

        # Kelly formula: W - [(1-W) / R]
        kelly = win_rate - ((1 - win_rate) / r)

        # Can be negative (don't trade this strategy)
        if kelly <= 0:
            return 0.0

        # Apply fraction (half Kelly is common)
        return kelly * self.max_fraction * 100  # Return as percentage

    def calculate_from_trades(
        self,
        winning_trades: list,
        losing_trades: list
    ) -> float:
        """
        Calculate Kelly from trade history.

        Args:
            winning_trades: List of winning pip amounts
            losing_trades: List of losing pip amounts (positive values)

        Returns:
            Optimal risk percentage
        """
        total_trades = len(winning_trades) + len(losing_trades)
        if total_trades == 0:
            return 0.0

        win_rate = len(winning_trades) / total_trades

        avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0

        return self.calculate(win_rate, avg_win, avg_loss)


class ForexATRPositionSizer:
    """
    ATR-based position sizing for Forex.

    Uses Average True Range to set dynamic stops and position sizes.

    Stop Distance = ATR * Multiplier
    Position Size = Risk / (ATR Stop * Pip Value)

    Example:
        atr_sizer = ForexATRPositionSizer(atr_multiplier=2.0)
        lots = atr_sizer.calculate(
            symbol="EUR/USD",
            atr_pips=15,
            account_equity=10000,
            risk_percent=1.0,
            current_price=1.1000
        )
    """

    def __init__(
        self,
        atr_multiplier: float = 2.0,
        pip_calculator: Optional[PipCalculator] = None
    ):
        """
        Initialize ATR position sizer.

        Args:
            atr_multiplier: Multiplier for ATR to get stop distance
            pip_calculator: PipCalculator instance
        """
        self.atr_multiplier = atr_multiplier
        self.pip_calc = pip_calculator or PipCalculator()

    def calculate(
        self,
        symbol: str,
        atr_pips: float,
        account_equity: float,
        risk_percent: float,
        current_price: float,
        fx_rates: Optional[Dict[str, float]] = None
    ) -> Tuple[float, float]:
        """
        Calculate position size using ATR.

        Args:
            symbol: Currency pair
            atr_pips: ATR in pips
            account_equity: Account equity
            risk_percent: Risk percentage
            current_price: Current price
            fx_rates: FX rates for cross pairs

        Returns:
            Tuple of (lots, stop_pips)

        Example:
            lots, stop = atr_sizer.calculate(
                "EUR/USD", 15, 10000, 1.0, 1.1000
            )
            # With ATR 15, multiplier 2 -> stop = 30 pips
            # Risk $100, pip value $10 -> lots = $100/(30*$10) = 0.33
        """
        if atr_pips <= 0 or account_equity <= 0:
            return (0.0, 0.0)

        # Calculate stop distance
        stop_pips = atr_pips * self.atr_multiplier

        # Calculate risk amount
        risk_amount = account_equity * (risk_percent / 100)

        # Get pip value
        pip_value = self.pip_calc.get_pip_value(
            symbol, current_price, LotType.STANDARD.value, fx_rates
        )

        if pip_value <= 0:
            return (0.0, stop_pips)

        # Calculate lots
        lots = risk_amount / (stop_pips * pip_value)

        return (round(lots, 2), stop_pips)
