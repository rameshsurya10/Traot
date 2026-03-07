"""
Portfolio Manager (Lean-Inspired)
=================================
Central hub for portfolio state and operations.

Features:
- Holdings management (positions + cash)
- Multi-currency support
- Buying power calculation
- Portfolio targets and rebalancing
- Performance tracking
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class InsightDirection(Enum):
    """Signal direction (Lean-compatible)."""
    UP = 1
    FLAT = 0
    DOWN = -1


@dataclass
class Holding:
    """
    A single position in the portfolio (Lean-inspired).

    Tracks quantity, cost basis, and real-time P&L.
    """
    symbol: str
    quantity: float = 0.0
    average_price: float = 0.0
    market_price: float = 0.0

    # Tracking
    _total_cost: float = field(default=0.0, repr=False)
    _realized_pnl: float = field(default=0.0, repr=False)

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0

    @property
    def invested(self) -> bool:
        """Check if any position exists."""
        return abs(self.quantity) > 1e-8

    @property
    def holdings_value(self) -> float:
        """Current market value of holdings."""
        return self.quantity * self.market_price

    @property
    def holdings_cost(self) -> float:
        """Total cost basis."""
        return self._total_cost

    @property
    def unrealized_profit(self) -> float:
        """Unrealized P&L."""
        if not self.invested:
            return 0.0
        return self.holdings_value - self._total_cost

    @property
    def unrealized_profit_percent(self) -> float:
        """Unrealized P&L as percentage."""
        if abs(self._total_cost) < 1e-8:
            return 0.0
        return (self.unrealized_profit / abs(self._total_cost)) * 100

    @property
    def realized_profit(self) -> float:
        """Realized P&L from closed positions."""
        return self._realized_pnl

    @property
    def total_profit(self) -> float:
        """Total P&L (realized + unrealized)."""
        return self.realized_profit + self.unrealized_profit

    def add_to_position(self, quantity: float, price: float) -> float:
        """
        Add to existing position.

        Returns commission-adjusted cost of trade.
        """
        trade_cost = quantity * price
        old_quantity = self.quantity

        # Update position
        if abs(self.quantity + quantity) < 1e-8:
            # Position fully closed
            self._realized_pnl += self.holdings_value - self._total_cost
            self.quantity = 0.0
            self._total_cost = 0.0
            self.average_price = 0.0
        elif (self.quantity >= 0 and quantity > 0) or (self.quantity <= 0 and quantity < 0):
            # Increasing position
            self._total_cost += trade_cost
            self.quantity += quantity
            self.average_price = self._total_cost / self.quantity if self.quantity != 0 else 0
        else:
            # Reducing position
            if abs(quantity) > abs(self.quantity):
                # Reversing position
                realized = (abs(self.quantity) / abs(old_quantity)) * self._total_cost
                self._realized_pnl += self.quantity * self.market_price - realized
                remaining_qty = quantity + self.quantity
                self.quantity = remaining_qty
                self._total_cost = abs(remaining_qty) * price
                self.average_price = price
            else:
                # Partial close
                close_ratio = abs(quantity) / abs(self.quantity)
                close_cost = close_ratio * self._total_cost
                self._realized_pnl += abs(quantity) * self.market_price - close_cost
                self.quantity += quantity
                self._total_cost -= close_cost
                # average_price stays the same

        return trade_cost

    def update_market_price(self, price: float):
        """Update current market price."""
        self.market_price = price

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'average_price': self.average_price,
            'market_price': self.market_price,
            'holdings_value': self.holdings_value,
            'unrealized_pnl': self.unrealized_profit,
            'unrealized_pnl_pct': self.unrealized_profit_percent,
            'realized_pnl': self.realized_profit,
        }


@dataclass
class PortfolioTarget:
    """
    Target position for a symbol (Lean-compatible).

    Used for rebalancing and order generation.
    """
    symbol: str
    quantity: float
    direction: InsightDirection = InsightDirection.UP
    tag: str = ""

    @classmethod
    def percent(cls, symbol: str, percent: float, portfolio_value: float,
                price: float, direction: InsightDirection = InsightDirection.UP) -> 'PortfolioTarget':
        """Create target from portfolio percentage."""
        target_value = portfolio_value * percent
        quantity = target_value / price if price > 0 else 0
        return cls(symbol=symbol, quantity=quantity, direction=direction)


class PortfolioManager:
    """
    Central Portfolio Manager (Lean-Inspired).

    Manages:
    - Cash positions (multi-currency)
    - Holdings (positions)
    - Buying power
    - Portfolio-level risk constraints
    - Performance tracking

    Example:
        portfolio = PortfolioManager(initial_cash=100000)

        # Process a fill
        portfolio.process_fill("AAPL", 100, 150.0, "BUY")

        # Update prices
        portfolio.update_price("AAPL", 155.0)

        # Check status
        print(f"Equity: ${portfolio.total_value:,.2f}")
        print(f"P&L: ${portfolio.total_unrealized_pnl:,.2f}")

        # Calculate position size
        size = portfolio.calculate_position_size(
            symbol="AAPL",
            entry_price=155.0,
            stop_price=150.0,
            risk_percent=0.02
        )
    """

    def __init__(
        self,
        initial_cash: float = 100000.0,
        base_currency: str = "USD",
        margin_ratio: float = 1.0,  # 1.0 = no margin, 2.0 = 2x margin
        max_position_percent: float = 0.25,  # Max 25% per position
        max_total_positions: int = 10,
    ):
        """
        Initialize portfolio manager.

        Args:
            initial_cash: Starting cash balance
            base_currency: Account base currency
            margin_ratio: Margin multiplier (1.0 = cash only)
            max_position_percent: Maximum portfolio % per position
            max_total_positions: Maximum concurrent positions
        """
        # Cash management (multi-currency support)
        self._cash: Dict[str, float] = {base_currency: initial_cash}
        self.base_currency = base_currency

        # Holdings
        self._holdings: Dict[str, Holding] = {}

        # Constraints
        self.margin_ratio = margin_ratio
        self.max_position_percent = max_position_percent
        self.max_total_positions = max_total_positions

        # Tracking
        self._initial_value = initial_cash
        self._high_water_mark = initial_cash
        self._trades_count = 0
        self._winning_trades = 0
        self._total_commission = 0.0

        # Thread safety
        self._lock = threading.RLock()

        # Callbacks
        self._on_position_changed: List[Callable] = []

        logger.info(f"PortfolioManager initialized with ${initial_cash:,.2f}")

    @property
    def initial_cash(self) -> float:
        return self._initial_value

    # =========================================================================
    # CASH MANAGEMENT
    # =========================================================================

    @property
    def cash(self) -> float:
        """Get cash in base currency."""
        with self._lock:
            return self._cash.get(self.base_currency, 0.0)

    def get_cash(self, currency: str = None) -> float:
        """Get cash in specific currency."""
        currency = currency or self.base_currency
        with self._lock:
            return self._cash.get(currency, 0.0)

    def set_cash(self, amount: float, currency: str = None):
        """Set cash amount."""
        currency = currency or self.base_currency
        with self._lock:
            self._cash[currency] = amount

    def add_cash(self, amount: float, currency: str = None):
        """Add cash (can be negative)."""
        currency = currency or self.base_currency
        with self._lock:
            self._cash[currency] = self._cash.get(currency, 0.0) + amount

    @property
    def buying_power(self) -> float:
        """Calculate available buying power."""
        with self._lock:
            # Cash * margin ratio - reserved for existing positions margin
            return self.cash * self.margin_ratio

    # =========================================================================
    # HOLDINGS MANAGEMENT
    # =========================================================================

    def get_holding(self, symbol: str) -> Holding:
        """Get or create holding for symbol."""
        with self._lock:
            if symbol not in self._holdings:
                self._holdings[symbol] = Holding(symbol=symbol)
            return self._holdings[symbol]

    @property
    def holdings(self) -> Dict[str, Holding]:
        """Get all holdings."""
        with self._lock:
            return dict(self._holdings)

    @property
    def invested_symbols(self) -> List[str]:
        """Get symbols with active positions."""
        with self._lock:
            return [s for s, h in self._holdings.items() if h.invested]

    @property
    def position_count(self) -> int:
        """Count of open positions."""
        return len(self.invested_symbols)

    def has_position(self, symbol: str) -> bool:
        """Check if symbol has an open position."""
        with self._lock:
            holding = self._holdings.get(symbol)
            return holding.invested if holding else False

    # =========================================================================
    # ORDER PROCESSING
    # =========================================================================

    def process_fill(
        self,
        symbol: str,
        quantity: float,
        fill_price: float,
        side: str,  # "BUY" or "SELL"
        commission: float = 0.0
    ):
        """
        Process an order fill.

        Updates holdings and cash based on the fill.

        Args:
            symbol: Ticker symbol
            quantity: Fill quantity (always positive)
            fill_price: Execution price
            side: "BUY" or "SELL"
            commission: Commission charged
        """
        with self._lock:
            # Adjust sign based on side
            signed_qty = quantity if side.upper() == "BUY" else -quantity

            # Get/create holding
            holding = self.get_holding(symbol)
            old_qty = holding.quantity

            # Update position
            trade_cost = holding.add_to_position(signed_qty, fill_price)
            holding.update_market_price(fill_price)

            # Update cash
            self._cash[self.base_currency] -= trade_cost + commission
            self._total_commission += commission

            # Track trades
            self._trades_count += 1

            # Update high water mark
            current_value = self.total_value
            if current_value > self._high_water_mark:
                self._high_water_mark = current_value

            logger.info(
                f"Fill: {side} {quantity} {symbol} @ ${fill_price:.2f} "
                f"(Position: {old_qty:.2f} -> {holding.quantity:.2f})"
            )

            # Notify callbacks
            self._notify_position_changed(symbol, holding)

    def _notify_position_changed(self, symbol: str, holding: Holding):
        """Notify position change callbacks."""
        for callback in self._on_position_changed:
            try:
                callback(symbol, holding)
            except Exception as e:
                logger.error(f"Position callback error: {e}")

    def on_position_changed(self, callback: Callable):
        """Register position change callback."""
        self._on_position_changed.append(callback)

    # =========================================================================
    # PRICE UPDATES
    # =========================================================================

    def update_price(self, symbol: str, price: float):
        """Update market price for a symbol."""
        with self._lock:
            if symbol in self._holdings:
                self._holdings[symbol].update_market_price(price)

    def update_prices(self, prices: Dict[str, float]):
        """Batch update prices."""
        with self._lock:
            for symbol, price in prices.items():
                if symbol in self._holdings:
                    self._holdings[symbol].update_market_price(price)

    # =========================================================================
    # PORTFOLIO VALUE
    # =========================================================================

    @property
    def total_holdings_value(self) -> float:
        """Total value of all positions."""
        with self._lock:
            return sum(h.holdings_value for h in self._holdings.values())

    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + holdings)."""
        with self._lock:
            return self.cash + self.total_holdings_value

    @property
    def total_unrealized_pnl(self) -> float:
        """Total unrealized P&L."""
        with self._lock:
            return sum(h.unrealized_profit for h in self._holdings.values())

    @property
    def total_realized_pnl(self) -> float:
        """Total realized P&L."""
        with self._lock:
            return sum(h.realized_profit for h in self._holdings.values())

    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)."""
        return self.total_realized_pnl + self.total_unrealized_pnl

    @property
    def total_return(self) -> float:
        """Total return percentage."""
        if self._initial_value == 0:
            return 0.0
        return ((self.total_value - self._initial_value) / self._initial_value) * 100

    @property
    def drawdown(self) -> float:
        """Current drawdown from high water mark."""
        if self._high_water_mark == 0:
            return 0.0
        return ((self._high_water_mark - self.total_value) / self._high_water_mark) * 100

    # =========================================================================
    # POSITION SIZING
    # =========================================================================

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float,
        risk_percent: float = 0.02,
        max_percent: float = None,
    ) -> float:
        """
        Calculate position size based on risk.

        Uses the risk-per-trade model: risk a fixed percentage of portfolio.

        Args:
            symbol: Ticker symbol
            entry_price: Expected entry price
            stop_price: Stop loss price
            risk_percent: Portfolio risk per trade (default 2%)
            max_percent: Max position size as portfolio % (default from config)

        Returns:
            Recommended position size (quantity)
        """
        max_percent = max_percent or self.max_position_percent

        # Risk-based sizing
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share == 0:
            logger.warning(f"Zero risk per share for {symbol}")
            return 0.0

        portfolio_risk = self.total_value * risk_percent
        risk_based_size = portfolio_risk / risk_per_share

        # Max position constraint
        max_position_value = self.total_value * max_percent
        max_based_size = max_position_value / entry_price if entry_price > 0 else 0

        # Buying power constraint
        bp_based_size = self.buying_power / entry_price if entry_price > 0 else 0

        # Take minimum of all constraints
        position_size = min(risk_based_size, max_based_size, bp_based_size)

        # Round to reasonable precision
        # Crypto pairs (symbol contains '/') use 6 decimal precision
        # Stocks use whole shares
        if '/' in symbol:
            position_size = round(position_size, 6)
        elif entry_price > 1:
            position_size = int(position_size)
        else:
            position_size = round(position_size, 4)

        logger.debug(
            f"Position size for {symbol}: {position_size} "
            f"(risk-based: {risk_based_size:.2f}, max: {max_based_size:.2f}, bp: {bp_based_size:.2f})"
        )

        return position_size

    def can_open_position(self, symbol: str) -> tuple:
        """
        Check if a new position can be opened.

        Returns:
            (can_open: bool, reason: str)
        """
        with self._lock:
            # Check position count
            if self.position_count >= self.max_total_positions:
                if not self.has_position(symbol):
                    return False, f"Max positions reached ({self.max_total_positions})"

            # Check buying power
            if self.buying_power <= 0:
                return False, "No buying power available"

            return True, "OK"

    # =========================================================================
    # TARGETS AND REBALANCING
    # =========================================================================

    def set_holdings(self, symbol: str, percent: float, current_price: float) -> Optional[PortfolioTarget]:
        """
        Set target holdings as percentage of portfolio (Lean-style).

        Args:
            symbol: Ticker symbol
            percent: Target portfolio percentage (0.10 = 10%)
            current_price: Current market price

        Returns:
            PortfolioTarget if order needed, None if already at target
        """
        target_value = self.total_value * percent
        target_quantity = target_value / current_price if current_price > 0 else 0

        holding = self.get_holding(symbol)
        current_quantity = holding.quantity

        # Calculate difference
        diff = target_quantity - current_quantity

        if abs(diff * current_price) < self.total_value * 0.001:  # < 0.1% change
            return None  # Already at target

        direction = InsightDirection.UP if diff > 0 else InsightDirection.DOWN

        return PortfolioTarget(
            symbol=symbol,
            quantity=abs(diff),
            direction=direction,
            tag=f"SetHoldings({percent:.1%})"
        )

    def liquidate(self, symbol: str) -> Optional[PortfolioTarget]:
        """
        Create target to close a position.

        Returns:
            PortfolioTarget if position exists, None otherwise
        """
        holding = self.get_holding(symbol)
        if not holding.invested:
            return None

        direction = InsightDirection.DOWN if holding.is_long else InsightDirection.UP

        return PortfolioTarget(
            symbol=symbol,
            quantity=abs(holding.quantity),
            direction=direction,
            tag="Liquidate"
        )

    # =========================================================================
    # REPORTING
    # =========================================================================

    def get_summary(self) -> dict:
        """Get portfolio summary."""
        with self._lock:
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'total_value': self.total_value,
                'cash': self.cash,
                'holdings_value': self.total_holdings_value,
                'buying_power': self.buying_power,
                'position_count': self.position_count,
                'unrealized_pnl': self.total_unrealized_pnl,
                'realized_pnl': self.total_realized_pnl,
                'total_return_pct': self.total_return,
                'drawdown_pct': self.drawdown,
                'trades_count': self._trades_count,
                'total_commission': self._total_commission,
            }

    def get_holdings_report(self) -> List[dict]:
        """Get detailed holdings report."""
        with self._lock:
            return [h.to_dict() for h in self._holdings.values() if h.invested]

    def __repr__(self) -> str:
        return (
            f"PortfolioManager(value=${self.total_value:,.2f}, "
            f"cash=${self.cash:,.2f}, positions={self.position_count})"
        )
