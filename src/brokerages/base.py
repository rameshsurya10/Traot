"""
Base Brokerage Interface (Lean-Inspired)
========================================
Abstract brokerage class that all implementations inherit from.
Provides unified interface regardless of broker.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
import logging

from .orders import Order, OrderTicket
from .events import OrderEvent

logger = logging.getLogger(__name__)


@dataclass
class CashBalance:
    """Cash balance in a currency."""
    currency: str
    amount: float
    available: float  # After margin requirements

    @property
    def locked(self) -> float:
        """Get locked/reserved amount."""
        return self.amount - self.available


@dataclass
class Position:
    """Current holding in a security."""
    symbol: str
    quantity: float
    average_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    side: str = "long"  # "long" or "short"

    @property
    def cost_basis(self) -> float:
        """Get total cost basis."""
        return self.quantity * self.average_price

    @property
    def is_long(self) -> bool:
        """Check if long position."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if short position."""
        return self.quantity < 0


class BaseBrokerage(ABC):
    """
    Abstract brokerage interface (Lean-inspired).

    All broker implementations inherit from this.
    Provides unified interface regardless of broker.

    Implementations:
    - PaperBrokerage: Simulated trading
    - AlpacaBrokerage: Alpaca Securities
    - BinanceBrokerage: Binance exchange

    Example:
        brokerage = AlpacaBrokerage(paper=True)
        brokerage.connect()

        # Place order
        order = Order(symbol="AAPL", side=OrderSide.BUY, ...)
        ticket = brokerage.place_order(order)

        # Check positions
        positions = brokerage.get_positions()
    """

    def __init__(self, name: str):
        self.name = name
        self._is_connected = False

        # Event callbacks
        self._on_order_event: List[Callable[[OrderEvent], None]] = []
        self._on_message: List[Callable[[str], None]] = []

        # Order tracking
        self._orders: Dict[str, Order] = {}

    # ========== Connection ==========

    @property
    def is_connected(self) -> bool:
        """Check if connected to broker."""
        return self._is_connected

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to broker.

        Returns:
            True if connected successfully
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from broker."""
        pass

    # ========== Orders ==========

    @abstractmethod
    def place_order(self, order: Order) -> OrderTicket:
        """
        Submit order to broker.

        Args:
            order: Order to submit

        Returns:
            OrderTicket for tracking/modifying
        """
        pass

    @abstractmethod
    def update_order(
        self,
        order: Order,
        quantity: Optional[float] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> bool:
        """
        Modify an existing order.

        Args:
            order: Order to modify
            quantity: New quantity (optional)
            limit_price: New limit price (optional)
            stop_price: New stop price (optional)

        Returns:
            True if update accepted
        """
        pass

    @abstractmethod
    def cancel_order(self, order: Order) -> bool:
        """
        Cancel an open order.

        Args:
            order: Order to cancel

        Returns:
            True if cancellation accepted
        """
        pass

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get all open orders.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of open orders
        """
        orders = [o for o in self._orders.values() if o.is_open]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order or None if not found
        """
        return self._orders.get(order_id)

    # ========== Account ==========

    @abstractmethod
    def get_cash_balance(self) -> List[CashBalance]:
        """
        Get cash balances in all currencies.

        Returns:
            List of cash balances
        """
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """
        Get all current positions.

        Returns:
            List of positions
        """
        pass

    @abstractmethod
    def get_account_value(self) -> float:
        """
        Get total account value (cash + positions).

        Returns:
            Total account value
        """
        pass

    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol.

        Args:
            symbol: Symbol to look up

        Returns:
            Position or None
        """
        for pos in self.get_positions():
            if pos.symbol == symbol:
                return pos
        return None

    def get_buying_power(self) -> float:
        """
        Get available buying power.

        Returns:
            Available buying power
        """
        balances = self.get_cash_balance()
        return sum(b.available for b in balances)

    # ========== Events ==========

    def on_order_event(self, callback: Callable[[OrderEvent], None]) -> None:
        """
        Register callback for order events.

        Args:
            callback: Function to call on order events
        """
        self._on_order_event.append(callback)

    def on_message(self, callback: Callable[[str], None]) -> None:
        """
        Register callback for broker messages.

        Args:
            callback: Function to call on messages
        """
        self._on_message.append(callback)

    def _emit_order_event(self, event: OrderEvent) -> None:
        """Emit order event to all listeners."""
        for callback in self._on_order_event:
            try:
                callback(event)
            except Exception as e:
                self._emit_message(f"Order event callback error: {e}")

    def _emit_message(self, message: str) -> None:
        """Emit message to all listeners."""
        logger.info(f"[{self.name}] {message}")
        for callback in self._on_message:
            try:
                callback(message)
            except Exception:
                pass

    # ========== Convenience Methods ==========

    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all open orders.

        Args:
            symbol: Cancel only for this symbol (optional)

        Returns:
            Number of orders canceled
        """
        canceled = 0
        for order in self.get_open_orders(symbol):
            if self.cancel_order(order):
                canceled += 1
        return canceled

    def get_order_history(self, limit: int = 100) -> List[Order]:
        """
        Get recent order history.

        Args:
            limit: Maximum orders to return

        Returns:
            List of orders (newest first)
        """
        orders = sorted(
            self._orders.values(),
            key=lambda o: o.created_time,
            reverse=True
        )
        return orders[:limit]
