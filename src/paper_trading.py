"""
Paper Trading Simulator (Lean-Inspired)
=======================================
Full trading simulator implementing BaseBrokerage interface.
Practice trading without risking real money.

Enhanced with:
- All order types (Market, Limit, Stop, Trailing)
- Realistic slippage simulation
- Event-driven order management
- Compatible with live brokerage interface
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import threading

from .brokerages.base import BaseBrokerage, CashBalance, Position
from .brokerages.orders import (
    Order, OrderTicket, OrderType, OrderSide, OrderStatus
)
from .brokerages.events import OrderEvent, OrderEventType

logger = logging.getLogger(__name__)


@dataclass
class PendingOrder:
    """Internal tracking for pending orders with SL/TP."""
    order: Order
    stop_loss_order: Optional[Order] = None
    take_profit_order: Optional[Order] = None


@dataclass
class Trade:
    """Completed trade record."""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    realized_pnl: float
    commission: float
    timestamp: datetime

    @property
    def price(self) -> float:
        """Get exit price (alias for dashboard compatibility)."""
        return self.exit_price


class PaperBrokerage(BaseBrokerage):
    """
    Paper trading brokerage (Lean-inspired).

    Implements full BaseBrokerage interface for seamless
    switching between paper and live trading.

    Features:
    - All order types (Market, Limit, Stop, Trailing)
    - Realistic slippage simulation
    - Commission modeling
    - Position tracking with P&L
    - Automatic SL/TP handling
    - Event callbacks

    Example:
        brokerage = PaperBrokerage(initial_cash=10000)
        brokerage.connect()

        # Register for events
        brokerage.on_order_event(lambda e: print(f"Order: {e}"))

        # Place order
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
            stop_loss=45000,
            take_profit=55000
        )
        ticket = brokerage.place_order(order)

        # Simulate price update
        brokerage.update_price("BTCUSDT", 50500)
    """

    def __init__(
        self,
        initial_cash: float = 10000,
        commission_percent: float = 0.1,
        slippage_percent: float = 0.05,
        base_currency: str = "USD"
    ):
        """
        Initialize paper trading brokerage.

        Args:
            initial_cash: Starting virtual capital
            commission_percent: Commission per trade (0.1 = 0.1%)
            slippage_percent: Slippage simulation (0.05 = 0.05%)
            base_currency: Base currency for account
        """
        super().__init__("Paper")

        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.base_currency = base_currency
        self.commission_pct = commission_percent / 100
        self.slippage_pct = slippage_percent / 100

        # Position tracking
        self._positions: Dict[str, _InternalPosition] = {}

        # Pending orders (limit, stop, etc.)
        self._pending_orders: List[Order] = []

        # Current prices
        self._prices: Dict[str, float] = {}

        # Trade history
        self._trade_history: List[Dict] = []

        # Threading
        self._lock = threading.Lock()
        self._order_counter = 0

    # ========== Connection ==========

    def connect(self) -> bool:
        """Connect (always succeeds for paper trading)."""
        self._is_connected = True
        self._emit_message(f"Paper trading connected with ${self.initial_cash:,.2f}")
        return True

    def disconnect(self) -> None:
        """Disconnect from paper trading."""
        self._is_connected = False
        self._emit_message("Paper trading disconnected")

    # ========== Orders ==========

    def place_order(self, order: Order) -> OrderTicket:
        """
        Place a trading order.

        Supports:
        - MARKET: Execute immediately at current price
        - LIMIT: Execute when price reaches limit
        - STOP_MARKET: Execute market order when stop price hit
        - STOP_LIMIT: Execute limit order when stop price hit
        - TRAILING_STOP: Trailing stop order

        Args:
            order: Order to place

        Returns:
            OrderTicket for tracking
        """
        with self._lock:
            # Assign internal ID
            self._order_counter += 1
            if not order.id:
                order.id = f"PAPER_{self._order_counter:06d}"

            order.submitted_time = datetime.utcnow()
            self._orders[order.id] = order

            # Handle by order type
            if order.order_type == OrderType.MARKET:
                self._execute_market_order(order)
            else:
                # Queue for later execution
                order.status = OrderStatus.SUBMITTED
                self._pending_orders.append(order)
                self._emit_order_event(OrderEvent(
                    order_id=order.id,
                    event_type=OrderEventType.SUBMITTED,
                    order=order
                ))

            # Create SL/TP orders if specified
            if order.status == OrderStatus.FILLED:
                self._create_sl_tp_orders(order)

            return OrderTicket(order=order, _brokerage=self)

    def _execute_market_order(self, order: Order) -> bool:
        """Execute market order at current price."""
        symbol = order.symbol
        current_price = self._prices.get(symbol, 0)

        if current_price <= 0:
            order.status = OrderStatus.REJECTED
            self._emit_order_event(OrderEvent(
                order_id=order.id,
                event_type=OrderEventType.REJECTED,
                order=order,
                message=f"No price data for {symbol}"
            ))
            return False

        # Apply slippage
        if order.side == OrderSide.BUY:
            fill_price = current_price * (1 + self.slippage_pct)
        else:
            fill_price = current_price * (1 - self.slippage_pct)

        return self._fill_order(order, fill_price)

    def _fill_order(self, order: Order, fill_price: float) -> bool:
        """Execute order fill and update positions."""
        trade_value = order.quantity * fill_price
        commission = trade_value * self.commission_pct

        if order.side == OrderSide.BUY:
            total_cost = trade_value + commission

            if total_cost > self.cash:
                order.status = OrderStatus.REJECTED
                self._emit_order_event(OrderEvent(
                    order_id=order.id,
                    event_type=OrderEventType.REJECTED,
                    order=order,
                    message=f"Insufficient cash: need ${total_cost:,.2f}, have ${self.cash:,.2f}"
                ))
                return False

            self.cash -= total_cost
            self._update_position_buy(order.symbol, order.quantity, fill_price)

        else:  # SELL
            pos = self._positions.get(order.symbol)
            if not pos or pos.quantity < order.quantity:
                order.status = OrderStatus.REJECTED
                self._emit_order_event(OrderEvent(
                    order_id=order.id,
                    event_type=OrderEventType.REJECTED,
                    order=order,
                    message="Insufficient position to sell"
                ))
                return False

            proceeds = trade_value - commission
            self.cash += proceeds

            # Record trade
            pnl = (fill_price - pos.avg_price) * order.quantity - commission
            self._trade_history.append({
                'symbol': order.symbol,
                'side': 'sell',
                'quantity': order.quantity,
                'entry_price': pos.avg_price,
                'exit_price': fill_price,
                'pnl': pnl,
                'commission': commission,
                'time': datetime.utcnow()
            })

            self._update_position_sell(order.symbol, order.quantity, fill_price)

        # Update order
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_fill_price = fill_price
        order.commission = commission
        order.filled_time = datetime.utcnow()

        self._emit_order_event(OrderEvent(
            order_id=order.id,
            event_type=OrderEventType.FILLED,
            order=order,
            fill_price=fill_price,
            fill_quantity=order.quantity,
            commission=commission
        ))

        logger.info(
            f"Order filled: {order.side.value} {order.quantity} {order.symbol} "
            f"@ ${fill_price:,.2f} (commission: ${commission:.2f})"
        )

        return True

    def _update_position_buy(self, symbol: str, quantity: float, price: float):
        """Update position after buy."""
        if symbol in self._positions:
            pos = self._positions[symbol]
            total_qty = pos.quantity + quantity
            pos.avg_price = (pos.avg_price * pos.quantity + price * quantity) / total_qty
            pos.quantity = total_qty
        else:
            self._positions[symbol] = _InternalPosition(
                symbol=symbol,
                quantity=quantity,
                avg_price=price,
                entry_time=datetime.utcnow()
            )

    def _update_position_sell(self, symbol: str, quantity: float, price: float):
        """Update position after sell."""
        pos = self._positions.get(symbol)
        if pos:
            pos.quantity -= quantity
            if pos.quantity <= 0:
                del self._positions[symbol]

    def _create_sl_tp_orders(self, order: Order):
        """Create stop loss and take profit orders."""
        if order.stop_loss:
            sl_order = Order(
                symbol=order.symbol,
                side=OrderSide.SELL if order.side == OrderSide.BUY else OrderSide.BUY,
                order_type=OrderType.STOP_MARKET,
                quantity=order.quantity,
                stop_price=order.stop_loss,
                tag=f"SL_{order.id}"
            )
            sl_order.status = OrderStatus.SUBMITTED
            self._orders[sl_order.id] = sl_order
            self._pending_orders.append(sl_order)

        if order.take_profit:
            tp_order = Order(
                symbol=order.symbol,
                side=OrderSide.SELL if order.side == OrderSide.BUY else OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=order.quantity,
                limit_price=order.take_profit,
                tag=f"TP_{order.id}"
            )
            tp_order.status = OrderStatus.SUBMITTED
            self._orders[tp_order.id] = tp_order
            self._pending_orders.append(tp_order)

    def update_order(
        self,
        order: Order,
        quantity: Optional[float] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> bool:
        """Update a pending order."""
        with self._lock:
            if not order.is_open:
                return False

            if quantity is not None:
                order.quantity = quantity
            if limit_price is not None:
                order.limit_price = limit_price
            if stop_price is not None:
                order.stop_price = stop_price

            self._emit_order_event(OrderEvent(
                order_id=order.id,
                event_type=OrderEventType.UPDATED,
                order=order
            ))

            return True

    def cancel_order(self, order: Order) -> bool:
        """Cancel a pending order."""
        with self._lock:
            if not order.is_open:
                return False

            order.status = OrderStatus.CANCELED

            if order in self._pending_orders:
                self._pending_orders.remove(order)

            self._emit_order_event(OrderEvent(
                order_id=order.id,
                event_type=OrderEventType.CANCELED,
                order=order
            ))

            return True

    # ========== Learning System Integration ==========

    def execute_signal(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        signal_id: str = None
    ) -> bool:
        """
        Execute a trading signal from the continuous learning system.

        This is the bridge between the learning system and paper trading.
        It creates and executes a market order based on the prediction signal.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            signal: Direction ("UP" or "DOWN")
            confidence: Prediction confidence (0.0 to 1.0)
            signal_id: Signal ID for tracking

        Returns:
            True if order was executed
        """
        current_price = self._prices.get(symbol, 0)
        if current_price <= 0:
            logger.warning(f"[Paper] Cannot execute signal for {symbol}: no price data")
            return False

        # Determine order side
        if signal == 'UP':
            side = OrderSide.BUY
        elif signal == 'DOWN':
            # Only sell if we have a position
            pos = self._positions.get(symbol)
            if not pos or pos.quantity <= 0:
                logger.debug(f"[Paper] Skipping DOWN signal for {symbol}: no position to sell")
                return False
            side = OrderSide.SELL
        else:
            return False

        # Calculate position size (risk 2% of portfolio per trade)
        risk_pct = 0.02
        portfolio_value = self.cash + sum(
            p.quantity * self._prices.get(s, 0)
            for s, p in self._positions.items()
        )

        if side == OrderSide.BUY:
            trade_value = portfolio_value * risk_pct * confidence
            quantity = trade_value / current_price
            # Crypto precision
            if '/' in symbol:
                quantity = round(quantity, 6)
            else:
                quantity = max(1, int(quantity))
        else:
            # Sell entire position
            quantity = self._positions[symbol].quantity

        if quantity <= 0:
            return False

        # Create and place order
        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity
        )

        if signal_id:
            order.tag = signal_id

        self.place_order(order)

        if order.status == OrderStatus.FILLED:
            logger.info(
                f"[Paper] Signal executed: {side.name} {quantity:.6f} {symbol} "
                f"@ {current_price:.2f} (conf: {confidence:.1%}, id: {signal_id})"
            )
            return True

        return False

    # ========== Price Updates ==========

    def update_price(self, symbol: str, price: float) -> None:
        """
        Update current price for a symbol.

        This triggers pending order checks and position P&L updates.

        Args:
            symbol: Symbol to update
            price: New price
        """
        with self._lock:
            self._prices[symbol] = price
            self._check_pending_orders(symbol, price)
            self._update_position_pnl(symbol, price)

    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Update multiple prices at once.

        Args:
            prices: Dict of {symbol: price}
        """
        for symbol, price in prices.items():
            self.update_price(symbol, price)

    def _check_pending_orders(self, symbol: str, price: float):
        """Check if any pending orders should be triggered."""
        orders_to_remove = []

        for order in self._pending_orders:
            if order.symbol != symbol:
                continue

            triggered = False

            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and price <= order.limit_price:
                    triggered = True
                elif order.side == OrderSide.SELL and price >= order.limit_price:
                    triggered = True

            elif order.order_type == OrderType.STOP_MARKET:
                if order.side == OrderSide.BUY and price >= order.stop_price:
                    triggered = True
                elif order.side == OrderSide.SELL and price <= order.stop_price:
                    triggered = True

            elif order.order_type == OrderType.STOP_LIMIT:
                if order.side == OrderSide.BUY and price >= order.stop_price:
                    # Convert to limit order
                    order.order_type = OrderType.LIMIT
                elif order.side == OrderSide.SELL and price <= order.stop_price:
                    order.order_type = OrderType.LIMIT

            elif order.order_type == OrderType.TRAILING_STOP:
                self._update_trailing_stop(order, price)
                if order.side == OrderSide.SELL and price <= order.stop_price:
                    triggered = True
                elif order.side == OrderSide.BUY and price >= order.stop_price:
                    triggered = True

            if triggered:
                fill_price = price
                if order.order_type == OrderType.LIMIT:
                    fill_price = order.limit_price

                if self._fill_order(order, fill_price):
                    orders_to_remove.append(order)
                    # Cancel related SL/TP orders
                    self._cancel_related_orders(order)

        for order in orders_to_remove:
            if order in self._pending_orders:
                self._pending_orders.remove(order)

    def _update_trailing_stop(self, order: Order, price: float):
        """Update trailing stop price."""
        if order.trailing_amount is None:
            return

        if order.trailing_as_percent:
            trail = price * (order.trailing_amount / 100)
        else:
            trail = order.trailing_amount

        if order.side == OrderSide.SELL:
            new_stop = price - trail
            if order.stop_price is None or new_stop > order.stop_price:
                order.stop_price = new_stop
        else:
            new_stop = price + trail
            if order.stop_price is None or new_stop < order.stop_price:
                order.stop_price = new_stop

    def _cancel_related_orders(self, filled_order: Order):
        """Cancel SL/TP orders when position is closed."""
        tag_prefix = filled_order.tag.split('_')[0] if filled_order.tag else None
        if not tag_prefix:
            return

        parent_id = filled_order.tag.split('_')[1] if '_' in filled_order.tag else None
        if not parent_id:
            return

        for order in self._pending_orders[:]:
            if order.tag and parent_id in order.tag and order.id != filled_order.id:
                self.cancel_order(order)

    def _update_position_pnl(self, symbol: str, price: float):
        """Update unrealized P&L for position."""
        pos = self._positions.get(symbol)
        if pos:
            pos.current_price = price
            pos.unrealized_pnl = (price - pos.avg_price) * pos.quantity

    # ========== Account ==========

    def get_cash_balance(self) -> List[CashBalance]:
        """Get cash balance."""
        with self._lock:
            return [CashBalance(
                currency=self.base_currency,
                amount=self.cash,
                available=self.cash
            )]

    def get_positions(self) -> List[Position]:
        """Get all current positions."""
        with self._lock:
            positions = []
            for symbol, pos in self._positions.items():
                market_value = pos.quantity * pos.current_price
                unrealized_pnl = (pos.current_price - pos.avg_price) * pos.quantity
                unrealized_pct = (unrealized_pnl / (pos.avg_price * pos.quantity)) * 100 if pos.avg_price > 0 else 0

                positions.append(Position(
                    symbol=symbol,
                    quantity=pos.quantity,
                    average_price=pos.avg_price,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_percent=unrealized_pct
                ))
            return positions

    def get_account_value(self) -> float:
        """Get total account value (cash + positions)."""
        with self._lock:
            positions_value = sum(
                pos.quantity * pos.current_price
                for pos in self._positions.values()
            )
            return self.cash + positions_value

    # ========== Additional Methods ==========

    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """Get trade history as dictionaries."""
        with self._lock:
            return self._trade_history[-limit:]

    def get_trades(self, limit: int = 100) -> List[Trade]:
        """Get trade history as Trade objects."""
        with self._lock:
            trades = []
            for t in self._trade_history[-limit:]:
                trades.append(Trade(
                    symbol=t.get('symbol', ''),
                    side=t.get('side', ''),
                    quantity=t.get('quantity', 0),
                    entry_price=t.get('entry_price', 0),
                    exit_price=t.get('exit_price', 0),
                    realized_pnl=t.get('pnl', 0),
                    commission=t.get('commission', 0),
                    timestamp=t.get('time', datetime.utcnow())
                ))
            return trades

    def get_portfolio_stats(self) -> Dict:
        """Get comprehensive portfolio statistics."""
        with self._lock:
            total_value = self.get_account_value()
            total_pnl = total_value - self.initial_cash
            total_return = (total_pnl / self.initial_cash) * 100 if self.initial_cash > 0 else 0

            realized_pnl = sum(t['pnl'] for t in self._trade_history)
            unrealized_pnl = sum(pos.unrealized_pnl for pos in self._positions.values())

            winning = sum(1 for t in self._trade_history if t['pnl'] > 0)
            total = len(self._trade_history)
            win_rate = (winning / total * 100) if total > 0 else 0

            return {
                'total_value': total_value,
                'cash': self.cash,
                'positions_value': total_value - self.cash,
                'total_pnl': total_pnl,
                'total_return_pct': total_return,
                'realized_pnl': realized_pnl,
                'unrealized_pnl': unrealized_pnl,
                'num_positions': len(self._positions),
                'total_trades': total,
                'win_rate': win_rate,
                'initial_capital': self.initial_cash
            }

    def reset(self) -> None:
        """Reset to initial state."""
        with self._lock:
            self.cash = self.initial_cash
            self._positions.clear()
            self._pending_orders.clear()
            self._orders.clear()
            self._trade_history.clear()
            self._prices.clear()
            self._order_counter = 0

            logger.info("Paper trading reset")

    def set_price(self, symbol: str, price: float) -> None:
        """
        Set initial price for a symbol (without triggering orders).

        Args:
            symbol: Symbol
            price: Initial price
        """
        with self._lock:
            self._prices[symbol] = price


@dataclass
class _InternalPosition:
    """Internal position tracking."""
    symbol: str
    quantity: float
    avg_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0


# Backward compatibility alias
PaperTradingSimulator = PaperBrokerage
