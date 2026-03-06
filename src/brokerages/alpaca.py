"""
Alpaca Brokerage Implementation (Lean-Inspired)
================================================
Live trading integration with Alpaca Securities.

Features:
- Commission-free US stocks & options
- Crypto trading
- Paper trading mode
- Real-time WebSocket updates

Setup:
    export ALPACA_API_KEY="your-key"
    export ALPACA_SECRET_KEY="your-secret"
    export ALPACA_PAPER="true"  # For paper trading

Install:
    pip install alpaca-py
"""

import os
from typing import List, Optional
from datetime import datetime
import logging
import threading

from .base import BaseBrokerage, CashBalance, Position
from .orders import Order, OrderTicket, OrderType, OrderSide, OrderStatus, TimeInForce
from .events import OrderEvent, OrderEventType

logger = logging.getLogger(__name__)

# Check for Alpaca SDK
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        StopOrderRequest,
        StopLimitOrderRequest,
        TrailingStopOrderRequest,
        GetOrdersRequest
    )
    from alpaca.trading.enums import (
        OrderSide as AlpacaSide,
        TimeInForce as AlpacaTIF,
        OrderStatus as AlpacaStatus,
    )
    from alpaca.trading.stream import TradingStream
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("Alpaca SDK not installed. Run: pip install alpaca-py")


class AlpacaBrokerage(BaseBrokerage):
    """
    Alpaca brokerage implementation (Lean-inspired).

    Supports:
    - US Equities (commission-free)
    - Options
    - Crypto (BTC, ETH, etc.)
    - Paper trading mode

    Example:
        # Set environment variables first
        os.environ["ALPACA_API_KEY"] = "your-key"
        os.environ["ALPACA_SECRET_KEY"] = "your-secret"

        brokerage = AlpacaBrokerage(paper=True)
        brokerage.connect()

        # Place order
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10
        )
        ticket = brokerage.place_order(order)

        # Check positions
        positions = brokerage.get_positions()
        print(f"Positions: {positions}")

        brokerage.disconnect()
    """

    def __init__(self, paper: bool = True):
        """
        Initialize Alpaca brokerage.

        Args:
            paper: Use paper trading (default True for safety)
        """
        super().__init__("Alpaca")

        if not ALPACA_AVAILABLE:
            raise ImportError(
                "Alpaca SDK not installed. Run: pip install alpaca-py"
            )

        self.paper = paper
        self._client: Optional[TradingClient] = None
        self._stream: Optional[TradingStream] = None
        self._stream_thread: Optional[threading.Thread] = None

        # Load credentials
        self._api_key = os.getenv("ALPACA_API_KEY")
        self._secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not self._api_key or not self._secret_key:
            raise ValueError(
                "Alpaca credentials required. Set environment variables:\n"
                "  ALPACA_API_KEY=your-api-key\n"
                "  ALPACA_SECRET_KEY=your-secret-key"
            )

    def connect(self) -> bool:
        """Connect to Alpaca."""
        try:
            self._client = TradingClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
                paper=self.paper
            )

            # Verify connection
            account = self._client.get_account()
            self._is_connected = True

            mode = "PAPER" if self.paper else "LIVE"
            self._emit_message(
                f"Connected to Alpaca ({mode}) - "
                f"Account: ${float(account.equity):,.2f}"
            )

            # Start WebSocket stream for order updates
            self._start_stream()

            return True

        except Exception as e:
            logger.error(f"Alpaca connection failed: {e}")
            self._is_connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from Alpaca."""
        self._stop_stream()
        self._client = None
        self._is_connected = False
        self._emit_message("Disconnected from Alpaca")

    def _start_stream(self):
        """Start WebSocket stream for real-time updates."""
        try:
            self._stream = TradingStream(
                api_key=self._api_key,
                secret_key=self._secret_key,
                paper=self.paper
            )

            @self._stream.on_trade_update
            async def on_trade_update(update):
                self._handle_trade_update(update)

            def run_stream():
                try:
                    self._stream.run()
                except Exception as e:
                    logger.error(f"Stream error: {e}")

            self._stream_thread = threading.Thread(target=run_stream, daemon=True)
            self._stream_thread.start()

        except Exception as e:
            logger.warning(f"Could not start stream: {e}")

    def _stop_stream(self):
        """Stop WebSocket stream."""
        if self._stream:
            try:
                self._stream.stop()
            except Exception:
                pass

    def _handle_trade_update(self, update):
        """Handle trade update from WebSocket."""
        try:
            order_id = str(update.order.client_order_id or update.order.id)
            order = self._orders.get(order_id)

            if not order:
                return

            event_type = None
            if update.event == "fill":
                order.status = OrderStatus.FILLED
                order.filled_quantity = float(update.order.filled_qty)
                order.average_fill_price = float(update.order.filled_avg_price)
                event_type = OrderEventType.FILLED

            elif update.event == "partial_fill":
                order.status = OrderStatus.PARTIALLY_FILLED
                order.filled_quantity = float(update.order.filled_qty)
                order.average_fill_price = float(update.order.filled_avg_price)
                event_type = OrderEventType.PARTIALLY_FILLED

            elif update.event == "canceled":
                order.status = OrderStatus.CANCELED
                event_type = OrderEventType.CANCELED

            elif update.event == "rejected":
                order.status = OrderStatus.REJECTED
                event_type = OrderEventType.REJECTED

            if event_type:
                self._emit_order_event(OrderEvent(
                    order_id=order.id,
                    event_type=event_type,
                    order=order,
                    fill_price=order.average_fill_price,
                    fill_quantity=order.filled_quantity
                ))

        except Exception as e:
            logger.error(f"Error handling trade update: {e}")

    # ========== Orders ==========

    def place_order(self, order: Order) -> OrderTicket:
        """Submit order to Alpaca."""
        if not self._is_connected:
            raise ConnectionError("Not connected to Alpaca")

        try:
            # Convert to Alpaca order request
            alpaca_request = self._to_alpaca_order(order)

            # Submit
            alpaca_order = self._client.submit_order(alpaca_request)

            # Update our order with broker ID
            order.broker_id = str(alpaca_order.id)
            order.status = OrderStatus.SUBMITTED
            order.submitted_time = datetime.utcnow()

            # Track
            self._orders[order.id] = order

            logger.info(f"Order submitted: {order.id} -> {order.broker_id}")

            self._emit_order_event(OrderEvent(
                order_id=order.id,
                event_type=OrderEventType.SUBMITTED,
                order=order
            ))

            return OrderTicket(order=order, _brokerage=self)

        except Exception as e:
            order.status = OrderStatus.REJECTED
            logger.error(f"Order rejected: {e}")

            self._emit_order_event(OrderEvent(
                order_id=order.id,
                event_type=OrderEventType.REJECTED,
                order=order,
                message=str(e)
            ))

            return OrderTicket(order=order, _brokerage=self)

    def _to_alpaca_order(self, order: Order):
        """Convert Order to Alpaca request."""
        side = AlpacaSide.BUY if order.side == OrderSide.BUY else AlpacaSide.SELL

        # Time in force
        tif_map = {
            TimeInForce.DAY: AlpacaTIF.DAY,
            TimeInForce.GTC: AlpacaTIF.GTC,
            TimeInForce.IOC: AlpacaTIF.IOC,
            TimeInForce.FOK: AlpacaTIF.FOK,
        }
        tif = tif_map.get(order.time_in_force, AlpacaTIF.GTC)

        # Build request based on order type
        if order.order_type == OrderType.MARKET:
            return MarketOrderRequest(
                symbol=order.symbol,
                qty=order.quantity,
                side=side,
                time_in_force=tif,
                client_order_id=order.id
            )

        elif order.order_type == OrderType.LIMIT:
            return LimitOrderRequest(
                symbol=order.symbol,
                qty=order.quantity,
                side=side,
                time_in_force=tif,
                limit_price=order.limit_price,
                client_order_id=order.id
            )

        elif order.order_type == OrderType.STOP_MARKET:
            return StopOrderRequest(
                symbol=order.symbol,
                qty=order.quantity,
                side=side,
                time_in_force=tif,
                stop_price=order.stop_price,
                client_order_id=order.id
            )

        elif order.order_type == OrderType.STOP_LIMIT:
            return StopLimitOrderRequest(
                symbol=order.symbol,
                qty=order.quantity,
                side=side,
                time_in_force=tif,
                stop_price=order.stop_price,
                limit_price=order.limit_price,
                client_order_id=order.id
            )

        elif order.order_type == OrderType.TRAILING_STOP:
            return TrailingStopOrderRequest(
                symbol=order.symbol,
                qty=order.quantity,
                side=side,
                time_in_force=tif,
                trail_percent=order.trailing_amount if order.trailing_as_percent else None,
                trail_price=order.trailing_amount if not order.trailing_as_percent else None,
                client_order_id=order.id
            )

        raise ValueError(f"Unsupported order type: {order.order_type}")

    def update_order(
        self,
        order: Order,
        quantity: Optional[float] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> bool:
        """
        Update order on Alpaca.

        Note: Alpaca doesn't support direct order modification.
        This cancels and replaces the order.
        """
        if not order.broker_id:
            return False

        # Cancel existing
        if not self.cancel_order(order):
            return False

        # Create new order with updated params
        new_order = Order(
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=quantity or order.quantity,
            limit_price=limit_price or order.limit_price,
            stop_price=stop_price or order.stop_price,
            time_in_force=order.time_in_force
        )

        ticket = self.place_order(new_order)
        return ticket.status != OrderStatus.REJECTED

    def cancel_order(self, order: Order) -> bool:
        """Cancel order on Alpaca."""
        if not order.broker_id:
            return False

        try:
            self._client.cancel_order_by_id(order.broker_id)
            order.status = OrderStatus.CANCELED

            self._emit_order_event(OrderEvent(
                order_id=order.id,
                event_type=OrderEventType.CANCELED,
                order=order
            ))

            return True

        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False

    # ========== Account ==========

    def get_cash_balance(self) -> List[CashBalance]:
        """Get Alpaca cash balance."""
        if not self._client:
            return []

        account = self._client.get_account()

        return [CashBalance(
            currency="USD",
            amount=float(account.cash),
            available=float(account.buying_power)
        )]

    def get_positions(self) -> List[Position]:
        """Get Alpaca positions."""
        if not self._client:
            return []

        alpaca_positions = self._client.get_all_positions()

        return [Position(
            symbol=p.symbol,
            quantity=float(p.qty),
            average_price=float(p.avg_entry_price),
            market_value=float(p.market_value),
            unrealized_pnl=float(p.unrealized_pl),
            unrealized_pnl_percent=float(p.unrealized_plpc) * 100
        ) for p in alpaca_positions]

    def get_account_value(self) -> float:
        """Get total account value."""
        if not self._client:
            return 0.0

        account = self._client.get_account()
        return float(account.equity)

    # ========== Additional Methods ==========

    def get_account_info(self) -> dict:
        """Get detailed account information."""
        if not self._client:
            return {}

        account = self._client.get_account()

        return {
            'id': account.id,
            'status': account.status,
            'currency': account.currency,
            'cash': float(account.cash),
            'portfolio_value': float(account.portfolio_value),
            'equity': float(account.equity),
            'buying_power': float(account.buying_power),
            'daytrading_buying_power': float(account.daytrading_buying_power),
            'pattern_day_trader': account.pattern_day_trader,
            'trading_blocked': account.trading_blocked,
            'transfers_blocked': account.transfers_blocked,
            'account_blocked': account.account_blocked,
        }

    def get_open_orders_from_broker(self) -> List[Order]:
        """Get open orders directly from Alpaca."""
        if not self._client:
            return []

        request = GetOrdersRequest(status="open")
        alpaca_orders = self._client.get_orders(request)

        orders = []
        for ao in alpaca_orders:
            order = Order(
                id=ao.client_order_id or str(ao.id),
                broker_id=str(ao.id),
                symbol=ao.symbol,
                side=OrderSide.BUY if ao.side == AlpacaSide.BUY else OrderSide.SELL,
                quantity=float(ao.qty),
                filled_quantity=float(ao.filled_qty) if ao.filled_qty else 0,
                average_fill_price=float(ao.filled_avg_price) if ao.filled_avg_price else 0,
                status=self._convert_status(ao.status)
            )

            if ao.limit_price:
                order.limit_price = float(ao.limit_price)
                order.order_type = OrderType.LIMIT

            if ao.stop_price:
                order.stop_price = float(ao.stop_price)
                if order.order_type == OrderType.LIMIT:
                    order.order_type = OrderType.STOP_LIMIT
                else:
                    order.order_type = OrderType.STOP_MARKET

            orders.append(order)

        return orders

    def _convert_status(self, alpaca_status) -> OrderStatus:
        """Convert Alpaca status to our OrderStatus."""
        status_map = {
            AlpacaStatus.NEW: OrderStatus.NEW,
            AlpacaStatus.ACCEPTED: OrderStatus.SUBMITTED,
            AlpacaStatus.PENDING_NEW: OrderStatus.SUBMITTED,
            AlpacaStatus.PARTIALLY_FILLED: OrderStatus.PARTIALLY_FILLED,
            AlpacaStatus.FILLED: OrderStatus.FILLED,
            AlpacaStatus.CANCELED: OrderStatus.CANCELED,
            AlpacaStatus.REJECTED: OrderStatus.REJECTED,
            AlpacaStatus.EXPIRED: OrderStatus.EXPIRED,
        }
        return status_map.get(alpaca_status, OrderStatus.NEW)
