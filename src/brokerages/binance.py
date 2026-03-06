"""
Binance Brokerage Implementation (Lean-Inspired)
=================================================
Live trading integration with Binance exchange.

Features:
- Spot and futures crypto trading
- Real-time WebSocket updates
- Testnet (paper) mode

Setup:
    export BINANCE_API_KEY="your-key"
    export BINANCE_SECRET_KEY="your-secret"
    export BINANCE_TESTNET="true"  # For testnet

Install:
    pip install python-binance
"""

import os
from typing import List, Optional, Dict
from datetime import datetime
import logging

from .base import BaseBrokerage, CashBalance, Position
from .orders import Order, OrderTicket, OrderType, OrderSide, OrderStatus, TimeInForce
from .events import OrderEvent, OrderEventType

logger = logging.getLogger(__name__)

# Check for Binance SDK
try:
    from binance.client import Client
    from binance.enums import (
        SIDE_BUY, SIDE_SELL,
        ORDER_TYPE_MARKET, ORDER_TYPE_LIMIT,
        ORDER_TYPE_STOP_LOSS, ORDER_TYPE_STOP_LOSS_LIMIT,
        TIME_IN_FORCE_GTC, TIME_IN_FORCE_IOC, TIME_IN_FORCE_FOK
    )
    from binance import ThreadedWebsocketManager
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    logger.warning("Binance SDK not installed. Run: pip install python-binance")


class BinanceBrokerage(BaseBrokerage):
    """
    Binance brokerage implementation (Lean-inspired).

    Supports:
    - Spot trading (BTC, ETH, etc.)
    - All major order types
    - Testnet for paper trading
    - Real-time WebSocket updates

    Example:
        # Set environment variables first
        os.environ["BINANCE_API_KEY"] = "your-key"
        os.environ["BINANCE_SECRET_KEY"] = "your-secret"

        brokerage = BinanceBrokerage(testnet=True)
        brokerage.connect()

        # Place order
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001
        )
        ticket = brokerage.place_order(order)

        # Check positions
        positions = brokerage.get_positions()

        brokerage.disconnect()
    """

    def __init__(self, testnet: bool = True):
        """
        Initialize Binance brokerage.

        Args:
            testnet: Use testnet (default True for safety)
        """
        super().__init__("Binance")

        if not BINANCE_AVAILABLE:
            raise ImportError(
                "Binance SDK not installed. Run: pip install python-binance"
            )

        self.testnet = testnet
        self._client: Optional[Client] = None
        self._ws_manager: Optional[ThreadedWebsocketManager] = None

        # Load credentials
        self._api_key = os.getenv("BINANCE_API_KEY")
        self._secret_key = os.getenv("BINANCE_SECRET_KEY")

        if not self._api_key or not self._secret_key:
            raise ValueError(
                "Binance credentials required. Set environment variables:\n"
                "  BINANCE_API_KEY=your-api-key\n"
                "  BINANCE_SECRET_KEY=your-secret-key"
            )

        # Symbol info cache
        self._symbol_info: Dict[str, dict] = {}

    def connect(self) -> bool:
        """Connect to Binance."""
        try:
            self._client = Client(
                api_key=self._api_key,
                api_secret=self._secret_key,
                testnet=self.testnet
            )

            # Verify connection (call get_account to validate API keys)
            self._client.get_account()
            self._is_connected = True

            # Cache symbol info for precision
            self._load_symbol_info()

            mode = "TESTNET" if self.testnet else "LIVE"
            self._emit_message(f"Connected to Binance ({mode})")

            # Start WebSocket for order updates
            self._start_websocket()

            return True

        except Exception as e:
            logger.error(f"Binance connection failed: {e}")
            self._is_connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from Binance."""
        self._stop_websocket()
        self._client = None
        self._is_connected = False
        self._emit_message("Disconnected from Binance")

    def _load_symbol_info(self):
        """Load symbol precision info."""
        try:
            info = self._client.get_exchange_info()
            for s in info['symbols']:
                self._symbol_info[s['symbol']] = {
                    'base': s['baseAsset'],
                    'quote': s['quoteAsset'],
                    'price_precision': s['quotePrecision'],
                    'qty_precision': s['baseAssetPrecision'],
                    'min_qty': None,
                    'min_notional': None,
                    'step_size': None
                }
                for f in s['filters']:
                    if f['filterType'] == 'LOT_SIZE':
                        self._symbol_info[s['symbol']]['min_qty'] = float(f['minQty'])
                        self._symbol_info[s['symbol']]['step_size'] = float(f['stepSize'])
                    elif f['filterType'] == 'NOTIONAL':
                        self._symbol_info[s['symbol']]['min_notional'] = float(f.get('minNotional', 0))
        except Exception as e:
            logger.warning(f"Could not load symbol info: {e}")

    def _start_websocket(self):
        """Start WebSocket for user data stream."""
        try:
            self._ws_manager = ThreadedWebsocketManager(
                api_key=self._api_key,
                api_secret=self._secret_key,
                testnet=self.testnet
            )
            self._ws_manager.start()
            self._ws_manager.start_user_socket(callback=self._handle_user_update)
        except Exception as e:
            logger.warning(f"Could not start WebSocket: {e}")

    def _stop_websocket(self):
        """Stop WebSocket."""
        if self._ws_manager:
            try:
                self._ws_manager.stop()
            except Exception:
                pass

    def _handle_user_update(self, msg):
        """Handle user data stream updates."""
        try:
            if msg.get('e') == 'executionReport':
                order_id = msg.get('c')  # Client order ID
                order = self._orders.get(order_id)

                if not order:
                    return

                exec_type = msg.get('x')
                status = msg.get('X')

                if exec_type == 'TRADE':
                    filled_qty = float(msg.get('l', 0))
                    fill_price = float(msg.get('L', 0))

                    order.filled_quantity += filled_qty
                    # Update average price
                    if order.average_fill_price == 0:
                        order.average_fill_price = fill_price
                    else:
                        total_qty = order.filled_quantity
                        prev_qty = total_qty - filled_qty
                        order.average_fill_price = (
                            (order.average_fill_price * prev_qty + fill_price * filled_qty)
                            / total_qty
                        )

                    if status == 'FILLED':
                        order.status = OrderStatus.FILLED
                        self._emit_order_event(OrderEvent(
                            order_id=order.id,
                            event_type=OrderEventType.FILLED,
                            order=order,
                            fill_price=fill_price,
                            fill_quantity=filled_qty
                        ))
                    else:
                        order.status = OrderStatus.PARTIALLY_FILLED
                        self._emit_order_event(OrderEvent(
                            order_id=order.id,
                            event_type=OrderEventType.PARTIALLY_FILLED,
                            order=order,
                            fill_price=fill_price,
                            fill_quantity=filled_qty
                        ))

                elif exec_type == 'CANCELED':
                    order.status = OrderStatus.CANCELED
                    self._emit_order_event(OrderEvent(
                        order_id=order.id,
                        event_type=OrderEventType.CANCELED,
                        order=order
                    ))

                elif exec_type == 'REJECTED':
                    order.status = OrderStatus.REJECTED
                    self._emit_order_event(OrderEvent(
                        order_id=order.id,
                        event_type=OrderEventType.REJECTED,
                        order=order,
                        message=msg.get('r', 'Unknown rejection reason')
                    ))

        except Exception as e:
            logger.error(f"Error handling user update: {e}")

    # ========== Orders ==========

    def place_order(self, order: Order) -> OrderTicket:
        """Submit order to Binance."""
        if not self._is_connected:
            raise ConnectionError("Not connected to Binance")

        try:
            # Format symbol (remove dash if present)
            symbol = order.symbol.replace("-", "").replace("/", "")

            # Convert to Binance order
            side = SIDE_BUY if order.side == OrderSide.BUY else SIDE_SELL

            # Round quantity to valid precision
            qty = self._round_quantity(symbol, order.quantity)

            params = {
                "symbol": symbol,
                "side": side,
                "quantity": qty,
                "newClientOrderId": order.id
            }

            if order.order_type == OrderType.MARKET:
                params["type"] = ORDER_TYPE_MARKET
                result = self._client.create_order(**params)

            elif order.order_type == OrderType.LIMIT:
                params["type"] = ORDER_TYPE_LIMIT
                params["price"] = self._round_price(symbol, order.limit_price)
                params["timeInForce"] = self._convert_tif(order.time_in_force)
                result = self._client.create_order(**params)

            elif order.order_type == OrderType.STOP_MARKET:
                params["type"] = ORDER_TYPE_STOP_LOSS
                params["stopPrice"] = self._round_price(symbol, order.stop_price)
                result = self._client.create_order(**params)

            elif order.order_type == OrderType.STOP_LIMIT:
                params["type"] = ORDER_TYPE_STOP_LOSS_LIMIT
                params["price"] = self._round_price(symbol, order.limit_price)
                params["stopPrice"] = self._round_price(symbol, order.stop_price)
                params["timeInForce"] = self._convert_tif(order.time_in_force)
                result = self._client.create_order(**params)

            else:
                raise ValueError(f"Unsupported order type: {order.order_type}")

            # Update order
            order.broker_id = str(result["orderId"])
            order.status = OrderStatus.SUBMITTED
            order.submitted_time = datetime.utcnow()

            # Check if already filled (market orders)
            if result.get("status") == "FILLED":
                order.status = OrderStatus.FILLED
                order.filled_quantity = float(result.get("executedQty", 0))
                fills = result.get("fills", [])
                if fills:
                    total_cost = sum(float(f["price"]) * float(f["qty"]) for f in fills)
                    total_qty = sum(float(f["qty"]) for f in fills)
                    order.average_fill_price = total_cost / total_qty if total_qty > 0 else 0
                    order.commission = sum(float(f.get("commission", 0)) for f in fills)

            self._orders[order.id] = order

            logger.info(f"Order submitted: {order.id} -> {order.broker_id}")

            event_type = OrderEventType.FILLED if order.status == OrderStatus.FILLED else OrderEventType.SUBMITTED
            self._emit_order_event(OrderEvent(
                order_id=order.id,
                event_type=event_type,
                order=order,
                fill_price=order.average_fill_price,
                fill_quantity=order.filled_quantity
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

    def _round_quantity(self, symbol: str, quantity: float) -> str:
        """Round quantity to valid precision."""
        info = self._symbol_info.get(symbol, {})
        precision = info.get('qty_precision', 8)
        step = info.get('step_size')

        if step:
            # Round to step size
            quantity = round(quantity / step) * step

        return f"{quantity:.{precision}f}".rstrip('0').rstrip('.')

    def _round_price(self, symbol: str, price: float) -> str:
        """Round price to valid precision."""
        info = self._symbol_info.get(symbol, {})
        precision = info.get('price_precision', 8)
        return f"{price:.{precision}f}".rstrip('0').rstrip('.')

    def _convert_tif(self, tif: TimeInForce) -> str:
        """Convert TimeInForce to Binance format."""
        mapping = {
            TimeInForce.GTC: TIME_IN_FORCE_GTC,
            TimeInForce.IOC: TIME_IN_FORCE_IOC,
            TimeInForce.FOK: TIME_IN_FORCE_FOK,
            TimeInForce.DAY: TIME_IN_FORCE_GTC,  # Binance doesn't have DAY
        }
        return mapping.get(tif, TIME_IN_FORCE_GTC)

    def update_order(
        self,
        order: Order,
        quantity: Optional[float] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> bool:
        """Binance doesn't support order modification."""
        logger.warning("Binance does not support order modification. Cancel and resubmit.")
        return False

    def cancel_order(self, order: Order) -> bool:
        """Cancel order on Binance."""
        if not order.broker_id:
            return False

        try:
            symbol = order.symbol.replace("-", "").replace("/", "")
            self._client.cancel_order(
                symbol=symbol,
                orderId=int(order.broker_id)
            )
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
        """Get Binance balances."""
        if not self._client:
            return []

        account = self._client.get_account()
        balances = []

        for b in account["balances"]:
            free = float(b["free"])
            locked = float(b["locked"])
            if free > 0 or locked > 0:
                balances.append(CashBalance(
                    currency=b["asset"],
                    amount=free + locked,
                    available=free
                ))

        return balances

    def get_positions(self) -> List[Position]:
        """
        Get Binance positions.

        For spot trading, positions are non-stablecoin balances.
        """
        if not self._client:
            return []

        stables = {"USDT", "BUSD", "USDC", "USD", "DAI", "TUSD"}
        balances = self.get_cash_balance()

        positions = []
        for b in balances:
            if b.currency not in stables and b.amount > 0:
                # Get current price
                try:
                    ticker = self._client.get_symbol_ticker(symbol=f"{b.currency}USDT")
                    price = float(ticker["price"])
                    market_value = b.amount * price
                except Exception:
                    price = 0
                    market_value = 0

                positions.append(Position(
                    symbol=b.currency,
                    quantity=b.amount,
                    average_price=0,  # Not tracked by Binance
                    market_value=market_value,
                    unrealized_pnl=0,  # Can't calculate without entry price
                    unrealized_pnl_percent=0
                ))

        return positions

    def get_account_value(self) -> float:
        """Get total account value in USDT."""
        if not self._client:
            return 0.0

        balances = self.get_cash_balance()
        total = 0.0

        stables = {"USDT", "BUSD", "USDC", "DAI", "TUSD"}

        for b in balances:
            if b.currency in stables:
                total += b.amount
            elif b.amount > 0:
                try:
                    ticker = self._client.get_symbol_ticker(symbol=f"{b.currency}USDT")
                    price = float(ticker["price"])
                    total += b.amount * price
                except Exception:
                    pass  # Skip if no USDT pair

        return total

    # ========== Additional Methods ==========

    def get_ticker(self, symbol: str) -> dict:
        """Get current ticker for symbol."""
        if not self._client:
            return {}

        symbol = symbol.replace("-", "").replace("/", "")
        ticker = self._client.get_symbol_ticker(symbol=symbol)

        return {
            'symbol': symbol,
            'price': float(ticker['price'])
        }

    def get_orderbook(self, symbol: str, limit: int = 10) -> dict:
        """Get order book for symbol."""
        if not self._client:
            return {}

        symbol = symbol.replace("-", "").replace("/", "")
        depth = self._client.get_order_book(symbol=symbol, limit=limit)

        return {
            'bids': [[float(p), float(q)] for p, q in depth['bids']],
            'asks': [[float(p), float(q)] for p, q in depth['asks']]
        }

    def get_recent_trades(self, symbol: str, limit: int = 50) -> List[dict]:
        """Get recent trades for symbol."""
        if not self._client:
            return []

        symbol = symbol.replace("-", "").replace("/", "")
        trades = self._client.get_recent_trades(symbol=symbol, limit=limit)

        return [{
            'price': float(t['price']),
            'quantity': float(t['qty']),
            'time': datetime.fromtimestamp(t['time'] / 1000),
            'is_buyer_maker': t['isBuyerMaker']
        } for t in trades]
