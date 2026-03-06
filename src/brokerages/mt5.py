"""
MetaTrader 5 Brokerage Implementation
======================================

Full brokerage integration for MT5 Forex trading.

Features:
- Order execution via mt5.order_send()
- Position tracking via mt5.positions_get()
- Account balance and equity queries
- Symbol normalization (EUR/USD <-> EURUSD)
- Thread-safe via shared MT5Worker

Requirements:
- MetaTrader 5 terminal (running on Windows or via bridge)
- MT5 account (demo or live)

Environment Variables:
    MT5_LOGIN: MT5 account number
    MT5_PASSWORD: MT5 account password
    MT5_SERVER: Broker server name
    MT5_TERMINAL_PATH: Path to MT5 terminal.exe (Windows only)

Usage:
    from src.brokerages.mt5 import MT5Brokerage

    brokerage = MT5Brokerage(demo=True)
    brokerage.connect()

    order = Order(
        symbol="EUR/USD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1  # lots
    )
    ticket = brokerage.place_order(order)
"""

import os
import sys
import logging
import threading
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

from .base import BaseBrokerage, CashBalance, Position
from .orders import Order, OrderTicket, OrderType, OrderSide, OrderStatus
from .events import OrderEvent, OrderEventType
from .utils.symbol_normalizer import get_symbol_normalizer

logger = logging.getLogger(__name__)

# MT5 trade action constants (used when module not yet imported)
TRADE_ACTION_DEAL = 1       # Market order
TRADE_ACTION_PENDING = 5    # Pending order
TRADE_ACTION_SLTP = 6       # Modify SL/TP
TRADE_ACTION_MODIFY = 7     # Modify pending order
TRADE_ACTION_REMOVE = 8     # Remove pending order

# MT5 order type constants
ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
ORDER_TYPE_BUY_LIMIT = 2
ORDER_TYPE_SELL_LIMIT = 3
ORDER_TYPE_BUY_STOP = 4
ORDER_TYPE_SELL_STOP = 5

# MT5 order filling modes
ORDER_FILLING_FOK = 0
ORDER_FILLING_IOC = 1
ORDER_FILLING_RETURN = 2

# MT5 order time types
ORDER_TIME_GTC = 0
ORDER_TIME_DAY = 1

# MT5 trade return codes
TRADE_RETCODE_DONE = 10009
TRADE_RETCODE_PLACED = 10008
TRADE_RETCODE_REQUOTE = 10004


class MT5Brokerage(BaseBrokerage):
    """
    MetaTrader 5 Brokerage Implementation.

    Implements BaseBrokerage interface for trade execution via MT5 terminal.
    Uses mt5.order_send() for execution, mt5.positions_get() for position tracking.

    Forex-specific:
    - Quantities are in lots (0.01 micro, 0.1 mini, 1.0 standard)
    - Magic number identifies orders from this bot
    - Auto-detects filling mode per symbol

    Example:
        brokerage = MT5Brokerage(demo=True)
        brokerage.connect()

        balance = brokerage.get_cash_balance()
        print(f"Balance: ${balance[0].amount:,.2f}")

        order = Order(
            symbol="EUR/USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.10  # 0.10 lots = 10,000 units
        )
        ticket = brokerage.place_order(order)
    """

    def __init__(
        self,
        demo: bool = True,
        terminal_path: Optional[str] = None,
        magic_number: int = 234000,
        deviation: int = 20,
    ):
        """
        Initialize MT5 brokerage.

        Args:
            demo: Use demo account
            terminal_path: Path to MT5 terminal (or MT5_TERMINAL_PATH env var)
            magic_number: Magic number to identify bot orders
            deviation: Max price deviation in points (slippage tolerance)
        """
        super().__init__("MetaTrader5")

        self._demo = demo
        self._terminal_path = terminal_path or os.environ.get('MT5_TERMINAL_PATH', '')
        self._login = int(os.environ.get('MT5_LOGIN', '0'))
        self._password = os.environ.get('MT5_PASSWORD', '')
        self._server = os.environ.get('MT5_SERVER', '')
        self._magic_number = magic_number
        self._deviation = deviation

        # MT5 module/client reference
        self._mt5 = None
        self._mt5_lock = threading.Lock()
        self._external_worker = None  # Shared MT5Worker from MT5DataProvider

        # Symbol normalizer
        self._normalizer = get_symbol_normalizer()

        # Filling mode cache per symbol
        self._filling_modes: Dict[str, int] = {}

        logger.info(
            f"MT5Brokerage initialized: demo={demo}, magic={magic_number}, "
            f"deviation={deviation}pts"
        )

    # ========== Connection ==========

    def connect(self) -> bool:
        """
        Initialize MT5 terminal and login.

        Returns:
            True if connected successfully
        """
        try:
            self._mt5 = self._get_mt5_module()

            # Initialize terminal
            init_kwargs = {}
            if self._terminal_path:
                init_kwargs['path'] = self._terminal_path

            if not self._mt5_call('initialize', **init_kwargs):
                error = self._mt5_call('last_error')
                logger.error(f"MT5 initialize failed: {error}")
                self._emit_message(f"Connection failed: {error}")
                return False

            # Login
            if self._login:
                if not self._mt5_call(
                    'login',
                    self._login,
                    password=self._password,
                    server=self._server,
                ):
                    error = self._mt5_call('last_error')
                    logger.error(f"MT5 login failed: {error}")
                    self._emit_message(f"Login failed: {error}")
                    return False

            # Verify connection
            account = self._mt5_call('account_info')
            if not account:
                logger.error("MT5 account info unavailable")
                return False

            self._is_connected = True
            env_name = "Demo" if self._demo else "Live"
            self._emit_message(
                f"Connected to MT5 {env_name}: "
                f"login={self._login}, balance={account.balance:.2f} {account.currency}"
            )
            logger.info(
                f"Connected to MT5 {env_name}: login={self._login}, "
                f"server={account.server}, balance={account.balance:.2f}"
            )
            return True

        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            self._emit_message(f"Connection failed: {e}")
            return False

    def disconnect(self) -> None:
        """Shutdown MT5 connection."""
        if self._mt5 and self._is_connected:
            try:
                self._mt5_call('shutdown')
            except Exception as e:
                logger.warning(f"MT5 shutdown error: {e}")

        self._is_connected = False
        self._emit_message("Disconnected from MT5")
        logger.info("Disconnected from MT5")

    def set_worker(self, worker) -> None:
        """
        Set a shared MT5Worker from MT5DataProvider.

        On Windows, MT5 Python API is not thread-safe. Sharing the worker
        ensures all MT5 calls (data + trading) go through a single thread.
        On Linux (bridge mode), this is less critical since the bridge server
        serializes calls, but still beneficial for consistency.

        Args:
            worker: MT5Worker instance from MT5DataProvider
        """
        self._external_worker = worker
        logger.info("MT5Brokerage: using shared MT5Worker for thread safety")

    def _mt5_call(self, func_name: str, *args, **kwargs):
        """
        Call an MT5 function, routing through shared worker if available.

        This ensures thread safety on Windows where both MT5DataProvider
        and MT5Brokerage need to call the MT5 module.
        """
        if self._external_worker:
            return self._external_worker.call(func_name, *args, **kwargs)
        else:
            with self._mt5_lock:
                func = getattr(self._mt5, func_name)
                return func(*args, **kwargs)

    def _get_mt5_module(self):
        """Get MT5 interface (direct or bridge)."""
        if sys.platform == 'win32':
            try:
                import MetaTrader5
                return MetaTrader5
            except ImportError:
                raise ImportError(
                    "MetaTrader5 package not installed. Run: pip install MetaTrader5"
                )
        else:
            from .mt5_bridge.client import MT5BridgeClient
            return MT5BridgeClient(
                host=os.environ.get('MT5_BRIDGE_HOST', 'localhost'),
                port=int(os.environ.get('MT5_BRIDGE_PORT', '5555')),
            )

    # ========== Orders ==========

    def place_order(self, order: Order) -> OrderTicket:
        """
        Submit order to MT5.

        Args:
            order: Order to submit (quantity in lots for forex)

        Returns:
            OrderTicket for tracking/modifying
        """
        if not self._is_connected:
            order.status = OrderStatus.REJECTED
            return OrderTicket(order=order, _brokerage=self)

        # Validate
        if order.quantity <= 0:
            order.status = OrderStatus.REJECTED
            self._emit_order_event(OrderEvent(
                order_id=order.id,
                event_type=OrderEventType.REJECTED,
                order=order,
                message=f"Invalid quantity: {order.quantity}",
            ))
            return OrderTicket(order=order, _brokerage=self)

        try:
            # Convert symbol: EUR/USD -> EURUSD
            mt5_symbol = self._normalizer.to_mt4(order.symbol)

            # Build MT5 request
            request = self._build_order_request(order, mt5_symbol)
            if request is None:
                order.status = OrderStatus.REJECTED
                return OrderTicket(order=order, _brokerage=self)

            # Send order
            result = self._mt5_call('order_send', request)

            if result is None:
                error = self._mt5_call('last_error')
                order.status = OrderStatus.REJECTED
                self._emit_order_event(OrderEvent(
                    order_id=order.id,
                    event_type=OrderEventType.REJECTED,
                    order=order,
                    message=f"MT5 error: {error}",
                ))
                logger.error(f"Order send returned None: {error}")
                return OrderTicket(order=order, _brokerage=self)

            # Handle result
            if result.retcode in (TRADE_RETCODE_DONE, TRADE_RETCODE_PLACED):
                order.broker_id = str(result.order)
                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                order.average_fill_price = result.price if result.price else 0
                order.filled_time = datetime.now(tz=timezone.utc)

                self._emit_order_event(OrderEvent(
                    order_id=order.id,
                    event_type=OrderEventType.FILLED,
                    order=order,
                    fill_price=order.average_fill_price,
                    fill_quantity=order.filled_quantity,
                ))
                logger.info(
                    f"Order filled: {order.side.value} {order.quantity} lots "
                    f"{order.symbol} @ {order.average_fill_price}"
                )
            else:
                order.status = OrderStatus.REJECTED
                reason = f"MT5 retcode {result.retcode}: {result.comment}"
                self._emit_order_event(OrderEvent(
                    order_id=order.id,
                    event_type=OrderEventType.REJECTED,
                    order=order,
                    message=reason,
                ))
                logger.warning(f"Order rejected: {reason}")

            # Track order
            self._orders[order.id] = order
            return OrderTicket(order=order, _brokerage=self)

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            order.status = OrderStatus.REJECTED
            self._emit_order_event(OrderEvent(
                order_id=order.id,
                event_type=OrderEventType.REJECTED,
                order=order,
                message=str(e),
            ))
            return OrderTicket(order=order, _brokerage=self)

    def _build_order_request(self, order: Order, mt5_symbol: str) -> Optional[dict]:
        """Build MT5 order request dict."""
        filling_mode = self._get_filling_mode(mt5_symbol)

        if order.order_type == OrderType.MARKET:
            # Get current price for market order
            tick = self._mt5_call('symbol_info_tick', mt5_symbol)
            if not tick:
                logger.error(f"Cannot get tick for {mt5_symbol}")
                return None

            if order.side == OrderSide.BUY:
                price = tick.ask
                order_type_mt5 = ORDER_TYPE_BUY
            else:
                price = tick.bid
                order_type_mt5 = ORDER_TYPE_SELL

            request = {
                "action": TRADE_ACTION_DEAL,
                "symbol": mt5_symbol,
                "volume": round(order.quantity, 2),
                "type": order_type_mt5,
                "price": price,
                "deviation": self._deviation,
                "magic": self._magic_number,
                "comment": "Traot",
                "type_time": ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }

        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY:
                order_type_mt5 = ORDER_TYPE_BUY_LIMIT
            else:
                order_type_mt5 = ORDER_TYPE_SELL_LIMIT

            request = {
                "action": TRADE_ACTION_PENDING,
                "symbol": mt5_symbol,
                "volume": round(order.quantity, 2),
                "type": order_type_mt5,
                "price": order.limit_price,
                "deviation": self._deviation,
                "magic": self._magic_number,
                "comment": "Traot",
                "type_time": ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }

        elif order.order_type == OrderType.STOP_MARKET:
            if order.side == OrderSide.BUY:
                order_type_mt5 = ORDER_TYPE_BUY_STOP
            else:
                order_type_mt5 = ORDER_TYPE_SELL_STOP

            request = {
                "action": TRADE_ACTION_PENDING,
                "symbol": mt5_symbol,
                "volume": round(order.quantity, 2),
                "type": order_type_mt5,
                "price": order.stop_price,
                "deviation": self._deviation,
                "magic": self._magic_number,
                "comment": "Traot",
                "type_time": ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }

        else:
            logger.error(f"Unsupported order type: {order.order_type}")
            return None

        # Add stop loss and take profit
        if order.stop_loss is not None:
            request["sl"] = order.stop_loss
        if order.take_profit is not None:
            request["tp"] = order.take_profit

        return request

    def _get_filling_mode(self, mt5_symbol: str) -> int:
        """
        Get the supported filling mode for a symbol.

        Different brokers support different filling modes.
        Auto-detect from symbol info.
        """
        if mt5_symbol in self._filling_modes:
            return self._filling_modes[mt5_symbol]

        try:
            info = self._mt5_call('symbol_info', mt5_symbol)
            if info and hasattr(info, 'filling_mode'):
                # filling_mode is a bitmask: bit 0=FOK, bit 1=IOC, bit 2=RETURN
                mode = info.filling_mode
                if mode & 1:
                    result = ORDER_FILLING_FOK
                elif mode & 2:
                    result = ORDER_FILLING_IOC
                else:
                    result = ORDER_FILLING_RETURN
                self._filling_modes[mt5_symbol] = result
                return result
        except Exception as e:
            logger.warning(f"Could not detect filling mode for {mt5_symbol}: {e}")

        # Default to IOC
        self._filling_modes[mt5_symbol] = ORDER_FILLING_IOC
        return ORDER_FILLING_IOC

    def update_order(
        self,
        order: Order,
        quantity: Optional[float] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> bool:
        """
        Modify an existing pending order.

        Args:
            order: Order to modify
            quantity: New volume in lots (optional)
            limit_price: New limit price (optional)
            stop_price: New stop price (optional)

        Returns:
            True if modification accepted
        """
        if not self._is_connected or not order.broker_id:
            return False

        try:
            mt5_symbol = self._normalizer.to_mt4(order.symbol)

            request = {
                "action": TRADE_ACTION_MODIFY,
                "order": int(order.broker_id),
                "symbol": mt5_symbol,
                "magic": self._magic_number,
            }

            if limit_price is not None and stop_price is not None:
                logger.error("Cannot set both limit_price and stop_price")
                return False
            if limit_price is not None:
                request["price"] = limit_price
            elif stop_price is not None:
                request["price"] = stop_price
            if quantity is not None:
                request["volume"] = round(quantity, 2)
            if order.stop_loss is not None:
                request["sl"] = order.stop_loss
            if order.take_profit is not None:
                request["tp"] = order.take_profit

            result = self._mt5_call('order_send', request)

            if result and result.retcode == TRADE_RETCODE_DONE:
                if limit_price is not None:
                    order.limit_price = limit_price
                if stop_price is not None:
                    order.stop_price = stop_price
                logger.info(f"Order modified: {order}")
                return True

            reason = result.comment if result else "Unknown"
            logger.warning(f"Order modification failed: {reason}")
            return False

        except Exception as e:
            logger.error(f"Failed to modify order: {e}")
            return False

    def cancel_order(self, order: Order) -> bool:
        """
        Cancel a pending order.

        Args:
            order: Order to cancel

        Returns:
            True if cancellation accepted
        """
        if not self._is_connected or not order.broker_id:
            return False

        try:
            request = {
                "action": TRADE_ACTION_REMOVE,
                "order": int(order.broker_id),
            }

            result = self._mt5_call('order_send', request)

            if result and result.retcode == TRADE_RETCODE_DONE:
                order.status = OrderStatus.CANCELED
                self._emit_order_event(OrderEvent(
                    order_id=order.id,
                    event_type=OrderEventType.CANCELED,
                    order=order,
                ))
                logger.info(f"Order canceled: {order}")
                return True

            reason = result.comment if result else "Unknown"
            logger.warning(f"Order cancellation failed: {reason}")
            return False

        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False

    # ========== Account ==========

    def get_cash_balance(self) -> List[CashBalance]:
        """
        Get account cash balance.

        Returns:
            List with single CashBalance entry
        """
        if not self._is_connected:
            return []

        try:
            info = self._mt5_call('account_info')
            if not info:
                return []

            return [CashBalance(
                currency=info.currency,
                amount=info.balance,
                available=info.margin_free,
            )]

        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return []

    def get_positions(self) -> List[Position]:
        """
        Get all open positions.

        Returns:
            List of Position objects
        """
        if not self._is_connected:
            return []

        try:
            mt5_positions = self._mt5_call('positions_get')

            if mt5_positions is None:
                return []

            positions = []
            for pos in mt5_positions:
                # Only include positions from this bot (by magic number)
                if hasattr(pos, 'magic') and pos.magic != self._magic_number:
                    continue

                # Convert MT5 symbol back to standard
                symbol = self._normalizer.to_standard(pos.symbol)
                is_long = pos.type == ORDER_TYPE_BUY

                # Get contract size for accurate notional calculation
                # Forex standard lot = 100,000 units
                contract_size = 100_000
                try:
                    sym_info = self._mt5_call('symbol_info', pos.symbol)
                    if sym_info and hasattr(sym_info, 'trade_contract_size'):
                        contract_size = sym_info.trade_contract_size
                except Exception:
                    pass

                unrealized = pos.profit
                notional = pos.volume * contract_size * pos.price_open
                unrealized_pct = (unrealized / notional * 100) if notional > 0 else 0
                market_value = pos.volume * contract_size * pos.price_current

                positions.append(Position(
                    symbol=symbol,
                    quantity=abs(pos.volume),
                    average_price=pos.price_open,
                    market_value=market_value,
                    unrealized_pnl=unrealized,
                    unrealized_pnl_percent=unrealized_pct,
                    side="long" if is_long else "short",
                ))

            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def get_account_value(self) -> float:
        """
        Get total account equity.

        Returns:
            Account equity (balance + unrealized P&L)
        """
        if not self._is_connected:
            return 0.0

        try:
            info = self._mt5_call('account_info')
            return info.equity if info else 0.0

        except Exception as e:
            logger.error(f"Failed to get account value: {e}")
            return 0.0

    # ========== Convenience Methods ==========

    def get_account_summary(self) -> Dict[str, Any]:
        """Get comprehensive account summary."""
        if not self._is_connected:
            return {}

        try:
            info = self._mt5_call('account_info')
            if not info:
                return {}

            return {
                "login": info.login,
                "server": info.server,
                "currency": info.currency,
                "balance": info.balance,
                "equity": info.equity,
                "margin": info.margin,
                "margin_free": info.margin_free,
                "margin_level": info.margin_level,
                "profit": info.profit,
                "leverage": info.leverage,
            }

        except Exception as e:
            logger.error(f"Failed to get account summary: {e}")
            return {}

    def close_position(self, symbol: str) -> bool:
        """
        Close all positions for a symbol opened by this bot.

        Args:
            symbol: Currency pair (standard format)

        Returns:
            True if all positions closed successfully
        """
        if not self._is_connected:
            return False

        try:
            mt5_symbol = self._normalizer.to_mt4(symbol)

            positions = self._mt5_call('positions_get', symbol=mt5_symbol)

            if not positions:
                logger.info(f"No positions found for {symbol}")
                return True

            all_closed = True
            for pos in positions:
                if hasattr(pos, 'magic') and pos.magic != self._magic_number:
                    continue

                # Close by opening opposite position
                close_type = ORDER_TYPE_SELL if pos.type == ORDER_TYPE_BUY else ORDER_TYPE_BUY

                tick = self._mt5_call('symbol_info_tick', mt5_symbol)
                if not tick:
                    all_closed = False
                    continue

                price = tick.bid if close_type == ORDER_TYPE_SELL else tick.ask

                request = {
                    "action": TRADE_ACTION_DEAL,
                    "symbol": mt5_symbol,
                    "volume": pos.volume,
                    "type": close_type,
                    "position": pos.ticket,
                    "price": price,
                    "deviation": self._deviation,
                    "magic": self._magic_number,
                    "comment": "Traot close",
                    "type_time": ORDER_TIME_GTC,
                    "type_filling": self._get_filling_mode(mt5_symbol),
                }

                result = self._mt5_call('order_send', request)

                if result and result.retcode == TRADE_RETCODE_DONE:
                    logger.info(f"Closed position: {symbol} {pos.volume} lots")
                else:
                    reason = result.comment if result else "Unknown"
                    logger.error(f"Failed to close position: {reason}")
                    all_closed = False

            return all_closed

        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return False

    def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get current bid/ask/mid price for a symbol.

        Args:
            symbol: Currency pair

        Returns:
            Dict with 'bid', 'ask', 'mid' or None
        """
        if not self._is_connected:
            return None

        try:
            mt5_symbol = self._normalizer.to_mt4(symbol)
            tick = self._mt5_call('symbol_info_tick', mt5_symbol)
            if not tick:
                return None

            bid = tick.bid
            ask = tick.ask
            return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2}

        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return None
