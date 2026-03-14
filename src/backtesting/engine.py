"""
Backtest Engine (Lean-Inspired)
===============================
Event-driven backtesting with realistic order execution.

Enhanced with:
- Event-driven architecture (OnData, OnOrderEvent)
- Multiple order types (Market, Limit, Stop, Trailing)
- Realistic slippage and commission modeling
- Uses same brokerage interface as live trading
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Callable
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config import Config
from src.brokerages.base import BaseBrokerage, CashBalance, Position
from src.brokerages.orders import Order, OrderTicket, OrderType, OrderSide, OrderStatus
from src.brokerages.events import OrderEvent, OrderEventType
from .metrics import BacktestMetrics, calculate_metrics

logger = logging.getLogger(__name__)


@dataclass
class Slice:
    """
    Data slice at a point in time (Lean-inspired).

    Contains all market data available at this moment.

    Performance: Uses lazy evaluation for bars to avoid DataFrame copies in hot loop.
    """
    time: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    # Lazy bar access (avoids DataFrame copy per iteration)
    _source_df: Optional[pd.DataFrame] = field(default=None, repr=False)
    _bar_start_idx: int = field(default=0, repr=False)
    _bar_end_idx: int = field(default=0, repr=False)

    # Legacy support - will be None if using lazy access
    bars: Optional[pd.DataFrame] = field(default=None, repr=False)

    def get_bars(self) -> Optional[pd.DataFrame]:
        """
        Get historical bars for this slice (lazy evaluation).

        Returns a view/slice of the source DataFrame only when called,
        avoiding expensive copies in the hot loop.
        """
        if self.bars is not None:
            return self.bars
        if self._source_df is not None:
            # Return a view, not a copy (10x faster)
            return self._source_df.iloc[self._bar_start_idx:self._bar_end_idx]
        return None

    def get_bars_copy(self) -> Optional[pd.DataFrame]:
        """Get a copy of bars (use only if modification needed)."""
        bars = self.get_bars()
        return bars.copy() if bars is not None else None

    def __repr__(self):
        return f"Slice({self.time}: {self.symbol} @ ${self.close:.2f})"


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    # Capital
    initial_cash: float = 100000

    # Data range
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    # Position management
    max_open_positions: int = 1
    allow_concurrent: bool = False

    # Simulation settings
    slippage_percent: float = 0.05
    commission_percent: float = 0.1

    # Exit rules
    max_hold_candles: int = 24
    use_trailing_stop: bool = False
    trailing_stop_percent: float = 1.0


class BacktestBrokerage(BaseBrokerage):
    """
    Simulated brokerage for backtesting (Lean-inspired).

    Provides realistic order execution with:
    - Slippage modeling
    - Commission simulation
    - Multiple order types
    - Position tracking
    """

    def __init__(
        self,
        initial_cash: float = 100000,
        slippage_percent: float = 0.05,
        commission_percent: float = 0.1
    ):
        super().__init__("Backtest")

        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.slippage_pct = slippage_percent / 100
        self.commission_pct = commission_percent / 100

        # Position tracking
        self._positions: Dict[str, _BacktestPosition] = {}
        self._pending_orders: List[Order] = []
        self._trade_history: List[Dict] = []

        self._is_connected = True

    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        pass

    def place_order(self, order: Order) -> OrderTicket:
        """Queue order for execution on next bar."""
        order.status = OrderStatus.SUBMITTED
        order.submitted_time = datetime.utcnow()
        self._orders[order.id] = order
        self._pending_orders.append(order)

        self._emit_order_event(OrderEvent(
            order_id=order.id,
            event_type=OrderEventType.SUBMITTED,
            order=order
        ))

        return OrderTicket(order=order, _brokerage=self)

    def process_bar(self, bar: Slice) -> None:
        """
        Process pending orders against new bar.

        Called for each bar during backtest.
        """
        orders_to_remove = []

        for order in self._pending_orders:
            if order.symbol != bar.symbol:
                continue

            filled = self._try_fill_order(order, bar)
            if filled:
                orders_to_remove.append(order)

        for order in orders_to_remove:
            self._pending_orders.remove(order)

        # Update position P&L
        if bar.symbol in self._positions:
            pos = self._positions[bar.symbol]
            pos.current_price = bar.close
            pos.unrealized_pnl = (bar.close - pos.avg_price) * pos.quantity

    def _try_fill_order(self, order: Order, bar: Slice) -> bool:
        """Attempt to fill order at current bar."""
        fill_price = None

        if order.order_type == OrderType.MARKET:
            fill_price = bar.open
            if order.side == OrderSide.BUY:
                fill_price *= (1 + self.slippage_pct)
            else:
                fill_price *= (1 - self.slippage_pct)

        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY:
                if bar.low <= order.limit_price:
                    fill_price = min(order.limit_price, bar.open)
            else:
                if bar.high >= order.limit_price:
                    fill_price = max(order.limit_price, bar.open)

        elif order.order_type == OrderType.STOP_MARKET:
            if order.side == OrderSide.BUY:
                if bar.high >= order.stop_price:
                    fill_price = max(order.stop_price, bar.open)
                    fill_price *= (1 + self.slippage_pct)
            else:
                if bar.low <= order.stop_price:
                    fill_price = min(order.stop_price, bar.open)
                    fill_price *= (1 - self.slippage_pct)

        elif order.order_type == OrderType.STOP_LIMIT:
            if order.side == OrderSide.BUY and bar.high >= order.stop_price:
                if bar.low <= order.limit_price:
                    fill_price = order.limit_price
            elif order.side == OrderSide.SELL and bar.low <= order.stop_price:
                if bar.high >= order.limit_price:
                    fill_price = order.limit_price

        elif order.order_type == OrderType.TRAILING_STOP:
            self._update_trailing_stop(order, bar)
            if order.side == OrderSide.SELL and bar.low <= order.stop_price:
                fill_price = min(order.stop_price, bar.open)
            elif order.side == OrderSide.BUY and bar.high >= order.stop_price:
                fill_price = max(order.stop_price, bar.open)

        if fill_price is None:
            return False

        return self._execute_fill(order, fill_price, bar.time)

    def _update_trailing_stop(self, order: Order, bar: Slice):
        """Update trailing stop price."""
        if order.trailing_amount is None:
            return

        if order.trailing_as_percent:
            trail = bar.close * (order.trailing_amount / 100)
        else:
            trail = order.trailing_amount

        if order.side == OrderSide.SELL:
            new_stop = bar.high - trail
            if order.stop_price is None or new_stop > order.stop_price:
                order.stop_price = new_stop
        else:
            new_stop = bar.low + trail
            if order.stop_price is None or new_stop < order.stop_price:
                order.stop_price = new_stop

    def _execute_fill(self, order: Order, fill_price: float, fill_time: datetime) -> bool:
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
                    message="Insufficient cash"
                ))
                return False

            self.cash -= total_cost

            # Update position
            if order.symbol in self._positions:
                pos = self._positions[order.symbol]
                total_qty = pos.quantity + order.quantity
                pos.avg_price = (pos.avg_price * pos.quantity + fill_price * order.quantity) / total_qty
                pos.quantity = total_qty
            else:
                self._positions[order.symbol] = _BacktestPosition(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    avg_price=fill_price,
                    entry_time=fill_time
                )

        else:  # SELL
            pos = self._positions.get(order.symbol)
            if not pos or pos.quantity < order.quantity:
                order.status = OrderStatus.REJECTED
                self._emit_order_event(OrderEvent(
                    order_id=order.id,
                    event_type=OrderEventType.REJECTED,
                    order=order,
                    message="Insufficient position"
                ))
                return False

            proceeds = trade_value - commission
            self.cash += proceeds

            # Calculate P&L
            pnl = (fill_price - pos.avg_price) * order.quantity - commission
            pnl_pct = (pnl / (pos.avg_price * order.quantity)) * 100

            # Record trade
            self._trade_history.append({
                'symbol': order.symbol,
                'entry_price': pos.avg_price,
                'exit_price': fill_price,
                'entry_time': pos.entry_time,
                'exit_time': fill_time,
                'quantity': order.quantity,
                'pnl': pnl,
                'pnl_percent': pnl_pct,
                'commission': commission,
                'direction': 'long',
                'is_winner': pnl > 0
            })

            # Update position
            pos.quantity -= order.quantity
            if pos.quantity <= 0:
                del self._positions[order.symbol]

        # Update order
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_fill_price = fill_price
        order.commission = commission
        order.filled_time = fill_time

        self._emit_order_event(OrderEvent(
            order_id=order.id,
            event_type=OrderEventType.FILLED,
            order=order,
            fill_price=fill_price,
            fill_quantity=order.quantity,
            commission=commission
        ))

        return True

    def update_order(self, order: Order, **kwargs) -> bool:
        if not order.is_open:
            return False
        for key, value in kwargs.items():
            if value is not None and hasattr(order, key):
                setattr(order, key, value)
        return True

    def cancel_order(self, order: Order) -> bool:
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

    def get_cash_balance(self) -> List[CashBalance]:
        return [CashBalance(currency="USD", amount=self.cash, available=self.cash)]

    def get_positions(self) -> List[Position]:
        return [Position(
            symbol=p.symbol,
            quantity=p.quantity,
            average_price=p.avg_price,
            market_value=p.quantity * p.current_price,
            unrealized_pnl=p.unrealized_pnl,
            unrealized_pnl_percent=(p.unrealized_pnl / (p.avg_price * p.quantity) * 100) if p.avg_price > 0 else 0
        ) for p in self._positions.values()]

    def get_account_value(self) -> float:
        positions_value = sum(p.quantity * p.current_price for p in self._positions.values())
        return self.cash + positions_value

    def get_trade_history(self) -> List[Dict]:
        return self._trade_history

    def reset(self):
        """Reset brokerage state."""
        self.cash = self.initial_cash
        self._positions.clear()
        self._pending_orders.clear()
        self._orders.clear()
        self._trade_history.clear()


@dataclass
class _BacktestPosition:
    """Internal position tracking."""
    symbol: str
    quantity: float
    avg_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0


class BacktestEngine:
    """
    Event-driven backtest engine (Lean-inspired).

    Usage:
        engine = BacktestEngine(config_path="config.yaml")

        # Define your strategy
        def on_data(slice, brokerage):
            prediction = model.predict(slice.bars)
            if prediction.signal == "BUY":
                order = Order(
                    symbol=slice.symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=1.0
                )
                brokerage.place_order(order)

        results = engine.run(on_data_callback=on_data)
        print(results.summary())
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        backtest_config: Optional[BacktestConfig] = None,
        config_dict: Optional[dict] = None,
    ):
        if config_dict is not None:
            self.config = Config.load(config_path)
            self.config.raw = config_dict
            # Override typed fields that the backtest engine uses
            signals = config_dict.get('signals', {})
            if signals:
                from src.core.config import SignalConfig
                self.config.signals = SignalConfig(
                    risk_per_trade=signals.get('risk_per_trade', 0.02),
                    risk_reward_ratio=signals.get('risk_reward_ratio', 2.0),
                    strong_signal=signals.get('strong_signal', 0.65),
                    medium_signal=signals.get('medium_signal', 0.55),
                    cooldown_minutes=signals.get('cooldown_minutes', 60),
                )
            analysis = config_dict.get('analysis', {})
            if analysis:
                from src.core.config import AnalysisConfig
                self.config.analysis = AnalysisConfig(
                    update_interval=analysis.get('update_interval', 60),
                    min_confidence=analysis.get('min_confidence', 0.55),
                    lookback_period=analysis.get('lookback_period', 100),
                )
        else:
            self.config = Config.load(config_path)
        self.bt_config = backtest_config or BacktestConfig()

        # Brokerage
        self.brokerage = BacktestBrokerage(
            initial_cash=self.bt_config.initial_cash,
            slippage_percent=self.bt_config.slippage_percent,
            commission_percent=self.bt_config.commission_percent
        )

        # State
        self.df: Optional[pd.DataFrame] = None
        self.current_index: int = 0

    def load_data(self, df: Optional[pd.DataFrame] = None, copy: bool = True) -> bool:
        """Load historical data for backtesting."""
        if df is not None:
            self.df = df.copy() if copy else df
        else:
            from src.data_service import DataService
            data_service = DataService()
            self.df = data_service.get_candles(limit=100000)

        if self.df.empty or len(self.df) < 200:
            logger.error(f"Insufficient data: {len(self.df) if self.df is not None else 0} candles")
            return False

        # Apply date filters
        if self.bt_config.start_date:
            self.df = self.df[self.df['datetime'] >= self.bt_config.start_date]
        if self.bt_config.end_date:
            self.df = self.df[self.df['datetime'] <= self.bt_config.end_date]

        self.df = self.df.reset_index(drop=True)

        logger.info(f"Loaded {len(self.df)} candles for backtesting")
        logger.info(f"Date range: {self.df['datetime'].min()} to {self.df['datetime'].max()}")

        return True

    def run(
        self,
        on_data_callback: Optional[Callable[[Slice, BacktestBrokerage], None]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BacktestMetrics:
        """
        Run the backtest.

        Args:
            on_data_callback: Called for each bar with (Slice, Brokerage)
            progress_callback: Called with (current, total) for progress

        Returns:
            BacktestMetrics with results
        """
        if self.df is None or self.df.empty:
            if not self.load_data():
                return BacktestMetrics()

        # Reset brokerage
        self.brokerage.reset()

        # Get symbol from config
        symbol = self.config.data.symbol

        # Minimum lookback
        min_lookback = max(200, self.config.model.sequence_length + 50)
        total_candles = len(self.df) - min_lookback

        logger.info(f"Starting backtest with {total_candles} tradeable candles")

        # Main loop
        for i in range(min_lookback, len(self.df)):
            self.current_index = i
            row = self.df.iloc[i]

            # Create slice with lazy bar access (avoids DataFrame copy - 10x faster)
            slice_data = Slice(
                time=row['datetime'],
                symbol=symbol,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                _source_df=self.df,
                _bar_start_idx=i - min_lookback,
                _bar_end_idx=i + 1
            )

            # Process pending orders from previous bar
            self.brokerage.process_bar(slice_data)

            # Call user strategy
            if on_data_callback:
                try:
                    on_data_callback(slice_data, self.brokerage)
                except Exception as e:
                    logger.error(f"Strategy error at {slice_data.time}: {e}")

            # Progress update
            if progress_callback and (i - min_lookback) % 100 == 0:
                progress_callback(i - min_lookback, total_candles)

        # Close remaining positions
        self._close_all_positions()

        logger.info("-" * 60)
        trades = self.brokerage.get_trade_history()
        logger.info(f"Backtest complete: {len(trades)} trades")

        return calculate_metrics(trades)

    def _close_all_positions(self):
        """Close all remaining positions at last price."""
        if self.df is None or self.df.empty:
            return

        last_row = self.df.iloc[-1]
        symbol = self.config.data.symbol

        for pos in self.brokerage.get_positions():
            order = Order(
                symbol=pos.symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=pos.quantity
            )
            self.brokerage.place_order(order)

            # Process immediately
            slice_data = Slice(
                time=last_row['datetime'],
                symbol=symbol,
                open=last_row['close'],
                high=last_row['close'],
                low=last_row['close'],
                close=last_row['close'],
                volume=0
            )
            self.brokerage.process_bar(slice_data)

    def run_with_model(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BacktestMetrics:
        """
        Run backtest using the configured ML model.

        This is the default mode using your existing prediction engine.
        """
        from src.analysis_engine import AnalysisEngine

        # Load model
        analysis = AnalysisEngine(self.config._config_path or "config.yaml")
        analysis.load_model()

        signals_generated = 0
        open_position = None

        def on_data(slice_data: Slice, brokerage: BacktestBrokerage):
            nonlocal signals_generated, open_position

            # Check for exit first
            if open_position:
                self._check_exit(slice_data, brokerage, open_position)
                if open_position.get('closed'):
                    open_position = None

            # Skip if at max positions
            if open_position and not self.bt_config.allow_concurrent:
                return

            # Generate prediction
            prediction = analysis.predict(slice_data.bars)

            signal = prediction.get('signal', 'NEUTRAL')
            confidence = prediction.get('confidence', 0)

            # Skip neutral/low confidence
            if signal in ['NEUTRAL', 'WAIT']:
                return
            if confidence < self.config.analysis.min_confidence:
                return

            signals_generated += 1

            # Determine direction and create order
            if 'BUY' in signal:
                side = OrderSide.BUY
            else:
                side = OrderSide.SELL

            # Calculate position size (simplified)
            account_value = brokerage.get_account_value()
            risk_amount = account_value * self.config.signals.risk_per_trade
            position_value = risk_amount * 10  # 10:1 simplified
            quantity = position_value / slice_data.close

            order = Order(
                symbol=slice_data.symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity,
                stop_loss=prediction.get('stop_loss'),
                take_profit=prediction.get('take_profit')
            )

            ticket = brokerage.place_order(order)

            if ticket.status == OrderStatus.FILLED or ticket.status == OrderStatus.SUBMITTED:
                open_position = {
                    'order': order,
                    'entry_time': slice_data.time,
                    'entry_index': self.current_index,
                    'stop_loss': prediction.get('stop_loss'),
                    'take_profit': prediction.get('take_profit'),
                    'direction': side,
                    'closed': False
                }

                # Create SL/TP orders
                if order.stop_loss:
                    sl_order = Order(
                        symbol=slice_data.symbol,
                        side=OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY,
                        order_type=OrderType.STOP_MARKET,
                        quantity=quantity,
                        stop_price=order.stop_loss
                    )
                    brokerage.place_order(sl_order)

                if order.take_profit:
                    tp_order = Order(
                        symbol=slice_data.symbol,
                        side=OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY,
                        order_type=OrderType.LIMIT,
                        quantity=quantity,
                        limit_price=order.take_profit
                    )
                    brokerage.place_order(tp_order)

        def _check_exit(slice_data: Slice, brokerage: BacktestBrokerage, position: dict):
            """Check if position should be closed."""
            candles_held = self.current_index - position['entry_index']

            # Max hold time
            if candles_held >= self.bt_config.max_hold_candles:
                for pos in brokerage.get_positions():
                    order = Order(
                        symbol=pos.symbol,
                        side=OrderSide.SELL if position['direction'] == OrderSide.BUY else OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=pos.quantity
                    )
                    brokerage.place_order(order)
                    brokerage.cancel_all_orders(pos.symbol)
                position['closed'] = True

        self._check_exit = _check_exit

        return self.run(on_data_callback=on_data, progress_callback=progress_callback)

    def get_equity_curve(self) -> List[float]:
        """Get cumulative P&L over time."""
        trades = self.brokerage.get_trade_history()
        equity = [0]
        for trade in trades:
            equity.append(equity[-1] + trade['pnl_percent'])
        return equity

    def get_trade_details(self) -> pd.DataFrame:
        """Get all trades as DataFrame."""
        trades = self.brokerage.get_trade_history()
        if not trades:
            return pd.DataFrame()
        return pd.DataFrame(trades)


def run_backtest(
    config_path: str = "config.yaml",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    initial_cash: float = 100000,
    verbose: bool = True
) -> BacktestMetrics:
    """
    Convenience function to run a backtest.

    Args:
        config_path: Path to config file
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
        initial_cash: Starting capital
        verbose: Whether to print progress

    Returns:
        BacktestMetrics
    """
    bt_config = BacktestConfig(initial_cash=initial_cash)

    if start_date:
        bt_config.start_date = datetime.fromisoformat(start_date)
    if end_date:
        bt_config.end_date = datetime.fromisoformat(end_date)

    engine = BacktestEngine(config_path, bt_config)

    def progress(current, total):
        if verbose:
            pct = current / total * 100 if total > 0 else 0
            print(f"\rBacktesting: {pct:.1f}% ({current}/{total})", end="", flush=True)

    results = engine.run_with_model(progress_callback=progress if verbose else None)

    if verbose:
        print("\n")
        print(results.summary())

    return results


# Legacy support - keep old names working
OpenPosition = _BacktestPosition


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

    parser = argparse.ArgumentParser(description='Run backtest')
    parser.add_argument('--config', default='config.yaml', help='Config file')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    parser.add_argument('--cash', type=float, default=100000, help='Initial cash')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    results = run_backtest(
        config_path=args.config,
        start_date=args.start,
        end_date=args.end,
        initial_cash=args.cash,
        verbose=not args.quiet
    )
