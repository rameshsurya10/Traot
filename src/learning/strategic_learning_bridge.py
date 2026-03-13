"""
Strategic Learning Bridge
==========================

BRIDGES live trading runner with continuous learning system.

This is the MISSING PIECE that connects:
1. LiveTradingRunner (live trading)
2. ContinuousLearningSystem (learning orchestrator)
3. MultiTimeframeModelManager (model management)
4. OutcomeTracker (performance tracking)
5. RetrainingEngine (automatic retraining)

WORKFLOW (Complete End-to-End):
================================

1. CANDLE ARRIVES (from WebSocket)
   ↓
2. STRATEGIC LEARNING TRIGGER
   - Check if this timeframe needs prediction
   - Fetch historical data (1 year for long-term patterns)
   - Calculate features (32 technical + 7 sentiment = 39)
   ↓
3. MULTI-TIMEFRAME PREDICTION
   - Get predictions from ALL timeframes (15m, 1h, 4h, 1d)
   - Each model analyzes patterns at its scale
   - Aggregate using weighted voting
   ↓
4. CONFIDENCE GATING
   - Check if confidence ≥ 80% (TRADING mode)
   - Or < 80% (LEARNING mode - paper trading only)
   ↓
5. TRADE EXECUTION
   - TRADING mode: Real trades (if live_brokerage configured)
   - LEARNING mode: Paper trades (always)
   - Record signal to database
   ↓
6. POSITION MONITORING
   - Track open positions
   - Check stop-loss, take-profit, time limits
   - Close when conditions met
   ↓
7. OUTCOME TRACKING
   - Record win/loss
   - Add to experience replay buffer
   - Check retraining triggers
   ↓
8. AUTOMATIC RETRAINING (if triggered)
   - Fetch recent 5,000 candles
   - Mix with experience replay (30% ratio)
   - Train with EWC (prevent forgetting)
   - Validate until confidence ≥ 80%
   - Save improved model
   ↓
9. MODE TRANSITION
   - LEARNING → TRADING (when confidence ≥ 80%)
   - TRADING → LEARNING (when accuracy drops)
   ↓
10. REPEAT FOR EVERY CANDLE

This file makes your system TRULY INTELLIGENT by:
- Learning from EVERY trade outcome
- Automatically adapting to market changes
- Using 1-year historical patterns for strategic decisions
- Preventing catastrophic forgetting with EWC
- Multi-timeframe analysis (15m, 1h, 4h, 1d)
"""

import logging
import threading
from typing import Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque

from src.core.database import Database
from src.learning.continuous_learner import ContinuousLearningSystem
from src.brokerages.base import BaseBrokerage
from src.data.provider import Candle

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Tracks an active trade for outcome recording."""
    signal_id: int
    symbol: str
    interval: str
    entry_price: float
    entry_time: datetime
    direction: str  # 'BUY' or 'SELL'
    confidence: float
    stop_loss: float
    take_profit: float
    features: any  # numpy array
    regime: str
    is_paper: bool


class StrategicLearningBridge:
    """
    Bridges LiveTradingRunner with Continuous Learning System.

    This class is the GLUE that makes everything work together.

    Responsibilities:
    1. Trigger predictions after candle close
    2. Execute trades via brokerage
    3. Monitor open positions
    4. Record outcomes when trades complete
    5. Trigger automatic retraining when needed
    6. Transition between LEARNING and TRADING modes

    Thread-safe: All operations are thread-safe
    """

    def __init__(
        self,
        database: Database,
        predictor: any,  # UnbreakablePredictor or AdvancedPredictor
        paper_brokerage: BaseBrokerage,
        live_brokerage: Optional[BaseBrokerage] = None,
        config: dict = None,
        boosted_predictor: any = None
    ):
        """
        Initialize Strategic Learning Bridge.

        Args:
            database: Database instance
            predictor: Prediction system (UnbreakablePredictor, AdvancedPredictor, etc.)
            paper_brokerage: Paper trading brokerage
            live_brokerage: Live trading brokerage (optional)
            config: Configuration dict from config.yaml
            boosted_predictor: BoostedPredictor instance (optional)
        """
        self.database = database
        self.predictor = predictor
        self.paper_brokerage = paper_brokerage
        self.live_brokerage = live_brokerage
        self.config = config or {}

        # Initialize Continuous Learning System
        logger.info("Initializing Continuous Learning System...")
        self.learning_system = ContinuousLearningSystem(
            predictor=predictor,
            database=database,
            paper_brokerage=paper_brokerage,
            live_brokerage=live_brokerage,
            config=config,
            boosted_predictor=boosted_predictor
        )

        # Track open trades (for outcome recording)
        self._open_trades: Dict[int, TradeRecord] = {}  # signal_id -> TradeRecord
        self._trades_lock = threading.Lock()
        self._mode_lock = threading.Lock()  # Thread safety for mode tracking

        # Statistics
        self._stats = {
            'candles_processed': 0,
            'predictions_made': 0,
            'trades_opened': 0,
            'trades_closed': 0,
            'wins': 0,
            'losses': 0,
            'retrainings_triggered': 0,
            'learning_mode_time': 0.0,
            'trading_mode_time': 0.0
        }
        self._stats_lock = threading.Lock()

        # Mode tracking per symbol
        self._current_modes: Dict[str, str] = {}  # symbol -> 'LEARNING' or 'TRADING'
        self._mode_transitions: deque = deque(maxlen=1000)  # Bounded to prevent memory leak

        # Recover any PENDING signals from a previous session so they can be
        # closed/evaluated by _check_and_close_trades() on the next candle.
        self._recover_open_trades_from_db()

        logger.info(
            "Strategic Learning Bridge initialized\n"
            f"  Paper brokerage: {paper_brokerage.__class__.__name__}\n"
            f"  Live brokerage: {live_brokerage.__class__.__name__ if live_brokerage else 'None'}\n"
            f"  Enabled timeframes: {self.learning_system.enabled_intervals}"
        )

    def on_candle_close(
        self,
        symbol: str,
        interval: str,
        candle: Candle
    ) -> dict:
        """
        MAIN ENTRY POINT - Called when a candle completes.

        This is where ALL the magic happens:
        1. Get multi-timeframe predictions
        2. Aggregate signals
        3. Check confidence gate
        4. Execute trade (paper or live)
        5. Monitor open positions
        6. Record outcomes
        7. Trigger retraining if needed

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            interval: Timeframe (e.g., '1h', '4h', '1d')
            candle: Closed candle data

        Returns:
            dict: Result with prediction, execution status, mode, etc.
        """
        with self._stats_lock:
            self._stats['candles_processed'] += 1

        logger.info(f"[{symbol} @ {interval}] 📊 Candle closed at {datetime.fromtimestamp(candle.timestamp / 1000)}")

        try:
            # 1. TRIGGER CONTINUOUS LEARNING SYSTEM
            result = self.learning_system.on_candle_closed(
                symbol=symbol,
                interval=interval,
                candle=candle,
                data=None  # Will fetch from database
            )

            # 2. UPDATE STATISTICS
            with self._stats_lock:
                self._stats['predictions_made'] += 1

                # Track mode time
                mode = result.get('mode', 'LEARNING')
                if mode == 'TRADING':
                    self._stats['trading_mode_time'] += 1
                else:
                    self._stats['learning_mode_time'] += 1

            # 3. UPDATE MODE TRACKING (thread-safe)
            with self._mode_lock:
                current_mode = self._current_modes.get(symbol)
                new_mode = result.get('mode')

                if current_mode != new_mode:
                    self._record_mode_transition(symbol, interval, current_mode, new_mode, result)
                    self._current_modes[symbol] = new_mode

            # 4. TRACK TRADE IF EXECUTED
            # Skip during replay — historical candles must NEVER open/close real trades.
            # Replay is pre-workout for the model only; only live candles create real trades.
            is_replay = result.get('reason') == 'replay_mode'
            if not is_replay and result.get('executed') and result.get('signal_id'):
                # Guard: prevent accumulating too many open trades per symbol.
                # This stops the batch-dump pattern where dozens of stale trades
                # all close against one candle price producing identical exits.
                max_open = self.config.get('continuous_learning', {}).get(
                    'exit_logic', {}
                ).get('max_open_trades_per_symbol', 2)
                with self._trades_lock:
                    open_for_symbol = sum(
                        1 for t in self._open_trades.values() if t.symbol == symbol
                    )
                if open_for_symbol >= max_open:
                    logger.warning(
                        f"[{symbol}] Max open trades guard: already {open_for_symbol} open "
                        f"(limit {max_open}), expiring signal {result['signal_id']}"
                    )
                    # Mark the orphan signal as EXPIRED so it doesn't sit PENDING forever
                    try:
                        self.database.close_signal(
                            signal_id=result['signal_id'],
                            outcome='EXPIRED',
                            exit_price=0.0,
                            pnl_percent=0.0
                        )
                    except Exception as e:
                        logger.debug(f"Failed to expire over-limit signal: {e}")
                else:
                    # Merge entry_price into prediction dict for trade tracking
                    prediction_dict = result['aggregated_signal']
                    if result.get('entry_price'):
                        prediction_dict['entry_price'] = result['entry_price']
                    self._track_new_trade(
                        signal_id=result['signal_id'],
                        symbol=symbol,
                        interval=interval,
                        prediction=prediction_dict,
                        is_paper=(result['mode'] == 'LEARNING')
                    )

            # 5. CHECK FOR COMPLETED TRADES
            # Skip during replay — exit prices from historical candles are not real live prices.
            if not is_replay:
                self._check_and_close_trades(symbol, candle)

            # Log result
            if result.get('executed'):
                aggregated = result.get('aggregated_signal', {})
                logger.info(
                    f"[{symbol}] {result['mode']} MODE: "
                    f"{aggregated.get('direction', 'UNKNOWN')} @ "
                    f"{aggregated.get('confidence', 0):.2%} "
                    f"({result.get('brokerage', 'unknown')} brokerage)"
                )

            return result

        except Exception as e:
            logger.error(
                f"[{symbol} @ {interval}] Error in on_candle_close: {e}",
                exc_info=True
            )
            return {
                'error': str(e),
                'mode': 'LEARNING',
                'executed': False
            }

    def _recover_open_trades_from_db(self):
        """
        Reload PENDING signals from DB into _open_trades on startup.

        Only recovers trades opened within the last max_holding_hours window.
        Any older pending signals are expired immediately — they are either
        stale replay signals or orphaned trades that can no longer be evaluated
        meaningfully against current live prices.
        """
        try:
            from datetime import timedelta

            # First expire ALL signals older than max_holding_hours (default 24h).
            # This clears any fake replay signals or stale trades from before restarts.
            exit_config = self.config.get('continuous_learning', {}).get('exit_logic', {})
            max_holding_hours = exit_config.get('max_holding_hours', 24)
            expired = self.database.close_stale_signals(max_age_hours=max_holding_hours)
            if expired:
                logger.info(f"Startup cleanup: expired {expired} stale/replay signals older than {max_holding_hours}h")

            # Only recover trades opened within the holding window
            cutoff = datetime.utcnow() - timedelta(hours=max_holding_hours)
            cutoff_ms = int(cutoff.timestamp() * 1000)

            recovered = 0
            with self.database.connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, datetime, signal_type, confidence, price,
                           stop_loss, take_profit, symbol, interval
                    FROM signals
                    WHERE (actual_outcome IS NULL OR actual_outcome = 'PENDING')
                      AND timestamp > ?
                    ORDER BY timestamp DESC
                ''', (cutoff_ms,))
                rows = cursor.fetchall()

            for row in rows:
                sig_id = row['id']
                if sig_id in self._open_trades:
                    continue

                symbol = row['symbol'] or 'UNKNOWN'
                interval = row['interval'] or '1h'
                direction = row['signal_type'] if row['signal_type'] in ('BUY', 'SELL') else 'BUY'

                try:
                    entry_time = datetime.fromisoformat(row['datetime'])
                except Exception:
                    entry_time = datetime.utcnow()

                # Infer is_paper from confidence: if the signal's confidence was above
                # the trading threshold, it was opened in TRADING mode (live).
                # Default to is_paper=False so live trade outcomes are never silently lost.
                conf_threshold = self.config.get('continuous_learning', {}).get(
                    'confidence', {}
                ).get('trading_threshold', 0.80)
                sig_confidence = row['confidence'] or 0.0
                was_live = sig_confidence >= conf_threshold

                trade = TradeRecord(
                    signal_id=sig_id,
                    symbol=symbol,
                    interval=interval,
                    entry_price=row['price'] or 0.0,
                    entry_time=entry_time,
                    direction=direction,
                    confidence=sig_confidence,
                    stop_loss=row['stop_loss'] or 0.0,
                    take_profit=row['take_profit'] or 0.0,
                    features=None,
                    regime='NORMAL',
                    is_paper=not was_live
                )
                with self._trades_lock:
                    self._open_trades[sig_id] = trade
                recovered += 1

            if recovered:
                logger.info(f"Recovered {recovered} pending trades from DB into open_trades")
        except Exception as e:
            logger.warning(f"Failed to recover open trades from DB: {e}")

    def _track_new_trade(
        self,
        signal_id: int,
        symbol: str,
        interval: str,
        prediction: dict,
        is_paper: bool
    ):
        """
        Track a newly opened trade for outcome recording.

        Args:
            signal_id: Database signal ID
            symbol: Trading pair
            interval: Timeframe
            prediction: Prediction dict from aggregated signal
            is_paper: True if paper trade
        """
        try:
            # Extract trade details from prediction
            direction = prediction.get('direction', 'BUY')
            confidence = prediction.get('confidence', 0.5)
            entry_price = prediction.get('entry_price', 0.0)
            stop_loss = prediction.get('stop_loss') or 0.0
            take_profit = prediction.get('take_profit') or 0.0
            regime = prediction.get('regime', 'NORMAL')

            # Create trade record
            trade = TradeRecord(
                signal_id=signal_id,
                symbol=symbol,
                interval=interval,
                entry_price=entry_price,
                entry_time=datetime.utcnow(),
                direction=direction,
                confidence=confidence,
                stop_loss=stop_loss,
                take_profit=take_profit,
                features=None,  # TODO: Extract from prediction if available
                regime=regime,
                is_paper=is_paper
            )

            # Store in open trades
            with self._trades_lock:
                self._open_trades[signal_id] = trade

            with self._stats_lock:
                self._stats['trades_opened'] += 1

            logger.info(
                f"[{symbol}] Tracking new trade: "
                f"{direction} @ {entry_price:.2f} "
                f"(SL: {stop_loss:.2f}, TP: {take_profit:.2f}) "
                f"{'[PAPER]' if is_paper else '[LIVE]'}"
            )

        except Exception as e:
            logger.error(f"Failed to track new trade: {e}", exc_info=True)

    def clear_open_trades(self) -> int:
        """Clear all tracked open trades. Returns count cleared. Used post-replay."""
        with self._trades_lock:
            count = len(self._open_trades)
            self._open_trades.clear()
        return count

    def _check_and_close_trades(self, symbol: str, candle: Candle):
        """
        Check open trades for exit conditions and record outcomes.

        Exit conditions (in priority order):
        1. Stop-loss hit
        2. Take-profit hit
        3. Max holding period
        4. Opposite signal

        Args:
            symbol: Trading pair
            candle: Latest candle
        """
        try:
            current_price = candle.close
            current_time = datetime.fromtimestamp(candle.timestamp / 1000)

            # Get config parameters
            exit_config = self.config.get('continuous_learning', {}).get('exit_logic', {})
            stop_loss_pct = exit_config.get('stop_loss_pct', 2.0)
            take_profit_pct = exit_config.get('take_profit_pct', 4.0)
            max_holding_hours = exit_config.get('max_holding_hours', 24)

            # Guard: cap how many trades can close per candle to prevent batch dumps.
            # If many stale signals accumulated (e.g. from restart/replay), closing them
            # all against one candle price produces fake identical-exit results.
            max_closes_per_candle = exit_config.get('max_closes_per_candle', 3)

            # Check all open trades for this symbol
            trades_to_close = []

            with self._trades_lock:
                for signal_id, trade in list(self._open_trades.items()):
                    # Only evaluate trades for this exact symbol AND timeframe.
                    # A 15m candle must not close a 1h trade — each trade is
                    # only evaluated when its own timeframe candle closes.
                    if trade.symbol != symbol or trade.interval != candle.interval:
                        continue

                    # VALIDATION: Sanity check prices
                    if trade.entry_price <= 0 or current_price <= 0:
                        logger.error(
                            f"[{symbol}] Invalid prices detected: "
                            f"entry={trade.entry_price}, current={current_price}. Skipping trade."
                        )
                        continue

                    # Calculate P&L
                    if trade.direction == 'BUY':
                        pnl_pct = ((current_price - trade.entry_price) / trade.entry_price) * 100
                    else:  # SELL
                        pnl_pct = ((trade.entry_price - current_price) / trade.entry_price) * 100

                    should_close = False
                    close_reason = None

                    # Resolve effective SL/TP: use per-trade ATR-based levels when available,
                    # fall back to config flat-percentage levels otherwise.
                    if trade.stop_loss and trade.stop_loss > 0:
                        # Price-based: convert absolute price level to % distance from entry
                        if trade.direction == 'BUY':
                            effective_sl_pct = ((trade.entry_price - trade.stop_loss) / trade.entry_price) * 100
                            effective_tp_pct = ((trade.take_profit - trade.entry_price) / trade.entry_price) * 100
                        else:
                            effective_sl_pct = ((trade.stop_loss - trade.entry_price) / trade.entry_price) * 100
                            effective_tp_pct = ((trade.entry_price - trade.take_profit) / trade.entry_price) * 100
                    else:
                        effective_sl_pct = stop_loss_pct
                        effective_tp_pct = take_profit_pct

                    # 1. STOP-LOSS CHECK
                    if pnl_pct <= -effective_sl_pct:
                        should_close = True
                        close_reason = "stop_loss"
                        logger.info(
                            f"[{symbol}] Stop-loss hit: {pnl_pct:.2f}% <= -{effective_sl_pct:.2f}%"
                        )

                    # 2. TAKE-PROFIT CHECK
                    elif pnl_pct >= effective_tp_pct:
                        should_close = True
                        close_reason = "take_profit"
                        logger.info(
                            f"[{symbol}] Take-profit hit: {pnl_pct:.2f}% >= {effective_tp_pct:.2f}%"
                        )

                    # 3. MAX HOLDING PERIOD CHECK
                    elif (current_time - trade.entry_time) > timedelta(hours=max_holding_hours):
                        should_close = True
                        close_reason = "max_holding_period"
                        logger.info(
                            f"[{symbol}] ⏰ Max holding period reached: "
                            f"{(current_time - trade.entry_time).total_seconds() / 3600:.1f}h > {max_holding_hours}h"
                        )

                    if should_close:
                        trades_to_close.append((signal_id, trade, current_price, close_reason))

            # Enforce max closes per candle to prevent batch-dump artifacts.
            # EXCEPTION: stop-loss closures are NEVER deferred — they limit losses.
            if len(trades_to_close) > max_closes_per_candle:
                # Separate stop-losses (must always close) from other reasons
                sl_trades = [t for t in trades_to_close if t[3] == 'stop_loss']
                other_trades = [t for t in trades_to_close if t[3] != 'stop_loss']

                remaining_slots = max(0, max_closes_per_candle - len(sl_trades))
                if other_trades and remaining_slots < len(other_trades):
                    other_trades.sort(key=lambda t: t[1].entry_time)
                    other_trades = other_trades[:remaining_slots]
                    logger.warning(
                        f"[{symbol}] Batch-close guard: capped non-SL closes to {remaining_slots} "
                        f"(+{len(sl_trades)} stop-losses always close). "
                        f"Deferred trades will be evaluated next candle."
                    )
                trades_to_close = sl_trades + other_trades

            # Close trades (outside lock to prevent deadlock)
            for signal_id, trade, exit_price, close_reason in trades_to_close:
                self._close_trade_and_record_outcome(
                    trade=trade,
                    exit_price=exit_price,
                    close_reason=close_reason
                )

                # Remove from open trades
                with self._trades_lock:
                    if signal_id in self._open_trades:
                        del self._open_trades[signal_id]

                with self._stats_lock:
                    self._stats['trades_closed'] += 1

        except Exception as e:
            logger.error(f"Error checking trades: {e}", exc_info=True)

    def _close_trade_and_record_outcome(
        self,
        trade: TradeRecord,
        exit_price: float,
        close_reason: str
    ):
        """
        Close trade and record outcome to continuous learning system.

        This triggers:
        1. Outcome recording in database
        2. Experience replay buffer update
        3. Retraining trigger check
        4. Performance statistics update

        Args:
            trade: TradeRecord instance
            exit_price: Exit price
            close_reason: Reason for closing ('stop_loss', 'take_profit', etc.)
        """
        try:
            # Record outcome via OutcomeTracker
            outcome = self.learning_system.outcome_tracker.record_outcome(
                signal_id=trade.signal_id,
                symbol=trade.symbol,
                interval=trade.interval,
                entry_price=trade.entry_price,
                exit_price=exit_price,
                predicted_direction=trade.direction,
                confidence=trade.confidence,
                features=trade.features,
                regime=trade.regime,
                is_paper_trade=trade.is_paper
            )

            # Update statistics
            with self._stats_lock:
                if outcome['was_correct']:
                    self._stats['wins'] += 1
                else:
                    self._stats['losses'] += 1

                # Check if retraining was triggered
                if outcome.get('should_retrain'):
                    self._stats['retrainings_triggered'] += 1

            # Mark signal as closed in the signals table (WIN/LOSS/EXPIRED)
            # This is the critical step that moves the signal out of PENDING state.
            try:
                outcome_label = 'WIN' if outcome['was_correct'] else 'LOSS'
                self.database.close_signal(
                    signal_id=trade.signal_id,
                    outcome=outcome_label,
                    exit_price=exit_price,
                    pnl_percent=outcome['pnl_percent']
                )
            except Exception as db_err:
                logger.error(f"Failed to close signal {trade.signal_id} in DB: {db_err}")

            # Log outcome
            logger.info(
                f"[{trade.symbol}] Trade closed: "
                f"{'WIN' if outcome['was_correct'] else 'LOSS'} "
                f"({outcome['pnl_percent']:+.2f}%) "
                f"- Reason: {close_reason} "
                f"{'[PAPER]' if trade.is_paper else '[LIVE]'}"
            )

            # If retraining triggered, actually execute it
            if outcome.get('should_retrain'):
                trigger_reason = outcome.get('trigger_reason', 'unknown')
                logger.info(
                    f"[{trade.symbol}] 🔄 Retraining triggered: {trigger_reason}"
                )
                try:
                    self.learning_system._schedule_retrain(
                        symbol=trade.symbol,
                        interval=trade.interval,
                        reason=trigger_reason
                    )
                except Exception as retrain_err:
                    logger.error(
                        f"[{trade.symbol}] Failed to schedule retraining: {retrain_err}"
                    )

        except Exception as e:
            logger.error(f"Failed to record outcome: {e}", exc_info=True)

    def _record_mode_transition(
        self,
        symbol: str,
        interval: str,
        old_mode: Optional[str],
        new_mode: str,
        result: dict
    ):
        """
        Record mode transition for analytics.

        Args:
            symbol: Trading pair
            interval: Timeframe
            old_mode: Previous mode ('LEARNING' or 'TRADING')
            new_mode: New mode
            result: Result dict from continuous learning system
        """
        transition = {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': symbol,
            'interval': interval,
            'from_mode': old_mode,
            'to_mode': new_mode,
            'confidence': result.get('aggregated_signal', {}).get('confidence', 0.0),
            'reason': result.get('reason', 'unknown')
        }

        self._mode_transitions.append(transition)

        # Log transition
        logger.info(
            f"[{symbol} @ {interval}] Mode transition: "
            f"{old_mode or 'INITIAL'} → {new_mode} "
            f"(reason: {transition['reason']})"
        )

    def get_stats(self) -> dict:
        """Get comprehensive statistics."""
        with self._stats_lock:
            stats = self._stats.copy()

        # Add derived metrics
        total_trades = stats['trades_closed']
        if total_trades > 0:
            stats['win_rate'] = stats['wins'] / total_trades
        else:
            stats['win_rate'] = 0.0

        # Add learning system stats
        stats['learning_system'] = self.learning_system.get_stats()

        # Add open trades count
        with self._trades_lock:
            stats['open_trades'] = len(self._open_trades)

        # Add mode distribution
        total_candles = stats['learning_mode_time'] + stats['trading_mode_time']
        if total_candles > 0:
            stats['learning_mode_pct'] = (stats['learning_mode_time'] / total_candles) * 100
            stats['trading_mode_pct'] = (stats['trading_mode_time'] / total_candles) * 100
        else:
            stats['learning_mode_pct'] = 0.0
            stats['trading_mode_pct'] = 0.0

        return stats

    def stop(self):
        """Stop the bridge and underlying learning system."""
        logger.info("Stopping Strategic Learning Bridge...")

        # Stop continuous learning system
        self.learning_system.stop()

        # Log final stats
        stats = self.get_stats()
        logger.info(
            f"Strategic Learning Bridge stopped\n"
            f"  Total candles processed: {stats['candles_processed']}\n"
            f"  Total predictions: {stats['predictions_made']}\n"
            f"  Total trades: {stats['trades_closed']}\n"
            f"  Win rate: {stats['win_rate']:.2%}\n"
            f"  Retrainings triggered: {stats['retrainings_triggered']}\n"
            f"  Learning mode: {stats['learning_mode_pct']:.1f}%\n"
            f"  Trading mode: {stats['trading_mode_pct']:.1f}%"
        )
