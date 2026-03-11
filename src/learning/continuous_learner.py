"""
Continuous Learning System
===========================

Main orchestrator for continuous learning trading system.

Workflow:
1. Get predictions from all enabled timeframes
2. Aggregate signals using configured method
3. Check confidence gate
4. Execute trade (live or paper based on mode)
5. Track outcomes when trades close
6. Trigger retraining if needed
7. Perform online learning updates

Features:
- 80% confidence threshold for live trading
- Immediate retraining on losses
- Multi-timeframe signal aggregation
- Paper trading during learning mode
- Event-driven architecture (webhook-based)
"""

import logging
import math
import threading
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

from src.core.database import Database
from src.core.types import Signal, SignalType, SignalStrength
from src.multi_timeframe.model_manager import MultiTimeframeModelManager
from src.multi_timeframe.aggregator import SignalAggregator, TimeframeSignal, AggregatedSignal
from src.learning.confidence_gate import ConfidenceGate
from src.learning.state_manager import LearningStateManager
from src.learning.outcome_tracker import OutcomeTracker
from src.learning.retraining_engine import RetrainingEngine
from src.learning.strategy_analyzer import StrategyAnalyzer
from src.ml.learning.continual import ContinualLearner

logger = logging.getLogger(__name__)


class ContinuousLearningSystem:
    """
    Main orchestrator for continuous learning.

    Thread-safe: All operations are thread-safe
    Event-driven: Triggered by candle close events
    Configurable: All parameters from config.yaml
    """

    def __init__(
        self,
        predictor: any,  # UnbreakablePredictor instance
        database: Database,
        paper_brokerage: any,
        live_brokerage: any = None,
        config: dict = None,
        boosted_predictor: any = None
    ):
        """
        Initialize continuous learning system.

        Args:
            predictor: UnbreakablePredictor instance
            database: Database instance
            paper_brokerage: Paper trading brokerage
            live_brokerage: Live trading brokerage (optional)
            config: Configuration dict from config.yaml
            boosted_predictor: BoostedPredictor instance (optional)
        """
        self.predictor = predictor
        self.database = database
        self.paper_brokerage = paper_brokerage
        self.live_brokerage = live_brokerage
        self.config = config or {}
        self.boosted_predictor = boosted_predictor

        # Get configurations
        self.cl_config = self.config.get('continuous_learning', {})
        self.timeframes_config = self.config.get('timeframes', {})

        # Signal strength thresholds from config
        signals_cfg = self.config.get('signals', {})
        self._strong_signal_threshold = signals_cfg.get('strong_signal', 0.65)
        self._medium_signal_threshold = signals_cfg.get('medium_signal', 0.55)

        # Cache sequence lengths as dict for O(1) lookup (called multiple times per candle)
        self._sequence_lengths = {
            cfg['interval']: cfg.get('sequence_length', 60)
            for cfg in self.timeframes_config.get('intervals', [])
            if 'interval' in cfg
        }

        # Initialize components
        self.signal_aggregator = SignalAggregator(
            config=self.timeframes_config
        )

        # Create ConfidenceGateConfig from dict
        conf_dict = self.cl_config.get('confidence', {})
        gate_cfg = self.config.get('confidence_gate', {})
        from src.learning.confidence_gate import ConfidenceGateConfig
        conf_config = ConfidenceGateConfig(
            trading_threshold=conf_dict.get('trading_threshold', 0.8),
            hysteresis=conf_dict.get('hysteresis', 0.05),
            smoothing_alpha=conf_dict.get('smoothing_alpha', 0.3),
            regime_adjustment=conf_dict.get('regime_adjustment', True),
            min_threshold_clamp=gate_cfg.get('min_threshold_clamp', 0.50),
            max_threshold_clamp=gate_cfg.get('max_threshold_clamp', 0.95),
            max_history_size=gate_cfg.get('max_history_size', 100)
        )

        self.confidence_gate = ConfidenceGate(config=conf_config)

        self.state_manager = LearningStateManager(
            database=database
        )

        # Get continual learner from predictor
        continual_learner = getattr(predictor, 'continual_learner', None)
        if not continual_learner and hasattr(predictor, 'model'):
            logger.warning("Predictor has no continual_learner, creating new one")
            ewc_config = self.cl_config.get("ewc", {})
            continual_learner = ContinualLearner(
                model=predictor.model,
                ewc_lambda=ewc_config.get("lambda", 1000.0),
                replay_buffer_size=self.cl_config.get("experience_replay", {}).get("buffer_size", 10000),
                replay_batch_size=self.cl_config.get("experience_replay", {}).get("batch_size", 32),
                drift_window=self.cl_config.get("drift_window", 100)
            )
        elif not continual_learner:
            logger.info("Predictor is not a neural network model, skipping continual learner")
            continual_learner = None

        # Build retraining config, injecting trading_threshold so OutcomeTracker
        # uses the same threshold for high-confidence loss detection
        retrain_cfg = dict(self.cl_config.get('retraining', {}))
        retrain_cfg.setdefault(
            'high_conf_retrain_threshold',
            conf_dict.get('trading_threshold', 0.80)
        )
        self.outcome_tracker = OutcomeTracker(
            database=database,
            continual_learner=continual_learner,
            config=retrain_cfg
        )

        # Always create a MultiTimeframeModelManager for retraining
        # (predictor.model_manager is multi_currency_system.ModelManager with incompatible API)
        retrain_model_manager = MultiTimeframeModelManager(
            models_dir=self.config.get('model', {}).get('models_dir', 'models'),
            config=self.config.get('model', {})
        )

        self.retraining_engine = RetrainingEngine(
            model_manager=retrain_model_manager,
            continual_learner=continual_learner,
            database=database,
            config=self.config
        )

        # Track active retraining threads
        self.retraining_threads: Dict[str, threading.Thread] = {}
        self._threads_lock = threading.Lock()

        # Replay mode: suppresses retraining during historical replay
        self.replay_mode = False

        # Statistics
        self._stats_lock = threading.Lock()
        self._stats = {
            'candles_processed': 0,
            'predictions_made': 0,
            'trades_executed': 0,
            'paper_trades': 0,
            'live_trades': 0,
            'retrainings_triggered': 0,
            'online_updates': 0,
            'mode_transitions': 0
        }

        # Store latest predictions per symbol for opposite signal detection
        self._latest_predictions: Dict[str, AggregatedSignal] = {}
        self._predictions_lock = threading.Lock()

        # Track last processed outcome per symbol/interval to avoid duplicate updates
        self._last_online_update_outcome_id: Dict[str, int] = {}
        self._outcome_tracking_lock = threading.Lock()

        # Track previous candle close per (symbol, interval) for correct online learning target
        # LSTM target = "next_close > current_close", so fallback needs previous close
        self._prev_close: Dict[str, float] = {}

        # Cooldown tracking: prevent duplicate signals per (symbol, interval)
        # Key = "symbol_interval", value = datetime of last saved signal
        self._last_signal_time: Dict[str, datetime] = {}
        self._cooldown_lock = threading.Lock()  # Guards _last_signal_time (thread-safe read-check-write)

        # Strategy analyzer for strategy-based trade decisions
        db_path = self.config.get('database', {}).get('path', 'data/trading.db')
        try:
            self.strategy_analyzer = StrategyAnalyzer(
                database_path=db_path,
                config_path='config.yaml'
            )
        except Exception as e:
            logger.warning(f"Strategy analyzer initialization failed: {e}. Strategy-based filtering disabled.")
            self.strategy_analyzer = None

        self._strategy_cache: Dict[str, dict] = {}  # symbol -> best strategy
        self._strategy_cache_lock = threading.Lock()
        self._last_strategy_analysis: Dict[str, datetime] = {}

        # Get enabled intervals
        self.enabled_intervals = self._get_enabled_intervals()

        # Initialize PerformanceBasedLearner for per-candle learning
        try:
            from src.learning.performance_learner import create_performance_learner

            perf_config = self.cl_config.get('performance_learning', {})
            self.performance_learner = create_performance_learner(
                predictor=predictor,
                database=database,
                timeframes=perf_config.get('timeframes', self.enabled_intervals or ['15m', '1h']),
                loss_retrain=perf_config.get('loss_retrain', True),
                reinforce_wins=perf_config.get('reinforce_wins', True)
            )
            logger.info("PerformanceBasedLearner initialized for per-candle learning")
        except Exception as e:
            logger.warning(f"PerformanceBasedLearner initialization failed: {e}. Per-candle learning disabled.")
            self.performance_learner = None

        logger.info(
            f"ContinuousLearningSystem initialized: "
            f"intervals={self.enabled_intervals}, "
            f"aggregation={self.timeframes_config.get('aggregation_method', 'weighted_vote')}"
        )

    def _get_enabled_intervals(self) -> List[str]:
        """Get list of enabled intervals from config."""
        intervals = []

        for interval_config in self.timeframes_config.get('intervals', []):
            if interval_config.get('enabled', True):
                intervals.append(interval_config['interval'])

        return intervals

    def on_candle_closed(
        self,
        symbol: str,
        interval: str,
        candle: any,
        data: dict = None
    ) -> dict:
        """
        Main event handler - called when a candle completes.

        This is the entry point for all continuous learning logic.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            interval: Timeframe that closed (e.g., '1h')
            candle: Candle object
            data: Pre-fetched data (optional)

        Returns:
            Dict with prediction results, execution status, mode, etc.
        """
        try:
            with self._stats_lock:
                self._stats['candles_processed'] += 1

            logger.info(
                f"[{symbol} @ {interval}] Candle closed at {candle.timestamp}"
            )

            # 1. ONLINE LEARNING UPDATE (fast, incremental)
            if self.cl_config.get('online_learning', {}).get('enabled', True):
                self._online_update(symbol, interval, candle, data)

            # 2. GET PREDICTIONS FROM ALL TIMEFRAMES
            timeframe_signals = self._get_all_timeframe_predictions(
                symbol=symbol,
                data=data
            )

            # 3. AGGREGATE SIGNALS
            aggregated = self.signal_aggregator.aggregate(timeframe_signals)

            # Extract SL/TP from primary interval's advanced_result (BUG 2 fix)
            primary_interval = self._get_primary_interval()
            primary_signal = timeframe_signals.get(primary_interval)
            sl = None
            tp = None
            if primary_signal and primary_signal.advanced_result:
                sl = getattr(primary_signal.advanced_result, 'stop_loss', None)
                tp = getattr(primary_signal.advanced_result, 'take_profit', None)

            # 4. GET CURRENT MODE
            # Use highest weighted timeframe for mode determination
            current_mode = self.state_manager.get_current_mode(
                symbol=symbol,
                interval=primary_interval
            )

            # 5. CHECK CONFIDENCE GATE
            can_trade_live, reason = self.confidence_gate.should_trade(
                confidence=aggregated.confidence,
                current_mode=current_mode,
                regime=aggregated.regime,
                symbol=symbol,
                interval=primary_interval
            )

            # 6. DETERMINE MODE AND BROKERAGE
            # confidence_gate.should_trade() is the single authority for mode decision
            # (it already checks confidence vs trading_threshold with hysteresis + regime)
            if can_trade_live:
                mode = 'TRADING'
                brokerage = self.live_brokerage if self.live_brokerage else self.paper_brokerage
                brokerage_type = 'live' if self.live_brokerage else 'paper_fallback'
            else:
                mode = 'LEARNING'
                brokerage = self.paper_brokerage
                brokerage_type = 'paper'

            # Update state if changed
            if mode != current_mode:
                self._handle_mode_transition(
                    symbol=symbol,
                    interval=primary_interval,
                    new_mode=mode,
                    confidence=aggregated.confidence,
                    reason=reason
                )

            # During replay: execute paper trades for learning data, but skip live execution
            if self.replay_mode:
                with self._stats_lock:
                    self._stats['predictions_made'] += 1

                # Still save signals during replay so strategies can be evaluated later
                replay_signal_id = None
                replay_executed = False
                entry_price = candle.close if hasattr(candle, 'close') else None
                if aggregated.direction != 'NEUTRAL' and entry_price:
                    try:
                        replay_signal_id = self._execute_trade(
                            brokerage=self.paper_brokerage,
                            symbol=symbol,
                            interval=primary_interval if primary_interval else interval,
                            prediction=aggregated,
                            is_paper=True,
                            strategy_name=None,
                            perf_signal_id=None,
                            entry_price=entry_price,
                            stop_loss=sl,
                            take_profit=tp
                        )
                        replay_executed = replay_signal_id is not None
                    except Exception as e:
                        logger.debug(f"Replay trade save failed for {symbol}: {e}")

                return {
                    'aggregated_signal': aggregated.to_dict(),
                    'mode': mode,
                    'executed': replay_executed,
                    'brokerage': 'replay_paper',
                    'timeframe_signals': {
                        k: v.to_dict() for k, v in timeframe_signals.items()
                    },
                    'reason': 'replay_mode',
                    'signal_id': replay_signal_id,
                    'strategy': None,
                    'strategy_approved': False,
                    'entry_price': entry_price
                }

            # 7. STRATEGY EVALUATION - Check all strategies before trading
            best_strategy = self._evaluate_strategies_for_trade(
                symbol=symbol,
                interval=primary_interval,
                direction=aggregated.direction,
                confidence=aggregated.confidence,
                regime=aggregated.regime
            )

            # 8. EXECUTE TRADE (only if signal exists AND strategy approves)
            executed = False
            signal_id = None
            perf_signal_id = None
            entry_price = candle.close if hasattr(candle, 'close') else None
            strategy_approved = best_strategy is not None and best_strategy.get('is_recommended', False)

            logger.info(
                f"[{symbol} @ {interval}] Decision: direction={aggregated.direction}, "
                f"confidence={aggregated.confidence:.1%}, mode={mode}"
            )

            if aggregated.direction != 'NEUTRAL':
                # In LEARNING mode, always execute paper trades for training data
                # In TRADING mode, trade if strategy is recommended OR no strategies exist yet (fresh start)
                no_strategies_yet = best_strategy is None
                should_execute = (mode == 'LEARNING') or (mode == 'TRADING' and (strategy_approved or no_strategies_yet))

                if should_execute:
                    # Generate perf_signal_id BEFORE executing trade
                    if self.performance_learner and entry_price:
                        perf_signal_id = f"perf_{symbol}_{interval}_{int(datetime.utcnow().timestamp() * 1000)}"

                    signal_id = self._execute_trade(
                        brokerage=brokerage,
                        symbol=symbol,
                        interval=interval,
                        prediction=aggregated,
                        is_paper=(mode == 'LEARNING'),
                        strategy_name=best_strategy.get('strategy_name') if best_strategy else None,
                        perf_signal_id=perf_signal_id,
                        entry_price=entry_price,
                        stop_loss=sl,
                        take_profit=tp
                    )
                    executed = True

                    # Track prediction in PerformanceBasedLearner (AFTER trade executed)
                    if self.performance_learner and perf_signal_id and entry_price:
                        try:
                            self.performance_learner.on_prediction_made(
                                signal_id=perf_signal_id,
                                symbol=symbol,
                                interval=interval,
                                direction=aggregated.direction,
                                confidence=aggregated.confidence,
                                entry_price=entry_price
                            )
                        except Exception as e:
                            logger.warning(f"PerformanceBasedLearner.on_prediction_made failed: {e}")

                    strategy_info = f" [Strategy: {best_strategy['strategy_name']}]" if best_strategy else ""
                    logger.info(
                        f"[{symbol}] {mode} MODE: {aggregated.direction} @ "
                        f"{aggregated.confidence:.2%} ({brokerage_type}){strategy_info}"
                    )

                    with self._stats_lock:
                        self._stats['trades_executed'] += 1
                        if mode == 'LEARNING':
                            self._stats['paper_trades'] += 1
                        else:
                            self._stats['live_trades'] += 1
                else:
                    logger.info(
                        f"[{symbol}] Trade SKIPPED: {aggregated.direction} @ "
                        f"{aggregated.confidence:.2%} - No recommended strategy "
                        f"(best: {best_strategy.get('strategy_name', 'None') if best_strategy else 'None'})"
                    )

            # 9. STORE LATEST PREDICTION (for opposite signal detection)
            with self._predictions_lock:
                self._latest_predictions[symbol] = aggregated

            # Periodic cleanup of stale predictions (every 100 candles)
            if self.performance_learner:
                with self._stats_lock:
                    candles_processed = self._stats['candles_processed']
                if candles_processed % self.cl_config.get('stale_cleanup_interval', 100) == 0:
                    try:
                        self.performance_learner.cleanup_stale_predictions(
                            max_age_minutes=self.cl_config.get('prediction_max_age_minutes', 60)
                        )
                    except Exception as e:
                        logger.debug(f"Performance learner cleanup failed: {e}")

            # Trade closing is handled by StrategicLearningBridge._check_and_close_trades()

            # Record prediction
            with self._stats_lock:
                self._stats['predictions_made'] += 1

            return {
                'aggregated_signal': aggregated.to_dict(),
                'mode': mode,
                'executed': executed,
                'brokerage': brokerage_type,
                'timeframe_signals': {
                    k: v.to_dict() for k, v in timeframe_signals.items()
                },
                'reason': reason,
                'signal_id': signal_id,
                'strategy': best_strategy,
                'strategy_approved': strategy_approved,
                'entry_price': entry_price
            }

        except Exception as e:
            logger.error(
                f"[{symbol} @ {interval}] Error in on_candle_closed: {e}",
                exc_info=True
            )
            return {
                'error': str(e),
                'mode': 'LEARNING',
                'executed': False
            }

    def _get_all_timeframe_predictions(
        self,
        symbol: str,
        data: dict = None
    ) -> Dict[str, TimeframeSignal]:
        """
        Get predictions from all enabled timeframes.

        Args:
            symbol: Trading pair
            data: Pre-fetched data (optional)

        Returns:
            Dict mapping interval to TimeframeSignal
        """
        predictions = {}

        for interval in self.enabled_intervals:
            try:
                # Get data for this timeframe
                if data and data.get('interval') == interval:
                    df = data.get('candles')
                else:
                    # Fetch from database — live_only=True ensures predictions are
                    # always based on real current market data, never old backfill.
                    sequence_length = self._get_sequence_length(interval)
                    df = self.database.get_candles(
                        symbol=symbol,
                        interval=interval,
                        limit=sequence_length + 100,  # Extra for feature calculation
                        live_only=True,
                        live_days=14  # 14 days of live candles for feature context
                    )

                if df is None or len(df) < 10:
                    logger.warning(
                        f"Insufficient data for {symbol} @ {interval}: {len(df) if df is not None else 0} candles"
                    )
                    continue

                # Get LSTM probability from model manager if available
                lstm_prob = 0.5  # Default if no LSTM model
                if hasattr(self.predictor, 'model_manager') and self.predictor.model_manager:
                    model_mgr = self.predictor.model_manager
                    model = model_mgr.loaded_models.get(symbol)
                    if model is not None:
                        try:
                            from src.analysis_engine import FeatureCalculator
                            import torch
                            import numpy as np

                            df_features = FeatureCalculator.calculate_all(df)
                            feature_cols = FeatureCalculator.get_feature_columns()
                            features = df_features[feature_cols].values

                            # Get normalization params from model config
                            model_config = model_mgr.model_configs.get(symbol, {})
                            means = model_config.get('feature_means')
                            stds = model_config.get('feature_stds')
                            seq_len = model_config.get('config', {}).get('sequence_length', 60)

                            if means is not None and stds is not None:
                                features = (features - means) / (stds + 1e-8)
                                features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

                                if len(features) >= seq_len:
                                    x = torch.FloatTensor(features[-seq_len:]).unsqueeze(0)
                                    model.eval()
                                    with torch.no_grad():
                                        lstm_prob = model(x).item()
                                    logger.debug(f"[{symbol}@{interval}] LSTM prob={lstm_prob:.4f} (seq_len={seq_len}, features={features.shape})")
                                else:
                                    logger.warning(f"[{symbol}@{interval}] LSTM skipped: not enough features ({len(features)} < {seq_len})")
                            else:
                                logger.warning(f"[{symbol}@{interval}] LSTM skipped: no normalization params (means={means is not None}, stds={stds is not None})")
                        except Exception as e:
                            logger.warning(f"LSTM prediction failed for {symbol}: {e}")
                    else:
                        logger.debug(f"[{symbol}@{interval}] No LSTM model loaded")
                else:
                    logger.debug(f"[{symbol}@{interval}] No model_manager on predictor")

                # Blend LSTM + boosted prediction (same logic as MultiCurrencySystem.predict)
                has_boost = False
                boost_prob = 0.5
                if self.boosted_predictor is not None:
                    try:
                        if self.boosted_predictor.is_symbol_fitted(symbol):
                            boost_prob = self.boosted_predictor.predict(df, symbol)
                            has_boost = True
                    except Exception as e:
                        logger.debug(f"Boosted prediction failed for {symbol}: {e}")

                from src.core.constants import blend_probabilities
                boost_cfg = self.config.get('boosting', {})
                has_lstm = lstm_prob != 0.5
                combined_prob = blend_probabilities(
                    lstm_prob=lstm_prob, boost_prob=boost_prob,
                    has_lstm=has_lstm, has_boost=has_boost,
                    boost_weight=boost_cfg.get('ensemble_weight', 0.6),
                )

                # Get prediction from model
                result = self.predictor.predict(
                    df=df,
                    symbol=symbol,
                    interval=interval,
                    lstm_probability=combined_prob
                )

                # Create TimeframeSignal
                signal = TimeframeSignal(
                    interval=interval,
                    direction=result.direction,
                    confidence=result.confidence,
                    lstm_prob=result.lstm_probability,
                    advanced_result=result,
                    timestamp=datetime.utcnow()
                )

                predictions[interval] = signal

                logger.debug(
                    f"[{symbol} @ {interval}] Prediction: {result.direction} @ "
                    f"{result.confidence:.2%}"
                )

            except Exception as e:
                logger.error(
                    f"Failed to get prediction for {symbol} @ {interval}: {e}",
                    exc_info=True
                )
                continue

        return predictions

    def _get_sequence_length(self, interval: str) -> int:
        """Get sequence length for interval from config (O(1) cached lookup)."""
        return self._sequence_lengths.get(interval, 60)

    def _get_primary_interval(self) -> str:
        """Get highest weighted interval."""
        intervals = self.timeframes_config.get('intervals', [])

        if not intervals:
            return '1h'  # Default

        # Find max weight
        primary = max(
            [i for i in intervals if i.get('enabled', True)],
            key=lambda x: x.get('weight', 0.0)
        )

        return primary['interval']

    def _handle_mode_transition(
        self,
        symbol: str,
        interval: str,
        new_mode: str,
        confidence: float,
        reason: str
    ):
        """Handle mode transition (LEARNING <-> TRADING)."""
        if new_mode == 'TRADING':
            self.state_manager.transition_to_trading(
                symbol=symbol,
                interval=interval,
                confidence=confidence
            )
            logger.info(
                f"✓ [{symbol} @ {interval}] Transitioned to TRADING mode "
                f"(confidence: {confidence:.2%})"
            )
        else:
            self.state_manager.transition_to_learning(
                symbol=symbol,
                interval=interval,
                reason=reason
            )
            logger.info(
                f"← [{symbol} @ {interval}] Transitioned to LEARNING mode "
                f"(reason: {reason})"
            )

        with self._stats_lock:
            self._stats['mode_transitions'] += 1

    def _evaluate_strategies_for_trade(
        self,
        symbol: str,
        interval: str,
        direction: str,
        confidence: float,
        regime: str
    ) -> Optional[dict]:
        """
        Evaluate all strategies and determine if we should trade.

        This method:
        1. Analyzes historical trade outcomes to discover strategies
        2. Ranks strategies by Sharpe ratio
        3. Saves strategy performance to database
        4. Returns the best strategy if it's recommended for trading

        Args:
            symbol: Trading pair
            interval: Timeframe
            direction: Predicted direction ('BUY', 'SELL', 'NEUTRAL')
            confidence: Model confidence
            regime: Current market regime

        Returns:
            Best strategy dict with is_recommended flag, or None
        """
        # If strategy analyzer not available, skip strategy-based filtering
        if self.strategy_analyzer is None:
            logger.debug("Strategy analyzer not available, skipping strategy evaluation")
            return None

        try:
            # Check if we need to refresh strategy analysis (every hour)
            key = f"{symbol}_{interval}"
            now = datetime.utcnow()

            # Thread-safe check for cache freshness
            with self._strategy_cache_lock:
                last_analysis = self._last_strategy_analysis.get(key)
                should_refresh = (
                    last_analysis is None or
                    (now - last_analysis).total_seconds() > 3600  # 1 hour
                )

                # If another thread is already refreshing, use existing cache
                if should_refresh and key in self._strategy_cache:
                    # Return cached while another thread refreshes
                    cached = self._strategy_cache.get(key)
                    if cached:
                        # Another thread might be refreshing, just use cache
                        pass

            if should_refresh:
                logger.info(f"[{symbol} @ {interval}] Refreshing strategy analysis...")

                # Discover strategies from historical data
                strategies = self.strategy_analyzer.discover_strategies(lookback_days=365)

                if strategies:
                    # Rank strategies
                    ranked = self.strategy_analyzer.rank_strategies(by='sharpe')

                    # Save all strategies to database
                    for strategy_name, strategy in strategies.items():
                        is_recommended = (
                            strategy.sharpe_ratio > 1.0 and
                            strategy.win_rate > 0.50 and
                            strategy.total_trades >= 10
                        )

                        recommendation = self.strategy_analyzer._get_recommendation(strategy)

                        self.database.save_strategy_performance(
                            strategy_name=strategy_name,
                            symbol=symbol,
                            interval=interval,
                            metrics={
                                'total_trades': strategy.total_trades,
                                'win_rate': strategy.win_rate,
                                'avg_profit_pct': strategy.avg_profit_pct,
                                'avg_loss_pct': strategy.avg_loss_pct,
                                'profit_factor': strategy.profit_factor,
                                'sharpe_ratio': strategy.sharpe_ratio,
                                'max_drawdown_pct': strategy.max_drawdown_pct,
                                'avg_holding_hours': strategy.avg_holding_hours,
                                'best_regime': strategy.best_regime,
                                'confidence_threshold': strategy.confidence_threshold,
                                'is_recommended': is_recommended,
                                'recommendation': recommendation
                            }
                        )

                    logger.info(
                        f"[{symbol} @ {interval}] Saved {len(strategies)} strategies to database"
                    )

                    # Cache the best strategy
                    if ranked:
                        best_name, best_strategy = ranked[0]
                        is_rec = (
                            best_strategy.sharpe_ratio > 1.0 and
                            best_strategy.win_rate > 0.50 and
                            best_strategy.total_trades >= 10
                        )

                        with self._strategy_cache_lock:
                            self._strategy_cache[key] = {
                                'strategy_name': best_name,
                                'sharpe_ratio': best_strategy.sharpe_ratio,
                                'win_rate': best_strategy.win_rate,
                                'profit_factor': best_strategy.profit_factor,
                                'total_trades': best_strategy.total_trades,
                                'best_regime': best_strategy.best_regime,
                                'is_recommended': is_rec,
                                'recommendation': self.strategy_analyzer._get_recommendation(best_strategy)
                            }

                # Update timestamp inside the lock
                self._last_strategy_analysis[key] = now

            # Get cached best strategy (already within lock scope above for refresh case)
            with self._strategy_cache_lock:
                best_strategy = self._strategy_cache.get(key)

            # If no strategy found, try from database
            if not best_strategy:
                db_strategy = self.database.get_best_strategy(symbol, interval)
                if db_strategy:
                    best_strategy = dict(db_strategy)

            return best_strategy

        except Exception as e:
            logger.error(f"Strategy evaluation failed: {e}", exc_info=True)
            return None

    def _execute_trade(
        self,
        brokerage: any,
        symbol: str,
        interval: str,
        prediction: AggregatedSignal,
        is_paper: bool,
        strategy_name: str = None,
        perf_signal_id: str = None,
        entry_price: float = None,
        stop_loss: float = None,
        take_profit: float = None
    ) -> Optional[int]:
        """
        Execute trade via brokerage.

        Args:
            brokerage: Brokerage instance
            symbol: Trading pair
            interval: Timeframe
            prediction: Aggregated prediction
            is_paper: Whether this is a paper trade
            strategy_name: Name of the strategy being used
            perf_signal_id: Performance learner signal ID for tracking
            entry_price: Entry price for the trade

        Returns:
            Signal ID or None
        """
        try:
            # Map direction to SignalType
            direction_map = {'BUY': SignalType.BUY, 'SELL': SignalType.SELL}
            signal_type = direction_map.get(prediction.direction, SignalType.NEUTRAL)

            # Determine strength from confidence (thresholds from config signals section)
            if prediction.confidence >= self._strong_signal_threshold:
                strength = SignalStrength.STRONG
            elif prediction.confidence >= self._medium_signal_threshold:
                strength = SignalStrength.MEDIUM
            else:
                strength = SignalStrength.WEAK

            # Create proper Signal object
            signal = Signal(
                timestamp=prediction.timestamp,
                signal_type=signal_type,
                strength=strength,
                confidence=prediction.confidence,
                price=entry_price or 0.0,
                stop_loss=stop_loss,
                take_profit=take_profit,
                actual_outcome='PENDING'
            )

            # Cooldown guard: prevent duplicate signals within half the interval period.
            # Thread-safe: entire check-and-update is atomic under _cooldown_lock.
            from src.core.constants import INTERVAL_MINUTES
            _candle_minutes = INTERVAL_MINUTES.get(interval, 60)
            _cooldown_minutes = max(1, math.floor(_candle_minutes / 2))  # minimum 1 min
            _cooldown_key = f"{symbol}_{interval}"
            _now = datetime.utcnow()
            with self._cooldown_lock:
                _last = self._last_signal_time.get(_cooldown_key)
                if _last is not None:
                    _elapsed = (_now - _last).total_seconds() / 60
                    if _elapsed < _cooldown_minutes:
                        logger.debug(
                            f"[{symbol} @ {interval}] Duplicate signal suppressed: "
                            f"{_elapsed:.1f}m < cooldown {_cooldown_minutes}m"
                        )
                        return None
                self._last_signal_time[_cooldown_key] = _now

            signal_id = self.database.save_signal(signal, symbol=symbol, interval=interval)

            # Update brokerage price before executing
            if entry_price and hasattr(brokerage, 'update_price'):
                brokerage.update_price(symbol, entry_price)

            # Execute via brokerage
            if hasattr(brokerage, 'execute_signal'):
                brokerage.execute_signal(
                    symbol=symbol,
                    signal=prediction.direction,
                    confidence=prediction.confidence,
                    signal_id=signal_id
                )

            return signal_id

        except Exception as e:
            logger.error(f"Failed to execute trade: {e}", exc_info=True)
            return None

    # Trade closing is handled by StrategicLearningBridge._check_and_close_trades()

    def _schedule_retrain(
        self,
        symbol: str,
        interval: str,
        reason: str
    ):
        """
        Schedule background retraining.

        Args:
            symbol: Trading pair
            interval: Timeframe
            reason: Trigger reason
        """
        key = f"{symbol}_{interval}"

        # Skip retraining during historical replay
        if self.replay_mode:
            return

        with self._threads_lock:
            # Check if already retraining
            if key in self.retraining_threads and self.retraining_threads[key].is_alive():
                logger.debug(f"Retraining already in progress for {key}")
                return

            def retrain_task():
                try:
                    logger.info(f"⚙ [{key}] Starting retraining (reason: {reason})")

                    result = self.retraining_engine.retrain(
                        symbol=symbol,
                        interval=interval,
                        trigger_reason=reason
                    )

                    if result['success']:
                        # Transition to TRADING mode
                        self.state_manager.transition_to_trading(
                            symbol=symbol,
                            interval=interval,
                            confidence=result['validation_confidence']
                        )

                        logger.info(
                            f"✓ [{key}] Retraining successful! "
                            f"Confidence: {result['validation_confidence']:.2%}, "
                            f"Duration: {result['duration_seconds']:.1f}s"
                        )
                    else:
                        logger.warning(
                            f"[{key}] Retraining completed but confidence still low: "
                            f"{result.get('validation_confidence', 0):.2%}"
                        )

                except Exception as e:
                    logger.error(f"[{key}] Retraining failed: {e}", exc_info=True)

            # Start background thread
            thread = threading.Thread(
                target=retrain_task,
                daemon=True,
                name=f"Retrain-{key}"
            )
            self.retraining_threads[key] = thread
            thread.start()

            with self._stats_lock:
                self._stats['retrainings_triggered'] += 1

        logger.info(f"[{key}] Retraining scheduled (reason: {reason})")

    def _online_update(
        self,
        symbol: str,
        interval: str,
        candle: any,
        data: dict = None
    ):
        """
        Check for completed trades and perform online learning update with confirmed outcomes.

        Online learning requires actual outcomes to avoid information leakage.
        This method checks if we have pending trades that can now be evaluated,
        and only then performs the online update.

        Args:
            symbol: Trading pair
            interval: Timeframe
            candle: Latest candle
            data: Pre-fetched data (optional)
        """
        try:
            # Get online learning config
            online_config = self.cl_config.get('online_learning', {})

            if not online_config.get('enabled', True):
                return

            if not hasattr(self.predictor, 'online_update'):
                return

            # Get recent confirmed outcomes for this symbol/interval
            # These are outcomes that have been recorded with actual price movements
            recent_outcomes = self.outcome_tracker.get_performance_summary(
                symbol=symbol,
                interval=interval,
                limit=5
            ).get('recent_outcomes', [])

            if not recent_outcomes:
                # Fallback: use price movement vs previous close for online learning
                # This matches the LSTM training target: next_close > current_close
                # Track previous close per (symbol, interval) to align targets
                prev_close_key = f"{symbol}_{interval}"
                if hasattr(candle, 'close') and candle.close != 0:
                    prev_close = self._prev_close.get(prev_close_key)
                    # Always update prev_close for next iteration
                    self._prev_close[prev_close_key] = candle.close

                    if prev_close is not None:
                        # Require minimum move to filter noise
                        price_change_pct = abs(candle.close - prev_close) / prev_close
                        min_pct = self.cl_config.get('online_learning', {}).get('min_price_change_pct', 0.001)
                        if price_change_pct < min_pct:
                            return  # Skip noisy candles with <0.1% move
                        # Target: did this candle close higher than previous close?
                        # Matches training: y = 1 if next_close > current_close
                        actual_outcome = 1 if candle.close > prev_close else 0

                        if data and data.get('candles') is not None:
                            df = data['candles']
                        else:
                            seq_length = self._get_sequence_length(interval)
                            df = self.database.get_candles(
                                symbol=symbol,
                                interval=interval,
                                limit=seq_length + 10,
                                live_only=True,
                                live_days=7
                            )

                        if df is not None and len(df) >= 10:
                            self.predictor.online_update(
                                df=df,
                                symbol=symbol,
                                interval=interval,
                                learning_rate=online_config.get('learning_rate', 0.0001),
                                actual_outcome=actual_outcome
                            )
                            with self._stats_lock:
                                self._stats['online_updates'] = self._stats.get('online_updates', 0) + 1
                return

            # Only update with confirmed outcomes that haven't been used yet
            latest_outcome = recent_outcomes[0] if recent_outcomes else None
            if latest_outcome is None:
                return

            # Get outcome ID for tracking (avoid duplicate updates)
            outcome_id = latest_outcome.get('id') or latest_outcome.get('signal_id') or latest_outcome.get('outcome_id')
            key = f"{symbol}_{interval}"

            with self._outcome_tracking_lock:
                if self._last_online_update_outcome_id.get(key) == outcome_id:
                    # Already processed this outcome
                    return
                self._last_online_update_outcome_id[key] = outcome_id

            # Get the actual price direction (1 if UP, 0 if DOWN)
            # Use actual_direction from outcome data directly — avoids circular feedback
            # where the model learns from its own prediction correctness
            actual_direction = latest_outcome.get('actual_direction', None)
            if actual_direction == 'BUY':  # Price went up
                actual_outcome = 1
            elif actual_direction == 'SELL':  # Price went down
                actual_outcome = 0
            else:
                # Fallback: derive from entry/exit price
                entry_price = latest_outcome.get('entry_price', 0)
                exit_price = latest_outcome.get('exit_price', 0)
                if entry_price > 0 and exit_price > 0:
                    actual_outcome = 1 if exit_price > entry_price else 0
                else:
                    logger.debug(f"Cannot determine actual outcome for {symbol}, skipping")
                    return

            # Get data for features
            if data and data.get('candles') is not None:
                df = data['candles']
            else:
                sequence_length = self._get_sequence_length(interval)
                df = self.database.get_candles(
                    symbol=symbol,
                    interval=interval,
                    limit=sequence_length + 10
                )

            if df is None or len(df) < 10:
                return

            # Perform online update with the actual outcome
            self.predictor.online_update(
                df=df,
                symbol=symbol,
                interval=interval,
                learning_rate=online_config.get('learning_rate', 0.0001),
                actual_outcome=actual_outcome
            )

            with self._stats_lock:
                self._stats['online_updates'] += 1

            logger.debug(
                f"[{symbol} @ {interval}] Online learning update performed "
                f"(actual_direction={actual_direction}, outcome={actual_outcome})"
            )

        except Exception as e:
            logger.error(f"Online update failed: {e}", exc_info=True)

    def get_stats(self) -> dict:
        """Get continuous learning system statistics."""
        with self._stats_lock:
            stats = {
                **self._stats,
                'aggregator_stats': self.signal_aggregator.get_stats(),
                'confidence_gate_stats': self.confidence_gate.get_stats(),
                'outcome_tracker_stats': self.outcome_tracker.get_stats(),
                'retraining_stats': self.retraining_engine.get_stats(),
                'active_retrainings': sum(
                    1 for t in self.retraining_threads.values() if t.is_alive()
                )
            }

        # Add PerformanceBasedLearner stats
        if self.performance_learner:
            try:
                stats['performance_learner_stats'] = self.performance_learner.get_stats()
            except Exception as e:
                logger.debug(f"Failed to get performance_learner stats: {e}")

        return stats

    def stop(self):
        """Stop continuous learning system and wait for retraining threads."""
        logger.info("Stopping ContinuousLearningSystem...")

        # Shutdown PerformanceBasedLearner first (waits for retraining to complete)
        if self.performance_learner:
            try:
                self.performance_learner.shutdown()
                logger.info("PerformanceBasedLearner shut down")
            except Exception as e:
                logger.warning(f"Error shutting down PerformanceBasedLearner: {e}")

        # Wait for active retraining threads
        with self._threads_lock:
            active_threads = [
                t for t in self.retraining_threads.values() if t.is_alive()
            ]

        if active_threads:
            logger.info(f"Waiting for {len(active_threads)} retraining threads...")
            for thread in active_threads:
                thread.join(timeout=30)

        logger.info("ContinuousLearningSystem stopped")
