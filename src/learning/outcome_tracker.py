"""
Outcome Tracker
===============

Tracks trade outcomes and triggers retraining when needed.

Key Features:
- Record trade outcomes (win/loss, PnL)
- Detect performance degradation
- Trigger retraining based on multiple criteria
- Store failed trades for experience replay
- Thread-safe operations

Retraining Triggers:
1. ANY loss when confidence was ≥ 80% (user requirement)
2. 3 consecutive losses
3. Win rate drops below 45%
4. Concept drift detected

NO hardcoded values - all from config
Production-ready with comprehensive logging
"""

import logging
import json
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

from src.core.database import Database

logger = logging.getLogger(__name__)


class OutcomeTracker:
    """
    Tracks trade outcomes and determines when retraining is needed.

    Thread-safe: All database operations are atomic via Database class.

    Integrates with:
    - Database: Persists outcomes to trade_outcomes table
    - Experience Replay: Stores important samples for retraining
    - RetrainingEngine: Triggers retraining when needed
    """

    def __init__(
        self,
        database: Database,
        continual_learner: any = None,
        config: dict = None
    ):
        """
        Initialize outcome tracker.

        Args:
            database: Database instance
            continual_learner: ContinualLearner instance for drift detection
            config: Configuration dict from config.yaml
        """
        self.db = database
        self.continual_learner = continual_learner
        self.config = config or {}

        # Retraining triggers (from config)
        # Note: self.config IS the retraining section (passed from continuous_learner)
        retrain_config = self.config
        self.retrain_on_loss = retrain_config.get('on_loss', False)  # Disabled: fires on every loss, drowns statistical triggers
        self.consecutive_loss_threshold = retrain_config.get('consecutive_loss_threshold', 3)
        self.win_rate_threshold = retrain_config.get('win_rate_threshold', 0.45)
        self.drift_threshold = retrain_config.get('drift_threshold', 0.7)
        # High-confidence loss threshold: retrain when a loss occurs at this confidence level
        # Mirrors the trading_threshold from confidence config so they stay in sync
        self.high_conf_retrain_threshold = retrain_config.get('high_conf_retrain_threshold', 0.80)

        # Experience replay importance weights
        self.replay_loss_importance = retrain_config.get('replay_loss_importance', 2.0)
        self.replay_win_importance = retrain_config.get('replay_win_importance', 1.0)

        # Win rate lookback for retraining trigger
        self.win_rate_lookback = retrain_config.get('win_rate_lookback', 100)

        # Drift detection params
        self.drift_min_samples = retrain_config.get('drift_min_samples', 10)
        self.drift_decay_factor = retrain_config.get('drift_decay_factor', 0.95)
        self.drift_high_conf_threshold = retrain_config.get('drift_high_conf_threshold', 0.7)
        self.drift_high_conf_weight = retrain_config.get('drift_high_conf_weight', 1.5)

        # Experience replay buffer (in-memory for fast access)
        self._replay_buffer: Dict[Tuple[str, str], List[dict]] = {}
        self._buffer_max_size = retrain_config.get('replay_buffer_size', 10000)
        self._replay_lock = threading.Lock()  # Thread safety for replay buffer

        # Drift detection state (sliding window of recent predictions)
        self._drift_window_size = retrain_config.get('drift_window_size', 50)
        self._drift_history: Dict[Tuple[str, str], List[dict]] = {}
        self._drift_lock = threading.Lock()

        # Statistics
        self._stats = {
            'outcomes_recorded': 0,
            'losses_detected': 0,
            'wins_detected': 0,
            'retraining_triggered': 0,
            'drift_detected': 0
        }

        logger.info(
            f"OutcomeTracker initialized: "
            f"retrain_on_loss={self.retrain_on_loss}, "
            f"consecutive_threshold={self.consecutive_loss_threshold}, "
            f"drift_detection={'enabled' if continual_learner else 'disabled'}"
        )

    def record_outcome(
        self,
        signal_id: int,
        symbol: str,
        interval: str,
        entry_price: float,
        exit_price: float,
        predicted_direction: str,
        confidence: float,
        features: np.ndarray = None,
        regime: str = None,
        is_paper_trade: bool = False,
        strategy_name: str = None
    ) -> dict:
        """
        Record trade outcome and check retraining triggers.

        Args:
            signal_id: ID of the signal that generated the trade
            symbol: Trading pair
            interval: Timeframe
            entry_price: Entry price
            exit_price: Exit price
            predicted_direction: 'BUY' or 'SELL'
            confidence: Model confidence (0.0 to 1.0)
            features: Feature vector used for prediction (for replay buffer)
            regime: Market regime at trade time
            is_paper_trade: True if paper trade
            strategy_name: Name of the strategy used for this trade

        Returns:
            {
                'was_correct': bool,
                'pnl_percent': float,
                'should_retrain': bool,
                'trigger_reason': str or None
            }
        """
        self._stats['outcomes_recorded'] += 1

        # Calculate if prediction was correct
        was_correct = self._check_correctness(
            entry_price=entry_price,
            exit_price=exit_price,
            predicted_direction=predicted_direction
        )

        # Calculate PnL
        pnl_percent = ((exit_price - entry_price) / entry_price) * 100

        # Adjust PnL sign for SELL predictions
        if predicted_direction == 'SELL':
            pnl_percent *= -1

        pnl_absolute = exit_price - entry_price
        if predicted_direction == 'SELL':
            pnl_absolute *= -1

        # Determine actual direction
        if exit_price > entry_price:
            actual_direction = 'BUY'  # Price went up
        elif exit_price < entry_price:
            actual_direction = 'SELL'  # Price went down
        else:
            actual_direction = 'NEUTRAL'  # No change

        # Save to database
        try:
            # JSON-encode features for database storage
            features_json = None
            if features is not None:
                try:
                    features_json = json.dumps(features.tolist())
                except (TypeError, ValueError) as e:
                    logger.warning(f"Failed to serialize features: {e}")
                    features_json = None

            outcome_id = self.db.record_trade_outcome(
                signal_id=signal_id,
                symbol=symbol,
                interval=interval,
                entry_price=entry_price,
                exit_price=exit_price,
                entry_time=datetime.utcnow().isoformat(),
                exit_time=datetime.utcnow().isoformat(),
                predicted_direction=predicted_direction,
                predicted_confidence=confidence,
                predicted_probability=confidence,  # Same as confidence for now
                actual_direction=actual_direction,
                was_correct=was_correct,
                pnl_percent=pnl_percent,
                pnl_absolute=pnl_absolute,
                features_snapshot=features_json,
                regime=regime,
                is_paper_trade=is_paper_trade,
                strategy_name=strategy_name
            )

            logger.info(
                f"[{symbol} @ {interval}] Trade outcome recorded: "
                f"{'✓ CORRECT' if was_correct else '✗ INCORRECT'} "
                f"(PnL: {pnl_percent:+.2f}%, confidence: {confidence:.1%})"
                + (" [PAPER]" if is_paper_trade else "")
            )

        except Exception as e:
            logger.error(f"Failed to record outcome: {e}", exc_info=True)
            return {
                'was_correct': was_correct,
                'pnl_percent': pnl_percent,
                'should_retrain': False,
                'trigger_reason': None,
                'error': str(e)
            }

        # Update statistics
        if was_correct:
            self._stats['wins_detected'] += 1
        else:
            self._stats['losses_detected'] += 1

        # Add to experience replay buffer (if features provided)
        if features is not None:
            self._add_to_replay(
                symbol=symbol,
                interval=interval,
                features=features,
                target=float(was_correct),
                importance=self.replay_loss_importance if not was_correct else self.replay_win_importance
            )

        # Check if retraining needed
        # Paper trades in LEARNING mode still need retraining to improve the model
        should_retrain = False
        trigger_reason = None

        should_retrain, trigger_reason = self.check_retraining_triggers(
            symbol=symbol,
            interval=interval,
            was_correct=was_correct,
            confidence=confidence
        )

        return {
            'was_correct': was_correct,
            'pnl_percent': pnl_percent,
            'pnl_absolute': pnl_absolute,
            'actual_direction': actual_direction,
            'should_retrain': should_retrain,
            'trigger_reason': trigger_reason,
            'outcome_id': outcome_id
        }

    def _check_correctness(
        self,
        entry_price: float,
        exit_price: float,
        predicted_direction: str
    ) -> bool:
        """
        Check if prediction was correct.

        Args:
            entry_price: Entry price
            exit_price: Exit price
            predicted_direction: 'BUY' or 'SELL'

        Returns:
            True if prediction matched price movement
        """
        price_change = exit_price - entry_price

        if predicted_direction == 'BUY':
            # Correct if price went up
            return price_change > 0
        elif predicted_direction == 'SELL':
            # Correct if price went down
            return price_change < 0
        else:
            # NEUTRAL or unknown - consider incorrect
            return False

    def check_retraining_triggers(
        self,
        symbol: str,
        interval: str,
        was_correct: bool,
        confidence: float = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if retraining should be triggered.

        Triggers:
        1. ANY loss when confidence ≥ 80% (immediate retrain)
        2. 3 consecutive losses
        3. Win rate < 45% (over last 100 trades)
        4. Concept drift detected (drift_score > 0.7)

        Args:
            symbol: Trading pair
            interval: Timeframe
            was_correct: Was the trade correct?
            confidence: Model confidence (optional)

        Returns:
            (should_retrain: bool, reason: str or None)
        """
        # Trigger 1: Retrain only on high-confidence losses (on_loss=true + conf>=threshold)
        # Low-confidence losses fall through to statistical triggers below.
        # A model that was highly confident and wrong is suspicious; lower confidence
        # losses are expected variance and should not trigger immediate retraining.
        if not was_correct and self.retrain_on_loss:
            if confidence is not None and confidence >= self.high_conf_retrain_threshold:
                self._stats['retraining_triggered'] += 1
                return (True, f"loss_high_confidence (conf={confidence:.1%})")

        # Trigger 2: Consecutive losses
        try:
            recent_outcomes = self.db.get_recent_outcomes(
                symbol=symbol,
                interval=interval,
                limit=self.consecutive_loss_threshold
            )

            if len(recent_outcomes) >= self.consecutive_loss_threshold:
                # Check if all are losses
                all_losses = all(not outcome['was_correct'] for outcome in recent_outcomes)

                if all_losses:
                    self._stats['retraining_triggered'] += 1
                    return (True, f"consecutive_losses ({self.consecutive_loss_threshold})")

        except Exception as e:
            logger.error(f"Failed to check consecutive losses: {e}")

        # Trigger 3: Win rate degradation
        try:
            win_rate = self.db.get_win_rate(
                symbol=symbol,
                interval=interval,
                limit=self.win_rate_lookback
            )

            if win_rate < self.win_rate_threshold:
                self._stats['retraining_triggered'] += 1
                return (True, f"win_rate_low ({win_rate:.1%} < {self.win_rate_threshold:.1%})")

        except Exception as e:
            logger.error(f"Failed to check win rate: {e}")

        # Trigger 4: Concept drift detection
        # Uses sliding window to detect when model accuracy degrades significantly
        if self.continual_learner is not None:
            try:
                drift_detected, drift_score = self._check_concept_drift(
                    symbol=symbol,
                    interval=interval,
                    was_correct=was_correct,
                    confidence=confidence
                )

                if drift_detected:
                    self._stats['retraining_triggered'] += 1
                    self._stats['drift_detected'] += 1
                    return (True, f"concept_drift (score={drift_score:.2f} > {self.drift_threshold:.2f})")

            except Exception as e:
                logger.error(f"Failed to check concept drift: {e}")

        return (False, None)

    def _check_concept_drift(
        self,
        symbol: str,
        interval: str,
        was_correct: bool,
        confidence: float
    ) -> Tuple[bool, float]:
        """
        Detect concept drift using sliding window accuracy monitoring.

        Drift is detected when high-confidence predictions start failing.
        This indicates the underlying market regime has changed.

        Args:
            symbol: Trading pair
            interval: Timeframe
            was_correct: Whether the prediction was correct
            confidence: Prediction confidence

        Returns:
            (drift_detected: bool, drift_score: float)
        """
        key = (symbol, interval)

        with self._drift_lock:
            if key not in self._drift_history:
                self._drift_history[key] = []

            # Add current outcome to history
            self._drift_history[key].append({
                'was_correct': was_correct,
                'confidence': confidence,
                'timestamp': datetime.now(timezone.utc)  # Always use timezone-aware UTC
            })

            # Maintain sliding window
            if len(self._drift_history[key]) > self._drift_window_size:
                self._drift_history[key] = self._drift_history[key][-self._drift_window_size:]

            # Need minimum samples for drift detection
            if len(self._drift_history[key]) < self.drift_min_samples:
                return (False, 0.0)

            # Calculate weighted accuracy (higher weight for recent samples)
            history = self._drift_history[key]
            total_weight = 0.0
            weighted_correct = 0.0

            for i, outcome in enumerate(history):
                # Exponential decay weight (recent samples matter more)
                weight = 1.0 * (self.drift_decay_factor ** (len(history) - i - 1))

                # Extra weight for high-confidence predictions (validate confidence first)
                outcome_confidence = outcome.get('confidence')
                if outcome_confidence is not None and outcome_confidence >= self.drift_high_conf_threshold:
                    weight *= self.drift_high_conf_weight

                total_weight += weight
                if outcome['was_correct']:
                    weighted_correct += weight

            # Calculate drift score (1.0 - accuracy = drift)
            # Handle edge case: if total_weight is 0, cannot determine drift
            if total_weight <= 0:
                logger.warning(
                    f"[{symbol}@{interval}] Invalid total_weight=0 in drift detection, "
                    f"cannot determine drift"
                )
                return (False, 0.0)

            accuracy = weighted_correct / total_weight
            drift_score = 1.0 - accuracy

            # Check against threshold
            drift_detected = drift_score > self.drift_threshold

            if drift_detected:
                logger.warning(
                    f"[{symbol}@{interval}] Concept drift detected: "
                    f"drift_score={drift_score:.2f}, accuracy={accuracy:.2%}, "
                    f"window_size={len(history)}"
                )

            return (drift_detected, drift_score)

    def _add_to_replay(
        self,
        symbol: str,
        interval: str,
        features: np.ndarray,
        target: float,
        importance: float = 1.0
    ):
        """
        Add sample to experience replay buffer.

        Thread-safe: Protected by self._replay_lock

        Args:
            symbol: Trading pair
            interval: Timeframe
            features: Feature vector
            target: Target value (0.0 or 1.0)
            importance: Sample importance (higher = prioritized)
        """
        key = (symbol, interval)

        with self._replay_lock:
            if key not in self._replay_buffer:
                self._replay_buffer[key] = []

            sample = {
                'features': features,
                'target': target,
                'importance': importance,
                'timestamp': datetime.utcnow()
            }

            self._replay_buffer[key].append(sample)

            # Keep buffer size limited
            if len(self._replay_buffer[key]) > self._buffer_max_size:
                # Remove oldest (least important) samples
                # Sort by importance (descending) and timestamp (ascending)
                self._replay_buffer[key].sort(
                    key=lambda x: (-x['importance'], x['timestamp'])
                )
                # Keep most important samples
                self._replay_buffer[key] = self._replay_buffer[key][:self._buffer_max_size]

    def get_replay_buffer(
        self,
        symbol: str,
        interval: str,
        limit: int = None
    ) -> List[dict]:
        """
        Get experience replay buffer samples.

        Thread-safe: Protected by self._replay_lock

        Args:
            symbol: Trading pair
            interval: Timeframe
            limit: Max samples to return (None = all)

        Returns:
            List of sample dicts (copy to prevent external modification)
        """
        key = (symbol, interval)

        with self._replay_lock:
            if key not in self._replay_buffer:
                return []

            samples = self._replay_buffer[key].copy()  # Return copy for safety

            if limit:
                return samples[:limit]

            return samples

    def clear_replay_buffer(self, symbol: str = None, interval: str = None):
        """
        Clear experience replay buffer.

        Args:
            symbol: If specified, clear only this symbol
            interval: If specified (with symbol), clear only that specific model
        """
        if symbol is None:
            # Clear all
            self._replay_buffer.clear()
            logger.info("Cleared all replay buffers")
        elif interval is None:
            # Clear all intervals for symbol
            keys_to_remove = [k for k in self._replay_buffer.keys() if k[0] == symbol]
            for key in keys_to_remove:
                del self._replay_buffer[key]
            logger.info(f"Cleared replay buffer for {symbol} (all intervals)")
        else:
            # Clear specific model
            key = (symbol, interval)
            self._replay_buffer.pop(key, None)
            logger.info(f"Cleared replay buffer for {symbol} @ {interval}")

    def get_stats(self) -> dict:
        """Get outcome tracker statistics."""
        total_outcomes = self._stats['outcomes_recorded']
        if total_outcomes > 0:
            win_rate = self._stats['wins_detected'] / total_outcomes
        else:
            win_rate = 0.0

        return {
            **self._stats,
            'win_rate': win_rate,
            'replay_buffer_size': sum(len(samples) for samples in self._replay_buffer.values()),
            'symbols_tracked': len(self._replay_buffer)
        }

    def get_performance_summary(
        self,
        symbol: str,
        interval: str,
        limit: int = 100
    ) -> dict:
        """
        Get performance summary for (symbol, interval).

        Args:
            symbol: Trading pair
            interval: Timeframe
            limit: Number of recent trades to analyze

        Returns:
            Performance metrics dict
        """
        try:
            outcomes = self.db.get_recent_outcomes(
                symbol=symbol,
                interval=interval,
                limit=limit
            )

            if not outcomes:
                return {
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'avg_pnl': 0.0,
                    'total_pnl': 0.0
                }

            total_trades = len(outcomes)
            wins = sum(1 for o in outcomes if o['was_correct'])
            win_rate = wins / total_trades

            pnls = [o['pnl_percent'] for o in outcomes if o['pnl_percent'] is not None]
            avg_pnl = np.mean(pnls) if pnls else 0.0
            total_pnl = np.sum(pnls) if pnls else 0.0

            return {
                'total_trades': total_trades,
                'wins': wins,
                'losses': total_trades - wins,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'total_pnl': total_pnl,
                'recent_outcomes': outcomes[:10]  # Last 10 for display
            }

        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'total_pnl': 0.0,
                'error': str(e)
            }
