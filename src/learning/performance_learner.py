"""
Performance-Based Continuous Learning System
=============================================

Implements continuous learning based on trade performance:
1. Initial training on 1 year of 15min + 1hr candle data
2. Per-candle feedback:
   - PROFIT: Reinforce current model weights (small positive update)
   - LOSS: Trigger thorough retraining to correct the model

Retraining Levels:
- LIGHT: Quick update on recent data (30 epochs) - minor corrections
- MEDIUM: Standard retraining (50 epochs) - significant drift
- FULL: Complete retraining (100 epochs) - consecutive losses or major failures

This approach ensures:
- Model continuously adapts to current market conditions
- Profitable patterns are reinforced
- Losing patterns are quickly corrected
- No catastrophic forgetting (via EWC)

Author: Traot System
"""

import logging
import threading
import numpy as np
import torch
import torch.nn as nn
from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


class RetrainLevel(Enum):
    """Retraining intensity levels."""
    NONE = "none"           # No retraining needed
    LIGHT = "light"         # Quick update (30 epochs)
    MEDIUM = "medium"       # Standard retraining (50 epochs)
    FULL = "full"           # Complete retraining (100 epochs)


@dataclass
class CandleOutcome:
    """Outcome of a prediction after candle closes."""
    timestamp: datetime
    symbol: str
    interval: str
    predicted_direction: str  # 'BUY' or 'SELL'
    actual_direction: str     # 'UP' or 'DOWN'
    confidence: float
    was_correct: bool
    pnl_percent: float
    entry_price: float
    exit_price: float
    features: Optional[np.ndarray] = None


@dataclass
class LearningState:
    """Current state of the learning system."""
    total_candles_processed: int = 0
    total_wins: int = 0
    total_losses: int = 0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    last_retrain_time: Optional[datetime] = None
    last_retrain_reason: Optional[str] = None
    retrains_triggered: int = 0
    reinforcements_applied: int = 0

    @property
    def win_rate(self) -> float:
        total = self.total_wins + self.total_losses
        return self.total_wins / total if total > 0 else 0.0


@dataclass
class PerformanceLearnerConfig:
    """Configuration for performance-based learning."""
    # Retraining triggers
    loss_retrain_enabled: bool = True           # Retrain on any loss
    consecutive_loss_threshold: int = 3         # Trigger FULL retrain after N consecutive losses
    win_rate_threshold: float = 0.45            # Trigger MEDIUM retrain if win rate drops below
    high_confidence_loss_threshold: float = 0.80  # FULL retrain if loss at this confidence

    # Reinforcement settings
    reinforce_on_win: bool = True               # Apply reinforcement learning on wins
    reinforce_learning_rate: float = 0.0001     # Small LR for reinforcement
    reinforce_epochs: int = 5                   # Quick reinforcement epochs

    # Retraining epochs by level
    light_epochs: int = 30
    medium_epochs: int = 50
    full_epochs: int = 100

    # Data settings
    recent_candles_light: int = 2000            # Candles for LIGHT retraining
    recent_candles_medium: int = 5000           # Candles for MEDIUM retraining
    recent_candles_full: int = 10000            # Candles for FULL retraining

    # Timeframes to train on
    timeframes: List[str] = field(default_factory=lambda: ['15m', '1h'])

    # Performance tracking
    performance_window: int = 100               # Track last N trades for win rate
    cooldown_minutes: int = 30                  # Min time between retrainings


class PerformanceBasedLearner:
    """
    Continuous learning system that adapts based on trade performance.

    Key Principles:
    1. Learn from every candle close
    2. Reinforce successful predictions (strengthen good patterns)
    3. Retrain on failed predictions (correct mistakes)
    4. Multiple retraining levels based on severity

    Thread-safe for real-time trading environments.
    """

    def __init__(
        self,
        predictor,  # UnbreakablePredictor instance
        database,   # Database instance
        config: Optional[PerformanceLearnerConfig] = None
    ):
        """
        Initialize performance-based learner.

        Args:
            predictor: UnbreakablePredictor instance with trained models
            database: Database for fetching candles and storing outcomes
            config: Learning configuration
        """
        self.predictor = predictor
        self.database = database
        self.config = config or PerformanceLearnerConfig()

        # Learning state
        self.state = LearningState()

        # Performance tracking (sliding window)
        self._recent_outcomes: deque = deque(maxlen=self.config.performance_window)
        self._outcomes_lock = threading.Lock()

        # Retraining lock (prevent concurrent retraining)
        self._retrain_lock = threading.Lock()
        self._is_retraining = False

        # Track pending predictions awaiting outcome
        self._pending_predictions: Dict[str, dict] = {}  # signal_id -> prediction_info
        self._pending_lock = threading.Lock()

        # Statistics per timeframe
        self._timeframe_stats: Dict[str, LearningState] = {
            tf: LearningState() for tf in self.config.timeframes
        }

        # Track active retrain thread for graceful shutdown
        self._active_retrain_thread: Optional[threading.Thread] = None

        logger.info(
            f"PerformanceBasedLearner initialized: "
            f"timeframes={self.config.timeframes}, "
            f"loss_retrain={self.config.loss_retrain_enabled}, "
            f"reinforce_on_win={self.config.reinforce_on_win}"
        )

    def on_prediction_made(
        self,
        signal_id: str,
        symbol: str,
        interval: str,
        direction: str,
        confidence: float,
        entry_price: float,
        features: np.ndarray = None
    ):
        """
        Record a prediction for later outcome tracking.

        Args:
            signal_id: Unique ID for this prediction
            symbol: Trading pair
            interval: Timeframe
            direction: Predicted direction ('BUY' or 'SELL')
            confidence: Model confidence
            entry_price: Price at prediction time
            features: Feature vector used (for replay buffer)
        """
        with self._pending_lock:
            self._pending_predictions[signal_id] = {
                'timestamp': datetime.utcnow(),
                'symbol': symbol,
                'interval': interval,
                'direction': direction,
                'confidence': confidence,
                'entry_price': entry_price,
                'features': features
            }

        logger.debug(
            f"[{symbol}@{interval}] Prediction recorded: {direction} @ {confidence:.1%}"
        )

    def on_candle_closed(
        self,
        signal_id: str,
        exit_price: float,
        actual_direction: str = None
    ) -> Dict:
        """
        Process candle close and trigger learning.

        This is the main entry point called when a candle closes.

        Args:
            signal_id: ID of the prediction to evaluate
            exit_price: Closing price of the candle
            actual_direction: Optional actual direction ('UP' or 'DOWN')

        Returns:
            Dict with outcome, learning action taken, and stats
        """
        with self._pending_lock:
            prediction = self._pending_predictions.pop(signal_id, None)

        if prediction is None:
            logger.warning(f"No pending prediction found for signal_id={signal_id}")
            return {'error': 'prediction_not_found'}

        # Determine actual direction from price movement
        if actual_direction is None:
            if exit_price > prediction['entry_price']:
                actual_direction = 'UP'
            elif exit_price < prediction['entry_price']:
                actual_direction = 'DOWN'
            else:
                actual_direction = 'FLAT'

        # Check if prediction was correct
        predicted = prediction['direction']
        was_correct = (
            (predicted == 'BUY' and actual_direction == 'UP') or
            (predicted == 'SELL' and actual_direction == 'DOWN')
        )

        # Calculate PnL
        pnl_pct = ((exit_price - prediction['entry_price']) / prediction['entry_price']) * 100
        if predicted == 'SELL':
            pnl_pct *= -1  # Invert for short positions

        # Create outcome
        outcome = CandleOutcome(
            timestamp=datetime.utcnow(),
            symbol=prediction['symbol'],
            interval=prediction['interval'],
            predicted_direction=predicted,
            actual_direction=actual_direction,
            confidence=prediction['confidence'],
            was_correct=was_correct,
            pnl_percent=pnl_pct,
            entry_price=prediction['entry_price'],
            exit_price=exit_price,
            features=prediction.get('features')
        )

        # Update state
        self._update_state(outcome)

        # Determine learning action
        learning_action = self._determine_learning_action(outcome)

        # Execute learning action
        result = self._execute_learning(outcome, learning_action)

        return {
            'outcome': {
                'was_correct': was_correct,
                'pnl_percent': pnl_pct,
                'predicted': predicted,
                'actual': actual_direction,
                'confidence': prediction['confidence']
            },
            'learning_action': learning_action.value,
            'learning_result': result,
            'state': {
                'win_rate': self.state.win_rate,
                'consecutive_wins': self.state.consecutive_wins,
                'consecutive_losses': self.state.consecutive_losses,
                'total_candles': self.state.total_candles_processed
            }
        }

    def _update_state(self, outcome: CandleOutcome):
        """Update learning state based on outcome."""
        with self._outcomes_lock:
            self._recent_outcomes.append(outcome)
            self.state.total_candles_processed += 1

            if outcome.was_correct:
                self.state.total_wins += 1
                self.state.consecutive_wins += 1
                self.state.consecutive_losses = 0

                if self.state.consecutive_wins > self.state.max_consecutive_wins:
                    self.state.max_consecutive_wins = self.state.consecutive_wins
            else:
                self.state.total_losses += 1
                self.state.consecutive_losses += 1
                self.state.consecutive_wins = 0

                if self.state.consecutive_losses > self.state.max_consecutive_losses:
                    self.state.max_consecutive_losses = self.state.consecutive_losses

            # Update timeframe-specific stats
            interval = outcome.interval
            if interval in self._timeframe_stats:
                tf_state = self._timeframe_stats[interval]
                tf_state.total_candles_processed += 1

                if outcome.was_correct:
                    tf_state.total_wins += 1
                    tf_state.consecutive_wins += 1
                    tf_state.consecutive_losses = 0
                else:
                    tf_state.total_losses += 1
                    tf_state.consecutive_losses += 1
                    tf_state.consecutive_wins = 0

        status = "✓ WIN" if outcome.was_correct else "✗ LOSS"
        logger.info(
            f"[{outcome.symbol}@{outcome.interval}] {status}: "
            f"{outcome.predicted_direction} (conf={outcome.confidence:.1%}) -> "
            f"{outcome.actual_direction} | PnL: {outcome.pnl_percent:+.2f}% | "
            f"WR: {self.state.win_rate:.1%} | "
            f"Streak: {'+' + str(self.state.consecutive_wins) if outcome.was_correct else '-' + str(self.state.consecutive_losses)}"
        )

    def _determine_learning_action(self, outcome: CandleOutcome) -> RetrainLevel:
        """
        Determine what learning action to take based on outcome.

        Decision tree:
        1. WIN + config.reinforce_on_win → REINFORCE (not retrain)
        2. LOSS at HIGH confidence → FULL retrain
        3. LOSS with consecutive_losses >= threshold → FULL retrain
        4. LOSS with dropping win rate → MEDIUM retrain
        5. LOSS (any other) → LIGHT retrain

        Also checks cooldown to prevent too frequent retraining.
        """
        # Check cooldown
        if self.state.last_retrain_time:
            elapsed = (datetime.utcnow() - self.state.last_retrain_time).total_seconds()
            if elapsed < self.config.cooldown_minutes * 60:
                remaining = self.config.cooldown_minutes - (elapsed / 60)
                logger.debug(f"Cooldown active: {remaining:.1f} minutes remaining")
                return RetrainLevel.NONE

        # WIN case: No retraining, but may reinforce
        if outcome.was_correct:
            # Reinforcement is handled separately in _execute_learning
            return RetrainLevel.NONE

        # LOSS cases (only if loss retraining is enabled)
        if not self.config.loss_retrain_enabled:
            return RetrainLevel.NONE

        # High confidence loss → FULL retrain immediately
        if outcome.confidence >= self.config.high_confidence_loss_threshold:
            logger.warning(
                f"High confidence loss detected: {outcome.confidence:.1%} >= "
                f"{self.config.high_confidence_loss_threshold:.1%} → FULL retrain"
            )
            return RetrainLevel.FULL

        # Consecutive losses → FULL retrain
        if self.state.consecutive_losses >= self.config.consecutive_loss_threshold:
            logger.warning(
                f"Consecutive losses: {self.state.consecutive_losses} >= "
                f"{self.config.consecutive_loss_threshold} → FULL retrain"
            )
            return RetrainLevel.FULL

        # Check win rate degradation → MEDIUM retrain
        if self.state.win_rate < self.config.win_rate_threshold:
            logger.warning(
                f"Win rate dropped: {self.state.win_rate:.1%} < "
                f"{self.config.win_rate_threshold:.1%} → MEDIUM retrain"
            )
            return RetrainLevel.MEDIUM

        # Default: LIGHT retrain on any loss
        return RetrainLevel.LIGHT

    def _execute_learning(
        self,
        outcome: CandleOutcome,
        retrain_level: RetrainLevel
    ) -> Dict:
        """
        Execute the learning action.

        Args:
            outcome: The candle outcome
            retrain_level: Level of retraining to perform

        Returns:
            Result dictionary
        """
        result = {'action': 'none', 'success': True}

        # Handle WIN: Apply reinforcement
        if outcome.was_correct and self.config.reinforce_on_win:
            result = self._apply_reinforcement(outcome)
            result['action'] = 'reinforcement'
            return result

        # Handle LOSS: Retrain based on level
        if retrain_level == RetrainLevel.NONE:
            return result

        # Thread-safe check if already retraining (prevent race condition)
        with self._retrain_lock:
            if self._is_retraining:
                logger.debug("Skipping retrain: already in progress")
                return {'action': 'skipped', 'reason': 'already_retraining'}
            # Set flag atomically while holding lock
            self._is_retraining = True

        # Execute retraining in background (flag already set)
        thread = threading.Thread(
            target=self._retrain_async,
            args=(outcome, retrain_level),
            daemon=False,  # Non-daemon to allow graceful shutdown
            name=f"Retrain-{retrain_level.value}"
        )
        thread.start()

        # Track thread for graceful shutdown
        self._active_retrain_thread = thread

        result['action'] = f'retrain_{retrain_level.value}'
        result['async'] = True

        return result

    def _apply_reinforcement(self, outcome: CandleOutcome) -> Dict:
        """
        Apply reinforcement learning for successful predictions.

        Reinforcement:
        - Small learning rate to avoid overshooting
        - Few epochs to nudge weights slightly
        - Uses the exact features that led to success

        This strengthens the patterns that led to correct predictions.
        """
        if outcome.features is None:
            logger.debug("No features available for reinforcement")
            return {'success': False, 'reason': 'no_features'}

        try:
            # Get the model
            if not hasattr(self.predictor, 'tcn_lstm_model') or self.predictor.tcn_lstm_model is None:
                logger.debug("No neural network model available for reinforcement")
                return {'success': False, 'reason': 'no_nn_model'}

            model = self.predictor.tcn_lstm_model

            # Prepare data (single sample reinforcement)
            features = torch.FloatTensor(outcome.features).unsqueeze(0)

            # Target: 1 for BUY that was correct, 0 for SELL that was correct
            target = torch.FloatTensor([1.0 if outcome.predicted_direction == 'BUY' else 0.0])

            # Small learning rate optimizer for gentle reinforcement
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config.reinforce_learning_rate
            )
            criterion = nn.BCELoss()

            # Quick reinforcement epochs
            model.train()
            for _ in range(self.config.reinforce_epochs):
                optimizer.zero_grad()
                output = model(features).squeeze()
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            model.eval()

            self.state.reinforcements_applied += 1

            logger.debug(
                f"[{outcome.symbol}@{outcome.interval}] Reinforcement applied: "
                f"{self.config.reinforce_epochs} epochs @ LR={self.config.reinforce_learning_rate}"
            )

            return {
                'success': True,
                'epochs': self.config.reinforce_epochs,
                'learning_rate': self.config.reinforce_learning_rate
            }

        except Exception as e:
            logger.error(f"Reinforcement failed: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def _retrain_async(self, outcome: CandleOutcome, level: RetrainLevel):
        """
        Perform retraining asynchronously.

        Retraining levels:
        - LIGHT: 30 epochs, 2000 recent candles
        - MEDIUM: 50 epochs, 5000 recent candles
        - FULL: 100 epochs, 10000 recent candles

        Note: _is_retraining flag is set by caller before thread starts
        """
        try:
            # Determine parameters based on level
            if level == RetrainLevel.LIGHT:
                epochs = self.config.light_epochs
                candles = self.config.recent_candles_light
                reason = "loss_light"
            elif level == RetrainLevel.MEDIUM:
                epochs = self.config.medium_epochs
                candles = self.config.recent_candles_medium
                reason = "win_rate_degradation"
            else:  # FULL
                epochs = self.config.full_epochs
                candles = self.config.recent_candles_full
                reason = f"consecutive_losses_{self.state.consecutive_losses}"

            logger.info(
                f"⚙ [{outcome.symbol}@{outcome.interval}] Starting {level.value.upper()} retrain: "
                f"{epochs} epochs, {candles} candles | Reason: {reason}"
            )

            start_time = datetime.utcnow()

            # Fetch training data for all configured timeframes
            for interval in self.config.timeframes:
                df = self.database.get_candles(
                    symbol=outcome.symbol,
                    interval=interval,
                    limit=candles + 200  # Extra for indicators
                )

                if df is None or len(df) < 500:
                    logger.warning(
                        f"Insufficient data for {interval}: {len(df) if df is not None else 0}"
                    )
                    continue

                # Retrain predictor
                self.predictor.fit(
                    df=df,
                    epochs=epochs,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=False
                )

            duration = (datetime.utcnow() - start_time).total_seconds()

            # Update state
            self.state.last_retrain_time = datetime.utcnow()
            self.state.last_retrain_reason = reason
            self.state.retrains_triggered += 1

            # Reset consecutive losses after retraining (thread-safe)
            with self._outcomes_lock:
                self.state.consecutive_losses = 0

            logger.info(
                f"✓ [{outcome.symbol}@{outcome.interval}] {level.value.upper()} retrain complete: "
                f"{duration:.1f}s | Total retrains: {self.state.retrains_triggered}"
            )

        except Exception as e:
            logger.error(f"Retraining failed: {e}", exc_info=True)

        finally:
            with self._retrain_lock:
                self._is_retraining = False

    def get_stats(self) -> Dict:
        """Get learning statistics (thread-safe)."""
        with self._outcomes_lock:
            recent_wins = sum(1 for o in self._recent_outcomes if o.was_correct)
            recent_total = len(self._recent_outcomes)
            recent_win_rate = recent_wins / recent_total if recent_total > 0 else 0.0

            # Capture all state under lock to prevent race conditions
            timeframe_stats = {
                tf: {
                    'candles': state.total_candles_processed,
                    'win_rate': state.win_rate,
                    'consecutive_losses': state.consecutive_losses
                }
                for tf, state in self._timeframe_stats.items()
            }

            return {
                'total_candles': self.state.total_candles_processed,
                'total_wins': self.state.total_wins,
                'total_losses': self.state.total_losses,
                'overall_win_rate': self.state.win_rate,
                'recent_win_rate': recent_win_rate,
                'consecutive_wins': self.state.consecutive_wins,
                'consecutive_losses': self.state.consecutive_losses,
                'max_consecutive_wins': self.state.max_consecutive_wins,
                'max_consecutive_losses': self.state.max_consecutive_losses,
                'retrains_triggered': self.state.retrains_triggered,
                'reinforcements_applied': self.state.reinforcements_applied,
                'last_retrain': self.state.last_retrain_time.isoformat() if self.state.last_retrain_time else None,
                'last_retrain_reason': self.state.last_retrain_reason,
                'is_retraining': self._is_retraining,
                'timeframe_stats': timeframe_stats
            }

    def get_recommended_action(self) -> str:
        """Get recommended action based on current state."""
        if self.state.consecutive_losses >= self.config.consecutive_loss_threshold:
            return "PAUSE_TRADING"

        if self.state.win_rate < 0.40:
            return "REDUCE_POSITION_SIZE"

        if self.state.consecutive_wins >= 5 and self.state.win_rate > 0.60:
            return "NORMAL_TRADING"

        return "CAUTIOUS_TRADING"

    def cleanup_stale_predictions(self, max_age_minutes: int = 60) -> int:
        """
        Remove predictions older than max_age_minutes.

        Prevents memory leak from predictions that never got closed.

        Args:
            max_age_minutes: Maximum age in minutes before removal

        Returns:
            Number of stale predictions cleaned up
        """
        cutoff = datetime.utcnow() - timedelta(minutes=max_age_minutes)
        stale_count = 0

        with self._pending_lock:
            stale = [
                sid for sid, p in self._pending_predictions.items()
                if p['timestamp'] < cutoff
            ]
            for sid in stale:
                del self._pending_predictions[sid]
            stale_count = len(stale)

        if stale_count > 0:
            logger.warning(f"Cleaned up {stale_count} stale predictions")

        return stale_count

    def shutdown(self, timeout: float = 300):
        """
        Graceful shutdown - wait for retraining to complete.

        Args:
            timeout: Maximum time to wait in seconds (default: 5 minutes)
        """
        logger.info("Shutting down PerformanceBasedLearner...")

        # Wait for any ongoing retraining
        if self._active_retrain_thread and self._active_retrain_thread.is_alive():
            logger.info(f"Waiting for retraining to complete (timeout: {timeout}s)...")
            self._active_retrain_thread.join(timeout=timeout)

            if self._active_retrain_thread.is_alive():
                logger.warning("Timeout waiting for retraining to complete")
            else:
                logger.info("Retraining completed successfully")

        # Final stats
        stats = self.get_stats()
        logger.info(
            f"PerformanceBasedLearner shutdown complete\n"
            f"  Total candles: {stats['total_candles']}\n"
            f"  Win rate: {stats['overall_win_rate']:.1%}\n"
            f"  Retrains: {stats['retrains_triggered']}\n"
            f"  Reinforcements: {stats['reinforcements_applied']}"
        )


def create_performance_learner(
    predictor,
    database,
    timeframes: List[str] = None,
    loss_retrain: bool = True,
    reinforce_wins: bool = True
) -> PerformanceBasedLearner:
    """
    Factory function to create a configured PerformanceBasedLearner.

    Args:
        predictor: UnbreakablePredictor instance
        database: Database instance
        timeframes: List of timeframes to train on
        loss_retrain: Whether to retrain on losses
        reinforce_wins: Whether to reinforce on wins

    Returns:
        Configured PerformanceBasedLearner
    """
    config = PerformanceLearnerConfig(
        timeframes=timeframes or ['15m', '1h'],
        loss_retrain_enabled=loss_retrain,
        reinforce_on_win=reinforce_wins
    )

    return PerformanceBasedLearner(
        predictor=predictor,
        database=database,
        config=config
    )
