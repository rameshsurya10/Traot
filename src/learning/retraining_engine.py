"""
Retraining Engine
=================

Handles incremental model retraining with continual learning.

Strategy:
1. Use recent data (configurable lookback)
2. Mix with experience replay buffer (configurable ratio)
3. Apply EWC regularization to prevent catastrophic forgetting
4. Train until validation confidence ≥ target (default 80%)
5. Early stopping if no improvement (configurable patience)

Integrates with:
- MultiTimeframeModelManager: Load/save models
- EWC: Prevent catastrophic forgetting
- ExperienceReplayBuffer: Replay important samples
- Database: Fetch training data and record metrics

NO hardcoded values - all from config
Production-ready with comprehensive logging
"""

import logging
import time
import threading
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from datetime import datetime

from src.core.database import Database
from src.multi_timeframe.model_manager import MultiTimeframeModelManager
from src.ml.learning.continual import EWC

logger = logging.getLogger(__name__)


class RetrainingEngine:
    """
    Handles incremental model retraining with continual learning.

    Thread-safe: Can be called from background threads.

    Workflow:
    1. Fetch recent candles from database
    2. Compute features and labels
    3. Mix with experience replay samples
    4. Create train/validation splits
    5. Train with EWC regularization until confidence ≥ target
    6. Update Fisher Information matrix
    7. Save improved model
    8. Record metrics to database
    """

    def __init__(
        self,
        model_manager: MultiTimeframeModelManager,
        database: Database,
        continual_learner: any = None,
        config: dict = None
    ):
        """
        Initialize retraining engine.

        Args:
            model_manager: MultiTimeframeModelManager instance
            database: Database instance
            continual_learner: ContinualLearner instance (provides replay buffers and EWC)
            config: Configuration dict from config.yaml
        """
        self.model_manager = model_manager
        self.db = database
        self.continual_learner = continual_learner
        self.config = config or {}

        # Get replay buffers from continual learner if available
        if hasattr(continual_learner, 'replay_buffers'):
            self.replay_buffers = continual_learner.replay_buffers
        elif hasattr(continual_learner, 'replay_buffer'):
            self.replay_buffers = {'default': continual_learner.replay_buffer}
        else:
            self.replay_buffers = {}
            logger.debug("No replay buffers available (mathematical predictor - expected)")

        # Retraining parameters (from config)
        retrain_config = self.config.get('continuous_learning', {}).get('retraining', {})
        self.target_confidence = retrain_config.get('target_confidence', 0.80)
        self.max_epochs = retrain_config.get('max_epochs', 50)
        self.patience = retrain_config.get('patience', 10)
        self.recent_candles = retrain_config.get('recent_candles', 5000)
        self.replay_mix_ratio = retrain_config.get('replay_mix_ratio', 0.3)
        self.min_samples = retrain_config.get('min_samples', 1000)
        self.min_candles_for_features = retrain_config.get('min_candles_for_features', 100)

        # EWC parameters (from config)
        ewc_config = self.config.get('continuous_learning', {}).get('ewc', {})
        self.ewc_lambda = ewc_config.get('lambda', 1000.0)
        self.fisher_sample_size = ewc_config.get('fisher_sample_size', 200)

        # Learning parameters
        self.learning_rate = self.config.get('model', {}).get('learning_rate', 0.001)
        self.batch_size = self.config.get('model', {}).get('batch_size', 32)
        self.validation_split = self.config.get('model', {}).get('validation_split', 0.2)

        # Statistics (thread-safe)
        self._stats_lock = threading.Lock()
        self._stats = {
            'retrainings_attempted': 0,
            'retrainings_successful': 0,
            'retrainings_failed': 0,
            'total_epochs': 0,
            'total_duration_seconds': 0.0
        }

        logger.info(
            f"RetrainingEngine initialized: "
            f"target_confidence={self.target_confidence:.1%}, "
            f"max_epochs={self.max_epochs}, "
            f"patience={self.patience}"
        )

    def retrain(
        self,
        symbol: str,
        interval: str,
        trigger_reason: str,
        recent_candles: int = None,
        replay_mix_ratio: float = None
    ) -> dict:
        """
        Retrain model until confidence ≥ target.

        Args:
            symbol: Trading pair
            interval: Timeframe
            trigger_reason: Why retraining was triggered
            recent_candles: Number of recent candles to use (None = use config)
            replay_mix_ratio: Replay buffer mix ratio (None = use config)

        Returns:
            {
                'success': bool,
                'validation_accuracy': float,
                'validation_confidence': float,
                'epochs_trained': int,
                'improvement_pct': float,
                'duration_seconds': float,
                'error': str (if failed)
            }
        """
        with self._stats_lock:
            self._stats['retrainings_attempted'] += 1

        # Use config defaults if not specified
        if recent_candles is None:
            recent_candles = self.recent_candles
        if replay_mix_ratio is None:
            replay_mix_ratio = self.replay_mix_ratio

        logger.info(
            f"[{symbol} @ {interval}] Starting retraining. "
            f"Reason: {trigger_reason}, "
            f"Recent candles: {recent_candles}, "
            f"Replay mix: {replay_mix_ratio:.1%}"
        )

        # Record start in database
        retrain_id = self.db.start_retraining_event(
            symbol=symbol,
            interval=interval,
            trigger_reason=trigger_reason,
            trigger_metadata={
                'recent_candles': recent_candles,
                'replay_mix_ratio': replay_mix_ratio,
                'ewc_lambda': self.ewc_lambda
            }
        )

        start_time = time.time()

        try:
            # 1. Load current model
            model = self.model_manager.load_model(symbol, interval)
            if model is None:
                raise ValueError(f"Model not found for {symbol} @ {interval}")

            # Get model metadata for baseline
            metadata = self.model_manager.get_metadata(symbol, interval)
            baseline_confidence = metadata.validation_confidence if metadata else 0.0

            # 2. Prepare training data
            logger.debug(f"[{symbol} @ {interval}] Fetching training data...")
            X_train, y_train, X_val, y_val = self._prepare_training_data(
                symbol=symbol,
                interval=interval,
                recent_candles=recent_candles,
                replay_ratio=replay_mix_ratio
            )

            if len(X_train) < self.min_samples:
                raise ValueError(
                    f"Insufficient training data: {len(X_train)} < {self.min_samples}"
                )

            logger.info(
                f"[{symbol} @ {interval}] Training data: "
                f"{len(X_train)} train, {len(X_val)} val"
            )

            # 3. Initialize EWC
            ewc = EWC(
                model=model,
                ewc_lambda=self.ewc_lambda,
                fisher_sample_size=self.fisher_sample_size
            )

            # Load existing Fisher Information if available
            # (This preserves knowledge from previous retraining)
            model_path = self.model_manager.get_model_path(symbol, interval)
            fisher_path = model_path.with_suffix('.fisher')
            if fisher_path.exists():
                try:
                    fisher_state = torch.load(fisher_path)
                    ewc._fisher = fisher_state['fisher']
                    ewc._optimal_params = fisher_state['optimal_params']
                    ewc._is_consolidated = True
                    logger.debug("Loaded existing Fisher Information")
                except Exception as e:
                    logger.warning(f"Failed to load Fisher: {e}")

            # 4. Train until confident
            best_accuracy = 0.0
            best_confidence = 0.0
            best_model_state = model.state_dict()  # Save initial state
            patience_counter = 0
            val_loss = None  # Initialize to prevent UnboundLocalError

            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            criterion = nn.BCELoss()

            # Create data loaders
            train_loader = self._create_data_loader(X_train, y_train, shuffle=True)
            val_loader = self._create_data_loader(X_val, y_val, shuffle=False)

            for epoch in range(self.max_epochs):
                # Train one epoch with EWC
                train_loss = self._train_epoch(
                    model=model,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    ewc=ewc
                )

                # Validate
                val_accuracy, val_confidence, val_loss = self._evaluate(
                    model=model,
                    val_loader=val_loader,
                    criterion=criterion
                )

                with self._stats_lock:
                    self._stats['total_epochs'] += 1

                logger.info(
                    f"[{symbol} @ {interval}] Epoch {epoch+1}/{self.max_epochs}: "
                    f"Train Loss={train_loss:.4f}, "
                    f"Val Loss={val_loss:.4f}, "
                    f"Val Acc={val_accuracy:.2%}, "
                    f"Val Conf={val_confidence:.2%}"
                )

                # Check if target reached
                if val_confidence >= self.target_confidence:
                    logger.info(
                        f"✓ [{symbol} @ {interval}] Target confidence reached: "
                        f"{val_confidence:.2%} ≥ {self.target_confidence:.2%}"
                    )

                    # Save best model state
                    best_model_state = model.state_dict()
                    best_accuracy = val_accuracy
                    best_confidence = val_confidence

                    # Consolidate knowledge (compute Fisher for EWC)
                    logger.debug("Computing Fisher Information...")
                    ewc.compute_fisher(val_loader, criterion)

                    # Save improved model
                    self._save_model_and_fisher(
                        model=model,
                        ewc=ewc,
                        symbol=symbol,
                        interval=interval,
                        accuracy=val_accuracy,
                        confidence=val_confidence,
                        samples_trained=len(X_train)
                    )

                    duration = time.time() - start_time

                    with self._stats_lock:
                        self._stats['total_duration_seconds'] += duration
                        self._stats['retrainings_successful'] += 1

                    improvement_pct = ((val_confidence - baseline_confidence) / max(baseline_confidence, 0.01)) * 100

                    # Record success
                    self.db.complete_retraining_event(
                        retrain_id=retrain_id,
                        status='success',
                        validation_accuracy=val_accuracy,
                        validation_confidence=val_confidence,
                        epochs_trained=epoch + 1,
                        duration_seconds=duration
                    )

                    return {
                        'success': True,
                        'validation_accuracy': val_accuracy,
                        'validation_confidence': val_confidence,
                        'epochs_trained': epoch + 1,
                        'improvement_pct': improvement_pct,
                        'duration_seconds': duration
                    }

                # Track best performance
                if val_confidence > best_confidence:
                    best_confidence = val_confidence
                    best_accuracy = val_accuracy
                    best_model_state = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1

                    # Early stopping
                    if patience_counter >= self.patience:
                        logger.warning(
                            f"[{symbol} @ {interval}] Early stopping - "
                            f"no improvement for {self.patience} epochs"
                        )
                        break

            # Failed to reach target, but save progress
            if best_model_state is not None:
                model.load_state_dict(best_model_state)

                # Consolidate knowledge with best model
                ewc.compute_fisher(val_loader, criterion)

                # Save progress
                self._save_model_and_fisher(
                    model=model,
                    ewc=ewc,
                    symbol=symbol,
                    interval=interval,
                    accuracy=best_accuracy,
                    confidence=best_confidence,
                    samples_trained=len(X_train)
                )

            duration = time.time() - start_time

            with self._stats_lock:
                self._stats['total_duration_seconds'] += duration
                self._stats['retrainings_failed'] += 1

            improvement_pct = ((best_confidence - baseline_confidence) / max(baseline_confidence, 0.01)) * 100

            logger.warning(
                f"[{symbol} @ {interval}] Failed to reach target confidence. "
                f"Best: {best_confidence:.2%}, Target: {self.target_confidence:.2%}"
            )

            # Record failure (but note progress was saved)
            self.db.complete_retraining_event(
                retrain_id=retrain_id,
                status='below_target',
                validation_accuracy=best_accuracy,
                validation_confidence=best_confidence,
                epochs_trained=epoch + 1,
                duration_seconds=duration
            )

            return {
                'success': False,
                'validation_accuracy': best_accuracy,
                'validation_confidence': best_confidence,
                'epochs_trained': epoch + 1,
                'improvement_pct': improvement_pct,
                'duration_seconds': duration
            }

        except Exception as e:
            logger.error(f"[{symbol} @ {interval}] Retraining failed: {e}", exc_info=True)

            duration = time.time() - start_time

            with self._stats_lock:
                self._stats['retrainings_failed'] += 1

            self.db.complete_retraining_event(
                retrain_id=retrain_id,
                status='error',
                error_message=str(e),
                duration_seconds=duration
            )

            return {
                'success': False,
                'error': str(e),
                'duration_seconds': duration
            }

    def _prepare_training_data(
        self,
        symbol: str,
        interval: str,
        recent_candles: int,
        replay_ratio: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data by mixing recent candles with replay buffer.

        Args:
            symbol: Trading pair
            interval: Timeframe
            recent_candles: Number of recent candles to fetch
            replay_ratio: Ratio of replay samples to mix in

        Returns:
            (X_train, y_train, X_val, y_val)
        """
        from src.analysis_engine import FeatureCalculator
        import pandas as pd

        # 1. Fetch recent candles from database
        logger.debug(f"Fetching {recent_candles} candles for {symbol} @ {interval}")

        # live_only=True — retraining must use real recent market data only.
        # Historical backfill is for initial pre-training; ongoing retraining
        # must reflect the current market regime, not data from months ago.
        candles = self.db.get_candles(
            symbol=symbol,
            interval=interval,
            limit=recent_candles + 100,  # Extra for indicator warmup
            live_only=True,
            live_days=30  # Use last 30 days for retraining context
        )

        # Use stored retrain_config instead of re-traversing config path
        min_candles = getattr(self, 'min_candles_for_features', 100)
        if candles is None or len(candles) < min_candles:
            raise ValueError(
                f"Insufficient candle data: got {len(candles) if candles is not None else 0} candles, "
                f"need at least {min_candles} for feature calculation"
            )

        # Convert to DataFrame if needed
        if not isinstance(candles, pd.DataFrame):
            df = pd.DataFrame(candles)
        else:
            df = candles.copy()

        # Ensure required columns exist
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # 2. Calculate all features (32 technical + 7 sentiment = 39 features)
        logger.debug("Calculating technical and sentiment features...")

        include_sentiment = self.config.get('model', {}).get('features', {}).get('include_sentiment', True)

        df_features = FeatureCalculator.calculate_all_with_sentiment(
            df=df,
            database=self.db,
            symbol=symbol,
            include_sentiment=include_sentiment
        )

        # 3. Generate labels (price direction: 1 if next close > current close, 0 otherwise)
        df_features['label'] = (df_features['close'].shift(-1) > df_features['close']).astype(np.float32)

        # Drop rows with NaN (from indicators and label shift)
        df_features = df_features.dropna()

        if len(df_features) < self.min_samples:
            raise ValueError(
                f"After feature calculation, only {len(df_features)} samples remain. "
                f"Need at least {self.min_samples}. Try increasing recent_candles."
            )

        # 4. Extract feature matrix and labels
        feature_cols = FeatureCalculator.get_feature_columns()

        # Verify all feature columns exist
        missing_features = [col for col in feature_cols if col not in df_features.columns]
        if missing_features:
            logger.warning(f"Missing feature columns: {missing_features}. Setting to 0.0")
            for col in missing_features:
                df_features[col] = 0.0

        X = df_features[feature_cols].values.astype(np.float32)
        y = df_features['label'].values.astype(np.float32)

        # Limit to requested number of recent candles
        if len(X) > recent_candles:
            X = X[-recent_candles:]
            y = y[-recent_candles:]

        logger.info(
            f"Prepared {len(X)} training samples with {X.shape[1]} features from real market data"
        )

        # 5. Create sequences for LSTM (2D → 3D)
        # LSTM expects (batch, sequence_length, features), not (batch, features)
        seq_len = self._get_sequence_length(interval)
        X_seq, y_seq = self._create_sequences(X, y, seq_len)

        if len(X_seq) < self.min_samples:
            raise ValueError(
                f"After sequencing (seq_len={seq_len}), only {len(X_seq)} samples remain. "
                f"Need at least {self.min_samples}. Try increasing recent_candles."
            )

        logger.info(
            f"Created {len(X_seq)} sequences of length {seq_len} "
            f"(shape: {X_seq.shape}) for LSTM training"
        )

        # 7. Train/validation split
        split_idx = int(len(X_seq) * (1 - self.validation_split))

        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

        logger.info(
            f"Training data prepared: {len(X_train)} train samples, {len(X_val)} validation samples"
        )

        return X_train, y_train, X_val, y_val

    def _get_sequence_length(self, interval: str) -> int:
        """
        Get sequence length for a given interval from config.

        Falls back to model.sequence_length, then default 60.
        """
        # Check interval-specific config first
        intervals = self.config.get('timeframes', {}).get('intervals', [])
        for interval_config in intervals:
            if interval_config.get('interval') == interval:
                seq_len = interval_config.get('sequence_length')
                if seq_len is not None:
                    return seq_len

        # Fallback to global model config
        return self.config.get('model', {}).get('sequence_length', 60)

    def _create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding-window sequences from 2D feature matrix.

        Transforms:
            X: (n_samples, n_features) → (n_samples - seq_len, seq_len, n_features)
            y: (n_samples,) → (n_samples - seq_len,)

        Each sequence X_seq[i] = X[i:i+seq_len] with label y_seq[i] = y[i+seq_len-1]
        (the label at the end of the window).
        """
        n_samples = len(X)
        if n_samples <= sequence_length:
            raise ValueError(
                f"Not enough samples ({n_samples}) for sequence_length={sequence_length}"
            )

        n_sequences = n_samples - sequence_length
        X_seq = np.empty((n_sequences, sequence_length, X.shape[1]), dtype=np.float32)
        y_seq = np.empty(n_sequences, dtype=np.float32)

        for i in range(n_sequences):
            X_seq[i] = X[i:i + sequence_length]
            # Label = "did price go up after the last candle in this window?"
            y_seq[i] = y[i + sequence_length - 1]

        return X_seq, y_seq

    def _create_data_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool = True
    ) -> torch.utils.data.DataLoader:
        """Create PyTorch DataLoader from numpy arrays."""
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(y)
        )

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0  # Avoid multiprocessing issues
        )

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        ewc: EWC
    ) -> float:
        """
        Train one epoch with EWC regularization.

        Returns:
            Average training loss
        """
        model.train()
        total_loss = 0.0
        n_batches = 0

        for inputs, targets in train_loader:
            # EWC training step
            loss = ewc.training_step(inputs, targets, optimizer, criterion)
            total_loss += loss
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _evaluate(
        self,
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float, float]:
        """
        Evaluate model on validation set.

        Returns:
            (accuracy, average_confidence, loss)
        """
        model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        confidences = []

        # Confidence calibration params (outside loop for performance)
        pred_config = self.config.get('prediction', {})
        floor = pred_config.get('confidence_floor', 0.48)
        ceiling = pred_config.get('confidence_ceiling', 0.55)
        conf_range = ceiling - floor

        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)

                total_loss += loss.item()

                # Predictions
                predictions = (outputs > 0.5).float()
                correct += (predictions == targets).sum().item()
                total += len(targets)

                # Confidence using calibrated range mapping (aligned with AdvancedPredictor)
                conf = (torch.abs(outputs - 0.5) + 0.5 - floor) / conf_range if conf_range > 0 else torch.abs(outputs - 0.5) * 2
                conf = torch.clamp(conf, 0.0, 1.0)
                confidences.extend(conf.tolist())

        accuracy = correct / max(total, 1)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        avg_loss = total_loss / len(val_loader)

        return accuracy, avg_confidence, avg_loss

    def _save_model_and_fisher(
        self,
        model: nn.Module,
        ewc: EWC,
        symbol: str,
        interval: str,
        accuracy: float,
        confidence: float,
        samples_trained: int
    ):
        """Save model and Fisher Information."""
        from src.multi_timeframe.model_manager import ModelMetadata

        # Save model
        metadata = ModelMetadata(
            symbol=symbol,
            interval=interval,
            trained_at=datetime.utcnow(),
            samples_trained=samples_trained,
            validation_accuracy=accuracy,
            validation_confidence=confidence,
            version=1,
            config={
                'ewc_lambda': self.ewc_lambda,
                'fisher_sample_size': self.fisher_sample_size
            }
        )

        self.model_manager.save_model(symbol, interval, model, metadata)

        # Save Fisher Information
        model_path = self.model_manager.get_model_path(symbol, interval)
        fisher_path = model_path.with_suffix('.fisher')

        if ewc._is_consolidated:
            torch.save({
                'fisher': ewc._fisher,
                'optimal_params': ewc._optimal_params
            }, fisher_path)

            logger.debug(f"Saved Fisher Information to {fisher_path}")

    def get_stats(self) -> dict:
        """Get retraining engine statistics (thread-safe)."""
        with self._stats_lock:
            success_rate = 0.0
            if self._stats['retrainings_attempted'] > 0:
                success_rate = (
                    self._stats['retrainings_successful'] /
                    self._stats['retrainings_attempted']
                )

            avg_duration = 0.0
            if self._stats['retrainings_attempted'] > 0:
                avg_duration = (
                    self._stats['total_duration_seconds'] /
                    self._stats['retrainings_attempted']
                )

            return {
                **self._stats,
                'success_rate': success_rate,
                'avg_duration_seconds': avg_duration,
                'avg_epochs_per_retrain': (
                    self._stats['total_epochs'] /
                    max(self._stats['retrainings_attempted'], 1)
                )
            }
