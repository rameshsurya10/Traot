"""
Unbreakable Predictor
======================

The main prediction engine that combines all components:
1. SVMD Signal Decomposition
2. GMM-HMM Regime Detection
3. TCN-LSTM-Attention + XGBoost + LightGBM Base Models
4. Stacking Meta-Learner
5. Risk Management (Kelly + Dynamic SL/TP)
6. Continuous Learning (EWC + Concept Drift)

This is the central orchestrator for making trading predictions.
"""

import numpy as np
import pandas as pd
import torch
import threading
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import logging
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

from .regime import RegimeDetector
from .models import create_base_models
from .ensemble import StackingEnsemble
from .risk import RiskManager
from .learning import ContinualLearner
from .features import FeatureEngineer
from .features.selector import AdaptiveFeatureSelector, MarketRegime, get_features_for_regime

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Complete trading signal with all components."""
    # Core signal
    direction: str  # BUY, SELL, NEUTRAL
    confidence: float  # 0.0 to 1.0
    probability: float  # Raw probability

    # Price levels
    entry_price: float
    stop_loss: float
    take_profit: float

    # Risk metrics
    position_size_pct: float
    risk_reward_ratio: float
    expected_value: float

    # Analysis details
    regime: str
    regime_confidence: float
    base_model_predictions: Dict[str, float]

    # Technical indicators
    rsi: float
    macd_hist: float
    atr: float
    bb_position: float

    # Metadata
    timestamp: datetime
    model_confidence: float  # Model's internal confidence
    drift_score: float  # Concept drift score
    lstm_probability: float = 0.5  # Raw LSTM probability (for continuous learning compatibility)

    # Warnings
    warnings: list = None

    def __post_init__(self):
        """Initialize default values after dataclass init."""
        if self.warnings is None:
            self.warnings = []
        if self.lstm_probability is None:
            self.lstm_probability = self.probability


class UnbreakablePredictor:
    """
    The Unbreakable Trading Prediction System.

    Combines all research-backed components into a single, robust predictor.

    Architecture:
    1. Data → SVMD Decomposition → Feature Engineering
    2. Features → Regime Detection (GMM-HMM)
    3. Features + Regime → Base Models (TCN-LSTM, XGBoost, LightGBM)
    4. Base Predictions → Stacking Meta-Learner
    5. Final Prediction → Risk Management
    6. Continuous Learning monitors and adapts
    """

    # Signal thresholds (configurable)
    BUY_THRESHOLD = 0.55
    SELL_THRESHOLD = 0.45

    def __init__(
        self,
        config_path: str = "config.yaml",
        model_dir: str = "models/unbreakable",
        sequence_length: int = 60,
        hidden_size: int = 128,
        use_gpu: bool = True,
        capital: float = 100000.0,
        adaptive_features: bool = True,
        max_features: int = 15
    ):
        """
        Initialize the Unbreakable Predictor.

        Args:
            config_path: Path to config file
            model_dir: Directory for model storage
            sequence_length: Sequence length for LSTM
            hidden_size: Hidden layer size
            use_gpu: Whether to use GPU
            capital: Trading capital for risk calculations
            adaptive_features: Enable adaptive feature selection per regime
            max_features: Maximum features to use (if adaptive_features=True)
        """
        self.config_path = config_path
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.capital = capital
        self.adaptive_features = adaptive_features
        self.max_features = max_features
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

        # Load regime mismatch penalties from config
        try:
            from src.core.config import load_config
            raw_config = load_config(config_path)
            pred_cfg = raw_config.get('prediction', {})
        except Exception:
            pred_cfg = {}
        regime_mismatch_cfg = pred_cfg.get('regime_mismatch', {})
        self._low_regime_conf_penalty = regime_mismatch_cfg.get('low_regime_confidence', 0.8)
        self._bear_buy_penalty = regime_mismatch_cfg.get('bear_buy', 0.7)
        self._bull_sell_penalty = regime_mismatch_cfg.get('bull_sell', 0.7)

        # Initialize components
        self.regime_detector = RegimeDetector()
        self.feature_engineer = FeatureEngineer(
            sequence_length=sequence_length,
            include_svmd=True,
            include_regime=True
        )
        self.risk_manager = RiskManager(
            kelly_fraction=0.25,
            max_risk_per_trade=0.02,
            atr_multiplier_sl=2.0,
            atr_multiplier_tp=4.0
        )

        # Adaptive feature selector
        self.feature_selector: Optional[AdaptiveFeatureSelector] = None
        if adaptive_features:
            self.feature_selector = AdaptiveFeatureSelector(
                n_features=max_features,
                use_shap=True,
                regime_weight=0.4,
                importance_weight=0.6
            )

        # These will be initialized after fitting
        self.base_models: Dict[str, Any] = {}
        self.ensemble: Optional[StackingEnsemble] = None
        self.continual_learner: Optional[ContinualLearner] = None

        self._is_fitted = False
        self._last_prediction: Optional[TradingSignal] = None
        self._selected_features: Dict[str, List[str]] = {}  # Cache selected features per regime
        self._training_features: List[str] = []  # Features used during training (for prediction consistency)

        # Thread safety lock for model operations
        self._model_lock = threading.Lock()

        logger.info(f"UnbreakablePredictor initialized on {self.device}")

    @property
    def tcn_lstm_model(self):
        """
        Get the TCN-LSTM-Attention neural network model.

        Used by PerformanceBasedLearner for reinforcement learning.

        Returns:
            The neural network model or None if not fitted
        """
        return self.base_models.get('tcn_lstm_attention')

    def fit(
        self,
        df: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: bool = True
    ) -> 'UnbreakablePredictor':
        """
        Train the complete prediction system.

        Args:
            df: DataFrame with OHLCV data
            epochs: Training epochs for neural networks
            batch_size: Batch size for training
            validation_split: Validation split ratio
            verbose: Print progress

        Returns:
            self
        """
        logger.info("="*60)
        logger.info("TRAINING UNBREAKABLE PREDICTION SYSTEM")
        logger.info("="*60)

        # 1. Fit regime detector
        if verbose:
            logger.info("Step 1: Fitting regime detector (GMM-HMM)...")
        self.regime_detector.fit(df)

        # 2. Engineer features
        if verbose:
            logger.info("Step 2: Engineering features with SVMD...")
        feature_set = self.feature_engineer.fit_transform(df, self.regime_detector)

        n_features_original = feature_set.tabular_data.shape[1]
        logger.info(f"   Created {n_features_original} features from {len(df)} samples")

        # 3. Apply adaptive feature selection BEFORE training
        if self.adaptive_features and self.feature_selector is not None:
            if verbose:
                logger.info("Step 3: Applying adaptive feature selection...")

            # Get all feature names from engineer
            all_feature_names = list(self.feature_engineer._feature_names)  # Copy to avoid mutation

            # Calculate feature importance using sampled data for efficiency
            X_full = feature_set.tabular_data
            y_full = feature_set.targets

            # Sample data for importance calculation (memory efficient)
            sample_size = min(5000, len(X_full))
            X_sample = X_full[:sample_size]
            y_sample = y_full[:sample_size]

            # Use a lightweight RF for initial importance
            rf = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42, n_jobs=2)
            rf.fit(X_sample, y_sample)

            # Calculate importance from RF + regime effectiveness
            self.feature_selector.calculate_importance(
                {'random_forest': rf}, X_sample, y_sample, all_feature_names
            )

            # Select features using UNION of top features across ALL regimes
            # This ensures model works well regardless of market conditions
            all_selected = set()
            feature_counts = {}
            all_regimes = [MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.SIDEWAYS, MarketRegime.VOLATILE]

            for regime in all_regimes:
                regime_selection = self.feature_selector.select_features(
                    all_feature_names, regime, n_features=self.max_features
                )
                for f in regime_selection.selected_features:
                    all_selected.add(f)
                    feature_counts[f] = feature_counts.get(f, 0) + 1

            # Limit to max_features by taking most common across regimes
            if len(all_selected) > self.max_features:
                # Sort by count (most common first), then by importance
                sorted_features = sorted(
                    all_selected,
                    key=lambda f: (feature_counts.get(f, 0), self.feature_selector._feature_importance.get(f, 0)),
                    reverse=True
                )
                selected_features = sorted_features[:self.max_features]
            else:
                selected_features = list(all_selected)

            # Validate all selected features exist in original
            valid_selected = [f for f in selected_features if f in all_feature_names]
            if len(valid_selected) != len(selected_features):
                logger.warning(f"Some selected features not found, using {len(valid_selected)} of {len(selected_features)}")
            selected_features = valid_selected

            if not selected_features:
                raise ValueError("Feature selection returned no valid features - cannot train model")

            self._training_features = selected_features  # Store for prediction consistency

            logger.info(f"   Selected {len(selected_features)} of {n_features_original} features (cross-regime optimal)")
            logger.info(f"   Top features: {', '.join(selected_features[:5])}...")

            # Build feature index mapping
            feature_indices = [all_feature_names.index(f) for f in selected_features]

            # Validate dimensions before slicing
            if feature_set.sequence_data is not None and len(feature_set.sequence_data) > 0:
                if feature_set.sequence_data.shape[2] != len(all_feature_names):
                    logger.warning(f"Sequence data feature mismatch: {feature_set.sequence_data.shape[2]} vs {len(all_feature_names)}")

            # Update tabular data to only include selected features
            feature_set.tabular_data = feature_set.tabular_data[:, feature_indices]

            # Update sequence data if present (filter last dimension)
            if feature_set.sequence_data is not None and len(feature_set.sequence_data) > 0:
                feature_set.sequence_data = feature_set.sequence_data[:, :, feature_indices]

            # Update feature engineer's internal state to match
            self.feature_engineer._feature_names = selected_features
            self.feature_engineer._feature_means = self.feature_engineer._feature_means[feature_indices]
            self.feature_engineer._feature_stds = self.feature_engineer._feature_stds[feature_indices]

            # Cache selection for all regimes
            for regime in all_regimes:
                self._selected_features[regime.value] = selected_features

            n_features = len(selected_features)
            logger.info(f"   Training will use {n_features} features")
        else:
            n_features = n_features_original
            self._training_features = list(self.feature_engineer._feature_names) if self.feature_engineer._feature_names else []

        # 4. Create base models with CORRECT feature count
        if verbose:
            logger.info(f"Step {'4' if self.adaptive_features else '3'}: Creating base models (input_size={n_features})...")
        self.base_models = create_base_models(
            input_size=n_features,
            hidden_size=self.hidden_size,
            sequence_length=self.sequence_length,
            use_gpu=(self.device.type == 'cuda')
        )

        # 5. Train base models
        if verbose:
            step = 5 if self.adaptive_features else 4
            logger.info(f"Step {step}: Training base models...")

        X = feature_set.tabular_data
        y = feature_set.targets
        X_seq = feature_set.sequence_data

        # Time series split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        if X_seq is not None and len(X_seq) > 0:
            # Align sequence data with tabular
            X_seq_train = X_seq[:split_idx]
            X_seq_val = X_seq[split_idx:]
        else:
            X_seq_train, X_seq_val = None, None

        # Train each base model
        for name, model in self.base_models.items():
            if verbose:
                logger.info(f"   Training {name}...")

            if isinstance(model, torch.nn.Module):
                self._train_pytorch_model(
                    model, X_seq_train, y_train,
                    X_val=X_seq_val, y_val=y_val,
                    epochs=epochs, batch_size=batch_size
                )
            else:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=20,
                    verbose=False
                )

        # 6. Create and train stacking ensemble
        if verbose:
            step = 6 if self.adaptive_features else 5
            logger.info(f"Step {step}: Training stacking ensemble...")

        self.ensemble = StackingEnsemble(
            base_models=self.base_models,
            meta_learner='ridge',
            n_folds=5,
            use_probabilities=True
        )

        self.ensemble.fit(
            X_train, y_train,
            X_seq=X_seq_train,
            fit_base_models=False,  # Already trained
            verbose=verbose
        )

        # 7. Initialize continual learner
        if verbose:
            step = 7 if self.adaptive_features else 6
            logger.info(f"Step {step}: Initializing continual learning...")

        # Get the PyTorch model for continual learning
        pytorch_model = None
        for name, model in self.base_models.items():
            if isinstance(model, torch.nn.Module):
                pytorch_model = model
                break

        if pytorch_model is not None:
            self.continual_learner = ContinualLearner(
                model=pytorch_model,
                ewc_lambda=1000.0,
                replay_buffer_size=10000
            )

            # Create data loader for consolidation
            train_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_seq_train if X_seq_train is not None else X_train),
                torch.FloatTensor(y_train)
            )
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

            self.continual_learner.consolidate(train_loader)

        self._is_fitted = True

        # 8. Evaluate
        if verbose:
            step = 8 if self.adaptive_features else 7
            logger.info(f"Step {step}: Evaluating model...")
            self._evaluate(X_val, y_val, X_seq_val)

        # 9. Save models
        self.save()

        logger.info("="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("="*60)

        return self

    def _train_pytorch_model(
        self,
        model: torch.nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 0.001
    ):
        """Train a PyTorch model."""
        model = model.to(self.device)
        model.train()

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            train_loss = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = model(X_batch).squeeze(-1)  # Squeeze only last dimension
                # Ensure both tensors have same shape for BCE loss
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                if y_batch.dim() == 0:
                    y_batch = y_batch.unsqueeze(0)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            if X_val is not None:
                model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val).to(self.device)
                    y_val_tensor = torch.FloatTensor(y_val).to(self.device)
                    val_outputs = model(X_val_tensor).squeeze(-1)
                    # Ensure both tensors have same shape
                    if val_outputs.dim() == 0:
                        val_outputs = val_outputs.unsqueeze(0)
                    if y_val_tensor.dim() == 0:
                        y_val_tensor = y_val_tensor.unsqueeze(0)
                    val_loss = criterion(val_outputs, y_val_tensor).item()

                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 10:
                        logger.info(f"   Early stopping at epoch {epoch+1}")
                        break

    def _evaluate(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_seq_val: np.ndarray = None
    ):
        """Evaluate model performance."""
        probs = self.ensemble.predict_proba(X_val, X_seq_val)
        preds = (probs > 0.5).astype(int)

        accuracy = (preds == y_val).mean()
        logger.info(f"   Validation Accuracy: {accuracy:.2%}")

        # High confidence accuracy
        high_conf_mask = (probs > 0.6) | (probs < 0.4)
        if high_conf_mask.sum() > 0:
            high_conf_acc = (preds[high_conf_mask] == y_val[high_conf_mask]).mean()
            logger.info(f"   High Confidence Accuracy: {high_conf_acc:.2%} ({high_conf_mask.sum()} samples)")

    def predict(
        self,
        df: pd.DataFrame,
        symbol: str = None,
        interval: str = None
    ) -> TradingSignal:
        """
        Make a trading prediction.

        Thread-safe: Uses model lock to prevent conflicts with online_update.

        Args:
            df: DataFrame with recent OHLCV data
            symbol: Trading pair (optional, for logging/multi-asset support)
            interval: Timeframe (optional, for logging/multi-timeframe support)

        Returns:
            TradingSignal with complete analysis

        Raises:
            ValueError: If model not fitted or required columns missing
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Validate input DataFrame
        required_cols = ['close', 'high', 'low', 'open', 'volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if len(df) < self.sequence_length:
            raise ValueError(f"Insufficient data: need at least {self.sequence_length} rows, got {len(df)}")

        # Get current price
        current_price = df['close'].iloc[-1]
        current_time = df['datetime'].iloc[-1] if 'datetime' in df.columns else datetime.now()

        warnings = []

        # 1. Detect regime
        regime_result = self.regime_detector.detect(df)
        regime = regime_result.current_regime.name
        regime_confidence = regime_result.confidence

        # 2. Engineer features
        feature_set = self.feature_engineer.transform(df, self.regime_detector)

        if len(feature_set.tabular_data) == 0:
            return self._create_neutral_signal(current_price, current_time, "Insufficient data")

        # 3. Get ensemble prediction (thread-safe)
        X = feature_set.tabular_data[-1:] if len(feature_set.tabular_data) > 0 else None
        X_seq = feature_set.sequence_data[-1:] if feature_set.sequence_data is not None and len(feature_set.sequence_data) > 0 else None

        # Use lock to ensure model stays in eval mode during prediction
        with self._model_lock:
            ensemble_pred = self.ensemble.predict_detailed(X, X_seq)

        probability = ensemble_pred.probability
        model_confidence = ensemble_pred.confidence

        # 4. Determine direction
        if probability > self.BUY_THRESHOLD:
            direction = 'BUY'
        elif probability < self.SELL_THRESHOLD:
            direction = 'SELL'
        else:
            direction = 'NEUTRAL'

        # 5. Get technical indicators from features
        df_features = feature_set.features
        rsi = df_features['rsi_14'].iloc[-1] if 'rsi_14' in df_features else 50
        macd_hist = df_features['macd_hist'].iloc[-1] if 'macd_hist' in df_features else 0
        atr = df_features['atr_14'].iloc[-1] if 'atr_14' in df_features else current_price * 0.02
        bb_position = df_features['bb_position'].iloc[-1] if 'bb_position' in df_features else 0.5

        # 6. Risk assessment
        win_probability = probability if direction == 'BUY' else (1 - probability)

        risk_assessment = self.risk_manager.assess_trade(
            capital=self.capital,
            entry_price=current_price,
            direction=direction,
            win_probability=win_probability,
            atr=atr,
            prediction_confidence=model_confidence
        )

        warnings.extend(risk_assessment.warnings)

        # 7. Check for concept drift
        drift_score = 0.0
        if self.continual_learner is not None:
            drift_score = self.continual_learner.drift_detector.get_drift_score()
            if drift_score > 0.5:
                warnings.append(f"High concept drift detected: {drift_score:.2f}")

        # 8. Adjust confidence based on regime
        final_confidence = model_confidence

        # Reduce confidence in uncertain regimes (configurable via prediction.regime_mismatch)
        if regime_confidence < 0.5:
            final_confidence *= self._low_regime_conf_penalty
            warnings.append("Low regime confidence - reduced signal confidence")

        # Reduce confidence if regime doesn't match signal
        if regime == 'BEAR' and direction == 'BUY':
            final_confidence *= self._bear_buy_penalty
            warnings.append("BUY signal in BEAR regime - use caution")
        elif regime == 'BULL' and direction == 'SELL':
            final_confidence *= self._bull_sell_penalty
            warnings.append("SELL signal in BULL regime - use caution")

        # Get LSTM-specific probability for continuous learning
        lstm_prob = ensemble_pred.base_model_predictions.get('tcn_lstm_attention', probability)

        signal = TradingSignal(
            direction=direction,
            confidence=final_confidence,
            probability=probability,
            entry_price=current_price,
            stop_loss=risk_assessment.stop_loss,
            take_profit=risk_assessment.take_profit,
            position_size_pct=risk_assessment.position_size_pct,
            risk_reward_ratio=risk_assessment.risk_reward_ratio,
            expected_value=risk_assessment.expected_value,
            regime=regime,
            regime_confidence=regime_confidence,
            base_model_predictions=ensemble_pred.base_model_predictions,
            rsi=rsi,
            macd_hist=macd_hist,
            atr=atr,
            bb_position=bb_position,
            timestamp=current_time,
            model_confidence=model_confidence,
            drift_score=drift_score,
            lstm_probability=lstm_prob,
            warnings=warnings
        )

        self._last_prediction = signal
        return signal

    def _create_neutral_signal(
        self,
        price: float,
        timestamp: datetime,
        reason: str
    ) -> TradingSignal:
        """Create a neutral signal when prediction isn't possible."""
        return TradingSignal(
            direction='NEUTRAL',
            confidence=0.0,
            probability=0.5,
            entry_price=price,
            stop_loss=price,
            take_profit=price,
            position_size_pct=0.0,
            risk_reward_ratio=0.0,
            expected_value=0.0,
            regime='UNKNOWN',
            regime_confidence=0.0,
            base_model_predictions={},
            lstm_probability=0.5,
            rsi=50,
            macd_hist=0,
            atr=0,
            bb_position=0.5,
            timestamp=timestamp,
            model_confidence=0.0,
            drift_score=0.0,
            warnings=[reason]
        )

    def online_update(
        self,
        df: pd.DataFrame,
        symbol: str = None,
        interval: str = None,
        learning_rate: float = 0.0001,
        actual_outcome: int = None
    ):
        """
        Perform online learning update with new candle data.

        This is called by ContinuousLearningSystem after each candle.
        Performs small incremental update to the neural network.

        Args:
            df: DataFrame with recent OHLCV data (excluding the outcome candle)
            symbol: Trading pair (for logging, optional)
            interval: Timeframe (for logging, optional)
            learning_rate: Learning rate for online update
            actual_outcome: Actual outcome (1 if price went up, 0 if down).
                           If None, uses the last prediction's confirmed outcome.
        """
        if not self._is_fitted:
            logger.debug("Model not fitted, skipping online update")
            return

        if self.continual_learner is None:
            logger.debug("Continual learner not initialized, skipping online update")
            return

        # Require actual_outcome to be explicitly provided to avoid information leakage
        if actual_outcome is None:
            logger.debug("No actual_outcome provided, skipping online update")
            return

        # Validate actual_outcome
        if actual_outcome not in (0, 1, 0.0, 1.0):
            logger.warning(f"Invalid actual_outcome: {actual_outcome}, must be 0 or 1")
            return

        # Thread-safe model update
        with self._model_lock:
            try:
                # Get features for the latest candle
                feature_set = self.feature_engineer.transform(df, self.regime_detector)

                if feature_set.tabular_data is None or len(feature_set.tabular_data) == 0:
                    return

                # Get sequence data for LSTM
                X_seq = feature_set.sequence_data[-1:] if feature_set.sequence_data is not None else None

                if X_seq is None or len(X_seq) == 0:
                    return

                # Only update PyTorch model (LSTM/TCN) with online learning
                # Boosting models don't support online updates
                lstm_model = self.base_models.get('tcn_lstm_attention')
                if lstm_model is None:
                    return

                # Perform small gradient step with guaranteed eval mode restoration
                lstm_model.train()
                try:
                    X_tensor = torch.FloatTensor(X_seq).to(self.device)
                    y_tensor = torch.FloatTensor([float(actual_outcome)]).to(self.device)

                    # Small learning rate for online update
                    optimizer = torch.optim.SGD(lstm_model.parameters(), lr=learning_rate)
                    criterion = torch.nn.BCELoss()

                    optimizer.zero_grad()
                    output = lstm_model(X_tensor).squeeze()
                    if output.dim() == 0:
                        output = output.unsqueeze(0)

                    loss = criterion(output, y_tensor)
                    loss.backward()

                    # Gradient clipping for stability in online learning
                    torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=1.0)

                    optimizer.step()

                    log_msg = f"Online update: loss={loss.item():.4f}, outcome={actual_outcome}"
                    if symbol and interval:
                        log_msg = f"[{symbol} @ {interval}] {log_msg}"
                    logger.debug(log_msg)

                finally:
                    # Always restore eval mode even if exception occurs
                    lstm_model.eval()

            except Exception as e:
                logger.error(f"Online update failed: {e}", exc_info=True)

    def save(self, path: str = None):
        """Save all models to disk."""
        if path is None:
            path = self.model_dir

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save ensemble
        if self.ensemble is not None:
            self.ensemble.save(path / 'ensemble')

        # Save regime detector
        joblib.dump(self.regime_detector, path / 'regime_detector.joblib')

        # Save feature engineer parameters and training features
        joblib.dump({
            'feature_means': self.feature_engineer._feature_means,
            'feature_stds': self.feature_engineer._feature_stds,
            'feature_names': self.feature_engineer._feature_names,
            'training_features': self._training_features,
            'adaptive_features': self.adaptive_features,
            'max_features': self.max_features
        }, path / 'feature_params.joblib')

        logger.info(f"Models saved to {path}")

    def load(self, path: str = None):
        """Load all models from disk."""
        if path is None:
            path = self.model_dir

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model directory not found: {path}")

        # Load regime detector
        self.regime_detector = joblib.load(path / 'regime_detector.joblib')

        # Load feature parameters
        feature_params = joblib.load(path / 'feature_params.joblib')
        self.feature_engineer._feature_means = feature_params['feature_means']
        self.feature_engineer._feature_stds = feature_params['feature_stds']
        self.feature_engineer._feature_names = feature_params['feature_names']
        self.feature_engineer._is_fitted = True

        # Load training features (for prediction consistency)
        self._training_features = feature_params.get('training_features', feature_params['feature_names'])
        self.adaptive_features = feature_params.get('adaptive_features', self.adaptive_features)
        self.max_features = feature_params.get('max_features', self.max_features)

        # Load ensemble
        if (path / 'ensemble').exists():
            # Recreate base models
            n_features = len(self.feature_engineer._feature_names)
            self.base_models = create_base_models(
                input_size=n_features,
                hidden_size=self.hidden_size,
                use_gpu=(self.device.type == 'cuda')
            )

            self.ensemble = StackingEnsemble(base_models=self.base_models)
            self.ensemble.load(path / 'ensemble')

        self._is_fitted = True
        logger.info(f"Models loaded from {path}")

    def get_status(self) -> Dict[str, Any]:
        """Get predictor status."""
        return {
            'is_fitted': self._is_fitted,
            'device': str(self.device),
            'n_base_models': len(self.base_models),
            'base_models': list(self.base_models.keys()),
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'adaptive_features': self.adaptive_features,
            'max_features': self.max_features,
            'n_training_features': len(self._training_features) if self._training_features else 0,
            'training_features': self._training_features[:5] if self._training_features else [],  # Show top 5
            'n_total_features': len(self.feature_engineer._feature_names) if self.feature_engineer._feature_names else 0,
            'continual_learning': self.continual_learner.get_status() if self.continual_learner else None,
            'last_prediction': self._last_prediction.direction if self._last_prediction else None
        }

    def get_feature_importance(self, regime: str = None) -> Dict[str, float]:
        """
        Get feature importance rankings.

        Args:
            regime: Market regime (bull/bear/sideways/volatile). If None, uses current.

        Returns:
            Dict of feature name -> importance score
        """
        if not self._is_fitted or self.feature_selector is None:
            return {}

        feature_names = self.feature_engineer._feature_names
        if not feature_names:
            return {}

        # Calculate importance if not already done
        if not self.feature_selector._feature_importance:
            X = np.zeros((100, len(feature_names)))  # Dummy for structure
            y = np.zeros(100)
            self.feature_selector.calculate_importance(
                self.base_models, X, y, feature_names
            )

        return self.feature_selector._feature_importance

    def get_selected_features(self, regime: str) -> List[str]:
        """
        Get selected features for a specific regime.

        Args:
            regime: Market regime (bull/bear/sideways/volatile)

        Returns:
            List of selected feature names
        """
        if not self.adaptive_features:
            return self.feature_engineer._feature_names or []

        # Check cache
        if regime in self._selected_features:
            return self._selected_features[regime]

        # Calculate selection
        try:
            market_regime = MarketRegime(regime.lower())
            feature_names = self.feature_engineer._feature_names or []

            if self.feature_selector and feature_names:
                result = self.feature_selector.select_features(
                    feature_names, market_regime, self.max_features
                )
                self._selected_features[regime] = result.selected_features
                return result.selected_features
        except (ValueError, KeyError):
            pass

        # Fallback to research-backed defaults
        return get_features_for_regime(regime)

    def print_feature_rankings(self, regime: str = 'sideways', top_n: int = 20) -> str:
        """
        Print feature importance rankings.

        Args:
            regime: Market regime
            top_n: Number of top features to show

        Returns:
            Formatted string output
        """
        if self.feature_selector is None:
            return "Adaptive feature selection not enabled"

        feature_names = self.feature_engineer._feature_names or []
        if not feature_names:
            return "No features available (model not fitted?)"

        try:
            market_regime = MarketRegime(regime.lower())
            return self.feature_selector.print_rankings(feature_names, market_regime, top_n)
        except ValueError:
            return f"Invalid regime: {regime}"
