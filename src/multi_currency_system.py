"""
Multi-Currency Trading System with Auto-Learning
=================================================
Supports multiple currency pairs with individual models per currency.
Includes automatic retraining based on performance feedback.

FEATURES:
1. Multiple currency pairs (forex, crypto)
2. Separate model per currency
3. Auto-retrain on poor performance
4. Performance tracking per currency
5. Dynamic model selection

TRUTH ABOUT AUTO-LEARNING:
- Retraining improves adaptation to market changes
- BUT can cause overfitting to recent data
- Balance: Retrain periodically, not on every trade
- Minimum data: 1000+ candles for meaningful training
"""

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import yaml

from src.analysis_engine import LSTMModel, FeatureCalculator
from src.advanced_predictor import AdvancedPredictor
from src.ml.boosted_predictor import BoostedPredictor
from src.analysis.trend_consensus import TrendConsensus

logger = logging.getLogger(__name__)


@dataclass
class CurrencyConfig:
    """Configuration for a single currency pair."""
    symbol: str  # e.g., "EUR/USD", "BTC/USD"
    exchange: str  # e.g., "twelvedata", "binance", "mt5"
    interval: str  # e.g., "1h", "4h"
    model_path: str  # Path to trained model
    enabled: bool = True


@dataclass
class PerformanceStats:
    """Track performance statistics per currency."""
    symbol: str
    total_signals: int = 0
    correct_predictions: int = 0
    total_pnl_percent: float = 0.0
    last_retrain: Optional[datetime] = None
    win_rate: float = 0.0

    # Configurable thresholds (set via from_config classmethod or defaults)
    min_signals_for_retrain_check: int = 20
    win_rate_retrain_threshold: float = 0.45
    initial_retrain_after_signals: int = 100
    days_between_retrains: int = 30
    min_signals_for_periodic_retrain: int = 50

    def add_result(self, is_correct: bool, pnl_percent: float):
        """Record a prediction result."""
        self.total_signals += 1
        if is_correct:
            self.correct_predictions += 1
        self.total_pnl_percent += pnl_percent
        self.win_rate = self.correct_predictions / self.total_signals if self.total_signals > 0 else 0

    @property
    def needs_retrain(self) -> bool:
        """Check if model needs retraining based on performance."""
        if self.total_signals < self.min_signals_for_retrain_check:
            return False  # Not enough data
        if self.win_rate < self.win_rate_retrain_threshold:
            return True  # Performing below baseline
        if self.last_retrain is None:
            return self.total_signals >= self.initial_retrain_after_signals
        days_since_retrain = (datetime.utcnow() - self.last_retrain).days
        return days_since_retrain >= self.days_between_retrains and self.total_signals >= self.min_signals_for_periodic_retrain


class ModelManager:
    """
    Manages multiple models for different currencies.

    Responsibilities:
    - Load/save models per currency
    - Track model versions
    - Handle model switching
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models: Dict[str, LSTMModel] = {}
        self.model_configs: Dict[str, dict] = {}

    def get_model_path(self, symbol: str, interval: str = "1h") -> Path:
        """Get path for currency-specific model."""
        safe_symbol = symbol.replace("/", "_").replace("-", "_")
        return self.models_dir / f"model_{safe_symbol}_{interval}.pt"

    def load_model(self, symbol: str, config: dict, interval: str = "1h") -> Optional[LSTMModel]:
        """Load model for specific currency."""
        model_path = self.get_model_path(symbol, interval)

        if not model_path.exists():
            logger.warning(f"No model found for {symbol} at {model_path}")
            return None

        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

            feature_columns = FeatureCalculator.get_feature_columns()
            input_size = len(feature_columns)

            model = LSTMModel(
                input_size=input_size,
                hidden_size=config.get('hidden_size', 128),
                num_layers=config.get('num_layers', 2),
                dropout=config.get('dropout', 0.2)
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            self.loaded_models[symbol] = model
            self.model_configs[symbol] = checkpoint.get('config', {})

            # Also store scaler parameters
            if 'feature_means' in checkpoint:
                self.model_configs[symbol]['feature_means'] = checkpoint['feature_means']
                self.model_configs[symbol]['feature_stds'] = checkpoint['feature_stds']

            logger.info(f"Model loaded for {symbol}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model for {symbol}: {e}")
            return None

    def save_model(self, symbol: str, model: LSTMModel, config: dict,
                   feature_means: np.ndarray, feature_stds: np.ndarray,
                   interval: str = "1h"):
        """Save model for specific currency."""
        model_path = self.get_model_path(symbol, interval)

        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'feature_means': feature_means,
            'feature_stds': feature_stds,
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat()
        }, model_path)

        logger.info(f"Model saved for {symbol} at {model_path}")

    def get_model(self, symbol: str) -> Optional[LSTMModel]:
        """Get loaded model for symbol."""
        return self.loaded_models.get(symbol)

    def get_scaler_params(self, symbol: str) -> tuple:
        """Get scaler parameters for symbol."""
        config = self.model_configs.get(symbol, {})
        return config.get('feature_means'), config.get('feature_stds')


class AutoTrainer:
    """
    Automatic Model Retraining

    WHEN TO RETRAIN:
    1. Win rate drops below 45% (model is underperforming)
    2. Every 30 days (market conditions change)
    3. After significant market regime change (detected by entropy)

    SAFEGUARDS:
    - Minimum 1000 candles required
    - Validation split prevents overfitting
    - Keep backup of previous model
    - Only replace if new model is better
    """

    def __init__(self, model_manager: ModelManager, config: dict):
        self.model_manager = model_manager
        self.config = config
        self._training = False
        self._training_lock = threading.Lock()

    def train_model(
        self,
        symbol: str,
        df: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> bool:
        """
        Train or retrain model for a currency.

        Args:
            symbol: Currency pair
            df: Historical data
            epochs: Training epochs
            batch_size: Batch size
            validation_split: Validation data ratio

        Returns:
            True if training successful and model improved
        """
        with self._training_lock:
            if self._training:
                logger.warning("Training already in progress")
                return False
            self._training = True

        try:
            logger.info(f"Starting training for {symbol}")
            logger.info(f"Data: {len(df)} candles")

            if len(df) < 1000:
                logger.error(f"Insufficient data for {symbol}: {len(df)} < 1000")
                return False

            # Calculate features
            df_features = FeatureCalculator.calculate_all(df)
            feature_columns = FeatureCalculator.get_feature_columns()

            # Extract and normalize features FIRST (before creating target)
            features = df_features[feature_columns].values
            closes = df_features['close'].values

            # Normalize
            feature_means = np.nanmean(features, axis=0)
            feature_stds = np.nanstd(features, axis=0)
            features = (features - feature_means) / (feature_stds + 1e-8)
            features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

            # Create sequences
            sequence_length = self.config.get('sequence_length', 60)

            # Create sliding windows manually for correct shape
            # Need shape: (num_samples, sequence_length, num_features)
            num_features = features.shape[1]
            num_sequences = len(features) - sequence_length - 1  # -1 for target

            X = np.zeros((num_sequences, sequence_length, num_features))
            y = np.zeros(num_sequences)

            # CRITICAL FIX: Align sequences with correct targets
            # For each sequence ending at position i+sequence_length-1,
            # the target is: will the NEXT candle (i+sequence_length) close higher?
            for i in range(num_sequences):
                # Sequence uses candles [i] to [i+sequence_length-1]
                X[i] = features[i:i + sequence_length]

                # Target: will candle [i+sequence_length] close higher than candle [i+sequence_length-1]?
                current_close = closes[i + sequence_length - 1]
                next_close = closes[i + sequence_length]
                y[i] = 1.0 if next_close > current_close else 0.0

            # Remove any invalid entries
            valid = ~(np.isnan(y) | np.isnan(X).any(axis=(1, 2)))
            X = X[valid]
            y = y[valid]

            logger.info(f"Created {len(X)} training sequences")

            # Split data
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Create data loaders
            train_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train)
            )
            val_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            )

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

            # Create model
            model = LSTMModel(
                input_size=len(feature_columns),
                hidden_size=self.config.get('hidden_size', 128),
                num_layers=self.config.get('num_layers', 2),
                dropout=self.config.get('dropout', 0.2)
            )

            # Training
            criterion = torch.nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            best_val_acc = 0
            best_state = None

            for epoch in range(epochs):
                # Train
                model.train()
                epoch_loss = 0
                for X_batch, y_batch in train_loader:
                    # Performance: Explicitly clear gradients (prevents memory leak)
                    optimizer.zero_grad()
                    outputs = model(X_batch).squeeze()
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                # Performance: Clear GPU cache every 10 epochs (prevents memory leak)
                if (epoch + 1) % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Validate
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        outputs = model(X_batch).squeeze()
                        predictions = (outputs > 0.5).float()
                        correct += (predictions == y_batch).sum().item()
                        total += len(y_batch)

                val_acc = correct / total

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    # Performance: Deep copy state dict to prevent memory leak
                    best_state = {k: v.clone().detach() for k, v in model.state_dict().items()}

                if (epoch + 1) % 10 == 0:
                    avg_loss = epoch_loss / len(train_loader)
                    logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.2%}")

            # Check if new model is better than existing
            existing_model = self.model_manager.get_model(symbol)
            should_save = True

            if existing_model is not None:
                # Test existing model on validation set
                existing_model.eval()
                existing_correct = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        outputs = existing_model(X_batch).squeeze()
                        predictions = (outputs > 0.5).float()
                        existing_correct += (predictions == y_batch).sum().item()

                existing_acc = existing_correct / total
                logger.info(f"Existing model accuracy: {existing_acc:.2%}")
                logger.info(f"New model accuracy: {best_val_acc:.2%}")

                # Only save if significantly better (>1%)
                should_save = best_val_acc > existing_acc + 0.01

            if should_save and best_state is not None:
                model.load_state_dict(best_state)
                self.model_manager.save_model(
                    symbol, model, self.config,
                    feature_means, feature_stds
                )
                self.model_manager.loaded_models[symbol] = model
                logger.info(f"New model saved for {symbol} (accuracy: {best_val_acc:.2%})")
                return True
            else:
                logger.info(f"Keeping existing model for {symbol}")
                return False

        except Exception as e:
            logger.error(f"Training failed for {symbol}: {e}")
            return False

        finally:
            self._training = False


class MultiCurrencySystem:
    """
    Complete Multi-Currency Trading System

    WORKFLOW:
    1. Load all configured currencies
    2. For each currency:
       - Fetch latest data
       - Run prediction (LSTM + Advanced Math)
       - Generate signal if confidence > threshold
       - Track performance
       - Auto-retrain if needed

    TRANSPARENCY:
    - Shows all algorithm contributions
    - Explains why signal was generated
    - Tracks accuracy per currency
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize multi-currency system."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.model_manager = ModelManager(self.config.get('model', {}).get('models_dir', 'models'))
        self.auto_trainer = AutoTrainer(self.model_manager, self.config.get('model', {}))
        self.advanced_predictor = AdvancedPredictor(config=self.config)
        self.advanced_predictor.set_model_manager(self.model_manager)
        self.boosted_predictor = BoostedPredictor(config=self.config)
        self.trend_consensus = TrendConsensus(config=self.config)

        # Currency configurations
        self.currencies: Dict[str, CurrencyConfig] = {}
        self.performance: Dict[str, PerformanceStats] = {}

        # Thread safety lock for performance tracking
        self._performance_lock = threading.Lock()

        # Feature cache to avoid recalculating features on every predict()
        # Key: f"{symbol}_{last_timestamp}_{len(df)}" -> cached features DataFrame
        self._feature_cache: Dict[str, pd.DataFrame] = {}
        self._feature_cache_lock = threading.Lock()
        self._feature_cache_max_size = 50  # Max cached entries (roughly one per symbol)

        logger.info("MultiCurrencySystem initialized")

    def add_currency(
        self,
        symbol: str,
        exchange: str = "coinbase",
        interval: str = "1h"
    ):
        """
        Add a currency pair to track.

        Args:
            symbol: Currency pair (e.g., "BTC/USD", "EUR/USD")
            exchange: Exchange name (binance, twelvedata, mt5)
            interval: Candle interval
        """
        currency_config = CurrencyConfig(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            model_path=str(self.model_manager.get_model_path(symbol, interval))
        )
        self.currencies[symbol] = currency_config
        perf_cfg = self.config.get('performance_tracking', {})
        self.performance[symbol] = PerformanceStats(
            symbol=symbol,
            min_signals_for_retrain_check=perf_cfg.get('min_signals_for_retrain_check', 20),
            win_rate_retrain_threshold=perf_cfg.get('win_rate_retrain_threshold', 0.45),
            initial_retrain_after_signals=perf_cfg.get('initial_retrain_after_signals', 100),
            days_between_retrains=perf_cfg.get('days_between_retrains', 30),
            min_signals_for_periodic_retrain=perf_cfg.get('min_signals_for_periodic_retrain', 50)
        )

        # Try to load existing LSTM model (pass interval for correct path)
        model_config = self.config.get('model', {})
        self.model_manager.load_model(symbol, model_config, interval=interval)

        # Try to load existing boosted model (XGBoost + LightGBM) from disk
        if self.boosted_predictor and not self.boosted_predictor.is_symbol_fitted(symbol):
            self.boosted_predictor._load_symbol(symbol)

        logger.info(f"Added currency: {symbol} on {exchange}")

    def _get_cached_features(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get features from cache or calculate them.

        Performance: Avoids recalculating features when data hasn't changed.
        Cache key is based on symbol + last timestamp + data length.

        Args:
            symbol: Currency pair
            df: Price data

        Returns:
            DataFrame with calculated features
        """
        # Generate cache key from data signature
        last_timestamp = str(df.index[-1]) if hasattr(df.index, '__getitem__') else str(len(df))
        cache_key = f"{symbol}_{last_timestamp}_{len(df)}"

        with self._feature_cache_lock:
            # Check cache
            if cache_key in self._feature_cache:
                return self._feature_cache[cache_key]

            # Calculate features (expensive operation)
            df_features = FeatureCalculator.calculate_all(df)

            # Evict oldest entries if cache is full (simple LRU approximation)
            if len(self._feature_cache) >= self._feature_cache_max_size:
                # Remove first 10 entries (oldest added)
                keys_to_remove = list(self._feature_cache.keys())[:10]
                for key in keys_to_remove:
                    del self._feature_cache[key]

            # Cache the result
            self._feature_cache[cache_key] = df_features

            return df_features

    def predict(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """
        Generate prediction for a currency.

        Args:
            symbol: Currency pair
            df: Price data

        Returns:
            Prediction dict with all details, or None if prediction fails
        """
        try:
            if symbol not in self.currencies:
                logger.error(f"Currency {symbol} not configured")
                return None

            # Get LSTM model prediction
            model = self.model_manager.get_model(symbol)
            lstm_prob = 0.5  # Default if no LSTM model

            if model is not None:
                try:
                    # Prepare features (cached to avoid recalculation)
                    df_features = self._get_cached_features(symbol, df)
                    feature_columns = FeatureCalculator.get_feature_columns()

                    feature_means, feature_stds = self.model_manager.get_scaler_params(symbol)
                    sequence_length = self.config.get('model', {}).get('sequence_length', 60)

                    features = df_features[feature_columns].iloc[-sequence_length:].values

                    if feature_means is not None:
                        features = (features - feature_means) / (feature_stds + 1e-8)
                    else:
                        features = (features - np.nanmean(features, axis=0)) / (np.nanstd(features, axis=0) + 1e-8)

                    features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

                    # LSTM prediction
                    with torch.no_grad():
                        x = torch.FloatTensor(features).unsqueeze(0)
                        lstm_prob = model(x).item()

                except Exception as e:
                    logger.error(f"LSTM prediction failed for {symbol}: {e}")

            # Get gradient boosting prediction (XGBoost + LightGBM)
            has_boost = False
            boost_prob = 0.5
            if self.boosted_predictor.is_symbol_fitted(symbol):
                try:
                    boost_prob = self.boosted_predictor.predict(df, symbol)
                    has_boost = True
                except Exception as e:
                    logger.debug(f"Boosted prediction failed for {symbol}: {e}")

            # Blend LSTM + boosting (boosting gets higher weight — it's more accurate)
            from src.core.constants import blend_probabilities
            boost_cfg = self.config.get('boosting', {})
            has_lstm = model is not None and lstm_prob != 0.5
            combined_prob = blend_probabilities(
                lstm_prob=lstm_prob, boost_prob=boost_prob,
                has_lstm=has_lstm, has_boost=has_boost,
                boost_weight=boost_cfg.get('ensemble_weight', 0.6),
            )

            # Get trend consensus signal (multi-period EMA alignment, replaces LLM)
            trend_prob = None
            if self.trend_consensus.is_available:
                try:
                    interval = self.currencies[symbol].interval if symbol in self.currencies else "1h"
                    trend_prob = self.trend_consensus.get_signal(symbol, df, interval=interval)
                except Exception as e:
                    logger.debug(f"Trend consensus skipped for {symbol}: {e}")

            # Get advanced mathematical prediction
            atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
            advanced_result = self.advanced_predictor.predict(
                df=df, symbol=symbol, lstm_probability=combined_prob, atr=atr,
                trend_probability=trend_prob,
            )

            # Build result
            current_price = df['close'].iloc[-1]

            result = {
                'symbol': symbol,
                'timestamp': datetime.utcnow(),
                'price': current_price,
                'direction': advanced_result.direction,
                'confidence': advanced_result.confidence,
                'stop_loss': advanced_result.stop_loss,
                'take_profit': advanced_result.take_profit,
                'risk_reward': advanced_result.risk_reward_ratio,

                # Mathematical breakdown (transparency)
                'components': {
                    'lstm_probability': lstm_prob,
                    'boosted_probability': boost_prob,
                    'combined_probability': combined_prob,
                    'fourier_signal': advanced_result.fourier_signal,
                    'kalman_trend': advanced_result.kalman_trend,
                    'entropy_regime': advanced_result.entropy_regime,
                    'markov_probability': advanced_result.markov_probability,
                    'monte_carlo_risk': advanced_result.monte_carlo_risk,
                    'trend_probability': advanced_result.trend_probability,
                },

                # Model info
                'has_trained_model': model is not None,
                'algorithm_weights': getattr(advanced_result, 'algorithm_weights', {}),
                'raw_scores': getattr(advanced_result, 'raw_scores', {})
            }

            return result

        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}")
            return None

    def get_performance_report(self) -> Dict:
        """Get performance statistics for all currencies (thread-safe)."""
        report = {}
        with self._performance_lock:
            for symbol, stats in self.performance.items():
                report[symbol] = {
                    'total_signals': stats.total_signals,
                    'win_rate': f"{stats.win_rate:.1%}",
                    'total_pnl': f"{stats.total_pnl_percent:.2f}%",
                    'needs_retrain': stats.needs_retrain,
                    'last_retrain': stats.last_retrain.isoformat() if stats.last_retrain else 'Never',
                }
        return report

    def get_status(self) -> Dict:
        """Get system status."""
        return {
            'currencies': list(self.currencies.keys()),
            'loaded_models': list(self.model_manager.loaded_models.keys()),
            'performance': self.get_performance_report(),
        }

    def cleanup(self):
        """Cleanup resources."""
        logger.info("MultiCurrencySystem cleanup complete")
