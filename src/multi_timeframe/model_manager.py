"""
Multi-Timeframe Model Manager
==============================

Manages LSTM models per (symbol, interval) combination for continuous learning.

Key Features:
- Load/save models per (symbol, interval)
- Model naming: models/model_BTC_USDT_1h.pt
- Caching for fast access
- Thread-safe operations
- Model versioning support

NO hardcoded values - all from config
NO duplicate code - DRY principle
Centralized model management
"""

import logging
import threading
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from src.analysis_engine import LSTMModel, FeatureCalculator

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a trained model."""
    symbol: str
    interval: str
    trained_at: datetime
    samples_trained: int
    validation_accuracy: float
    validation_confidence: float
    feature_means: Optional[torch.Tensor] = None
    feature_stds: Optional[torch.Tensor] = None
    version: int = 1
    config: Optional[dict] = None


class MultiTimeframeModelManager:
    """
    Manages models per (symbol, interval) for multi-timeframe trading.

    Thread-safe with RLock for concurrent access.

    Model Path Pattern:
        models/model_{symbol}_{interval}.pt
        Example: models/model_BTC_USDT_1h.pt

    Cache Structure:
        {(symbol, interval): LSTMModel}
    """

    def __init__(self, models_dir: str = "models", config: dict = None):
        """
        Initialize multi-timeframe model manager.

        Args:
            models_dir: Directory to store models (from config)
            config: Model configuration dict (from config.yaml)
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or {}

        # Cache: (symbol, interval) -> LSTMModel
        self._model_cache: Dict[Tuple[str, str], LSTMModel] = {}

        # Metadata: (symbol, interval) -> ModelMetadata
        self._metadata_cache: Dict[Tuple[str, str], ModelMetadata] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Stats
        self._stats = {
            'models_loaded': 0,
            'models_saved': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        logger.info(f"MultiTimeframeModelManager initialized (dir: {self.models_dir})")

    def get_model_path(self, symbol: str, interval: str) -> Path:
        """
        Get path for (symbol, interval) model.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            interval: Timeframe (e.g., "1h", "4h")

        Returns:
            Path to model file

        Example:
            >>> manager.get_model_path("BTC/USDT", "1h")
            Path("models/model_BTC_USDT_1h.pt")
        """
        # Sanitize symbol for filename
        safe_symbol = symbol.replace("/", "_").replace("-", "_")

        # Format: model_{symbol}_{interval}.pt (matches MultiCurrencySystem convention)
        filename = f"model_{safe_symbol}_{interval}.pt"

        return self.models_dir / filename

    def model_exists(self, symbol: str, interval: str) -> bool:
        """Check if model file exists for (symbol, interval)."""
        model_path = self.get_model_path(symbol, interval)
        return model_path.exists()

    def load_model(
        self,
        symbol: str,
        interval: str,
        force_reload: bool = False
    ) -> Optional[LSTMModel]:
        """
        Load model for (symbol, interval).

        Args:
            symbol: Trading pair
            interval: Timeframe
            force_reload: If True, bypass cache and reload from disk

        Returns:
            Loaded LSTMModel or None if not found

        Thread-safe: Uses RLock for cache access
        """
        key = (symbol, interval)

        with self._lock:
            # Check cache first (unless force reload)
            if not force_reload and key in self._model_cache:
                self._stats['cache_hits'] += 1
                logger.debug(f"Cache hit for {symbol} @ {interval}")
                return self._model_cache[key]

            self._stats['cache_misses'] += 1

            # Get model path
            model_path = self.get_model_path(symbol, interval)

            if not model_path.exists():
                # Fallback to 1h model if interval-specific model doesn't exist
                if interval != '1h':
                    fallback_path = self.get_model_path(symbol, '1h')
                    if fallback_path.exists():
                        logger.info(f"No {interval} model for {symbol}, falling back to 1h model")
                        model_path = fallback_path
                    else:
                        logger.warning(f"No model found for {symbol} @ {interval} or 1h")
                        return None
                else:
                    logger.warning(f"No model found for {symbol} @ {interval} at {model_path}")
                    return None

            try:
                # Load checkpoint
                checkpoint = torch.load(
                    model_path,
                    map_location='cpu',
                    weights_only=False  # Need to load metadata
                )

                # Get input size from feature columns
                feature_columns = FeatureCalculator.get_feature_columns()
                input_size = len(feature_columns)

                # Create model with config
                model_config = checkpoint.get('config', {})
                model = LSTMModel(
                    input_size=input_size,
                    hidden_size=model_config.get('hidden_size', 128),
                    num_layers=model_config.get('num_layers', 2),
                    dropout=model_config.get('dropout', 0.2),
                    bidirectional=model_config.get('bidirectional', False)
                )

                # Load weights
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()

                # Cache model
                self._model_cache[key] = model

                # Cache metadata
                metadata = ModelMetadata(
                    symbol=symbol,
                    interval=interval,
                    trained_at=checkpoint.get('trained_at', datetime.utcnow()),
                    samples_trained=checkpoint.get('samples_trained', 0),
                    validation_accuracy=checkpoint.get('validation_accuracy', 0.0),
                    validation_confidence=checkpoint.get('validation_confidence', 0.0),
                    feature_means=checkpoint.get('feature_means'),
                    feature_stds=checkpoint.get('feature_stds'),
                    version=checkpoint.get('version', 1),
                    config=model_config
                )
                self._metadata_cache[key] = metadata

                self._stats['models_loaded'] += 1

                logger.info(
                    f"Loaded model for {symbol} @ {interval}: "
                    f"acc={metadata.validation_accuracy:.2%}, "
                    f"conf={metadata.validation_confidence:.2%}"
                )

                return model

            except Exception as e:
                logger.error(f"Failed to load model for {symbol} @ {interval}: {e}", exc_info=True)
                return None

    def save_model(
        self,
        symbol: str,
        interval: str,
        model: LSTMModel,
        metadata: ModelMetadata,
        feature_means: Optional[torch.Tensor] = None,
        feature_stds: Optional[torch.Tensor] = None
    ):
        """
        Save model for (symbol, interval).

        Args:
            symbol: Trading pair
            interval: Timeframe
            model: Trained LSTMModel
            metadata: Model metadata
            feature_means: Feature normalization means
            feature_stds: Feature normalization standard deviations

        Thread-safe: Uses RLock for cache updates
        """
        key = (symbol, interval)
        model_path = self.get_model_path(symbol, interval)

        try:
            with self._lock:
                # Prepare checkpoint
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'symbol': symbol,
                    'interval': interval,
                    'trained_at': metadata.trained_at,
                    'samples_trained': metadata.samples_trained,
                    'validation_accuracy': metadata.validation_accuracy,
                    'validation_confidence': metadata.validation_confidence,
                    'version': metadata.version,
                    'config': metadata.config or {},
                    'feature_means': feature_means,
                    'feature_stds': feature_stds
                }

                # Save to disk
                torch.save(checkpoint, model_path)

                # Update cache
                self._model_cache[key] = model
                self._metadata_cache[key] = metadata

                self._stats['models_saved'] += 1

                logger.info(
                    f"Saved model for {symbol} @ {interval} to {model_path}: "
                    f"acc={metadata.validation_accuracy:.2%}"
                )

        except Exception as e:
            logger.error(f"Failed to save model for {symbol} @ {interval}: {e}", exc_info=True)
            raise

    def get_metadata(self, symbol: str, interval: str) -> Optional[ModelMetadata]:
        """Get metadata for (symbol, interval) model."""
        key = (symbol, interval)

        with self._lock:
            # Check cache
            if key in self._metadata_cache:
                return self._metadata_cache[key]

            # Try loading model to populate cache
            self.load_model(symbol, interval)

            # Return from cache (will be None if load failed)
            return self._metadata_cache.get(key)

    def get_all_models(self, symbol: str) -> Dict[str, LSTMModel]:
        """
        Get all loaded models for a symbol across all intervals.

        Args:
            symbol: Trading pair

        Returns:
            Dict[interval, LSTMModel]
        """
        with self._lock:
            result = {}
            for (sym, interval), model in self._model_cache.items():
                if sym == symbol:
                    result[interval] = model
            return result

    def get_all_intervals(self, symbol: str) -> list:
        """Get all intervals that have models for a symbol."""
        intervals = []
        safe_symbol = symbol.replace("/", "_").replace("-", "_")

        for file in self.models_dir.glob(f"model_{safe_symbol}_*.pt"):
            # Parse filename: model_{symbol}_{interval}.pt
            stem = file.stem  # e.g. "model_BTC_USDT_1h"
            # Remove "model_" prefix, then remove the symbol prefix to get interval
            suffix = stem[len(f"model_{safe_symbol}_"):]
            if suffix:
                intervals.append(suffix)

        return sorted(set(intervals))

    def clear_cache(self, symbol: str = None, interval: str = None):
        """
        Clear model cache.

        Args:
            symbol: If specified, clear only models for this symbol
            interval: If specified (with symbol), clear only that specific model
        """
        with self._lock:
            if symbol is None:
                # Clear all
                self._model_cache.clear()
                self._metadata_cache.clear()
                logger.info("Cleared all model cache")
            elif interval is None:
                # Clear all intervals for symbol
                keys_to_remove = [k for k in self._model_cache.keys() if k[0] == symbol]
                for key in keys_to_remove:
                    del self._model_cache[key]
                    self._metadata_cache.pop(key, None)
                logger.info(f"Cleared cache for {symbol} (all intervals)")
            else:
                # Clear specific model
                key = (symbol, interval)
                self._model_cache.pop(key, None)
                self._metadata_cache.pop(key, None)
                logger.info(f"Cleared cache for {symbol} @ {interval}")

    def delete_model(self, symbol: str, interval: str):
        """Delete model file and remove from cache."""
        key = (symbol, interval)
        model_path = self.get_model_path(symbol, interval)

        with self._lock:
            # Remove from cache
            self._model_cache.pop(key, None)
            self._metadata_cache.pop(key, None)

            # Delete file
            if model_path.exists():
                model_path.unlink()
                logger.info(f"Deleted model for {symbol} @ {interval}")
            else:
                logger.warning(f"Model file not found: {model_path}")

    def get_stats(self) -> dict:
        """Get manager statistics."""
        with self._lock:
            return {
                **self._stats,
                'cached_models': len(self._model_cache),
                'cache_hit_rate': (
                    self._stats['cache_hits'] /
                    (self._stats['cache_hits'] + self._stats['cache_misses'])
                    if (self._stats['cache_hits'] + self._stats['cache_misses']) > 0
                    else 0.0
                )
            }

    def list_all_models(self) -> list:
        """
        List all model files in directory.

        Returns:
            List of dicts with symbol, interval, and path info
        """
        models = []

        for file in self.models_dir.glob("model_*.pt"):
            # Parse filename: model_{symbol}_{interval}.pt
            stem = file.stem  # e.g. "model_BTC_USDT_1h"
            if not stem.startswith("model_"):
                continue

            remainder = stem[len("model_"):]  # e.g. "BTC_USDT_1h"
            # The interval is the last segment; symbol parts are everything before
            parts = remainder.rsplit('_', 1)
            if len(parts) == 2:
                raw_symbol = parts[0]  # e.g. "BTC_USDT"
                interval = parts[1]    # e.g. "1h"
                symbol = raw_symbol.replace("_", "/")

                models.append({
                    'symbol': symbol,
                    'interval': interval,
                    'path': str(file),
                    'size_mb': file.stat().st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(file.stat().st_mtime)
                })

        return sorted(models, key=lambda x: (x['symbol'], x['interval']))


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def get_model_manager(config: dict = None) -> MultiTimeframeModelManager:
    """
    Get shared model manager instance.

    Args:
        config: Configuration dict from config.yaml

    Returns:
        MultiTimeframeModelManager instance
    """
    if config is None:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

    models_dir = config.get('model', {}).get('models_dir', 'models')

    return MultiTimeframeModelManager(models_dir=models_dir, config=config.get('model', {}))
