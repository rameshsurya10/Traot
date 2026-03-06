"""
Model Factory
==============

Creates and configures base models for the ensemble.
Handles model availability and provides fallbacks.
"""

import numpy as np
import torch
from typing import Dict, Any
import logging

from .tcn_lstm import TCNLSTMAttention, SimpleLSTM
from .boosting import (
    XGBoostModel,
    LightGBMModel,
    CatBoostModel,
    NGBoostModel,
    SklearnBoostingModel,
    get_available_boosting_models
)

logger = logging.getLogger(__name__)


def create_base_models(
    input_size: int,
    hidden_size: int = 128,
    sequence_length: int = 60,
    use_gpu: bool = True,
    include_ngboost: bool = True
) -> Dict[str, Any]:
    """
    Create all base models for the ensemble.

    Args:
        input_size: Number of input features
        hidden_size: Hidden layer size for neural networks
        sequence_length: Sequence length for time series
        use_gpu: Whether to use GPU for PyTorch models
        include_ngboost: Whether to include NGBoost for uncertainty

    Returns:
        Dictionary of model name -> model instance
    """
    models = {}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

    # 1. TCN-LSTM-Attention (primary deep learning model)
    try:
        tcn_lstm = TCNLSTMAttention(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            tcn_channels=[64, 64, 128],
            kernel_size=3,
            n_heads=4,
            dropout=0.2
        ).to(device)
        models['tcn_lstm_attention'] = tcn_lstm
        logger.info(f"TCN-LSTM-Attention created on {device}")
    except Exception as e:
        logger.warning(f"Failed to create TCN-LSTM-Attention: {e}")
        # Fallback to simple LSTM
        simple_lstm = SimpleLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.2
        ).to(device)
        models['simple_lstm'] = simple_lstm
        logger.info("Using SimpleLSTM as fallback")

    # 2. XGBoost
    available_boosting = get_available_boosting_models()

    if 'xgboost' in available_boosting:
        models['xgboost'] = XGBoostModel(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0
        )
        logger.info("XGBoost model created")

    # 3. LightGBM
    if 'lightgbm' in available_boosting:
        models['lightgbm'] = LightGBMModel(
            n_estimators=100,
            num_leaves=31,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0
        )
        logger.info("LightGBM model created")

    # 4. CatBoost
    if 'catboost' in available_boosting:
        models['catboost'] = CatBoostModel(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            l2_leaf_reg=3.0
        )
        logger.info("CatBoost model created")

    # 5. NGBoost (for uncertainty quantification)
    if include_ngboost and 'ngboost' in available_boosting:
        models['ngboost'] = NGBoostModel(
            n_estimators=100,
            learning_rate=0.1
        )
        logger.info("NGBoost model created for uncertainty quantification")

    # 6. Sklearn fallback if no boosting models available
    if len([k for k in models.keys() if k not in ['tcn_lstm_attention', 'simple_lstm']]) == 0:
        models['sklearn_gbm'] = SklearnBoostingModel(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        )
        logger.info("Using sklearn GBM as fallback")

    logger.info(f"Created {len(models)} base models: {list(models.keys())}")
    return models


def create_regime_specific_models(
    input_size: int,
    n_regimes: int = 3,
    hidden_size: int = 128
) -> Dict[str, Dict[str, Any]]:
    """
    Create separate model sets for each regime.

    Args:
        input_size: Number of input features
        n_regimes: Number of market regimes
        hidden_size: Hidden layer size

    Returns:
        Dictionary of regime -> {model_name -> model}
    """
    regime_names = ['bull', 'bear', 'sideways'][:n_regimes]
    regime_models = {}

    for regime in regime_names:
        regime_models[regime] = create_base_models(
            input_size=input_size,
            hidden_size=hidden_size,
            include_ngboost=(regime == 'bear')  # NGBoost especially important for bear markets
        )
        logger.info(f"Created model set for {regime} regime")

    return regime_models


class ModelWrapper:
    """
    Unified wrapper for different model types.

    Provides consistent interface for:
    - PyTorch models
    - Sklearn-style models
    - Boosting models
    """

    def __init__(self, model: Any, model_type: str = 'auto'):
        self.model = model
        self.model_type = self._detect_type(model) if model_type == 'auto' else model_type
        self._device = None

    def _detect_type(self, model: Any) -> str:
        """Detect model type."""
        if isinstance(model, torch.nn.Module):
            return 'pytorch'
        elif hasattr(model, 'fit') and hasattr(model, 'predict_proba'):
            return 'sklearn'
        else:
            return 'unknown'

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit model with unified interface."""
        if self.model_type == 'pytorch':
            # PyTorch models need special training loop
            raise NotImplementedError("Use dedicated PyTorch training function")
        else:
            return self.model.fit(X, y, **kwargs)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model_type == 'pytorch':
            self.model.eval()
            with torch.no_grad():
                if isinstance(X, np.ndarray):
                    X = torch.FloatTensor(X)
                if self._device:
                    X = X.to(self._device)
                probs = self.model(X).cpu().numpy()
                # Return as 2-column probability (class 0, class 1)
                return np.column_stack([1 - probs.flatten(), probs.flatten()])
        else:
            return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get class predictions."""
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)

    def to(self, device: torch.device) -> 'ModelWrapper':
        """Move model to device (for PyTorch models)."""
        if self.model_type == 'pytorch':
            self.model = self.model.to(device)
            self._device = device
        return self

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        if self.model_type == 'pytorch':
            # PyTorch models are "fitted" if they have been trained
            return True  # Assume trained after creation
        else:
            return getattr(self.model, '_is_fitted', False) or getattr(self.model, 'is_fitted', False)
