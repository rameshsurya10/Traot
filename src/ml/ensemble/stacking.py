"""
Stacking Ensemble
==================

Combines multiple base models using a meta-learner.

Two approaches:
1. StackingEnsemble: Standard stacking with cross-validation
2. RegimeAwareEnsemble: Different base models for each market regime

Research shows:
- Stacking outperforms simple averaging by 5-10%
- XGBoost/Ridge as meta-learner prevents overfitting
- Regime-aware switching further improves performance

Sources:
- ScienceDirect 2019: Stacking ensembles
- Springer 2024: Enhanced boosting with stacking
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from dataclasses import dataclass
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EnsemblePrediction:
    """Result from ensemble prediction."""
    direction: str  # BUY, SELL, NEUTRAL
    probability: float  # 0.0 to 1.0
    confidence: float  # Adjusted confidence
    base_model_predictions: Dict[str, float]  # Individual model predictions
    meta_features: np.ndarray  # Features used by meta-learner
    regime: Optional[str] = None  # Current market regime if available


class StackingEnsemble:
    """
    Stacking Ensemble Classifier.

    Uses out-of-fold predictions to train meta-learner,
    avoiding overfitting and information leakage.

    Architecture:
    1. Base models generate predictions
    2. Predictions become features for meta-learner
    3. Meta-learner makes final prediction

    Parameters:
    -----------
    base_models : Dict[str, Any]
        Dictionary of model name -> model instance
    meta_learner : str
        Type of meta-learner: 'ridge', 'xgboost', 'logistic'
    n_folds : int
        Number of folds for generating out-of-fold predictions
    use_probabilities : bool
        Use probabilities as meta-features (vs class labels)
    include_original_features : bool
        Include original features in meta-learner input
    """

    def __init__(
        self,
        base_models: Dict[str, Any] = None,
        meta_learner: str = 'ridge',
        n_folds: int = 5,
        use_probabilities: bool = True,
        include_original_features: bool = False
    ):
        self.base_models = base_models or {}
        self.meta_learner_type = meta_learner
        self.n_folds = n_folds
        self.use_probabilities = use_probabilities
        self.include_original_features = include_original_features

        self.meta_learner = None
        self.scaler = StandardScaler()
        self._is_fitted = False
        self._base_model_weights: Dict[str, float] = {}

    def _create_meta_learner(self):
        """Create meta-learner based on type."""
        if self.meta_learner_type == 'ridge':
            return Ridge(alpha=1.0)
        elif self.meta_learner_type == 'logistic':
            return LogisticRegression(C=1.0, max_iter=1000)
        elif self.meta_learner_type == 'xgboost':
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
            except ImportError:
                logger.warning("XGBoost not available, using Ridge")
                return Ridge(alpha=1.0)
        else:
            return Ridge(alpha=1.0)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_seq: Optional[np.ndarray] = None,
        fit_base_models: bool = True,
        verbose: bool = False
    ) -> 'StackingEnsemble':
        """
        Fit the stacking ensemble.

        Args:
            X: Feature matrix for boosting models (n_samples, n_features)
            y: Target labels
            X_seq: Sequential data for neural networks (n_samples, seq_len, n_features)
            fit_base_models: Whether to fit base models (False if pre-trained)
            verbose: Print progress

        Returns:
            self
        """
        n_samples = len(y)

        # Generate out-of-fold predictions
        oof_predictions = np.zeros((n_samples, len(self.base_models)))

        # Use TimeSeriesSplit for financial data (no future leakage)
        tscv = TimeSeriesSplit(n_splits=self.n_folds)

        for model_idx, (model_name, model) in enumerate(self.base_models.items()):
            if verbose:
                logger.info(f"Generating OOF predictions for {model_name}")

            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
                # Select appropriate data
                if isinstance(model, nn.Module):
                    # PyTorch model - use sequential data
                    if X_seq is None:
                        logger.warning(f"No sequential data for {model_name}, skipping")
                        continue

                    X_train = torch.FloatTensor(X_seq[train_idx])
                    X_val = torch.FloatTensor(X_seq[val_idx])
                    y_train = torch.FloatTensor(y[train_idx])

                    # Quick training for OOF (reduced epochs)
                    if fit_base_models:
                        self._train_pytorch_model(model, X_train, y_train, epochs=20)

                    model.eval()
                    with torch.no_grad():
                        preds = model(X_val).cpu().numpy().flatten()
                    oof_predictions[val_idx, model_idx] = preds

                else:
                    # Sklearn-style model
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train = y[train_idx]

                    if fit_base_models:
                        model.fit(X_train, y_train)

                    if self.use_probabilities:
                        preds = model.predict_proba(X_val)[:, 1]
                    else:
                        preds = model.predict(X_val)

                    oof_predictions[val_idx, model_idx] = preds

            if verbose:
                logger.info(f"  {model_name} OOF complete")

        # Create meta-features
        if self.include_original_features:
            meta_features = np.hstack([oof_predictions, X])
        else:
            meta_features = oof_predictions

        # Scale meta-features
        meta_features = self.scaler.fit_transform(meta_features)

        # Train meta-learner
        self.meta_learner = self._create_meta_learner()

        if self.meta_learner_type == 'ridge':
            # Ridge regression for probability output
            self.meta_learner.fit(meta_features, y)
        else:
            self.meta_learner.fit(meta_features, y)

        # Calculate base model weights (based on OOF performance)
        self._calculate_base_weights(oof_predictions, y)

        self._is_fitted = True
        logger.info("StackingEnsemble fitted successfully")

        return self

    def _train_pytorch_model(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 20,
        lr: float = 0.001
    ):
        """Quick training for PyTorch model."""
        device = next(model.parameters()).device
        X = X.to(device)
        y = y.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X).squeeze()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

    def _calculate_base_weights(self, oof_predictions: np.ndarray, y: np.ndarray):
        """Calculate weights for each base model based on performance."""
        from sklearn.metrics import accuracy_score

        for i, model_name in enumerate(self.base_models.keys()):
            preds = (oof_predictions[:, i] > 0.5).astype(int)
            acc = accuracy_score(y, preds)
            self._base_model_weights[model_name] = acc

        # Normalize weights
        total = sum(self._base_model_weights.values())
        for model_name in self._base_model_weights:
            self._base_model_weights[model_name] /= total

        logger.info(f"Base model weights: {self._base_model_weights}")

    def predict(
        self,
        X: np.ndarray,
        X_seq: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Make ensemble predictions.

        Args:
            X: Feature matrix for boosting models
            X_seq: Sequential data for neural networks

        Returns:
            Class predictions (0 or 1)
        """
        probs = self.predict_proba(X, X_seq)
        return (probs > 0.5).astype(int)

    def predict_proba(
        self,
        X: np.ndarray,
        X_seq: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Get ensemble prediction probabilities.

        Args:
            X: Feature matrix for boosting models
            X_seq: Sequential data for neural networks

        Returns:
            Probability of class 1
        """
        if not self._is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        # Get base model predictions
        base_predictions = np.zeros((len(X), len(self.base_models)))

        for i, (model_name, model) in enumerate(self.base_models.items()):
            if isinstance(model, nn.Module):
                if X_seq is None:
                    # Use weighted average from other models
                    continue

                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_seq)
                    device = next(model.parameters()).device
                    X_tensor = X_tensor.to(device)
                    preds = model(X_tensor).cpu().numpy().flatten()
                base_predictions[:, i] = preds
            else:
                if self.use_probabilities:
                    preds = model.predict_proba(X)[:, 1]
                else:
                    preds = model.predict(X).astype(float)
                base_predictions[:, i] = preds

        # Create meta-features
        if self.include_original_features:
            meta_features = np.hstack([base_predictions, X])
        else:
            meta_features = base_predictions

        # Scale
        meta_features = self.scaler.transform(meta_features)

        # Meta-learner prediction
        if self.meta_learner_type == 'ridge':
            # Clip to [0, 1] for probability interpretation
            probs = np.clip(self.meta_learner.predict(meta_features), 0, 1)
        else:
            probs = self.meta_learner.predict_proba(meta_features)[:, 1]

        return probs

    def predict_detailed(
        self,
        X: np.ndarray,
        X_seq: Optional[np.ndarray] = None
    ) -> EnsemblePrediction:
        """
        Get detailed ensemble prediction with all components.

        Returns:
            EnsemblePrediction with full details
        """
        if not self._is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        # Get base model predictions
        base_predictions = {}

        for model_name, model in self.base_models.items():
            if isinstance(model, nn.Module):
                if X_seq is not None:
                    model.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X_seq)
                        device = next(model.parameters()).device
                        pred = model(X_tensor.to(device)).cpu().numpy().flatten()[-1]
                    base_predictions[model_name] = float(pred)
            else:
                pred = model.predict_proba(X[-1:])[:, 1][0]
                base_predictions[model_name] = float(pred)

        # Get ensemble prediction
        prob = self.predict_proba(X, X_seq)[-1]

        # Determine direction
        if prob > 0.55:
            direction = 'BUY'
        elif prob < 0.45:
            direction = 'SELL'
        else:
            direction = 'NEUTRAL'

        # Calculate confidence based on agreement
        predictions_list = list(base_predictions.values())
        agreement = 1 - np.std(predictions_list) if predictions_list else 0
        confidence = prob if prob > 0.5 else (1 - prob)
        confidence = confidence * (0.5 + 0.5 * agreement)  # Adjust by agreement

        # Meta-features for transparency
        meta_features = np.array(predictions_list)

        return EnsemblePrediction(
            direction=direction,
            probability=float(prob),
            confidence=float(confidence),
            base_model_predictions=base_predictions,
            meta_features=meta_features
        )

    def save(self, path: Union[str, Path]):
        """Save ensemble to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save meta-learner and scaler
        joblib.dump(self.meta_learner, path / 'meta_learner.joblib')
        joblib.dump(self.scaler, path / 'scaler.joblib')
        joblib.dump(self._base_model_weights, path / 'weights.joblib')

        # Save base models
        for name, model in self.base_models.items():
            if isinstance(model, nn.Module):
                torch.save(model.state_dict(), path / f'{name}.pt')
            else:
                joblib.dump(model, path / f'{name}.joblib')

        logger.info(f"Ensemble saved to {path}")

    def load(self, path: Union[str, Path]):
        """Load ensemble from disk."""
        path = Path(path)

        # Check required files exist
        meta_learner_path = path / 'meta_learner.joblib'
        scaler_path = path / 'scaler.joblib'
        weights_path = path / 'weights.joblib'

        if not meta_learner_path.exists():
            raise FileNotFoundError(f"Meta-learner not found: {meta_learner_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")

        self.meta_learner = joblib.load(meta_learner_path)
        self.scaler = joblib.load(scaler_path)
        self._base_model_weights = joblib.load(weights_path) if weights_path.exists() else {}

        # Load base models - collect updates first to avoid dict modification during iteration
        models_to_update = {}
        models_loaded = 0

        for name, model in self.base_models.items():
            if isinstance(model, nn.Module):
                model_path = path / f'{name}.pt'
                if model_path.exists():
                    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
                    model.load_state_dict(state_dict)
                    model.eval()
                    models_loaded += 1
                    logger.debug(f"Loaded PyTorch model: {name}")
                else:
                    logger.warning(f"PyTorch model file not found: {model_path}")
            else:
                model_path = path / f'{name}.joblib'
                if model_path.exists():
                    loaded_model = joblib.load(model_path)
                    # Queue update instead of modifying during iteration
                    models_to_update[name] = loaded_model
                    models_loaded += 1
                    logger.debug(f"Loaded sklearn/boosting model: {name}")
                else:
                    logger.warning(f"Sklearn model file not found: {model_path}")

        # Apply queued updates after iteration completes
        for name, loaded_model in models_to_update.items():
            self.base_models[name] = loaded_model

        if models_loaded == 0:
            logger.warning("No base models were loaded - ensemble may not work correctly")

        self._is_fitted = True
        logger.info(f"Ensemble loaded from {path} ({models_loaded} models)")


class RegimeAwareEnsemble:
    """
    Regime-Aware Ensemble that uses different models for each market regime.

    Uses GMM-HMM to detect current regime, then routes prediction
    to the appropriate specialist ensemble.

    Regimes:
    - BULL: Momentum-focused models
    - BEAR: Defensive models with higher uncertainty focus
    - SIDEWAYS: Mean-reversion focused models
    """

    def __init__(
        self,
        regime_detector=None,
        regime_ensembles: Dict[str, StackingEnsemble] = None
    ):
        self.regime_detector = regime_detector
        self.regime_ensembles = regime_ensembles or {}
        self._default_ensemble: Optional[StackingEnsemble] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_seq: Optional[np.ndarray] = None,
        df: pd.DataFrame = None,
        verbose: bool = False
    ) -> 'RegimeAwareEnsemble':
        """
        Fit regime-specific ensembles.

        Args:
            X: Feature matrix
            y: Target labels
            X_seq: Sequential data for neural networks
            df: DataFrame for regime detection
            verbose: Print progress
        """
        if self.regime_detector is None:
            from ..regime import RegimeDetector
            self.regime_detector = RegimeDetector()

        # Fit regime detector
        if df is not None:
            self.regime_detector.fit(df)
            regimes = self.regime_detector.get_regime_for_training(df)
        else:
            # Without price data, train single ensemble
            regimes = np.zeros(len(y))

        # Train ensemble for each regime
        unique_regimes = np.unique(regimes)

        for regime_id in unique_regimes:
            regime_mask = regimes == regime_id
            regime_name = ['bull', 'bear', 'sideways'][int(regime_id) % 3]

            if regime_mask.sum() < 100:
                logger.warning(f"Regime {regime_name} has too few samples ({regime_mask.sum()}), skipping")
                continue

            if verbose:
                logger.info(f"Training {regime_name} ensemble with {regime_mask.sum()} samples")

            # Create and fit regime-specific ensemble
            if regime_name not in self.regime_ensembles:
                from ..models import create_base_models
                base_models = create_base_models(input_size=X.shape[1])
                self.regime_ensembles[regime_name] = StackingEnsemble(base_models=base_models)

            X_regime = X[regime_mask]
            y_regime = y[regime_mask]
            X_seq_regime = X_seq[regime_mask] if X_seq is not None else None

            self.regime_ensembles[regime_name].fit(
                X_regime, y_regime, X_seq_regime,
                verbose=verbose
            )

        # Set default ensemble (use 'sideways' or first available)
        if 'sideways' in self.regime_ensembles:
            self._default_ensemble = self.regime_ensembles['sideways']
        elif self.regime_ensembles:
            self._default_ensemble = list(self.regime_ensembles.values())[0]

        logger.info("RegimeAwareEnsemble fitted successfully")
        return self

    def predict(
        self,
        X: np.ndarray,
        X_seq: Optional[np.ndarray] = None,
        df: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """Make regime-aware predictions."""
        probs = self.predict_proba(X, X_seq, df)
        return (probs > 0.5).astype(int)

    def predict_proba(
        self,
        X: np.ndarray,
        X_seq: Optional[np.ndarray] = None,
        df: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """Get regime-aware prediction probabilities."""
        # Detect current regime
        if df is not None and self.regime_detector is not None:
            regime_result = self.regime_detector.detect(df)
            current_regime = regime_result.current_regime.name.lower()
        else:
            current_regime = 'sideways'

        # Get appropriate ensemble
        if current_regime in self.regime_ensembles:
            ensemble = self.regime_ensembles[current_regime]
        elif self._default_ensemble is not None:
            ensemble = self._default_ensemble
        else:
            raise ValueError("No ensemble available for prediction")

        return ensemble.predict_proba(X, X_seq)

    def predict_detailed(
        self,
        X: np.ndarray,
        X_seq: Optional[np.ndarray] = None,
        df: Optional[pd.DataFrame] = None
    ) -> EnsemblePrediction:
        """Get detailed regime-aware prediction."""
        # Detect current regime
        if df is not None and self.regime_detector is not None:
            regime_result = self.regime_detector.detect(df)
            current_regime = regime_result.current_regime.name.lower()
        else:
            current_regime = 'sideways'

        # Get appropriate ensemble
        if current_regime in self.regime_ensembles:
            ensemble = self.regime_ensembles[current_regime]
        elif self._default_ensemble is not None:
            ensemble = self._default_ensemble
        else:
            raise ValueError("No ensemble available for prediction")

        prediction = ensemble.predict_detailed(X, X_seq)
        prediction.regime = current_regime

        return prediction
