"""
Gradient Boosting Models
=========================

Implements XGBoost, LightGBM, and CatBoost for the ensemble.
These models excel at tabular data and provide diversity in the ensemble.

Research findings:
- XGBoost: 55.9% accuracy on BTC (Neptune.ai 2024)
- LightGBM: Fastest training, great for large data
- CatBoost: Best for categorical features

All models wrapped with sklearn-compatible interface.
"""

import numpy as np
from typing import Optional, Dict
from abc import ABC, abstractmethod
import logging
import warnings

# Suppress specific warnings from boosting libraries (not all warnings globally)
warnings.filterwarnings('ignore', category=FutureWarning, module='xgboost')
warnings.filterwarnings('ignore', category=FutureWarning, module='lightgbm')
warnings.filterwarnings('ignore', category=UserWarning, module='catboost')
logger = logging.getLogger(__name__)


class BaseBoostingModel(ABC):
    """Abstract base class for boosting models."""

    def __init__(self, **params):
        self.params = params
        self.model = None
        self._is_fitted = False
        self._feature_importance = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseBoostingModel':
        """Fit the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        pass

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def feature_importance(self) -> Optional[np.ndarray]:
        return self._feature_importance


class XGBoostModel(BaseBoostingModel):
    """
    XGBoost wrapper for trading prediction.

    XGBoost excels at:
    - Handling missing values automatically
    - Built-in regularization (L1/L2)
    - Feature importance for interpretability
    - Robust to overfitting

    Default parameters optimized for trading:
    - n_estimators: 100 (balanced speed/accuracy)
    - max_depth: 6 (prevent overfitting)
    - learning_rate: 0.1 (standard)
    - subsample: 0.8 (bagging for robustness)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[list] = None,
        early_stopping_rounds: Optional[int] = 20,
        verbose: bool = False
    ) -> 'XGBoostModel':
        """Fit XGBoost model."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

        # Build model params - early_stopping_rounds goes to constructor in newer XGBoost
        model_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            **self.params
        }

        # Add early stopping to constructor if eval_set provided
        if eval_set is not None and early_stopping_rounds:
            model_params['early_stopping_rounds'] = early_stopping_rounds

        self.model = xgb.XGBClassifier(**model_params)

        fit_params = {'verbose': verbose}

        if eval_set is not None:
            fit_params['eval_set'] = eval_set

        self.model.fit(X, y, **fit_params)
        self._is_fitted = True
        self._feature_importance = self.model.feature_importances_

        logger.info(f"XGBoost fitted with {self.params['n_estimators']} estimators")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(X)


class LightGBMModel(BaseBoostingModel):
    """
    LightGBM wrapper for trading prediction.

    LightGBM advantages:
    - Fastest training among boosting methods
    - Lower memory usage
    - Better accuracy on large datasets
    - Handles categorical features efficiently

    Uses leaf-wise tree growth (faster than level-wise).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = -1,  # No limit by default
        num_leaves: int = 31,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_samples: int = 20,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_samples=min_child_samples,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[list] = None,
        early_stopping_rounds: Optional[int] = 20,
        verbose: bool = False
    ) -> 'LightGBMModel':
        """Fit LightGBM model."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")

        callbacks = []
        if not verbose:
            callbacks.append(lgb.log_evaluation(period=0))
        if early_stopping_rounds and eval_set:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))

        self.model = lgb.LGBMClassifier(
            objective='binary',
            verbose=-1,
            **self.params
        )

        fit_params = {}
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
        if callbacks:
            fit_params['callbacks'] = callbacks

        self.model.fit(X, y, **fit_params)
        self._is_fitted = True
        self._feature_importance = self.model.feature_importances_

        logger.info(f"LightGBM fitted with {self.params['n_estimators']} estimators")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(X)


class CatBoostModel(BaseBoostingModel):
    """
    CatBoost wrapper for trading prediction.

    CatBoost advantages:
    - Best handling of categorical features (one-hot not needed)
    - Ordered boosting (reduces overfitting)
    - GPU support out of the box
    - Robust to hyperparameter choices

    Uses symmetric trees for faster prediction.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        l2_leaf_reg: float = 3.0,
        random_state: int = 42,
        thread_count: int = -1,
        **kwargs
    ):
        super().__init__(
            iterations=n_estimators,
            depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            l2_leaf_reg=l2_leaf_reg,
            random_seed=random_state,
            thread_count=thread_count,
            **kwargs
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[list] = None,
        early_stopping_rounds: Optional[int] = 20,
        verbose: bool = False
    ) -> 'CatBoostModel':
        """Fit CatBoost model."""
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            raise ImportError("CatBoost not installed. Run: pip install catboost")

        # Build model params
        model_params = {
            'loss_function': 'Logloss',
            'verbose': verbose,
            **self.params
        }

        # Add early stopping to constructor if eval_set will be provided
        if eval_set is not None and early_stopping_rounds:
            model_params['early_stopping_rounds'] = early_stopping_rounds

        self.model = CatBoostClassifier(**model_params)

        fit_params = {}
        if eval_set is not None:
            fit_params['eval_set'] = eval_set

        self.model.fit(X, y, **fit_params)
        self._is_fitted = True
        self._feature_importance = self.model.feature_importances_

        logger.info(f"CatBoost fitted with {self.params['iterations']} iterations")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(X)


class NGBoostModel(BaseBoostingModel):
    """
    NGBoost (Natural Gradient Boosting) for uncertainty quantification.

    Unlike standard boosting which outputs point estimates, NGBoost
    outputs a full probability distribution, enabling:
    - Confidence intervals
    - Uncertainty quantification
    - Risk-aware predictions

    Critical for trading risk management.

    Research: Stanford ML Group 2019, still state-of-art for UQ.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        minibatch_frac: float = 1.0,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            minibatch_frac=minibatch_frac,
            random_state=random_state,
            **kwargs
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[list] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = False
    ) -> 'NGBoostModel':
        """Fit NGBoost model."""
        try:
            from ngboost import NGBClassifier
            from ngboost.distns import Bernoulli
        except ImportError:
            raise ImportError("NGBoost not installed. Run: pip install ngboost")

        self.model = NGBClassifier(
            Dist=Bernoulli,
            n_estimators=self.params['n_estimators'],
            learning_rate=self.params['learning_rate'],
            minibatch_frac=self.params['minibatch_frac'],
            random_state=self.params['random_state'],
            verbose=verbose
        )

        if eval_set is not None:
            X_val, y_val = eval_set[0]
            self.model.fit(X, y, X_val=X_val, Y_val=y_val)
        else:
            self.model.fit(X, y)

        self._is_fitted = True

        logger.info(f"NGBoost fitted with {self.params['n_estimators']} estimators")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(X)

    def predict_dist(self, X: np.ndarray):
        """
        Get full probability distribution for uncertainty quantification.

        Returns distribution object that can be used to get:
        - Mean: dist.mean()
        - Variance: dist.var()
        - Confidence intervals: dist.ppf([0.025, 0.975])
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.pred_dist(X)


class SklearnBoostingModel(BaseBoostingModel):
    """
    Sklearn GradientBoostingClassifier fallback.

    Used when XGBoost/LightGBM/CatBoost are not installed.
    Slower but always available.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=random_state,
            **kwargs
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[list] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = False
    ) -> 'SklearnBoostingModel':
        """Fit sklearn GradientBoostingClassifier."""
        from sklearn.ensemble import GradientBoostingClassifier

        self.model = GradientBoostingClassifier(
            n_estimators=self.params['n_estimators'],
            max_depth=self.params['max_depth'],
            learning_rate=self.params['learning_rate'],
            subsample=self.params['subsample'],
            random_state=self.params['random_state'],
            verbose=int(verbose)
        )

        self.model.fit(X, y)
        self._is_fitted = True
        self._feature_importance = self.model.feature_importances_

        logger.info(f"Sklearn GBM fitted with {self.params['n_estimators']} estimators")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(X)


def get_available_boosting_models() -> Dict[str, type]:
    """Get dictionary of available boosting models."""
    models = {'sklearn': SklearnBoostingModel}

    try:
        import xgboost
        models['xgboost'] = XGBoostModel
    except ImportError:
        logger.warning("XGBoost not available")

    try:
        import lightgbm
        models['lightgbm'] = LightGBMModel
    except ImportError:
        logger.warning("LightGBM not available")

    try:
        import catboost
        models['catboost'] = CatBoostModel
    except ImportError:
        logger.warning("CatBoost not available")

    try:
        import ngboost
        models['ngboost'] = NGBoostModel
    except ImportError:
        logger.warning("NGBoost not available")

    return models
