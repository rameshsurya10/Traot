"""
Boosted Predictor
==================

XGBoost + LightGBM ensemble for trading predictions.
Uses pandas-ta technical indicators as features.

This integrates into the existing MultiCurrencySystem alongside LSTM,
providing a much stronger probability signal than math-only algorithms.

Research basis:
- XGBoost: 55.9% accuracy on BTC (Neptune.ai 2024)
- LightGBM: Fastest training, great for large data
- Gradient boosting consistently outperforms deep learning on tabular financial data
"""

import logging
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BoostedPredictor:
    """
    XGBoost + LightGBM ensemble predictor with pandas-ta features.

    Trained per-symbol on historical OHLCV data.
    Returns a probability (0.0-1.0) that can be blended with LSTM.
    """

    def __init__(self, config: dict = None):
        cfg = (config or {}).get('boosting', {})
        self._models: Dict[str, Dict] = {}  # symbol -> {'xgb': model, 'lgb': model, ...}
        self._feature_names: Dict[str, List[str]] = {}  # symbol -> feature names
        self._models_dir = (config or {}).get('model', {}).get('models_dir', 'models')

        # XGBoost config
        xgb_cfg = cfg.get('xgboost', {})
        self._xgb_params = {
            'n_estimators': xgb_cfg.get('n_estimators', 200),
            'max_depth': xgb_cfg.get('max_depth', 6),
            'learning_rate': xgb_cfg.get('learning_rate', 0.05),
            'subsample': xgb_cfg.get('subsample', 0.8),
            'colsample_bytree': xgb_cfg.get('colsample_bytree', 0.8),
            'min_child_weight': xgb_cfg.get('min_child_weight', 3),
            'reg_alpha': xgb_cfg.get('reg_alpha', 0.1),
            'reg_lambda': xgb_cfg.get('reg_lambda', 1.0),
        }

        # LightGBM config
        lgb_cfg = cfg.get('lightgbm', {})
        self._lgb_params = {
            'n_estimators': lgb_cfg.get('n_estimators', 200),
            'num_leaves': lgb_cfg.get('num_leaves', 31),
            'learning_rate': lgb_cfg.get('learning_rate', 0.05),
            'subsample': lgb_cfg.get('subsample', 0.8),
            'colsample_bytree': lgb_cfg.get('colsample_bytree', 0.8),
            'min_child_samples': lgb_cfg.get('min_child_samples', 20),
            'reg_alpha': lgb_cfg.get('reg_alpha', 0.1),
            'reg_lambda': lgb_cfg.get('reg_lambda', 1.0),
        }

        self._min_samples = cfg.get('min_samples', 500)
        self._enabled = cfg.get('enabled', True)

    @property
    def is_fitted(self) -> bool:
        return len(self._models) > 0

    def is_symbol_fitted(self, symbol: str) -> bool:
        return symbol in self._models

    def fit(self, df: pd.DataFrame, symbol: str) -> float:
        """
        Train XGBoost + LightGBM on OHLCV data for a specific symbol.

        Args:
            df: OHLCV DataFrame (must have open, high, low, close, volume)
            symbol: Trading pair (e.g. 'EUR/USD')

        Returns:
            Best validation accuracy (0.0-1.0)
        """
        if not self._enabled:
            logger.info(f"[{symbol}] Boosted predictor disabled in config")
            return 0.0

        if len(df) < self._min_samples:
            logger.warning(
                f"[{symbol}] Not enough data for boosting: {len(df)} < {self._min_samples}"
            )
            return 0.0

        try:
            # Generate features
            df_feat = self._generate_features(df)
            if len(df_feat) < self._min_samples // 2:
                logger.warning(f"[{symbol}] Too few rows after feature generation: {len(df_feat)}")
                return 0.0

            # Create target: 1 if next close > current close, 0 otherwise
            df_feat = df_feat.copy()
            df_feat['target'] = (df_feat['close'].shift(-1) > df_feat['close']).astype(int)
            df_feat = df_feat.iloc[:-1]  # Drop last row (no future target)

            # Get feature columns (everything except OHLCV and target)
            exclude = {'open', 'high', 'low', 'close', 'volume', 'target',
                       'datetime', 'timestamp', 'date', 'time'}
            feature_cols = [c for c in df_feat.columns if c not in exclude]

            # Build clean feature matrix with column names
            X_df = df_feat[feature_cols].copy()
            X_df = X_df.fillna(0.0).replace([np.inf, -np.inf], 0.0)
            y = df_feat['target'].values

            # Time-series split (80/20)
            split_idx = int(len(X_df) * 0.8)
            X_train, X_val = X_df.iloc[:split_idx], X_df.iloc[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Train XGBoost
            import xgboost as xgb
            xgb_model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1,
                early_stopping_rounds=20,
                **self._xgb_params
            )
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            xgb_acc = (xgb_model.predict(X_val) == y_val).mean()

            # Train LightGBM
            import lightgbm as lgb
            lgb_model = lgb.LGBMClassifier(
                objective='binary',
                random_state=42,
                n_jobs=-1,
                verbose=-1,
                **self._lgb_params
            )
            lgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(20, verbose=False)]
            )
            lgb_acc = (lgb_model.predict(X_val) == y_val).mean()

            # Store models
            self._models[symbol] = {
                'xgb': xgb_model,
                'lgb': lgb_model,
            }
            self._feature_names[symbol] = feature_cols

            # Ensemble accuracy (average probability)
            xgb_proba = xgb_model.predict_proba(X_val)[:, 1]
            lgb_proba = lgb_model.predict_proba(X_val)[:, 1]
            ensemble_proba = (xgb_proba + lgb_proba) / 2.0
            ensemble_preds = (ensemble_proba > 0.5).astype(int)
            ensemble_acc = (ensemble_preds == y_val).mean()

            logger.info(
                f"[{symbol}] Boosted training complete: "
                f"XGB={xgb_acc:.2%}, LGB={lgb_acc:.2%}, Ensemble={ensemble_acc:.2%} "
                f"({len(X_train)} train, {len(X_val)} val, {len(feature_cols)} features)"
            )

            # Save to disk
            self._save_symbol(symbol)

            return ensemble_acc

        except Exception as e:
            logger.error(f"[{symbol}] Boosted training failed: {e}", exc_info=True)
            return 0.0

    def predict(self, df: pd.DataFrame, symbol: str) -> float:
        """
        Predict probability that next candle closes higher.

        Args:
            df: Recent OHLCV data
            symbol: Trading pair

        Returns:
            Probability (0.0-1.0), or 0.5 if not fitted
        """
        if symbol not in self._models:
            # Try loading from disk
            if not self._load_symbol(symbol):
                return 0.5

        models = self._models[symbol]
        feature_cols = self._feature_names[symbol]

        try:
            df_feat = self._generate_features(df)
            if len(df_feat) == 0:
                return 0.5

            # Get last row features
            available_cols = [c for c in feature_cols if c in df_feat.columns]
            if len(available_cols) < len(feature_cols) * 0.8:
                logger.warning(
                    f"[{symbol}] Missing features: {len(available_cols)}/{len(feature_cols)}"
                )
                return 0.5

            # Build feature DataFrame with proper column names (avoids sklearn warnings)
            X_df = pd.DataFrame(
                np.nan_to_num(
                    df_feat[available_cols].iloc[[-1]].values,
                    nan=0.0, posinf=0.0, neginf=0.0
                ),
                columns=available_cols
            )

            # Pad missing columns with zeros if needed
            if len(available_cols) < len(feature_cols):
                for col in feature_cols:
                    if col not in X_df.columns:
                        X_df[col] = 0.0
                X_df = X_df[feature_cols]

            # Ensemble prediction
            xgb_prob = models['xgb'].predict_proba(X_df)[:, 1][0]
            lgb_prob = models['lgb'].predict_proba(X_df)[:, 1][0]
            ensemble_prob = (xgb_prob + lgb_prob) / 2.0

            return float(ensemble_prob)

        except Exception as e:
            logger.error(f"[{symbol}] Boosted prediction failed: {e}")
            return 0.5

    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate technical indicator features using pandas-ta."""

        df = df.copy()

        # Ensure proper column names (lowercase)
        df.columns = [c.lower() for c in df.columns]

        # Trend indicators
        df.ta.sma(length=10, append=True)
        df.ta.sma(length=20, append=True)
        df.ta.sma(length=50, append=True)
        df.ta.ema(length=10, append=True)
        df.ta.ema(length=20, append=True)
        df.ta.macd(append=True)
        df.ta.adx(append=True)

        # Momentum indicators
        df.ta.rsi(length=14, append=True)
        df.ta.rsi(length=7, append=True)
        df.ta.stoch(append=True)
        df.ta.willr(append=True)
        df.ta.cci(append=True)
        df.ta.roc(length=10, append=True)
        df.ta.mom(length=10, append=True)

        # Volatility indicators
        df.ta.bbands(length=20, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.kc(append=True)
        df.ta.natr(length=14, append=True)

        # Volume indicators (skip if volume is zero — e.g. forex)
        has_volume = df['volume'].sum() > 0
        if has_volume:
            df.ta.obv(append=True)
            df.ta.mfi(append=True)
            df.ta.cmf(append=True)

        # Price-derived features (always available)
        df['returns_1'] = df['close'].pct_change(1)
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_10'] = df['close'].pct_change(10)
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['hl_range'] = (df['high'] - df['low']) / (df['close'] + 1e-8)
        df['oc_range'] = (df['close'] - df['open']) / (df['close'] + 1e-8)
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['close'] + 1e-8)
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['close'] + 1e-8)

        # Rolling statistics
        for w in [5, 10, 20]:
            df[f'vol_{w}'] = df['close'].pct_change().rolling(w).std()
            df[f'mean_{w}'] = df['close'].rolling(w).mean() / df['close'] - 1
            df[f'skew_{w}'] = df['close'].pct_change().rolling(w).skew()

        # Drop NaN rows
        df = df.dropna()

        return df

    def _save_symbol(self, symbol: str):
        """Save models for a symbol to disk."""
        safe_name = symbol.replace('/', '_')
        path = Path(self._models_dir)
        path.mkdir(parents=True, exist_ok=True)

        models = self._models.get(symbol)
        if models is None:
            return

        joblib.dump({
            'xgb': models['xgb'],
            'lgb': models['lgb'],
            'feature_names': self._feature_names[symbol],
            'symbol': symbol,
        }, path / f"boosted_{safe_name}.joblib")

        logger.debug(f"[{symbol}] Boosted models saved")

    def _load_symbol(self, symbol: str) -> bool:
        """Load models for a symbol from disk."""
        safe_name = symbol.replace('/', '_')
        path = Path(self._models_dir) / f"boosted_{safe_name}.joblib"

        if not path.exists():
            return False

        try:
            data = joblib.load(path)
            self._models[symbol] = {
                'xgb': data['xgb'],
                'lgb': data['lgb'],
            }
            self._feature_names[symbol] = data['feature_names']
            logger.info(f"[{symbol}] Boosted models loaded from disk")
            return True
        except Exception as e:
            logger.warning(f"[{symbol}] Failed to load boosted models: {e}")
            return False

    def get_status(self) -> Dict:
        """Get predictor status."""
        return {
            'enabled': self._enabled,
            'fitted_symbols': list(self._models.keys()),
            'n_features': {
                s: len(f) for s, f in self._feature_names.items()
            },
        }
