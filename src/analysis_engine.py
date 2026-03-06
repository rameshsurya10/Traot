"""
Analysis Engine - Core ML Interface
====================================
Provides unified interface for ML predictions and feature engineering.

This module serves as the main entry point for:
- Feature calculation (50+ technical indicators)
- LSTM-based price prediction
- Signal generation

Re-exports from the advanced ML module for backward compatibility.
"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# =============================================================================
# LSTM MODEL (Backward Compatible Interface)
# =============================================================================

class LSTMModel(nn.Module):
    """
    LSTM model for price direction prediction.

    Architecture:
    - Multi-layer LSTM with dropout
    - Fully connected output layer
    - Sigmoid activation for probability output

    This is a simplified version. For production, use TCNLSTMAttention
    from src.ml.models which includes attention mechanism.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        """
        Initialize LSTM model.

        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Output layer
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, sequence, features)

        Returns:
            Probability tensor of shape (batch,) or (batch, 1)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)

        # Take last timestep output
        last_output = lstm_out[:, -1, :]

        # Fully connected layers
        output = self.fc(last_output)

        return output.squeeze(-1)


# =============================================================================
# FEATURE CALCULATOR (Backward Compatible Interface)
# =============================================================================

class FeatureCalculator:
    """
    Static feature calculator for technical indicators.

    Provides 50+ technical indicators including:
    - Trend: SMA, EMA, MACD, ADX
    - Momentum: RSI, Stochastic, ROC, Williams %R
    - Volatility: ATR, Bollinger Bands, Historical Vol
    - Volume: OBV, Volume Ratio, Money Flow
    - Pattern: Candle analysis, Higher Highs/Lower Lows

    All methods are static for easy use without instantiation.
    """

    # Standard feature columns (32 technical + 7 sentiment = 39 total)
    _FEATURE_COLUMNS = [
        # Returns
        'returns', 'log_returns',
        # Moving averages ratios
        'price_sma_7_ratio', 'price_sma_21_ratio', 'price_sma_50_ratio',
        # Volatility
        'atr_14', 'bb_width', 'bb_position', 'volatility_14',
        # Momentum
        'rsi_14', 'rsi_7', 'stoch_k', 'stoch_d', 'macd', 'macd_signal', 'macd_hist',
        'roc_10', 'williams_r', 'cci',
        # Volume
        'volume_ratio', 'obv',
        # Trend
        'adx', 'plus_di', 'minus_di', 'di_diff',
        'trend_7', 'trend_14',
        # Pattern
        'candle_body_ratio', 'higher_high', 'higher_close',
        # Note: Sentiment features removed - they're optional and require news integration
        # If needed, they should be added separately via database.get_sentiment_features()
    ]

    @staticmethod
    def get_feature_columns() -> List[str]:
        """Get list of feature column names."""
        return FeatureCalculator._FEATURE_COLUMNS.copy()

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators.

        Args:
            df: DataFrame with OHLCV columns (open, high, low, close, volume)

        Returns:
            DataFrame with all technical indicators added
        """
        df = df.copy()

        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # =================================================================
        # RETURNS
        # =================================================================
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # =================================================================
        # MOVING AVERAGES
        # =================================================================
        for period in [7, 14, 21, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # Price ratios to MAs
        df['price_sma_7_ratio'] = df['close'] / (df['sma_7'] + 1e-10)
        df['price_sma_21_ratio'] = df['close'] / (df['sma_21'] + 1e-10)
        df['price_sma_50_ratio'] = df['close'] / (df['sma_50'] + 1e-10)

        # =================================================================
        # VOLATILITY
        # =================================================================
        # True Range
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        df['atr_14'] = tr.rolling(14).mean()
        df['atr_7'] = tr.rolling(7).mean()

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-10)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

        # Historical volatility
        df['volatility_7'] = df['returns'].rolling(7).std() * np.sqrt(252)
        df['volatility_14'] = df['returns'].rolling(14).std() * np.sqrt(252)
        df['volatility_30'] = df['returns'].rolling(30).std() * np.sqrt(252)

        # =================================================================
        # MOMENTUM
        # =================================================================
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # RSI 7
        gain7 = delta.where(delta > 0, 0).rolling(7).mean()
        loss7 = (-delta.where(delta < 0, 0)).rolling(7).mean()
        rs7 = gain7 / (loss7 + 1e-10)
        df['rsi_7'] = 100 - (100 / (1 + rs7))

        # Stochastic
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # ROC
        df['roc_5'] = df['close'].pct_change(5) * 100
        df['roc_10'] = df['close'].pct_change(10) * 100
        df['roc_20'] = df['close'].pct_change(20) * 100

        # Williams %R
        df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14 + 1e-10)

        # CCI
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['cci'] = (typical_price - typical_price.rolling(20).mean()) / \
                    (0.015 * typical_price.rolling(20).std() + 1e-10)

        # =================================================================
        # VOLUME
        # =================================================================
        if 'volume' in df.columns and df['volume'].sum() > 0:
            df['volume_sma_14'] = df['volume'].rolling(14).mean()
            df['volume_ratio'] = df['volume'] / (df['volume_sma_14'] + 1e-10)

            # OBV
            volume_direction = np.sign(df['close'].diff())
            df['obv'] = (df['volume'] * volume_direction).fillna(0).cumsum()
        else:
            df['volume_ratio'] = 1.0
            df['obv'] = 0.0

        # =================================================================
        # TREND (ADX)
        # =================================================================
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / (atr_14 + 1e-10))
        minus_di = 100 * (minus_dm.rolling(14).mean() / (atr_14 + 1e-10))
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(14).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        df['di_diff'] = plus_di - minus_di

        # Trend strength
        df['trend_7'] = (df['close'] - df['close'].shift(7)) / (df['close'].shift(7) + 1e-10)
        df['trend_14'] = (df['close'] - df['close'].shift(14)) / (df['close'].shift(14) + 1e-10)

        # =================================================================
        # PATTERN FEATURES
        # =================================================================
        df['candle_body'] = abs(df['close'] - df['open'])
        df['candle_body_ratio'] = df['candle_body'] / (df['high'] - df['low'] + 1e-10)
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['higher_close'] = (df['close'] > df['close'].shift(1)).astype(int)

        return df

    @staticmethod
    def add_sentiment_features(
        df: pd.DataFrame,
        database: any = None,
        symbol: str = None,
        include_sentiment: bool = True
    ) -> pd.DataFrame:
        """
        Add sentiment features from database to technical features.

        CRITICAL: Fails gracefully if news data unavailable.
        All sentiment features default to neutral (0.0) if missing.

        Args:
            df: DataFrame with OHLCV and technical features
            database: Database instance (optional)
            symbol: Trading symbol (optional, for fetching sentiment)
            include_sentiment: Whether to include sentiment (from config)

        Returns:
            DataFrame with sentiment features added
        """
        df = df.copy()

        # Initialize all sentiment features with neutral values
        sentiment_cols = [
            'sentiment_1h', 'sentiment_6h', 'sentiment_24h',
            'sentiment_momentum', 'sentiment_volatility',
            'news_volume_1h', 'source_diversity'
        ]

        for col in sentiment_cols:
            df[col] = 0.0

        # If sentiment disabled in config, return with neutral values
        if not include_sentiment:
            logger.debug("Sentiment features disabled - using neutral values")
            return df

        # If no database provided, return with neutral values
        if database is None:
            logger.debug("No database provided - using neutral sentiment values")
            return df

        # Try to fetch sentiment features
        try:
            # Requires 'timestamp' column in df
            if 'timestamp' not in df.columns:
                logger.warning("No timestamp column - cannot fetch sentiment features")
                return df

            for idx, row in df.iterrows():
                try:
                    timestamp = int(row['timestamp'])

                    # Get sentiment features from database
                    sentiment = database.get_sentiment_features(timestamp)

                    if sentiment:
                        # Update row with sentiment data
                        for col in sentiment_cols:
                            if col in sentiment:
                                df.loc[idx, col] = sentiment[col]

                        # Normalize news_volume_1h to 0-1 scale (cap at 10 articles)
                        if 'news_volume_1h' in sentiment:
                            df.loc[idx, 'news_volume_1h'] = min(
                                sentiment['news_volume_1h'] / 10.0,
                                1.0
                            )

                except Exception as e:
                    # Individual row failure - continue with neutral
                    logger.debug(f"Failed to get sentiment for timestamp {row.get('timestamp')}: {e}")
                    continue

            logger.debug(
                f"Sentiment features added for {symbol or 'unknown'}: "
                f"{len(df)} candles processed"
            )

        except Exception as e:
            logger.warning(
                f"Sentiment feature integration failed: {e}. "
                f"Continuing with neutral sentiment values."
            )

        return df

    @staticmethod
    def calculate_all_with_sentiment(
        df: pd.DataFrame,
        database: any = None,
        symbol: str = None,
        include_sentiment: bool = True
    ) -> pd.DataFrame:
        """
        Calculate all features (technical + sentiment).

        Convenience method that combines calculate_all() and add_sentiment_features().

        Args:
            df: DataFrame with OHLCV columns
            database: Database instance (optional)
            symbol: Trading symbol (optional)
            include_sentiment: Whether to include sentiment features

        Returns:
            DataFrame with 39 features (32 technical + 7 sentiment)
        """
        # Calculate technical features
        df = FeatureCalculator.calculate_all(df)

        # Add sentiment features
        df = FeatureCalculator.add_sentiment_features(
            df=df,
            database=database,
            symbol=symbol,
            include_sentiment=include_sentiment
        )

        return df


# =============================================================================
# ANALYSIS ENGINE (Main Orchestrator)
# =============================================================================

@dataclass
class AnalysisResult:
    """Result of analysis engine prediction."""
    signal: str  # "BUY", "SELL", "NEUTRAL"
    probability: float  # 0.0 to 1.0
    confidence: float  # Adjusted confidence

    # Price levels
    current_price: float
    stop_loss: float
    take_profit: float

    # Technical summary
    rsi: float
    macd_signal: str
    trend: str
    volatility: float

    # Metadata
    features_used: int
    timestamp: str


class AnalysisEngine:
    """
    Main analysis engine for trading predictions.

    Combines feature engineering and ML prediction into a single interface.

    Usage:
        engine = AnalysisEngine()
        result = engine.analyze(df)
        print(f"Signal: {result.signal}, Confidence: {result.confidence}")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize analysis engine.

        Args:
            model_path: Path to trained model weights
            config: Configuration dictionary
        """
        self.config = config or {}
        self.model_path = model_path
        self.model: Optional[LSTMModel] = None
        self._is_fitted = False

        # Default thresholds (configurable)
        self.buy_threshold = self.config.get('buy_threshold', 0.6)
        self.sell_threshold = self.config.get('sell_threshold', 0.4)
        self.min_confidence = self.config.get('min_confidence', 0.55)

        # Try to load model if path provided
        if model_path:
            self.load_model(model_path)

    def load_model(self, path: str) -> bool:
        """Load trained model from path."""
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)

            feature_count = len(FeatureCalculator.get_feature_columns())
            self.model = LSTMModel(
                input_size=feature_count,
                hidden_size=checkpoint.get('hidden_size', 128),
                num_layers=checkpoint.get('num_layers', 2)
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self._is_fitted = True

            logger.info(f"Model loaded from {path}")
            return True

        except Exception as e:
            logger.warning(f"Could not load model: {e}")
            return False

    def analyze(self, df: pd.DataFrame) -> AnalysisResult:
        """
        Analyze price data and generate trading signal.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            AnalysisResult with signal and analysis details
        """
        from datetime import datetime

        # Calculate features
        df_features = FeatureCalculator.calculate_all(df)

        # Get current values
        latest = df_features.iloc[-1]
        current_price = float(latest['close'])
        atr = float(latest.get('atr_14', current_price * 0.02))
        rsi = float(latest.get('rsi_14', 50))

        # Determine probability
        if self._is_fitted and self.model is not None:
            probability = self._get_model_prediction(df_features)
        else:
            # Fallback to rule-based
            probability = self._get_rule_based_prediction(df_features)

        # Generate signal
        if probability > self.buy_threshold:
            signal = "BUY"
            confidence = (probability - 0.5) * 2
            stop_loss = current_price - (2 * atr)
            take_profit = current_price + (3 * atr)
        elif probability < self.sell_threshold:
            signal = "SELL"
            confidence = (0.5 - probability) * 2
            stop_loss = current_price + (2 * atr)
            take_profit = current_price - (3 * atr)
        else:
            signal = "NEUTRAL"
            confidence = 0.0
            stop_loss = current_price - atr
            take_profit = current_price + atr

        # Technical summary
        macd_val = float(latest.get('macd_hist', 0))
        trend_val = float(latest.get('trend_14', 0))

        return AnalysisResult(
            signal=signal,
            probability=probability,
            confidence=confidence,
            current_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            rsi=rsi,
            macd_signal="BULLISH" if macd_val > 0 else "BEARISH",
            trend="UP" if trend_val > 0 else "DOWN",
            volatility=float(latest.get('volatility_14', 0)),
            features_used=len(FeatureCalculator.get_feature_columns()),
            timestamp=datetime.utcnow().isoformat()
        )

    def _get_model_prediction(self, df_features: pd.DataFrame) -> float:
        """Get prediction from LSTM model."""
        try:
            feature_cols = FeatureCalculator.get_feature_columns()
            # Filter to existing columns
            available_cols = [c for c in feature_cols if c in df_features.columns]

            sequence_length = 60
            features = df_features[available_cols].iloc[-sequence_length:].values

            # Normalize
            features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
            mean = np.nanmean(features, axis=0)
            std = np.nanstd(features, axis=0) + 1e-8
            features = (features - mean) / std

            # Predict
            with torch.no_grad():
                x = torch.FloatTensor(features).unsqueeze(0)
                prob = self.model(x).item()

            return prob

        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            return 0.5

    def _get_rule_based_prediction(self, df_features: pd.DataFrame) -> float:
        """Fallback rule-based prediction."""
        latest = df_features.iloc[-1]

        score = 0.5

        # RSI
        rsi = latest.get('rsi_14', 50)
        if rsi < 30:
            score += 0.15
        elif rsi > 70:
            score -= 0.15

        # MACD
        macd_hist = latest.get('macd_hist', 0)
        if macd_hist > 0:
            score += 0.10
        else:
            score -= 0.10

        # Trend
        trend = latest.get('trend_14', 0)
        if trend > 0:
            score += 0.10
        else:
            score -= 0.10

        # Bollinger position
        bb_pos = latest.get('bb_position', 0.5)
        if bb_pos < 0.2:
            score += 0.10  # Oversold
        elif bb_pos > 0.8:
            score -= 0.10  # Overbought

        return max(0.0, min(1.0, score))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'LSTMModel',
    'FeatureCalculator',
    'AnalysisEngine',
    'AnalysisResult'
]
