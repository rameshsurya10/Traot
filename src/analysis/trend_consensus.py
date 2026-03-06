"""
Trend Consensus Analyzer
=========================
Measures multi-period trend alignment to produce a directional probability.

Instead of asking an LLM to interpret price action (expensive, slow, non-deterministic),
this analyzer measures whether short, medium, and long-term moving averages agree on
market direction. When all periods align, the signal is strong. When they disagree,
the signal is neutral (0.5).

Technique:
    - EMA crossover scoring: close vs EMA at multiple periods
    - Slope agreement: are all EMAs trending the same direction?
    - ADX filter: only trust alignment when trend strength is meaningful
    - Output: probability 0.0 (strong SELL) to 1.0 (strong BUY)

This replaces the LLM signal in the ensemble with a deterministic, zero-cost,
sub-millisecond computation that captures trend alignment — something the other
5 math algorithms (Fourier, Kalman, Entropy, Markov, Monte Carlo) don't measure.

Usage:
    analyzer = TrendConsensus(config)
    prob = analyzer.get_signal("BTC/USDT", df, interval="1h")
    # prob: 0.0 = strong bearish alignment, 0.5 = no consensus, 1.0 = strong bullish
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default EMA periods — short to long term
DEFAULT_PERIODS = [8, 21, 55, 200]


class TrendConsensus:
    """
    Multi-period trend alignment signal.

    Scores how many EMA periods agree on direction, weighted by trend strength (ADX).
    Thread-safe (stateless computation, no shared mutable state).
    """

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        tc_cfg = config.get('trend_consensus', {})

        self._enabled = tc_cfg.get('enabled', True)
        self._periods = tc_cfg.get('periods', DEFAULT_PERIODS)
        self._adx_period = tc_cfg.get('adx_period', 14)
        self._slope_lookback = tc_cfg.get('slope_lookback', 5)
        self._prob_clamp_min = tc_cfg.get('prob_clamp_min', 0.15)
        self._prob_clamp_max = tc_cfg.get('prob_clamp_max', 0.85)

        # Weight for ensemble (mirrors old LLM weight slot)
        self._ensemble_weight = tc_cfg.get('ensemble_weight', 0.10)

        logger.info(
            f"TrendConsensus initialized (periods={self._periods}, "
            f"adx={self._adx_period}, weight={self._ensemble_weight})"
        )

    @property
    def is_available(self) -> bool:
        return self._enabled

    @property
    def ensemble_weight(self) -> float:
        return self._ensemble_weight

    def get_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        interval: str = "1h"
    ) -> float:
        """
        Compute trend consensus probability.

        Args:
            symbol: Trading pair (for logging only)
            df: DataFrame with at least 'close', 'high', 'low' columns
            interval: Timeframe (for logging only)

        Returns:
            Probability 0.0 (bearish) to 1.0 (bullish), 0.5 = no consensus
        """
        if not self._enabled:
            return 0.5

        try:
            close = df['close'].values
            if len(close) < max(self._periods) + self._slope_lookback:
                return 0.5  # Not enough data

            # 1. Compute EMAs and score crossover direction
            crossover_score = self._score_crossovers(close)

            # 2. Compute slope agreement
            slope_score = self._score_slopes(close)

            # 3. Compute ADX for trend strength filter
            adx = self._compute_adx(df)
            # Scale ADX: below 20 = weak trend (dampen signal), above 25 = strong
            adx_factor = min(1.0, max(0.3, (adx - 15) / 15))

            # 4. Combine: crossover and slope each contribute, scaled by ADX
            raw_score = (0.6 * crossover_score + 0.4 * slope_score)
            # raw_score is in [-1, 1], map to probability [0, 1]
            prob = 0.5 + (raw_score * 0.5 * adx_factor)

            # Clamp
            prob = max(self._prob_clamp_min, min(self._prob_clamp_max, prob))

            logger.debug(
                f"TrendConsensus [{symbol}@{interval}]: "
                f"crossover={crossover_score:.3f}, slope={slope_score:.3f}, "
                f"adx={adx:.1f}, prob={prob:.3f}"
            )
            return prob

        except Exception as e:
            logger.debug(f"TrendConsensus failed for {symbol}: {e}")
            return 0.5

    def _score_crossovers(self, close: np.ndarray) -> float:
        """
        Score how many EMA periods the close is above (bullish) or below (bearish).

        Returns:
            Score in [-1, 1]. +1 = close above all EMAs, -1 = close below all.
        """
        current_close = close[-1]
        scores = []

        for period in self._periods:
            ema = self._ema(close, period)
            if current_close > ema:
                scores.append(1.0)
            elif current_close < ema:
                scores.append(-1.0)
            else:
                scores.append(0.0)

        return np.mean(scores)

    def _score_slopes(self, close: np.ndarray) -> float:
        """
        Score whether EMA slopes all agree on direction.

        Returns:
            Score in [-1, 1]. +1 = all slopes up, -1 = all slopes down.
        """
        slopes = []

        for period in self._periods:
            ema_now = self._ema(close, period)
            ema_prev = self._ema(close[:-self._slope_lookback], period)
            slope = ema_now - ema_prev
            if slope > 0:
                slopes.append(1.0)
            elif slope < 0:
                slopes.append(-1.0)
            else:
                slopes.append(0.0)

        return np.mean(slopes)

    def _compute_adx(self, df: pd.DataFrame) -> float:
        """Compute Average Directional Index (ADX) for trend strength."""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        n = self._adx_period

        if len(close) < n * 3:
            return 20.0  # Neutral default — need enough bars for smoothed DI + ADX

        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Wilder-smoothed series (not scalars) for ATR, +DM, -DM
        atr_series = self._wilder_smooth_series(tr, n)
        plus_dm_series = self._wilder_smooth_series(plus_dm, n)
        minus_dm_series = self._wilder_smooth_series(minus_dm, n)

        # +DI and -DI series
        plus_di = 100 * plus_dm_series / (atr_series + 1e-10)
        minus_di = 100 * minus_dm_series / (atr_series + 1e-10)

        # DX series, then smooth to get ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = self._wilder_smooth(dx, n)

        return float(adx)

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> float:
        """Compute EMA of the last value efficiently."""
        if len(data) < period:
            return float(np.mean(data))
        alpha = 2.0 / (period + 1)
        ema = data[0]
        for val in data[1:]:
            ema = alpha * val + (1 - alpha) * ema
        return float(ema)

    @staticmethod
    def _wilder_smooth_series(data: np.ndarray, period: int) -> np.ndarray:
        """Wilder's smoothing returning full series (for DI/DX computation)."""
        if len(data) < period:
            return data.copy()
        result = np.empty(len(data) - period + 1)
        result[0] = np.mean(data[:period])
        for i in range(1, len(result)):
            result[i] = (result[i - 1] * (period - 1) + data[period + i - 1]) / period
        return result

    @staticmethod
    def _wilder_smooth(data: np.ndarray, period: int) -> float:
        """Wilder's smoothing returning final scalar value."""
        if len(data) < period:
            return float(np.mean(data))
        result = np.mean(data[:period])
        for i in range(period, len(data)):
            result = (result * (period - 1) + data[i]) / period
        return float(result)
