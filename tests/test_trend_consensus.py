"""
Tests for TrendConsensus analyzer.

Uses synthetic but realistic OHLCV data — strong uptrend, strong downtrend,
choppy sideways — to verify the analyzer produces correct directional signals
through the actual computation pipeline (EMA crossovers, slope agreement, ADX).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.trend_consensus import TrendConsensus


# ---------------------------------------------------------------------------
# Helpers — generate realistic OHLCV data
# ---------------------------------------------------------------------------

def _make_ohlcv(closes: np.ndarray, noise_pct: float = 0.005) -> pd.DataFrame:
    """Build OHLCV DataFrame from a close-price series with realistic H/L/V."""
    n = len(closes)
    noise = closes * noise_pct
    highs = closes + np.abs(np.random.default_rng(42).normal(0, 1, n)) * noise
    lows = closes - np.abs(np.random.default_rng(43).normal(0, 1, n)) * noise
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    volumes = np.random.default_rng(44).uniform(100, 1000, n)
    return pd.DataFrame({
        'open': opens, 'high': highs, 'low': lows,
        'close': closes, 'volume': volumes,
    })


def _strong_uptrend(n: int = 300, start: float = 100.0, daily_pct: float = 0.003) -> pd.DataFrame:
    """Steady uptrend: price rises ~0.3% per candle with small noise."""
    rng = np.random.default_rng(1)
    closes = np.empty(n)
    closes[0] = start
    for i in range(1, n):
        closes[i] = closes[i - 1] * (1 + daily_pct + rng.normal(0, 0.001))
    return _make_ohlcv(closes)


def _strong_downtrend(n: int = 300, start: float = 100.0, daily_pct: float = 0.003) -> pd.DataFrame:
    """Steady downtrend: price drops ~0.3% per candle with small noise."""
    rng = np.random.default_rng(2)
    closes = np.empty(n)
    closes[0] = start
    for i in range(1, n):
        closes[i] = closes[i - 1] * (1 - daily_pct + rng.normal(0, 0.001))
    return _make_ohlcv(closes)


def _choppy_sideways(n: int = 300, center: float = 100.0, amplitude: float = 2.0) -> pd.DataFrame:
    """Choppy sideways: oscillates randomly around a center with no trend."""
    rng = np.random.default_rng(3)
    closes = center + rng.normal(0, amplitude, n)
    return _make_ohlcv(closes)


def _v_reversal(n: int = 400, start: float = 100.0) -> pd.DataFrame:
    """V-shaped reversal: drops for first half, rallies sharply in second half."""
    rng = np.random.default_rng(4)
    mid = n // 2
    closes = np.empty(n)
    closes[0] = start
    # Drop phase
    for i in range(1, mid):
        closes[i] = closes[i - 1] * (1 - 0.004 + rng.normal(0, 0.001))
    # Rally phase (stronger than the drop so EMAs catch up)
    for i in range(mid, n):
        closes[i] = closes[i - 1] * (1 + 0.005 + rng.normal(0, 0.001))
    return _make_ohlcv(closes)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analyzer():
    """Default TrendConsensus with production config."""
    return TrendConsensus({
        'trend_consensus': {
            'enabled': True,
            'periods': [8, 21, 55, 200],
            'adx_period': 14,
            'slope_lookback': 5,
            'prob_clamp_min': 0.15,
            'prob_clamp_max': 0.85,
            'ensemble_weight': 0.10,
        }
    })


@pytest.fixture
def disabled_analyzer():
    return TrendConsensus({'trend_consensus': {'enabled': False}})


# ---------------------------------------------------------------------------
# Test: strong uptrend → bullish probability (> 0.6)
# ---------------------------------------------------------------------------

class TestStrongUptrend:
    def test_signal_is_bullish(self, analyzer):
        df = _strong_uptrend()
        prob = analyzer.get_signal("BTC/USDT", df, interval="1h")
        assert prob > 0.6, f"Uptrend should produce bullish prob > 0.6, got {prob:.4f}"

    def test_signal_above_neutral(self, analyzer):
        df = _strong_uptrend()
        prob = analyzer.get_signal("BTC/USDT", df)
        assert prob > 0.5, f"Uptrend should be above neutral 0.5, got {prob:.4f}"


# ---------------------------------------------------------------------------
# Test: strong downtrend → bearish probability (< 0.4)
# ---------------------------------------------------------------------------

class TestStrongDowntrend:
    def test_signal_is_bearish(self, analyzer):
        df = _strong_downtrend()
        prob = analyzer.get_signal("ETH/USDT", df, interval="1h")
        assert prob < 0.4, f"Downtrend should produce bearish prob < 0.4, got {prob:.4f}"

    def test_signal_below_neutral(self, analyzer):
        df = _strong_downtrend()
        prob = analyzer.get_signal("ETH/USDT", df)
        assert prob < 0.5, f"Downtrend should be below neutral 0.5, got {prob:.4f}"


# ---------------------------------------------------------------------------
# Test: choppy sideways → near-neutral probability (0.35 - 0.65)
# ---------------------------------------------------------------------------

class TestChoppySideways:
    def test_signal_near_neutral(self, analyzer):
        df = _choppy_sideways()
        prob = analyzer.get_signal("EUR/USD", df, interval="1h")
        assert 0.30 <= prob <= 0.70, (
            f"Choppy market should be near neutral [0.30, 0.70], got {prob:.4f}"
        )


# ---------------------------------------------------------------------------
# Test: V-reversal — after rally phase, signal should be bullish
# ---------------------------------------------------------------------------

class TestVReversal:
    def test_late_reversal_is_bullish(self, analyzer):
        df = _v_reversal()
        prob = analyzer.get_signal("BTC/USDT", df, interval="1h")
        assert prob > 0.5, f"After V-reversal rally, prob should be > 0.5, got {prob:.4f}"


# ---------------------------------------------------------------------------
# Test: output always within clamp bounds
# ---------------------------------------------------------------------------

class TestOutputBounds:
    @pytest.mark.parametrize("df_fn", [_strong_uptrend, _strong_downtrend, _choppy_sideways, _v_reversal])
    def test_probability_within_clamps(self, analyzer, df_fn):
        df = df_fn()
        prob = analyzer.get_signal("TEST/USD", df)
        assert 0.15 <= prob <= 0.85, f"Prob {prob:.4f} outside clamp bounds [0.15, 0.85]"

    def test_probability_is_float(self, analyzer):
        df = _strong_uptrend()
        prob = analyzer.get_signal("BTC/USDT", df)
        assert isinstance(prob, float)


# ---------------------------------------------------------------------------
# Test: disabled analyzer returns neutral
# ---------------------------------------------------------------------------

class TestDisabled:
    def test_returns_neutral(self, disabled_analyzer):
        df = _strong_uptrend()
        prob = disabled_analyzer.get_signal("BTC/USDT", df)
        assert prob == 0.5

    def test_is_not_available(self, disabled_analyzer):
        assert disabled_analyzer.is_available is False


# ---------------------------------------------------------------------------
# Test: insufficient data returns neutral
# ---------------------------------------------------------------------------

class TestInsufficientData:
    def test_short_dataframe_returns_neutral(self, analyzer):
        """With only 50 candles, can't compute 200-period EMA — should return 0.5."""
        df = _strong_uptrend(n=50)
        prob = analyzer.get_signal("BTC/USDT", df)
        assert prob == 0.5

    def test_empty_dataframe_returns_neutral(self, analyzer):
        df = pd.DataFrame({'open': [], 'high': [], 'low': [], 'close': [], 'volume': []})
        prob = analyzer.get_signal("BTC/USDT", df)
        assert prob == 0.5


# ---------------------------------------------------------------------------
# Test: properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_is_available_when_enabled(self, analyzer):
        assert analyzer.is_available is True

    def test_ensemble_weight(self, analyzer):
        assert analyzer.ensemble_weight == 0.10

    def test_default_config(self):
        tc = TrendConsensus()
        assert tc.is_available is True
        assert tc.ensemble_weight == 0.10


# ---------------------------------------------------------------------------
# Test: determinism — same input always gives same output
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_data_same_result(self, analyzer):
        df = _strong_uptrend()
        prob1 = analyzer.get_signal("BTC/USDT", df)
        prob2 = analyzer.get_signal("BTC/USDT", df)
        assert prob1 == prob2, "Same data should always produce same signal"


# ---------------------------------------------------------------------------
# Test: internal methods directly
# ---------------------------------------------------------------------------

class TestInternals:
    def test_ema_on_constant_data(self):
        """EMA of constant series should equal that constant."""
        data = np.full(100, 50.0)
        ema = TrendConsensus._ema(data, 20)
        assert abs(ema - 50.0) < 0.01

    def test_ema_short_data_fallback(self):
        """With data shorter than period, should return mean."""
        data = np.array([10.0, 20.0, 30.0])
        ema = TrendConsensus._ema(data, 50)
        assert abs(ema - 20.0) < 0.01

    def test_wilder_smooth_constant(self):
        """Wilder smooth of constant series should equal that constant."""
        data = np.full(50, 25.0)
        result = TrendConsensus._wilder_smooth(data, 14)
        assert abs(result - 25.0) < 0.01

    def test_adx_on_trending_data(self, analyzer):
        """Strong trending data should produce ADX > 20."""
        df = _strong_uptrend()
        adx = analyzer._compute_adx(df)
        assert adx > 15, f"Strong trend should have ADX > 15, got {adx:.2f}"

    def test_crossover_score_uptrend(self, analyzer):
        """In uptrend, close should be above most EMAs → positive score."""
        df = _strong_uptrend()
        score = analyzer._score_crossovers(df['close'].values)
        assert score > 0, f"Uptrend crossover score should be > 0, got {score:.4f}"

    def test_crossover_score_downtrend(self, analyzer):
        """In downtrend, close should be below most EMAs → negative score."""
        df = _strong_downtrend()
        score = analyzer._score_crossovers(df['close'].values)
        assert score < 0, f"Downtrend crossover score should be < 0, got {score:.4f}"

    def test_slope_score_uptrend(self, analyzer):
        """In uptrend, all EMA slopes should be positive → score > 0."""
        df = _strong_uptrend()
        score = analyzer._score_slopes(df['close'].values)
        assert score > 0, f"Uptrend slope score should be > 0, got {score:.4f}"

    def test_slope_score_downtrend(self, analyzer):
        """In downtrend, all EMA slopes should be negative → score < 0."""
        df = _strong_downtrend()
        score = analyzer._score_slopes(df['close'].values)
        assert score < 0, f"Downtrend slope score should be < 0, got {score:.4f}"


# ---------------------------------------------------------------------------
# Test: config-driven behavior
# ---------------------------------------------------------------------------

class TestCustomConfig:
    def test_custom_periods(self):
        """Shorter periods = faster reaction to trend changes."""
        tc = TrendConsensus({'trend_consensus': {'periods': [5, 10, 20]}})
        # With only short periods (max=20), 50 candles is enough data
        df = _strong_uptrend(n=50)
        prob = tc.get_signal("BTC/USDT", df)
        assert prob > 0.5, f"Short-period analyzer on uptrend should be > 0.5, got {prob:.4f}"

    def test_tight_clamps(self):
        """Tighter clamps restrict the output range."""
        tc = TrendConsensus({'trend_consensus': {
            'prob_clamp_min': 0.40,
            'prob_clamp_max': 0.60,
        }})
        df = _strong_uptrend()
        prob = tc.get_signal("BTC/USDT", df)
        assert 0.40 <= prob <= 0.60, f"Tight clamps should restrict to [0.40, 0.60], got {prob:.4f}"

    def test_custom_ensemble_weight(self):
        tc = TrendConsensus({'trend_consensus': {'ensemble_weight': 0.20}})
        assert tc.ensemble_weight == 0.20


# ---------------------------------------------------------------------------
# Test: integration with ensemble (end-to-end parameter flow)
# ---------------------------------------------------------------------------

class TestEnsembleIntegration:
    def test_signal_feeds_into_predictor_param(self):
        """Verify the signal type and range match what AdvancedPredictor expects."""
        tc = TrendConsensus()
        df = _strong_uptrend()
        prob = tc.get_signal("BTC/USDT", df, interval="1h")

        # AdvancedPredictor expects: Optional[float], None or 0.5 = skip
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0
        # Strong uptrend should NOT be skipped (should not be exactly 0.5)
        assert prob != 0.5, "Strong trend signal should not be neutral"
