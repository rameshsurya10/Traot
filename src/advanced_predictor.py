"""
Advanced Predictor - Mathematical Analysis Algorithms
======================================================
Ensemble of advanced mathematical algorithms for price prediction.

Algorithms:
1. Fourier Transform - Cycle detection
2. Kalman Filter - State estimation and trend smoothing
3. Shannon Entropy - Market regime detection
4. Markov Chain - Transition probabilities
5. Monte Carlo - Risk assessment via GBM simulations
6. Ensemble - Weighted combination of all algorithms

All algorithms include numerical stability safeguards (epsilon checks).
"""

import logging
import math
import threading
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from scipy.fft import fft, fftfreq
from src.analysis_engine import FeatureCalculator

logger = logging.getLogger(__name__)

# Numerical stability constant
EPSILON = 1e-10


# =============================================================================
# RESULT CLASSES
# =============================================================================

@dataclass
class PredictionResult:
    """Result from AdvancedPredictor with comprehensive metrics."""
    direction: str  # "BUY", "SELL", "NEUTRAL"
    confidence: float  # 0.0 to 1.0

    # Individual algorithm outputs
    fourier_signal: str  # "BULLISH", "BEARISH", "NEUTRAL"
    fourier_cycle_phase: float  # 0.0 to 1.0
    fourier_dominant_period: float  # Dominant cycle length

    kalman_trend: str  # "UP", "DOWN", "SIDEWAYS"
    kalman_smoothed_price: float
    kalman_velocity: float  # Price momentum
    kalman_error_covariance: float  # Estimation uncertainty

    entropy_regime: str  # "TRENDING", "CHOPPY", "VOLATILE", "NORMAL"
    entropy_value: float  # Normalized entropy 0-1
    entropy_raw_value: float  # Raw entropy value
    entropy_n_samples: int  # Number of samples analyzed

    markov_probability: float  # P(up | current_state)
    markov_state: str  # Current market state
    markov_prob_down: float  # P(down | current_state)
    markov_prob_neutral: float  # P(neutral | current_state)

    monte_carlo_risk: float  # Risk score 0-1
    monte_carlo_expected_return: float  # Expected return
    monte_carlo_prob_profit: float  # Probability of profit
    monte_carlo_prob_stop_loss: float  # Probability of hitting SL
    monte_carlo_prob_take_profit: float  # Probability of hitting TP
    monte_carlo_var_5pct: float  # Value at Risk (5th percentile)
    monte_carlo_volatility_daily: float  # Daily volatility
    monte_carlo_volatility_annual: float  # Annualized volatility
    monte_carlo_drift_annual: float  # Annualized drift

    # Price levels
    stop_loss: float
    take_profit: float

    # Risk metrics (calculated)
    risk_reward_ratio: float  # R:R ratio
    expected_profit_pct: float  # Expected profit %
    expected_loss_pct: float  # Expected loss %
    kelly_fraction: float  # Kelly criterion position size (0-1)

    # Sentiment features (optional)
    sentiment_score: Optional[float] = None  # Overall sentiment -1 to 1
    sentiment_1h: Optional[float] = None  # 1-hour sentiment
    sentiment_6h: Optional[float] = None  # 6-hour sentiment
    sentiment_momentum: Optional[float] = None  # Sentiment trend
    news_volume_1h: Optional[int] = None  # Number of articles in last hour

    # 8 out of 10 Rule Validation
    # Each rule returns (passed: bool, description: str)
    rules_passed: int = 0  # Number of rules that passed
    rules_total: int = 10  # Total rules checked
    rules_details: List[Tuple[str, bool, str]] = field(default_factory=list)  # [(name, passed, reason)]

    # Meta
    ensemble_weights: Dict[str, float] = field(default_factory=dict)

    # LSTM input probability (for backward compatibility)
    lstm_probability: float = 0.5  # Input LSTM probability passed to predict()

    # Trend consensus signal (multi-period EMA alignment)
    trend_probability: Optional[float] = None  # Trend consensus probability 0-1


# =============================================================================
# FOURIER TRANSFORM ANALYZER
# =============================================================================

class FourierAnalyzer:
    """
    Fourier Transform for cycle detection.

    Detects dominant price cycles to identify where we are
    in the current market cycle (near top/bottom).
    """

    def __init__(self, n_harmonics: int = 5, config: Optional[Dict] = None):
        """
        Args:
            n_harmonics: Number of dominant frequencies to use
            config: Full app config dict (reads prediction.fourier.*)
        """
        cfg = (config or {}).get('prediction', {}).get('fourier', {})
        self.n_harmonics = cfg.get('n_harmonics', n_harmonics)
        self.min_samples = cfg.get('min_samples', 32)
        self.phase_bullish_upper = cfg.get('phase_bullish_upper', 0.25)
        self.phase_neutral_upper = cfg.get('phase_neutral_upper', 0.50)
        self.phase_bearish_upper = cfg.get('phase_bearish_upper', 0.75)
        self.prob_bullish = cfg.get('prob_bullish', 0.75)
        self.prob_bearish = cfg.get('prob_bearish', 0.25)
        self.prob_neutral = cfg.get('prob_neutral', 0.50)
        self.default_period = cfg.get('default_period', 20.0)

    def analyze(self, prices: np.ndarray) -> Dict:
        """
        Analyze price series for dominant cycles.

        Args:
            prices: Price array (close prices)

        Returns:
            Dict with cycle analysis results
        """
        if len(prices) < self.min_samples:
            return self._default_result()

        try:
            # Detrend prices
            detrended = self._detrend(prices)

            # Apply FFT
            n = len(detrended)
            yf = fft(detrended)
            xf = fftfreq(n, 1.0)

            # Get power spectrum (positive frequencies only)
            power = np.abs(yf[:n // 2]) ** 2
            freqs = xf[:n // 2]

            # Find dominant frequencies (skip DC component)
            valid_idx = freqs > EPSILON
            power_valid = power[valid_idx]
            freqs_valid = freqs[valid_idx]

            if len(power_valid) == 0:
                return self._default_result()

            # Get top harmonics
            top_idx = np.argsort(power_valid)[-self.n_harmonics:]
            dominant_freqs = freqs_valid[top_idx]
            dominant_powers = power_valid[top_idx]

            # Calculate dominant period
            if dominant_freqs[-1] > EPSILON:
                dominant_period = 1.0 / dominant_freqs[-1]
            else:
                dominant_period = len(prices)

            # Determine cycle phase (0 = trough, 0.5 = peak)
            position_in_cycle = (len(prices) % dominant_period) / (dominant_period + EPSILON)
            cycle_phase = position_in_cycle

            # Signal based on phase
            if cycle_phase < self.phase_bullish_upper:
                signal = "BULLISH"  # Rising from trough
            elif cycle_phase < self.phase_neutral_upper:
                signal = "NEUTRAL"  # Near peak
            elif cycle_phase < self.phase_bearish_upper:
                signal = "BEARISH"  # Falling from peak
            else:
                signal = "BULLISH"  # Near trough

            return {
                'signal': signal,
                'dominant_period': float(dominant_period),
                'cycle_phase': float(cycle_phase),
                'dominant_frequencies': dominant_freqs.tolist(),
                'power_spectrum': dominant_powers.tolist()
            }

        except Exception as e:
            logger.error(f"Fourier analysis error: {e}")
            return self._default_result()

    def _detrend(self, prices: np.ndarray) -> np.ndarray:
        """Remove linear trend from prices."""
        x = np.arange(len(prices))
        slope, intercept = np.polyfit(x, prices, 1)
        trend = slope * x + intercept
        return prices - trend

    def _default_result(self) -> Dict:
        return {
            'signal': 'NEUTRAL',
            'dominant_period': self.default_period,
            'cycle_phase': 0.5,
            'dominant_frequencies': [],
            'power_spectrum': []
        }


# =============================================================================
# KALMAN FILTER
# =============================================================================

class KalmanFilter:
    """
    Kalman Filter for price smoothing and trend estimation.

    Provides noise-filtered price estimate and trend direction.
    """

    def __init__(
        self,
        process_variance: float = 1e-5,
        measurement_variance: float = 1e-2,
        config: Optional[Dict] = None
    ):
        """
        Args:
            process_variance: Q - process noise variance
            measurement_variance: R - measurement noise variance
            config: Full app config dict (reads prediction.kalman.*)
        """
        cfg = (config or {}).get('prediction', {}).get('kalman', {})
        self.Q = cfg.get('process_variance', process_variance)
        self.R = cfg.get('measurement_variance', measurement_variance)
        self.initial_error_covariance = cfg.get('initial_error_covariance', 1.0)
        self.velocity_lookback = cfg.get('velocity_lookback', 10)
        self.prob_up = cfg.get('prob_up', 0.70)
        self.prob_down = cfg.get('prob_down', 0.30)
        self.prob_sideways = cfg.get('prob_sideways', 0.50)

    def filter(self, prices: np.ndarray) -> Dict:
        """
        Apply Kalman filter to price series.

        Args:
            prices: Price array

        Returns:
            Dict with filtered prices and trend info
        """
        if len(prices) < 2:
            return self._default_result(prices[-1] if len(prices) > 0 else 0)

        try:
            # Initialize state
            x_est = prices[0]  # Initial estimate
            P = self.initial_error_covariance  # Initial error covariance

            filtered_prices = []

            for z in prices:
                # Prediction step
                x_pred = x_est
                P_pred = P + self.Q

                # Update step
                K = P_pred / (P_pred + self.R + EPSILON)  # Kalman gain
                x_est = x_pred + K * (z - x_pred)
                P = (1 - K) * P_pred

                filtered_prices.append(x_est)

            filtered_prices = np.array(filtered_prices)

            # Calculate velocity (trend)
            velocity = np.diff(filtered_prices)
            avg_velocity = np.mean(velocity[-self.velocity_lookback:]) if len(velocity) >= self.velocity_lookback else np.mean(velocity)

            # Determine trend
            if avg_velocity > EPSILON:
                trend = "UP"
            elif avg_velocity < -EPSILON:
                trend = "DOWN"
            else:
                trend = "SIDEWAYS"

            return {
                'smoothed_price': float(filtered_prices[-1]),
                'filtered_prices': filtered_prices.tolist(),
                'velocity': float(avg_velocity),
                'trend': trend,
                'error_covariance': float(P)
            }

        except Exception as e:
            logger.error(f"Kalman filter error: {e}")
            return self._default_result(prices[-1])

    def _default_result(self, price: float) -> Dict:
        return {
            'smoothed_price': float(price),
            'filtered_prices': [float(price)],
            'velocity': 0.0,
            'trend': 'SIDEWAYS',
            'error_covariance': 1.0
        }


# =============================================================================
# ENTROPY ANALYZER
# =============================================================================

class EntropyAnalyzer:
    """
    Shannon Entropy for market regime detection.

    High entropy = chaotic/choppy market
    Low entropy = trending market
    """

    def __init__(self, n_bins: int = 10, lookback: int = 50, config: Optional[Dict] = None):
        """
        Args:
            n_bins: Number of bins for histogram
            lookback: Lookback period for entropy calculation
            config: Full app config dict (reads prediction.entropy.*)
        """
        cfg = (config or {}).get('prediction', {}).get('entropy', {})
        self.n_bins = cfg.get('n_bins', n_bins)
        self.lookback = cfg.get('lookback', lookback)
        self.min_samples = cfg.get('min_samples', 10)
        self.trending_threshold = cfg.get('trending_threshold', 0.55)
        self.normal_threshold = cfg.get('normal_threshold', 0.75)
        self.choppy_threshold = cfg.get('choppy_threshold', 0.88)
        self.prob_trending_up = cfg.get('prob_trending_up', 0.65)
        self.prob_trending_down = cfg.get('prob_trending_down', 0.35)
        self.prob_normal_up = cfg.get('prob_normal_up', 0.55)
        self.prob_normal_down = cfg.get('prob_normal_down', 0.45)
        self.low_entropy_threshold = cfg.get('low_entropy_threshold', 0.5)

    def analyze(self, returns: np.ndarray) -> Dict:
        """
        Calculate Shannon entropy of returns.

        Args:
            returns: Array of price returns

        Returns:
            Dict with entropy analysis
        """
        if len(returns) < self.lookback:
            return self._default_result()

        try:
            # Use recent returns
            recent_returns = returns[-self.lookback:]

            # Remove NaN/Inf
            recent_returns = recent_returns[np.isfinite(recent_returns)]
            if len(recent_returns) < self.min_samples:
                return self._default_result()

            # Calculate histogram (probability distribution)
            hist, bin_edges = np.histogram(recent_returns, bins=self.n_bins, density=True)

            # Calculate bin widths
            bin_widths = np.diff(bin_edges)

            # Probability for each bin
            probs = hist * bin_widths
            probs = probs[probs > EPSILON]  # Remove zero probabilities

            # Shannon entropy: H = -sum(p * log2(p))
            entropy = -np.sum(probs * np.log2(probs + EPSILON))

            # Normalize entropy (0-1 scale)
            max_entropy = np.log2(self.n_bins)
            normalized_entropy = entropy / (max_entropy + EPSILON)

            # Determine regime
            if normalized_entropy < self.trending_threshold:
                regime = "TRENDING"
            elif normalized_entropy < self.normal_threshold:
                regime = "NORMAL"
            elif normalized_entropy < self.choppy_threshold:
                regime = "CHOPPY"
            else:
                regime = "VOLATILE"

            return {
                'entropy': float(entropy),
                'normalized_entropy': float(normalized_entropy),
                'regime': regime,
                'n_samples': len(recent_returns)
            }

        except Exception as e:
            logger.error(f"Entropy analysis error: {e}")
            return self._default_result()

    def _default_result(self) -> Dict:
        return {
            'entropy': 0.5,
            'normalized_entropy': 0.5,
            'regime': 'NORMAL',
            'n_samples': 0
        }


# =============================================================================
# MARKOV CHAIN
# =============================================================================

class MarkovChain:
    """
    Markov Chain for state transition probabilities.

    Models market as discrete states and calculates
    probability of transitions.
    """

    def __init__(self, n_states: int = 3, config: Optional[Dict] = None):
        """
        Args:
            n_states: Number of market states (3 = down/neutral/up)
            config: Full app config dict (reads prediction.markov.*)
        """
        cfg = (config or {}).get('prediction', {}).get('markov', {})
        self.n_states = cfg.get('n_states', n_states)
        self.min_samples = cfg.get('min_samples', 20)
        self.damping = cfg.get('damping', 0.8)
        self.percentile_lower = cfg.get('percentile_lower', 33)
        self.percentile_upper = cfg.get('percentile_upper', 67)
        self.states = ['DOWN', 'NEUTRAL', 'UP']

    def analyze(self, returns: np.ndarray) -> Dict:
        """
        Build transition matrix and calculate probabilities.

        Args:
            returns: Array of price returns

        Returns:
            Dict with Markov analysis
        """
        if len(returns) < self.min_samples:
            return self._default_result()

        try:
            # Discretize returns into states
            states = self._discretize_returns(returns)

            # Build transition matrix
            transition_matrix = self._build_transition_matrix(states)

            # Current state
            current_state_idx = states[-1]
            current_state = self.states[current_state_idx]

            # Probability of going up from current state
            raw_prob_up = transition_matrix[current_state_idx, 2]  # UP state index
            raw_prob_down = transition_matrix[current_state_idx, 0]  # DOWN state index
            # Dampen all three probabilities toward uniform (1/3) to prevent
            # extreme values dominating ensemble while preserving probability axioms
            raw_prob_neutral = transition_matrix[current_state_idx, 1]  # NEUTRAL state
            damping = self.damping
            uniform = 1.0 / 3.0
            prob_up = uniform + (raw_prob_up - uniform) * damping
            prob_down = uniform + (raw_prob_down - uniform) * damping
            prob_neutral = uniform + (raw_prob_neutral - uniform) * damping

            # Steady state (long-term probabilities)
            steady_state = self._calculate_steady_state(transition_matrix)

            # Binary ensemble probability (centered at 0.50)
            ensemble_prob_val = prob_up / (prob_up + prob_down + 1e-10)

            return {
                'current_state': current_state,
                'prob_up': float(prob_up),
                'prob_down': float(prob_down),
                'prob_neutral': float(prob_neutral),
                'ensemble_prob': float(ensemble_prob_val),
                'transition_matrix': transition_matrix.tolist(),
                'steady_state': steady_state.tolist()
            }

        except Exception as e:
            logger.error(f"Markov chain error: {e}")
            return self._default_result()

    def _discretize_returns(self, returns: np.ndarray) -> np.ndarray:
        """Convert continuous returns to discrete states."""
        states = np.zeros(len(returns), dtype=int)

        # Use percentiles for thresholds
        lower = np.percentile(returns, self.percentile_lower)
        upper = np.percentile(returns, self.percentile_upper)

        states[returns < lower] = 0  # DOWN
        states[(returns >= lower) & (returns <= upper)] = 1  # NEUTRAL
        states[returns > upper] = 2  # UP

        return states

    def _build_transition_matrix(self, states: np.ndarray) -> np.ndarray:
        """Build transition probability matrix."""
        n = self.n_states
        # Laplace smoothing: start with 1 to avoid extreme probabilities from sparse counts
        counts = np.ones((n, n))

        for i in range(len(states) - 1):
            from_state = states[i]
            to_state = states[i + 1]
            counts[from_state, to_state] += 1

        # Normalize rows
        row_sums = counts.sum(axis=1, keepdims=True)
        transition_matrix = counts / row_sums

        return transition_matrix

    def _calculate_steady_state(self, P: np.ndarray) -> np.ndarray:
        """Calculate steady state distribution."""
        try:
            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eig(P.T)

            # Find eigenvector for eigenvalue = 1
            idx = np.argmin(np.abs(eigenvalues - 1))
            steady_state = np.real(eigenvectors[:, idx])
            steady_state = steady_state / (steady_state.sum() + EPSILON)

            return steady_state
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            # Equal distribution fallback
            return np.ones(self.n_states) / self.n_states

    def _default_result(self) -> Dict:
        return {
            'current_state': 'NEUTRAL',
            'prob_up': 0.33,
            'prob_down': 0.33,
            'prob_neutral': 0.34,
            'ensemble_prob': 0.50,
            'transition_matrix': [[0.33, 0.34, 0.33]] * 3,
            'steady_state': [0.33, 0.34, 0.33]
        }


# =============================================================================
# MONTE CARLO SIMULATION
# =============================================================================

class MonteCarlo:
    """
    Monte Carlo simulation for risk assessment.

    Uses Geometric Brownian Motion (GBM) to simulate
    potential future price paths.
    """

    def __init__(
        self,
        n_simulations: int = None,
        time_horizon: int = None,
        default_volatility: float = None,
        config: Optional[Dict] = None
    ):
        """
        Args:
            n_simulations: Number of simulation paths (overrides config if provided)
            time_horizon: Number of candle steps to simulate forward
            default_volatility: Default per-candle volatility if calculation fails
            config: Full app config dict (reads prediction.monte_carlo.*)
        """
        cfg = (config or {}).get('prediction', {}).get('monte_carlo', {})
        self.n_simulations = n_simulations if n_simulations is not None else cfg.get('n_simulations', 1000)
        self.time_horizon = time_horizon if time_horizon is not None else cfg.get('time_horizon', 24)
        self.default_volatility = default_volatility if default_volatility is not None else cfg.get('default_volatility', 0.02)
        self.default_stop_loss_pct = cfg.get('default_stop_loss_pct', 0.02)
        self.default_take_profit_pct = cfg.get('default_take_profit_pct', 0.03)
        self.min_returns_samples = cfg.get('min_returns_samples', 10)
        self.var_weight = cfg.get('var_weight', 0.5)
        self.annualization_factor = cfg.get('annualization_factor', 252)

    def simulate(
        self,
        current_price: float,
        returns: np.ndarray,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.03
    ) -> Dict:
        """
        Run Monte Carlo simulation using vectorized NumPy operations.

        Uses Geometric Brownian Motion (GBM) to simulate price paths.
        Fully vectorized for performance (100K simulations in ~1 second).

        Args:
            current_price: Current asset price
            returns: Historical returns array
            stop_loss_pct: Stop loss as percentage (e.g., 0.02 = 2%)
            take_profit_pct: Take profit as percentage (e.g., 0.03 = 3%)

        Returns:
            Dict with simulation results including probabilities and risk metrics
        """
        if len(returns) < self.min_returns_samples or current_price <= 0:
            return self._default_result()

        try:
            # Calculate drift and volatility from historical data
            clean_returns = returns[np.isfinite(returns)]
            if len(clean_returns) < self.min_returns_samples:
                return self._default_result()

            drift = np.mean(clean_returns)
            volatility = np.std(clean_returns)

            # Handle edge cases
            if np.isnan(volatility) or volatility < EPSILON:
                volatility = self.default_volatility
            if np.isnan(drift):
                drift = 0.0

            # Annualize
            drift_annual = drift * self.annualization_factor
            vol_annual = volatility * np.sqrt(self.annualization_factor)

            # Per-candle parameters (returns are per-candle, not per-year)
            dt = 1.0

            # Price targets
            stop_loss_price = current_price * (1 - stop_loss_pct)
            take_profit_price = current_price * (1 + take_profit_pct)

            # Vectorized GBM simulation (100-1000x faster than loops)
            # Generate all random shocks at once: shape (n_simulations, time_horizon)
            random_shocks = np.random.normal(0, 1, (self.n_simulations, self.time_horizon))

            # Calculate log returns for each step
            log_returns = (drift - 0.5 * volatility ** 2) * dt + volatility * np.sqrt(dt) * random_shocks

            # Cumulative sum to get price paths (in log space)
            cumulative_log_returns = np.cumsum(log_returns, axis=1)

            # Convert to price paths
            price_paths = current_price * np.exp(cumulative_log_returns)

            # Check for stop loss and take profit hits along each path
            hit_sl = np.any(price_paths <= stop_loss_price, axis=1)
            hit_tp = np.any(price_paths >= take_profit_price, axis=1)

            # For paths that hit both, determine which was hit first
            # Find first index where SL or TP was hit
            sl_indices = np.argmax(price_paths <= stop_loss_price, axis=1)
            tp_indices = np.argmax(price_paths >= take_profit_price, axis=1)

            # argmax returns 0 if no True found, so we need to handle that
            sl_indices = np.where(hit_sl, sl_indices, self.time_horizon + 1)
            tp_indices = np.where(hit_tp, tp_indices, self.time_horizon + 1)

            # Count paths where SL was hit first vs TP was hit first
            sl_hit_first = hit_sl & (sl_indices < tp_indices)
            tp_hit_first = hit_tp & (tp_indices < sl_indices)

            # Get final prices (last column of price paths)
            final_prices = price_paths[:, -1]

            # Override final price for paths that hit stops
            # If SL hit first, final price is SL price; if TP hit first, final price is TP price
            final_prices = np.where(sl_hit_first, stop_loss_price, final_prices)
            final_prices = np.where(tp_hit_first, take_profit_price, final_prices)

            # Calculate statistics
            expected_return = (np.mean(final_prices) - current_price) / current_price
            prob_profit = np.mean(final_prices > current_price)
            prob_stop_loss = np.mean(sl_hit_first)
            prob_take_profit = np.mean(tp_hit_first)

            # Value at Risk (VaR) - 5th percentile
            var_5 = np.percentile(final_prices, 5)
            var_pct = (current_price - var_5) / current_price

            # Risk score (0-1, higher = riskier)
            risk_score = min(1.0, max(0.0, prob_stop_loss + var_pct * self.var_weight))

            return {
                'expected_return': float(expected_return),
                'prob_profit': float(prob_profit),
                'prob_stop_loss': float(prob_stop_loss),
                'prob_take_profit': float(prob_take_profit),
                'var_5_pct': float(var_pct),
                'risk_score': float(risk_score),
                'volatility_daily': float(volatility),
                'volatility_annual': float(vol_annual),
                'drift_annual': float(drift_annual),
                'n_simulations': self.n_simulations
            }

        except Exception as e:
            logger.error(f"Monte Carlo error: {e}")
            return self._default_result()

    def _default_result(self) -> Dict:
        return {
            'expected_return': 0.0,
            'prob_profit': 0.5,
            'prob_stop_loss': 0.1,
            'prob_take_profit': 0.1,
            'var_5_pct': 0.05,
            'risk_score': 0.5,
            'volatility_daily': self.default_volatility,
            'volatility_annual': self.default_volatility * np.sqrt(self.annualization_factor),
            'drift_annual': 0.0,
            'n_simulations': self.n_simulations
        }


# =============================================================================
# ADVANCED PREDICTOR (ENSEMBLE)
# =============================================================================

class AdvancedPredictor:
    """
    Advanced Predictor - Ensemble of mathematical algorithms.

    Combines:
    1. Fourier Transform - Cycle detection
    2. Kalman Filter - Trend smoothing
    3. Shannon Entropy - Regime detection
    4. Markov Chain - State transitions
    5. Monte Carlo - Risk assessment

    Usage:
        predictor = AdvancedPredictor()
        result = predictor.predict(df, lstm_probability, atr)
    """

    # Algorithm weights for ensemble (configurable)
    DEFAULT_WEIGHTS = {
        'lstm': 0.35,
        'fourier': 0.15,
        'kalman': 0.20,
        'markov': 0.15,
        'entropy': 0.10,
        'monte_carlo': 0.05
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None, prediction_validator=None, config: Optional[Dict] = None):
        """
        Args:
            weights: Custom algorithm weights (must sum to 1.0)
            prediction_validator: PredictionValidator instance for streak tracking
            config: Full application config dict for calibration settings
        """
        self.prediction_validator = prediction_validator
        self.config = config or {}

        # Load ensemble weights from config (or use explicit weights, or defaults)
        if weights is not None:
            self.weights = weights
        else:
            ens_weights_cfg = self.config.get('prediction', {}).get('ensemble', {}).get('weights', {})
            if ens_weights_cfg:
                self.weights = {
                    'lstm': ens_weights_cfg.get('lstm', 0.35),
                    'fourier': ens_weights_cfg.get('fourier', 0.15),
                    'kalman': ens_weights_cfg.get('kalman', 0.20),
                    'markov': ens_weights_cfg.get('markov', 0.15),
                    'entropy': ens_weights_cfg.get('entropy', 0.10),
                    'monte_carlo': ens_weights_cfg.get('monte_carlo', 0.05),
                }
            else:
                self.weights = self.DEFAULT_WEIGHTS.copy()

        # Confidence calibration from config
        pred_config = self.config.get('prediction', {})
        self.confidence_floor = pred_config.get('confidence_floor', 0.48)
        self.confidence_ceiling = pred_config.get('confidence_ceiling', 0.55)
        self.regime_penalties = pred_config.get('regime_penalties', {
            'TRENDING': 1.0,
            'NORMAL': 1.0,
            'CHOPPY': 0.85,
            'VOLATILE': 0.80,
        })

        if self.confidence_floor >= self.confidence_ceiling:
            logger.warning(
                f"confidence_floor ({self.confidence_floor}) >= ceiling ({self.confidence_ceiling}), "
                f"using defaults (0.48, 0.55)"
            )
            self.confidence_floor = 0.48
            self.confidence_ceiling = 0.55

        # Buy/sell thresholds
        self.buy_threshold = pred_config.get('buy_threshold', 0.52)
        self.sell_threshold = pred_config.get('sell_threshold', 0.48)

        # Ensemble config
        ens_cfg = pred_config.get('ensemble', {})
        self.sentiment_weight = ens_cfg.get('sentiment_weight', 0.05)
        self.sentiment_1h_weight = ens_cfg.get('sentiment_1h_weight', 0.6)
        self.sentiment_6h_weight = ens_cfg.get('sentiment_6h_weight', 0.4)
        self.sentiment_clamp_min = ens_cfg.get('sentiment_clamp_min', 0.2)
        self.sentiment_clamp_max = ens_cfg.get('sentiment_clamp_max', 0.8)
        self.mc_scaling = ens_cfg.get('mc_scaling', 20)
        self.mc_clamp_min = ens_cfg.get('mc_clamp_min', 0.2)
        self.mc_clamp_max = ens_cfg.get('mc_clamp_max', 0.8)

        # Trend consensus weight (added dynamically like sentiment)
        tc_cfg = self.config.get('trend_consensus', {})
        self.trend_weight = tc_cfg.get('ensemble_weight', 0.10)

        # Dynamic probability config
        fourier_cfg = pred_config.get('fourier', {})
        self.fourier_dynamic = fourier_cfg.get('dynamic_prob', True)
        self.fourier_amplitude = fourier_cfg.get('amplitude', 0.15)

        kalman_cfg = pred_config.get('kalman', {})
        self.kalman_dynamic = kalman_cfg.get('dynamic_prob', True)
        self.kalman_sigmoid_sensitivity = kalman_cfg.get('sigmoid_sensitivity', 2.0)

        entropy_cfg = pred_config.get('entropy', {})
        self.entropy_dynamic = entropy_cfg.get('dynamic_prob', True)
        self.entropy_certainty_scaling = entropy_cfg.get('certainty_scaling', 0.8)

        self.symmetric_confidence = ens_cfg.get('symmetric_confidence', True)

        # ATR config
        atr_cfg = pred_config.get('atr', {})
        self.atr_period = atr_cfg.get('period', 14)
        self.atr_default_pct = atr_cfg.get('default_pct', 0.02)
        self.atr_sl_multiplier = atr_cfg.get('sl_multiplier', 2)
        self.atr_tp_multiplier = atr_cfg.get('tp_multiplier', 3)
        self.atr_sl_pct_min = atr_cfg.get('sl_pct_min', 0.01)
        self.atr_sl_pct_max = atr_cfg.get('sl_pct_max', 0.05)
        self.atr_tp_pct_min = atr_cfg.get('tp_pct_min', 0.015)
        self.atr_tp_pct_max = atr_cfg.get('tp_pct_max', 0.10)

        # Kelly config
        kelly_cfg = pred_config.get('kelly', {})
        self.kelly_max_fraction = kelly_cfg.get('max_fraction', 0.25)

        # Rules config
        rules_cfg = pred_config.get('rules', {})
        self.rule_cycle_buy_max_phase = rules_cfg.get('cycle_buy_max_phase', 0.7)
        self.rule_cycle_sell_min_phase = rules_cfg.get('cycle_sell_min_phase', 0.3)
        self.rule_markov_min_prob = rules_cfg.get('markov_min_prob', 0.55)
        self.rule_mc_min_win_prob = rules_cfg.get('mc_min_win_prob', 0.50)
        self.rule_min_risk_reward = rules_cfg.get('min_risk_reward', 1.5)
        self.rule_max_daily_volatility = rules_cfg.get('max_daily_volatility', 0.05)
        self.rule_min_confidence = rules_cfg.get('min_confidence', 0.60)
        self.rule_min_kelly = rules_cfg.get('min_kelly', 0.01)

        # Online update config
        ou_cfg = pred_config.get('online_update', {})
        self.ou_clamp_min = ou_cfg.get('output_clamp_min', 0.01)
        self.ou_clamp_max = ou_cfg.get('output_clamp_max', 0.99)
        self.ou_max_loss = ou_cfg.get('max_loss_threshold', 2.0)
        self.ou_grad_clip = ou_cfg.get('grad_clip_max_norm', 1.0)

        # Normalize weights
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            self.weights = {k: v / total for k, v in self.weights.items()}

        # Initialize analyzers (pass config for configurable params)
        self.fourier = FourierAnalyzer(config=self.config)
        self.kalman = KalmanFilter(config=self.config)
        self.entropy = EntropyAnalyzer(config=self.config)
        self.markov = MarkovChain(config=self.config)
        self.monte_carlo = MonteCarlo(config=self.config)

        # Model manager reference for online learning (injected externally)
        self._model_manager = None
        self._online_lock = threading.Lock()

        # Cached optimizer/criterion for online updates (avoid re-allocation per call)
        self._online_optimizer = None
        self._online_criterion = torch.nn.BCELoss()
        self._online_lr = None
        self._online_model_symbol = None
        self._online_param_id = None  # Track parameter tensor identity for retraining invalidation

    @property
    def model_manager(self):
        """Public accessor for continuous learner compatibility."""
        return self._model_manager

    def set_model_manager(self, model_manager) -> None:
        """
        Inject model manager reference so online_update() can access LSTM models.

        Args:
            model_manager: ModelManager instance that holds per-symbol LSTM models.
        """
        self._model_manager = model_manager

    def online_update(
        self,
        df: pd.DataFrame,
        symbol: str = None,
        interval: str = None,
        learning_rate: float = 0.0001,
        actual_outcome: int = None
    ) -> None:
        """
        Perform online learning update on the LSTM model for a given symbol.

        Called by ContinuousLearningSystem after confirmed trade outcomes.
        Performs a single SGD gradient step on the neural network.

        Args:
            df: DataFrame with recent OHLCV data
            symbol: Trading pair (e.g., "BTC/USDT")
            interval: Timeframe (for logging)
            learning_rate: Learning rate for the SGD step
            actual_outcome: 1 if price went up, 0 if down
        """
        if self._model_manager is None:
            logger.debug("No model manager set, skipping online update")
            return

        if actual_outcome is None:
            logger.debug("No actual_outcome provided, skipping online update")
            return

        if actual_outcome not in (0, 1, 0.0, 1.0):
            logger.warning(f"Invalid actual_outcome: {actual_outcome}, must be 0 or 1")
            return

        if symbol is None:
            logger.debug("No symbol provided, skipping online update")
            return

        model = self._model_manager.loaded_models.get(symbol)
        if model is None:
            logger.debug(f"No LSTM model loaded for {symbol}, skipping online update")
            return

        with self._online_lock:
            try:
                feature_columns = FeatureCalculator.get_feature_columns()
                config = self._model_manager.model_configs.get(symbol, {})
                feature_means = config.get('feature_means')
                feature_stds = config.get('feature_stds')
                sequence_length = config.get('sequence_length', 60)

                # Calculate features from raw OHLCV (static method, no instantiation needed)
                df_features = FeatureCalculator.calculate_all(df)

                if len(df_features) < sequence_length:
                    return

                features = df_features[feature_columns].iloc[-sequence_length:].values

                # Normalize features
                if feature_means is not None:
                    features = (features - feature_means) / (feature_stds + 1e-8)
                else:
                    features = (features - np.nanmean(features, axis=0)) / (np.nanstd(features, axis=0) + 1e-8)

                features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

                # Validate features aren't degenerate
                if np.all(features == 0):
                    logger.debug(f"All-zero features for {symbol}, skipping online update")
                    return

                # Single SGD step on the LSTM
                model.train()
                try:
                    x = torch.FloatTensor(features).unsqueeze(0)
                    y = torch.FloatTensor([float(actual_outcome)])

                    # Reuse cached optimizer (recreate if lr, model, or params change)
                    # Track param tensor id to detect load_state_dict() after retraining
                    param_id = id(next(model.parameters()))
                    if (self._online_optimizer is None
                            or self._online_lr != learning_rate
                            or self._online_model_symbol != symbol
                            or self._online_param_id != param_id):
                        self._online_optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
                        self._online_lr = learning_rate
                        self._online_model_symbol = symbol
                        self._online_param_id = param_id
                    criterion = self._online_criterion

                    self._online_optimizer.zero_grad()
                    output = model(x).squeeze()
                    if output.dim() == 0:
                        output = output.unsqueeze(0)

                    # Clamp output to prevent log(0) explosion in BCELoss
                    output = output.clamp(self.ou_clamp_min, self.ou_clamp_max)

                    loss = criterion(output, y)

                    # Skip update if loss is abnormally high (model diverging)
                    if loss.item() > self.ou_max_loss:
                        logger.warning(
                            f"Online update [{symbol}@{interval}]: "
                            f"loss={loss.item():.4f} too high (>{self.ou_max_loss}), skipping to prevent divergence"
                        )
                        return

                    loss.backward()

                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.ou_grad_clip)
                    self._online_optimizer.step()

                    logger.info(
                        f"Online update [{symbol}@{interval}]: "
                        f"outcome={actual_outcome}, pred={output.item():.4f}, loss={loss.item():.4f}"
                    )
                finally:
                    model.eval()

            except Exception as e:
                logger.error(f"Online update failed for {symbol}: {e}")

    def predict(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str = None,
        lstm_probability: float = 0.5,
        atr: Optional[float] = None,
        sentiment_features: Optional[Dict] = None,
        interval: str = None,  # Alias for timeframe (backward compatibility)
        trend_probability: Optional[float] = None,  # Trend consensus signal
        **kwargs  # Accept extra params for forward compatibility
    ) -> PredictionResult:
        """
        Generate prediction using ensemble of algorithms.

        Args:
            df: DataFrame with OHLCV data
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Timeframe (e.g., "1h", "4h")
            lstm_probability: Probability from LSTM model (0-1)
            atr: Average True Range for stop loss calculation
            sentiment_features: Optional sentiment data from news analysis
            interval: Alias for timeframe (API compatibility)

        Returns:
            PredictionResult with comprehensive analysis
        """
        # Handle interval as alias for timeframe
        if timeframe is None and interval is not None:
            timeframe = interval
        if timeframe is None:
            timeframe = "1h"  # Default

        # Extract price data
        prices = df['close'].values
        current_price = float(prices[-1])

        # Calculate returns
        returns = np.diff(prices) / (prices[:-1] + EPSILON)

        # Calculate ATR if not provided
        if atr is None or np.isnan(atr):
            high_low = df['high'] - df['low']
            if len(high_low) >= self.atr_period:
                atr = float(high_low.rolling(self.atr_period).mean().iloc[-1])
            else:
                atr = current_price * self.atr_default_pct

        if np.isnan(atr) or atr < EPSILON:
            atr = current_price * self.atr_default_pct

        # Run all analyzers
        fourier_result = self.fourier.analyze(prices)
        kalman_result = self.kalman.filter(prices)
        entropy_result = self.entropy.analyze(returns)
        markov_result = self.markov.analyze(returns)

        # Monte Carlo with risk-adjusted stops
        stop_loss_pct = min(self.atr_sl_pct_max, max(self.atr_sl_pct_min, atr / current_price * self.atr_sl_multiplier))
        take_profit_pct = min(self.atr_tp_pct_max, max(self.atr_tp_pct_min, atr / current_price * self.atr_tp_multiplier))
        monte_carlo_result = self.monte_carlo.simulate(
            current_price, returns, stop_loss_pct, take_profit_pct
        )

        # Process sentiment features (if available)
        sentiment_score = None
        sentiment_1h = None
        sentiment_6h = None
        sentiment_momentum = None
        news_volume_1h = None
        sentiment_prob = 0.5

        if sentiment_features:
            sentiment_1h = sentiment_features.get('sentiment_1h', 0.0)
            sentiment_6h = sentiment_features.get('sentiment_6h', 0.0)
            sentiment_momentum = sentiment_features.get('sentiment_momentum', 0.0)
            news_volume_1h = sentiment_features.get('news_volume_1h', 0)

            # Calculate weighted sentiment score
            sentiment_score = (sentiment_1h * self.sentiment_1h_weight + sentiment_6h * self.sentiment_6h_weight) if sentiment_1h is not None and sentiment_6h is not None else 0.0

            # Convert sentiment to probability (sentiment range -1 to 1, prob 0 to 1)
            sentiment_prob = 0.5 + (sentiment_score * 0.5)
            sentiment_prob = max(self.sentiment_clamp_min, min(self.sentiment_clamp_max, sentiment_prob))

        # Calculate ensemble probability
        prob_scores = []

        # LSTM contribution
        prob_scores.append(('lstm', lstm_probability))

        # Fourier contribution — dynamic cosine on cycle phase
        if self.fourier_dynamic:
            cycle_phase = fourier_result.get('cycle_phase', 0.5)
            if not fourier_result.get('dominant_frequencies'):
                fourier_prob = 0.5
            else:
                fourier_prob = 0.5 + self.fourier_amplitude * math.cos(2 * math.pi * cycle_phase)
                fourier_prob = max(0.05, min(0.95, fourier_prob))
        else:
            fourier_prob = self.fourier.prob_neutral
            if fourier_result['signal'] == 'BULLISH':
                fourier_prob = self.fourier.prob_bullish
            elif fourier_result['signal'] == 'BEARISH':
                fourier_prob = self.fourier.prob_bearish
        prob_scores.append(('fourier', fourier_prob))

        # Kalman contribution — dynamic sigmoid on velocity
        if self.kalman_dynamic:
            velocity = kalman_result.get('velocity', 0.0)
            norm_velocity = velocity / (atr + 1e-10) if atr > 0 else 0.0
            exp_arg = max(-500, min(500, -self.kalman_sigmoid_sensitivity * norm_velocity))
            kalman_prob = 1.0 / (1.0 + math.exp(exp_arg))
            kalman_prob = max(0.05, min(0.95, kalman_prob))
        else:
            kalman_prob = self.kalman.prob_sideways
            if kalman_result['trend'] == 'UP':
                kalman_prob = self.kalman.prob_up
            elif kalman_result['trend'] == 'DOWN':
                kalman_prob = self.kalman.prob_down
        prob_scores.append(('kalman', kalman_prob))

        # Markov contribution
        markov_prob = markov_result.get('ensemble_prob', markov_result['prob_up'])
        prob_scores.append(('markov', markov_prob))

        # Entropy contribution — certainty-weighted scaling of Kalman
        if self.entropy_dynamic:
            normalized_entropy = entropy_result.get('normalized_entropy', 0.5)
            certainty = max(0.0, 1.0 - normalized_entropy)
            entropy_prob = 0.5 + (kalman_prob - 0.5) * certainty * self.entropy_certainty_scaling
            entropy_prob = max(0.05, min(0.95, entropy_prob))
        else:
            entropy_val = entropy_result.get('entropy', 0.5)
            entropy_prob = 0.5
            if entropy_result['regime'] == 'TRENDING':
                entropy_prob = self.entropy.prob_trending_up if kalman_result['trend'] == 'UP' else self.entropy.prob_trending_down
            elif entropy_result['regime'] in ('NORMAL', 'CHOPPY'):
                if entropy_val < self.entropy.low_entropy_threshold:
                    entropy_prob = self.entropy.prob_normal_up if kalman_result['trend'] == 'UP' else self.entropy.prob_normal_down
        prob_scores.append(('entropy', entropy_prob))

        # Monte Carlo contribution — increase scaling for meaningful signal
        mc_prob = 0.5 + (monte_carlo_result['expected_return'] * self.mc_scaling)
        mc_prob = max(self.mc_clamp_min, min(self.mc_clamp_max, mc_prob))
        prob_scores.append(('monte_carlo', mc_prob))

        # Trend consensus contribution (if available)
        has_trend = trend_probability is not None and trend_probability != 0.5
        if has_trend:
            prob_scores.append(('trend', trend_probability))

        # Sentiment contribution (if available)
        has_sentiment = bool(sentiment_features)
        if has_sentiment:
            prob_scores.append(('sentiment', sentiment_prob))

        # Build dynamic weights: start from base, add optional signals, renormalize
        active_weights = self.weights.copy()
        optional_weight_total = 0.0

        if has_trend:
            active_weights['trend'] = self.trend_weight
            optional_weight_total += self.trend_weight

        if has_sentiment:
            active_weights['sentiment'] = self.sentiment_weight
            optional_weight_total += self.sentiment_weight

        # Normalize base weights down to make room for optional signals
        if optional_weight_total > 0:
            base_total = sum(self.weights.values())
            scale = (1.0 - optional_weight_total) / base_total if base_total > 0 else 1.0
            for key in self.weights:
                active_weights[key] = self.weights[key] * scale

        weights_with_sentiment = active_weights

        # Weighted ensemble
        ensemble_prob = sum(
            weights_with_sentiment.get(name, 0) * prob
            for name, prob in prob_scores
        )
        ensemble_prob = max(0.0, min(1.0, ensemble_prob))

        # Determine direction and confidence
        conf_range = self.confidence_ceiling - self.confidence_floor
        if self.symmetric_confidence:
            # Symmetric: confidence = distance from 0.50, same for BUY and SELL
            if ensemble_prob > self.buy_threshold:
                direction = "BUY"
                confidence = (ensemble_prob - 0.5) / conf_range if conf_range > 0 else 0.0
            elif ensemble_prob < self.sell_threshold:
                direction = "SELL"
                confidence = (0.5 - ensemble_prob) / conf_range if conf_range > 0 else 0.0
            else:
                direction = "NEUTRAL"
                confidence = 0.0
        else:
            # Legacy: asymmetric mirror calculation
            if ensemble_prob > self.buy_threshold:
                direction = "BUY"
                confidence = (ensemble_prob - self.confidence_floor) / conf_range if conf_range > 0 else 0.0
            elif ensemble_prob < self.sell_threshold:
                direction = "SELL"
                sell_prob = 1.0 - ensemble_prob
                confidence = (sell_prob - self.confidence_floor) / conf_range if conf_range > 0 else 0.0
            else:
                direction = "NEUTRAL"
                confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        # Apply per-regime penalty from config
        regime = entropy_result.get('regime', 'NORMAL')
        regime_multiplier = self.regime_penalties.get(regime, 1.0)
        confidence *= regime_multiplier
        confidence = max(0.0, min(1.0, confidence))

        # Calculate stop loss and take profit
        if direction == "BUY":
            stop_loss = current_price - (self.atr_sl_multiplier * atr)
            take_profit = current_price + (self.atr_tp_multiplier * atr)
        elif direction == "SELL":
            stop_loss = current_price + (self.atr_sl_multiplier * atr)
            take_profit = current_price - (self.atr_tp_multiplier * atr)
        else:
            stop_loss = current_price - atr
            take_profit = current_price + atr

        # Calculate risk metrics
        risk_amount = abs(current_price - stop_loss)
        reward_amount = abs(take_profit - current_price)
        risk_reward_ratio = reward_amount / (risk_amount + EPSILON)

        expected_profit_pct = abs((take_profit - current_price) / current_price)
        expected_loss_pct = abs((current_price - stop_loss) / current_price)

        # Position sizing using Kelly Criterion approximation
        win_prob = monte_carlo_result['prob_profit']
        kelly_fraction = (win_prob * (1 + risk_reward_ratio) - 1) / (risk_reward_ratio + EPSILON)
        kelly_fraction = max(0.0, min(self.kelly_max_fraction, kelly_fraction))

        # =================================================================
        # 8 OUT OF 10 RULE VALIDATION
        # =================================================================
        rules_details = []

        # Rule 1: Trend Alignment (Kalman agrees with signal)
        trend_aligned = (
            (direction == "BUY" and kalman_result['trend'] == "UP") or
            (direction == "SELL" and kalman_result['trend'] == "DOWN") or
            direction == "NEUTRAL"
        )
        rules_details.append(("Trend Alignment", trend_aligned,
            f"Kalman: {kalman_result['trend']}" if trend_aligned else f"Kalman shows {kalman_result['trend']}"))

        # Rule 2: Cycle Position (Fourier - not buying at top, not selling at bottom)
        cycle_phase = fourier_result['cycle_phase']
        cycle_ok = (
            (direction == "BUY" and cycle_phase < self.rule_cycle_buy_max_phase) or
            (direction == "SELL" and cycle_phase > self.rule_cycle_sell_min_phase) or
            direction == "NEUTRAL"
        )
        rules_details.append(("Cycle Position", cycle_ok,
            f"Phase {cycle_phase:.0%}" if cycle_ok else f"Phase {cycle_phase:.0%} - bad entry"))

        # Rule 3: Market Regime (Not choppy/volatile for trades)
        regime = entropy_result['regime']
        regime_ok = regime in ['TRENDING', 'NORMAL'] or direction == "NEUTRAL"
        rules_details.append(("Market Regime", regime_ok,
            f"{regime}" if regime_ok else f"{regime} - too risky"))

        # Rule 4: Markov Probability
        markov_ok = (
            (direction == "BUY" and markov_result['prob_up'] > self.rule_markov_min_prob) or
            (direction == "SELL" and markov_result['prob_down'] > self.rule_markov_min_prob) or
            direction == "NEUTRAL"
        )
        prob_used = markov_result['prob_up'] if direction == "BUY" else markov_result['prob_down']
        rules_details.append(("Markov Probability", markov_ok,
            f"{prob_used:.0%}" if markov_ok else f"Only {prob_used:.0%}"))

        # Rule 5: Monte Carlo Win Rate
        mc_win_ok = monte_carlo_result['prob_profit'] > self.rule_mc_min_win_prob
        rules_details.append(("Win Probability", mc_win_ok,
            f"{monte_carlo_result['prob_profit']:.0%}" if mc_win_ok else f"Only {monte_carlo_result['prob_profit']:.0%}"))

        # Rule 6: Risk/Reward Ratio
        rr_ok = risk_reward_ratio >= self.rule_min_risk_reward
        rules_details.append(("Risk/Reward", rr_ok,
            f"1:{risk_reward_ratio:.1f}" if rr_ok else f"1:{risk_reward_ratio:.1f} - too low"))

        # Rule 7: Volatility Check
        vol_ok = monte_carlo_result['volatility_daily'] < self.rule_max_daily_volatility
        rules_details.append(("Volatility", vol_ok,
            f"{monte_carlo_result['volatility_daily']*100:.1f}%" if vol_ok else f"{monte_carlo_result['volatility_daily']*100:.1f}% - too high"))

        # Rule 8: Confidence Level
        conf_ok = confidence > self.rule_min_confidence
        rules_details.append(("Confidence", conf_ok,
            f"{confidence*100:.0f}%" if conf_ok else f"Only {confidence*100:.0f}%"))

        # Rule 9: Position Size Valid
        kelly_ok = kelly_fraction > self.rule_min_kelly or direction == "NEUTRAL"
        rules_details.append(("Position Size", kelly_ok,
            f"{kelly_fraction*100:.1f}%" if kelly_ok else "Too small"))

        # Rule 10: Fourier Signal Agreement
        fourier_ok = (
            (direction == "BUY" and fourier_result['signal'] in ["BULLISH", "NEUTRAL"]) or
            (direction == "SELL" and fourier_result['signal'] in ["BEARISH", "NEUTRAL"]) or
            direction == "NEUTRAL"
        )
        rules_details.append(("Fourier Signal", fourier_ok,
            fourier_result['signal'] if fourier_ok else f"{fourier_result['signal']} disagrees"))

        # Count passed rules
        rules_passed = sum(1 for _, passed, _ in rules_details if passed)

        # Record prediction for validation tracking (8/10 streak system)
        if self.prediction_validator:
            try:
                self.prediction_validator.record_prediction(
                    symbol=symbol,
                    timeframe=timeframe,
                    direction=direction,
                    current_price=current_price,
                    confidence=float(confidence),
                    target_price=current_price,  # Will be compared to next candle close
                    stop_loss=float(stop_loss),
                    take_profit=float(take_profit),
                    rules_passed=rules_passed,
                    rules_total=10,
                    market_context={
                        'regime': entropy_result['regime'],
                        'trend': kalman_result['trend'],
                        'volatility': monte_carlo_result['volatility_daily'],
                        'cycle_phase': fourier_result['cycle_phase']
                    }
                )
                logger.info(
                    f"✅ Prediction recorded: {symbol} {timeframe} {direction} "
                    f"@ ${current_price:,.2f} (conf: {confidence:.1%}, rules: {rules_passed}/10)"
                )
            except Exception as e:
                logger.error(f"❌ Failed to record prediction: {e}")

        return PredictionResult(
            # Core signal
            direction=direction,
            confidence=float(confidence),

            # Fourier analysis
            fourier_signal=fourier_result['signal'],
            fourier_cycle_phase=fourier_result['cycle_phase'],
            fourier_dominant_period=fourier_result['dominant_period'],

            # Kalman filter
            kalman_trend=kalman_result['trend'],
            kalman_smoothed_price=kalman_result['smoothed_price'],
            kalman_velocity=kalman_result['velocity'],
            kalman_error_covariance=kalman_result['error_covariance'],

            # Entropy & regime
            entropy_regime=entropy_result['regime'],
            entropy_value=entropy_result['normalized_entropy'],
            entropy_raw_value=entropy_result['entropy'],
            entropy_n_samples=entropy_result['n_samples'],

            # Markov chain
            markov_probability=markov_result['prob_up'],
            markov_state=markov_result['current_state'],
            markov_prob_down=markov_result['prob_down'],
            markov_prob_neutral=markov_result['prob_neutral'],

            # Monte Carlo simulation
            monte_carlo_risk=monte_carlo_result['risk_score'],
            monte_carlo_expected_return=monte_carlo_result['expected_return'],
            monte_carlo_prob_profit=monte_carlo_result['prob_profit'],
            monte_carlo_prob_stop_loss=monte_carlo_result['prob_stop_loss'],
            monte_carlo_prob_take_profit=monte_carlo_result['prob_take_profit'],
            monte_carlo_var_5pct=monte_carlo_result['var_5_pct'],
            monte_carlo_volatility_daily=monte_carlo_result['volatility_daily'],
            monte_carlo_volatility_annual=monte_carlo_result['volatility_annual'],
            monte_carlo_drift_annual=monte_carlo_result['drift_annual'],

            # Price levels
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),

            # Risk metrics
            risk_reward_ratio=float(risk_reward_ratio),
            expected_profit_pct=float(expected_profit_pct),
            expected_loss_pct=float(expected_loss_pct),
            kelly_fraction=float(kelly_fraction),

            # Sentiment (optional)
            sentiment_score=sentiment_score,
            sentiment_1h=sentiment_1h,
            sentiment_6h=sentiment_6h,
            sentiment_momentum=sentiment_momentum,
            news_volume_1h=news_volume_1h,

            # 8 out of 10 Rule Validation
            rules_passed=rules_passed,
            rules_total=10,
            rules_details=rules_details,

            # Meta
            ensemble_weights=weights_with_sentiment.copy(),

            # LSTM probability (input preserved for backward compatibility)
            lstm_probability=lstm_probability,

            # Trend consensus (optional)
            trend_probability=trend_probability,
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'AdvancedPredictor',
    'PredictionResult',
    'FourierAnalyzer',
    'KalmanFilter',
    'EntropyAnalyzer',
    'MarkovChain',
    'MonteCarlo',
]
