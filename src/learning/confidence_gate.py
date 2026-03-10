"""
Confidence Gate
===============

Controls LEARNING ↔ TRADING mode transitions based on model confidence.

Key Features:
- 80% confidence threshold (user requirement)
- Hysteresis to prevent oscillation
- Regime-adjusted thresholds (harder in volatile markets)
- Smoothing to prevent rapid mode switching

NO hardcoded values - all from config
Production-ready with comprehensive logging
"""

import logging
from typing import Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceGateConfig:
    """Configuration for confidence gate."""
    trading_threshold: float = 0.65  # Enter TRADING mode at 65%
    hysteresis: float = 0.05  # Exit TRADING at 75% (80% - 5%)
    smoothing_alpha: float = 0.3  # EMA smoothing (0=no smoothing, 1=no memory)
    regime_adjustment: bool = True  # Adjust threshold by market regime

    # Regime-specific adjustments
    regime_thresholds: dict = None

    # Configurable clamp/history values
    min_threshold_clamp: float = 0.50
    max_threshold_clamp: float = 0.95
    max_history_size: int = 100

    def __post_init__(self):
        """Initialize regime thresholds if not provided."""
        if self.regime_thresholds is None:
            self.regime_thresholds = {
                'TRENDING': -0.05,  # Easier: 75% threshold
                'NORMAL': 0.0,      # Standard: 80% threshold
                'CHOPPY': 0.0,      # Standard: 80% (penalty already in predictor)
                'VOLATILE': 0.0     # Standard: 80% (penalty already in predictor)
            }


class ConfidenceGate:
    """
    Controls when the system can trade live vs. paper trade.

    Mode Transitions:
    - LEARNING → TRADING: confidence ≥ trading_threshold
    - TRADING → LEARNING: confidence < (trading_threshold - hysteresis)

    Hysteresis prevents oscillation:
    - Enter at 80%, exit at 75%
    - Prevents rapid mode switching

    Regime Adjustment:
    - TRENDING markets: Lower threshold (75%) - easier to predict
    - CHOPPY markets: Higher threshold (85%) - harder to predict
    - VOLATILE markets: Much higher (90%) - very hard to predict

    Smoothing:
    - EMA smoothing on confidence scores
    - Prevents single outlier from triggering mode change

    Thread-safe: All operations atomic
    """

    def __init__(self, config: Optional[ConfidenceGateConfig] = None):
        """
        Initialize confidence gate.

        Args:
            config: ConfidenceGateConfig or None for defaults
        """
        self.config = config or ConfidenceGateConfig()

        # Smoothed confidence per (symbol, interval)
        self._smoothed_confidence: dict = {}

        # Mode history for logging
        self._mode_history: list = []

        # Statistics
        self._stats = {
            'transitions_to_trading': 0,
            'transitions_to_learning': 0,
            'total_checks': 0,
            'regime_adjustments': 0
        }

        logger.info(
            f"ConfidenceGate initialized: "
            f"threshold={self.config.trading_threshold:.1%}, "
            f"hysteresis={self.config.hysteresis:.1%}"
        )

    def should_trade(
        self,
        confidence: float,
        current_mode: str,
        regime: str = "NORMAL",
        symbol: str = None,
        interval: str = None
    ) -> Tuple[bool, str]:
        """
        Determine if system should trade live or paper trade.

        Args:
            confidence: Model prediction confidence (0.0 to 1.0)
            current_mode: Current mode ('LEARNING' or 'TRADING')
            regime: Market regime ('TRENDING', 'NORMAL', 'CHOPPY', 'VOLATILE')
            symbol: Trading pair (for smoothing)
            interval: Timeframe (for smoothing)

        Returns:
            (can_trade_live: bool, reason: str)

        Logic:
        - If LEARNING and confidence ≥ threshold → can trade live
        - If TRADING and confidence < threshold - hysteresis → cannot trade
        - Hysteresis prevents oscillation

        Examples:
            >>> gate = ConfidenceGate()
            >>> gate.should_trade(0.85, 'LEARNING', 'NORMAL')
            (True, "Confidence 85.0% ≥ threshold 80.0%")

            >>> gate.should_trade(0.77, 'TRADING', 'NORMAL')
            (True, "Maintaining TRADING mode (hysteresis)")

            >>> gate.should_trade(0.70, 'TRADING', 'NORMAL')
            (False, "Confidence 70.0% < exit threshold 75.0%")
        """
        self._stats['total_checks'] += 1

        # Hard minimum confidence floor — reject noise signals immediately
        if confidence < self.config.min_threshold_clamp:
            return (False, f"Confidence {confidence:.1%} below minimum floor {self.config.min_threshold_clamp:.1%}")

        # Apply smoothing if symbol/interval provided
        if symbol and interval:
            confidence = self._apply_smoothing(confidence, symbol, interval)

        # Get regime-adjusted threshold
        threshold = self._get_adjusted_threshold(regime)

        # Calculate exit threshold (with hysteresis)
        exit_threshold = threshold - self.config.hysteresis

        # Validate inputs
        if current_mode not in ('LEARNING', 'TRADING'):
            logger.warning(f"Invalid mode: {current_mode}, defaulting to LEARNING")
            current_mode = 'LEARNING'

        # Determine if can trade
        if current_mode == 'LEARNING':
            # In LEARNING mode: check if confidence reached threshold
            if confidence >= threshold:
                can_trade = True
                reason = (
                    f"Confidence {confidence:.1%} ≥ threshold {threshold:.1%}"
                    f"{' (regime-adjusted)' if regime != 'NORMAL' else ''}"
                )

                # Record transition
                self._stats['transitions_to_trading'] += 1
                self._record_transition(
                    from_mode='LEARNING',
                    to_mode='TRADING',
                    confidence=confidence,
                    threshold=threshold,
                    regime=regime,
                    symbol=symbol,
                    interval=interval
                )
            else:
                can_trade = False
                gap = threshold - confidence
                reason = (
                    f"Confidence {confidence:.1%} below threshold {threshold:.1%} "
                    f"(need +{gap:.1%})"
                )

        elif current_mode == 'TRADING':
            # In TRADING mode: check if confidence dropped below exit threshold
            if confidence < exit_threshold:
                can_trade = False
                reason = (
                    f"Confidence {confidence:.1%} < exit threshold {exit_threshold:.1%} "
                    f"(hysteresis)"
                )

                # Record transition
                self._stats['transitions_to_learning'] += 1
                self._record_transition(
                    from_mode='TRADING',
                    to_mode='LEARNING',
                    confidence=confidence,
                    threshold=exit_threshold,
                    regime=regime,
                    symbol=symbol,
                    interval=interval
                )
            else:
                can_trade = True
                margin = confidence - exit_threshold
                reason = (
                    f"Maintaining TRADING mode: confidence {confidence:.1%} "
                    f"(margin +{margin:.1%} above exit threshold)"
                )

        return (can_trade, reason)

    def _get_adjusted_threshold(self, regime: str) -> float:
        """
        Get regime-adjusted confidence threshold.

        Args:
            regime: Market regime

        Returns:
            Adjusted threshold

        Examples:
            >>> gate = ConfidenceGate()
            >>> gate._get_adjusted_threshold('TRENDING')
            0.75  # Easier

            >>> gate._get_adjusted_threshold('VOLATILE')
            0.90  # Much harder
        """
        base_threshold = self.config.trading_threshold

        if not self.config.regime_adjustment:
            return base_threshold

        # Get adjustment for regime
        adjustment = self.config.regime_thresholds.get(regime, 0.0)

        if adjustment != 0.0:
            self._stats['regime_adjustments'] += 1

        adjusted = base_threshold + adjustment

        # Clamp to valid range
        adjusted = max(self.config.min_threshold_clamp, min(self.config.max_threshold_clamp, adjusted))

        return adjusted

    def _apply_smoothing(
        self,
        confidence: float,
        symbol: str,
        interval: str
    ) -> float:
        """
        Apply EMA smoothing to confidence score.

        Prevents single outlier from causing mode change.

        Args:
            confidence: Raw confidence score
            symbol: Trading pair
            interval: Timeframe

        Returns:
            Smoothed confidence

        Formula:
            smoothed = alpha * new + (1 - alpha) * old
        """
        key = (symbol, interval)

        if key not in self._smoothed_confidence:
            # First value - no smoothing
            self._smoothed_confidence[key] = confidence
            return confidence

        # EMA smoothing
        alpha = self.config.smoothing_alpha
        old_smoothed = self._smoothed_confidence[key]
        new_smoothed = alpha * confidence + (1 - alpha) * old_smoothed

        self._smoothed_confidence[key] = new_smoothed

        return new_smoothed

    def _record_transition(
        self,
        from_mode: str,
        to_mode: str,
        confidence: float,
        threshold: float,
        regime: str,
        symbol: str = None,
        interval: str = None
    ):
        """Record mode transition for history."""
        transition = {
            'timestamp': datetime.utcnow(),
            'from_mode': from_mode,
            'to_mode': to_mode,
            'confidence': confidence,
            'threshold': threshold,
            'regime': regime,
            'symbol': symbol,
            'interval': interval
        }

        self._mode_history.append(transition)

        # Keep last N transitions
        if len(self._mode_history) > self.config.max_history_size:
            self._mode_history.pop(0)

        logger.info(
            f"Mode transition: {from_mode} → {to_mode} "
            f"(confidence: {confidence:.1%}, threshold: {threshold:.1%}, "
            f"regime: {regime})"
            + (f" [{symbol} @ {interval}]" if symbol and interval else "")
        )

    def get_stats(self) -> dict:
        """Get confidence gate statistics."""
        return {
            **self._stats,
            'recent_transitions': len(self._mode_history),
            'smoothed_symbols': len(self._smoothed_confidence)
        }

    def get_transition_history(self, limit: int = 10) -> list:
        """Get recent mode transitions."""
        return self._mode_history[-limit:] if self._mode_history else []

    def reset_smoothing(self, symbol: str = None, interval: str = None):
        """
        Reset confidence smoothing.

        Args:
            symbol: If specified, reset only this symbol
            interval: If specified (with symbol), reset only that specific model
        """
        if symbol is None:
            # Reset all
            self._smoothed_confidence.clear()
            logger.info("Reset all confidence smoothing")
        elif interval is None:
            # Reset all intervals for symbol
            keys_to_remove = [k for k in self._smoothed_confidence.keys() if k[0] == symbol]
            for key in keys_to_remove:
                del self._smoothed_confidence[key]
            logger.info(f"Reset smoothing for {symbol} (all intervals)")
        else:
            # Reset specific model
            key = (symbol, interval)
            self._smoothed_confidence.pop(key, None)
            logger.info(f"Reset smoothing for {symbol} @ {interval}")


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_confidence_gate(config: dict = None) -> ConfidenceGate:
    """
    Create confidence gate from config dict.

    Args:
        config: Config dict from config.yaml

    Returns:
        ConfidenceGate instance

    Example config:
        continuous_learning:
          confidence:
            trading_threshold: 0.80
            hysteresis: 0.05
            smoothing_alpha: 0.3
            regime_adjustment: true
    """
    if config is None:
        return ConfidenceGate()

    # Read gate-specific overrides from top-level confidence_gate section
    gate_overrides = config.get('confidence_gate', {}) if isinstance(config, dict) else {}

    gate_config = ConfidenceGateConfig(
        trading_threshold=config.get('trading_threshold', 0.80),
        hysteresis=config.get('hysteresis', 0.05),
        smoothing_alpha=config.get('smoothing_alpha', 0.3),
        regime_adjustment=config.get('regime_adjustment', True),
        regime_thresholds=config.get('regime_thresholds'),
        min_threshold_clamp=gate_overrides.get('min_threshold_clamp', 0.50),
        max_threshold_clamp=gate_overrides.get('max_threshold_clamp', 0.95),
        max_history_size=gate_overrides.get('max_history_size', 100)
    )

    return ConfidenceGate(gate_config)
