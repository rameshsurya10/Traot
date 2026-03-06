"""
Multi-Timeframe Signal Aggregator
==================================

Aggregates predictions across multiple timeframes using configurable methods.

Methods:
- weighted_vote: Σ(weight × confidence × direction_score)
- majority: Most common direction wins
- alignment_required: All timeframes must agree (conservative)

Integration:
- Combines signals from 1m, 5m, 15m, 1h, 4h, 1d models
- Respects user-configured weights
- Returns aggregated confidence score
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Signal aggregation methods."""
    WEIGHTED_VOTE = "weighted_vote"
    MAJORITY = "majority"
    ALIGNMENT_REQUIRED = "alignment_required"


@dataclass
class TimeframeSignal:
    """
    Prediction from a single timeframe.

    Attributes:
        interval: Timeframe (e.g., '1h', '5m')
        direction: 'BUY', 'SELL', or 'NEUTRAL'
        confidence: Model confidence (0.0 to 1.0)
        lstm_prob: LSTM output probability
        advanced_result: Full AdvancedPredictionResult object
        timestamp: When prediction was made
    """
    interval: str
    direction: str
    confidence: float
    lstm_prob: float
    advanced_result: any = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'interval': self.interval,
            'direction': self.direction,
            'confidence': self.confidence,
            'lstm_prob': self.lstm_prob,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class AggregatedSignal:
    """
    Final aggregated signal from all timeframes.

    Attributes:
        direction: Final trading direction
        confidence: Aggregated confidence score
        timeframe_signals: Dict of individual timeframe signals
        method: Aggregation method used
        regime: Detected market regime
        metadata: Additional information
    """
    direction: str
    confidence: float
    timeframe_signals: Dict[str, TimeframeSignal]
    method: str
    regime: str = "NORMAL"
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'direction': self.direction,
            'confidence': self.confidence,
            'timeframe_signals': {
                interval: signal.to_dict()
                for interval, signal in self.timeframe_signals.items()
            },
            'method': self.method,
            'regime': self.regime,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
        }


class SignalAggregator:
    """
    Aggregate signals across multiple timeframes.

    Thread-safe: Can be called from multiple threads
    Configurable: All parameters from config.yaml
    """

    def __init__(self, config: dict = None):
        """
        Initialize signal aggregator.

        Args:
            config: Configuration dict from config.yaml['timeframes']
        """
        self.config = config or {}

        # Get aggregation method from config
        method_str = self.config.get('aggregation_method', 'weighted_vote')
        try:
            self.method = AggregationMethod(method_str)
        except ValueError:
            logger.warning(
                f"Invalid aggregation method '{method_str}', "
                f"using weighted_vote"
            )
            self.method = AggregationMethod.WEIGHTED_VOTE

        # Get interval weights from config
        self.interval_weights = self._load_interval_weights()

        # Thresholds
        self.min_confidence = self.config.get('min_confidence', 0.50)
        self.alignment_threshold = self.config.get('alignment_threshold', 1.0)
        self.agreement_floor = self.config.get('agreement_floor', 0.8)

        # Statistics
        self._stats = {
            'aggregations_performed': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'neutral_signals': 0,
            'avg_confidence': 0.0
        }

        logger.info(
            f"SignalAggregator initialized: method={self.method.value}, "
            f"intervals={list(self.interval_weights.keys())}"
        )

    def _load_interval_weights(self) -> Dict[str, float]:
        """
        Load interval weights from config.

        Returns:
            Dict mapping interval to weight
        """
        weights = {}

        intervals = self.config.get('intervals', [])
        for interval_config in intervals:
            if interval_config.get('enabled', True):
                interval = interval_config['interval']
                weight = interval_config.get('weight', 1.0)
                weights[interval] = weight

        # Normalize weights to sum to 1.0
        if weights:
            total_weight = sum(weights.values())
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    def aggregate(
        self,
        signals: Dict[str, TimeframeSignal]
    ) -> AggregatedSignal:
        """
        Aggregate signals from multiple timeframes.

        Args:
            signals: Dict mapping interval to TimeframeSignal

        Returns:
            AggregatedSignal with final direction and confidence
        """
        if not signals:
            logger.warning("No signals provided for aggregation")
            return AggregatedSignal(
                direction='NEUTRAL',
                confidence=0.0,
                timeframe_signals={},
                method=self.method.value
            )

        # Filter to only enabled intervals
        valid_signals = {
            interval: signal
            for interval, signal in signals.items()
            if interval in self.interval_weights
        }

        if not valid_signals:
            logger.warning("No valid signals after filtering")
            return AggregatedSignal(
                direction='NEUTRAL',
                confidence=0.0,
                timeframe_signals=signals,
                method=self.method.value
            )

        # Route to appropriate aggregation method
        if self.method == AggregationMethod.WEIGHTED_VOTE:
            result = self._weighted_vote(valid_signals)
        elif self.method == AggregationMethod.MAJORITY:
            result = self._majority_vote(valid_signals)
        elif self.method == AggregationMethod.ALIGNMENT_REQUIRED:
            result = self._alignment_required(valid_signals)
        else:
            logger.error(f"Unknown aggregation method: {self.method}")
            result = self._weighted_vote(valid_signals)  # Fallback

        # Add all signals to result
        result.timeframe_signals = signals
        result.method = self.method.value

        # Detect market regime from signals
        result.regime = self._detect_regime(signals)

        # Update statistics
        self._update_stats(result)

        logger.debug(
            f"Aggregated {len(valid_signals)} signals: "
            f"{result.direction} @ {result.confidence:.2%} ({result.regime})"
        )

        return result

    def _weighted_vote(
        self,
        signals: Dict[str, TimeframeSignal]
    ) -> AggregatedSignal:
        """
        Weighted voting: Σ(weight × confidence × direction_score).

        Direction scores:
        - BUY: +1.0
        - SELL: -1.0
        - NEUTRAL: 0.0

        Args:
            signals: Valid timeframe signals

        Returns:
            AggregatedSignal
        """
        weighted_score = 0.0
        total_weight = 0.0
        confidence_sum = 0.0

        for interval, signal in signals.items():
            weight = self.interval_weights.get(interval, 0.0)

            # Convert direction to score
            if signal.direction == 'BUY':
                direction_score = 1.0
            elif signal.direction == 'SELL':
                direction_score = -1.0
            else:
                direction_score = 0.0

            # Weighted contribution
            weighted_score += weight * signal.confidence * direction_score
            total_weight += weight
            confidence_sum += weight * signal.confidence

        # Calculate aggregated confidence
        if total_weight > 0:
            avg_confidence = confidence_sum / total_weight
        else:
            avg_confidence = 0.0

        # Determine final direction and confidence
        # weighted_score sign determines direction
        # avg_confidence is the actual confidence level
        # agreement_ratio measures how aligned the timeframes are (0-1)
        if abs(weighted_score) < 0.01:  # Nearly neutral
            direction = 'NEUTRAL'
            confidence = avg_confidence
        else:
            direction = 'BUY' if weighted_score > 0 else 'SELL'
            # Agreement: ratio of net directional strength to total confidence
            # 1.0 = all timeframes agree, 0.0 = perfectly split
            agreement_ratio = abs(weighted_score) / (confidence_sum + 1e-10) if confidence_sum > 0 else 0.0
            agreement_ratio = min(agreement_ratio, 1.0)
            # Confidence = avg confidence, lightly scaled by agreement
            # Floor prevents crushing strong signals from aligned timeframes
            confidence = avg_confidence * max(agreement_ratio, self.agreement_floor)

        # Cap confidence at 1.0
        confidence = min(confidence, 1.0)

        return AggregatedSignal(
            direction=direction,
            confidence=confidence,
            timeframe_signals={},
            method=self.method.value,
            metadata={
                'weighted_score': weighted_score,
                'avg_confidence': avg_confidence,
                'total_weight': total_weight
            }
        )

    def _majority_vote(
        self,
        signals: Dict[str, TimeframeSignal]
    ) -> AggregatedSignal:
        """
        Majority voting: Most common direction wins.

        Confidence is average of agreeing signals.

        Args:
            signals: Valid timeframe signals

        Returns:
            AggregatedSignal
        """
        from collections import Counter

        # Count directions
        direction_counts = Counter()
        direction_confidences = {'BUY': [], 'SELL': [], 'NEUTRAL': []}

        for signal in signals.values():
            direction_counts[signal.direction] += 1
            direction_confidences[signal.direction].append(signal.confidence)

        # Find majority direction
        most_common = direction_counts.most_common(1)
        if not most_common:
            return AggregatedSignal(
                direction='NEUTRAL',
                confidence=0.0,
                timeframe_signals={},
                method=self.method.value
            )

        direction, count = most_common[0]

        # Calculate confidence (average of agreeing signals)
        confidences = direction_confidences[direction]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
        else:
            avg_confidence = 0.0

        # Apply agreement penalty if not unanimous
        agreement_ratio = count / len(signals)
        confidence = avg_confidence * agreement_ratio

        return AggregatedSignal(
            direction=direction,
            confidence=confidence,
            timeframe_signals={},
            method=self.method.value,
            metadata={
                'vote_count': count,
                'total_votes': len(signals),
                'agreement_ratio': agreement_ratio,
                'direction_counts': dict(direction_counts)
            }
        )

    def _alignment_required(
        self,
        signals: Dict[str, TimeframeSignal]
    ) -> AggregatedSignal:
        """
        Alignment required: ALL timeframes must agree.

        Very conservative strategy.

        Args:
            signals: Valid timeframe signals

        Returns:
            AggregatedSignal
        """
        # Get all directions
        directions = [signal.direction for signal in signals.values()]

        # Check if all agree (excluding NEUTRAL)
        non_neutral_directions = [d for d in directions if d != 'NEUTRAL']

        if not non_neutral_directions:
            # All neutral
            return AggregatedSignal(
                direction='NEUTRAL',
                confidence=0.0,
                timeframe_signals={},
                method=self.method.value,
                metadata={'reason': 'all_neutral'}
            )

        # Check alignment
        unique_directions = set(non_neutral_directions)

        if len(unique_directions) == 1:
            # Perfect alignment
            direction = unique_directions.pop()

            # Confidence is average of all signals
            confidences = [s.confidence for s in signals.values()]
            avg_confidence = sum(confidences) / len(confidences)

            return AggregatedSignal(
                direction=direction,
                confidence=avg_confidence,
                timeframe_signals={},
                method=self.method.value,
                metadata={'alignment': 'perfect'}
            )
        else:
            # No alignment - return NEUTRAL
            return AggregatedSignal(
                direction='NEUTRAL',
                confidence=0.0,
                timeframe_signals={},
                method=self.method.value,
                metadata={
                    'alignment': 'none',
                    'directions': list(unique_directions)
                }
            )

    def _detect_regime(self, signals: Dict[str, TimeframeSignal]) -> str:
        """
        Detect market regime from timeframe signals.

        Uses advanced_result from each signal if available.

        Args:
            signals: Timeframe signals

        Returns:
            Regime string: 'TRENDING', 'CHOPPY', 'VOLATILE', 'NORMAL'
        """
        regimes = []

        for signal in signals.values():
            if signal.advanced_result and hasattr(signal.advanced_result, 'regime'):
                regimes.append(signal.advanced_result.regime)

        if not regimes:
            return 'NORMAL'

        # Use most common regime
        from collections import Counter
        regime_counts = Counter(regimes)
        most_common = regime_counts.most_common(1)[0][0]

        return most_common

    def _update_stats(self, result: AggregatedSignal):
        """Update aggregator statistics."""
        self._stats['aggregations_performed'] += 1

        if result.direction == 'BUY':
            self._stats['buy_signals'] += 1
        elif result.direction == 'SELL':
            self._stats['sell_signals'] += 1
        else:
            self._stats['neutral_signals'] += 1

        # Running average of confidence
        n = self._stats['aggregations_performed']
        old_avg = self._stats['avg_confidence']
        self._stats['avg_confidence'] = (old_avg * (n - 1) + result.confidence) / n

    def get_stats(self) -> dict:
        """Get aggregator statistics."""
        total = self._stats['aggregations_performed']

        if total == 0:
            return {**self._stats, 'signal_distribution': {}}

        return {
            **self._stats,
            'signal_distribution': {
                'buy': self._stats['buy_signals'] / total,
                'sell': self._stats['sell_signals'] / total,
                'neutral': self._stats['neutral_signals'] / total
            }
        }
