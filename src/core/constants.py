"""
Shared Constants
=================
Single source of truth for values used across multiple modules.
"""

# Maps candle interval strings to their duration in minutes
INTERVAL_MINUTES = {
    '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
    '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480,
    '12h': 720, '1d': 1440,
}


def blend_probabilities(
    lstm_prob: float,
    boost_prob: float,
    has_lstm: bool,
    has_boost: bool,
    boost_weight: float = 0.6,
) -> float:
    """
    Blend LSTM and boosted model probabilities.

    Args:
        lstm_prob: LSTM model probability (0.0-1.0)
        boost_prob: Boosted model probability (0.0-1.0)
        has_lstm: Whether LSTM signal is available (not 0.5)
        has_boost: Whether boosted signal is available (not 0.5)
        boost_weight: Weight for boosted model (LSTM gets 1.0 - this)

    Returns:
        Combined probability. Falls back gracefully when one model is unavailable.
    """
    lstm_weight = 1.0 - boost_weight
    if has_lstm and has_boost:
        return lstm_weight * lstm_prob + boost_weight * boost_prob
    elif has_boost:
        return boost_prob
    elif has_lstm:
        return lstm_prob
    return 0.5
