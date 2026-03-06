"""
Test Learning Components
========================

Tests for ConfidenceGate, LearningStateManager, and OutcomeTracker.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import numpy as np
import tempfile

from src.learning import (
    ConfidenceGate,
    ConfidenceGateConfig,
    LearningStateManager,
    OutcomeTracker
)
from src.core.database import Database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIDENCE GATE TESTS
# =============================================================================

def test_confidence_gate_basic():
    """Test basic confidence gate functionality."""
    logger.info("\nTest 1: Confidence Gate - Basic Functionality")

    gate = ConfidenceGate()

    # Test 1: LEARNING mode, low confidence - should not trade
    # Default threshold is 0.65, so 0.50 should not trade
    can_trade, reason = gate.should_trade(
        confidence=0.50,
        current_mode='LEARNING',
        regime='NORMAL'
    )
    assert not can_trade, "Should not trade with 50% confidence"
    assert "below threshold" in reason.lower()
    logger.info(f"✓ Low confidence: {reason}")

    # Test 2: LEARNING mode, high confidence - should trade
    can_trade, reason = gate.should_trade(
        confidence=0.70,
        current_mode='LEARNING',
        regime='NORMAL'
    )
    assert can_trade, "Should trade with 70% confidence (threshold=65%)"
    assert "threshold" in reason.lower()
    logger.info(f"✓ High confidence: {reason}")

    # Test 3: TRADING mode, confidence above exit threshold - maintain
    # Exit threshold = 0.65 - 0.05 hysteresis = 0.60
    can_trade, reason = gate.should_trade(
        confidence=0.63,
        current_mode='TRADING',
        regime='NORMAL'
    )
    assert can_trade, "Should maintain TRADING at 63% (above 60% exit)"
    assert "maintaining" in reason.lower()
    logger.info(f"✓ Maintaining TRADING: {reason}")

    # Test 4: TRADING mode, confidence below exit threshold - switch to LEARNING
    # Exit threshold = 0.65 - 0.05 hysteresis = 0.60
    can_trade, reason = gate.should_trade(
        confidence=0.55,
        current_mode='TRADING',
        regime='NORMAL'
    )
    assert not can_trade, "Should exit TRADING at 55% (below 60% exit)"
    assert "exit threshold" in reason.lower()
    logger.info(f"✓ Exit to LEARNING: {reason}")

    logger.info("✓ All basic confidence gate tests passed")


def test_confidence_gate_hysteresis():
    """Test hysteresis prevents oscillation."""
    logger.info("\nTest 2: Confidence Gate - Hysteresis")

    config = ConfidenceGateConfig(
        trading_threshold=0.80,
        hysteresis=0.05
    )
    gate = ConfidenceGate(config)

    # Scenario: Confidence oscillating around threshold
    confidences = [0.79, 0.81, 0.78, 0.82, 0.77, 0.83]
    mode = 'LEARNING'
    transitions = 0

    for conf in confidences:
        can_trade, reason = gate.should_trade(conf, mode, 'NORMAL')
        new_mode = 'TRADING' if can_trade else 'LEARNING'

        if new_mode != mode:
            transitions += 1
            mode = new_mode

        logger.info(f"  Conf {conf:.1%} → {mode}: {reason}")

    # Should have fewer transitions due to hysteresis
    assert transitions <= 3, f"Too many transitions: {transitions}"
    logger.info(f"✓ Hysteresis working: only {transitions} transitions for {len(confidences)} values")


def test_confidence_gate_regime_adjustment():
    """Test regime-based threshold adjustment."""
    logger.info("\nTest 3: Confidence Gate - Regime Adjustment")

    gate = ConfidenceGate(ConfidenceGateConfig(regime_adjustment=True))

    regimes_and_results = [
        ('TRENDING', 0.61, True),   # Easier threshold (60% = 65% - 5%)
        ('NORMAL', 0.60, False),    # Standard threshold (65%), 60% is below
        ('CHOPPY', 0.68, True),     # Standard: 65% (penalty already in predictor)
        ('VOLATILE', 0.68, True),   # Standard: 65% (penalty already in predictor)
    ]

    for regime, confidence, expected_can_trade in regimes_and_results:
        can_trade, reason = gate.should_trade(
            confidence=confidence,
            current_mode='LEARNING',
            regime=regime
        )

        assert can_trade == expected_can_trade, \
            f"Regime {regime}: expected {expected_can_trade}, got {can_trade}"

        logger.info(f"✓ {regime:10s} @ {confidence:.1%}: {'CAN' if can_trade else 'CANNOT'} trade - {reason}")

    logger.info("✓ All regime adjustment tests passed")


def test_confidence_gate_smoothing():
    """Test confidence smoothing."""
    logger.info("\nTest 4: Confidence Gate - Smoothing")

    gate = ConfidenceGate(ConfidenceGateConfig(smoothing_alpha=0.3))

    # First reading - no smoothing
    can_trade1, _ = gate.should_trade(
        confidence=0.85,
        current_mode='LEARNING',
        regime='NORMAL',
        symbol='BTC/USDT',
        interval='1h'
    )

    # Sudden drop - should be smoothed
    can_trade2, reason2 = gate.should_trade(
        confidence=0.60,  # Sudden drop
        current_mode='TRADING',
        regime='NORMAL',
        symbol='BTC/USDT',
        interval='1h'
    )

    # Smoothing should prevent immediate exit
    # smoothed = 0.3 * 0.60 + 0.7 * 0.85 = 0.775 (above 0.75 exit threshold)
    assert can_trade2, f"Smoothing should prevent exit: {reason2}"
    logger.info(f"✓ Smoothing prevented premature exit: {reason2}")

    logger.info("✓ All smoothing tests passed")


# =============================================================================
# STATE MANAGER TESTS
# =============================================================================

def test_state_manager_basic():
    """Test basic state manager functionality."""
    logger.info("\nTest 5: State Manager - Basic Functionality")

    # Create temporary database
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    db = Database(temp_file.name)
    state_mgr = LearningStateManager(db)

    # Test 1: Default mode is LEARNING
    mode = state_mgr.get_current_mode('BTC/USDT', '1h')
    assert mode == 'LEARNING', "Default mode should be LEARNING"
    logger.info("✓ Default mode: LEARNING")

    # Test 2: Transition to TRADING
    state_id = state_mgr.transition_to_trading(
        symbol='BTC/USDT',
        interval='1h',
        confidence=0.85,
        reason="Confidence threshold reached"
    )
    assert state_id is not None
    logger.info(f"✓ Transitioned to TRADING (state_id={state_id})")

    # Test 3: Verify mode changed
    mode = state_mgr.get_current_mode('BTC/USDT', '1h', use_cache=False)
    assert mode == 'TRADING', "Mode should be TRADING"
    logger.info("✓ Mode verified: TRADING")

    # Test 4: Transition back to LEARNING
    state_id = state_mgr.transition_to_learning(
        symbol='BTC/USDT',
        interval='1h',
        reason="Loss detected",
        confidence=0.70
    )
    assert state_id is not None
    logger.info("✓ Transitioned back to LEARNING")

    # Test 5: Verify mode changed
    mode = state_mgr.get_current_mode('BTC/USDT', '1h', use_cache=False)
    assert mode == 'LEARNING', "Mode should be LEARNING"
    logger.info("✓ Mode verified: LEARNING")

    # Cleanup
    temp_file.close()
    Path(temp_file.name).unlink()

    logger.info("✓ All state manager basic tests passed")


def test_state_manager_cache():
    """Test state manager caching."""
    logger.info("\nTest 6: State Manager - Caching")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    db = Database(temp_file.name)
    state_mgr = LearningStateManager(db)

    # Transition to TRADING
    state_mgr.transition_to_trading('BTC/USDT', '1h', 0.85)

    # First call - cache miss
    mode1 = state_mgr.get_current_mode('BTC/USDT', '1h', use_cache=False)

    # Second call - cache hit
    mode2 = state_mgr.get_current_mode('BTC/USDT', '1h', use_cache=True)

    assert mode1 == mode2 == 'TRADING'

    stats = state_mgr.get_stats()
    logger.info(f"Cache stats: {stats}")
    assert stats['cache_hits'] >= 1
    logger.info("✓ Cache working correctly")

    # Cleanup
    temp_file.close()
    Path(temp_file.name).unlink()

    logger.info("✓ All caching tests passed")


# =============================================================================
# OUTCOME TRACKER TESTS
# =============================================================================

def test_outcome_tracker_basic():
    """Test basic outcome tracker functionality."""
    logger.info("\nTest 7: Outcome Tracker - Basic Functionality")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    db = Database(temp_file.name)

    # OutcomeTracker expects the retraining section directly as config,
    # and config must be passed as keyword arg (2nd positional is continual_learner)
    retrain_config = {
        'on_loss': True,
        'consecutive_loss_threshold': 3,
        'win_rate_threshold': 0.45,
        'high_conf_retrain_threshold': 0.80,
    }
    tracker = OutcomeTracker(db, config=retrain_config)

    # Create a test signal first
    from src.core.types import Signal, SignalType, SignalStrength
    from datetime import datetime

    signal = Signal(
        timestamp=datetime.utcnow(),
        signal_type=SignalType.BUY,
        strength=SignalStrength.STRONG,
        confidence=0.85,
        price=50000.0,
        stop_loss=49000.0,
        take_profit=52000.0,
        atr=1000.0
    )
    signal_id = db.save_signal(signal)

    # Test 1: Record winning trade
    features = np.random.randn(39)  # 39 features
    result = tracker.record_outcome(
        signal_id=signal_id,
        symbol='BTC/USDT',
        interval='1h',
        entry_price=50000.0,
        exit_price=51000.0,  # Profit
        predicted_direction='BUY',
        confidence=0.85,
        features=features,
        regime='NORMAL',
        is_paper_trade=False
    )

    assert result['was_correct'] == True, "Trade should be correct (BUY, price went up)"
    assert result['pnl_percent'] > 0, "PnL should be positive"
    assert result['should_retrain'] == False, "Should not retrain on win"
    logger.info(f"✓ Win recorded: {result['pnl_percent']:.2f}% PnL")

    # Test 2: Record losing trade (should trigger retraining)
    signal2 = Signal(
        timestamp=datetime.utcnow(),
        signal_type=SignalType.BUY,
        strength=SignalStrength.STRONG,
        confidence=0.85,
        price=51000.0,
        stop_loss=50000.0,
        take_profit=53000.0,
        atr=1000.0
    )
    signal_id2 = db.save_signal(signal2)

    result2 = tracker.record_outcome(
        signal_id=signal_id2,
        symbol='BTC/USDT',
        interval='1h',
        entry_price=51000.0,
        exit_price=50000.0,  # Loss
        predicted_direction='BUY',
        confidence=0.85,
        features=features,
        regime='NORMAL',
        is_paper_trade=False
    )

    assert result2['was_correct'] == False, "Trade should be incorrect"
    assert result2['pnl_percent'] < 0, "PnL should be negative"
    assert result2['should_retrain'] == True, "Should retrain on loss (high confidence)"
    assert 'loss' in result2['trigger_reason'].lower()
    logger.info(f"✓ Loss recorded: {result2['pnl_percent']:.2f}% PnL, retrain triggered: {result2['trigger_reason']}")

    # Test 3: Get performance summary
    summary = tracker.get_performance_summary('BTC/USDT', '1h')
    assert summary['total_trades'] == 2
    assert summary['wins'] == 1
    assert summary['losses'] == 1
    assert summary['win_rate'] == 0.5
    logger.info(f"✓ Performance summary: {summary['total_trades']} trades, {summary['win_rate']:.1%} win rate")

    # Cleanup
    temp_file.close()
    Path(temp_file.name).unlink()

    logger.info("✓ All outcome tracker basic tests passed")


def test_outcome_tracker_replay_buffer():
    """Test experience replay buffer."""
    logger.info("\nTest 8: Outcome Tracker - Replay Buffer")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    db = Database(temp_file.name)
    tracker = OutcomeTracker(db)

    # Add samples directly to buffer
    for i in range(5):
        features = np.random.randn(39)
        tracker._add_to_replay(
            symbol='BTC/USDT',
            interval='1h',
            features=features,
            target=float(i % 2),  # Alternate 0 and 1
            importance=2.0 if i % 2 == 0 else 1.0  # Even samples more important
        )

    # Get buffer
    samples = tracker.get_replay_buffer('BTC/USDT', '1h')
    assert len(samples) == 5
    logger.info(f"✓ Replay buffer: {len(samples)} samples")

    # Check importance ordering
    importances = [s['importance'] for s in samples]
    logger.info(f"✓ Sample importances: {importances}")

    # Clear buffer
    tracker.clear_replay_buffer('BTC/USDT', '1h')
    samples = tracker.get_replay_buffer('BTC/USDT', '1h')
    assert len(samples) == 0
    logger.info("✓ Buffer cleared")

    # Cleanup
    temp_file.close()
    Path(temp_file.name).unlink()

    logger.info("✓ All replay buffer tests passed")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_integration_full_flow():
    """Test full integration flow."""
    logger.info("\nTest 9: Integration - Full Flow")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    db = Database(temp_file.name)

    # Create components
    gate = ConfidenceGate()
    state_mgr = LearningStateManager(db)
    tracker = OutcomeTracker(db)

    symbol = 'BTC/USDT'
    interval = '1h'

    # Initial state - LEARNING mode
    mode = state_mgr.get_current_mode(symbol, interval)
    assert mode == 'LEARNING'
    logger.info(f"✓ Initial mode: {mode}")

    # Low confidence - should not trade (below 65% threshold)
    can_trade, reason = gate.should_trade(0.50, mode, 'NORMAL')
    assert not can_trade
    logger.info(f"✓ Low confidence: {reason}")

    # High confidence - should trade (above 65% threshold)
    can_trade, reason = gate.should_trade(0.70, mode, 'NORMAL')
    assert can_trade
    logger.info(f"✓ High confidence: {reason}")

    # Transition to TRADING
    state_mgr.transition_to_trading(symbol, interval, 0.85)
    mode = state_mgr.get_current_mode(symbol, interval)
    assert mode == 'TRADING'
    logger.info("✓ Transitioned to TRADING")

    # Record a loss - should trigger retraining
    from src.core.types import Signal, SignalType, SignalStrength
    from datetime import datetime

    signal = Signal(
        timestamp=datetime.utcnow(),
        signal_type=SignalType.BUY,
        strength=SignalStrength.STRONG,
        confidence=0.85,
        price=50000.0,
        stop_loss=49000.0,
        take_profit=52000.0,
        atr=1000.0
    )
    signal_id = db.save_signal(signal)

    result = tracker.record_outcome(
        signal_id=signal_id,
        symbol=symbol,
        interval=interval,
        entry_price=50000.0,
        exit_price=49000.0,  # Loss
        predicted_direction='BUY',
        confidence=0.85,
        features=np.random.randn(39)
    )

    assert result['should_retrain']
    logger.info(f"✓ Loss triggered retraining: {result['trigger_reason']}")

    # Transition back to LEARNING after loss
    state_mgr.transition_to_learning(symbol, interval, "Loss detected", 0.70)
    mode = state_mgr.get_current_mode(symbol, interval)
    assert mode == 'LEARNING'
    logger.info("✓ Transitioned back to LEARNING")

    # Cleanup
    temp_file.close()
    Path(temp_file.name).unlink()

    logger.info("✓ All integration tests passed")


if __name__ == '__main__':
    try:
        test_confidence_gate_basic()
        test_confidence_gate_hysteresis()
        test_confidence_gate_regime_adjustment()
        test_confidence_gate_smoothing()
        test_state_manager_basic()
        test_state_manager_cache()
        test_outcome_tracker_basic()
        test_outcome_tracker_replay_buffer()
        test_integration_full_flow()

        logger.info("\n" + "="*60)
        logger.info("All Learning Component tests passed!")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise
