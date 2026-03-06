"""
Test multi-timeframe data provider functionality.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from datetime import datetime
from src.data.provider import UnifiedDataProvider, Candle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_multi_timeframe_subscription():
    """Test subscribing to multiple intervals for same symbol."""

    provider = UnifiedDataProvider.get_instance()

    # Test 1: Subscribe to BTC/USDT with multiple intervals
    provider.subscribe('BTC/USDT', exchange='binance', interval='1h')
    provider.subscribe('BTC/USDT', exchange='binance', interval='4h')
    provider.subscribe('BTC/USDT', exchange='binance', interval='1d')

    assert ('BTC/USDT', '1h') in provider._subscriptions
    assert ('BTC/USDT', '4h') in provider._subscriptions
    assert ('BTC/USDT', '1d') in provider._subscriptions

    # Test 2: Candle callback registration
    received_candles = []
    provider.on_candle_closed(lambda candle: received_candles.append(candle))
    assert len(provider._callbacks) > 0

    # Test 3: Unsubscribe all intervals for symbol
    provider.unsubscribe('BTC/USDT')
    assert ('BTC/USDT', '1h') not in provider._subscriptions
    assert ('BTC/USDT', '4h') not in provider._subscriptions
    assert ('BTC/USDT', '1d') not in provider._subscriptions

    # Test 4: Duplicate subscription is idempotent
    provider.subscribe('ETH/USDT', exchange='binance', interval='1h')
    provider.subscribe('ETH/USDT', exchange='binance', interval='1h')
    count = sum(1 for k in provider._subscriptions if k[0] == 'ETH/USDT')
    assert count == 1

    # Cleanup
    provider.unsubscribe('ETH/USDT')


def test_candle_dataclass():
    """Test Candle dataclass has interval field."""

    candle = Candle(
        timestamp=1234567890,
        datetime=datetime(2023, 1, 1),
        open=50000.0,
        high=51000.0,
        low=49000.0,
        close=50500.0,
        volume=100.0,
        symbol='BTC/USDT',
        interval='1h',
        is_closed=True,
    )
    assert candle.interval == '1h'

    # Test default value
    candle2 = Candle(
        timestamp=1234567890,
        datetime=datetime(2023, 1, 1),
        open=3000.0,
        high=3100.0,
        low=2900.0,
        close=3050.0,
        volume=50.0,
        symbol='ETH/USDT',
    )
    assert candle2.interval == ''


if __name__ == '__main__':
    try:
        test_candle_dataclass()
        test_multi_timeframe_subscription()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise
