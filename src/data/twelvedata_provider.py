"""
Twelve Data Forex Provider
===========================
REST-based polling provider for forex/metals data via Twelve Data API.

Free tier: 8 requests/minute, 800 requests/day.
Candle close detection: fetch 2 most recent candles; second (older) is guaranteed closed.

Usage:
    provider = TwelveDataProvider(config)
    provider.connect()
    provider.subscribe("EUR/USD", "1h")
    provider.on_candle_closed(callback)
    provider.start()
"""

import calendar
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Set, Tuple

import requests

from src.core.types import Candle

logger = logging.getLogger(__name__)

# Maps internal interval names to Twelve Data API format
INTERVAL_MAP = {
    '1m': '1min',
    '5m': '5min',
    '15m': '15min',
    '30m': '30min',
    '1h': '1h',
    '4h': '4h',
    '1d': '1day',
    '1w': '1week',
}

# Maps interval to seconds (for poll interval calculation)
_INTERVAL_SECONDS = {
    '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
    '1h': 3600, '4h': 14400, '1d': 86400, '1w': 604800,
}


@dataclass
class _Subscription:
    """Internal subscription state for a (symbol, interval) pair."""
    symbol: str
    interval: str
    last_candle_ts: Optional[int] = None
    candle_buffer: deque = field(default_factory=lambda: deque(maxlen=5000))


class TwelveDataProvider:
    """
    Forex data provider using Twelve Data REST API.

    Thread-safe. Polls at configurable intervals.
    Detects new closed candles and fires callbacks.
    """

    BASE_URL = "https://api.twelvedata.com"

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}

        self._api_key = os.environ.get('TWELVE_DATA_API_KEY', '')
        self._backfill_count = cfg.get('backfill_count', 500)
        self._buffer_size = cfg.get('buffer_size', 5000)

        # Polling config
        poll_cfg = cfg.get('polling', {})
        self._poll_divisor = poll_cfg.get('divisor', 5)
        self._poll_min = poll_cfg.get('min_seconds', 8)
        self._poll_max = poll_cfg.get('max_seconds', 720)

        # Rate limit config
        rl_cfg = cfg.get('rate_limit', {})
        self._max_per_minute = rl_cfg.get('per_minute', 8)
        self._max_per_day = rl_cfg.get('per_day', 800)
        self._warn_threshold = rl_cfg.get('warn_threshold', 700)

        # State
        self._connected = False
        self._running = False
        self._session = requests.Session()
        self._subscriptions: Dict[Tuple[str, str], _Subscription] = {}
        self._callbacks: List[Callable] = []
        self._database = None
        self._lock = threading.Lock()
        self._rate_lock = threading.Lock()
        self._poll_thread: Optional[threading.Thread] = None

        # Rate limit counters (protected by _rate_lock)
        self._calls_this_minute = 0
        self._calls_today = 0
        self._minute_start = time.time()
        self._day_start = time.time()

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        """
        Test API connectivity. Idempotent — returns True if already connected.

        Returns:
            True if connected, False if API key missing or validation fails.
        """
        if self._connected:
            return True

        if not self._api_key:
            logger.warning("Twelve Data: No API key set (TWELVE_DATA_API_KEY)")
            return False

        try:
            resp = self._session.get(
                f"{self.BASE_URL}/time_series",
                params={
                    'symbol': 'EUR/USD',
                    'interval': '1h',
                    'outputsize': 1,
                    'apikey': self._api_key,
                },
                timeout=15,
            )
            data = resp.json()

            if data.get('status') == 'error' or 'values' not in data:
                logger.error(f"Twelve Data connect failed: {data.get('message', 'unknown error')}")
                return False

            self._connected = True
            logger.info("Twelve Data connected successfully")
            return True

        except Exception as e:
            logger.error(f"Twelve Data connect error: {e}")
            return False

    def disconnect(self):
        """Disconnect and stop polling."""
        self.stop()
        self._connected = False
        logger.info("Twelve Data disconnected")

    def subscribe(self, symbol: str, interval: str = "1h"):
        """
        Subscribe to a (symbol, interval) pair. Triggers backfill if connected.

        Args:
            symbol: e.g. "EUR/USD"
            interval: e.g. "1h", "15m"
        """
        if interval not in INTERVAL_MAP:
            logger.warning(f"Invalid interval '{interval}' — must be one of {list(INTERVAL_MAP.keys())}")
            return

        key = (symbol, interval)
        with self._lock:
            if key in self._subscriptions:
                return  # Already subscribed

            sub = _Subscription(
                symbol=symbol,
                interval=interval,
                candle_buffer=deque(maxlen=self._buffer_size),
            )
            self._subscriptions[key] = sub

        logger.info(f"Twelve Data: subscribed to {symbol} @ {interval}")

        # Backfill if connected
        if self._connected:
            self._backfill(sub)

    def unsubscribe(self, symbol: str, interval: Optional[str] = None):
        """
        Unsubscribe from a symbol. If interval is None, removes all intervals.

        Args:
            symbol: Symbol to unsubscribe
            interval: Specific interval, or None for all
        """
        with self._lock:
            if interval:
                key = (symbol, interval)
                self._subscriptions.pop(key, None)
            else:
                keys_to_remove = [k for k in self._subscriptions if k[0] == symbol]
                for k in keys_to_remove:
                    del self._subscriptions[k]

    def get_subscriptions(self) -> Set[Tuple[str, str]]:
        """Return set of active (symbol, interval) subscriptions."""
        with self._lock:
            return set(self._subscriptions.keys())

    def on_candle_closed(self, callback: Callable):
        """Register a callback for closed candles. callback(candle: Candle)."""
        self._callbacks.append(callback)

    def set_database(self, database):
        """Set database reference for saving candles."""
        self._database = database

    def start(self):
        """Start the background polling thread."""
        if self._running:
            return
        self._running = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name="TwelveData-Poller",
            daemon=True,
        )
        self._poll_thread.start()
        logger.info("Twelve Data polling started")

    def stop(self):
        """Stop the background polling thread."""
        self._running = False
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=10)
        self._poll_thread = None
        logger.info("Twelve Data polling stopped")

    def get_status(self) -> dict:
        """Return provider status for dashboard/diagnostics."""
        with self._lock:
            return {
                'connected': self._connected,
                'running': self._running,
                'subscriptions': len(self._subscriptions),
                'calls_today': self._calls_today,
                'calls_remaining': max(0, self._max_per_day - self._calls_today),
                'calls_this_minute': self._calls_this_minute,
            }

    # ── Internal methods ──────────────────────────────────────────────

    def _poll_loop(self):
        """Main polling loop — runs in background thread."""
        while self._running:
            with self._lock:
                subs = list(self._subscriptions.values())

            for sub in subs:
                if not self._running:
                    break
                try:
                    self._check_for_new_candle(sub)
                except Exception as e:
                    logger.error(f"Poll error for {sub.symbol}@{sub.interval}: {e}")

            # Calculate poll interval based on smallest subscribed interval
            poll_seconds = self._calc_poll_interval()
            # Sleep in small increments so stop() is responsive
            for _ in range(int(poll_seconds)):
                if not self._running:
                    break
                time.sleep(1)

    def _calc_poll_interval(self) -> float:
        """Calculate poll interval as candle_interval / divisor, clamped."""
        min_interval_sec = float('inf')
        with self._lock:
            for sub in self._subscriptions.values():
                sec = _INTERVAL_SECONDS.get(sub.interval, 3600)
                if sec < min_interval_sec:
                    min_interval_sec = sec

        if min_interval_sec == float('inf'):
            return self._poll_max

        poll = min_interval_sec / self._poll_divisor
        return max(self._poll_min, min(poll, self._poll_max))

    def _check_for_new_candle(self, sub: _Subscription):
        """
        Fetch latest 2 candles from API. The second (older) candle is guaranteed
        closed. Compare its timestamp to last_candle_ts to detect new candles.
        """
        self._wait_for_rate_limit()

        td_interval = INTERVAL_MAP.get(sub.interval, sub.interval)
        resp = self._session.get(
            f"{self.BASE_URL}/time_series",
            params={
                'symbol': sub.symbol,
                'interval': td_interval,
                'outputsize': 2,
                'apikey': self._api_key,
            },
            timeout=15,
        )
        self._increment_counters()
        data = resp.json()

        if 'values' not in data or len(data['values']) < 2:
            return

        # API returns newest first — index 1 is the closed candle
        closed_raw = data['values'][1]
        closed_ts = self._parse_timestamp(closed_raw['datetime'])

        if sub.last_candle_ts is not None and closed_ts <= sub.last_candle_ts:
            return  # Already seen this candle

        sub.last_candle_ts = closed_ts
        candle = self._build_candle(closed_raw, sub.symbol, sub.interval)
        sub.candle_buffer.append(candle)

        # Save to database
        if self._database:
            try:
                self._database.save_candles(candle)
            except Exception as e:
                logger.debug(f"DB save error for {sub.symbol}: {e}")

        # Fire callbacks (snapshot to avoid race with on_candle_closed)
        for cb in list(self._callbacks):
            try:
                cb(candle)
            except Exception as e:
                logger.error(f"Candle callback error: {e}")

    def _backfill(self, sub: _Subscription):
        """Fetch historical candles for a new subscription."""
        try:
            self._wait_for_rate_limit()

            td_interval = INTERVAL_MAP.get(sub.interval, sub.interval)
            resp = self._session.get(
                f"{self.BASE_URL}/time_series",
                params={
                    'symbol': sub.symbol,
                    'interval': td_interval,
                    'outputsize': self._backfill_count,
                    'apikey': self._api_key,
                },
                timeout=30,
            )
            self._increment_counters()
            data = resp.json()

            if 'values' not in data:
                logger.warning(f"Backfill failed for {sub.symbol}@{sub.interval}: {data.get('message', 'no values')}")
                return

            # API returns newest first — reverse for chronological order
            raw_candles = list(reversed(data['values']))

            candles = [self._build_candle(raw, sub.symbol, sub.interval) for raw in raw_candles]
            for candle in candles:
                sub.candle_buffer.append(candle)

            if self._database and candles:
                try:
                    import pandas as pd
                    df = pd.DataFrame([{
                        'timestamp': c.timestamp,
                        'datetime': c.datetime,
                        'open': c.open,
                        'high': c.high,
                        'low': c.low,
                        'close': c.close,
                        'volume': c.volume,
                    } for c in candles])
                    self._database.save_candles(df, symbol=sub.symbol, interval=sub.interval)
                except Exception as e:
                    logger.warning(f"Backfill DB save error: {e}")

            # Set last_candle_ts to most recent candle
            if sub.candle_buffer:
                sub.last_candle_ts = sub.candle_buffer[-1].timestamp

            logger.info(
                f"Twelve Data backfill: {sub.symbol}@{sub.interval} — "
                f"{len(raw_candles)} candles loaded"
            )

        except Exception as e:
            logger.error(f"Backfill error for {sub.symbol}@{sub.interval}: {e}")

    def _build_candle(self, raw: dict, symbol: str, interval: str) -> Candle:
        """Build a Candle from Twelve Data JSON row."""
        ts = self._parse_timestamp(raw['datetime'])
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)

        return Candle(
            timestamp=ts,
            datetime=dt,
            open=float(raw['open']),
            high=float(raw['high']),
            low=float(raw['low']),
            close=float(raw['close']),
            volume=float(raw.get('volume', 0)),
            symbol=symbol,
            interval=interval,
            is_closed=True,
        )

    @staticmethod
    def _parse_timestamp(dt_str: str) -> int:
        """
        Parse Twelve Data datetime string to millisecond epoch (UTC).

        Uses calendar.timegm() to avoid local timezone issues.
        """
        dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
        epoch = calendar.timegm(dt.timetuple())
        return int(epoch * 1000)

    def _wait_for_rate_limit(self):
        """Block until rate limits allow another call. Thread-safe."""
        with self._rate_lock:
            now = time.time()

            # Reset minute counter
            if now - self._minute_start >= 60:
                self._calls_this_minute = 0
                self._minute_start = now

            # Reset daily counter
            if now - self._day_start >= 86400:
                self._calls_today = 0
                self._day_start = now

            if self._calls_today >= self._max_per_day:
                raise RuntimeError(f"Twelve Data daily API limit reached ({self._max_per_day})")

            if self._calls_today >= self._warn_threshold:
                logger.warning(f"Twelve Data: {self._calls_today}/{self._max_per_day} daily calls used")

            # Wait if minute limit hit (release lock during sleep)
            need_wait = self._calls_this_minute >= self._max_per_minute
            wait = 60 - (now - self._minute_start) if need_wait else 0

        if need_wait and wait > 0:
            logger.debug(f"Rate limit: waiting {wait:.1f}s")
            for _ in range(int(wait)):
                if not self._running:
                    return
                time.sleep(1)
            with self._rate_lock:
                self._calls_this_minute = 0
                self._minute_start = time.time()

    def _increment_counters(self):
        """Increment rate limit counters after an API call. Thread-safe."""
        with self._rate_lock:
            self._calls_this_minute += 1
            self._calls_today += 1
