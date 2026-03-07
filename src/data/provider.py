"""
Unified Data Provider — CCXT-based crypto data streaming
=========================================================
Singleton provider for real-time candle data via CCXT Pro WebSocket,
with REST polling fallback.  Matches the TwelveDataProvider interface.

Usage:
    provider = UnifiedDataProvider.get_instance("config.yaml")
    provider.subscribe("BTC/USDT", exchange="binance", interval="1h")
    provider.on_candle_closed(my_callback)
    provider.start()
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Tuple

import ccxt
import pandas as pd
import yaml

from src.core.types import Candle

logger = logging.getLogger(__name__)

# Interval string → seconds mapping
INTERVAL_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900,
    "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400,
    "6h": 21600, "8h": 28800, "12h": 43200, "1d": 86400,
}


@dataclass
class _Subscription:
    """Track a single symbol+interval subscription."""
    symbol: str
    exchange: str
    interval: str
    candle_buffer: deque = field(default_factory=lambda: deque(maxlen=5000))
    last_candle_ts: int = 0


class UnifiedDataProvider:
    """
    Singleton crypto data provider using CCXT REST polling.

    Interface mirrors TwelveDataProvider so the runner treats all
    market-type providers identically.
    """

    _instance: Optional["UnifiedDataProvider"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def get_instance(cls, config_path: str = "config.yaml") -> "UnifiedDataProvider":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(config_path)
            return cls._instance

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        data_cfg = self.config.get("data", {})
        ws_cfg = data_cfg.get("websocket", {})

        self._subscriptions: Dict[Tuple[str, str], _Subscription] = {}
        self._callbacks: List[Callable] = []
        self._database = None
        self._exchanges: Dict[str, ccxt.Exchange] = {}

        self._running = False
        self._connected = False
        self._poll_thread: Optional[threading.Thread] = None

        # Buffer size from config
        self._buffer_size = ws_cfg.get("buffer_size", 5000)

        # Rate limiting
        self._min_poll_interval = 2.0  # seconds between API calls

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        if self._connected:
            return True
        self._connected = True
        logger.info("UnifiedDataProvider connected")
        return True

    def disconnect(self):
        self._connected = False
        self.stop()

    def subscribe(self, symbol: str, exchange: str = "binance", interval: str = "1h"):
        """Subscribe to a symbol+interval. Idempotent."""
        key = (symbol, interval)
        if key in self._subscriptions:
            return

        sub = _Subscription(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            candle_buffer=deque(maxlen=self._buffer_size),
        )
        self._subscriptions[key] = sub
        self._ensure_exchange(exchange)
        logger.info(f"Subscribed to {symbol} @ {interval} on {exchange}")

    def unsubscribe(self, symbol: str):
        """Remove all subscriptions for a symbol."""
        keys_to_remove = [k for k in self._subscriptions if k[0] == symbol]
        for k in keys_to_remove:
            del self._subscriptions[k]
        logger.info(f"Unsubscribed from {symbol}")

    def on_candle_closed(self, callback: Callable):
        """Register a callback for candle close events: callback(candle: Candle)."""
        self._callbacks.append(callback)

    def set_database(self, database):
        """Enable auto-persistence of candles to database."""
        self._database = database

    def start(self):
        """Start polling for new candles in a background thread."""
        if self._running:
            return
        self._running = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="CryptoDataPoll"
        )
        self._poll_thread.start()
        logger.info("UnifiedDataProvider started (REST polling)")

    def stop(self):
        """Stop the polling thread."""
        self._running = False
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=10)
        self._poll_thread = None
        logger.info("UnifiedDataProvider stopped")

    def get_status(self) -> dict:
        return {
            "connected": self._connected,
            "running": self._running,
            "subscriptions": len(self._subscriptions),
            "symbols": [k[0] for k in self._subscriptions],
        }

    # ------------------------------------------------------------------
    # Historical backfill (called externally from runner)
    # ------------------------------------------------------------------

    def fetch_historical(
        self,
        symbol: str,
        exchange: str = "binance",
        interval: str = "1h",
        days: int = 365,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV candles from exchange via CCXT REST.

        Returns DataFrame with columns:
            timestamp (int ms), datetime, open, high, low, close, volume
        """
        ex = self._ensure_exchange(exchange)
        timeframe = interval
        limit_per_req = 1000

        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
        since = start_ms

        all_candles = []
        logger.info(f"Fetching {days} days of {symbol} @ {interval} from {exchange}...")

        while since < end_ms:
            try:
                ohlcv = ex.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=limit_per_req,
                )
                if not ohlcv:
                    break

                all_candles.extend(ohlcv)
                last_ts = ohlcv[-1][0]
                if last_ts <= since:
                    break
                since = last_ts + 1

                if len(all_candles) % 5000 == 0:
                    logger.info(f"  {symbol}: fetched {len(all_candles)} candles so far...")

                # Respect rate limit
                time.sleep(self._min_poll_interval)

            except Exception as e:
                logger.error(f"Error fetching {symbol} OHLCV: {e}")
                time.sleep(5)

        if not all_candles:
            logger.warning(f"No historical data for {symbol}")
            return pd.DataFrame(
                columns=["timestamp", "datetime", "open", "high", "low", "close", "volume"]
            )

        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(
            f"Fetched {len(df)} candles for {symbol} @ {interval} "
            f"({df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]})"
        )
        return df

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_exchange(self, exchange_name: str) -> ccxt.Exchange:
        """Get or create a CCXT exchange instance."""
        if exchange_name not in self._exchanges:
            exchange_class = getattr(ccxt, exchange_name)
            self._exchanges[exchange_name] = exchange_class({"enableRateLimit": True})
            logger.info(f"CCXT exchange initialized: {exchange_name}")
        return self._exchanges[exchange_name]

    def _poll_loop(self):
        """Background thread: poll for new closed candles."""
        logger.info("Poll loop started")

        while self._running:
            for key, sub in list(self._subscriptions.items()):
                if not self._running:
                    break
                try:
                    self._poll_subscription(sub)
                except Exception as e:
                    logger.error(f"Poll error for {sub.symbol}@{sub.interval}: {e}")

                time.sleep(self._min_poll_interval)

            # Sleep between full cycles — poll interval based on smallest subscribed interval
            min_interval_sec = min(
                (INTERVAL_SECONDS.get(sub.interval, 3600) for sub in self._subscriptions.values()),
                default=3600,
            )
            # Poll at 1/5 of candle interval (matches TwelveData divisor pattern)
            sleep_time = max(10, min_interval_sec // 5)
            self._wait(sleep_time)

    def _wait(self, seconds: float):
        """Interruptible sleep."""
        end = time.monotonic() + seconds
        while self._running and time.monotonic() < end:
            time.sleep(1)

    def _poll_subscription(self, sub: _Subscription):
        """Fetch latest 2 candles; if 2nd-to-last is new and closed, emit it."""
        ex = self._ensure_exchange(sub.exchange)

        ohlcv = ex.fetch_ohlcv(
            symbol=sub.symbol,
            timeframe=sub.interval,
            limit=2,
        )
        if not ohlcv or len(ohlcv) < 2:
            return

        # The second-to-last candle is guaranteed closed
        closed = ohlcv[0]  # [timestamp_ms, open, high, low, close, volume]
        ts_ms = closed[0]

        if ts_ms <= sub.last_candle_ts:
            return  # Already processed

        sub.last_candle_ts = ts_ms

        candle = Candle(
            timestamp=ts_ms,
            datetime=datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
            open=float(closed[1]),
            high=float(closed[2]),
            low=float(closed[3]),
            close=float(closed[4]),
            volume=float(closed[5]),
            symbol=sub.symbol,
            interval=sub.interval,
            is_closed=True,
        )

        # Persist to database
        if self._database:
            try:
                df = pd.DataFrame([{
                    "timestamp": candle.timestamp,
                    "datetime": candle.datetime,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                }])
                self._database.save_candles(df, symbol=candle.symbol, interval=candle.interval)
            except Exception as e:
                logger.error(f"Failed to save candle to DB: {e}")

        # Buffer
        sub.candle_buffer.append(candle)

        # Fire callbacks
        for cb in self._callbacks:
            try:
                cb(candle)
            except Exception as e:
                logger.error(f"Candle callback error: {e}")
