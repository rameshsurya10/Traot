"""
Type Definitions
================
Clean, typed data structures for the trading system.
All data flows through these types for consistency.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any


class SignalType(Enum):
    """Trading signal direction."""
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"


class SignalStrength(Enum):
    """Signal confidence level."""
    STRONG = "STRONG"
    MEDIUM = "MEDIUM"
    WEAK = "WEAK"


@dataclass
class Candle:
    """Single OHLCV candle."""
    timestamp: int
    datetime: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = ""
    interval: str = ""
    is_closed: bool = True  # Candles from on_candle_closed are always closed

    @property
    def is_bullish(self) -> bool:
        """Check if candle closed higher than open."""
        return self.close > self.open

    @property
    def body_size(self) -> float:
        """Absolute size of candle body."""
        return abs(self.close - self.open)

    @property
    def range_size(self) -> float:
        """Total range from high to low."""
        return self.high - self.low

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'datetime': self.datetime.isoformat() if isinstance(self.datetime, datetime) else str(self.datetime),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'symbol': self.symbol,
            'interval': self.interval,
        }


@dataclass
class Tick:
    """Real-time price tick from exchange."""
    symbol: str
    price: float
    timestamp: int  # Unix timestamp in milliseconds
    quantity: float = 0.0


@dataclass
class Prediction:
    """ML model prediction result."""
    timestamp: datetime
    price: float
    probability: float  # 0-1, >0.5 means UP
    signal_type: SignalType
    confidence: float
    using_ml: bool = True

    # Technical indicators at prediction time
    rsi: Optional[float] = None
    macd_hist: Optional[float] = None
    bb_position: Optional[float] = None
    atr: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            'price': self.price,
            'probability': self.probability,
            'signal_type': self.signal_type.value,
            'confidence': self.confidence,
            'using_ml': self.using_ml,
            'rsi': self.rsi,
            'macd_hist': self.macd_hist,
            'bb_position': self.bb_position,
            'atr': self.atr,
        }


@dataclass
class Signal:
    """Actionable trading signal."""
    id: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    signal_type: SignalType = SignalType.NEUTRAL
    strength: SignalStrength = SignalStrength.WEAK
    confidence: float = 0.0
    price: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    atr: Optional[float] = None

    # For tracking performance
    actual_outcome: Optional[str] = None  # 'WIN', 'LOSS', 'PENDING'
    outcome_price: Optional[float] = None
    outcome_timestamp: Optional[datetime] = None
    pnl_percent: Optional[float] = None

    def __post_init__(self):
        """Validate signal data."""
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")

    @property
    def is_actionable(self) -> bool:
        """Check if signal should be acted upon."""
        return (
            self.signal_type != SignalType.NEUTRAL and
            self.strength != SignalStrength.WEAK
        )

    @property
    def risk_reward_ratio(self) -> Optional[float]:
        """Calculate risk:reward ratio if levels are set."""
        if not self.stop_loss or not self.take_profit:
            return None
        risk = abs(self.price - self.stop_loss)
        reward = abs(self.take_profit - self.price)
        if risk == 0:
            return None
        return reward / risk

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            'signal_type': self.signal_type.value,
            'strength': self.strength.value,
            'confidence': self.confidence,
            'price': self.price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'atr': self.atr,
            'actual_outcome': self.actual_outcome,
            'outcome_price': self.outcome_price,
            'pnl_percent': self.pnl_percent,
        }

    def format_message(self) -> str:
        """Format signal as human-readable message."""
        direction = self.signal_type.value
        emoji = '🟢' if direction == 'BUY' else '🔴' if direction == 'SELL' else '⚪'
        strength_emoji = '💪' if self.strength == SignalStrength.STRONG else '👍'

        lines = [
            f"{emoji} {self.strength.value} {direction} SIGNAL {strength_emoji}",
            "",
            f"📊 Price: ${self.price:,.2f}",
            f"🎯 Confidence: {self.confidence:.1%}",
        ]

        if self.stop_loss and self.take_profit:
            rr = self.risk_reward_ratio or 0
            lines.extend([
                "",
                "📍 Levels:",
                f"   🛑 Stop Loss: ${self.stop_loss:,.2f}",
                f"   ✅ Take Profit: ${self.take_profit:,.2f}",
                f"   📏 Risk:Reward = 1:{rr:.1f}",
            ])

        lines.extend([
            "",
            f"⏰ Time: {self.timestamp}",
        ])

        return "\n".join(lines)


@dataclass
class TradeResult:
    """Result of a simulated or real trade (for backtesting/tracking)."""
    signal_id: int
    entry_price: float
    entry_time: datetime
    exit_price: float
    exit_time: datetime
    direction: SignalType
    stop_loss: float
    take_profit: float

    # Outcome
    hit_target: bool = False
    hit_stop: bool = False
    pnl_percent: float = 0.0
    pnl_absolute: float = 0.0

    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl_percent > 0

    @property
    def duration_minutes(self) -> float:
        """Trade duration in minutes."""
        return (self.exit_time - self.entry_time).total_seconds() / 60

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'signal_id': self.signal_id,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat(),
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.isoformat(),
            'direction': self.direction.value,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'hit_target': self.hit_target,
            'hit_stop': self.hit_stop,
            'pnl_percent': self.pnl_percent,
            'pnl_absolute': self.pnl_absolute,
            'is_winner': self.is_winner,
            'duration_minutes': self.duration_minutes,
        }
