"""
Configuration Management
========================
Centralized configuration with validation and defaults.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


@dataclass
class DataConfig:
    """Data collection settings."""
    symbol: str = "BTC-USD"
    exchange: str = "coinbase"
    interval: str = "1h"
    history_days: int = 365
    websocket: Optional[Dict[str, Any]] = None


@dataclass
class AnalysisConfig:
    """Analysis engine settings."""
    update_interval: int = 60  # seconds
    min_confidence: float = 0.55
    lookback_period: int = 100


@dataclass
class ModelConfig:
    """ML model settings."""
    path: str = "models/best_model.pt"
    models_dir: str = "models"
    sequence_length: int = 60
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    features: Optional[Dict[str, Any]] = None


@dataclass
class SignalConfig:
    """Signal generation settings."""
    risk_per_trade: float = 0.02
    risk_reward_ratio: float = 2.0
    strong_signal: float = 0.65
    medium_signal: float = 0.55
    cooldown_minutes: int = 60


@dataclass
class NotificationConfig:
    """
    Notification settings.

    SECURITY: Telegram credentials should be set via environment variables:
        TELEGRAM_BOT_TOKEN
        TELEGRAM_CHAT_ID
    """
    desktop: bool = True
    sound: bool = True
    sound_file: Optional[str] = None
    telegram_enabled: bool = False
    # These are loaded from environment variables for security
    _telegram_bot_token: str = field(default="", repr=False)
    _telegram_chat_id: str = field(default="", repr=False)

    @property
    def telegram_bot_token(self) -> str:
        """Get Telegram bot token from environment variable (secure)."""
        import os
        return os.getenv('TELEGRAM_BOT_TOKEN', self._telegram_bot_token)

    @property
    def telegram_chat_id(self) -> str:
        """Get Telegram chat ID from environment variable (secure)."""
        import os
        return os.getenv('TELEGRAM_CHAT_ID', self._telegram_chat_id)


@dataclass
class DatabaseConfig:
    """Database settings."""
    path: str = "data/trading.db"
    backup_enabled: bool = True
    backup_interval_hours: int = 24


@dataclass
class LoggingConfig:
    """Logging settings."""
    level: str = "INFO"
    file: str = "data/trading.log"
    max_size_mb: int = 10
    backup_count: int = 3


@dataclass
class BrokerageConfig:
    """
    Brokerage settings (Lean-inspired).

    Supports multiple brokers:
    - paper: Simulated trading (default)
    - alpaca: Alpaca Securities (US stocks, options, crypto)
    - binance: Binance exchange (crypto)

    SECURITY: API credentials should be set via environment variables:
        ALPACA_API_KEY, ALPACA_SECRET_KEY
        BINANCE_API_KEY, BINANCE_SECRET_KEY
    """
    # Brokerage type: paper, alpaca, binance
    type: str = "paper"

    # Paper trading settings
    initial_cash: float = 10000.0
    commission_percent: float = 0.1
    slippage_percent: float = 0.05

    # Live trading settings
    paper_mode: bool = True  # Use paper/testnet for live brokers

    @property
    def is_paper(self) -> bool:
        """Check if using paper trading."""
        return self.type == "paper" or self.paper_mode

    @property
    def is_live(self) -> bool:
        """Check if using live trading (with real money)."""
        return self.type != "paper" and not self.paper_mode


@dataclass
class MT5Config:
    """
    MetaTrader 5 settings.

    SECURITY: MT5 credentials should be set via environment variables:
        MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_TERMINAL_PATH
    """
    enabled: bool = False
    demo: bool = True
    terminal_path: str = ""
    server: str = ""
    bridge_host: str = "localhost"
    bridge_port: int = 5555
    poll_interval_ms: int = 2000
    tick_poll_interval_ms: int = 500
    magic_number: int = 234000
    pairs: Optional[list] = None
    intervals: Optional[list] = None


@dataclass
class BacktestConfig:
    """Backtesting settings."""
    # Capital
    initial_cash: float = 100000.0

    # Simulation
    slippage_percent: float = 0.05
    commission_percent: float = 0.1

    # Position management
    max_open_positions: int = 1
    allow_concurrent: bool = False

    # Exit rules
    max_hold_candles: int = 24
    use_trailing_stop: bool = False
    trailing_stop_percent: float = 1.0


@dataclass
class Config:
    """
    Main configuration class.

    Loads from YAML file with sensible defaults.
    All settings are validated on load.
    """
    data: DataConfig = field(default_factory=DataConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    brokerage: BrokerageConfig = field(default_factory=BrokerageConfig)
    mt5: MT5Config = field(default_factory=MT5Config)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    # Raw config dict for backwards compatibility
    raw: Dict[str, Any] = field(default_factory=dict)

    _config_path: Optional[Path] = field(default=None, repr=False)

    @classmethod
    def load(cls, config_path: str = "config.yaml") -> "Config":
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to config file

        Returns:
            Config instance with loaded values
        """
        path = Path(config_path)

        if not path.exists():
            # Return defaults if no config file
            return cls()

        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}

        config = cls()
        config._config_path = path
        config.raw = data  # Store raw config dict

        # Load each section
        if 'data' in data:
            config.data = DataConfig(**data['data'])

        if 'analysis' in data:
            config.analysis = AnalysisConfig(**data['analysis'])

        if 'model' in data:
            config.model = ModelConfig(**data['model'])

        if 'signals' in data:
            config.signals = SignalConfig(**data['signals'])

        if 'notifications' in data:
            notif = data['notifications']
            telegram = notif.get('telegram', {})
            # Note: telegram credentials are loaded from environment variables
            # for security - see NotificationConfig properties
            config.notifications = NotificationConfig(
                desktop=notif.get('desktop', True),
                sound=notif.get('sound', True),
                sound_file=notif.get('sound_file'),
                telegram_enabled=telegram.get('enabled', False),
                _telegram_bot_token=telegram.get('bot_token', ''),
                _telegram_chat_id=telegram.get('chat_id', ''),
            )

        if 'database' in data:
            config.database = DatabaseConfig(**data['database'])

        if 'logging' in data:
            config.logging = LoggingConfig(**data['logging'])

        if 'brokerage' in data:
            config.brokerage = BrokerageConfig(**data['brokerage'])

        if 'mt5' in data:
            config.mt5 = MT5Config(**data['mt5'])

        if 'backtest' in data:
            config.backtest = BacktestConfig(**data['backtest'])

        # Validate
        config.validate()

        return config

    def validate(self):
        """Validate configuration values."""
        # Confidence thresholds
        if not 0 < self.analysis.min_confidence < 1:
            raise ValueError(f"min_confidence must be 0-1, got {self.analysis.min_confidence}")

        if not 0 < self.signals.strong_signal < 1:
            raise ValueError(f"strong_signal must be 0-1, got {self.signals.strong_signal}")

        if self.signals.medium_signal >= self.signals.strong_signal:
            raise ValueError("medium_signal must be less than strong_signal")

        # Risk settings
        if not 0 < self.signals.risk_per_trade < 0.5:
            raise ValueError(f"risk_per_trade must be 0-0.5, got {self.signals.risk_per_trade}")

        if self.signals.risk_reward_ratio < 1:
            raise ValueError(f"risk_reward_ratio must be >= 1, got {self.signals.risk_reward_ratio}")

        # Model settings
        if self.model.sequence_length < 10:
            raise ValueError(f"sequence_length must be >= 10, got {self.model.sequence_length}")

    def get_model_path(self) -> Path:
        """Get absolute path to model file."""
        return Path(self.model.path)

    def get_db_path(self) -> Path:
        """Get absolute path to database file."""
        path = Path(self.database.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_log_path(self) -> Path:
        """Get absolute path to log file."""
        path = Path(self.logging.file)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'data': {
                'symbol': self.data.symbol,
                'exchange': self.data.exchange,
                'interval': self.data.interval,
                'history_days': self.data.history_days,
            },
            'analysis': {
                'update_interval': self.analysis.update_interval,
                'min_confidence': self.analysis.min_confidence,
                'lookback_period': self.analysis.lookback_period,
            },
            'model': {
                'path': self.model.path,
                'sequence_length': self.model.sequence_length,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'dropout': self.model.dropout,
            },
            'signals': {
                'risk_per_trade': self.signals.risk_per_trade,
                'risk_reward_ratio': self.signals.risk_reward_ratio,
                'strong_signal': self.signals.strong_signal,
                'medium_signal': self.signals.medium_signal,
                'cooldown_minutes': self.signals.cooldown_minutes,
            },
            'notifications': {
                'desktop': self.notifications.desktop,
                'sound': self.notifications.sound,
                'telegram_enabled': self.notifications.telegram_enabled,
            },
            'database': {
                'path': self.database.path,
            },
            'logging': {
                'level': self.logging.level,
                'file': self.logging.file,
            },
            'brokerage': {
                'type': self.brokerage.type,
                'initial_cash': self.brokerage.initial_cash,
                'commission_percent': self.brokerage.commission_percent,
                'slippage_percent': self.brokerage.slippage_percent,
                'paper_mode': self.brokerage.paper_mode,
            },
            'backtest': {
                'initial_cash': self.backtest.initial_cash,
                'slippage_percent': self.backtest.slippage_percent,
                'commission_percent': self.backtest.commission_percent,
                'max_open_positions': self.backtest.max_open_positions,
                'max_hold_candles': self.backtest.max_hold_candles,
            },
        }

    def get_brokerage(self) -> 'BaseBrokerage':
        """
        Get configured brokerage instance.

        Returns:
            BaseBrokerage instance based on config
        """

        if self.brokerage.type == "paper":
            from src.paper_trading import PaperBrokerage
            return PaperBrokerage(
                initial_cash=self.brokerage.initial_cash,
                commission_percent=self.brokerage.commission_percent,
                slippage_percent=self.brokerage.slippage_percent
            )

        elif self.brokerage.type == "alpaca":
            from src.brokerages.alpaca import AlpacaBrokerage
            return AlpacaBrokerage(paper=self.brokerage.paper_mode)

        elif self.brokerage.type == "binance":
            from src.brokerages.binance import BinanceBrokerage
            return BinanceBrokerage(testnet=self.brokerage.paper_mode)

        elif self.brokerage.type == "mt5":
            from src.brokerages.mt5 import MT5Brokerage
            return MT5Brokerage(
                demo=self.mt5.demo,
                terminal_path=self.mt5.terminal_path,
                magic_number=self.mt5.magic_number,
            )

        else:
            raise ValueError(f"Unknown brokerage type: {self.brokerage.type}")


def load_config(config_path: str = 'config.yaml') -> dict:
    """
    Load configuration from YAML file as dictionary.

    This is a simple loader that returns the raw YAML structure.
    Use Config.load() for the dataclass-based configuration.

    Args:
        config_path: Path to config file

    Returns:
        Dictionary with configuration
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, 'r') as f:
        config = yaml.safe_load(f) or {}

    return config
