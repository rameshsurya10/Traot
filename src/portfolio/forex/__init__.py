"""
Forex Trading Module
====================

Comprehensive Forex trading support for the Traot.

This module provides:
- Currency pair configurations and constants
- Pip value calculations for all major and cross pairs
- Leverage management (1:50 US standard)
- Real-time spread tracking
- Swap/rollover rate management
- Forex-specific risk models
- Pip-based position sizing

Supported Pairs (14):
- Majors: EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD, NZD/USD
- Crosses: EUR/GBP, EUR/JPY, GBP/JPY, EUR/CHF, EUR/AUD, AUD/JPY, CHF/JPY

Usage:
    from src.portfolio.forex import (
        PipCalculator,
        LeverageManager,
        SpreadTracker,
        ForexPositionSizer,
        ForexLeverageRisk,
        SpreadAdjustmentRisk,
    )

    # Calculate position size
    sizer = ForexPositionSizer(max_risk_percent=1.0)
    result = sizer.calculate_position_size(
        symbol="EUR/USD",
        account_equity=10000,
        stop_pips=50,
        current_price=1.1000
    )

    # Check leverage
    leverage_mgr = LeverageManager(max_leverage=50.0)
    can_trade, margin, reason = leverage_mgr.can_open_position(
        account_equity=10000,
        position_value=100000,
        current_exposure=0
    )
"""

# Constants and configurations
from .constants import (
    # Dataclasses
    CurrencyPairConfig,
    # Enums
    LotType,
    # Constants
    US_MAX_LEVERAGE,
    EU_MAX_LEVERAGE,
    UK_MAX_LEVERAGE,
    MARGIN_CALL_LEVEL,
    STOP_OUT_LEVEL,
    ROLLOVER_HOUR_UTC,
    DEFAULT_ACCOUNT_CURRENCY,
    # Pair dictionaries
    MAJOR_PAIRS,
    CROSS_PAIRS,
    FOREX_PAIRS,
    CORRELATION_GROUPS,
    # Helper functions
    get_pair_config,
    is_forex_pair,
    is_jpy_pair,
    get_current_session,
)

# Pip calculations
from .pip_calculator import PipCalculator

# Leverage management
from .leverage_manager import (
    LeverageManager,
    LeverageState,
)

# Spread tracking
from .spread_tracker import (
    SpreadTracker,
    SpreadSnapshot,
    SpreadStats,
)

# Swap/rollover rates
from .swap_rates import (
    SwapRateManager,
    SwapRate,
    DEFAULT_SWAP_RATES,
)

# Risk models
from .risk_models import (
    ForexLeverageRisk,
    SpreadAdjustmentRisk,
    ForexCorrelationRisk,
    ForexSessionRisk,
)

# Position sizing
from .position_sizer import (
    ForexPositionSizer,
    PositionSizeResult,
    ForexKellyPosition,
    ForexATRPositionSizer,
)

__all__ = [
    # Constants
    "CurrencyPairConfig",
    "LotType",
    "US_MAX_LEVERAGE",
    "EU_MAX_LEVERAGE",
    "UK_MAX_LEVERAGE",
    "MARGIN_CALL_LEVEL",
    "STOP_OUT_LEVEL",
    "ROLLOVER_HOUR_UTC",
    "DEFAULT_ACCOUNT_CURRENCY",
    "MAJOR_PAIRS",
    "CROSS_PAIRS",
    "FOREX_PAIRS",
    "CORRELATION_GROUPS",
    "get_pair_config",
    "is_forex_pair",
    "is_jpy_pair",
    "get_current_session",
    # Pip Calculator
    "PipCalculator",
    # Leverage
    "LeverageManager",
    "LeverageState",
    # Spread
    "SpreadTracker",
    "SpreadSnapshot",
    "SpreadStats",
    # Swap
    "SwapRateManager",
    "SwapRate",
    "DEFAULT_SWAP_RATES",
    # Risk Models
    "ForexLeverageRisk",
    "SpreadAdjustmentRisk",
    "ForexCorrelationRisk",
    "ForexSessionRisk",
    # Position Sizing
    "ForexPositionSizer",
    "PositionSizeResult",
    "ForexKellyPosition",
    "ForexATRPositionSizer",
]

__version__ = "1.0.0"
