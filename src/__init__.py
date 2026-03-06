"""
Traot - Professional Trading Signal System
=================================================

A production-ready trading signal system with ML predictions.

Modules:
--------
- core: Configuration, database, types, logging, metrics, validation
- live_trading: Live/paper trading runner
- portfolio: Position sizing, risk management
- universe: Asset filtering and selection
- brokerages: Exchange integrations
- analysis_engine: Feature engineering and ML predictions
- advanced_predictor: Fourier, Kalman, Monte Carlo predictions
- multi_currency_system: Multi-currency trading with auto-learning

Usage:
------
    from src import DataService, AnalysisEngine
    from src.core import Config, Database, MetricsCalculator
    from src.advanced_predictor import AdvancedPredictor
    from src.live_trading import LiveTradingRunner

Quick Start:
------------
    # Run analysis (paper trading)
    python run_analysis.py

    # Start unified dashboard
    streamlit run dashboard_unified.py
"""

__version__ = "3.0.0"
__author__ = "Traot"

# Core services (non-torch)
try:
    from .data_service import DataService
except ImportError:
    DataService = None

from .notifier import Notifier

# Torch-dependent imports (optional)
try:
    from .analysis_engine import AnalysisEngine, FeatureCalculator, LSTMModel
    TORCH_AVAILABLE = True
except ImportError:
    AnalysisEngine = None
    FeatureCalculator = None
    LSTMModel = None
    TORCH_AVAILABLE = False

# Advanced predictor (mathematical algorithms - no torch required)
from .advanced_predictor import (
    AdvancedPredictor,
    FourierAnalyzer,
    KalmanFilter,
    EntropyAnalyzer,
    MarkovChain,
    MonteCarlo
)

# Multi-currency system (may require torch)
try:
    from .multi_currency_system import MultiCurrencySystem, CurrencyConfig, PerformanceStats
except ImportError:
    MultiCurrencySystem = None
    CurrencyConfig = None
    PerformanceStats = None

# Utilities
from .utils import (
    load_config,
    get_config_value,
    get_db_connection,
    get_db_path,
    get_project_root,
    get_data_config,
    get_signal_config,
    get_model_config,
    get_dashboard_config,
    get_notification_config,
    get_analysis_config
)

__all__ = [
    # Services
    "DataService",
    "AnalysisEngine",
    "FeatureCalculator",
    "LSTMModel",
    "Notifier",

    # Advanced Predictor
    "AdvancedPredictor",
    "FourierAnalyzer",
    "KalmanFilter",
    "EntropyAnalyzer",
    "MarkovChain",
    "MonteCarlo",

    # Multi-currency
    "MultiCurrencySystem",
    "CurrencyConfig",
    "PerformanceStats",

    # Utils
    "load_config",
    "get_config_value",
    "get_db_connection",
    "get_db_path",
    "get_project_root",
    "get_data_config",
    "get_signal_config",
    "get_model_config",
    "get_dashboard_config",
    "get_notification_config",
    "get_analysis_config",
]
