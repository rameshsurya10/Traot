"""
Traot - Unified Dashboard
=================================
Streamlined dashboard with consolidated navigation.

Pages:
- Dashboard: Real-time chart, AI predictions, market analysis
- Trading Center: AI Trading, Portfolio, Performance (3 tabs)
- Forex Markets: (forex mode only) Currency pairs and position sizing
- Settings: Trading, Risk, Learning, Notifications, System

Usage:
    streamlit run dashboard.py
"""

import sys
from pathlib import Path

# Add project root to path for imports
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import json
import sqlite3
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import logging
import yaml
import os
import signal
import subprocess
import html
import atexit
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_PATH = ROOT / "config.yaml"
PID_FILE = ROOT / "run_analysis.pid"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Traot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# MODULE IMPORTS (with fallbacks)
# =============================================================================

# WebSocket Data Provider (REQUIRED - no fallback)
# Import directly to avoid torch dependency from src/__init__.py
from src.data.provider import UnifiedDataProvider
from src.learning.strategy_analyzer import StrategyAnalyzer

try:
    from src.core.database import Database
    from src.core.metrics import MetricsCalculator
    from src.core.validation import OrderValidator
    from src.paper_trading import PaperTradingSimulator, OrderSide, OrderType
    from src.advanced_predictor import AdvancedPredictor
    from src.news.collector import NewsCollector
    from src.news.aggregator import SentimentAggregator

    # CONTINUOUS LEARNING: Strategic Learning Bridge
    from src.learning.strategic_learning_bridge import StrategicLearningBridge

    # PERFORMANCE-BASED LEARNING: Per-candle learning system
    from src.learning.performance_learner import PerformanceBasedLearner, create_performance_learner

    AI_AVAILABLE = True
    LEARNING_AVAILABLE = True
    PERF_LEARNING_AVAILABLE = True
except ImportError as e:
    AI_AVAILABLE = False
    LEARNING_AVAILABLE = False
    PERF_LEARNING_AVAILABLE = False
    logger.warning(f"AI modules not available: {e}")

# =============================================================================
# STYLES
# =============================================================================

st.markdown("""
<style>
    /* Hide Streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* Sidebar - Always Visible */
    [data-testid="stSidebar"] {
        min-width: 280px !important;
        width: 280px !important;
        transform: none !important;
    }

    [data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 280px !important;
        width: 280px !important;
        margin-left: 0 !important;
        transform: none !important;
    }

    [data-testid="stSidebar"] button[kind="header"] {
        display: none !important;
    }

    /* Responsive Chart Container */
    .stPlotlyChart {
        width: 100% !important;
    }
    .stPlotlyChart > div {
        width: 100% !important;
    }
    .js-plotly-plot {
        width: 100% !important;
    }
    .plot-container {
        width: 100% !important;
    }

    /* Main content area */
    .main .block-container {
        padding: 1rem 2rem;
        max-width: 100%;
    }

    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        margin-bottom: 0.5rem;
    }
    .metric-card .label {
        font-size: 0.75rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a2e;
        margin: 0.3rem 0;
    }
    .metric-card .delta { font-size: 0.9rem; }
    .metric-card.positive { border-left-color: #28a745; }
    .metric-card.positive .delta { color: #28a745; }
    .metric-card.negative { border-left-color: #dc3545; }
    .metric-card.negative .delta { color: #dc3545; }
    .metric-card.warning { border-left-color: #ffc107; }

    /* Signal Card */
    .signal-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    .signal-buy { border: 3px solid #28a745; }
    .signal-sell { border: 3px solid #dc3545; }
    .signal-neutral { border: 3px solid #6c757d; }

    /* Tables */
    .dataframe { font-size: 0.85rem; }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def _load_config_sync() -> dict:
    """Load config synchronously for initialization (no caching)."""
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, 'r') as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}


def init_session_state():
    """Initialize all session state variables from config.yaml."""
    # Load saved settings from config.yaml
    saved_config = _load_config_sync()
    data_config = saved_config.get('data', {})
    portfolio_config = saved_config.get('portfolio', {})

    defaults = {
        # Navigation
        'page': 'Dashboard',

        # Market mode (crypto or forex)
        'market_mode': saved_config.get('market_mode', 'crypto'),

        # Market settings - load from config.yaml if available
        'exchange': data_config.get('exchange', 'binance'),
        'symbol': data_config.get('symbol', 'BTC/USDT'),
        'timeframe': data_config.get('interval', '1h'),

        # Smart auto-refresh (off by default to save resources)
        # Auto-refreshes ONLY when: data loading, predictions pending, or manually enabled
        'auto_refresh': False,  # User can enable in settings
        'refresh_interval': 30,

        # WebSocket Data Provider (ONLY data source)
        'data_provider': None,

        # AI/ML
        'advanced_predictor': None,

        # CONTINUOUS LEARNING: Strategic Learning Bridge
        'learning_bridge': None,
        'use_continuous_learning': True,  # Enable continuous learning by default

        # PERFORMANCE-BASED LEARNING: Per-candle learning
        'performance_learner': None,
        'perf_learning_config': {
            'enabled': True,
            'loss_retrain': True,
            'reinforce_wins': True,
            'consecutive_loss_threshold': 3,
            'light_epochs': 30,
            'medium_epochs': 50,
            'full_epochs': 100,
            'win_rate_threshold': 0.45,
        },

        # Prediction Validation (8/10 streak system)
        'prediction_validator': None,
        'validation_worker_started': False,

        # Prediction cache (matches refresh interval)
        'last_prediction': None,
        'last_prediction_time': 0,
        'prediction_cache_seconds': 30,

        # Paper trading
        'paper_trader': None,
        'paper_capital': portfolio_config.get('initial_capital', 10000),

        # Database
        'db': None,

        # Metrics
        'metrics_calculator': None,

        # Order validator
        'order_validator': None,

        # News & Sentiment
        'news_collector': None,
        'sentiment_aggregator': None,

        # Tracked currencies (based on market mode)
        'tracked_symbols': ['EUR/USD', 'GBP/USD'] if saved_config.get('market_mode', 'crypto') == 'forex' else ['BTC/USDT', 'ETH/USDT'],

        # Model Training Status (MANDATORY before predictions)
        'model_ready': False,
        'model_accuracy': 0.0,
        'model_trained_at': None,
        'training_in_progress': False,
        'training_progress': 0,
        'training_epoch': 0,
        'training_status': '',
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# =============================================================================
# INITIALIZATION
# =============================================================================

@st.cache_resource
def get_data_provider():
    """Get singleton UnifiedDataProvider instance."""
    return UnifiedDataProvider.get_instance(str(CONFIG_PATH))


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_config() -> dict:
    """
    Load configuration file with caching.

    Returns:
        dict: Configuration dictionary with '_error' key if failed, empty dict if file doesn't exist.
    """
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in config file: {e}")
            return {'_error': f"Configuration file is malformed: {e}"}
        except PermissionError as e:
            logger.error(f"Permission denied reading config: {e}")
            return {'_error': f"Permission denied: Cannot read {CONFIG_PATH}. Check file permissions."}
        except Exception as e:
            logger.error(f"Error reading config file: {e}")
            return {'_error': f"Error reading configuration: {e}"}
    logger.warning(f"Config file not found: {CONFIG_PATH}")
    return {}


def check_model_readiness(symbol: str, timeframe: str) -> dict:
    """
    Check if model exists and is validated for predictions.

    Returns:
        dict with keys: ready (bool), accuracy (float), trained_at (str), reason (str)
    """
    import torch
    from pathlib import Path

    config = load_config()
    models_dir = Path(config.get('model', {}).get('models_dir', 'models'))

    # Get model path
    safe_symbol = symbol.replace("/", "_").replace("-", "_")
    model_path = models_dir / f"model_{safe_symbol}_{timeframe}.pt"

    if not model_path.exists():
        return {
            'ready': False,
            'accuracy': 0.0,
            'trained_at': None,
            'reason': 'No model found'
        }

    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        accuracy = checkpoint.get('validation_accuracy', 0.0)
        trained_at = checkpoint.get('trained_at', 'unknown')

        # Get minimum required accuracy from config
        min_accuracy = config.get('auto_training', {}).get('min_accuracy_required', 0.58)

        if accuracy >= min_accuracy:
            return {
                'ready': True,
                'accuracy': accuracy,
                'trained_at': trained_at,
                'reason': f'Model validated ({accuracy:.1%} >= {min_accuracy:.1%})'
            }
        else:
            return {
                'ready': False,
                'accuracy': accuracy,
                'trained_at': trained_at,
                'reason': f'Below minimum accuracy ({accuracy:.1%} < {min_accuracy:.1%})'
            }
    except Exception as e:
        return {
            'ready': False,
            'accuracy': 0.0,
            'trained_at': None,
            'reason': f'Could not read model: {e}'
        }


def train_model(symbol: str, timeframe: str, progress_callback=None):
    """
    Train model for symbol/timeframe with progress updates.

    Args:
        symbol: Trading pair
        timeframe: Candle interval
        progress_callback: Function to call with (epoch, max_epochs, accuracy, status_msg)
    """
    import torch
    import torch.nn as nn
    import numpy as np
    from pathlib import Path
    from datetime import datetime
    from src.analysis_engine import FeatureCalculator, LSTMModel

    config = load_config()
    auto_train_config = config.get('auto_training', {})
    model_config = config.get('model', {})

    # Training parameters
    training_candles = auto_train_config.get('training_candles', 5000)
    min_candles = auto_train_config.get('min_candles', 1000)
    target_accuracy = auto_train_config.get('target_accuracy', 0.65)
    min_accuracy_required = auto_train_config.get('min_accuracy_required', 0.58)
    max_epochs = auto_train_config.get('max_epochs', 100)
    batch_size = auto_train_config.get('batch_size', 32)
    learning_rate = auto_train_config.get('learning_rate', 0.001)

    # Model parameters
    hidden_size = model_config.get('hidden_size', 128)
    num_layers = model_config.get('num_layers', 2)
    dropout = model_config.get('dropout', 0.2)
    sequence_length = model_config.get('sequence_length', 60)

    # Get database
    db = st.session_state.db
    if not db:
        raise RuntimeError("Database not initialized")

    # Fetch training data
    if progress_callback:
        progress_callback(0, max_epochs, 0, f"Fetching {training_candles} candles...")

    candles_df = db.get_candles(
        symbol=symbol,
        interval=timeframe,
        limit=training_candles + 100
    )

    if candles_df is None or len(candles_df) < min_candles:
        raise ValueError(
            f"Insufficient data: {len(candles_df) if candles_df is not None else 0} < {min_candles} candles"
        )

    # Calculate features
    if progress_callback:
        progress_callback(0, max_epochs, 0, "Calculating features...")

    df_features = FeatureCalculator.calculate_all(candles_df)
    feature_columns = FeatureCalculator.get_feature_columns()

    # Extract and normalize features FIRST (before creating target)
    features = df_features[feature_columns].values
    closes = df_features['close'].values

    feature_means = np.nanmean(features, axis=0)
    feature_stds = np.nanstd(features, axis=0)
    features = (features - feature_means) / (feature_stds + 1e-8)
    features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

    # Create sequences
    if progress_callback:
        progress_callback(0, max_epochs, 0, "Creating sequences...")

    # Create sliding windows manually for correct shape
    # Need shape: (num_samples, sequence_length, num_features)
    num_features = features.shape[1]
    num_sequences = len(features) - sequence_length - 1  # -1 for target

    X = np.zeros((num_sequences, sequence_length, num_features))
    y = np.zeros(num_sequences)

    # ==================================================================================
    # STRATEGIC TARGET: Learn PROFITABLE trades with CLEAR TP/SL rules
    # ==================================================================================
    #
    # TRADING STRATEGY (CRYSTAL CLEAR):
    # - Take Profit (TP): 2.0% gain from entry
    # - Stop Loss (SL): 1.0% loss from entry
    # - Risk/Reward Ratio: 2:1 (risk $1 to make $2)
    # - Maximum Hold Time: 10 candles
    #   * For 15m timeframe: 10 × 15min = 2.5 hours maximum
    #   * For 1h timeframe: 10 × 1h = 10 hours maximum
    #
    # EXAMPLE FOR BTC @ $95,000:
    # - Entry: $95,000
    # - Take Profit: $96,900 (+2.0% = +$1,900 profit)
    # - Stop Loss: $94,050 (-1.0% = -$950 loss)
    # - Hold: Maximum 10 candles, then exit
    #
    # The model learns: "Which patterns lead to +2% BEFORE -1% within 10 candles?"
    # ==================================================================================

    highs = df_features['high'].values
    lows = df_features['low'].values

    # FIXED percentage targets (clear and simple)
    TP_PERCENT = 0.02  # 2% take profit (realistic for crypto)
    SL_PERCENT = 0.01  # 1% stop loss (tight but reasonable)
    MAX_HOLD_CANDLES = 10  # Maximum candles to hold position

    # Track target distribution for logging
    buy_signals = 0
    sell_signals = 0
    neutral_signals = 0

    for i in range(num_sequences):
        # Sequence uses candles [i] to [i+sequence_length-1]
        X[i] = features[i:i + sequence_length]

        # Entry: close price of the last candle in sequence
        entry_price = closes[i + sequence_length - 1]

        # === LONG (BUY) TRADE TARGETS ===
        tp_long = entry_price * (1.0 + TP_PERCENT)  # +2%
        sl_long = entry_price * (1.0 - SL_PERCENT)  # -1%

        # === SHORT (SELL) TRADE TARGETS ===
        tp_short = entry_price * (1.0 - TP_PERCENT)  # -2%
        sl_short = entry_price * (1.0 + SL_PERCENT)  # +1%

        # Look ahead to see what happens
        lookahead = min(MAX_HOLD_CANDLES, len(closes) - (i + sequence_length))

        # Track outcomes
        long_outcome = None  # 'WIN' if TP hit first, 'LOSS' if SL hit first
        short_outcome = None

        for j in range(lookahead):
            candle_idx = i + sequence_length + j
            high = highs[candle_idx]
            low = lows[candle_idx]

            # Check LONG trade (if not already decided)
            if long_outcome is None:
                # Check SL first (conservative: assume SL hit if both touched same candle)
                if low <= sl_long:
                    long_outcome = 'LOSS'
                elif high >= tp_long:
                    long_outcome = 'WIN'

            # Check SHORT trade (if not already decided)
            if short_outcome is None:
                # Check SL first (conservative: assume SL hit if both touched same candle)
                if high >= sl_short:
                    short_outcome = 'LOSS'
                elif low <= tp_short:
                    short_outcome = 'WIN'

        # === DECISION LOGIC ===
        # Only signal BUY/SELL if ONE direction wins clearly
        # Signal NEUTRAL if both lose, both win, or unclear

        if long_outcome == 'WIN' and short_outcome != 'WIN':
            # LONG wins, SHORT doesn't → BUY SIGNAL
            y[i] = 1.0
            buy_signals += 1

        elif short_outcome == 'WIN' and long_outcome != 'WIN':
            # SHORT wins, LONG doesn't → SELL SIGNAL
            y[i] = 0.0
            sell_signals += 1

        else:
            # NEUTRAL - don't trade when:
            # - Both lose (choppy market)
            # - Both win (contradictory - impossible but check anyway)
            # - Neither wins within time limit (weak signal)
            y[i] = 0.5
            neutral_signals += 1

    # Log target distribution
    total_samples = buy_signals + sell_signals + neutral_signals
    logger.info(
        f"Target distribution: "
        f"BUY={buy_signals} ({buy_signals/total_samples*100:.1f}%), "
        f"SELL={sell_signals} ({sell_signals/total_samples*100:.1f}%), "
        f"NEUTRAL={neutral_signals} ({neutral_signals/total_samples*100:.1f}%)"
    )

    # Remove any invalid entries
    valid = ~(np.isnan(y) | np.isnan(X).any(axis=(1, 2)))
    X = X[valid]
    y = y[valid]

    if len(X) < min_candles:
        raise ValueError(f"After sequence creation, only {len(X)} samples")

    # Train/val split
    split_idx = int(len(X) * 0.8)
    min_val_samples = 50
    if len(X) - split_idx < min_val_samples:
        if len(X) < min_val_samples * 2:
            raise ValueError("Insufficient data for reliable training")
        split_idx = len(X) - min_val_samples

    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size
    )

    # Create model
    device = torch.device('cpu')  # Use CPU for dashboard (GPU causes issues in Streamlit)
    model = LSTMModel(
        input_size=len(feature_columns),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0
    best_state = None
    patience = 10
    patience_counter = 0

    # Training loop
    for epoch in range(max_epochs):
        # Train
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                predictions = (outputs > 0.5).float()
                correct += (predictions == y_batch).sum().item()
                total += len(y_batch)

        val_acc = correct / total if total > 0 else 0

        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # Progress update
        if progress_callback:
            avg_loss = epoch_loss / len(train_loader)
            status = f"Epoch {epoch+1}/{max_epochs} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.2%}"
            progress_callback(epoch + 1, max_epochs, val_acc, status)

        # Early stopping
        if patience_counter >= patience:
            break

        # Target reached
        if val_acc >= target_accuracy:
            break

    # Save model
    if best_state is None:
        raise RuntimeError("Training failed - no valid model state")

    model.cpu()
    model.load_state_dict(best_state)

    models_dir = Path(config.get('model', {}).get('models_dir', 'models'))
    models_dir.mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.replace("/", "_").replace("-", "_")
    model_path = models_dir / f"model_{safe_symbol}_{timeframe}.pt"

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'sequence_length': sequence_length
        },
        'feature_means': feature_means,
        'feature_stds': feature_stds,
        'symbol': symbol,
        'interval': timeframe,
        'trained_at': datetime.utcnow().isoformat(),
        'samples_trained': len(X_train),
        'validation_accuracy': best_val_acc
    }, model_path)

    # Check if validated
    if best_val_acc < min_accuracy_required:
        raise RuntimeError(
            f"Model below minimum accuracy: {best_val_acc:.2%} < {min_accuracy_required:.2%}"
        )

    return {
        'accuracy': best_val_acc,
        'samples': len(X_train),
        'epochs': epoch + 1
    }


def initialize_components():
    """Initialize all components."""
    # Check model readiness on initialization
    if not st.session_state.training_in_progress:
        model_status = check_model_readiness(st.session_state.symbol, st.session_state.timeframe)
        st.session_state.model_ready = model_status['ready']
        st.session_state.model_accuracy = model_status['accuracy']
        st.session_state.model_trained_at = model_status['trained_at']

    # WebSocket Data Provider (ONLY data source)
    if st.session_state.data_provider is None:
        st.session_state.data_provider = get_data_provider()
        logger.info("Got data provider singleton")

    # Always check if provider needs to be started (may be cached but not running)
    provider = st.session_state.data_provider
    if provider and not provider.is_running:
        logger.info("Starting WebSocket provider...")

        # Subscribe to tracked symbols
        for symbol in st.session_state.tracked_symbols:
            provider.subscribe(
                symbol,
                exchange=st.session_state.exchange,
                interval=st.session_state.timeframe
            )

        # Start the provider
        provider.start()
        logger.info(f"WebSocket provider started for {st.session_state.tracked_symbols}")

    # Paper trader
    if st.session_state.paper_trader is None and AI_AVAILABLE:
        st.session_state.paper_trader = PaperTradingSimulator(
            initial_cash=st.session_state.paper_capital
        )

    # Database (required for prediction validator)
    if st.session_state.db is None and AI_AVAILABLE:
        db_path = ROOT / "data" / "trading.db"
        db_path.parent.mkdir(exist_ok=True)
        st.session_state.db = Database(str(db_path))

    # Prediction Validator (requires database)
    if st.session_state.prediction_validator is None and AI_AVAILABLE and st.session_state.db:
        from src.learning.prediction_validator import PredictionValidator
        st.session_state.prediction_validator = PredictionValidator(
            database=st.session_state.db,
            streak_required=8,
            min_data_years=1
        )

    # Start prediction validation background worker (runs once per session)
    if (not st.session_state.validation_worker_started and
        st.session_state.prediction_validator is not None and
        st.session_state.data_provider is not None):

        import threading

        # Initialize stop event for thread cleanup
        if 'validation_worker_stop' not in st.session_state:
            st.session_state.validation_worker_stop = threading.Event()

        # Capture instances in closure (thread-safe - no session_state access in thread)
        validator = st.session_state.prediction_validator
        data_provider = st.session_state.data_provider
        stop_event = st.session_state.validation_worker_stop

        def prediction_validation_worker():
            """
            Background thread that validates predictions when candles close.

            Runs every 60 seconds to check pending predictions and validate
            them against actual candle closes.

            Thread-safe: Uses captured instances, NOT session_state.
            """
            while not stop_event.is_set():
                try:
                    # Sleep in 5-second intervals to allow quick stop
                    for _ in range(12):  # 12 * 5 = 60 seconds
                        if stop_event.is_set():
                            logger.info("🛑 Validation worker stopped")
                            return
                        time.sleep(5)

                    # Use captured instances (thread-safe)
                    if validator and data_provider:
                        # Get all pending predictions from DATABASE (not just memory!)
                        # This ensures we validate ALL pending predictions, even after restart
                        pending_predictions = validator.get_pending_predictions_from_db()

                        logger.info(f"🔍 Checking {len(pending_predictions)} pending predictions from database")

                        for pred in pending_predictions:
                            if stop_event.is_set():
                                return

                            symbol = pred['symbol']
                            timeframe = pred['timeframe']
                            timestamp = pred['timestamp']

                            # Calculate when candle should close
                            candle_interval = validator._get_candle_interval_ms(timeframe)
                            expected_close = ((timestamp // candle_interval) + 1) * candle_interval
                            current_time = int(time.time() * 1000)

                            # Check if candle should have closed
                            if current_time >= expected_close:
                                try:
                                    # Fetch latest candle
                                    logger.debug(f"  Fetching data for {symbol} @ {timeframe}")
                                    df = data_provider.get_candles(
                                        symbol=symbol,
                                        interval=timeframe,
                                        limit=1
                                    )

                                    if df is not None and len(df) > 0:
                                        latest = df.iloc[-1]
                                        actual_price = float(latest['close'])
                                        candle_time = int(latest['timestamp'])

                                        # Validate prediction
                                        validator.validate_prediction(
                                            symbol=symbol,
                                            timeframe=timeframe,
                                            actual_price=actual_price,
                                            candle_close_time=candle_time
                                        )
                                        logger.info(f"✅ Auto-validated prediction: {symbol} {timeframe} @ ${actual_price:,.2f}")
                                    else:
                                        logger.warning(f"⚠️ No data for validation: {symbol} {timeframe}")

                                except Exception as e:
                                    logger.error(f"❌ Validation error for {symbol} {timeframe}: {e}")
                                    import traceback
                                    logger.error(traceback.format_exc())

                        # Cleanup stale predictions (older than 2x candle interval)
                        cleaned = validator.cleanup_stale_predictions()
                        if cleaned > 0:
                            logger.info(f"🗑️ Cleaned {cleaned} stale predictions")

                except Exception as e:
                    logger.error(f"❌ Validation worker error: {e}")

        # Start background thread
        validation_thread = threading.Thread(
            target=prediction_validation_worker,
            daemon=True,
            name="PredictionValidationWorker"
        )
        validation_thread.start()
        st.session_state.validation_thread = validation_thread
        st.session_state.validation_worker_started = True

        # Register cleanup on exit (prevent memory leak)
        def cleanup_validation_thread():
            """Stop validation thread when process exits."""
            logger.info("🧹 Cleaning up validation worker...")
            stop_event.set()
            if validation_thread.is_alive():
                validation_thread.join(timeout=2)
                logger.info("✅ Validation worker cleaned up")

        atexit.register(cleanup_validation_thread)
        logger.info("🚀 Prediction validation background worker started (thread-safe)")

    # Advanced predictor (requires prediction validator for recording predictions)
    if st.session_state.advanced_predictor is None and AI_AVAILABLE:
        config = load_config()
        st.session_state.advanced_predictor = AdvancedPredictor(
            prediction_validator=st.session_state.prediction_validator,
            config=config
        )

    # CONTINUOUS LEARNING: Initialize Strategic Learning Bridge
    if st.session_state.learning_bridge is None and LEARNING_AVAILABLE and st.session_state.use_continuous_learning:
        try:
            logger.info("🧠 Initializing Continuous Learning System...")

            # Get paper brokerage (uses paper trading simulator)
            paper_brokerage = st.session_state.paper_trader  # Fixed: was paper_simulator

            # Load config for learning bridge
            learning_config = load_config()

            # Initialize bridge (only if db and predictor are available)
            if st.session_state.db and st.session_state.advanced_predictor and paper_brokerage:
                st.session_state.learning_bridge = StrategicLearningBridge(
                    database=st.session_state.db,  # Fixed: was database
                    predictor=st.session_state.advanced_predictor,
                    paper_brokerage=paper_brokerage,
                    live_brokerage=None,  # Dashboard is paper-only
                    config=learning_config
                )
                logger.info("✅ Continuous Learning System initialized")
            else:
                logger.debug("Learning bridge dependencies not ready yet")

        except Exception as e:
            logger.error(f"Failed to initialize learning bridge: {e}", exc_info=True)
            st.session_state.learning_bridge = None

    # PERFORMANCE-BASED LEARNING: Initialize PerformanceBasedLearner
    if st.session_state.performance_learner is None and PERF_LEARNING_AVAILABLE:
        try:
            perf_config = st.session_state.get('perf_learning_config', {})
            if st.session_state.advanced_predictor and st.session_state.db and perf_config.get('enabled', True):
                st.session_state.performance_learner = create_performance_learner(
                    predictor=st.session_state.advanced_predictor,
                    database=st.session_state.db,
                    timeframes=['15m', '1h'],
                    loss_retrain=perf_config.get('loss_retrain', True),
                    reinforce_wins=perf_config.get('reinforce_wins', True)
                )
                logger.info("✅ PerformanceBasedLearner initialized")
        except Exception as e:
            logger.debug(f"PerformanceBasedLearner init deferred: {e}")

    # Metrics calculator
    if st.session_state.metrics_calculator is None and AI_AVAILABLE:
        st.session_state.metrics_calculator = MetricsCalculator()

    # Order validator
    if st.session_state.order_validator is None and AI_AVAILABLE:
        config = load_config()
        st.session_state.order_validator = OrderValidator(config.get('risk', {}))

    # News collector and sentiment aggregator
    if st.session_state.news_collector is None and AI_AVAILABLE:
        config = load_config()
        news_config = config.get('news', {})
        if news_config.get('enabled', False):
            # Validate required API keys
            required_keys = ['NEWSAPI_KEY']
            missing_keys = [k for k in required_keys if not os.getenv(k)]

            if missing_keys:
                warning_msg = f"News enabled but missing API keys: {', '.join(missing_keys)}"
                logger.warning(warning_msg)
                st.session_state.news_collector = None
                st.session_state.sentiment_aggregator = None
                st.warning(f"⚠️ {warning_msg}. Add them to .env file.")
            else:
                try:
                    st.session_state.news_collector = NewsCollector(
                        database=st.session_state.db,
                        config=news_config
                    )

                    st.session_state.sentiment_aggregator = SentimentAggregator(
                        database=st.session_state.db,
                        config=news_config.get('features', {})
                    )
                    logger.info("Sentiment aggregator initialized")

                    # Start news collector after both components are initialized
                    st.session_state.news_collector.start()
                    logger.info("News collector started")
                except ValueError as e:
                    error_msg = f"Invalid news configuration: {e}"
                    logger.error(error_msg)
                    st.session_state.news_collector = None
                    st.session_state.sentiment_aggregator = None
                    st.error(f"❌ {error_msg}")
                except Exception as e:
                    error_msg = f"Failed to initialize news components: {e}"
                    logger.error(error_msg)
                    st.session_state.news_collector = None
                    st.session_state.sentiment_aggregator = None
                    st.error(f"❌ {error_msg}")

initialize_components()

# =============================================================================
# CLEANUP HANDLERS
# =============================================================================

def cleanup_resources() -> None:
    """Cleanup background threads and connections on shutdown."""
    try:
        # Check if Streamlit session state is still available
        if not hasattr(st, 'session_state'):
            return

        # Stop news collector if running
        news_collector = st.session_state.get('news_collector')
        if news_collector:
            try:
                if hasattr(news_collector, 'stop'):
                    news_collector.stop()
                    logger.info("News collector stopped")
            except Exception as e:
                logger.warning(f"Error stopping news collector: {e}")

        # Stop data provider if running
        data_provider = st.session_state.get('data_provider')
        if data_provider:
            try:
                if hasattr(data_provider, 'stop'):
                    data_provider.stop()
                    logger.info("Data provider stopped")
            except Exception as e:
                logger.warning(f"Error stopping data provider: {e}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# Register cleanup handler (atexit imported at top)
atexit.register(cleanup_resources)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_engine_running() -> tuple:
    """Check if analysis engine is running."""
    if not PID_FILE.exists():
        return False, None
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)
        return True, pid
    except (ProcessLookupError, ValueError):
        PID_FILE.unlink(missing_ok=True)
        return False, None


def fetch_market_data(symbol: str, timeframe: str = '1h', limit: int = 200) -> dict:
    """Fetch market data from WebSocket provider (real-time only)."""
    provider = st.session_state.data_provider

    if not provider:
        return {'success': False, 'error': 'WebSocket provider not initialized'}

    if not provider.is_running:
        return {'success': False, 'error': 'WebSocket provider not running - click Refresh'}

    try:
        # Get real-time price from tick
        tick = provider.get_tick(symbol)
        price = tick.price if tick else 0

        # Get buffered candles (includes current in-progress candle)
        df = provider.get_candles(symbol, limit=limit)

        # Get provider status for debugging
        status = provider.get_status()
        ticks_received = status.get('ticks_received', 0)

        if df.empty:
            if ticks_received == 0:
                return {'success': False, 'error': f'Connecting to WebSocket... (no data yet)'}
            elif price > 0:
                # Have tick data but no candles yet - create minimal candle from tick
                df = pd.DataFrame([{
                    'timestamp': tick.timestamp,
                    'datetime': datetime.fromtimestamp(tick.timestamp / 1000) if tick.timestamp is not None else datetime.utcnow(),
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': tick.quantity if tick else 0
                }])
            else:
                return {'success': False, 'error': f'WebSocket connected, waiting for data... ({ticks_received} ticks)'}

        # Calculate 24h stats from candles
        if len(df) >= 24:
            last_24h = df.tail(24)
            high_24h = last_24h['high'].max()
            low_24h = last_24h['low'].min()
            volume_24h = last_24h['volume'].sum()
            open_24h = last_24h.iloc[0]['open']
            change_pct = ((price - open_24h) / open_24h * 100) if open_24h > 0 else 0
        else:
            high_24h = df['high'].max()
            low_24h = df['low'].min()
            volume_24h = df['volume'].sum()
            change_pct = 0

        return {
            'success': True,
            'source': 'websocket',
            'df': df,
            'price': price,
            'change_pct': change_pct,
            'high_24h': high_24h,
            'low_24h': low_24h,
            'volume_24h': volume_24h,
        }

    except Exception as e:
        logger.error(f"WebSocket data error: {e}")
        return {'success': False, 'error': str(e)}


def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
    """Calculate RSI."""
    if len(prices) < period:
        return 50.0

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss < 1e-10:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def format_number(value: float, prefix: str = '$') -> str:
    """Format large numbers."""
    if abs(value) >= 1e9:
        return f"{prefix}{value/1e9:.1f}B"
    elif abs(value) >= 1e6:
        return f"{prefix}{value/1e6:.1f}M"
    elif abs(value) >= 1e3:
        return f"{prefix}{value/1e3:.1f}K"
    else:
        return f"{prefix}{value:,.2f}"


# =============================================================================
# LEARNING PAGE HELPERS
# =============================================================================

def _render_db_learning_page():
    """Render learning page from database when bridge isn't available."""
    db_path = ROOT / "data" / "trading.db"
    if not db_path.exists():
        st.error("No database found. Run the trading bot first.")
        return

    conn = sqlite3.connect(str(db_path))

    # --- Overview metrics from DB ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        count = pd.read_sql("SELECT COUNT(*) as c FROM trade_outcomes", conn).iloc[0]['c']
        st.metric("Trade Outcomes", f"{count:,}")

    with col2:
        count = pd.read_sql("SELECT COUNT(*) as c FROM signals", conn).iloc[0]['c']
        st.metric("Signals Generated", f"{count:,}")

    with col3:
        count = pd.read_sql("SELECT COUNT(*) as c FROM retraining_history", conn).iloc[0]['c']
        st.metric("Retraining Runs", f"{count:,}")

    with col4:
        count = pd.read_sql("SELECT COUNT(*) as c FROM candles", conn).iloc[0]['c']
        st.metric("Candles Stored", f"{count:,}")

    st.markdown("---")

    # --- Tabs ---
    tab_outcomes, tab_signals, tab_retraining = st.tabs([
        "Trade Outcomes", "Recent Signals", "Retraining History"
    ])

    with tab_outcomes:
        try:
            df = pd.read_sql("""
                SELECT symbol, predicted_direction, was_correct, confidence,
                       pnl_percent, timestamp
                FROM trade_outcomes
                ORDER BY timestamp DESC
                LIMIT 200
            """, conn)
            if len(df) > 0:
                # Win rate summary
                if 'was_correct' in df.columns:
                    wins = df['was_correct'].sum()
                    total = len(df[df['was_correct'].notna()])
                    win_rate = wins / total * 100 if total > 0 else 0
                    st.metric("Recent Win Rate (last 200)", f"{win_rate:.1f}%")

                # Show table
                st.dataframe(df, use_container_width=True)

                # PnL chart
                if 'pnl_percent' in df.columns and df['pnl_percent'].notna().any():
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=df['pnl_percent'].cumsum(),
                        mode='lines',
                        name='Cumulative PnL %'
                    ))
                    fig.update_layout(title="Cumulative PnL %", height=300)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No trade outcomes recorded yet")
        except Exception as e:
            st.warning(f"Could not load trade outcomes: {e}")

    with tab_signals:
        try:
            df = pd.read_sql("""
                SELECT symbol, signal_type, confidence, entry_price,
                       actual_outcome, timestamp
                FROM signals
                ORDER BY timestamp DESC
                LIMIT 100
            """, conn)
            if len(df) > 0:
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No signals recorded yet")
        except Exception as e:
            st.warning(f"Could not load signals: {e}")

    with tab_retraining:
        try:
            df = pd.read_sql("""
                SELECT symbol, interval, accuracy, confidence, loss,
                       epochs, trigger_reason, timestamp
                FROM retraining_history
                ORDER BY timestamp DESC
                LIMIT 50
            """, conn)
            if len(df) > 0:
                st.dataframe(df, use_container_width=True)

                # Accuracy over time chart
                if 'accuracy' in df.columns and df['accuracy'].notna().any():
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df['accuracy'],
                        mode='lines+markers',
                        name='Accuracy'
                    ))
                    fig.update_layout(title="Retraining Accuracy Over Time", height=300)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No retraining history yet")
        except Exception as e:
            st.warning(f"Could not load retraining history: {e}")

    conn.close()


def _get_learning_systems() -> dict:
    """Safely access all learning subsystems from session state."""
    result = {
        'available': False, 'learning_system': None, 'confidence_gate': None,
        'state_manager': None, 'outcome_tracker': None, 'retraining_engine': None,
        'performance_learner': None, 'db': None
    }
    bridge = st.session_state.get('learning_bridge')
    if not bridge:
        return result
    ls = getattr(bridge, 'learning_system', None)
    if not ls:
        return result
    result['available'] = True
    result['learning_system'] = ls
    result['confidence_gate'] = getattr(ls, 'confidence_gate', None)
    result['state_manager'] = getattr(ls, 'state_manager', None)
    result['outcome_tracker'] = getattr(ls, 'outcome_tracker', None)
    result['retraining_engine'] = getattr(ls, 'retraining_engine', None)
    result['performance_learner'] = (
        getattr(ls, 'performance_learner', None)
        or st.session_state.get('performance_learner')
    )
    result['db'] = st.session_state.get('db')
    return result


def render_learning_overview(systems: dict):
    """Render learning system overview metric cards."""
    try:
        stats = systems['learning_system'].get_stats()
        perf = systems['performance_learner']
        action = perf.get_recommended_action() if perf else 'UNKNOWN'
        action_class = {
            'NORMAL_TRADING': 'positive', 'CAUTIOUS_TRADING': 'warning',
            'REDUCE_POSITION_SIZE': 'negative', 'PAUSE_TRADING': 'negative'
        }.get(action, '')
        action_label = action.replace('_', ' ').title()

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            render_metric_card("Candles Processed", str(stats.get('candles_processed', 0)))
        with c2:
            render_metric_card("Predictions Made", str(stats.get('predictions_made', 0)))
        with c3:
            paper = stats.get('paper_trades', 0)
            live = stats.get('live_trades', 0)
            render_metric_card("Trades Executed", str(stats.get('trades_executed', 0)),
                               delta=f"Paper: {paper} | Live: {live}")
        with c4:
            render_metric_card("Retrainings", str(stats.get('retrainings_triggered', 0)))
        with c5:
            render_metric_card("Online Updates", str(stats.get('online_updates', 0)))
        with c6:
            render_metric_card("Action", action_label, card_class=action_class)
    except Exception as e:
        st.warning(f"Unable to load learning stats: {e}")


def render_mode_status(systems: dict, tracked_symbols: list):
    """Show LEARNING/TRADING mode per symbol."""
    try:
        state_mgr = systems['state_manager']
        gate = systems['confidence_gate']
        timeframe = st.session_state.get('timeframe', '1h')

        if not state_mgr:
            st.info("State manager not available.")
            return

        for symbol in tracked_symbols:
            c1, c2, c3 = st.columns([2, 2, 3])
            try:
                mode = state_mgr.get_current_mode(symbol, timeframe)
            except Exception:
                mode = 'LEARNING'

            mode_color = '#28a745' if mode == 'TRADING' else '#ffc107'
            mode_bg = 'rgba(40,167,69,0.1)' if mode == 'TRADING' else 'rgba(255,193,7,0.1)'

            with c1:
                st.markdown(f"""<div style="padding:12px;border-radius:8px;border-left:4px solid {mode_color};
                    background:{mode_bg};">
                    <div style="font-size:0.85rem;color:#6c757d;">{html.escape(symbol)}</div>
                    <div style="font-size:1.3rem;font-weight:700;color:{mode_color};">{html.escape(str(mode))}</div>
                </div>""", unsafe_allow_html=True)

            with c2:
                try:
                    duration = state_mgr.get_time_in_mode(symbol, timeframe)
                    if duration:
                        hours = int(duration.total_seconds() // 3600)
                        mins = int((duration.total_seconds() % 3600) // 60)
                        render_metric_card("Time in Mode", f"{hours}h {mins}m")
                    else:
                        render_metric_card("Time in Mode", "N/A")
                except Exception:
                    render_metric_card("Time in Mode", "N/A")

            with c3:
                trading_thresh = getattr(gate.config, 'trading_threshold', 0.80) if gate else 0.80
                if gate and hasattr(gate, '_smoothed_confidence'):
                    conf = gate._smoothed_confidence.get((symbol, timeframe))
                    if conf is not None:
                        gap = max(0, trading_thresh - conf)
                        render_metric_card("Confidence", f"{conf:.1%}",
                                           delta=f"Need +{gap:.1%} for TRADING" if mode == 'LEARNING' else "Above threshold",
                                           card_class='positive' if conf >= trading_thresh else '')
                    else:
                        render_metric_card("Confidence", "No data")
                else:
                    render_metric_card("Confidence", "No data")

        with st.expander("Mode Transition History"):
            for symbol in tracked_symbols:
                try:
                    history = state_mgr.get_mode_history(symbol, timeframe, limit=10)
                    if history:
                        st.markdown(f"**{symbol}**")
                        df = pd.DataFrame(history)
                        display_cols = [c for c in ['entered_at', 'mode', 'confidence_score', 'reason'] if c in df.columns]
                        if display_cols:
                            st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
                    else:
                        st.caption(f"{symbol}: No transitions recorded yet.")
                except Exception:
                    st.caption(f"{symbol}: History unavailable.")

    except Exception as e:
        st.warning(f"Unable to load mode status: {e}")


def render_confidence_timeline(systems: dict, tracked_symbols: list):
    """Plotly confidence chart with 80% threshold line."""
    try:
        db = systems['db']
        if not db:
            st.info("Database not available. Start the trading system to enable confidence tracking.")
            return
        gate = systems['confidence_gate']
        timeframe = st.session_state.get('timeframe', '1h')

        c1, c2 = st.columns([2, 1])
        with c1:
            sel_symbol = st.selectbox("Symbol", options=['ALL'] + tracked_symbols, key='conf_symbol')
        with c2:
            days = st.slider("Lookback Days", 1, 30, 7, key='conf_days')

        symbols_to_plot = tracked_symbols if sel_symbol == 'ALL' else [sel_symbol]
        fig = go.Figure()
        has_data = False

        colors = {'BTC/USDT': '#f7931a', 'ETH/USDT': '#627eea'}

        for symbol in symbols_to_plot:
            try:
                df = db.get_confidence_trend(symbol, timeframe, days=days)
                if df is not None and not df.empty:
                    has_data = True
                    color = colors.get(symbol, '#667eea')
                    fig.add_trace(go.Scatter(
                        x=df['timestamp'], y=df['confidence_score'],
                        mode='lines+markers', name=f'{symbol}',
                        line=dict(width=2, color=color),
                        marker=dict(size=4),
                        hovertemplate="<b>%{x}</b><br>Confidence: %{y:.1%}<extra>%{fullData.name}</extra>"
                    ))
            except Exception as e:
                logging.debug(f"Failed to fetch confidence trend for {symbol}: {e}")

        # Threshold lines - read from gate config
        trading_thresh = getattr(gate.config, 'trading_threshold', 0.80) if gate else 0.80
        hysteresis = getattr(gate.config, 'hysteresis', 0.05) if gate else 0.05
        exit_thresh = trading_thresh - hysteresis

        fig.add_hline(y=trading_thresh, line_dash="dash", line_color="#dc3545", line_width=2,
                      annotation_text=f"{trading_thresh:.0%} Trading Threshold", annotation_position="top right",
                      annotation_font_color="#dc3545")
        fig.add_hline(y=exit_thresh, line_dash="dot", line_color="#ffc107", line_width=1,
                      annotation_text=f"{exit_thresh:.0%} Exit Threshold", annotation_position="bottom right",
                      annotation_font_color="#ffc107")

        # Retraining markers
        try:
            for symbol in symbols_to_plot:
                retrain_hist = db.get_retraining_history(symbol, timeframe, limit=20)
                if retrain_hist:
                    for r in retrain_hist[:10]:
                        ts = r.get('triggered_at')
                        if ts:
                            fig.add_vline(x=ts, line_dash="dot", line_color="#667eea", line_width=1)
        except Exception as e:
            logging.debug(f"Failed to fetch retraining markers: {e}")

        fig.update_layout(
            title="Model Confidence Over Time",
            xaxis_title="Time", yaxis_title="Confidence",
            yaxis=dict(range=[0, 1], tickformat='.0%'),
            height=420, template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)
        if not has_data:
            st.info("No confidence data recorded yet. The system needs to run and save confidence scores over time.")

        # Gate stats
        if gate:
            with st.expander("Confidence Gate Statistics"):
                g_stats = gate.get_stats()
                gc1, gc2, gc3, gc4 = st.columns(4)
                with gc1:
                    render_metric_card("To Trading", str(g_stats.get('transitions_to_trading', 0)), card_class='positive')
                with gc2:
                    render_metric_card("To Learning", str(g_stats.get('transitions_to_learning', 0)), card_class='warning')
                with gc3:
                    render_metric_card("Total Checks", str(g_stats.get('total_checks', 0)))
                with gc4:
                    render_metric_card("Regime Adjustments", str(g_stats.get('regime_adjustments', 0)))

    except Exception as e:
        st.warning(f"Unable to load confidence data: {e}")


def render_trade_outcomes_timeline(systems: dict, tracked_symbols: list):
    """Scatter plot of trade outcomes + cumulative P&L."""
    try:
        db = systems['db']
        if not db:
            st.info("Database not available. Start the trading system to enable trade outcome tracking.")
            return
        timeframe = st.session_state.get('timeframe', '1h')

        c1, c2 = st.columns([2, 1])
        with c1:
            sel_symbol = st.selectbox("Symbol", options=['ALL'] + tracked_symbols, key='outcomes_symbol')
        with c2:
            limit = st.slider("Recent Trades", 20, 200, 100, key='outcomes_limit')

        # Fetch outcomes
        all_outcomes = []
        symbols = tracked_symbols if sel_symbol == 'ALL' else [sel_symbol]
        for symbol in symbols:
            try:
                outcomes = db.get_recent_outcomes(symbol, timeframe, limit=limit)
                if outcomes:
                    all_outcomes.extend(outcomes)
            except Exception as e:
                logging.debug(f"Failed to fetch outcomes for {symbol}: {e}")

        # Summary cards
        try:
            perf_stats = db.get_performance_stats()
            sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
            with sc1:
                render_metric_card("Total Trades", str(perf_stats.get('resolved_trades', 0)))
            with sc2:
                wr = perf_stats.get('win_rate', 0)
                render_metric_card("Win Rate", f"{wr:.1%}",
                                   card_class='positive' if wr >= 0.5 else 'negative')
            with sc3:
                render_metric_card("Avg P&L", f"{perf_stats.get('avg_pnl', 0):.2f}%")
            with sc4:
                total_pnl = perf_stats.get('total_pnl', 0)
                render_metric_card("Total P&L", f"{total_pnl:.2f}%",
                                   card_class='positive' if total_pnl >= 0 else 'negative')
            with sc5:
                render_metric_card("Winners", str(perf_stats.get('winners', 0)), card_class='positive')
            with sc6:
                render_metric_card("Losers", str(perf_stats.get('losers', 0)), card_class='negative')
        except Exception as e:
            logging.debug(f"Failed to load performance stats: {e}")

        if not all_outcomes:
            st.info("No trade outcomes recorded yet. The system needs to place and close trades to show results here.")
            return

        outcomes_df = pd.DataFrame(all_outcomes)

        # Build dual chart
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            subplot_titles=('Trade Outcomes (Win/Loss)', 'Cumulative P&L (%)'),
            row_heights=[0.55, 0.45]
        )

        # Top: scatter of individual trades
        if 'pnl_percent' in outcomes_df.columns and 'entry_time' in outcomes_df.columns:
            outcomes_df['pnl_percent'] = pd.to_numeric(outcomes_df['pnl_percent'], errors='coerce').fillna(0)
            was_correct = outcomes_df.get('was_correct', pd.Series([0] * len(outcomes_df)))
            colors = ['#28a745' if w else '#dc3545' for w in was_correct.astype(bool)]
            sizes = [min(max(abs(p) * 3 + 6, 6), 25) for p in outcomes_df['pnl_percent']]

            fig.add_trace(go.Scatter(
                x=outcomes_df['entry_time'], y=outcomes_df['pnl_percent'],
                mode='markers', name='Trades',
                marker=dict(color=colors, size=sizes, line=dict(width=1, color='white'), opacity=0.8),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>P&L: %{y:.2f}%<br>"
                    "Conf: %{customdata[1]:.1%}<br>Strategy: %{customdata[2]}<extra></extra>"
                ),
                customdata=list(zip(
                    outcomes_df.get('predicted_direction', [''] * len(outcomes_df)),
                    pd.to_numeric(outcomes_df.get('predicted_confidence', outcomes_df.get('confidence', 0)), errors='coerce').fillna(0),
                    outcomes_df.get('strategy_name', [''] * len(outcomes_df)).fillna('AI Model')
                ))
            ), row=1, col=1)

            fig.add_hline(y=0, line_dash="solid", line_color="#6c757d", line_width=1, row=1, col=1)

            # Bottom: cumulative P&L
            sorted_df = outcomes_df.sort_values('entry_time')
            cum_pnl = sorted_df['pnl_percent'].cumsum()

            fig.add_trace(go.Scatter(
                x=sorted_df['entry_time'], y=cum_pnl,
                mode='lines', name='Cumulative P&L',
                fill='tozeroy', line=dict(color='#667eea', width=2),
                fillcolor='rgba(102, 126, 234, 0.1)'
            ), row=2, col=1)

        fig.update_layout(
            height=520, template='plotly_white', showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='x unified'
        )
        fig.update_yaxes(title_text="P&L (%)", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative (%)", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # Detail table
        with st.expander("Recent Trades Detail"):
            display_cols = ['entry_time', 'predicted_direction', 'entry_price', 'exit_price',
                            'pnl_percent', 'was_correct', 'predicted_confidence', 'strategy_name', 'closed_by']
            available_cols = [c for c in display_cols if c in outcomes_df.columns]
            if available_cols:
                detail = outcomes_df[available_cols].copy()
                rename_map = {
                    'entry_time': 'Time', 'predicted_direction': 'Dir',
                    'entry_price': 'Entry', 'exit_price': 'Exit',
                    'pnl_percent': 'P&L %', 'was_correct': 'Result',
                    'predicted_confidence': 'Conf', 'strategy_name': 'Strategy',
                    'closed_by': 'Closed By'
                }
                detail.rename(columns={k: v for k, v in rename_map.items() if k in detail.columns}, inplace=True)
                if 'Result' in detail.columns:
                    detail['Result'] = detail['Result'].apply(lambda x: 'WIN' if x else 'LOSS')
                st.dataframe(detail.head(30), use_container_width=True, hide_index=True)

    except Exception as e:
        st.warning(f"Unable to load trade outcomes: {e}")


def render_retraining_history(systems: dict, tracked_symbols: list):
    """Show retraining events with stats."""
    try:
        db = systems['db']
        if not db:
            st.info("Database not available. Start the trading system to enable retraining history.")
            return
        retrain_engine = systems['retraining_engine']
        timeframe = st.session_state.get('timeframe', '1h')

        # Stats cards
        if retrain_engine:
            try:
                r_stats = retrain_engine.get_stats()
                rc1, rc2, rc3, rc4 = st.columns(4)
                with rc1:
                    render_metric_card("Total Retrainings", str(r_stats.get('retrainings_attempted', 0)))
                with rc2:
                    sr = r_stats.get('success_rate', 0)
                    render_metric_card("Success Rate", f"{sr:.0%}",
                                       card_class='positive' if sr >= 0.5 else 'negative')
                with rc3:
                    avg_dur = r_stats.get('avg_duration_seconds', 0)
                    mins = int(avg_dur // 60)
                    secs = int(avg_dur % 60)
                    render_metric_card("Avg Duration", f"{mins}m {secs}s")
                with rc4:
                    ls_stats = systems['learning_system'].get_stats()
                    render_metric_card("Active", str(ls_stats.get('active_retrainings', 0)))
            except Exception:
                pass

        # History table
        all_history = []
        for symbol in tracked_symbols:
            try:
                hist = db.get_retraining_history(symbol, timeframe, limit=20)
                if hist:
                    all_history.extend(hist)
            except Exception:
                pass

        if all_history:
            df = pd.DataFrame(all_history)
            display_cols = [c for c in ['triggered_at', 'symbol', 'trigger_reason', 'status',
                                         'validation_accuracy', 'validation_confidence',
                                         'duration_seconds'] if c in df.columns]
            if display_cols:
                detail = df[display_cols].copy()
                rename_map = {
                    'triggered_at': 'Time', 'symbol': 'Symbol', 'trigger_reason': 'Reason',
                    'status': 'Status', 'validation_accuracy': 'Accuracy',
                    'validation_confidence': 'Confidence', 'duration_seconds': 'Duration (s)'
                }
                detail.rename(columns={k: v for k, v in rename_map.items() if k in detail.columns}, inplace=True)
                if 'Accuracy' in detail.columns:
                    detail['Accuracy'] = pd.to_numeric(detail['Accuracy'], errors='coerce').apply(
                        lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                if 'Confidence' in detail.columns:
                    detail['Confidence'] = pd.to_numeric(detail['Confidence'], errors='coerce').apply(
                        lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                st.dataframe(detail.head(20), use_container_width=True, hide_index=True)
        else:
            st.info("No retraining events recorded yet.")

    except Exception as e:
        st.warning(f"Unable to load retraining history: {e}")


# =============================================================================
# PAGE: LEARNING
# =============================================================================

def page_learning():
    """Learning Progress, Confidence Tracking & Trade Outcomes page."""
    st.title("Learning Progress")
    st.markdown("Real-time visibility into the continuous learning pipeline")

    # Get learning subsystems
    systems = _get_learning_systems()

    if not systems['available']:
        # Fallback: show database-based learning stats when bridge isn't available
        st.info("Reading learning data from database (bot running separately)")
        _render_db_learning_page()
        return

    # --- Section 1: Overview metric cards ---
    render_learning_overview(systems)

    st.markdown("---")

    # --- Section 2: Per-symbol mode status ---
    tracked_symbols = []
    if systems.get('state_manager'):
        try:
            sm = systems['state_manager']
            if hasattr(sm, '_mode_cache'):
                tracked_symbols = list({k[0] for k in sm._mode_cache.keys()})
        except Exception:
            pass
    # Fallback: get symbols from session state
    if not tracked_symbols:
        bridge = st.session_state.get('learning_bridge')
        if bridge and hasattr(bridge, 'symbols'):
            tracked_symbols = list(bridge.symbols)

    render_mode_status(systems, tracked_symbols)

    st.markdown("---")

    # --- Section 3: Tabbed detail views ---
    tab_confidence, tab_outcomes, tab_retraining = st.tabs([
        "Confidence Tracking",
        "Trade Outcomes",
        "Retraining History"
    ])

    with tab_confidence:
        render_confidence_timeline(systems, tracked_symbols)

    with tab_outcomes:
        render_trade_outcomes_timeline(systems, tracked_symbols)

    with tab_retraining:
        render_retraining_history(systems, tracked_symbols)


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_metric_card(label: str, value: str, delta: str = "", card_class: str = ""):
    """Render a metric card with XSS protection."""
    # Escape user-controlled content
    label_escaped = html.escape(label)
    value_escaped = html.escape(value)
    delta_html = f'<div class="delta">{html.escape(delta)}</div>' if delta else ''
    st.markdown(f"""
        <div class="metric-card {card_class}">
            <div class="label">{label_escaped}</div>
            <div class="value">{value_escaped}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)


def render_enhanced_signal_card(prediction, price: float, account_balance: float = 10000):
    """Render comprehensive signal card with all metrics."""
    signal_class = f"signal-{prediction.direction.lower()}"
    color = "#28a745" if prediction.direction == "BUY" else "#dc3545" if prediction.direction == "SELL" else "#6c757d"

    # Calculate dollar amounts for position
    position_size_dollars = account_balance * prediction.kelly_fraction
    shares = position_size_dollars / price if price > 0 else 0
    expected_profit_dollars = shares * abs(prediction.take_profit - price)
    expected_loss_dollars = shares * abs(price - prediction.stop_loss)

    html = f"""
    <div class="signal-card {signal_class}">
        <div style="font-size: 0.9rem; color: #6c757d; margin-bottom: 0.5rem;">🤖 AI SIGNAL</div>
        <div style="font-size: 3rem; font-weight: 800; color: {color};">{prediction.direction}</div>
        <div style="font-size: 1.2rem; margin-top: 0.5rem;">Confidence: {prediction.confidence*100:.1f}%</div>
        <div style="margin-top: 1.5rem; display: flex; justify-content: space-around; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
            <div style="text-align: center;">
                <div style="font-size: 0.75rem; color: #6c757d; font-weight: 600;">STOP LOSS</div>
                <div style="color: #dc3545; font-weight: 700; font-size: 1.1rem;">${prediction.stop_loss:,.2f}</div>
                <div style="font-size: 0.7rem; color: #999;">-{prediction.expected_loss_pct*100:.1f}%</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 0.75rem; color: #6c757d; font-weight: 600;">ENTRY</div>
                <div style="font-weight: 700; font-size: 1.1rem;">${price:,.2f}</div>
                <div style="font-size: 0.7rem; color: #999;">Current</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 0.75rem; color: #6c757d; font-weight: 600;">TAKE PROFIT</div>
                <div style="color: #28a745; font-weight: 700; font-size: 1.1rem;">${prediction.take_profit:,.2f}</div>
                <div style="font-size: 0.7rem; color: #999;">+{prediction.expected_profit_pct*100:.1f}%</div>
            </div>
        </div>
        <div style="margin-top: 1rem; padding: 1rem; background: white; border-radius: 8px; border: 1px solid #e0e0e0;">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                <div><span style="color: #6c757d; font-size: 0.8rem;">Risk/Reward:</span> <span style="font-weight: 700;">1:{prediction.risk_reward_ratio:.2f}</span></div>
                <div><span style="color: #6c757d; font-size: 0.8rem;">Position Size:</span> <span style="font-weight: 700;">{prediction.kelly_fraction*100:.1f}%</span></div>
                <div><span style="color: #6c757d; font-size: 0.8rem;">Expected Profit:</span> <span style="color: #28a745; font-weight: 700;">${expected_profit_dollars:,.0f}</span></div>
                <div><span style="color: #6c757d; font-size: 0.8rem;">Expected Loss:</span> <span style="color: #dc3545; font-weight: 700;">${expected_loss_dollars:,.0f}</span></div>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_monte_carlo_section(prediction, sl_pct: float, tp_pct: float):
    """Render Monte Carlo simulation results with clear advice."""
    prob_profit = prediction.monte_carlo_prob_profit * 100
    prob_tp = prediction.monte_carlo_prob_take_profit * 100
    prob_sl = prediction.monte_carlo_prob_stop_loss * 100
    var_5 = prediction.monte_carlo_var_5pct * 100
    daily_vol = prediction.monte_carlo_volatility_daily * 100
    annual_vol = prediction.monte_carlo_volatility_annual * 100

    # Pre-calculate all colors to avoid nested braces in f-string
    if prob_profit >= 60 and prob_tp > prob_sl:
        risk_level = "LOW RISK"
        risk_color = "#28a745"
        advice = "Good setup! Odds favor profit."
        advice_bg = "#d4edda"
    elif prob_profit >= 50:
        risk_level = "MODERATE"
        risk_color = "#ffc107"
        advice = "Fair odds. Use smaller position."
        advice_bg = "#fff3cd"
    else:
        risk_level = "HIGH RISK"
        risk_color = "#dc3545"
        advice = "Risky trade. Consider waiting."
        advice_bg = "#f8d7da"

    win_color = "#28a745" if prob_profit >= 50 else "#dc3545"

    # Use Streamlit columns for clean layout (avoids HTML rendering issues)
    st.markdown(f"""
<div style="background:white;border-radius:12px;padding:1.5rem;border:1px solid #e0e0e0;">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
<b style="font-size:1.1rem;">Risk Analysis (1 Year, 100K Simulations)</b>
<span style="background:{risk_color};color:white;padding:4px 12px;border-radius:20px;font-weight:700;font-size:0.85rem;">{risk_level}</span>
</div>
</div>
""", unsafe_allow_html=True)

    # Use columns for the metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
<div style="text-align:center;padding:1rem;background:#f8f9fa;border-radius:8px;">
<div style="font-size:0.75rem;color:#6c757d;">Win Chance</div>
<div style="font-size:1.5rem;font-weight:700;color:{win_color};">{prob_profit:.1f}%</div>
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
<div style="text-align:center;padding:1rem;background:#e8f5e9;border-radius:8px;">
<div style="font-size:0.75rem;color:#6c757d;">Hit TP (+{tp_pct:.1f}%)</div>
<div style="font-size:1.5rem;font-weight:700;color:#28a745;">{prob_tp:.1f}%</div>
</div>
""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
<div style="text-align:center;padding:1rem;background:#ffebee;border-radius:8px;">
<div style="font-size:0.75rem;color:#6c757d;">Hit SL (-{sl_pct:.1f}%)</div>
<div style="font-size:1.5rem;font-weight:700;color:#dc3545;">{prob_sl:.1f}%</div>
</div>
""", unsafe_allow_html=True)

    # Bottom metrics row
    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown(f"""
<div style="text-align:center;padding:0.5rem;">
<div style="font-size:0.7rem;color:#999;">Worst Case (5%)</div>
<div style="font-weight:600;color:#dc3545;">-{var_5:.1f}%</div>
</div>
""", unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
<div style="text-align:center;padding:0.5rem;">
<div style="font-size:0.7rem;color:#999;">Daily Swing</div>
<div style="font-weight:600;">{daily_vol:.2f}%</div>
</div>
""", unsafe_allow_html=True)

    with col6:
        st.markdown(f"""
<div style="text-align:center;padding:0.5rem;">
<div style="font-size:0.7rem;color:#999;">Yearly Swing</div>
<div style="font-weight:600;">{annual_vol:.1f}%</div>
</div>
""", unsafe_allow_html=True)

    # Advice box
    st.markdown(f"""
<div style="background:{advice_bg};padding:0.75rem 1rem;border-radius:8px;text-align:center;margin-top:0.5rem;">
<span style="font-weight:600;">{advice}</span>
</div>
""", unsafe_allow_html=True)


def render_action_advice(prediction, current_price: float):
    """Render clear action advice based on all signals."""
    direction = prediction.direction
    confidence = prediction.confidence * 100
    prob_profit = prediction.monte_carlo_prob_profit * 100
    kelly = prediction.kelly_fraction * 100
    rr = prediction.risk_reward_ratio

    # Build action steps based on signal
    if direction == "BUY" and confidence >= 60 and prob_profit >= 50:
        action_color = "#28a745"
        action_title = "Consider Buying"
        steps = [
            f"Entry: Around ${current_price:,.0f}",
            f"Stop Loss: ${prediction.stop_loss:,.0f}",
            f"Take Profit: ${prediction.take_profit:,.0f}",
            f"Position Size: {kelly:.0f}% of account"
        ]
        summary = f"Signals align for a long position with {rr:.1f}x reward potential."
    elif direction == "SELL" and confidence >= 60 and prob_profit >= 50:
        action_color = "#dc3545"
        action_title = "Consider Selling"
        steps = [
            f"Entry: Around ${current_price:,.0f}",
            f"Stop Loss: ${prediction.stop_loss:,.0f}",
            f"Take Profit: ${prediction.take_profit:,.0f}",
            f"Position Size: {kelly:.0f}% of account"
        ]
        summary = f"Signals suggest shorting with {rr:.1f}x reward potential."
    else:
        action_color = "#6c757d"
        action_title = "Wait for Better Setup"
        if confidence < 60:
            reason = f"Low confidence ({confidence:.0f}%)"
        elif prob_profit < 50:
            reason = f"Poor win odds ({prob_profit:.0f}%)"
        else:
            reason = "Mixed signals"
        steps = [
            f"Reason: {reason}",
            "Watch for clearer trend",
            "Check back in 15-30 min"
        ]
        summary = "Current conditions don't favor a trade. Patience pays."

    # Pre-calculate icon to avoid ternary in f-string
    icon = "✓" if "Consider" in action_title else "⏸"
    steps_html = "".join([f'<div style="padding:4px 0;border-bottom:1px solid #eee;">• {step}</div>' for step in steps])

    html = f"""
<div style="background:white;border-radius:12px;padding:1.5rem;border:2px solid {action_color};">
<div style="display:flex;align-items:center;margin-bottom:1rem;">
<div style="width:40px;height:40px;background:{action_color};border-radius:50%;display:flex;align-items:center;justify-content:center;margin-right:12px;">
<span style="color:white;font-size:1.2rem;">{icon}</span>
</div>
<div>
<div style="font-weight:700;font-size:1.1rem;color:{action_color};">{action_title}</div>
<div style="font-size:0.8rem;color:#666;">{summary}</div>
</div>
</div>
<div style="background:#f8f9fa;border-radius:8px;padding:1rem;">
{steps_html}
</div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


def render_rule_validation(prediction):
    """Render 8 out of 10 rule validation status."""
    # Check if prediction has rule validation (for backward compatibility with cached predictions)
    if not hasattr(prediction, 'rules_passed') or not prediction.rules_details:
        st.info("Rule validation will appear after next refresh")
        return

    rules_passed = prediction.rules_passed
    rules_total = prediction.rules_total
    rules_details = prediction.rules_details

    # Determine if signal is valid (8+ rules passed)
    is_valid = rules_passed >= 8
    status_color = "#28a745" if is_valid else "#dc3545" if rules_passed < 6 else "#ffc107"
    status_text = "VALID SIGNAL" if is_valid else "WEAK SIGNAL" if rules_passed >= 6 else "NO TRADE"

    # Header
    st.markdown(f"### Rule Validation ({rules_passed}/{rules_total})")

    # Status badge
    st.markdown(f"""
<div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem;">
<div style="background:{status_color};color:white;padding:8px 20px;border-radius:25px;font-weight:700;font-size:1.1rem;">
{status_text}
</div>
<div style="color:#666;font-size:0.9rem;">
{"Trade when 8+ rules pass" if not is_valid else "All key conditions met"}
</div>
</div>
""", unsafe_allow_html=True)

    # Rules grid - 2 columns of 5 rules each
    col1, col2 = st.columns(2)

    for i, (name, passed, reason) in enumerate(rules_details):
        icon = "✓" if passed else "✗"
        color = "#28a745" if passed else "#dc3545"
        bg = "#e8f5e9" if passed else "#ffebee"

        rule_html = f"""
<div style="display:flex;align-items:center;padding:8px 12px;margin-bottom:6px;background:{bg};border-radius:6px;border-left:4px solid {color};">
<span style="color:{color};font-weight:700;font-size:1.1rem;margin-right:10px;">{icon}</span>
<div>
<div style="font-weight:600;font-size:0.85rem;">{name}</div>
<div style="color:#666;font-size:0.75rem;">{reason}</div>
</div>
</div>
"""
        if i < 5:
            with col1:
                st.markdown(rule_html, unsafe_allow_html=True)
        else:
            with col2:
                st.markdown(rule_html, unsafe_allow_html=True)


def render_prediction_streak(validator, symbol: str, timeframe: str, df):
    """Render prediction streak progress and data requirement."""
    st.markdown("### 🎯 Prediction Validation")

    # Check data requirement
    has_data, data_msg = validator.check_data_requirement(df, timeframe)

    # Get streak status
    can_trade, trade_msg, current_streak = validator.can_trade(symbol, timeframe)

    # Status colors
    if can_trade:
        status_color = "#28a745"
        status_text = "✅ READY TO TRADE"
    elif current_streak >= 6:
        status_color = "#ffc107"
        status_text = "🔄 VALIDATING"
    else:
        status_color = "#dc3545"
        status_text = "❌ NOT READY"

    # Streak progress bar
    progress = (current_streak / validator.streak_required) * 100

    st.markdown(f"""
<div style="background:white;border-radius:12px;padding:1.5rem;border:2px solid {status_color};margin-bottom:1rem;">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
<div>
<div style="font-size:1.2rem;font-weight:700;color:{status_color};">{status_text}</div>
<div style="color:#666;font-size:0.9rem;">{trade_msg}</div>
</div>
<div style="font-size:2rem;font-weight:700;color:{status_color};">{current_streak}/{validator.streak_required}</div>
</div>

<div style="background:#e0e0e0;border-radius:10px;height:20px;overflow:hidden;margin-bottom:0.5rem;">
<div style="background:{status_color};height:100%;width:{progress}%;transition:width 0.3s;"></div>
</div>

<div style="font-size:0.85rem;color:#666;">
<div>📊 Data: {data_msg}</div>
<div>🎲 Model must prove accuracy before trading</div>
</div>
</div>
""", unsafe_allow_html=True)


def render_prediction_history(validator, symbol: str, timeframe: str):
    """Render recent prediction history table with detailed reasoning."""
    history = validator.get_history(symbol, timeframe, limit=10)

    if not history:
        st.info("No prediction history yet. Make your first prediction to start!")
        return

    st.markdown("### 📜 Recent Predictions")

    # Create DataFrame for display
    import pandas as pd
    records = []
    detailed_info = []  # Store detailed info for expanders

    for h in history:
        # Parse market context
        try:
            ctx = json.loads(h['market_context']) if h['market_context'] else {}
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning(f"Failed to parse market context: {e}")
            ctx = {}

        # Format timestamp
        from datetime import datetime
        ts = datetime.fromtimestamp(h['timestamp'] / 1000).strftime("%H:%M")
        full_ts = datetime.fromtimestamp(h['timestamp'] / 1000).strftime("%Y-%m-%d %H:%M:%S")

        # Determine result icon
        if h['is_correct'] is None:
            result = "⏳ Pending"
            result_color = "#999"
        elif h['is_correct'] == 1:
            result = "✅ Win"
            result_color = "#28a745"
        else:
            result = "❌ Loss"
            result_color = "#dc3545"

        records.append({
            "Time": ts,
            "Signal": h['predicted_direction'],
            "Price": f"${h['predicted_price']:,.0f}",
            "Actual": f"${h['actual_price']:,.0f}" if h['actual_price'] else "...",
            "Change": f"{h['profit_loss_pct']:+.2f}%" if h['profit_loss_pct'] else "...",
            "Result": result,
            "Confidence": f"{h['confidence']:.0%}",
            "Rules": f"{h['rules_passed']}/{h['rules_total']}"
        })

        # Store detailed info
        detailed_info.append({
            'timestamp': full_ts,
            'notes': h.get('notes', ''),
            'context': ctx,
            'predicted_direction': h['predicted_direction'],
            'predicted_price': h['predicted_price'],
            'actual_price': h.get('actual_price'),
            'actual_direction': h.get('actual_direction'),
            'confidence': h['confidence'],
            'rules_passed': h['rules_passed'],
            'rules_total': h['rules_total'],
            'is_correct': h['is_correct'],
            'profit_loss_pct': h.get('profit_loss_pct')
        })

    df_hist = pd.DataFrame(records)
    st.dataframe(df_hist, use_container_width=True, hide_index=True)

    # Show stats
    stats = validator.get_stats(symbol, timeframe)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total", stats.get('total_predictions', 0))
    with col2:
        st.metric("Correct", stats.get('correct_predictions', 0))
    with col3:
        st.metric("Accuracy", f"{stats.get('accuracy', 0):.1f}%")
    with col4:
        st.metric("Best Streak", stats.get('best_streak', 0))

    # Expandable detailed analysis section
    st.markdown("---")
    st.markdown("### 🔍 Detailed Analysis (Click to Expand)")

    for i, info in enumerate(detailed_info):
        result_emoji = "⏳" if info['is_correct'] is None else ("✅" if info['is_correct'] == 1 else "❌")
        expander_title = f"{result_emoji} {info['timestamp']} - {info['predicted_direction']} @ ${info['predicted_price']:,.0f}"

        with st.expander(expander_title, expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**📊 Prediction Details**")
                st.write(f"- **Signal:** {info['predicted_direction']}")
                st.write(f"- **Entry Price:** ${info['predicted_price']:,.2f}")
                st.write(f"- **Confidence:** {info['confidence']:.1%}")
                st.write(f"- **Rules Passed:** {info['rules_passed']}/{info['rules_total']}")

                if info['actual_price']:
                    st.markdown("**📈 Outcome**")
                    st.write(f"- **Actual Price:** ${info['actual_price']:,.2f}")
                    st.write(f"- **Change:** {info['profit_loss_pct']:+.2f}%" if info['profit_loss_pct'] else "- **Change:** N/A")
                    st.write(f"- **Result:** {'WIN ✅' if info['is_correct'] == 1 else 'LOSS ❌'}")

            with col2:
                st.markdown("**🌍 Market Context**")
                ctx = info['context']
                if ctx:
                    st.write(f"- **Regime:** {ctx.get('regime', 'N/A')}")
                    st.write(f"- **Trend:** {ctx.get('trend', 'N/A')}")
                    st.write(f"- **Volatility:** {ctx.get('volatility', 'N/A')}")
                    st.write(f"- **Cycle Phase:** {ctx.get('cycle_phase', 'N/A')}")
                    st.write(f"- **Fourier Signal:** {ctx.get('fourier_signal', 'N/A')}")
                    st.write(f"- **Kalman Trend:** {ctx.get('kalman_trend', 'N/A')}")
                else:
                    st.write("No market context available")

            # Show detailed notes if available
            if info['notes']:
                st.markdown("**📝 Analysis Notes**")
                st.code(info['notes'], language=None)
            else:
                # Generate explanation if notes not available
                st.markdown("**📝 Why This Prediction?**")
                ctx = info['context']

                reasons = []
                if info['predicted_direction'] == 'NEUTRAL':
                    reasons.append("🔸 Model was uncertain - confidence below threshold")
                    reasons.append("🔸 Mixed signals from multiple indicators")
                    if ctx.get('regime') in ['VOLATILE', 'CHOPPY']:
                        reasons.append(f"🔸 Market regime was {ctx.get('regime')} - risky to predict")
                elif info['predicted_direction'] == 'BUY':
                    reasons.append("🔸 Bullish signals detected")
                    if ctx.get('trend') == 'UP':
                        reasons.append("🔸 Trend was UP - following the trend")
                    if ctx.get('fourier_signal') == 'BULLISH':
                        reasons.append("🔸 Fourier analysis showed bullish cycle")
                elif info['predicted_direction'] == 'SELL':
                    reasons.append("🔸 Bearish signals detected")
                    if ctx.get('trend') == 'DOWN':
                        reasons.append("🔸 Trend was DOWN - following the trend")

                # Add loss analysis if applicable
                if info['is_correct'] == 0:
                    st.markdown("**❌ Why Loss Occurred:**")
                    loss_reasons = []
                    if info['predicted_direction'] == 'NEUTRAL':
                        loss_reasons.append("• NEUTRAL predictions count as loss if price moved significantly")
                    if ctx.get('regime') in ['VOLATILE', 'CHOPPY']:
                        loss_reasons.append(f"• Market was {ctx.get('regime')} - unpredictable conditions")
                    if info['confidence'] < 0.6:
                        loss_reasons.append(f"• Low confidence ({info['confidence']:.0%}) - model was unsure")
                    if info['rules_passed'] < 8:
                        loss_reasons.append(f"• Only {info['rules_passed']}/10 rules passed - weak signal")
                    if info.get('profit_loss_pct') and abs(info['profit_loss_pct']) < 0.5:
                        loss_reasons.append(f"• Small price change ({info['profit_loss_pct']:+.2f}%) - within noise range")

                    for reason in loss_reasons:
                        st.write(reason)

                for reason in reasons:
                    st.write(reason)


def render_performance_learning_status():
    """Render performance-based learning status and statistics."""
    perf_learner = st.session_state.get('performance_learner')

    if not perf_learner:
        st.info("Performance learner not initialized. Start trading to enable per-candle learning.")
        return

    try:
        stats = perf_learner.get_stats()
        action = perf_learner.get_recommended_action()

        # Status header with recommended action
        action_colors = {
            'NORMAL_TRADING': '🟢',
            'CAUTIOUS_TRADING': '🟡',
            'REDUCE_POSITION_SIZE': '🟠',
            'PAUSE_TRADING': '🔴'
        }
        action_icon = action_colors.get(action, '⚪')

        st.markdown(f"**Recommended Action:** {action_icon} {action.replace('_', ' ').title()}")

        # Main metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            win_rate = stats.get('overall_win_rate', 0)
            recent_rate = stats.get('recent_win_rate', 0)
            delta = f"Recent: {recent_rate:.0%}" if recent_rate else None
            st.metric("Win Rate", f"{win_rate:.1%}", delta=delta)

        with col2:
            total_candles = stats.get('total_candles', 0)
            wins = stats.get('total_wins', 0)
            losses = stats.get('total_losses', 0)
            st.metric("Trades", f"{wins + losses}", delta=f"W:{wins} L:{losses}")

        with col3:
            retrains = stats.get('retrains_triggered', 0)
            last_reason = stats.get('last_retrain_reason', '')
            delta_text = last_reason.replace('_', ' ') if last_reason else None
            st.metric("Retrains", retrains, delta=delta_text)

        with col4:
            reinforcements = stats.get('reinforcements_applied', 0)
            st.metric("Reinforcements", reinforcements)

        # Streak information
        st.markdown("---")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)

        with col_s1:
            consec_wins = stats.get('consecutive_wins', 0)
            max_wins = stats.get('max_consecutive_wins', 0)
            if consec_wins > 0:
                st.success(f"🔥 **{consec_wins}** consecutive wins (max: {max_wins})")
            else:
                st.info(f"Consecutive wins: 0 (max: {max_wins})")

        with col_s2:
            consec_losses = stats.get('consecutive_losses', 0)
            max_losses = stats.get('max_consecutive_losses', 0)
            if consec_losses > 0:
                st.warning(f"⚠️ **{consec_losses}** consecutive losses (max: {max_losses})")
            else:
                st.info(f"Consecutive losses: 0 (max: {max_losses})")

        with col_s3:
            is_retraining = stats.get('is_retraining', False)
            if is_retraining:
                st.warning("🔄 **Retraining in progress...**")
            else:
                st.success("✅ Ready for trading")

        with col_s4:
            last_retrain = stats.get('last_retrain')
            if last_retrain:
                st.info(f"Last retrain: {last_retrain[:16]}")
            else:
                st.info("No retrains yet")

        # Timeframe-specific stats
        tf_stats = stats.get('timeframe_stats', {})
        if tf_stats:
            st.markdown("---")
            st.markdown("**Per-Timeframe Stats:**")
            tf_cols = st.columns(len(tf_stats))
            for idx, (tf, tf_stat) in enumerate(tf_stats.items()):
                with tf_cols[idx]:
                    candles = tf_stat.get('candles', 0)
                    wr = tf_stat.get('win_rate', 0)
                    cl = tf_stat.get('consecutive_losses', 0)
                    st.metric(f"{tf}", f"{wr:.0%} WR", delta=f"{candles} candles")

    except Exception as e:
        st.error(f"Error loading performance stats: {e}")


def render_sentiment_section(prediction):
    """Render news sentiment analysis."""
    if prediction.sentiment_score is None:
        st.info("📰 Sentiment analysis not available (news collector not running)")
        return

    st.markdown("#### 📰 News Sentiment")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        sentiment = prediction.sentiment_score
        if sentiment > 0.2:
            color = "positive"
            emoji = "📈"
            label = "Bullish"
        elif sentiment < -0.2:
            color = "negative"
            emoji = "📉"
            label = "Bearish"
        else:
            color = ""
            emoji = "➖"
            label = "Neutral"

        render_metric_card(
            "Overall Sentiment",
            f"{emoji} {label}",
            f"{sentiment:+.2f}",
            card_class=color
        )

    with col2:
        if prediction.sentiment_1h is not None:
            render_metric_card(
                "1H Sentiment",
                f"{prediction.sentiment_1h:+.2f}"
            )

    with col3:
        if prediction.sentiment_6h is not None:
            render_metric_card(
                "6H Sentiment",
                f"{prediction.sentiment_6h:+.2f}"
            )

    with col4:
        if prediction.news_volume_1h is not None:
            render_metric_card(
                "News Volume (1H)",
                str(prediction.news_volume_1h)
            )

    # Sentiment momentum indicator
    if prediction.sentiment_momentum is not None:
        momentum = prediction.sentiment_momentum
        if abs(momentum) > 0.1:
            trend = "📈 Improving" if momentum > 0 else "📉 Declining"
            st.info(f"**Sentiment Trend:** {trend} ({momentum:+.2f}/hour)")


def render_algorithm_breakdown(prediction):
    """Render detailed algorithm analysis."""
    st.markdown("#### 🧠 Algorithm Analysis")

    # Create tabs for different algorithm categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Ensemble",
        "🌊 Fourier",
        "📈 Kalman",
        "🎰 Markov",
        "💹 Entropy"
    ])

    with tab1:
        st.markdown("**Algorithm Contribution Weights**")
        weights = prediction.ensemble_weights

        # Display as a table
        weight_df = pd.DataFrame([
            {"Algorithm": k.upper(), "Weight": f"{v*100:.1f}%", "Raw": v}
            for k, v in weights.items()
        ]).sort_values("Raw", ascending=False)

        st.dataframe(weight_df[["Algorithm", "Weight"]], use_container_width=True, hide_index=True)

        # Pie chart
        fig = go.Figure(data=[go.Pie(
            labels=[k.upper() for k in weights.keys()],
            values=list(weights.values()),
            hole=0.3
        )])
        fig.update_layout(title="Ensemble Composition", height=300, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Signal", prediction.fourier_signal)
        with col2:
            st.metric("Cycle Phase", f"{prediction.fourier_cycle_phase:.2f}")
        with col3:
            st.metric("Dominant Period", f"{prediction.fourier_dominant_period:.1f}")

        st.info(f"""
        **Interpretation:**
        - Phase **{prediction.fourier_cycle_phase:.2f}** means we are {int(prediction.fourier_cycle_phase*100)}% through the current cycle
        - The dominant cycle period is **{prediction.fourier_dominant_period:.0f}** candles
        - Signal: **{prediction.fourier_signal}** (cycle-based trend prediction)
        """)

    with tab3:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Trend", prediction.kalman_trend)
        with col2:
            st.metric("Smoothed Price", f"${prediction.kalman_smoothed_price:,.2f}")
        with col3:
            st.metric("Velocity", f"{prediction.kalman_velocity:.4f}")

        st.info(f"""
        **Interpretation:**
        - Kalman filter detected **{prediction.kalman_trend}** trend
        - Noise-filtered price estimate: **${prediction.kalman_smoothed_price:,.2f}**
        - Price velocity (momentum): **{prediction.kalman_velocity:.4f}**
        - Estimation uncertainty: **{prediction.kalman_error_covariance:.4f}**
        """)

    with tab4:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current State", prediction.markov_state)
        with col2:
            st.metric("P(Up)", f"{prediction.markov_probability*100:.1f}%")
        with col3:
            st.metric("P(Down)", f"{prediction.markov_prob_down*100:.1f}%")

        st.write("**State Transition Probabilities:**")
        st.write(f"- Probability of UP move: **{prediction.markov_probability*100:.1f}%**")
        st.write(f"- Probability of DOWN move: **{prediction.markov_prob_down*100:.1f}%**")
        st.write(f"- Probability of NEUTRAL move: **{prediction.markov_prob_neutral*100:.1f}%**")

    with tab5:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Regime", prediction.entropy_regime)
        with col2:
            st.metric("Entropy (Normalized)", f"{prediction.entropy_value:.2f}")
        with col3:
            st.metric("Sample Size", prediction.entropy_n_samples)

        regime_descriptions = {
            "TRENDING": "📈 Low entropy - Clear directional movement",
            "NORMAL": "➖ Medium entropy - Balanced market conditions",
            "CHOPPY": "📊 High entropy - Sideways price action",
            "VOLATILE": "⚡ Very high entropy - Unstable conditions"
        }

        st.info(f"""
        **Interpretation:** {regime_descriptions.get(prediction.entropy_regime, 'Unknown regime')}

        - Entropy score: **{prediction.entropy_value:.2f}** (0 = trending, 1 = random)
        - Raw entropy: **{prediction.entropy_raw_value:.2f}**
        - Based on **{prediction.entropy_n_samples}** recent observations
        """)


@st.cache_data(ttl=60)  # Cache for 60 seconds to reduce database queries
def load_trade_outcomes_for_strategies(lookback_days: int = 7):
    """Load recent trade outcomes for strategy analysis."""
    db_path = Path("data/trading.db")
    if not db_path.exists():
        return pd.DataFrame()

    cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()

    query = """
    SELECT
        symbol,
        interval,
        entry_price,
        exit_price,
        entry_time,
        exit_time,
        predicted_direction,
        predicted_confidence,
        was_correct,
        pnl_percent,
        regime
    FROM trade_outcomes
    WHERE entry_time >= ?
    ORDER BY entry_time DESC
    """

    try:
        with sqlite3.connect(str(db_path)) as conn:
            df = pd.read_sql_query(query, conn, params=(cutoff_date,))

            if len(df) > 0:
                df['entry_time'] = pd.to_datetime(df['entry_time'])
                df['exit_time'] = pd.to_datetime(df['exit_time'])
                df['holding_hours'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 3600

            return df
    except Exception as e:
        logger.error(f"Error loading trade outcomes: {e}")
        return pd.DataFrame()


def render_strategy_performance():
    """Render strategy performance section in dashboard."""
    st.markdown("---")
    st.markdown("### 📊 Strategy Performance (Continuous Learning)")

    # Load trade outcomes
    lookback_days = st.slider(
        "Analysis Period (days)",
        min_value=1,
        max_value=30,
        value=7,
        key="strategy_lookback"
    )

    trades_df = load_trade_outcomes_for_strategies(lookback_days=lookback_days)

    if len(trades_df) == 0:
        st.info("📝 No trades yet. Start trading with `python run_trading.py` to see strategy analysis.")
        st.markdown("""
        **How it works:**
        - Every candle close triggers multi-timeframe analysis
        - System discovers strategies from your trades automatically
        - Best strategy is highlighted based on Sharpe ratio

        Run `python scripts/analyze_strategies.py` for detailed analysis.
        """)
        return

    # Use StrategyAnalyzer to discover and calculate strategy metrics
    try:
        analyzer = StrategyAnalyzer("data/trading.db")
        strategies = analyzer.discover_strategies(lookback_days=lookback_days)

        if not strategies:
            st.warning("⚠️ Not enough trades to analyze strategies. Need at least 10 trades total.")
            st.info(f"Total trades: {len(trades_df)}. Keep trading to collect more data!")
            return

        # Convert Strategy objects to DataFrame for display
        strategy_metrics = pd.DataFrame([
            {
                'Strategy': s.name,
                'Trades': s.total_trades,
                'Win Rate': s.win_rate,
                'Avg Profit': s.avg_profit_pct,
                'Avg Loss': s.avg_loss_pct,
                'Profit Factor': s.profit_factor,
                'Sharpe Ratio': s.sharpe_ratio,
                'Total P&L': s.total_trades * (s.avg_profit_pct * s.win_rate - s.avg_loss_pct * (1 - s.win_rate))
            }
            for s in strategies.values()
        ])
        strategy_metrics = strategy_metrics.sort_values('Sharpe Ratio', ascending=False)
    except Exception as e:
        logger.error(f"Error analyzing strategies: {e}")
        st.error(f"Error analyzing strategies: {e}")
        return

    if len(strategy_metrics) == 0:
        st.warning("⚠️ Not enough trades to analyze strategies. Need at least 3 trades per strategy type.")
        st.info(f"Total trades: {len(trades_df)}. Keep trading to collect more data!")
        return

    # Display best strategy
    best_strategy = strategy_metrics.iloc[0]

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%); padding: 20px; border-radius: 10px; color: white; margin: 20px 0;">
        <h3 style="margin: 0;">🏆 BEST STRATEGY: {best_strategy['Strategy']}</h3>
        <p style="font-size: 18px; margin-top: 10px; margin-bottom: 0;">
            Win Rate: <strong>{best_strategy['Win Rate']*100:.1f}%</strong> |
            Sharpe Ratio: <strong>{best_strategy['Sharpe Ratio']:.2f}</strong> |
            Profit Factor: <strong>{best_strategy['Profit Factor']:.2f}x</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Overall performance
    col1, col2, col3, col4 = st.columns(4)

    total_trades = len(trades_df)
    wins = trades_df['was_correct'].sum()
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    total_pnl = trades_df['pnl_percent'].sum()

    with col1:
        st.metric("Total Trades", total_trades)

    with col2:
        st.metric("Overall Win Rate", f"{win_rate*100:.1f}%", delta=f"{wins} wins")

    with col3:
        pnl_color = "normal" if total_pnl > 0 else "inverse"
        st.metric("Total P&L", f"{total_pnl:.2f}%", delta_color=pnl_color)

    with col4:
        st.metric("Strategies Found", len(strategy_metrics))

    # Strategy comparison table
    st.markdown("#### All Strategies Comparison")

    # Format the dataframe for display
    display_df = strategy_metrics.copy()
    display_df['Win Rate'] = display_df['Win Rate'].apply(lambda x: f"{x*100:.1f}%")
    display_df['Avg Profit'] = display_df['Avg Profit'].apply(lambda x: f"+{x:.2f}%")
    display_df['Avg Loss'] = display_df['Avg Loss'].apply(lambda x: f"-{x:.2f}%")
    display_df['Profit Factor'] = display_df['Profit Factor'].apply(lambda x: f"{x:.2f}x" if x != float('inf') else "∞")
    display_df['Sharpe Ratio'] = display_df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
    display_df['Total P&L'] = display_df['Total P&L'].apply(lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Info box
    with st.expander("ℹ️ How Strategy Discovery Works"):
        st.markdown("""
        **Automatic Strategy Classification:**
        - **Scalping:** < 1 hour hold time
        - **Momentum Breakout:** High confidence (>85%), 1-4 hour hold
        - **Swing Trend Following:** 4-24 hours in trending markets
        - **Swing Mean Reversion:** 4-24 hours in choppy markets
        - **Position Trading:** > 24 hours hold
        - **Volatility Expansion:** Trades in volatile markets
        - **Range Trading:** Trades in sideways markets
        - **Trend Following:** Follows established trends

        **Performance Metrics:**
        - **Win Rate:** Percentage of profitable trades
        - **Profit Factor:** Total profit / Total loss (>1 = profitable)
        - **Sharpe Ratio:** Risk-adjusted returns (>1 = good, >2 = excellent)
        - **Total P&L:** Sum of all profit/loss percentages

        **Best Strategy** is determined by highest Sharpe ratio (best risk-adjusted returns).

        For detailed analysis, run: `python scripts/analyze_strategies.py`
        """)


def render_realtime_chart(symbol: str, timeframe: str, historical_data: list = None):
    """
    Render real-time candlestick chart using Lightweight Charts.
    Direct WebSocket connection - updates tick-by-tick without page refresh.
    """
    # Convert symbol to Binance format
    ws_symbol = symbol.replace("/", "").lower()

    # Convert historical data to JSON for initial load (VECTORIZED - 100x faster than iterrows)
    historical_json = "[]"
    has_data = historical_data is not None and not historical_data.empty
    if has_data:
        # Vectorized timestamp conversion
        timestamps = historical_data['timestamp'].values
        ts_converted = [int(ts / 1000) if ts > 1e12 else int(ts) for ts in timestamps]

        # List comprehension with zip (100-500x faster than iterrows)
        hist_list = [
            {
                "time": ts,
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c)
            }
            for ts, o, h, l, c in zip(
                ts_converted,
                historical_data['open'].values,
                historical_data['high'].values,
                historical_data['low'].values,
                historical_data['close'].values
            )
        ]
        historical_json = json.dumps(hist_list)

    # Volume data (VECTORIZED - 100x faster than iterrows)
    volume_json = "[]"
    if has_data:
        # Vectorized timestamp conversion
        timestamps = historical_data['timestamp'].values
        ts_converted = [int(ts / 1000) if ts > 1e12 else int(ts) for ts in timestamps]

        # Vectorized color calculation
        closes = historical_data['close'].values
        opens = historical_data['open'].values
        colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(closes, opens)]

        # List comprehension with zip
        vol_list = [
            {
                "time": ts,
                "value": float(v),
                "color": color
            }
            for ts, v, color in zip(
                ts_converted,
                historical_data['volume'].values,
                colors
            )
        ]
        volume_json = json.dumps(vol_list)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #fafafa; }}
            #chart-container {{ width: 100%; height: 420px; position: relative; }}
            #price-display {{
                position: absolute; top: 10px; left: 10px; z-index: 100;
                background: rgba(255,255,255,0.95); padding: 8px 12px; border-radius: 6px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1); font-size: 13px;
            }}
            #price-display .symbol {{ font-weight: 600; color: #333; }}
            #price-display .price {{ font-size: 18px; font-weight: 700; margin: 2px 0; }}
            #price-display .change {{ font-size: 12px; }}
            #price-display .status {{ font-size: 10px; color: #26a69a; margin-top: 4px; }}
            .up {{ color: #26a69a; }}
            .down {{ color: #ef5350; }}
        </style>
    </head>
    <body>
        <div id="chart-container">
            <div id="price-display">
                <div class="symbol">{symbol}</div>
                <div class="price" id="current-price">--</div>
                <div class="change" id="price-change">--</div>
                <div class="status" id="ws-status">● Connecting...</div>
            </div>
        </div>
        <script>
            const container = document.getElementById('chart-container');
            const chart = LightweightCharts.createChart(container, {{
                width: container.clientWidth,
                height: 420,
                layout: {{
                    background: {{ type: 'solid', color: '#fafafa' }},
                    textColor: '#333',
                }},
                grid: {{
                    vertLines: {{ color: '#f0f0f0' }},
                    horzLines: {{ color: '#f0f0f0' }},
                }},
                crosshair: {{
                    mode: LightweightCharts.CrosshairMode.Normal,
                }},
                rightPriceScale: {{
                    borderColor: '#e0e0e0',
                }},
                timeScale: {{
                    borderColor: '#e0e0e0',
                    timeVisible: true,
                    secondsVisible: false,
                }},
            }});

            // Candlestick series
            const candleSeries = chart.addCandlestickSeries({{
                upColor: '#26a69a',
                downColor: '#ef5350',
                borderDownColor: '#ef5350',
                borderUpColor: '#26a69a',
                wickDownColor: '#ef5350',
                wickUpColor: '#26a69a',
            }});

            // Volume series - separate scale at bottom
            const volumeSeries = chart.addHistogramSeries({{
                priceFormat: {{ type: 'volume' }},
                priceScaleId: 'volume',
            }});

            // Configure volume scale to be at bottom 20% of chart
            chart.priceScale('volume').applyOptions({{
                scaleMargins: {{ top: 0.8, bottom: 0 }},
            }});

            // Configure main price scale to leave room for volume
            chart.priceScale('right').applyOptions({{
                scaleMargins: {{ top: 0.05, bottom: 0.25 }},
            }});

            // Load historical data
            const historicalData = {historical_json};
            const volumeData = {volume_json};

            if (historicalData.length > 0) {{
                candleSeries.setData(historicalData);
                volumeSeries.setData(volumeData);
            }}

            // Current candle tracking
            let currentCandle = null;
            let openPrice = historicalData.length > 0 ? historicalData[0].close : 0;

            // Update price display
            function updatePriceDisplay(price, prevClose) {{
                const priceEl = document.getElementById('current-price');
                const changeEl = document.getElementById('price-change');

                priceEl.textContent = '$' + price.toLocaleString('en-US', {{minimumFractionDigits: 2, maximumFractionDigits: 2}});

                if (prevClose > 0) {{
                    const change = ((price - prevClose) / prevClose) * 100;
                    const changeStr = (change >= 0 ? '+' : '') + change.toFixed(2) + '%';
                    changeEl.textContent = changeStr;
                    changeEl.className = 'change ' + (change >= 0 ? 'up' : 'down');
                    priceEl.className = 'price ' + (change >= 0 ? 'up' : 'down');
                }}
            }}

            // WebSocket connection to Binance
            const wsUrl = 'wss://stream.binance.com:9443/ws/{ws_symbol}@kline_{timeframe}';
            let ws = null;
            let reconnectAttempts = 0;

            function connect() {{
                ws = new WebSocket(wsUrl);

                ws.onopen = () => {{
                    document.getElementById('ws-status').textContent = '● Live';
                    document.getElementById('ws-status').style.color = '#26a69a';
                    reconnectAttempts = 0;
                }};

                ws.onmessage = (event) => {{
                    const data = JSON.parse(event.data);
                    const kline = data.k;

                    const candle = {{
                        time: Math.floor(kline.t / 1000),
                        open: parseFloat(kline.o),
                        high: parseFloat(kline.h),
                        low: parseFloat(kline.l),
                        close: parseFloat(kline.c),
                    }};

                    const volume = {{
                        time: Math.floor(kline.t / 1000),
                        value: parseFloat(kline.v),
                        color: candle.close >= candle.open ? '#26a69a80' : '#ef535080',
                    }};

                    // Update chart
                    candleSeries.update(candle);
                    volumeSeries.update(volume);

                    // Update price display
                    updatePriceDisplay(candle.close, openPrice);

                    // Store current candle
                    currentCandle = candle;
                }};

                ws.onclose = () => {{
                    document.getElementById('ws-status').textContent = '● Reconnecting...';
                    document.getElementById('ws-status').style.color = '#ffa500';

                    if (reconnectAttempts < 10) {{
                        reconnectAttempts++;
                        setTimeout(connect, 2000);
                    }}
                }};

                ws.onerror = (error) => {{
                    console.error('WebSocket error:', error);
                    document.getElementById('ws-status').textContent = '● Error';
                    document.getElementById('ws-status').style.color = '#ef5350';
                }};
            }}

            // Start connection
            connect();

            // Resize handler
            window.addEventListener('resize', () => {{
                chart.applyOptions({{ width: container.clientWidth }});
            }});

            // Fit content
            chart.timeScale().fitContent();
        </script>
    </body>
    </html>
    """

    return html_content


# =============================================================================
# PAGE: DASHBOARD
# =============================================================================

def page_dashboard():
    """Main dashboard page."""
    # Calculate required candles for 1 year based on timeframe
    # Only 4 supported timeframes: 15m, 1h, 4h, 1d
    timeframe_to_year_candles = {
        '15m': 35040,  # 4 * 24 * 365
        '1h': 8760,    # 24 * 365
        '4h': 2190,    # 6 * 365
        '1d': 365,     # 365
    }
    # Get required candles for selected timeframe (default to 1h requirement)
    required_candles = timeframe_to_year_candles.get(st.session_state.timeframe, 8760)

    # Fetch market data with enough candles for 1 year
    # Use required amount (buffer size now supports 10,000 candles)
    fetch_limit = required_candles

    data = fetch_market_data(
        st.session_state.symbol,
        st.session_state.timeframe,
        limit=fetch_limit
    )

    if not data['success']:
        error_msg = data.get('error', 'Unknown error')
        # Show info message instead of error (WebSocket is loading data)
        if 'waiting for data' in error_msg.lower() or 'connecting' in error_msg.lower():
            st.info(f"🔄 Loading market data... {error_msg}")
            st.info("💡 The WebSocket is connected and receiving data. The dashboard will update automatically once candles are available.")
        else:
            st.warning(f"⚠️ {error_msg}")
        return

    df = data['df']
    price = data['price']
    change_pct = data['change_pct']

    # Header metrics
    st.markdown("### Market Overview")

    cols = st.columns(6)

    with cols[0]:
        color_class = "positive" if change_pct >= 0 else "negative"
        arrow = "▲" if change_pct >= 0 else "▼"
        render_metric_card(
            st.session_state.symbol,
            f"${price:,.2f}",
            f"{arrow} {abs(change_pct):.2f}%",
            color_class
        )

    with cols[1]:
        render_metric_card("24H High", f"${data['high_24h']:,.2f}")

    with cols[2]:
        render_metric_card("24H Low", f"${data['low_24h']:,.2f}")

    with cols[3]:
        render_metric_card("24H Volume", format_number(data['volume_24h']))

    with cols[4]:
        rsi = calculate_rsi(df['close'].values)
        rsi_class = "negative" if rsi > 70 else "positive" if rsi < 30 else ""
        render_metric_card("RSI (14)", f"{rsi:.1f}", "", rsi_class)

    with cols[5]:
        # Show data source and connection status
        provider = st.session_state.data_provider
        if provider and provider.is_connected:
            source = "WebSocket"
            source_class = "positive"
        elif data.get('source') == 'websocket':
            source = "WS Buffered"
            source_class = "warning"
        elif data.get('source') == 'rest':
            source = "REST"
            source_class = "warning"
        else:
            source = "Offline"
            source_class = "negative"
        render_metric_card("Data Source", source, "Real-time" if source == "WebSocket" else "", source_class)

    # Data Loading Status Indicator
    candles_loaded = len(df)
    data_progress = min((candles_loaded / required_candles) * 100, 100)

    # Check if fetch is actively in progress
    provider = st.session_state.data_provider
    key = f"{st.session_state.symbol}_{st.session_state.timeframe}"
    # Safe check - method may not exist if provider is cached from old version
    is_fetching = False
    if provider and hasattr(provider, 'is_fetching'):
        is_fetching = provider.is_fetching(key)

    # Set flag for smart auto-refresh
    st.session_state.data_loading = (candles_loaded < required_candles)

    if candles_loaded < required_candles:
        # Show different message based on fetch status
        if is_fetching:
            fetch_status = "🔄 **FETCHING IN PROGRESS** - Please wait..."
            fetch_note = "The fetch thread is running. Page will auto-refresh when complete."
        else:
            fetch_status = "⏳ Waiting for data..."
            fetch_note = "Data fetch may be starting or completing soon."

        st.info(f"""
        📊 **Historical Data Loading**: {candles_loaded:,} / {required_candles:,} candles ({data_progress:.1f}%)

        {fetch_status}

        💡 **Why?** Binance limits each request to 1,000 candles. Loading {required_candles:,} candles requires {required_candles // 1000 + 1} sequential API calls (~30-60 seconds).

        ✅ {fetch_note}
        """)

        # Progress bar
        st.progress(data_progress / 100)
    else:
        st.success(f"✅ Full historical data loaded: {candles_loaded:,} candles ({candles_loaded / 365:.1f} years for {st.session_state.timeframe} timeframe)")

    st.markdown("---")

    # Full-width chart at top
    chart_html = render_realtime_chart(
        st.session_state.symbol,
        st.session_state.timeframe,
        df
    )
    st.components.v1.html(chart_html, height=450)

    st.markdown("---")

    # AI Prediction and Monte Carlo side by side
    if AI_AVAILABLE and len(df) >= 50:
        # MANDATORY: Check if model is validated before predictions
        if not st.session_state.model_ready and not st.session_state.training_in_progress:
            # Model not ready - show training UI
            st.warning(f"### Model Training Required")

            config = load_config()
            min_accuracy = config.get('auto_training', {}).get('min_accuracy_required', 0.58)

            col1, col2 = st.columns([2, 1])

            with col1:
                if st.session_state.model_accuracy > 0:
                    st.error(
                        f"**Model accuracy is below minimum requirement**\n\n"
                        f"- Current accuracy: **{st.session_state.model_accuracy:.1%}**\n"
                        f"- Required accuracy: **{min_accuracy:.1%}**\n\n"
                        f"Predictions are blocked until the model meets the accuracy threshold."
                    )
                else:
                    st.info(
                        f"**No trained model found for {st.session_state.symbol} @ {st.session_state.timeframe}**\n\n"
                        f"A model must be trained and validated before predictions can be made.\n\n"
                        f"**Training requirements:**\n"
                        f"- Minimum accuracy: {min_accuracy:.1%}\n"
                        f"- Training data: 5000 candles\n"
                        f"- Expected duration: 5-10 minutes"
                    )

            with col2:
                # Check if database has enough data
                if st.session_state.db:
                    try:
                        df_check = st.session_state.db.get_candles(
                            st.session_state.symbol,
                            st.session_state.timeframe,
                            limit=1000
                        )
                        candles_in_db = len(df_check) if df_check is not None else 0
                    except Exception as e:
                        logger.debug(f"Failed to check candles in DB: {e}")
                        candles_in_db = 0
                else:
                    candles_in_db = 0

                if candles_in_db < 1000:
                    # Not enough data - show fetch button
                    st.error(f"**Insufficient data in database**\n\n{candles_in_db} / 1000 candles")

                    if st.button("📥 Fetch Historical Data", type="primary", use_container_width=True):
                        with st.spinner("Fetching historical data from Binance..."):
                            try:
                                provider = st.session_state.data_provider
                                if provider and st.session_state.db:
                                    # Ensure database is connected to provider
                                    provider.set_database(st.session_state.db)

                                    # Fetch historical data directly
                                    logger.info(f"Fetching historical data for {st.session_state.symbol} @ {st.session_state.timeframe}")

                                    # Calculate how many candles for 1 year
                                    timeframe_to_year = {
                                        '15m': 35040, '1h': 8760, '4h': 2190, '1d': 365
                                    }
                                    limit = timeframe_to_year.get(st.session_state.timeframe, 8760)

                                    # Fetch and save
                                    candles = provider.fetch_historical(
                                        st.session_state.symbol,
                                        st.session_state.timeframe,
                                        limit=limit
                                    )

                                    if candles and len(candles) > 0:
                                        # Data is automatically saved by fetch_historical when database is connected
                                        # Verify it was saved
                                        time.sleep(1)  # Wait for async save

                                        df_verify = st.session_state.db.get_candles(
                                            st.session_state.symbol,
                                            st.session_state.timeframe,
                                            limit=100
                                        )

                                        if df_verify is not None and len(df_verify) > 0:
                                            st.success(f"✓ Fetched and saved {len(candles)} candles to database!")
                                            time.sleep(1)
                                            st.rerun()
                                        else:
                                            st.error("Data was fetched but not saved to database - check logs")
                                    else:
                                        st.error("Failed to fetch data from Binance")
                                else:
                                    st.error("Provider or database not initialized")
                            except Exception as e:
                                st.error(f"Failed to fetch data: {e}")
                                logger.error(f"Historical data fetch error: {e}", exc_info=True)
                else:
                    # Enough data - show train button
                    st.success(f"✓ {candles_in_db:,} candles in database")

                    if st.button("Train Model Now", type="primary", use_container_width=True):
                        st.session_state.training_in_progress = True
                        st.rerun()

                if st.session_state.model_trained_at:
                    st.caption(f"Last trained: {st.session_state.model_trained_at}")

        elif st.session_state.training_in_progress:
            # Training in progress - show progress
            st.info("### Model Training in Progress")

            progress_placeholder = st.empty()
            status_placeholder = st.empty()

            def progress_callback(epoch, max_epochs, accuracy, status):
                progress = epoch / max_epochs
                progress_placeholder.progress(progress, text=f"Epoch {epoch}/{max_epochs}")
                status_placeholder.text(status)
                # Update session state
                st.session_state.training_epoch = epoch
                st.session_state.training_progress = progress
                st.session_state.training_status = status

            try:
                # Train the model
                result = train_model(
                    st.session_state.symbol,
                    st.session_state.timeframe,
                    progress_callback=progress_callback
                )

                # Training completed successfully
                st.session_state.training_in_progress = False
                st.session_state.model_ready = True
                st.session_state.model_accuracy = result['accuracy']

                # Update model status
                model_status = check_model_readiness(st.session_state.symbol, st.session_state.timeframe)
                st.session_state.model_trained_at = model_status['trained_at']

                st.success(
                    f"Model trained successfully!\n\n"
                    f"- Accuracy: {result['accuracy']:.2%}\n"
                    f"- Training samples: {result['samples']:,}\n"
                    f"- Epochs: {result['epochs']}\n\n"
                    f"Predictions are now enabled."
                )

                time.sleep(2)
                st.rerun()

            except Exception as e:
                st.session_state.training_in_progress = False
                st.error(f"Training failed: {e}")
                logger.error(f"Model training error: {e}", exc_info=True)

        else:
            # Model is ready - make predictions
            try:
                predictor = st.session_state.advanced_predictor
                if predictor:
                    # Calculate ATR
                    high_low = df['high'] - df['low']
                    atr = float(high_low.rolling(14).mean().iloc[-1]) if len(high_low) >= 14 else price * 0.02

                    # Calculate SL/TP percentages (same formula as advanced_predictor.py)
                    sl_pct = min(5.0, max(1.0, atr / price * 200))  # As percentage
                    tp_pct = min(10.0, max(1.5, atr / price * 300))  # As percentage

                    # Check if we can use cached prediction (smooth updates)
                    current_time = time.time()
                    # Check cache validity (also invalidate if missing new rule fields)
                    cached_pred = st.session_state.last_prediction
                    has_rules = cached_pred is not None and hasattr(cached_pred, 'rules_passed')
                    cache_valid = (
                        cached_pred is not None and
                        has_rules and
                        (current_time - st.session_state.last_prediction_time) < st.session_state.prediction_cache_seconds
                    )

                    if cache_valid:
                        # Use cached prediction
                        prediction = cached_pred
                    else:
                        # Fetch sentiment features
                        sentiment_features = None
                        if st.session_state.db and st.session_state.sentiment_aggregator:
                            try:
                                latest_timestamp = int(df.iloc[-1]['timestamp'])
                                sentiment_features = st.session_state.db.get_sentiment_features(latest_timestamp)
                                if sentiment_features:
                                    logger.info(f"Sentiment features loaded for AI prediction")
                            except Exception as e:
                                logger.warning(f"Could not fetch sentiment: {e}")
                                sentiment_features = None

                        # Get LSTM probability from trained model
                        lstm_probability = 0.5  # Default neutral
                        try:
                            import torch
                            from pathlib import Path
                            from src.analysis_engine import FeatureCalculator

                            # Load trained model
                            model_name = f"model_{st.session_state.symbol.replace('/', '_')}_{st.session_state.timeframe}.pt"
                            model_path = Path("models") / model_name

                            if model_path.exists():
                                # Load checkpoint
                                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

                                # Calculate features
                                df_features = FeatureCalculator.calculate_all(df.copy())
                                feature_columns = FeatureCalculator.get_feature_columns()

                                # Get features for last sequence
                                features = df_features[feature_columns].values
                                feature_means = checkpoint['feature_means']
                                feature_stds = checkpoint['feature_stds']

                                # Normalize
                                features = (features - feature_means) / (feature_stds + 1e-8)
                                features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

                                # Get sequence length from model config (with fallback for old models)
                                if 'model_config' in checkpoint:
                                    sequence_length = checkpoint['model_config']['sequence_length']
                                    model_config = checkpoint['model_config']
                                else:
                                    # Old model format - use defaults
                                    logger.warning("Old model format detected, using default config")
                                    sequence_length = 60
                                    model_config = {
                                        'input_size': 30,
                                        'hidden_size': 128,
                                        'num_layers': 2,
                                        'dropout': 0.2
                                    }

                                # Create sequence (last N candles)
                                if len(features) >= sequence_length:
                                    X = features[-sequence_length:]  # Shape: (sequence_length, num_features)
                                    X = torch.FloatTensor(X).unsqueeze(0)  # Shape: (1, sequence_length, num_features)

                                    # Load model architecture
                                    from src.analysis_engine import LSTMModel
                                    model = LSTMModel(
                                        input_size=model_config['input_size'],
                                        hidden_size=model_config['hidden_size'],
                                        num_layers=model_config['num_layers'],
                                        dropout=model_config['dropout']
                                    )
                                    model.load_state_dict(checkpoint['model_state_dict'])
                                    model.eval()

                                    # Make prediction
                                    with torch.no_grad():
                                        output = model(X)
                                        lstm_probability = float(output[0][0])

                                    logger.info(f"LSTM prediction: {lstm_probability:.3f} (from trained model)")
                                else:
                                    logger.warning(f"Insufficient data for LSTM: {len(features)} < {sequence_length}")
                            else:
                                logger.warning(f"Model not found: {model_path}")

                        except Exception as e:
                            logger.error(f"LSTM prediction failed: {e}", exc_info=True)
                            lstm_probability = 0.5  # Fallback to neutral

                        # Make prediction with all available data
                        prediction = predictor.predict(
                            df=df,
                            symbol=st.session_state.symbol,
                            timeframe=st.session_state.timeframe,
                            lstm_probability=lstm_probability,  # Using trained LSTM model
                            atr=atr,
                            sentiment_features=sentiment_features
                        )

                        # Cache the prediction
                        st.session_state.last_prediction = prediction
                        st.session_state.last_prediction_time = current_time

                    # Get account balance for position sizing
                    paper_trader = st.session_state.paper_trader
                    account_balance = paper_trader.get_account_value() if paper_trader else 10000

                    # ============================================================
                    # CLEAN LAYOUT: Show only what matters for trading decision
                    # ============================================================

                    # Main Trading Signal - FULL WIDTH, BIG & CLEAR
                    st.markdown("## 🎯 Trading Signal")
                    render_enhanced_signal_card(prediction, price, account_balance)

                    st.markdown("")  # Spacing

                    # Risk Analysis - Directly below signal
                    render_monte_carlo_section(prediction, sl_pct, tp_pct)

                    st.markdown("")  # Spacing

                    # Quick Recommendation
                    render_action_advice(prediction, price)

                    # ============================================================
                    # EVERYTHING ELSE: Collapsed by default
                    # ============================================================

                    st.markdown("---")

                    # Model Info - Collapsed
                    if st.session_state.model_accuracy > 0 or st.session_state.model_trained_at:
                        with st.expander("🤖 Model Information"):
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric(
                                    "Model Accuracy",
                                    f"{st.session_state.model_accuracy:.2%}" if st.session_state.model_accuracy > 0 else "N/A",
                                    delta="Validated ✓" if st.session_state.model_ready else "Not Ready"
                                )

                            with col2:
                                if st.session_state.model_trained_at:
                                    try:
                                        trained_time = datetime.fromisoformat(st.session_state.model_trained_at)
                                        time_ago = datetime.utcnow() - trained_time

                                        if time_ago.days > 0:
                                            time_str = f"{time_ago.days}d ago"
                                        elif time_ago.seconds > 3600:
                                            time_str = f"{time_ago.seconds // 3600}h ago"
                                        else:
                                            time_str = f"{time_ago.seconds // 60}m ago"

                                        st.metric("Last Trained", time_str)
                                    except Exception as e:
                                        logger.debug(f"Failed to parse last trained time: {e}")
                                        st.metric("Last Trained", "Unknown")
                                else:
                                    st.metric("Last Trained", "Never")

                            with col3:
                                st.metric(
                                    "Symbol/Timeframe",
                                    f"{st.session_state.symbol}",
                                    delta=st.session_state.timeframe
                                )

                    # Rule Validation - Collapsed
                    with st.expander("📋 Rule Validation (9/10)", expanded=False):
                        render_rule_validation(prediction)

                    # Prediction Validation - Collapsed
                    if st.session_state.prediction_validator:
                        with st.expander("🎯 Prediction Validation History", expanded=False):
                            render_prediction_streak(
                                st.session_state.prediction_validator,
                                st.session_state.symbol,
                                st.session_state.timeframe,
                                df
                            )

                            render_prediction_history(
                                st.session_state.prediction_validator,
                                st.session_state.symbol,
                                st.session_state.timeframe
                            )

                    # Performance Learning Status - Collapsed
                    if st.session_state.get('performance_learner'):
                        with st.expander("📊 Performance Learning Status", expanded=False):
                            render_performance_learning_status()

                    # Sentiment - Collapsed (only if available)
                    if prediction.sentiment_score is not None:
                        with st.expander("📰 News Sentiment", expanded=False):
                            render_sentiment_section(prediction)

                    # Algorithm Details - Collapsed
                    with st.expander("🧠 Algorithm Breakdown (Advanced)", expanded=False):
                        render_algorithm_breakdown(prediction)

            except Exception as e:
                logger.error(f"Prediction error: {e}")
                st.error(f"Prediction unavailable: {e}")

                # Show debug info
                if st.checkbox("Show debug info"):
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.info(f"AI predictions require 50+ candles of data. Currently: {len(df)}")

    # Strategy Performance Section
    render_strategy_performance()


# =============================================================================
# PAGE: FOREX MARKETS
# =============================================================================

def page_forex_markets():
    """Forex markets monitoring page."""
    st.markdown("### Forex Markets")

    # Try to import forex module
    try:
        from src.portfolio.forex import (
            FOREX_PAIRS, MAJOR_PAIRS, CROSS_PAIRS,
            get_current_session
        )
        forex_available = True
    except ImportError as e:
        st.error(f"Forex module not available: {e}")
        forex_available = False
        return

    # Load config
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        forex_config = config.get('forex', {})
        forex_enabled = forex_config.get('enabled', False)
    except Exception:
        forex_config = {}
        forex_enabled = False

    # Status row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        status = "Active" if forex_enabled else "Disabled"
        status_color = "positive" if forex_enabled else "warning"
        render_metric_card("Forex Status", status, card_class=status_color)

    with col2:
        session = get_current_session() if forex_available else "unknown"
        session_display = session.replace("_", " ").title()
        render_metric_card("Market Session", session_display)

    with col3:
        total_pairs = len(FOREX_PAIRS) if forex_available else 0
        render_metric_card("Available Pairs", str(total_pairs))

    with col4:
        leverage = forex_config.get('leverage', {}).get('default', 50)
        render_metric_card("Max Leverage", f"{leverage}:1")

    st.markdown("---")

    # Forex pairs display
    st.markdown("### Currency Pairs")

    # Tabs for Majors and Crosses
    tab_majors, tab_crosses, tab_all = st.tabs(["Majors (7)", "Crosses (7)", "All Pairs (14)"])

    with tab_majors:
        st.markdown("#### Major Currency Pairs")
        _render_forex_pairs_grid(MAJOR_PAIRS if forex_available else {})

    with tab_crosses:
        st.markdown("#### Cross Currency Pairs")
        _render_forex_pairs_grid(CROSS_PAIRS if forex_available else {})

    with tab_all:
        st.markdown("#### All Currency Pairs")
        _render_forex_pairs_grid(FOREX_PAIRS if forex_available else {})

    st.markdown("---")

    # Position sizing calculator
    st.markdown("### Position Size Calculator")

    col1, col2 = st.columns(2)

    with col1:
        calc_symbol = st.selectbox(
            "Currency Pair",
            list(FOREX_PAIRS.keys()) if forex_available else ["EUR/USD"],
            key="forex_calc_symbol"
        )
        account_balance = st.number_input("Account Balance ($)", value=10000.0, min_value=100.0, step=100.0)
        risk_percent = st.number_input("Risk Per Trade (%)", value=1.0, min_value=0.1, max_value=5.0, step=0.1)

    with col2:
        stop_pips = st.number_input("Stop Loss (pips)", value=50.0, min_value=1.0, step=1.0)
        current_price = st.number_input("Current Price", value=1.1000, min_value=0.0001, step=0.0001, format="%.4f")

    if st.button("Calculate Position Size"):
        try:
            from src.portfolio.forex import ForexPositionSizer
            sizer = ForexPositionSizer(max_risk_percent=risk_percent)
            result = sizer.calculate_position_size(
                symbol=calc_symbol,
                account_equity=account_balance,
                stop_pips=stop_pips,
                current_price=current_price
            )

            st.success(f"""
            **Position Size:** {result.lots:.2f} lots ({result.units:,.0f} units)

            **Risk Amount:** ${result.risk_amount:.2f}

            **Margin Required:** ${result.margin_required:,.2f}

            **Pip Value:** ${result.pip_value:.2f}/pip
            """)

            if result.was_reduced:
                st.warning(f"Position reduced: {result.reduction_reason}")

        except Exception as e:
            st.error(f"Calculation error: {e}")

    st.markdown("---")

    # Twelve Data connection status
    st.markdown("### Twelve Data Connection")

    td_key = os.getenv("TWELVE_DATA_API_KEY", "")

    if td_key and td_key != "your_twelve_data_key_here":
        st.success("Twelve Data API key configured")

        if st.button("Test Twelve Data Connection"):
            try:
                from src.data.twelvedata_provider import TwelveDataProvider
                provider = TwelveDataProvider({'enabled': True})
                if provider.connect():
                    status = provider.get_status()
                    st.success(f"""
                    **Connected to Twelve Data**

                    Status: Connected

                    Daily calls remaining: {status.get('calls_remaining', 'N/A')}
                    """)
                    provider.disconnect()
                else:
                    st.error("Failed to connect to Twelve Data. Check your API key.")
            except Exception as e:
                st.error(f"Connection error: {e}")
    else:
        st.warning("Twelve Data API key not configured. Set TWELVE_DATA_API_KEY in .env")

    # Trading session info
    st.markdown("---")
    st.markdown("### Trading Sessions (UTC)")

    session_data = {
        'Session': ['Sydney', 'Tokyo', 'London', 'New York'],
        'Open (UTC)': ['21:00', '00:00', '08:00', '13:00'],
        'Close (UTC)': ['06:00', '09:00', '17:00', '22:00'],
        'Best Pairs': ['AUD/USD, NZD/USD', 'USD/JPY, EUR/JPY', 'EUR/USD, GBP/USD', 'EUR/USD, USD/CAD'],
        'Liquidity': ['Low', 'Medium', 'High', 'High']
    }
    st.dataframe(pd.DataFrame(session_data), use_container_width=True, hide_index=True)


def _render_forex_pairs_grid(pairs: dict):
    """Render forex pairs in a grid layout."""
    if not pairs:
        st.info("No pairs available")
        return

    # Convert to list for grid
    pairs_list = list(pairs.items())

    # 4 columns grid
    cols = st.columns(4)

    for i, (symbol, config) in enumerate(pairs_list):
        with cols[i % 4]:
            pip_size = config.pip_size if hasattr(config, 'pip_size') else 0.0001
            typical_spread = config.typical_spread if hasattr(config, 'typical_spread') else 1.0
            category = config.category if hasattr(config, 'category') else "major"

            # Card styling based on category
            border_color = "#667eea" if category == "major" else "#764ba2"

            st.markdown(f"""
            <div style="
                background: white;
                border-radius: 12px;
                padding: 1rem;
                margin-bottom: 1rem;
                border-left: 4px solid {border_color};
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            ">
                <div style="font-size: 1.1rem; font-weight: 700; color: #1a1a2e;">{symbol}</div>
                <div style="font-size: 0.75rem; color: #6c757d; margin-top: 0.3rem;">
                    Pip: {pip_size} | Spread: ~{typical_spread:.1f}
                </div>
                <div style="font-size: 0.7rem; color: #999; margin-top: 0.2rem;">
                    {category.upper()}
                </div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# PAGE: PAPER TRADING
# =============================================================================

@st.cache_data(ttl=30)  # Cache for 30 seconds
def load_ai_trades_for_chart(symbol: str, lookback_days: int = 7) -> pd.DataFrame:
    """Load AI training trades from database for chart markers with TP/SL levels."""
    db_path = Path("data/trading.db")
    if not db_path.exists():
        return pd.DataFrame()

    # Use UTC for consistent timezone handling with database
    cutoff_date = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()

    # Join with signals table to get TP/SL levels
    query = """
    SELECT
        t.id,
        t.symbol,
        t.interval,
        t.entry_price,
        t.exit_price,
        t.entry_time,
        t.exit_time,
        t.predicted_direction,
        t.predicted_confidence,
        t.was_correct,
        t.pnl_percent,
        t.regime,
        t.is_paper_trade,
        t.strategy_name,
        t.closed_by,
        s.stop_loss,
        s.take_profit,
        s.atr
    FROM trade_outcomes t
    LEFT JOIN signals s ON t.signal_id = s.id
    WHERE t.symbol = ? AND t.entry_time >= ?
    ORDER BY t.entry_time DESC
    """

    try:
        with sqlite3.connect(str(db_path)) as conn:
            df = pd.read_sql_query(query, conn, params=(symbol, cutoff_date))

            if len(df) > 0:
                df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)
                df['exit_time'] = pd.to_datetime(df['exit_time'], utc=True)
                # Fill NaN values for numeric columns
                df['pnl_percent'] = df['pnl_percent'].fillna(0)
                df['predicted_confidence'] = df['predicted_confidence'].fillna(0)
                df['stop_loss'] = df['stop_loss'].fillna(0)
                df['take_profit'] = df['take_profit'].fillna(0)

            return df
    except Exception as e:
        logger.error(f"Error loading AI trades: {e}")
        return pd.DataFrame()


def render_chart_with_trades(symbol: str, timeframe: str, historical_data: pd.DataFrame, trades_df: pd.DataFrame):
    """
    Render candlestick chart with AI trade markers using Plotly.

    Shows:
    - BUY signals as green triangles pointing up
    - SELL signals as red triangles pointing down
    - Trade PnL color coding (green=profit, red=loss)
    - Strategy name labels
    """
    if historical_data is None or historical_data.empty:
        st.warning("No historical data available for chart")
        return

    # Create figure with secondary y-axis for volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{symbol} - AI Training Trades', 'Volume'),
        row_heights=[0.75, 0.25]
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=historical_data['datetime'],
            open=historical_data['open'],
            high=historical_data['high'],
            low=historical_data['low'],
            close=historical_data['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )

    # Volume bars
    colors = ['#26a69a' if c >= o else '#ef5350'
              for c, o in zip(historical_data['close'], historical_data['open'])]

    fig.add_trace(
        go.Bar(
            x=historical_data['datetime'],
            y=historical_data['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.5
        ),
        row=2, col=1
    )

    # Add trade markers if we have trades
    if trades_df is not None and len(trades_df) > 0:
        # Helper function to get color based on correctness (handles None/NaN)
        def get_result_color(correct):
            if pd.isna(correct) or correct is None:
                return '#999999'  # Gray for unknown
            return '#00ff00' if correct else '#ff6b6b'

        def get_result_text(correct):
            if pd.isna(correct) or correct is None:
                return '⏳ Pending'
            return '✓ Correct' if correct else '✗ Wrong'

        # BUY trades
        buy_trades = trades_df[trades_df['predicted_direction'] == 'BUY']
        if len(buy_trades) > 0:
            # Color based on correctness (handles None properly)
            buy_colors = [get_result_color(c) for c in buy_trades['was_correct']]

            fig.add_trace(
                go.Scatter(
                    x=buy_trades['entry_time'],
                    y=buy_trades['entry_price'],
                    mode='markers+text',
                    name='BUY Signals',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color=buy_colors,
                        line=dict(width=2, color='darkgreen')
                    ),
                    text=[f"BUY<br>{s if s else 'AI'}<br>{p:.1f}%"
                          for s, p in zip(buy_trades['strategy_name'].fillna(''), buy_trades['pnl_percent'].fillna(0))],
                    textposition='bottom center',
                    textfont=dict(size=9),
                    hovertemplate=(
                        "<b>BUY Signal</b><br>"
                        "Time: %{x}<br>"
                        "Price: $%{y:,.2f}<br>"
                        "Strategy: %{customdata[0]}<br>"
                        "Confidence: %{customdata[1]:.1%}<br>"
                        "PnL: %{customdata[2]:.2f}%<br>"
                        "Result: %{customdata[3]}<extra></extra>"
                    ),
                    customdata=list(zip(
                        buy_trades['strategy_name'].fillna('AI Model'),
                        buy_trades['predicted_confidence'].fillna(0),
                        buy_trades['pnl_percent'].fillna(0),
                        [get_result_text(c) for c in buy_trades['was_correct']]
                    ))
                ),
                row=1, col=1
            )

        # SELL trades
        sell_trades = trades_df[trades_df['predicted_direction'] == 'SELL']
        if len(sell_trades) > 0:
            # Color based on correctness (handles None properly)
            sell_colors = [get_result_color(c) for c in sell_trades['was_correct']]

            fig.add_trace(
                go.Scatter(
                    x=sell_trades['entry_time'],
                    y=sell_trades['entry_price'],
                    mode='markers+text',
                    name='SELL Signals',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color=sell_colors,
                        line=dict(width=2, color='darkred')
                    ),
                    text=[f"SELL<br>{s if s else 'AI'}<br>{p:.1f}%"
                          for s, p in zip(sell_trades['strategy_name'].fillna(''), sell_trades['pnl_percent'].fillna(0))],
                    textposition='top center',
                    textfont=dict(size=9),
                    hovertemplate=(
                        "<b>SELL Signal</b><br>"
                        "Time: %{x}<br>"
                        "Price: $%{y:,.2f}<br>"
                        "Strategy: %{customdata[0]}<br>"
                        "Confidence: %{customdata[1]:.1%}<br>"
                        "PnL: %{customdata[2]:.2f}%<br>"
                        "Result: %{customdata[3]}<extra></extra>"
                    ),
                    customdata=list(zip(
                        sell_trades['strategy_name'].fillna('AI Model'),
                        sell_trades['predicted_confidence'].fillna(0),
                        sell_trades['pnl_percent'].fillna(0),
                        [get_result_text(c) for c in sell_trades['was_correct']]
                    ))
                ),
                row=1, col=1
            )

        # TP/SL LINES - Draw Take Profit and Stop Loss levels for each trade
        # This shows the true structure of paper trades for verification
        for idx, trade in trades_df.iterrows():
            entry_time = trade['entry_time']
            exit_time = trade.get('exit_time')
            entry_price = trade['entry_price']
            stop_loss = trade.get('stop_loss', 0)
            take_profit = trade.get('take_profit', 0)
            direction = trade['predicted_direction']
            was_correct = trade.get('was_correct')
            closed_by = trade.get('closed_by', '')

            # Skip if no TP/SL data
            if stop_loss == 0 and take_profit == 0:
                continue

            # Calculate time range for lines (entry to exit, or entry + 24h if no exit)
            if pd.notna(exit_time):
                line_end = exit_time
            else:
                line_end = entry_time + pd.Timedelta(hours=24)

            # Take Profit line (green dashed)
            if take_profit > 0:
                tp_color = '#00ff00' if closed_by == 'take_profit' else '#00ff0080'
                fig.add_trace(
                    go.Scatter(
                        x=[entry_time, line_end],
                        y=[take_profit, take_profit],
                        mode='lines',
                        name=f'TP ${take_profit:,.2f}',
                        line=dict(color=tp_color, width=2, dash='dash'),
                        showlegend=False,
                        hovertemplate=f"<b>Take Profit</b><br>Price: ${take_profit:,.2f}<br>Direction: {direction}<extra></extra>"
                    ),
                    row=1, col=1
                )

            # Stop Loss line (red dashed)
            if stop_loss > 0:
                sl_color = '#ff0000' if closed_by == 'stop_loss' else '#ff000080'
                fig.add_trace(
                    go.Scatter(
                        x=[entry_time, line_end],
                        y=[stop_loss, stop_loss],
                        mode='lines',
                        name=f'SL ${stop_loss:,.2f}',
                        line=dict(color=sl_color, width=2, dash='dash'),
                        showlegend=False,
                        hovertemplate=f"<b>Stop Loss</b><br>Price: ${stop_loss:,.2f}<br>Direction: {direction}<extra></extra>"
                    ),
                    row=1, col=1
                )

            # Entry price line (blue solid)
            fig.add_trace(
                go.Scatter(
                    x=[entry_time, line_end],
                    y=[entry_price, entry_price],
                    mode='lines',
                    name=f'Entry ${entry_price:,.2f}',
                    line=dict(color='#2196F3', width=1, dash='dot'),
                    showlegend=False,
                    hovertemplate=f"<b>Entry</b><br>Price: ${entry_price:,.2f}<br>Direction: {direction}<extra></extra>"
                ),
                row=1, col=1
            )

            # Exit price marker (if trade is closed)
            if pd.notna(exit_time) and trade.get('exit_price'):
                exit_price = trade['exit_price']
                exit_color = '#00ff00' if was_correct else '#ff6b6b'
                fig.add_trace(
                    go.Scatter(
                        x=[exit_time],
                        y=[exit_price],
                        mode='markers',
                        name=f'Exit ${exit_price:,.2f}',
                        marker=dict(
                            symbol='x',
                            size=12,
                            color=exit_color,
                            line=dict(width=2, color='black')
                        ),
                        showlegend=False,
                        hovertemplate=f"<b>Exit</b><br>Price: ${exit_price:,.2f}<br>PnL: {trade.get('pnl_percent', 0):.2f}%<extra></extra>"
                    ),
                    row=1, col=1
                )

    # Update layout
    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_rangeslider_visible=False,
        template='plotly_white'
    )

    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def page_paper_trading():
    """Paper trading page with AI training visualization, portfolio, and performance."""
    st.markdown("### 📊 Trading Center")

    paper_trader = st.session_state.paper_trader

    # Create tabs for Trading, Portfolio, Performance
    tab_trading, tab_portfolio, tab_performance = st.tabs(["🤖 AI Trading", "💼 Portfolio", "📈 Performance"])

    # =========================================================================
    # TAB: AI TRADING
    # =========================================================================
    with tab_trading:
        _render_ai_trading_tab(paper_trader)

    # =========================================================================
    # TAB: PORTFOLIO
    # =========================================================================
    with tab_portfolio:
        _render_portfolio_tab(paper_trader)

    # =========================================================================
    # TAB: PERFORMANCE
    # =========================================================================
    with tab_performance:
        _render_performance_tab(paper_trader)


def _render_ai_trading_tab(paper_trader):
    """Render the AI Trading tab content."""
    # Settings row
    col_settings1, col_settings2 = st.columns([1, 1])

    with col_settings1:
        lookback_days = st.slider(
            "Show trades from last (days)",
            min_value=1,
            max_value=30,
            value=7,
            key="paper_trade_lookback"
        )

    with col_settings2:
        show_chart = st.checkbox("Show Candlestick Chart with Trades", value=True)

    # Fetch market data for chart
    data = fetch_market_data(
        st.session_state.symbol,
        st.session_state.timeframe,
        limit=500  # ~3 weeks of 1h data
    )

    current_price = data.get('price', 0) if data['success'] else 0

    # Load AI trades
    ai_trades = load_ai_trades_for_chart(st.session_state.symbol, lookback_days)

    # Show chart with trade markers
    if show_chart and data['success']:
        st.markdown("---")
        st.markdown("#### 📈 Price Chart with AI Trade Signals & TP/SL Levels")
        st.markdown("""
        <div style="font-size: 12px; color: #666; margin-bottom: 10px;">
        <span style="color: #26a69a;">▲</span> BUY signals |
        <span style="color: #ef5350;">▼</span> SELL signals |
        <span style="color: #00ff00;">●</span> Correct |
        <span style="color: #ff6b6b;">●</span> Wrong |
        <span style="color: #00ff00;">- - -</span> Take Profit |
        <span style="color: #ff0000;">- - -</span> Stop Loss |
        <span style="color: #2196F3;">···</span> Entry |
        <span style="color: black;">✕</span> Exit
        </div>
        """, unsafe_allow_html=True)

        render_chart_with_trades(
            symbol=st.session_state.symbol,
            timeframe=st.session_state.timeframe,
            historical_data=data['df'],
            trades_df=ai_trades
        )

    # AI Learning Trades Summary
    st.markdown("---")
    st.markdown("#### 🤖 AI Learning Trades")

    if len(ai_trades) > 0:
        # Summary metrics
        total_trades = len(ai_trades)
        paper_trades = ai_trades['is_paper_trade'].sum()
        wins = ai_trades['was_correct'].sum()
        win_rate = wins / total_trades if total_trades > 0 else 0
        total_pnl = ai_trades['pnl_percent'].sum()

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Trades", total_trades)
        with col2:
            st.metric("Paper Trades", f"{paper_trades} ({100*paper_trades/total_trades:.0f}%)" if total_trades > 0 else "0")
        with col3:
            st.metric("Win Rate", f"{win_rate*100:.1f}%")
        with col4:
            pnl_color = "normal" if total_pnl >= 0 else "inverse"
            st.metric("Total PnL", f"{total_pnl:+.2f}%", delta_color=pnl_color)
        with col5:
            # Unique strategies used
            strategies_used = ai_trades['strategy_name'].dropna().nunique()
            st.metric("Strategies Used", strategies_used)

        # Performance Learning Stats (if available)
        if st.session_state.get('performance_learner'):
            try:
                perf_stats = st.session_state.performance_learner.get_stats()
                st.markdown("##### 📊 Performance Learning")

                col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)

                with col_p1:
                    consec_wins = perf_stats.get('consecutive_wins', 0)
                    max_wins = perf_stats.get('max_consecutive_wins', 0)
                    st.metric("Consec. Wins", consec_wins, delta=f"max: {max_wins}")

                with col_p2:
                    consec_losses = perf_stats.get('consecutive_losses', 0)
                    max_losses = perf_stats.get('max_consecutive_losses', 0)
                    delta_color = "inverse" if consec_losses > 0 else "normal"
                    st.metric("Consec. Losses", consec_losses, delta=f"max: {max_losses}", delta_color=delta_color)

                with col_p3:
                    retrains = perf_stats.get('retrains_triggered', 0)
                    last_reason = perf_stats.get('last_retrain_reason', '-')
                    st.metric("Retrains", retrains, delta=last_reason if retrains > 0 else None)

                with col_p4:
                    reinforcements = perf_stats.get('reinforcements_applied', 0)
                    st.metric("Reinforcements", reinforcements)

                with col_p5:
                    is_retraining = perf_stats.get('is_retraining', False)
                    action = st.session_state.performance_learner.get_recommended_action()
                    status_icon = "🔄" if is_retraining else "✅"
                    st.metric("Status", f"{status_icon} {action}")

            except Exception as e:
                logger.debug(f"Could not load performance learning stats: {e}")

        # Show trade table
        st.markdown("##### Recent Trades (with TP/SL)")

        # Format for display - include TP/SL columns
        columns_to_show = [
            'entry_time', 'predicted_direction', 'entry_price', 'stop_loss', 'take_profit',
            'exit_price', 'pnl_percent', 'was_correct', 'predicted_confidence', 'strategy_name', 'closed_by'
        ]

        # Only include columns that exist
        available_columns = [c for c in columns_to_show if c in ai_trades.columns]
        display_df = ai_trades[available_columns].copy()

        # Rename columns for display
        column_names = {
            'entry_time': 'Time',
            'predicted_direction': 'Dir',
            'entry_price': 'Entry',
            'stop_loss': 'SL',
            'take_profit': 'TP',
            'exit_price': 'Exit',
            'pnl_percent': 'PnL %',
            'was_correct': '✓',
            'predicted_confidence': 'Conf',
            'strategy_name': 'Strategy',
            'closed_by': 'Closed By'
        }
        display_df.rename(columns=column_names, inplace=True)

        # Format values
        if 'Time' in display_df.columns:
            display_df['Time'] = display_df['Time'].dt.strftime('%m-%d %H:%M')
        if 'Entry' in display_df.columns:
            display_df['Entry'] = display_df['Entry'].apply(lambda x: f"${x:,.0f}" if x > 1000 else f"${x:,.2f}")
        if 'SL' in display_df.columns:
            display_df['SL'] = display_df['SL'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) and x > 0 else "-")
        if 'TP' in display_df.columns:
            display_df['TP'] = display_df['TP'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) and x > 0 else "-")
        if 'Exit' in display_df.columns:
            display_df['Exit'] = display_df['Exit'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) and x > 1000 else (f"${x:,.2f}" if pd.notna(x) else "⏳"))
        if 'PnL %' in display_df.columns:
            display_df['PnL %'] = display_df['PnL %'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-")
        if '✓' in display_df.columns:
            display_df['✓'] = display_df['✓'].apply(lambda x: "✅" if x == True else ("❌" if x == False else "⏳"))
        if 'Conf' in display_df.columns:
            display_df['Conf'] = display_df['Conf'].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "-")
        if 'Strategy' in display_df.columns:
            display_df['Strategy'] = display_df['Strategy'].fillna('AI')
        if 'Dir' in display_df.columns:
            display_df['Dir'] = display_df['Dir'].apply(lambda x: "🟢" if x == 'BUY' else "🔴")
        if 'Closed By' in display_df.columns:
            display_df['Closed By'] = display_df['Closed By'].apply(
                lambda x: "🎯 TP" if x == 'take_profit' else ("🛑 SL" if x == 'stop_loss' else (x if pd.notna(x) else "⏳"))
            )

        st.dataframe(display_df, use_container_width=True, hide_index=True)

    else:
        st.info("""
        📝 **No AI trades yet.**

        Start the trading system with:
        ```bash
        python run_trading.py
        ```

        The AI will:
        1. Analyze market data with multiple strategies
        2. Execute paper trades in LEARNING mode
        3. Record outcomes for continuous learning
        4. Trades will appear here with strategy markers
        """)

    # Manual Order Form (collapsible)
    st.markdown("---")
    with st.expander("📝 Place Manual Order", expanded=False):
        if paper_trader is None:
            st.warning("Paper trading not initialized")
        else:
            col1, col2 = st.columns(2)

            with col1:
                order_side = st.radio("Side", ["BUY", "SELL"], horizontal=True, key="manual_order_side")
                quantity = st.number_input("Quantity", min_value=0.0001, value=0.01, step=0.001, format="%.4f")

            with col2:
                order_type = st.selectbox("Order Type", ["MARKET", "LIMIT"])
                if order_type == "LIMIT":
                    limit_price = st.number_input("Limit Price", value=float(current_price), step=0.01)
                else:
                    limit_price = current_price

            order_value = quantity * limit_price
            st.markdown(f"**Order Value:** ${order_value:,.2f}")

            validation = None
            if st.session_state.order_validator:
                order = {
                    'symbol': st.session_state.symbol,
                    'quantity': quantity,
                    'side': order_side,
                    'order_type': order_type,
                    'price': limit_price,
                    'current_price': current_price
                }

                portfolio = {
                    'total_value': paper_trader.get_account_value(),
                    'cash': paper_trader.cash,
                    'daily_pnl': 0
                }

                validation = st.session_state.order_validator.validate(order, portfolio)

                if validation.warnings:
                    for w in validation.warnings:
                        st.warning(w)

                if validation.errors:
                    for e in validation.errors:
                        st.error(e)

            if st.button("Submit Order", type="primary", disabled=not validation.is_valid if validation else False):
                try:
                    side = OrderSide.BUY if order_side == "BUY" else OrderSide.SELL
                    otype = OrderType.MARKET if order_type == "MARKET" else OrderType.LIMIT

                    trade = paper_trader.place_order(
                        symbol=st.session_state.symbol,
                        side=side,
                        quantity=quantity,
                        order_type=otype,
                        price=limit_price
                    )

                    if trade:
                        st.success(f"Order executed: {order_side} {quantity} {st.session_state.symbol} @ ${limit_price:,.2f}")
                        st.rerun()
                    else:
                        st.error("Order failed")
                except Exception as e:
                    st.error(f"Order error: {e}")


def _render_portfolio_tab(paper_trader):
    """Render the Portfolio tab content."""
    if paper_trader is None:
        st.warning("Paper trading not initialized")
        return

    # Portfolio metrics
    cols = st.columns(4)

    with cols[0]:
        render_metric_card("Total Value", f"${paper_trader.get_account_value():,.2f}")

    with cols[1]:
        render_metric_card("Cash", f"${paper_trader.cash:,.2f}")

    with cols[2]:
        pnl = paper_trader.get_account_value() - paper_trader.initial_cash
        pnl_pct = (pnl / paper_trader.initial_cash) * 100
        card_class = "positive" if pnl >= 0 else "negative"
        render_metric_card("P&L", f"${pnl:,.2f}", f"{pnl_pct:+.2f}%", card_class)

    with cols[3]:
        positions = paper_trader.get_positions()
        render_metric_card("Positions", str(len(positions)))

    st.markdown("---")

    # Positions table
    st.markdown("#### Open Positions")

    if positions:
        pos_data = []
        for pos in positions:
            pos_data.append({
                'Symbol': pos.symbol,
                'Side': pos.side.value,
                'Quantity': f"{pos.quantity:.6f}",
                'Entry Price': f"${pos.entry_price:,.2f}",
                'Current Price': f"${pos.current_price:,.2f}",
                'P&L': f"${pos.unrealized_pnl:,.2f}",
                'P&L %': f"{pos.unrealized_pnl_percent:.2f}%"
            })
        st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)
    else:
        st.info("No open positions")

    # Trade history
    st.markdown("#### Trade History")

    trades = paper_trader.get_trades()
    if trades:
        trade_data = []
        for t in trades[-20:]:  # Last 20 trades
            side_str = t.side.value if hasattr(t.side, 'value') else str(t.side)
            trade_data.append({
                'Time': t.timestamp.strftime('%Y-%m-%d %H:%M'),
                'Symbol': t.symbol,
                'Side': side_str.upper(),
                'Quantity': f"{t.quantity:.6f}",
                'Price': f"${t.price:,.2f}",
                'P&L': f"${t.realized_pnl:,.2f}" if t.realized_pnl else "-"
            })
        st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)
    else:
        st.info("No trades yet")

    # Show continuous learning trade outcomes from database
    st.markdown("---")
    st.markdown("#### AI Learning Trades (from Database)")

    try:
        db = st.session_state.db
        if db:
            with db.connection() as conn:
                outcomes_df = pd.read_sql_query('''
                    SELECT
                        entry_time,
                        symbol,
                        interval,
                        predicted_direction,
                        predicted_confidence,
                        entry_price,
                        exit_price,
                        was_correct,
                        pnl_percent,
                        regime,
                        is_paper_trade
                    FROM trade_outcomes
                    ORDER BY entry_time DESC
                    LIMIT 50
                ''', conn)

            if len(outcomes_df) > 0:
                outcomes_df['Result'] = outcomes_df['was_correct'].apply(
                    lambda x: '✅ Win' if x else '❌ Loss'
                )
                outcomes_df['Type'] = outcomes_df['is_paper_trade'].apply(
                    lambda x: '📝 Paper' if x else '💰 Live'
                )
                outcomes_df['Confidence'] = outcomes_df['predicted_confidence'].apply(
                    lambda x: f"{x*100:.1f}%" if x else 'N/A'
                )
                outcomes_df['P&L'] = outcomes_df['pnl_percent'].apply(
                    lambda x: f"{x:+.2f}%" if x else 'N/A'
                )

                display_df = outcomes_df[[
                    'entry_time', 'symbol', 'interval', 'predicted_direction',
                    'Confidence', 'P&L', 'Result', 'Type', 'regime'
                ]].rename(columns={
                    'entry_time': 'Time',
                    'symbol': 'Symbol',
                    'interval': 'Timeframe',
                    'predicted_direction': 'Direction',
                    'regime': 'Regime'
                })

                st.dataframe(display_df, use_container_width=True, hide_index=True)

                total_trades = len(outcomes_df)
                wins = outcomes_df['was_correct'].sum()
                win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
                paper_trades = outcomes_df['is_paper_trade'].sum()

                st.markdown(f"""
                **Summary:** {total_trades} trades | {wins} wins ({win_rate:.1f}% win rate) |
                {paper_trades} paper / {total_trades - paper_trades} live
                """)
            else:
                st.info("No AI learning trades recorded yet. Start the analysis engine to generate trades.")
        else:
            st.warning("Database not initialized")
    except Exception as e:
        st.error(f"Error loading trade outcomes: {e}")


def _render_performance_tab(paper_trader):
    """Render the Performance tab content."""
    if paper_trader is None:
        st.warning("Paper trading not initialized")
        return

    trades = paper_trader.get_trades()

    if len(trades) < 2:
        st.info("Need at least 2 trades for performance analysis")
        return

    # Build equity curve
    equity = [paper_trader.initial_cash]
    for t in trades:
        if t.realized_pnl:
            equity.append(equity[-1] + t.realized_pnl)

    equity_series = pd.Series(equity)
    equity_series.index = pd.date_range(end=datetime.now(), periods=len(equity), freq='h')

    # Calculate metrics
    trade_dicts = [{'pnl': t.realized_pnl or 0} for t in trades]

    if st.session_state.metrics_calculator:
        metrics = st.session_state.metrics_calculator.calculate(equity_series, trade_dicts)

        st.markdown("#### Key Metrics")

        cols = st.columns(5)

        with cols[0]:
            render_metric_card("Total Return", f"{metrics.total_return:.2%}")
        with cols[1]:
            render_metric_card("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
        with cols[2]:
            render_metric_card("Sortino Ratio", f"{metrics.sortino_ratio:.2f}")
        with cols[3]:
            render_metric_card("Max Drawdown", f"{metrics.max_drawdown:.2%}")
        with cols[4]:
            render_metric_card("Win Rate", f"{metrics.win_rate:.1%}")

        cols2 = st.columns(5)

        with cols2[0]:
            render_metric_card("Total Trades", str(metrics.total_trades))
        with cols2[1]:
            render_metric_card("Winners", str(metrics.winning_trades))
        with cols2[2]:
            render_metric_card("Losers", str(metrics.losing_trades))
        with cols2[3]:
            render_metric_card("Profit Factor", f"{metrics.profit_factor:.2f}")
        with cols2[4]:
            render_metric_card("Avg Trade", f"${metrics.avg_win - abs(metrics.avg_loss):,.2f}")

        st.markdown("---")

        # Equity curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_series.index,
            y=equity_series.values,
            mode='lines',
            fill='tozeroy',
            name='Equity',
            line=dict(color='#667eea', width=2)
        ))
        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Time",
            yaxis_title="Equity ($)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # Risk management section (merged from Risk page)
    st.markdown("---")
    st.markdown("#### Risk Status")

    config = load_config()
    risk_config = config.get('risk', {})

    cols = st.columns(4)

    with cols[0]:
        max_dd = risk_config.get('max_drawdown_percent', 20)
        render_metric_card("Max DD Limit", f"{max_dd}%")

    with cols[1]:
        daily_limit = risk_config.get('daily_loss_limit', 0.05) * 100
        render_metric_card("Daily Loss Limit", f"{daily_limit}%")

    with cols[2]:
        max_pos = risk_config.get('max_position_percent', 0.25) * 100
        render_metric_card("Max Position", f"{max_pos}%")

    with cols[3]:
        # Circuit breaker status
        validator = st.session_state.order_validator
        if validator and validator.circuit_breaker_active:
            render_metric_card("Circuit Breaker", "🔴 ACTIVE", "", "negative")
        else:
            render_metric_card("Circuit Breaker", "🟢 OK", "", "positive")


# =============================================================================
# PAGE: SETTINGS
# =============================================================================

def page_settings():
    """Settings page."""
    st.markdown("### Settings")

    config = load_config()

    tabs = st.tabs(["Trading", "Risk", "Learning", "Notifications", "System"])

    with tabs[0]:  # Trading
        st.markdown("#### Trading Settings")

        # Market Mode Selection
        st.markdown("##### Market Mode")
        current_market_mode = config.get('market_mode', 'crypto')
        market_modes = {'Cryptocurrency (Binance)': 'crypto', 'Forex (Twelve Data)': 'forex'}
        market_mode_labels = list(market_modes.keys())
        current_mode_label = [k for k, v in market_modes.items() if v == current_market_mode]
        current_mode_label = current_mode_label[0] if current_mode_label else market_mode_labels[0]

        new_market_mode_label = st.radio(
            "Select market type:",
            market_mode_labels,
            index=market_mode_labels.index(current_mode_label),
            horizontal=True
        )
        new_market_mode = market_modes[new_market_mode_label]

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            # Exchange and symbol options based on market mode
            if new_market_mode == 'crypto':
                exchange_options = ['binance']
                crypto_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
                                 'ADA/USDT', 'AVAX/USDT', 'DOGE/USDT', 'DOT/USDT', 'LINK/USDT',
                                 'MATIC/USDT', 'UNI/USDT', 'ATOM/USDT', 'LTC/USDT', 'TRX/USDT']
                symbol_options = crypto_symbols
                st.info("📈 **Crypto Mode**: Trading on Binance exchange")
            else:  # forex
                exchange_options = ['twelvedata']
                forex_symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD',
                                'USD/CAD', 'NZD/USD', 'EUR/GBP', 'EUR/JPY', 'GBP/JPY']
                symbol_options = forex_symbols
                st.info("💱 **Forex Mode**: Data via Twelve Data API (requires API key)")

                td_key = os.getenv("TWELVE_DATA_API_KEY", "")
                if not td_key or td_key == "your_twelve_data_key_here":
                    st.warning("⚠️ **Twelve Data API key not configured.** Set TWELVE_DATA_API_KEY in your .env file to enable forex data.")

            current_symbol = config.get('data', {}).get('symbol', symbol_options[0])
            if current_symbol not in symbol_options:
                current_symbol = symbol_options[0]
            new_symbol = st.selectbox("Symbol", symbol_options,
                                     index=symbol_options.index(current_symbol) if current_symbol in symbol_options else 0)

            current_exchange = config.get('data', {}).get('exchange', exchange_options[0])
            new_exchange = st.selectbox("Exchange", exchange_options,
                                        index=exchange_options.index(current_exchange) if current_exchange in exchange_options else 0)

            all_timeframes = ['15m', '1h', '4h', '1d']  # Only 4 optimized timeframes
            current_interval = config.get('data', {}).get('interval', '1h')
            new_interval = st.selectbox("Timeframe", all_timeframes,
                                        index=all_timeframes.index(current_interval) if current_interval in all_timeframes else 1)

        with col2:
            new_capital = st.number_input("Initial Capital", value=config.get('portfolio', {}).get('initial_capital', 10000))
            new_position_method = st.selectbox("Position Sizing", ['equal_weight', 'risk_parity', 'kelly', 'volatility_target'])

    with tabs[1]:  # Risk
        st.markdown("#### Risk Settings")

        col1, col2 = st.columns(2)

        with col1:
            max_dd = st.slider("Max Drawdown %", 5, 50, int(config.get('risk', {}).get('max_drawdown_percent', 20)))
            daily_limit = st.slider("Daily Loss Limit %", 1, 20, int(config.get('risk', {}).get('daily_loss_limit', 0.05) * 100))

        with col2:
            max_pos = st.slider("Max Position Size %", 5, 50, int(config.get('risk', {}).get('max_position_percent', 0.25) * 100))
            sector_exp = st.slider("Max Sector Exposure %", 10, 80, int(config.get('risk', {}).get('max_sector_exposure', 0.40) * 100))

    with tabs[2]:  # Learning
        st.markdown("#### Continuous Learning Settings")

        # Get current learning config from session state or config
        perf_config = st.session_state.get('perf_learning_config', {})
        cl_config = config.get('continuous_learning', {}).get('performance_learning', {})

        st.markdown("##### Per-Candle Learning")
        st.caption("Learn from every trade: reinforce wins, retrain on losses")

        col1, col2 = st.columns(2)

        with col1:
            perf_enabled = st.checkbox(
                "Enable Performance Learning",
                value=perf_config.get('enabled', cl_config.get('enabled', True)),
                help="Enable per-candle learning based on trade outcomes"
            )
            loss_retrain = st.checkbox(
                "Retrain on Loss",
                value=perf_config.get('loss_retrain', cl_config.get('loss_retrain', True)),
                help="Trigger retraining when trades result in losses"
            )
            reinforce_wins = st.checkbox(
                "Reinforce on Win",
                value=perf_config.get('reinforce_wins', cl_config.get('reinforce_wins', True)),
                help="Apply reinforcement learning for winning trades"
            )

        with col2:
            consecutive_loss = st.slider(
                "Consecutive Loss Threshold",
                min_value=2, max_value=10,
                value=perf_config.get('consecutive_loss_threshold', cl_config.get('consecutive_loss_threshold', 3)),
                help="Trigger FULL retrain after N consecutive losses"
            )
            win_rate_threshold = st.slider(
                "Win Rate Threshold %",
                min_value=30, max_value=60,
                value=int(perf_config.get('win_rate_threshold', cl_config.get('win_rate_threshold', 0.45)) * 100),
                help="Trigger MEDIUM retrain if win rate drops below this"
            )

        st.markdown("##### Retraining Epochs")
        st.caption("Number of training epochs for each retrain level")

        col3, col4, col5 = st.columns(3)

        with col3:
            light_epochs = st.number_input(
                "Light (minor loss)",
                min_value=10, max_value=100,
                value=perf_config.get('light_epochs', cl_config.get('light_epochs', 30)),
                help="Quick update for single losses"
            )
        with col4:
            medium_epochs = st.number_input(
                "Medium (win rate drop)",
                min_value=30, max_value=150,
                value=perf_config.get('medium_epochs', cl_config.get('medium_epochs', 50)),
                help="Standard retrain for declining performance"
            )
        with col5:
            full_epochs = st.number_input(
                "Full (consecutive losses)",
                min_value=50, max_value=200,
                value=perf_config.get('full_epochs', cl_config.get('full_epochs', 100)),
                help="Complete retrain for major issues"
            )

        # Update session state
        st.session_state.perf_learning_config = {
            'enabled': perf_enabled,
            'loss_retrain': loss_retrain,
            'reinforce_wins': reinforce_wins,
            'consecutive_loss_threshold': consecutive_loss,
            'win_rate_threshold': win_rate_threshold / 100,
            'light_epochs': light_epochs,
            'medium_epochs': medium_epochs,
            'full_epochs': full_epochs,
        }

        # Show current performance learner status
        st.markdown("---")
        st.markdown("##### Learning Status")

        if st.session_state.get('performance_learner'):
            try:
                stats = st.session_state.performance_learner.get_stats()
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)

                with col_s1:
                    st.metric("Win Rate", f"{stats.get('overall_win_rate', 0):.1%}")
                with col_s2:
                    st.metric("Retrains", stats.get('retrains_triggered', 0))
                with col_s3:
                    st.metric("Reinforcements", stats.get('reinforcements_applied', 0))
                with col_s4:
                    is_retraining = stats.get('is_retraining', False)
                    st.metric("Status", "🔄 Retraining" if is_retraining else "✅ Ready")

                # Show streak info
                consec_wins = stats.get('consecutive_wins', 0)
                consec_losses = stats.get('consecutive_losses', 0)
                if consec_wins > 0:
                    st.success(f"🔥 {consec_wins} consecutive wins!")
                elif consec_losses > 0:
                    st.warning(f"⚠️ {consec_losses} consecutive losses")

            except Exception as e:
                st.info("Performance learner stats unavailable")
        else:
            st.info("Performance learner not initialized. Start trading to enable.")

    with tabs[3]:  # Notifications
        st.markdown("#### Notification Settings")

        desktop = st.checkbox("Desktop Notifications", value=config.get('notifications', {}).get('desktop', {}).get('enabled', True))
        sound = st.checkbox("Sound Alerts", value=config.get('notifications', {}).get('desktop', {}).get('sound', True))
        telegram = st.checkbox("Telegram", value=config.get('notifications', {}).get('telegram', {}).get('enabled', False))

        if telegram:
            bot_token = st.text_input("Bot Token", type="password")
            chat_id = st.text_input("Chat ID")

    with tabs[4]:  # System
        st.markdown("#### System Settings")

        auto_refresh = st.checkbox("Auto Refresh", value=True)
        refresh_interval = st.slider("Refresh Interval (seconds)", 1, 60, 5)

        st.markdown("#### Engine Control")

        running, pid = is_engine_running()

        if running:
            st.success(f"Engine running (PID: {pid})")
            if st.button("Stop Engine"):
                try:
                    os.kill(pid, signal.SIGTERM)
                    PID_FILE.unlink(missing_ok=True)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to stop: {e}")
        else:
            st.warning("Engine stopped")
            if st.button("Start Engine"):
                try:
                    venv_py = ROOT / "venv" / "bin" / "python"
                    py = str(venv_py) if venv_py.exists() else "python3"
                    subprocess.Popen([py, str(ROOT / "run_analysis.py")],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=ROOT)
                    time.sleep(2)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to start: {e}")

    if st.button("Save Settings", type="primary"):
        # Build updated config
        updated_config = config.copy()

        # Update market mode
        updated_config['market_mode'] = new_market_mode

        # Update data section
        if 'data' not in updated_config:
            updated_config['data'] = {}
        updated_config['data']['symbol'] = new_symbol
        updated_config['data']['exchange'] = new_exchange
        updated_config['data']['interval'] = new_interval

        # Update portfolio section
        if 'portfolio' not in updated_config:
            updated_config['portfolio'] = {}
        updated_config['portfolio']['initial_capital'] = int(new_capital)
        updated_config['portfolio']['position_sizing'] = new_position_method

        # Update risk section
        if 'risk' not in updated_config:
            updated_config['risk'] = {}
        updated_config['risk']['max_drawdown_percent'] = max_dd
        updated_config['risk']['daily_loss_limit'] = daily_limit / 100
        updated_config['risk']['max_position_percent'] = max_pos / 100
        updated_config['risk']['max_sector_exposure'] = sector_exp / 100

        # Update notifications section
        if 'notifications' not in updated_config:
            updated_config['notifications'] = {}
        if 'desktop' not in updated_config['notifications']:
            updated_config['notifications']['desktop'] = {}
        updated_config['notifications']['desktop']['enabled'] = desktop
        updated_config['notifications']['desktop']['sound'] = sound
        if 'telegram' not in updated_config['notifications']:
            updated_config['notifications']['telegram'] = {}
        updated_config['notifications']['telegram']['enabled'] = telegram

        # Update continuous learning / performance learning section
        if 'continuous_learning' not in updated_config:
            updated_config['continuous_learning'] = {}
        perf_cfg = st.session_state.get('perf_learning_config', {})
        updated_config['continuous_learning']['performance_learning'] = {
            'enabled': perf_cfg.get('enabled', True),
            'loss_retrain': perf_cfg.get('loss_retrain', True),
            'reinforce_wins': perf_cfg.get('reinforce_wins', True),
            'consecutive_loss_threshold': perf_cfg.get('consecutive_loss_threshold', 3),
            'win_rate_threshold': perf_cfg.get('win_rate_threshold', 0.45),
            'light_epochs': perf_cfg.get('light_epochs', 30),
            'medium_epochs': perf_cfg.get('medium_epochs', 50),
            'full_epochs': perf_cfg.get('full_epochs', 100),
        }

        # Save to config.yaml
        try:
            with open(CONFIG_PATH, 'w') as f:
                yaml.dump(updated_config, f, default_flow_style=False, sort_keys=False)

            # Update session state immediately
            st.session_state.market_mode = new_market_mode
            st.session_state.symbol = new_symbol
            st.session_state.exchange = new_exchange
            st.session_state.timeframe = new_interval
            st.session_state.paper_capital = int(new_capital)
            st.session_state.auto_refresh = auto_refresh
            st.session_state.refresh_interval = refresh_interval

            # Clear the config cache so it reloads
            load_config.clear()

            st.success("Settings saved!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to save settings: {e}")


# =============================================================================
# MAIN NAVIGATION
# =============================================================================

def main():
    """Main application."""
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## Traot")
        st.markdown("---")

        # Page selection (dynamic based on market mode)
        market_mode = st.session_state.get('market_mode', 'crypto')

        # Streamlined navigation - consolidated pages
        # Trading Center includes: AI Trading, Portfolio, Performance tabs
        pages = {
            "📊 Dashboard": "Dashboard",
            "📝 Trading Center": "Paper Trading",
            "📈 Learning": "Learning",
            "⚙️ Settings": "Settings"
        }

        # Add Forex Markets page when in forex mode
        if market_mode == 'forex':
            pages = {
                "📊 Dashboard": "Dashboard",
                "💱 Forex Markets": "Forex Markets",
                "📝 Trading Center": "Paper Trading",
                "📈 Learning": "Learning",
                "⚙️ Settings": "Settings"
            }

        for label, page_name in pages.items():
            if st.button(label, use_container_width=True,
                        type="primary" if st.session_state.page == page_name else "secondary"):
                st.session_state.page = page_name
                st.rerun()

        st.markdown("---")

        # Auto-refresh with explanation
        st.markdown("### 🔄 Auto-Refresh Settings")

        auto_refresh = st.checkbox(
            "Enable Manual Auto-Refresh",
            value=st.session_state.auto_refresh,
            help="Enable to always auto-refresh. Dashboard auto-refreshes automatically during data loading and active predictions."
        )
        st.session_state.auto_refresh = auto_refresh

        st.info("""
        **📊 Chart is ALWAYS LIVE** - updates in real-time via WebSocket (no refresh needed!)

        **Smart Auto-Refresh** is only for:
        - 📈 Historical data loading progress bar
        - 🎯 Prediction validation streak updates
        - ✅ Manually enabled above

        **When idle** (data loaded, no pending predictions), auto-refresh stops automatically.
        """)

        if st.button("🔄 Refresh Now"):
            st.rerun()

    # Render selected page (streamlined navigation)
    page_functions = {
        "Dashboard": page_dashboard,
        "Forex Markets": page_forex_markets,
        "Paper Trading": page_paper_trading,  # Now includes Portfolio + Performance tabs
        "Learning": page_learning,
        "Settings": page_settings,
    }

    # Smart Auto-refresh - ONLY for prediction validation updates
    # NOTE: Chart is ALREADY LIVE via JavaScript WebSocket - no refresh needed for chart!
    should_auto_refresh = False
    refresh_interval_ms = st.session_state.refresh_interval * 1000  # Default 30s

    # Check if provider is actively fetching historical data
    provider = st.session_state.data_provider
    # Safe check - method may not exist if provider is cached from old version
    is_fetching = False
    if provider and hasattr(provider, 'is_fetching'):
        is_fetching = provider.is_fetching()

    # Reason 1: User explicitly enabled manual auto-refresh
    if st.session_state.auto_refresh:
        should_auto_refresh = True

    # Reason 2: Historical data is still loading (to show progress bar updates)
    # BUT use a much longer interval if fetch is in progress to avoid killing threads
    if hasattr(st.session_state, 'data_loading') and st.session_state.data_loading:
        should_auto_refresh = True
        if is_fetching:
            # Use 90 second interval during active fetch to let it complete
            refresh_interval_ms = 90000
        else:
            # Use 10 second interval when data is loading but fetch finished
            refresh_interval_ms = 10000

    # Reason 3: Active predictions being validated (to show streak updates)
    if hasattr(st.session_state, 'prediction_validator'):
        validator = st.session_state.prediction_validator
        if validator and hasattr(validator, '_pending_predictions'):
            if len(validator._pending_predictions) > 0:
                should_auto_refresh = True

    # Only auto-refresh when necessary
    # Chart updates live via WebSocket - this is ONLY for prediction validation UI updates
    if should_auto_refresh:
        st_autorefresh(interval=refresh_interval_ms, key="data_refresh")

    page_func = page_functions.get(st.session_state.page, page_dashboard)
    page_func()


if __name__ == "__main__":
    main()
