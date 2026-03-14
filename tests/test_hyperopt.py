"""
Tests for Hyperopt — Parameter Spaces and Loss Functions
=========================================================
Covers:
  - Task 2: parameter_space.py  (TestParameterSpace)
  - Task 3: loss_functions.py   (TestLossFunctions)
"""

import math

import pytest

from src.backtesting.metrics import BacktestMetrics
from src.optimize.parameter_space import (
    ENSEMBLE_PARAMS,
    RISK_PARAMS,
    TRADING_PARAMS,
    apply_params_to_config,
    get_parameter_space,
    normalize_ensemble_weights,
    suggest_params,
)
from src.optimize.loss_functions import (
    MAX_LOSS,
    _hard_kill,
    compute_loss,
    get_available_loss_functions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeTrial:
    """Minimal Optuna trial mock that records suggestions."""

    def __init__(self):
        self.suggestions: dict = {}

    def suggest_float(self, name: str, low: float, high: float, *, step: float = None) -> float:
        value = (low + high) / 2
        self.suggestions[name] = value
        return value

    def suggest_int(self, name: str, low: int, high: int, *, step: int = 1) -> int:
        value = (low + high) // 2
        self.suggestions[name] = value
        return value


def _good_metrics(**overrides) -> BacktestMetrics:
    """Return metrics that pass all hard-kill checks by default."""
    defaults = dict(
        total_trades=50,
        winners=30,
        losers=20,
        win_rate=0.60,
        total_pnl_percent=25.0,
        avg_pnl_percent=0.50,
        avg_winner_pnl=2.0,
        avg_loser_pnl=-1.0,
        largest_winner=5.0,
        largest_loser=-3.0,
        max_drawdown=8.0,
        sharpe_ratio=1.8,
        profit_factor=2.0,
        expectancy=0.80,
    )
    defaults.update(overrides)
    return BacktestMetrics(**defaults)


# ===================================================================
# TestParameterSpace
# ===================================================================

class TestParameterSpace:
    """Tests for src/optimize/parameter_space.py."""

    # --- get_parameter_space ---

    def test_trading_space_count(self):
        params = get_parameter_space(["trading"])
        assert len(params) == 10

    def test_ensemble_space_count(self):
        params = get_parameter_space(["ensemble"])
        assert len(params) == 6

    def test_risk_space_count(self):
        params = get_parameter_space(["risk"])
        assert len(params) == 4

    def test_combined_spaces(self):
        params = get_parameter_space(["trading", "ensemble", "risk"])
        assert len(params) == 20

    def test_unknown_space_raises(self):
        with pytest.raises(ValueError, match="Unknown parameter space"):
            get_parameter_space(["nonexistent"])

    def test_all_params_have_required_keys(self):
        for p in get_parameter_space(["trading", "ensemble", "risk"]):
            assert "name" in p
            assert "type" in p
            assert "low" in p
            assert "high" in p
            assert "config_path" in p
            assert p["low"] < p["high"]

    # --- suggest_params ---

    def test_suggest_params_returns_all(self):
        trial = FakeTrial()
        params = suggest_params(trial, ["trading"])
        assert len(params) == 10
        for p_def in TRADING_PARAMS:
            assert p_def["name"] in params

    def test_suggest_params_int_type(self):
        trial = FakeTrial()
        params = suggest_params(trial, ["trading"])
        assert isinstance(params["cooldown_minutes"], int)

    def test_suggest_params_float_type(self):
        trial = FakeTrial()
        params = suggest_params(trial, ["trading"])
        assert isinstance(params["confidence_threshold"], float)

    def test_suggest_params_values_within_bounds(self):
        trial = FakeTrial()
        params = suggest_params(trial, ["trading", "ensemble", "risk"])
        all_defs = get_parameter_space(["trading", "ensemble", "risk"])
        for p_def in all_defs:
            val = params[p_def["name"]]
            assert p_def["low"] <= val <= p_def["high"], (
                f"{p_def['name']}: {val} not in [{p_def['low']}, {p_def['high']}]"
            )

    # --- apply_params_to_config ---

    def test_apply_creates_nested_keys(self):
        cfg: dict = {}
        apply_params_to_config(cfg, {"confidence_threshold": 0.85})
        assert cfg["continuous_learning"]["confidence"]["trading_threshold"] == 0.85

    def test_apply_overwrites_existing(self):
        cfg = {"signals": {"risk_per_trade": 0.02}}
        apply_params_to_config(cfg, {"risk_per_trade": 0.03})
        assert cfg["signals"]["risk_per_trade"] == 0.03

    def test_apply_multiple_params(self):
        cfg: dict = {}
        apply_params_to_config(cfg, {
            "sl_multiplier": 2.5,
            "tp_multiplier": 4.0,
            "kelly_fraction": 0.8,
        })
        assert cfg["prediction"]["atr"]["sl_multiplier"] == 2.5
        assert cfg["prediction"]["atr"]["tp_multiplier"] == 4.0
        assert cfg["position_sizing"]["kelly_fraction"] == 0.8

    def test_apply_ignores_unknown_param(self):
        cfg: dict = {}
        apply_params_to_config(cfg, {"unknown_param": 999})
        assert cfg == {}

    # --- normalize_ensemble_weights ---

    def test_normalize_sums_to_one(self):
        raw = {"lstm": 0.55, "fourier": 0.10, "kalman": 0.15,
               "markov": 0.10, "entropy": 0.05, "monte_carlo": 0.05}
        result = normalize_ensemble_weights(raw)
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_normalize_lstm_floor(self):
        raw = {"lstm": 0.10, "fourier": 0.30, "kalman": 0.30,
               "markov": 0.10, "entropy": 0.10, "monte_carlo": 0.10}
        result = normalize_ensemble_weights(raw)
        assert result["lstm"] >= 0.40
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_normalize_preserves_ratios(self):
        raw = {"lstm": 0.60, "fourier": 0.10, "kalman": 0.20,
               "markov": 0.05, "entropy": 0.03, "monte_carlo": 0.02}
        result = normalize_ensemble_weights(raw)
        # fourier/kalman ratio should be preserved
        assert abs(result["fourier"] / result["kalman"] - 0.10 / 0.20) < 1e-9

    def test_normalize_all_zeros(self):
        raw = {"lstm": 0.0, "fourier": 0.0, "kalman": 0.0,
               "markov": 0.0, "entropy": 0.0, "monte_carlo": 0.0}
        result = normalize_ensemble_weights(raw)
        assert result["lstm"] == 0.40
        assert abs(sum(result.values()) - 1.0) < 1e-9


# ===================================================================
# TestLossFunctions
# ===================================================================

class TestLossFunctions:
    """Tests for src/optimize/loss_functions.py."""

    # --- get_available_loss_functions ---

    def test_available_functions(self):
        names = get_available_loss_functions()
        assert set(names) == {"traot", "sharpe", "sortino", "calmar", "profit"}

    # --- _hard_kill ---

    def test_hard_kill_low_trades(self):
        m = _good_metrics(total_trades=5)
        assert _hard_kill(m, 15.0) is True

    def test_hard_kill_low_winrate(self):
        m = _good_metrics(win_rate=0.45)
        assert _hard_kill(m, 15.0) is True

    def test_hard_kill_high_drawdown(self):
        m = _good_metrics(max_drawdown=20.0)
        assert _hard_kill(m, 15.0) is True

    def test_hard_kill_large_single_loss(self):
        m = _good_metrics(largest_loser=-6.0)
        assert _hard_kill(m, 15.0) is True

    def test_hard_kill_low_profit_factor(self):
        m = _good_metrics(profit_factor=0.8)
        assert _hard_kill(m, 15.0) is True

    def test_hard_kill_passes_good_metrics(self):
        m = _good_metrics()
        assert _hard_kill(m, 15.0) is False

    # --- compute_loss returns MAX_LOSS on hard-kill ---

    def test_compute_loss_returns_max_on_kill(self):
        m = _good_metrics(total_trades=3)
        loss = compute_loss(m, "traot", backtest_days=30)
        assert loss == MAX_LOSS

    # --- compute_loss unknown function ---

    def test_compute_loss_unknown_raises(self):
        m = _good_metrics()
        with pytest.raises(ValueError, match="Unknown loss function"):
            compute_loss(m, "unknown", backtest_days=30)

    # --- traot loss ---

    def test_traot_loss_negative(self):
        m = _good_metrics()
        loss = compute_loss(m, "traot", backtest_days=30)
        assert loss < 0, "traot loss should be negative for good metrics"

    def test_traot_better_metrics_lower_loss(self):
        good = _good_metrics(win_rate=0.70, profit_factor=3.0)
        okay = _good_metrics(win_rate=0.55, profit_factor=1.5)
        loss_good = compute_loss(good, "traot", backtest_days=30)
        loss_okay = compute_loss(okay, "traot", backtest_days=30)
        assert loss_good < loss_okay

    # --- sharpe loss ---

    def test_sharpe_loss(self):
        m = _good_metrics(sharpe_ratio=2.0)
        loss = compute_loss(m, "sharpe", backtest_days=30)
        assert loss == pytest.approx(-2.0)

    # --- sortino loss ---

    def test_sortino_loss_negative(self):
        m = _good_metrics()
        loss = compute_loss(m, "sortino", backtest_days=30)
        assert loss < 0

    def test_sortino_fallback_to_sharpe(self):
        m = _good_metrics(losers=1, total_trades=50)
        loss = compute_loss(m, "sortino", backtest_days=30)
        # Should fall back to -sharpe_ratio
        assert loss == pytest.approx(-m.sharpe_ratio)

    # --- calmar loss ---

    def test_calmar_loss_negative(self):
        m = _good_metrics(total_pnl_percent=20.0, max_drawdown=5.0)
        loss = compute_loss(m, "calmar", backtest_days=30)
        assert loss == pytest.approx(-4.0)

    def test_calmar_zero_drawdown_positive_pnl(self):
        m = _good_metrics(total_pnl_percent=10.0, max_drawdown=0.0)
        loss = compute_loss(m, "calmar", backtest_days=30)
        assert loss == pytest.approx(-10.0)

    # --- profit loss ---

    def test_profit_loss(self):
        m = _good_metrics(total_pnl_percent=30.0)
        loss = compute_loss(m, "profit", backtest_days=30)
        assert loss == pytest.approx(-30.0)

    # --- max_dd_param ceiling ---

    def test_max_dd_param_none_default_ceiling(self):
        m = _good_metrics(max_drawdown=14.0)
        loss = compute_loss(m, "traot", backtest_days=30, max_dd_param=None)
        assert loss < 0, "Should pass with dd=14 and default ceiling 15"

    def test_max_dd_param_capped_at_20(self):
        m = _good_metrics(max_drawdown=19.0)
        # max_dd_param=30 should be capped to 20 -> 19 < 20 passes
        loss = compute_loss(m, "traot", backtest_days=30, max_dd_param=30)
        assert loss < 0

    def test_max_dd_param_uses_param_when_lower(self):
        m = _good_metrics(max_drawdown=12.0)
        # max_dd_param=10 < 20, so ceiling is 10 -> 12 > 10 -> killed
        loss = compute_loss(m, "traot", backtest_days=30, max_dd_param=10)
        assert loss == MAX_LOSS


# ===================================================================
# TestEngineIntegration
# ===================================================================

class TestEngineIntegration:
    """Tests for BacktestEngine hyperopt compatibility (config_dict, copy)."""

    def test_engine_accepts_config_dict(self):
        """BacktestEngine can be created with a raw config dict."""
        from src.backtesting.engine import BacktestEngine
        import yaml

        with open('config.yaml') as f:
            raw = yaml.safe_load(f)

        engine = BacktestEngine(config_dict=raw)
        assert engine.config is not None
        assert engine.config.raw == raw

    def test_engine_accepts_preloaded_df(self):
        """BacktestEngine can load data without copying."""
        from src.backtesting.engine import BacktestEngine
        import pandas as pd

        engine = BacktestEngine(config_path='config.yaml')
        df = pd.DataFrame({
            'datetime': pd.date_range('2025-01-01', periods=300, freq='15min'),
            'open': [100.0] * 300,
            'high': [101.0] * 300,
            'low': [99.0] * 300,
            'close': [100.5] * 300,
            'volume': [1000.0] * 300,
        })
        result = engine.load_data(df=df, copy=False)
        assert result is True

    def test_config_dict_overrides_signals(self):
        """Config dict should override signal config values."""
        from src.backtesting.engine import BacktestEngine
        import yaml

        with open('config.yaml') as f:
            raw = yaml.safe_load(f)

        raw['signals'] = {
            'risk_per_trade': 0.03,
            'risk_reward_ratio': 3.0,
            'strong_signal': 0.70,
            'medium_signal': 0.60,
            'cooldown_minutes': 30,
        }
        engine = BacktestEngine(config_dict=raw)
        assert engine.config.signals.risk_per_trade == 0.03
        assert engine.config.signals.cooldown_minutes == 30


# ===================================================================
# TestOptimizer
# ===================================================================

import pandas as pd


class TestOptimizer:
    """Tests for src/optimize/optimizer.py."""

    def test_create_objective_returns_callable(self):
        """create_objective should return a callable."""
        from src.optimize.optimizer import create_objective

        objective = create_objective(
            config_path="config.yaml",
            loss_name="traot",
            spaces=["trading"],
        )
        assert callable(objective)

    def test_create_objective_with_preloaded_data(self):
        """create_objective should accept a pre-loaded DataFrame."""
        from src.optimize.optimizer import create_objective

        df = pd.DataFrame({
            "datetime": pd.date_range("2025-01-01", periods=2000, freq="15min"),
            "open": [100.0] * 2000,
            "high": [101.0] * 2000,
            "low": [99.0] * 2000,
            "close": [100.5] * 2000,
            "volume": [1000.0] * 2000,
        })
        objective = create_objective(
            config_path="config.yaml",
            loss_name="traot",
            spaces=["trading"],
            data_df=df,
        )
        assert callable(objective)

    def test_split_train_val(self):
        """Train/val split should preserve total length."""
        from src.optimize.optimizer import split_train_val

        df = pd.DataFrame({"close": range(100)})
        train, val = split_train_val(df, train_ratio=0.7)
        assert len(train) == 70
        assert len(val) == 30

    def test_split_train_val_no_overlap(self):
        """Train and val should not overlap."""
        from src.optimize.optimizer import split_train_val

        df = pd.DataFrame({"close": range(100)})
        train, val = split_train_val(df, train_ratio=0.7)
        assert train["close"].max() < val["close"].min()

    def test_split_train_val_custom_ratio(self):
        """Custom train ratio should work correctly."""
        from src.optimize.optimizer import split_train_val

        df = pd.DataFrame({"close": range(200)})
        train, val = split_train_val(df, train_ratio=0.8)
        assert len(train) == 160
        assert len(val) == 40

    def test_load_historical_data_fallback(self):
        """Should generate synthetic data when no DB available."""
        from src.optimize.optimizer import _load_historical_data

        raw_config = {
            "database": {"path": "nonexistent.db"},
            "data": {"symbol": "TEST"},
        }
        df = _load_historical_data(raw_config)
        assert len(df) == 2000
        assert "datetime" in df.columns
        assert "close" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "volume" in df.columns

    def test_generate_synthetic_data_deterministic(self):
        """Synthetic data with same seed should be identical."""
        from src.optimize.optimizer import _generate_synthetic_data

        df1 = _generate_synthetic_data(seed=42)
        df2 = _generate_synthetic_data(seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_generate_synthetic_data_custom_size(self):
        """Synthetic data generation should respect the n parameter."""
        from src.optimize.optimizer import _generate_synthetic_data

        df = _generate_synthetic_data(n=500)
        assert len(df) == 500

    def test_estimate_backtest_days_with_datetime(self):
        """Should compute days from datetime column."""
        from src.optimize.optimizer import _estimate_backtest_days

        df = pd.DataFrame({
            "datetime": pd.date_range("2025-01-01", periods=960, freq="15min"),
        })
        days = _estimate_backtest_days(df)
        assert 9.0 <= days <= 11.0  # ~10 days of 15-min candles

    def test_estimate_backtest_days_no_datetime(self):
        """Should fall back to heuristic when no datetime column."""
        from src.optimize.optimizer import _estimate_backtest_days

        df = pd.DataFrame({"close": range(960)})
        days = _estimate_backtest_days(df)
        assert days == 10.0  # 960 / 96

    def test_estimate_backtest_days_empty_df(self):
        """Should return 30 for empty DataFrame."""
        from src.optimize.optimizer import _estimate_backtest_days

        df = pd.DataFrame()
        days = _estimate_backtest_days(df)
        assert days == 30.0

    def test_estimate_backtest_days_none(self):
        """Should return 30 for None input."""
        from src.optimize.optimizer import _estimate_backtest_days

        days = _estimate_backtest_days(None)
        assert days == 30.0

    def test_apply_normalized_ensemble(self):
        """Normalized ensemble weights should be written into config."""
        from src.optimize.optimizer import _apply_normalized_ensemble

        config = {}
        params = {
            "weight_lstm": 0.55,
            "weight_fourier": 0.10,
            "weight_kalman": 0.15,
            "weight_markov": 0.10,
            "weight_entropy": 0.05,
            "weight_monte_carlo": 0.05,
        }
        _apply_normalized_ensemble(config, params)
        weights = config["prediction"]["ensemble"]["weights"]
        assert "lstm" in weights
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_apply_normalized_ensemble_no_weight_params(self):
        """Should be a no-op when no weight_ params are present."""
        from src.optimize.optimizer import _apply_normalized_ensemble

        config = {}
        _apply_normalized_ensemble(config, {"confidence_threshold": 0.8})
        assert config == {}

    def test_objective_returns_max_loss_on_failure(self):
        """Objective should return MAX_LOSS when the trial fails."""
        from src.optimize.optimizer import create_objective
        from src.optimize.loss_functions import MAX_LOSS

        # Create objective with data that's too short for BacktestEngine
        short_df = pd.DataFrame({
            "datetime": pd.date_range("2025-01-01", periods=50, freq="15min"),
            "open": [100.0] * 50,
            "high": [101.0] * 50,
            "low": [99.0] * 50,
            "close": [100.5] * 50,
            "volume": [1000.0] * 50,
        })
        objective = create_objective(
            config_path="config.yaml",
            loss_name="traot",
            spaces=["trading"],
            data_df=short_df,
        )
        trial = FakeTrial()
        trial.number = 0
        loss = objective(trial)
        assert loss == MAX_LOSS


# ===================================================================
# TestResults
# ===================================================================

import json
import tempfile
import os


class TestResults:
    """Tests for src/optimize/results.py."""

    def test_save_best_result(self):
        from src.optimize.results import save_best_result

        with tempfile.TemporaryDirectory() as tmpdir:
            result = {
                "trial_number": 42,
                "loss": -0.25,
                "params": {"confidence_threshold": 0.78},
                "optimization_metrics": {"win_rate": 0.65},
            }
            path = save_best_result(result, output_dir=tmpdir)
            assert os.path.exists(path)
            with open(path) as f:
                saved = json.load(f)
            assert saved["trial_number"] == 42

    def test_save_best_result_creates_dir(self):
        from src.optimize.results import save_best_result

        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "nested", "deep")
            path = save_best_result({"test": 1}, output_dir=nested)
            assert os.path.exists(path)

    def test_load_latest_result(self):
        from src.optimize.results import save_best_result, load_latest_result

        with tempfile.TemporaryDirectory() as tmpdir:
            save_best_result({"trial": 1}, output_dir=tmpdir)
            result = load_latest_result(output_dir=tmpdir)
            assert result["trial"] == 1

    def test_load_latest_result_empty_dir(self):
        from src.optimize.results import load_latest_result

        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_latest_result(output_dir=tmpdir)
            assert result == {}

    def test_load_latest_result_nonexistent_dir(self):
        from src.optimize.results import load_latest_result

        result = load_latest_result(output_dir="/nonexistent/path")
        assert result == {}

    def test_export_trials_csv(self):
        from src.optimize.results import export_trials_csv
        import optuna

        study = optuna.create_study()
        study.add_trial(
            optuna.trial.create_trial(
                params={"x": 1.0},
                values=[0.5],
                distributions={
                    "x": optuna.distributions.FloatDistribution(0, 2)
                },
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_trials_csv(study, output_dir=tmpdir)
            assert os.path.exists(path)
            assert path.endswith(".csv")
            # Verify content
            df = pd.read_csv(path)
            assert len(df) == 1
            assert "param_x" in df.columns

    def test_format_results_table(self):
        from src.optimize.results import format_results_table
        import optuna

        study = optuna.create_study(direction="minimize")
        study.add_trial(
            optuna.trial.create_trial(
                params={"x": 1.0},
                values=[-0.5],
                distributions={
                    "x": optuna.distributions.FloatDistribution(0, 2)
                },
                user_attrs={
                    "total_trades": 50,
                    "win_rate": 0.65,
                    "total_pnl_percent": 7.43,
                    "max_drawdown": 4.1,
                },
            )
        )
        table = format_results_table(study, top_n=5)
        assert "Trial" in table
        assert "Loss" in table
        # Should contain the trial data
        assert "-0.5000" in table

    def test_param_registry_populated(self):
        from src.optimize.results import PARAM_REGISTRY

        assert len(PARAM_REGISTRY) == 20  # 10 trading + 6 ensemble + 4 risk
        assert "confidence_threshold" in PARAM_REGISTRY
        assert PARAM_REGISTRY["confidence_threshold"] == "continuous_learning.confidence.trading_threshold"


# ===================================================================
# TestCLI
# ===================================================================


class TestCLI:
    """Tests for src/optimize/hyperopt.py CLI argument parser."""

    def test_parse_args_defaults(self):
        from src.optimize.hyperopt import parse_args

        args = parse_args([])
        assert args.trials == 300
        assert args.loss == "traot"
        assert args.space == "trading"
        assert args.patience == 50
        assert args.config == "config.yaml"
        assert args.dry_run is False
        assert args.apply_best is False
        assert args.show_best is None
        assert args.resume is False
        assert args.download is False
        assert args.seed is None

    def test_parse_args_custom(self):
        from src.optimize.hyperopt import parse_args

        args = parse_args([
            "--trials", "500",
            "--loss", "sharpe",
            "--space", "trading,ensemble",
            "--seed", "42",
        ])
        assert args.trials == 500
        assert args.loss == "sharpe"
        assert args.space == "trading,ensemble"
        assert args.seed == 42

    def test_parse_args_dry_run(self):
        from src.optimize.hyperopt import parse_args

        args = parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_parse_args_show_best(self):
        from src.optimize.hyperopt import parse_args

        args = parse_args(["--show-best", "10"])
        assert args.show_best == 10

    def test_parse_args_apply_best(self):
        from src.optimize.hyperopt import parse_args

        args = parse_args(["--apply-best"])
        assert args.apply_best is True

    def test_parse_args_resume(self):
        from src.optimize.hyperopt import parse_args

        args = parse_args(["--resume"])
        assert args.resume is True

    def test_parse_args_download_with_days(self):
        from src.optimize.hyperopt import parse_args

        args = parse_args(["--download", "--days", "180"])
        assert args.download is True
        assert args.days == 180

    def test_parse_args_symbol_and_timeframe(self):
        from src.optimize.hyperopt import parse_args

        args = parse_args(["--symbol", "ETH/USDT", "--timeframe", "1h"])
        assert args.symbol == "ETH/USDT"
        assert args.timeframe == "1h"


# ===================================================================
# TestEarlyStopping
# ===================================================================


class TestEarlyStopping:
    """Tests for the EarlyStoppingCallback."""

    def test_early_stopping_triggers(self):
        from src.optimize.hyperopt import EarlyStoppingCallback
        import optuna

        callback = EarlyStoppingCallback(patience=3)
        study = optuna.create_study(direction="minimize")

        # Add trials with no improvement — catch RuntimeError from study.stop()
        # which can only be called inside an optimize loop
        for i in range(5):
            study.add_trial(
                optuna.trial.create_trial(
                    params={"x": float(i)},
                    values=[1.0],  # Same value every time
                    distributions={
                        "x": optuna.distributions.FloatDistribution(0, 10)
                    },
                )
            )
            try:
                callback(study, study.trials[-1])
            except RuntimeError:
                pass  # study.stop() raises outside optimize loop

        assert callback._no_improve_count >= 3

    def test_early_stopping_resets_on_improvement(self):
        from src.optimize.hyperopt import EarlyStoppingCallback
        import optuna

        callback = EarlyStoppingCallback(patience=5)
        study = optuna.create_study(direction="minimize")

        # Add improving trial
        study.add_trial(
            optuna.trial.create_trial(
                params={"x": 1.0},
                values=[1.0],
                distributions={
                    "x": optuna.distributions.FloatDistribution(0, 10)
                },
            )
        )
        callback(study, study.trials[-1])
        assert callback._no_improve_count == 0

        # Add worse trial
        study.add_trial(
            optuna.trial.create_trial(
                params={"x": 2.0},
                values=[2.0],
                distributions={
                    "x": optuna.distributions.FloatDistribution(0, 10)
                },
            )
        )
        callback(study, study.trials[-1])
        assert callback._no_improve_count == 1

        # Add better trial — should reset
        study.add_trial(
            optuna.trial.create_trial(
                params={"x": 0.5},
                values=[0.5],
                distributions={
                    "x": optuna.distributions.FloatDistribution(0, 10)
                },
            )
        )
        callback(study, study.trials[-1])
        assert callback._no_improve_count == 0
