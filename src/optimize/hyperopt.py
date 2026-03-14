"""
Traot Hyperparameter Optimization — CLI Orchestrator
======================================================
Main entry point for running Optuna-based hyperparameter search.

Usage:
    python -m src.optimize.hyperopt --trials 300 --loss traot --space trading
    python -m src.optimize.hyperopt --dry-run
    python -m src.optimize.hyperopt --show-best 10
    python -m src.optimize.hyperopt --apply-best
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the hyperopt CLI.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(
        description="Traot Hyperparameter Optimization",
    )

    # Optimisation settings
    parser.add_argument(
        "--trials", type=int, default=300,
        help="Number of optimisation trials (default: 300)",
    )
    parser.add_argument(
        "--loss", type=str, default="traot",
        choices=["traot", "sharpe", "sortino", "calmar", "profit"],
        help="Loss function to minimise (default: traot)",
    )
    parser.add_argument(
        "--space", type=str, default="trading",
        help="Comma-separated parameter spaces: trading,ensemble,risk (default: trading)",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )

    # Data options
    parser.add_argument(
        "--symbol", type=str, default=None,
        help="Trading pair override, e.g. BTC/USDT",
    )
    parser.add_argument(
        "--timeframe", type=str, default=None,
        help="Candle timeframe override, e.g. 15m, 1h",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download historical data via CCXT before optimising",
    )
    parser.add_argument(
        "--days", type=int, default=90,
        help="Days of data to download (default: 90)",
    )

    # Execution modes
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from SQLite-backed study (data/hyperopt/hyperopt.db)",
    )
    parser.add_argument(
        "--show-best", type=int, default=None, dest="show_best",
        help="Show top N results from the stored study",
    )
    parser.add_argument(
        "--apply-best", action="store_true", dest="apply_best",
        help="Apply the best result to config.yaml",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Run a single trial with current config (sanity check)",
    )

    # Performance / reproducibility
    parser.add_argument(
        "--jobs", type=int, default=-1,
        help="Number of parallel jobs (currently forced to 1)",
    )
    parser.add_argument(
        "--patience", type=int, default=50,
        help="Early-stop after N trials without improvement (default: 50)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Early Stopping Callback
# ---------------------------------------------------------------------------


class EarlyStoppingCallback:
    """Stop the study when no improvement is seen for *patience* trials."""

    def __init__(self, patience: int = 50) -> None:
        self._patience = patience
        self._best_value: Optional[float] = None
        self._no_improve_count: int = 0

    def __call__(self, study, trial) -> None:
        current_best = study.best_value
        if self._best_value is None or current_best < self._best_value:
            self._best_value = current_best
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1

        if self._no_improve_count >= self._patience:
            logger.info(
                "Early stopping: no improvement for %d trials", self._patience
            )
            study.stop()


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_hyperopt(args: argparse.Namespace) -> None:
    """Run the full hyperparameter optimisation loop.

    Steps:
        1. Parse spaces, print banner
        2. Load / download data
        3. Split train / val (70/30)
        4. Create Optuna study
        5. Run optimisation with early stopping
        6. Print results table
        7. Validate best params on held-out data
        8. Save results + export CSV
    """
    import optuna
    from optuna.samplers import TPESampler

    from src.optimize.optimizer import (
        create_objective,
        download_historical_data,
        split_train_val,
        _load_historical_data,
    )
    from src.optimize.parameter_space import get_parameter_space
    from src.optimize.results import (
        export_trials_csv,
        format_results_table,
        save_best_result,
    )
    from src.core.config import load_config

    # 1. Parse spaces
    spaces = [s.strip() for s in args.space.split(",")]
    param_count = len(get_parameter_space(spaces))

    print(f"\nTraot Hyperopt \u2014 Optimizing {param_count} parameters")
    print(f"  Spaces : {', '.join(spaces)}")
    print(f"  Loss   : {args.loss}")
    print(f"  Trials : {args.trials}")
    print(f"  Patience: {args.patience}")
    print()

    # 2. Load or download data
    if args.download:
        raw_config = load_config(args.config)
        symbol = args.symbol or raw_config.get("data", {}).get("symbol", "BTC/USDT")
        timeframe = args.timeframe or raw_config.get("data", {}).get("timeframe", "15m")
        data_df = download_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            days=args.days,
        )
    else:
        raw_config = load_config(args.config)
        data_df = _load_historical_data(raw_config, args.symbol, args.timeframe)

    print(f"  Data   : {len(data_df)} candles")

    # 3. Split train / val
    train_df, val_df = split_train_val(data_df, train_ratio=0.7)
    print(f"  Train  : {len(train_df)} candles")
    print(f"  Val    : {len(val_df)} candles")
    print()

    # 4. Create Optuna study
    if args.resume:
        db_dir = Path("data/hyperopt")
        db_dir.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///data/hyperopt/hyperopt.db"
    else:
        storage = None

    sampler = TPESampler(n_startup_trials=30, seed=args.seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="traot_hyperopt",
        storage=storage,
        load_if_exists=args.resume,
    )

    # 5. Create objective and run
    objective = create_objective(
        config_path=args.config,
        loss_name=args.loss,
        spaces=spaces,
        symbol=args.symbol,
        timeframe=args.timeframe,
        data_df=train_df,
        max_dd_param_active="risk" in spaces,
    )

    early_stop = EarlyStoppingCallback(patience=args.patience)

    # Suppress Optuna's trial-level logging for cleaner output
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("Starting optimisation...")
    study.optimize(
        objective,
        n_trials=args.trials,
        callbacks=[early_stop],
        n_jobs=1,  # Must be 1: BacktestEngine is not thread-safe
    )

    # 6. Print results table
    print("\n" + format_results_table(study, top_n=10))

    # 7. Validate best params on held-out data
    print("\nValidating best params on held-out data...")
    val_objective = create_objective(
        config_path=args.config,
        loss_name=args.loss,
        spaces=spaces,
        symbol=args.symbol,
        timeframe=args.timeframe,
        data_df=val_df,
        max_dd_param_active="risk" in spaces,
    )

    # Create a fixed trial with best params for validation
    best_params = study.best_params
    val_study = optuna.create_study(direction="minimize")

    # Use the best params in a new trial via enqueue
    val_study.enqueue_trial(best_params)
    val_study.optimize(val_objective, n_trials=1)

    train_loss = study.best_value
    val_loss = val_study.best_value
    print(f"  Train loss : {train_loss:.4f}")
    print(f"  Val loss   : {val_loss:.4f}")

    if val_loss > train_loss * 1.5:
        print("  WARNING: Possible overfitting — validation loss is significantly worse")
    else:
        print("  Validation looks healthy")

    # 8. Save results
    result = {
        "trial_number": study.best_trial.number,
        "loss": study.best_value,
        "params": study.best_params,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "n_trials_completed": len(study.trials),
        "loss_function": args.loss,
        "spaces": spaces,
    }
    # Include user_attrs if available
    if study.best_trial.user_attrs:
        result["optimization_metrics"] = dict(study.best_trial.user_attrs)

    result_path = save_best_result(result)
    csv_path = export_trials_csv(study)

    print(f"\nResults saved to {result_path}")
    print(f"Trials CSV saved to {csv_path}")
    print("\nRun --apply-best to update config.yaml with these parameters")


# ---------------------------------------------------------------------------
# Utility modes
# ---------------------------------------------------------------------------


def run_dry_run(args: argparse.Namespace) -> None:
    """Run a single evaluation with the current config (sanity check)."""
    from src.optimize.optimizer import (
        create_objective,
        _load_historical_data,
    )
    from src.core.config import load_config

    spaces = [s.strip() for s in args.space.split(",")]

    print("\nTraot Hyperopt \u2014 Dry Run")
    print("Evaluating current config with a single trial...\n")

    raw_config = load_config(args.config)
    data_df = _load_historical_data(raw_config, args.symbol, args.timeframe)

    objective = create_objective(
        config_path=args.config,
        loss_name=args.loss,
        spaces=spaces,
        data_df=data_df,
    )

    # Use a FakeTrial that returns midpoints
    class _MidpointTrial:
        """Trial mock that suggests midpoint values."""
        number = 0

        def suggest_float(self, name, low, high, *, step=None):
            return (low + high) / 2

        def suggest_int(self, name, low, high, *, step=1):
            return (low + high) // 2

    trial = _MidpointTrial()
    loss = objective(trial)

    from src.optimize.loss_functions import MAX_LOSS
    if loss >= MAX_LOSS:
        print(f"  Loss: {loss} (MAX_LOSS \u2014 trial killed by guardrails)")
    else:
        print(f"  Loss: {loss:.4f}")

    print("\nDry run complete.")


def show_best(args: argparse.Namespace) -> None:
    """Load and display the top N trials from the SQLite study."""
    import optuna

    from src.optimize.results import format_results_table

    db_path = Path("data/hyperopt/hyperopt.db")
    if not db_path.exists():
        print("No study database found at data/hyperopt/hyperopt.db")
        print("Run an optimisation with --resume first to create one.")
        return

    storage = f"sqlite:///data/hyperopt/hyperopt.db"
    study = optuna.load_study(study_name="traot_hyperopt", storage=storage)

    top_n = args.show_best or 10
    print(f"\nTop {top_n} trials:\n")
    print(format_results_table(study, top_n=top_n))

    # Also print best params
    print(f"\nBest trial #{study.best_trial.number} params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


def apply_best(args: argparse.Namespace) -> None:
    """Apply the latest best result to the config file."""
    from src.optimize.results import apply_best_to_config, load_latest_result

    result = load_latest_result()
    if not result:
        print("No results found in data/hyperopt/")
        print("Run an optimisation first.")
        return

    params = result.get("params", {})
    if not params:
        print("No params found in latest result.")
        return

    print(f"\nApplying {len(params)} parameters to {args.config}:")
    apply_best_to_config(args.config, params)
    print(f"\nConfig updated. Backup saved to {args.config}.backup")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    if args.show_best is not None:
        show_best(args)
    elif args.apply_best:
        apply_best(args)
    elif args.dry_run:
        run_dry_run(args)
    else:
        run_hyperopt(args)


if __name__ == "__main__":
    main()
