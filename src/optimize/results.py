"""
Results Storage and Export
==========================
Functions for saving, loading, and formatting hyperopt results.

- save_best_result: Save best trial as timestamped JSON
- load_latest_result: Load most recent result JSON
- apply_best_to_config: Write best params into config.yaml (with backup)
- export_trials_csv: Export all trials to CSV
- format_results_table: Pretty-print top N trials
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.optimize.parameter_space import get_parameter_space, _SPACE_MAP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Build param_name -> config_path lookup from all spaces
# ---------------------------------------------------------------------------

def _build_param_registry() -> Dict[str, str]:
    """Build a mapping of param_name -> config_path from all defined spaces."""
    all_params = get_parameter_space(list(_SPACE_MAP.keys()))
    return {p["name"]: p["config_path"] for p in all_params}


PARAM_REGISTRY: Dict[str, str] = _build_param_registry()


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


def save_best_result(
    result_dict: dict,
    output_dir: str = "data/hyperopt",
) -> str:
    """Save a result dict as a timestamped JSON file.

    Args:
        result_dict: Arbitrary dict to persist (trial number, params, metrics).
        output_dir: Directory to write into (created if missing).

    Returns:
        Absolute path to the saved JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(result_dict, f, indent=2, default=str)

    logger.info("Saved best result to %s", filepath)
    return filepath


def load_latest_result(output_dir: str = "data/hyperopt") -> dict:
    """Load the most recent ``results_*.json`` from *output_dir*.

    Returns:
        The parsed dict, or an empty dict if no results exist.
    """
    dirpath = Path(output_dir)
    if not dirpath.exists():
        return {}

    result_files = sorted(dirpath.glob("results_*.json"))
    if not result_files:
        return {}

    latest = result_files[-1]
    with open(latest) as f:
        data = json.load(f)

    logger.info("Loaded latest result from %s", latest)
    return data


# ---------------------------------------------------------------------------
# Apply best params to YAML config
# ---------------------------------------------------------------------------


def apply_best_to_config(config_path: str, params: dict) -> None:
    """Write optimised parameters into a YAML config file.

    Steps:
        1. Back up *config_path* to ``config_path.backup``.
        2. Read the YAML (preserving comments with ruamel.yaml if available).
        3. Walk each param's dot-path and update the value.
        4. Write back.

    Args:
        config_path: Path to the YAML configuration file.
        params: Dict of ``{param_name: value}`` to apply.
    """
    backup_path = config_path + ".backup"

    # Create backup
    import shutil
    shutil.copy2(config_path, backup_path)
    logger.info("Backed up config to %s", backup_path)

    # Try ruamel.yaml first for comment preservation, fallback to pyyaml
    try:
        from ruamel.yaml import YAML

        yaml = YAML()
        yaml.preserve_quotes = True
        with open(config_path) as f:
            config = yaml.load(f)

        _apply_params_in_place(config, params)

        with open(config_path, "w") as f:
            yaml.dump(config, f)

    except ImportError:
        import yaml as pyyaml

        with open(config_path) as f:
            config = pyyaml.safe_load(f)

        _apply_params_in_place(config, params)

        with open(config_path, "w") as f:
            pyyaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info("Applied %d parameters to %s", len(params), config_path)


def _apply_params_in_place(config: dict, params: dict) -> None:
    """Walk dot-paths from PARAM_REGISTRY and update values in *config*."""
    for name, value in params.items():
        config_path = PARAM_REGISTRY.get(name)
        if config_path is None:
            logger.warning("Unknown param '%s' — skipping", name)
            continue

        keys = config_path.split(".")
        node = config
        for key in keys[:-1]:
            if key not in node:
                node[key] = {}
            node = node[key]

        old_value = node.get(keys[-1], "<unset>")
        node[keys[-1]] = value
        logger.info("  %s: %s -> %s", name, old_value, value)


# ---------------------------------------------------------------------------
# Export trials to CSV
# ---------------------------------------------------------------------------


def export_trials_csv(
    study: Any,
    output_dir: str = "data/hyperopt",
) -> str:
    """Export all completed trials from an Optuna study to a CSV file.

    Args:
        study: An ``optuna.study.Study`` instance.
        output_dir: Directory to write the CSV into.

    Returns:
        Path to the saved CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trials_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)

    # Build rows from trials
    rows: List[Dict[str, Any]] = []
    for trial in study.trials:
        row: Dict[str, Any] = {
            "number": trial.number,
            "value": trial.value,
            "state": str(trial.state),
        }
        # Add params
        for k, v in trial.params.items():
            row[f"param_{k}"] = v
        # Add user attrs (metrics set by objective)
        for k, v in trial.user_attrs.items():
            row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    logger.info("Exported %d trials to %s", len(rows), filepath)
    return filepath


# ---------------------------------------------------------------------------
# Formatted results table
# ---------------------------------------------------------------------------


def format_results_table(study: Any, top_n: int = 10) -> str:
    """Return a formatted string showing the top N trials.

    Expected user_attrs on each trial (set by the objective function):
        total_trades, win_rate, total_pnl_percent, max_drawdown

    Args:
        study: An ``optuna.study.Study`` instance.
        top_n: Number of top trials to display.

    Returns:
        Multi-line formatted table string.
    """
    # Sort trials by value (lower is better for minimisation)
    completed = [
        t for t in study.trials
        if t.state.name == "COMPLETE" and t.value is not None
    ]
    completed.sort(key=lambda t: t.value)

    best_number = completed[0].number if completed else -1

    lines: List[str] = []
    header = f"{'Trial':>6}  {'Best':>4}  {'Trades':>6}  {'WR':>6}  {'P&L':>10}  {'MaxDD':>6}  {'Loss':>10}"
    separator = "\u2500" * len(header)
    lines.append(header)
    lines.append(separator)

    for trial in completed[:top_n]:
        ua = trial.user_attrs
        star = "\u2605" if trial.number == best_number else " "
        trades = ua.get("total_trades", "?")
        wr = ua.get("win_rate", 0)
        pnl = ua.get("total_pnl_percent", 0)
        dd = ua.get("max_drawdown", 0)
        loss = trial.value

        # Format win_rate: if stored as fraction convert to percent
        wr_str = f"{wr * 100:.0f}%" if isinstance(wr, float) and wr <= 1.0 else f"{wr}%"

        lines.append(
            f"{trial.number:>6}  {star:>4}  {trades:>6}  {wr_str:>6}  "
            f"{pnl:>+9.2f}%  {dd:>5.1f}%  {loss:>10.4f}"
        )

    return "\n".join(lines)
