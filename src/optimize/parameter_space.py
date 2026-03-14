"""
Parameter Space Definitions
============================
Defines searchable hyperparameter spaces for Optuna-based optimization.

Three spaces:
  - TRADING_PARAMS: Core trading thresholds and risk settings
  - ENSEMBLE_PARAMS: Model ensemble weight allocation
  - RISK_PARAMS: Portfolio-level risk controls
"""

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Parameter space definitions
# ---------------------------------------------------------------------------

TRADING_PARAMS: List[Dict[str, Any]] = [
    {"name": "confidence_threshold", "type": "float", "low": 0.60, "high": 0.95, "step": 0.01,
     "config_path": "continuous_learning.confidence.trading_threshold"},
    {"name": "hysteresis", "type": "float", "low": 0.01, "high": 0.10, "step": 0.01,
     "config_path": "continuous_learning.confidence.hysteresis"},
    {"name": "sl_multiplier", "type": "float", "low": 1.0, "high": 3.0, "step": 0.1,
     "config_path": "prediction.atr.sl_multiplier"},
    {"name": "tp_multiplier", "type": "float", "low": 1.5, "high": 5.0, "step": 0.1,
     "config_path": "prediction.atr.tp_multiplier"},
    {"name": "risk_per_trade", "type": "float", "low": 0.01, "high": 0.05, "step": 0.005,
     "config_path": "signals.risk_per_trade"},
    {"name": "risk_reward_ratio", "type": "float", "low": 1.0, "high": 4.0, "step": 0.1,
     "config_path": "signals.risk_reward_ratio"},
    {"name": "cooldown_minutes", "type": "int", "low": 15, "high": 240, "step": 15,
     "config_path": "signals.cooldown_minutes"},
    {"name": "strong_signal", "type": "float", "low": 0.55, "high": 0.80, "step": 0.01,
     "config_path": "signals.strong_signal"},
    {"name": "regime_choppy_penalty", "type": "float", "low": 0.80, "high": 1.0, "step": 0.01,
     "config_path": "prediction.regime_penalties.CHOPPY"},
    {"name": "regime_volatile_penalty", "type": "float", "low": 0.75, "high": 1.0, "step": 0.01,
     "config_path": "prediction.regime_penalties.VOLATILE"},
]

ENSEMBLE_PARAMS: List[Dict[str, Any]] = [
    {"name": "weight_lstm", "type": "float", "low": 0.40, "high": 0.70,
     "config_path": "prediction.ensemble.weights.lstm"},
    {"name": "weight_fourier", "type": "float", "low": 0.05, "high": 0.20,
     "config_path": "prediction.ensemble.weights.fourier"},
    {"name": "weight_kalman", "type": "float", "low": 0.05, "high": 0.25,
     "config_path": "prediction.ensemble.weights.kalman"},
    {"name": "weight_markov", "type": "float", "low": 0.05, "high": 0.15,
     "config_path": "prediction.ensemble.weights.markov"},
    {"name": "weight_entropy", "type": "float", "low": 0.02, "high": 0.10,
     "config_path": "prediction.ensemble.weights.entropy"},
    {"name": "weight_monte_carlo", "type": "float", "low": 0.02, "high": 0.10,
     "config_path": "prediction.ensemble.weights.monte_carlo"},
]

RISK_PARAMS: List[Dict[str, Any]] = [
    {"name": "max_drawdown_percent", "type": "int", "low": 10, "high": 30,
     "config_path": "risk.max_drawdown_percent"},
    {"name": "daily_loss_limit", "type": "float", "low": 0.02, "high": 0.10,
     "config_path": "risk.daily_loss_limit"},
    {"name": "max_position_percent", "type": "float", "low": 0.10, "high": 0.40,
     "config_path": "risk.max_position_percent"},
    {"name": "kelly_fraction", "type": "float", "low": 0.2, "high": 1.0,
     "config_path": "position_sizing.kelly_fraction"},
]

_SPACE_MAP: Dict[str, List[Dict[str, Any]]] = {
    "trading": TRADING_PARAMS,
    "ensemble": ENSEMBLE_PARAMS,
    "risk": RISK_PARAMS,
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_parameter_space(spaces: List[str]) -> List[Dict[str, Any]]:
    """Return the combined list of parameter definitions for the requested spaces.

    Args:
        spaces: List of space names, e.g. ``["trading", "ensemble"]``.

    Returns:
        Flat list of parameter dicts from all requested spaces.

    Raises:
        ValueError: If an unknown space name is provided.
    """
    combined: List[Dict[str, Any]] = []
    for name in spaces:
        key = name.lower()
        if key not in _SPACE_MAP:
            raise ValueError(
                f"Unknown parameter space '{name}'. "
                f"Available: {list(_SPACE_MAP.keys())}"
            )
        combined.extend(_SPACE_MAP[key])
    return combined


def suggest_params(trial: Any, spaces: List[str]) -> Dict[str, Any]:
    """Use an Optuna *trial* to suggest values for every parameter in *spaces*.

    Args:
        trial: An ``optuna.trial.Trial`` (or compatible mock).
        spaces: List of space names to sample from.

    Returns:
        Dict mapping parameter name to suggested value.
    """
    params: Dict[str, Any] = {}
    for p in get_parameter_space(spaces):
        if p["type"] == "int":
            params[p["name"]] = trial.suggest_int(
                p["name"], p["low"], p["high"], step=p.get("step", 1),
            )
        else:
            # float — step is optional
            kwargs: Dict[str, Any] = {"name": p["name"], "low": p["low"], "high": p["high"]}
            if "step" in p:
                kwargs["step"] = p["step"]
            params[p["name"]] = trial.suggest_float(**kwargs)
    return params


def apply_params_to_config(raw_config: dict, params: Dict[str, Any]) -> None:
    """Write suggested parameter values into a nested config dict in-place.

    Each parameter is looked up in the combined space definitions to find its
    ``config_path`` (dot-separated).  Intermediate dicts are created if they
    do not already exist.

    Args:
        raw_config: The mutable nested config dictionary (e.g. ``config.raw``).
        params: Dict of ``{param_name: value}`` as returned by :func:`suggest_params`.
    """
    # Build a lookup: param_name -> config_path
    all_params = get_parameter_space(list(_SPACE_MAP.keys()))
    path_lookup = {p["name"]: p["config_path"] for p in all_params}

    for name, value in params.items():
        config_path = path_lookup.get(name)
        if config_path is None:
            continue
        keys = config_path.split(".")
        node = raw_config
        for key in keys[:-1]:
            if key not in node:
                node[key] = {}
            node = node[key]
        node[keys[-1]] = value


def normalize_ensemble_weights(raw_weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize ensemble weights so they sum to 1.0 with an LSTM floor of 0.40.

    Algorithm:
        1. Normalize all weights proportionally so they sum to 1.0.
        2. If lstm < 0.40 after step 1, set lstm to 0.40 and redistribute
           the remaining 0.60 among the other models proportionally.

    Args:
        raw_weights: Dict like ``{"lstm": 0.55, "fourier": 0.10, ...}``.

    Returns:
        New dict with the same keys, values summing to 1.0, lstm >= 0.40.
    """
    total = sum(raw_weights.values())
    if total == 0:
        # Edge case: all zeros — give lstm the floor, split rest equally
        n_others = len(raw_weights) - 1
        result = {k: (0.60 / n_others if k != "lstm" else 0.40) for k in raw_weights}
        return result

    # Step 1: proportional normalization
    normalized = {k: v / total for k, v in raw_weights.items()}

    # Step 2: enforce LSTM floor
    lstm_key = "lstm"
    if lstm_key in normalized and normalized[lstm_key] < 0.40:
        normalized[lstm_key] = 0.40
        others_total = sum(v for k, v in normalized.items() if k != lstm_key)
        remaining = 0.60
        if others_total > 0:
            for k in normalized:
                if k != lstm_key:
                    normalized[k] = (normalized[k] / others_total) * remaining
        else:
            n_others = len(normalized) - 1
            for k in normalized:
                if k != lstm_key:
                    normalized[k] = remaining / n_others

    return normalized
