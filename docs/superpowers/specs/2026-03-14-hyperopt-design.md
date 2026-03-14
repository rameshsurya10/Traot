# Traot Hyperopt — Design Specification

**Date:** 2026-03-14
**Status:** Approved
**Scope:** Optuna-based hyperparameter optimization for trading parameters

---

## 1. Architecture

### File Structure

```
src/optimize/
├── __init__.py
├── hyperopt.py          # Main orchestrator + CLI entry point
├── parameter_space.py   # Parameter definitions & Optuna search spaces
├── optimizer.py         # Trial evaluation, backtesting integration
├── loss_functions.py    # Loss function registry (traot, sharpe, sortino, calmar, profit)
└── results.py           # Result storage, export, config application
```

### Integration Points

- **config.yaml** — Base configuration. Params accessed via `config.raw` dict (not the typed `Config` dataclass, which only covers a subset of sections). Hyperopt deep-copies `config.raw`, overrides values per trial, and passes the modified dict to the backtest engine.
- **src/backtesting/engine.py** — Backtest engine. Minor modifications needed: add a constructor overload that accepts a raw config dict and a pre-loaded DataFrame to avoid re-reading from disk per trial.
- **src/advanced_predictor.py** — Existing signal generation (used by backtest engine internally).
- **data/hyperopt.db** — Optuna study storage (SQLite, supports resume).
- **data/hyperopt/** — Results directory.

### Config Access Model

The typed `Config` dataclass only covers top-level sections (`signals`, `data`, `analysis`, `model`). Most hyperopt parameters live in sections accessible only via `config.raw` (`continuous_learning`, `prediction`, `risk`, `position_sizing`). Therefore:

- All config reads use dot-path traversal on the raw YAML dict (e.g., `raw['prediction']['atr']['sl_multiplier']`)
- `--apply-best` writes directly to the YAML file using `ruamel.yaml` to preserve comments and formatting
- The `Config` dataclass is not modified or extended

### Engine Modifications (Minimal)

The `BacktestEngine` needs two small changes to support efficient hyperopt:

1. **Accept raw config dict in constructor** — Currently calls `Config.load(path)`. Add overload: `BacktestEngine(config_dict=raw_dict, df=preloaded_df)` to skip file I/O.
2. **Accept pre-loaded DataFrame** — Currently copies `df` on each call. Add a `shared_data=True` flag that skips the copy for read-only backtests.

These are additive changes — existing API unchanged.

### Parallel Execution Safety

Each parallel worker creates its own `BacktestEngine` instance with its own mutable state (`current_index`, `brokerage`). The historical DataFrame is shared read-only across workers (no copies). Workers do not share any mutable state.

---

## 2. Parameter Space (~20 params)

### Trading Space (10 params) — Always optimized

| Parameter | Type | Range | Step | Config Path |
|-----------|------|-------|------|-------------|
| confidence_threshold | Float | 0.60 – 0.95 | 0.01 | `continuous_learning.confidence.trading_threshold` |
| hysteresis | Float | 0.01 – 0.10 | 0.01 | `continuous_learning.confidence.hysteresis` |
| sl_multiplier | Float | 1.0 – 3.0 | 0.1 | `prediction.atr.sl_multiplier` |
| tp_multiplier | Float | 1.5 – 5.0 | 0.1 | `prediction.atr.tp_multiplier` |
| risk_per_trade | Float | 0.01 – 0.05 | 0.005 | `signals.risk_per_trade` |
| risk_reward_ratio | Float | 1.0 – 4.0 | 0.1 | `signals.risk_reward_ratio` |
| cooldown_minutes | Int | 15 – 240 | 15 | `signals.cooldown_minutes` |
| strong_signal | Float | 0.55 – 0.80 | 0.01 | `signals.strong_signal` |
| regime_choppy_penalty | Float | 0.80 – 1.0 | 0.01 | `prediction.regime_penalties.CHOPPY` |
| regime_volatile_penalty | Float | 0.75 – 1.0 | 0.01 | `prediction.regime_penalties.VOLATILE` |

### Ensemble Space (6 params) — Optional (`--space trading,ensemble`)

| Parameter | Type | Range | Constraint |
|-----------|------|-------|------------|
| weight_lstm | Float | 0.40 – 0.70 | Normalized to sum=1.0, min 40% after normalization |
| weight_fourier | Float | 0.05 – 0.20 | |
| weight_kalman | Float | 0.05 – 0.25 | |
| weight_markov | Float | 0.05 – 0.15 | |
| weight_entropy | Float | 0.02 – 0.10 | |
| weight_monte_carlo | Float | 0.02 – 0.10 | |

Weights are normalized after Optuna suggests raw values: `w_i = raw_i / sum(raw)`.

**LSTM floor constraint:** After normalization, if `weight_lstm < 0.40`, redistribute: set lstm to 0.40 and scale others proportionally. This ensures the primary ML signal retains dominance, consistent with the design principle that "math algos average ~0.50" and should not outweigh the trained model.

### Risk Space (4 params) — Optional (`--space trading,risk`)

| Parameter | Type | Range | Config Path |
|-----------|------|-------|-------------|
| max_drawdown_percent | Int | 10 – 30 | `risk.max_drawdown_percent` |
| daily_loss_limit | Float | 0.02 – 0.10 | `risk.daily_loss_limit` |
| max_position_percent | Float | 0.10 – 0.40 | `risk.max_position_percent` |
| kelly_fraction | Float | 0.2 – 1.0 | `position_sizing.kelly_fraction` |

---

## 3. Loss Functions

### Default: `traot` (Loss-Averse)

Designed to minimize losses above all else. Capital preservation is the priority.

```python
# trade_quality: avg_win_pnl / abs(avg_loss_pnl)
# Fallback: if avg_loss == 0 (all wins), trade_quality = 10.0 (capped)
trade_quality = min(10.0, avg_win / abs(avg_loss)) if avg_loss != 0 else 10.0

score = (
    win_rate ** 2                    # Heavily reward high win rate
    * (1 - max_drawdown) ** 3        # Severely punish drawdown
    * profit_factor                   # Wins must outweigh losses in size
    * trade_quality                   # Each win should cover multiple losses
)
loss = -1 * score
```

**Hard-kill conditions (return MAX_LOSS = 100,000):**

| Condition | Threshold | Notes |
|-----------|-----------|-------|
| Max drawdown | > `trial_max_dd_pct` or > 20% (whichever is lower) | Dynamic when risk space active; fixed 15% when risk space inactive |
| Win rate | < 50% | Must win more than lose |
| Trade count | < 10 | Avoid "never trade" solutions |
| Largest single loss | > 5.0 (percentage points, i.e., 5%) | No oversized losers. Compared against `pnl_percent` which is stored as e.g. 5.0 = 5% |
| Profit factor | < 1.0 | Gross wins must exceed gross losses |

Note on drawdown hard-kill: When the risk space is active, the trial's own `max_drawdown_percent` parameter defines the acceptable ceiling. When risk space is not active, a fixed 15% ceiling applies. Either way, absolute cap is 20%.

### Alternative Loss Functions

| Name | Formula | Best For |
|------|---------|----------|
| `sharpe` | `-1 * (mean_return / std_return) * sqrt(N)` | Balanced risk/reward |
| `sortino` | `-1 * (mean_return / downside_deviation) * sqrt(N)` | Allowing upside volatility |
| `calmar` | `-1 * (total_return / max_drawdown)` | Drawdown-sensitive |
| `profit` | `-1 * total_pnl_percent` | Pure profit maximization |

**Annualization factor `N`:** Computed dynamically based on actual trade frequency:
`N = trades_per_year = trade_count / (backtest_days / 365)`. This avoids the incorrect `sqrt(252)` assumption which is only valid for daily returns. If trade count is too low for meaningful annualization (< 20), fall back to raw (non-annualized) ratio.

All loss functions share the same hard-kill guardrails.

---

## 4. Optimization Engine

### Trial Evaluation Flow

```
For each trial (1 to N):
  1. Optuna TPE sampler suggests parameter values
  2. Deep-copy config.raw dict, override with suggested params
  3. Create new BacktestEngine instance with modified config + shared data ref
  4. Run backtest on pre-loaded historical data
  5. Extract results: trades, P&L, drawdown, win rate
  6. Pass through hard-kill checks
  7. Compute loss via selected loss function
  8. Report loss to Optuna (feedback loop)
  9. Optuna learns, suggests smarter params next trial
```

### Performance Optimizations

- **Data loaded once** — Historical candles loaded into a pandas DataFrame once, shared read-only across all workers
- **Per-worker engine instances** — Each parallel worker creates its own `BacktestEngine` with separate mutable state, avoiding thread-safety issues
- **No pruning** — Backtest runs are fast enough (< 1 second for 90 days of 15m data) that pruning overhead isn't worth the complexity. Optuna's pruner requires mid-backtest loss reporting which would require significant engine changes. Instead, rely on hard-kills to reject bad trials quickly.
- **Parallel execution** — `joblib.Parallel` with `n_jobs` workers (default: CPU count - 1). Each worker is a separate process with its own engine instance.
- **Deduplication** — Identical parameter sets skipped
- **Resume** — Optuna study in SQLite, restart from where left off

### Early Stopping

- If best loss hasn't improved in `patience` trials (default: 50) → stop
- Implemented via Optuna callback checking `study.best_trial` stagnation
- Configurable via `--patience N`

### Optuna Configuration

- Sampler: `TPESampler(n_startup_trials=30, seed=None)` — 30 random trials, then smart sampling
- Direction: `minimize` (lower loss = better)
- Storage: `sqlite:///data/hyperopt.db`
- Seed: configurable via `--seed N` for reproducibility

---

## 5. Data Pipeline

### Mode 1: Existing Database (Default)

Reads candles from `trading.db` collected during live trading.

### Mode 2: Download Fresh Data

```bash
python -m src.optimize.hyperopt --download --days 90
```

Downloads historical candles from Binance via CCXT. Stores as Parquet
in `data/hyperopt_candles/`.

### Train/Validation Split

- First 70% of data → optimization (Optuna trains on this)
- Last 30% → validation (verify not overfit)

After optimization, best params re-tested on validation set:

```
Best Trial #247:
  Optimization:  +12.3% P&L, 68% WR, 4.2% max DD, Sharpe 2.1
  Validation:    +8.7% P&L, 64% WR, 5.1% max DD, Sharpe 1.8
  → Validation confirms results hold. Safe to apply.
```

**Overfit detection:** Compare validation Sharpe to optimization Sharpe. If ratio < 0.5 (validation Sharpe is less than half of optimization), warn: "Likely overfit. Consider more data or fewer parameters."

---

## 6. CLI Interface

```bash
# Basic — 300 trials, traot loss function
python -m src.optimize.hyperopt --trials 300

# Dry run — evaluate current config as single trial (no optimization)
python -m src.optimize.hyperopt --dry-run

# Specify loss function
python -m src.optimize.hyperopt --trials 300 --loss sharpe

# Optimize specific spaces
python -m src.optimize.hyperopt --trials 200 --space trading
python -m src.optimize.hyperopt --trials 500 --space trading,ensemble
python -m src.optimize.hyperopt --trials 500 --space trading,ensemble,risk

# Download data first
python -m src.optimize.hyperopt --download --days 90 --trials 300

# Specify symbol and timeframe
python -m src.optimize.hyperopt --symbol BTC/USDT --timeframe 15m --trials 300

# Resume interrupted run
python -m src.optimize.hyperopt --resume

# Show top N results
python -m src.optimize.hyperopt --show-best 10

# Apply best params to config.yaml (backs up original)
python -m src.optimize.hyperopt --apply-best

# Parallel workers
python -m src.optimize.hyperopt --trials 300 --jobs 4

# Early stopping patience
python -m src.optimize.hyperopt --trials 500 --patience 75

# Reproducible seed
python -m src.optimize.hyperopt --trials 300 --seed 42
```

### Output Format

```
Traot Hyperopt — Optimizing 20 parameters
Loss: traot | Data: 90 days BTC/USDT @ 15m | Workers: 4

Trial   Best    Trades  WR      P&L       MaxDD   Loss
─────────────────────────────────────────────────────────
  1           42      52%     +3.21%    8.2%    -0.0412
  3     ★     51      64%     +7.43%    4.1%    -0.1823
 247    ★     45      68%     +12.3%    4.2%    -0.2941

Early stopping: no improvement in 50 trials.
Best params → data/hyperopt/results_20260314_143022.json
```

---

## 7. Result Storage

### Directory Structure

```
data/hyperopt/
├── hyperopt.db                        # Optuna study (SQLite)
├── results_YYYYMMDD_HHMMSS.json       # Best params export
└── trials.csv                         # All trials for analysis
```

### Best Params JSON Format

```json
{
  "trial_number": 247,
  "loss": -0.2941,
  "loss_function": "traot",
  "optimization_metrics": {
    "total_pnl": 12.3,
    "win_rate": 0.68,
    "max_drawdown": 0.042,
    "trade_count": 45,
    "profit_factor": 2.1,
    "sharpe_ratio": 2.1
  },
  "validation_metrics": {
    "total_pnl": 8.7,
    "win_rate": 0.64,
    "max_drawdown": 0.051,
    "trade_count": 18,
    "profit_factor": 1.8,
    "sharpe_ratio": 1.8
  },
  "params": {
    "confidence_threshold": 0.78,
    "sl_multiplier": 1.8,
    "tp_multiplier": 3.2,
    "risk_per_trade": 0.015,
    "cooldown_minutes": 45
  },
  "data": {
    "symbol": "BTC/USDT",
    "timeframe": "15m",
    "days": 90,
    "optimization_period": "2025-12-15 to 2026-02-13",
    "validation_period": "2026-02-13 to 2026-03-14"
  },
  "timestamp": "2026-03-14T14:30:22Z"
}
```

### Apply Best Params

`--apply-best` reads the latest results JSON and:
1. Backs up `config.yaml` → `config.yaml.backup`
2. Updates each parameter at its YAML path using `ruamel.yaml` (preserves comments)
3. Logs every change: `confidence_threshold: 0.80 → 0.78`

---

## 8. Dependencies

- `optuna` — Optimization engine
- `joblib` — Parallel execution (installed with optuna)
- `pandas` — Data handling (already installed)
- `pyarrow` — Parquet storage for downloaded data
- `ruamel.yaml` — YAML writing with comment preservation
- `ccxt` — Exchange data download (already installed)

---

## 9. Constraints & Non-Goals

### In Scope
- Single-symbol optimization (BTC/USDT, etc.)
- Trading + ensemble + risk parameter spaces
- 5 loss functions with `traot` as default
- Resume, parallel execution
- Train/validation split with overfit detection
- Dry-run mode for pipeline validation
- Reproducible seeds

### Out of Scope (Future)
- Multi-symbol simultaneous optimization
- Walk-forward optimization (rolling windows)
- Multi-objective optimization (Pareto front)
- Neural architecture search (model structure)
- Training parameter optimization (handled by continuous learner)
- Mid-backtest pruning (backtests are fast enough without it)
