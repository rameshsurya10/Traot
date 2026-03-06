# Calibration Analyzer — Design Document

**Date**: 2026-03-02
**Status**: Approved
**Author**: Claude Code + user

---

## Problem

The trading system gates live capital deployment at `confidence >= 80%`. That 80% is computed as:

```python
confidence = (ensemble_prob - 0.5) / 0.07   # linear stretch
```

This is a transformed score, not a calibrated probability. There is no empirical evidence that an 80% confidence score corresponds to an 80% (or even 58%) win rate. Without this evidence, the confidence gate is arbitrary — it could be skipping profitable trades or letting through weak signals.

---

## Goal

Build a `CalibrationAnalyzer` that:
1. Reads closed trade outcomes from the `trade_outcomes` DB table
2. Groups trades into confidence buckets (50–60%, 60–70%, 70–80%, 80–90%, 90–100%)
3. Computes actual win rate per bucket
4. Tests monotonicity: does higher confidence → higher win rate?
5. Returns a structured `CalibrationReport` callable from the retraining engine and CLI

---

## Approach

**Approach B selected**: `CalibrationAnalyzer` class in `src/analysis/calibration_analyzer.py`, following the existing `LLMAnalyzer` pattern in the same directory.

Not chosen:
- Approach A (thin function): not reusable from retraining engine
- Approach C (full analytics pipeline): over-engineered for current data volume

---

## Architecture

### File
`src/analysis/calibration_analyzer.py`

### Data Structures

```python
@dataclass
class BucketStats:
    conf_low: float           # e.g. 0.50
    conf_high: float          # e.g. 0.60
    trade_count: int
    win_count: int
    win_rate: float           # win_count / trade_count
    avg_pnl: float            # mean pnl_percent
    avg_confidence: float     # mean actual confidence in bucket

@dataclass
class CalibrationReport:
    symbol: str
    interval: str
    total_trades: int
    overall_win_rate: float
    buckets: List[BucketStats]          # 5 buckets, some may be empty
    is_monotonic: bool                  # True if Spearman score > 0.7
    monotonicity_score: float           # Spearman rank correlation (-1 to +1)
    regime_breakdown: Dict[str, float]  # win rate per regime
    paper_vs_live: Dict[str, Any]       # separate stats for paper and live trades
    verdict: str          # CALIBRATED / WEAKLY_CALIBRATED / MISCALIBRATED / INSUFFICIENT_DATA
    recommendation: str   # human-readable action

class CalibrationAnalyzer:
    def __init__(self, database: Database)
    def analyze(symbol, interval, min_trades=50) -> CalibrationReport
    def analyze_all() -> List[CalibrationReport]
    def format_report(report) -> str
```

---

## Data Flow

### DB Query
```sql
SELECT predicted_confidence, was_correct, pnl_percent, regime, is_paper_trade
FROM trade_outcomes
WHERE symbol = ? AND interval = ?
  AND was_correct IS NOT NULL
  AND predicted_confidence IS NOT NULL
ORDER BY entry_time ASC
```

### Bucket Computation
- 5 fixed buckets: `[0.50–0.60)`, `[0.60–0.70)`, `[0.70–0.80)`, `[0.80–0.90)`, `[0.90–1.00]`
- Each trade assigned to bucket by `predicted_confidence`
- Per bucket: count trades, wins, avg `pnl_percent`, avg `predicted_confidence`
- Empty buckets included in output as zero-count rows (no divide-by-zero)

### Monotonicity Test
- `x = [avg_confidence per non-empty bucket]`
- `y = [win_rate per non-empty bucket]`
- Score = Spearman rank correlation of x vs y
- Rationale: Spearman tests ranking order (not linear fit) — correct metric for "does higher confidence rank higher in win rate?"

### Verdict Thresholds
| Condition | Verdict |
|-----------|---------|
| `total_trades < min_trades` | `INSUFFICIENT_DATA` |
| `score > 0.7` | `CALIBRATED` |
| `0.3 < score <= 0.7` | `WEAKLY_CALIBRATED` |
| `score <= 0.3` | `MISCALIBRATED` |

---

## Error Handling

| Case | Behavior |
|------|----------|
| No closed trades | Returns `INSUFFICIENT_DATA`, logs `"0/50 trades needed"` |
| Below threshold | Returns `INSUFFICIENT_DATA`, logs `"N/50 trades needed"` |
| Empty bucket | Skip bucket in monotonicity calc, show `"(no data)"` in output |
| DB error | Raises, does not swallow — caller handles |

---

## CLI Interface

```bash
# All symbols/intervals:
python -m src.analysis.calibration_analyzer

# Specific:
python -m src.analysis.calibration_analyzer --symbol BTC/USDT --interval 15m

# Options:
--min-trades 50          # override minimum threshold
--db data/trading.db     # override DB path
```

### Terminal Output Format
```
Calibration Report — BTC/USDT @ 15m
══════════════════════════════════════
Total trades: 312  |  Overall win rate: 54.2%
Paper: 298 trades (53.7%)  |  Live: 14 trades (64.3%)

Confidence   Trades   Win%     Avg P&L
─────────────────────────────────────────
50–60%          89   51.7%    -0.02%
60–70%          74   53.2%    +0.11%
70–80%          63   55.6%    +0.18%
80–90%          58   58.6%    +0.34%   ← TRADING ZONE
90–100%         28   64.3%    +0.51%

Monotonicity score: 0.94 (Spearman)
Verdict: ✓ CALIBRATED — confidence correlates with win rate

Regime breakdown:
  TRENDING:  58.1% win (116 trades)
  NORMAL:    54.0% win (89 trades)
  CHOPPY:    49.3% win (75 trades)
  VOLATILE:  51.2% win (32 trades)
```

---

## Integration Points

Once data exists (200+ trades), callable from:

```python
# Retraining engine — check calibration before deciding to retrain
from src.analysis.calibration_analyzer import CalibrationAnalyzer
report = CalibrationAnalyzer(db).analyze('BTC/USDT', '15m')
if report.verdict == 'MISCALIBRATED':
    logger.warning(f"Confidence miscalibrated: {report.recommendation}")

# Analyze all symbols:
reports = CalibrationAnalyzer(db).analyze_all()
```

---

## What It Does NOT Do (By Design)

- No Platt scaling / isotonic regression (future — after calibration curve is validated)
- No auto-trigger in live loop (on-demand only)
- No chart/HTML output (terminal only)
- No historical calibration tracking over time (future)

---

## Files Changed

| File | Action |
|------|--------|
| `src/analysis/calibration_analyzer.py` | Create |
| `src/analysis/__init__.py` | Add export |

No other files modified. Zero impact on live trading loop.
