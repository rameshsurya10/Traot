# CalibrationAnalyzer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a `CalibrationAnalyzer` class that reads closed trade outcomes from SQLite, buckets them by confidence, and computes a Spearman-based monotonicity verdict — answering "does higher confidence actually mean higher win rate?"

**Architecture:** Single file `src/analysis/calibration_analyzer.py` following the existing `LLMAnalyzer` pattern. Three dataclasses (`BucketStats`, `CalibrationReport`, `CalibrationAnalyzer`). Reads from `trade_outcomes` table. Returns structured data; also runnable as CLI via `__main__`.

**Tech Stack:** Python stdlib (`dataclasses`, `argparse`), `numpy`, `scipy.stats.spearmanr` (already in project deps via `advanced_predictor.py`)

---

## Context You Need

- DB path: `data/trading.db`
- Relevant table: `trade_outcomes` — columns: `predicted_confidence REAL`, `was_correct INTEGER`, `pnl_percent REAL`, `regime TEXT`, `is_paper_trade INTEGER`, `symbol TEXT`, `interval TEXT`, `entry_time TEXT`
- Existing pattern to follow: `src/analysis/llm_analyzer.py`
- Test DB fixture: use `temp_db` fixture from `tests/conftest.py` (creates in-memory SQLite via `tempfile`)
- Run tests: `pytest tests/test_calibration_analyzer.py -v`
- Confidence formula in live system: `(ensemble_prob - 0.5) / 0.07` — this maps raw probability to 0–1 confidence score stored in DB

---

## Task 1: Dataclasses — BucketStats and CalibrationReport

**Files:**
- Create: `tests/test_calibration_analyzer.py`
- Create: `src/analysis/calibration_analyzer.py`

### Step 1: Write the failing test

Create `tests/test_calibration_analyzer.py`:

```python
"""
Tests for CalibrationAnalyzer
"""
import sys
import tempfile
from pathlib import Path
from datetime import datetime

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.database import Database


@pytest.fixture
def temp_db():
    """Temp DB with trade_outcomes table."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    db = Database(db_path)
    yield db
    Path(db_path).unlink(missing_ok=True)


def _insert_outcome(db, symbol='BTC/USDT', interval='15m',
                    confidence=0.85, was_correct=1, pnl=0.5,
                    regime='TRENDING', is_paper=1):
    """Helper: insert one trade outcome."""
    db.save_trade_outcome(
        signal_id=None,
        symbol=symbol,
        interval=interval,
        entry_price=50000.0,
        exit_price=50250.0 if was_correct else 49750.0,
        entry_time=datetime.utcnow().isoformat(),
        exit_time=datetime.utcnow().isoformat(),
        predicted_direction='BUY',
        predicted_confidence=confidence,
        predicted_probability=0.53,
        actual_direction='BUY' if was_correct else 'SELL',
        was_correct=bool(was_correct),
        pnl_percent=pnl,
        pnl_absolute=pnl * 50000 / 100,
        features_snapshot=None,
        regime=regime,
        is_paper_trade=bool(is_paper),
        closed_by='take_profit' if was_correct else 'stop_loss',
        strategy_name=None
    )


class TestBucketStats:
    def test_fields_exist(self):
        from src.analysis.calibration_analyzer import BucketStats
        b = BucketStats(
            conf_low=0.50, conf_high=0.60,
            trade_count=10, win_count=6,
            win_rate=0.60, avg_pnl=0.12, avg_confidence=0.55
        )
        assert b.conf_low == 0.50
        assert b.win_rate == 0.60
        assert b.trade_count == 10

    def test_zero_trades_allowed(self):
        from src.analysis.calibration_analyzer import BucketStats
        b = BucketStats(
            conf_low=0.90, conf_high=1.00,
            trade_count=0, win_count=0,
            win_rate=0.0, avg_pnl=0.0, avg_confidence=0.0
        )
        assert b.trade_count == 0


class TestCalibrationReport:
    def test_fields_exist(self):
        from src.analysis.calibration_analyzer import CalibrationReport, BucketStats
        report = CalibrationReport(
            symbol='BTC/USDT',
            interval='15m',
            total_trades=100,
            overall_win_rate=0.54,
            buckets=[],
            is_monotonic=True,
            monotonicity_score=0.95,
            regime_breakdown={'TRENDING': 0.58},
            paper_vs_live={'paper': {'trades': 90, 'win_rate': 0.53},
                           'live': {'trades': 10, 'win_rate': 0.60}},
            verdict='CALIBRATED',
            recommendation='Confidence gate is empirically justified.'
        )
        assert report.verdict == 'CALIBRATED'
        assert report.is_monotonic is True
```

### Step 2: Run to verify it fails

```bash
cd /home/development1/Desktop/Ai-Trade-Bot
pytest tests/test_calibration_analyzer.py -v 2>&1 | head -20
```
Expected: `ImportError: cannot import name 'BucketStats' from 'src.analysis.calibration_analyzer'`

### Step 3: Create the dataclasses

Create `src/analysis/calibration_analyzer.py`:

```python
"""
Calibration Analyzer — Confidence vs Win Rate Diagnostic
=========================================================

Answers the key question: does the system's confidence score actually
correlate with real-world win probability?

The trading system gates capital deployment at confidence >= 80%, but
that confidence is a linear transform of ensemble probability — not an
empirically validated probability. This module measures whether higher
confidence buckets actually produce higher win rates.

Usage (programmatic):
    from src.analysis.calibration_analyzer import CalibrationAnalyzer
    report = CalibrationAnalyzer(db).analyze('BTC/USDT', '15m')
    print(report.verdict)   # CALIBRATED / MISCALIBRATED / INSUFFICIENT_DATA

Usage (CLI):
    python -m src.analysis.calibration_analyzer
    python -m src.analysis.calibration_analyzer --symbol BTC/USDT --interval 15m
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Confidence buckets — fixed, aligned with trading gate at 80%
BUCKETS = [
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),   # ← TRADING ZONE starts here
    (0.90, 1.01),   # 1.01 so conf=1.0 falls in last bucket
]

# Thresholds
MIN_TRADES_DEFAULT = 50
MONOTONIC_THRESHOLD = 0.7       # Spearman score > 0.7 = CALIBRATED
WEAK_MONOTONIC_THRESHOLD = 0.3  # Spearman score > 0.3 = WEAKLY_CALIBRATED


@dataclass
class BucketStats:
    """Win rate statistics for one confidence bucket."""
    conf_low: float
    conf_high: float
    trade_count: int
    win_count: int
    win_rate: float        # win_count / trade_count (0.0 if trade_count == 0)
    avg_pnl: float         # mean pnl_percent across trades in bucket
    avg_confidence: float  # mean predicted_confidence in bucket


@dataclass
class CalibrationReport:
    """Full calibration analysis result for one symbol/interval."""
    symbol: str
    interval: str
    total_trades: int
    overall_win_rate: float
    buckets: List[BucketStats]
    is_monotonic: bool
    monotonicity_score: float          # Spearman r (-1 to +1)
    regime_breakdown: Dict[str, float] # regime -> win_rate
    paper_vs_live: Dict[str, Any]      # paper/live trade split
    verdict: str         # CALIBRATED / WEAKLY_CALIBRATED / MISCALIBRATED / INSUFFICIENT_DATA
    recommendation: str
```

### Step 4: Run tests

```bash
pytest tests/test_calibration_analyzer.py::TestBucketStats tests/test_calibration_analyzer.py::TestCalibrationReport -v
```
Expected: `2 passed`

### Step 5: Commit

```bash
git add src/analysis/calibration_analyzer.py tests/test_calibration_analyzer.py
git commit -m "feat(calibration): add BucketStats and CalibrationReport dataclasses"
```

---

## Task 2: CalibrationAnalyzer.analyze() — Core Logic

**Files:**
- Modify: `src/analysis/calibration_analyzer.py`
- Modify: `tests/test_calibration_analyzer.py`

### Step 1: Write the failing tests

Add to `tests/test_calibration_analyzer.py`:

```python
class TestCalibrationAnalyzerInsufficientData:
    def test_empty_db_returns_insufficient(self, temp_db):
        from src.analysis.calibration_analyzer import CalibrationAnalyzer
        analyzer = CalibrationAnalyzer(temp_db)
        report = analyzer.analyze('BTC/USDT', '15m')
        assert report.verdict == 'INSUFFICIENT_DATA'
        assert report.total_trades == 0

    def test_below_threshold_returns_insufficient(self, temp_db):
        from src.analysis.calibration_analyzer import CalibrationAnalyzer
        # Insert 10 trades — below default threshold of 50
        for i in range(10):
            _insert_outcome(temp_db, confidence=0.82, was_correct=1)
        analyzer = CalibrationAnalyzer(temp_db)
        report = analyzer.analyze('BTC/USDT', '15m', min_trades=50)
        assert report.verdict == 'INSUFFICIENT_DATA'
        assert report.total_trades == 10

    def test_custom_min_trades(self, temp_db):
        from src.analysis.calibration_analyzer import CalibrationAnalyzer
        for i in range(10):
            _insert_outcome(temp_db, confidence=0.82, was_correct=1)
        analyzer = CalibrationAnalyzer(temp_db)
        # With min_trades=5, 10 trades is enough
        report = analyzer.analyze('BTC/USDT', '15m', min_trades=5)
        assert report.verdict != 'INSUFFICIENT_DATA'
        assert report.total_trades == 10


class TestCalibrationAnalyzerBuckets:
    def test_bucket_assignment(self, temp_db):
        """Trades in 80-90% bucket should be counted there."""
        from src.analysis.calibration_analyzer import CalibrationAnalyzer
        # Insert 60 trades all at 85% confidence
        for i in range(60):
            _insert_outcome(temp_db, confidence=0.85, was_correct=(i % 2 == 0))
        analyzer = CalibrationAnalyzer(temp_db)
        report = analyzer.analyze('BTC/USDT', '15m', min_trades=10)
        bucket_80_90 = next(b for b in report.buckets if b.conf_low == 0.80)
        assert bucket_80_90.trade_count == 60
        assert abs(bucket_80_90.win_rate - 0.5) < 0.05  # ~50% win rate

    def test_empty_bucket_has_zero_count(self, temp_db):
        """Buckets with no trades should have trade_count=0, not be missing."""
        from src.analysis.calibration_analyzer import CalibrationAnalyzer
        for i in range(60):
            _insert_outcome(temp_db, confidence=0.85, was_correct=1)
        analyzer = CalibrationAnalyzer(temp_db)
        report = analyzer.analyze('BTC/USDT', '15m', min_trades=10)
        # Always 5 buckets
        assert len(report.buckets) == 5
        # 50-60% bucket should be empty
        bucket_50_60 = next(b for b in report.buckets if b.conf_low == 0.50)
        assert bucket_50_60.trade_count == 0
        assert bucket_50_60.win_rate == 0.0

    def test_overall_win_rate_correct(self, temp_db):
        """Overall win rate = total wins / total trades."""
        from src.analysis.calibration_analyzer import CalibrationAnalyzer
        # 40 wins, 20 losses = 66.7% win rate
        for i in range(40):
            _insert_outcome(temp_db, confidence=0.82, was_correct=1)
        for i in range(20):
            _insert_outcome(temp_db, confidence=0.75, was_correct=0)
        analyzer = CalibrationAnalyzer(temp_db)
        report = analyzer.analyze('BTC/USDT', '15m', min_trades=10)
        assert abs(report.overall_win_rate - (40/60)) < 0.01


class TestCalibrationAnalyzerMonotonicity:
    def test_perfectly_monotonic_is_calibrated(self, temp_db):
        """Strictly increasing win rate per bucket → CALIBRATED."""
        from src.analysis.calibration_analyzer import CalibrationAnalyzer
        # Build perfect monotonic distribution:
        # 50-60%: 40% win, 60-70%: 50% win, 70-80%: 55% win,
        # 80-90%: 60% win, 90-100%: 70% win
        scenarios = [
            (0.55, [(1, 4), (0, 6)]),   # 40% win
            (0.65, [(1, 5), (0, 5)]),   # 50% win
            (0.75, [(1, 11), (0, 9)]),  # ~55% win
            (0.85, [(1, 12), (0, 8)]),  # 60% win
            (0.95, [(1, 7), (0, 3)]),   # 70% win
        ]
        for conf, outcomes in scenarios:
            for (correct, count) in outcomes:
                for _ in range(count):
                    _insert_outcome(temp_db, confidence=conf, was_correct=correct)
        analyzer = CalibrationAnalyzer(temp_db)
        report = analyzer.analyze('BTC/USDT', '15m', min_trades=10)
        assert report.verdict == 'CALIBRATED'
        assert report.is_monotonic is True
        assert report.monotonicity_score > 0.7

    def test_flat_win_rate_is_miscalibrated(self, temp_db):
        """Same win rate in every bucket → low Spearman → MISCALIBRATED."""
        from src.analysis.calibration_analyzer import CalibrationAnalyzer
        # Same ~50% win rate at every confidence level
        for conf in [0.55, 0.65, 0.75, 0.85, 0.95]:
            for i in range(20):
                _insert_outcome(temp_db, confidence=conf,
                                was_correct=(i % 2 == 0))
        analyzer = CalibrationAnalyzer(temp_db)
        report = analyzer.analyze('BTC/USDT', '15m', min_trades=10)
        # Flat curve: Spearman near 0 → MISCALIBRATED or WEAKLY_CALIBRATED
        assert report.verdict in ('MISCALIBRATED', 'WEAKLY_CALIBRATED')
        assert report.monotonicity_score < 0.7


class TestCalibrationAnalyzerRegimes:
    def test_regime_breakdown_computed(self, temp_db):
        """Win rate should be computed per regime."""
        from src.analysis.calibration_analyzer import CalibrationAnalyzer
        # 20 TRENDING wins, 10 TRENDING losses
        for _ in range(20):
            _insert_outcome(temp_db, confidence=0.82, was_correct=1, regime='TRENDING')
        for _ in range(10):
            _insert_outcome(temp_db, confidence=0.82, was_correct=0, regime='TRENDING')
        # 15 CHOPPY wins, 15 losses
        for _ in range(15):
            _insert_outcome(temp_db, confidence=0.82, was_correct=1, regime='CHOPPY')
        for _ in range(15):
            _insert_outcome(temp_db, confidence=0.82, was_correct=0, regime='CHOPPY')
        analyzer = CalibrationAnalyzer(temp_db)
        report = analyzer.analyze('BTC/USDT', '15m', min_trades=10)
        assert 'TRENDING' in report.regime_breakdown
        assert abs(report.regime_breakdown['TRENDING'] - (20/30)) < 0.01
        assert abs(report.regime_breakdown['CHOPPY'] - 0.5) < 0.01

    def test_paper_vs_live_split(self, temp_db):
        """Paper and live trades should be reported separately."""
        from src.analysis.calibration_analyzer import CalibrationAnalyzer
        for _ in range(40):
            _insert_outcome(temp_db, confidence=0.82, was_correct=1, is_paper=1)
        for _ in range(10):
            _insert_outcome(temp_db, confidence=0.82, was_correct=0, is_paper=0)
        analyzer = CalibrationAnalyzer(temp_db)
        report = analyzer.analyze('BTC/USDT', '15m', min_trades=10)
        assert report.paper_vs_live['paper']['trades'] == 40
        assert report.paper_vs_live['live']['trades'] == 10
```

### Step 2: Run to verify tests fail

```bash
pytest tests/test_calibration_analyzer.py -v -k "not TestBucketStats and not TestCalibrationReport" 2>&1 | head -30
```
Expected: `AttributeError: type object 'CalibrationAnalyzer' has no attribute ...` or `ImportError`

### Step 3: Implement CalibrationAnalyzer.analyze()

Add to `src/analysis/calibration_analyzer.py` after the dataclasses:

```python
import numpy as np
from scipy.stats import spearmanr


class CalibrationAnalyzer:
    """
    Measures whether confidence scores correlate with actual win rates.

    Usage:
        report = CalibrationAnalyzer(db).analyze('BTC/USDT', '15m')
        if report.verdict == 'MISCALIBRATED':
            logger.warning(report.recommendation)
    """

    def __init__(self, database):
        """
        Args:
            database: src.core.database.Database instance
        """
        self._db = database

    def analyze(
        self,
        symbol: str = None,
        interval: str = None,
        min_trades: int = MIN_TRADES_DEFAULT
    ) -> 'CalibrationReport':
        """
        Run calibration analysis for one symbol/interval.

        Args:
            symbol:     Trading pair, e.g. 'BTC/USDT'. None = all symbols.
            interval:   Timeframe, e.g. '15m'. None = all intervals.
            min_trades: Minimum closed trades needed to produce a report.

        Returns:
            CalibrationReport with verdict and per-bucket stats.
        """
        sym_label = symbol or 'ALL'
        int_label = interval or 'ALL'

        outcomes = self._fetch_outcomes(symbol, interval)

        if len(outcomes) < min_trades:
            logger.info(
                f"[CALIBRATION] {sym_label}@{int_label}: "
                f"{len(outcomes)}/{min_trades} trades needed"
            )
            return CalibrationReport(
                symbol=sym_label,
                interval=int_label,
                total_trades=len(outcomes),
                overall_win_rate=0.0,
                buckets=self._empty_buckets(),
                is_monotonic=False,
                monotonicity_score=0.0,
                regime_breakdown={},
                paper_vs_live={'paper': {'trades': 0, 'win_rate': 0.0},
                               'live':  {'trades': 0, 'win_rate': 0.0}},
                verdict='INSUFFICIENT_DATA',
                recommendation=(
                    f"Need {min_trades - len(outcomes)} more closed trades "
                    f"before calibration can be computed."
                )
            )

        buckets = self._compute_buckets(outcomes)
        score = self._spearman_score(buckets)
        verdict, recommendation = self._verdict(score)
        regime_breakdown = self._regime_breakdown(outcomes)
        paper_vs_live = self._paper_vs_live(outcomes)
        wins = sum(1 for o in outcomes if o['was_correct'])
        overall_win_rate = wins / len(outcomes) if outcomes else 0.0

        report = CalibrationReport(
            symbol=sym_label,
            interval=int_label,
            total_trades=len(outcomes),
            overall_win_rate=overall_win_rate,
            buckets=buckets,
            is_monotonic=score > MONOTONIC_THRESHOLD,
            monotonicity_score=round(score, 4),
            regime_breakdown=regime_breakdown,
            paper_vs_live=paper_vs_live,
            verdict=verdict,
            recommendation=recommendation
        )

        logger.info(
            f"[CALIBRATION] {sym_label}@{int_label}: "
            f"{len(outcomes)} trades | {verdict} | score={score:.2f}"
        )
        return report

    def analyze_all(self, min_trades: int = MIN_TRADES_DEFAULT) -> List['CalibrationReport']:
        """Run analyze() for every (symbol, interval) combination in the DB."""
        pairs = self._distinct_pairs()
        return [self.analyze(sym, ivl, min_trades) for sym, ivl in pairs]

    # ─── Private helpers ─────────────────────────────────────────────────────

    def _fetch_outcomes(self, symbol: Optional[str], interval: Optional[str]) -> List[Dict]:
        """Fetch closed trade outcomes from DB."""
        with self._db.connection() as conn:
            conn.row_factory = __import__('sqlite3').Row
            cur = conn.cursor()
            filters = [
                "was_correct IS NOT NULL",
                "predicted_confidence IS NOT NULL"
            ]
            params = []
            if symbol:
                filters.append("symbol = ?")
                params.append(symbol)
            if interval:
                filters.append("interval = ?")
                params.append(interval)
            where = " AND ".join(filters)
            cur.execute(
                f"SELECT predicted_confidence, was_correct, pnl_percent, "
                f"regime, is_paper_trade "
                f"FROM trade_outcomes WHERE {where} ORDER BY entry_time ASC",
                params
            )
            return [dict(row) for row in cur.fetchall()]

    def _compute_buckets(self, outcomes: List[Dict]) -> List['BucketStats']:
        """Assign outcomes to confidence buckets and compute per-bucket stats."""
        result = []
        for low, high in BUCKETS:
            in_bucket = [
                o for o in outcomes
                if low <= o['predicted_confidence'] < high
            ]
            if not in_bucket:
                result.append(BucketStats(
                    conf_low=low, conf_high=high,
                    trade_count=0, win_count=0,
                    win_rate=0.0, avg_pnl=0.0, avg_confidence=0.0
                ))
                continue
            wins = sum(1 for o in in_bucket if o['was_correct'])
            pnls = [o['pnl_percent'] for o in in_bucket if o['pnl_percent'] is not None]
            confs = [o['predicted_confidence'] for o in in_bucket]
            result.append(BucketStats(
                conf_low=low,
                conf_high=min(high, 1.0),   # display cap at 1.0
                trade_count=len(in_bucket),
                win_count=wins,
                win_rate=round(wins / len(in_bucket), 4),
                avg_pnl=round(float(np.mean(pnls)), 4) if pnls else 0.0,
                avg_confidence=round(float(np.mean(confs)), 4)
            ))
        return result

    def _empty_buckets(self) -> List['BucketStats']:
        """Return 5 zero-filled bucket stats."""
        return [
            BucketStats(conf_low=low, conf_high=min(high, 1.0),
                        trade_count=0, win_count=0,
                        win_rate=0.0, avg_pnl=0.0, avg_confidence=0.0)
            for low, high in BUCKETS
        ]

    def _spearman_score(self, buckets: List['BucketStats']) -> float:
        """Spearman rank correlation of avg_confidence vs win_rate."""
        non_empty = [b for b in buckets if b.trade_count > 0]
        if len(non_empty) < 2:
            return 0.0
        x = [b.avg_confidence for b in non_empty]
        y = [b.win_rate for b in non_empty]
        result = spearmanr(x, y)
        score = result.statistic if hasattr(result, 'statistic') else result.correlation
        return float(score) if not np.isnan(score) else 0.0

    def _verdict(self, score: float):
        """Map Spearman score to human verdict and recommendation."""
        if score > MONOTONIC_THRESHOLD:
            return (
                'CALIBRATED',
                'Confidence correlates with win rate. '
                '80% gate is empirically justified. Continue accumulating data.'
            )
        elif score > WEAK_MONOTONIC_THRESHOLD:
            return (
                'WEAKLY_CALIBRATED',
                'Weak correlation between confidence and win rate. '
                'Gate is partially meaningful. Consider widening confidence range '
                'or reviewing ensemble weights.'
            )
        else:
            return (
                'MISCALIBRATED',
                'No meaningful correlation between confidence and win rate. '
                'The 80% gate is arbitrary. Run LSTM weight ablation test, '
                'review ensemble probability formula, or recalibrate confidence floor/ceiling.'
            )

    def _regime_breakdown(self, outcomes: List[Dict]) -> Dict[str, float]:
        """Win rate per market regime."""
        from collections import defaultdict
        regime_wins = defaultdict(int)
        regime_total = defaultdict(int)
        for o in outcomes:
            regime = o.get('regime') or 'UNKNOWN'
            regime_total[regime] += 1
            if o['was_correct']:
                regime_wins[regime] += 1
        return {
            r: round(regime_wins[r] / regime_total[r], 4)
            for r in regime_total
        }

    def _paper_vs_live(self, outcomes: List[Dict]) -> Dict:
        """Separate win rates for paper and live trades."""
        paper = [o for o in outcomes if o.get('is_paper_trade')]
        live  = [o for o in outcomes if not o.get('is_paper_trade')]

        def stats(group):
            if not group:
                return {'trades': 0, 'win_rate': 0.0}
            wins = sum(1 for o in group if o['was_correct'])
            return {'trades': len(group), 'win_rate': round(wins / len(group), 4)}

        return {'paper': stats(paper), 'live': stats(live)}

    def _distinct_pairs(self) -> List[tuple]:
        """All (symbol, interval) combinations with closed trades."""
        with self._db.connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT DISTINCT symbol, interval FROM trade_outcomes "
                "WHERE was_correct IS NOT NULL"
            )
            return [(row[0], row[1]) for row in cur.fetchall()]
```

### Step 4: Run new tests

```bash
pytest tests/test_calibration_analyzer.py -v 2>&1 | tail -20
```
Expected: all tests pass. If `spearmanr` API differs between scipy versions, the `hasattr(result, 'statistic')` guard handles it.

### Step 5: Commit

```bash
git add src/analysis/calibration_analyzer.py tests/test_calibration_analyzer.py
git commit -m "feat(calibration): implement CalibrationAnalyzer.analyze() with Spearman monotonicity"
```

---

## Task 3: format_report() — Terminal Output

**Files:**
- Modify: `src/analysis/calibration_analyzer.py`
- Modify: `tests/test_calibration_analyzer.py`

### Step 1: Write the failing test

Add to `tests/test_calibration_analyzer.py`:

```python
class TestFormatReport:
    def test_format_shows_verdict(self, temp_db):
        from src.analysis.calibration_analyzer import CalibrationAnalyzer
        for conf in [0.55, 0.65, 0.75, 0.85, 0.95]:
            for i in range(12):
                _insert_outcome(temp_db, confidence=conf, was_correct=(i < 7))
        analyzer = CalibrationAnalyzer(temp_db)
        report = analyzer.analyze('BTC/USDT', '15m', min_trades=10)
        text = analyzer.format_report(report)
        assert 'BTC/USDT' in text
        assert '15m' in text
        assert report.verdict in text

    def test_format_shows_all_buckets(self, temp_db):
        from src.analysis.calibration_analyzer import CalibrationAnalyzer
        for conf in [0.55, 0.85]:
            for i in range(30):
                _insert_outcome(temp_db, confidence=conf, was_correct=(i < 18))
        analyzer = CalibrationAnalyzer(temp_db)
        report = analyzer.analyze('BTC/USDT', '15m', min_trades=10)
        text = analyzer.format_report(report)
        assert '50' in text and '60' in text   # bucket label
        assert '80' in text and '90' in text   # trading zone bucket

    def test_format_insufficient_data(self, temp_db):
        from src.analysis.calibration_analyzer import CalibrationAnalyzer
        analyzer = CalibrationAnalyzer(temp_db)
        report = analyzer.analyze('BTC/USDT', '15m')
        text = analyzer.format_report(report)
        assert 'INSUFFICIENT' in text
        assert '0/' in text   # "0/50 trades needed" style message
```

### Step 2: Run to verify fail

```bash
pytest tests/test_calibration_analyzer.py::TestFormatReport -v 2>&1 | head -15
```
Expected: `AttributeError: 'CalibrationAnalyzer' object has no attribute 'format_report'`

### Step 3: Implement format_report()

Add method to `CalibrationAnalyzer` class:

```python
def format_report(self, report: 'CalibrationReport') -> str:
    """
    Format CalibrationReport as a human-readable terminal string.

    Returns a multi-line string suitable for print() or logging.
    """
    lines = []
    lines.append(f"\nCalibration Report — {report.symbol} @ {report.interval}")
    lines.append("═" * 50)

    if report.verdict == 'INSUFFICIENT_DATA':
        needed = max(0, MIN_TRADES_DEFAULT - report.total_trades)
        lines.append(
            f"  INSUFFICIENT DATA: {report.total_trades}/{MIN_TRADES_DEFAULT} trades needed"
            f" ({needed} more required)"
        )
        lines.append(f"  {report.recommendation}")
        return "\n".join(lines)

    lines.append(
        f"Total trades: {report.total_trades}  |  "
        f"Overall win rate: {report.overall_win_rate:.1%}"
    )
    p = report.paper_vs_live.get('paper', {})
    lv = report.paper_vs_live.get('live', {})
    lines.append(
        f"Paper: {p.get('trades', 0)} trades "
        f"({p.get('win_rate', 0):.1%})  |  "
        f"Live: {lv.get('trades', 0)} trades "
        f"({lv.get('win_rate', 0):.1%})"
    )
    lines.append("")

    # Bucket table
    lines.append(f"{'Confidence':<12} {'Trades':>7} {'Win%':>8} {'Avg P&L':>9}")
    lines.append("─" * 42)
    for b in report.buckets:
        trading_zone = "  ← TRADING ZONE" if b.conf_low >= 0.80 and b.trade_count > 0 else ""
        if b.trade_count == 0:
            lines.append(
                f"{int(b.conf_low*100):2d}–{int(b.conf_high*100):3d}%"
                f"{'':>8} {'(no data)':>8}"
            )
        else:
            pnl_str = f"+{b.avg_pnl:.2f}%" if b.avg_pnl >= 0 else f"{b.avg_pnl:.2f}%"
            lines.append(
                f"{int(b.conf_low*100):2d}–{int(b.conf_high*100):3d}%"
                f"{b.trade_count:>10}"
                f"{b.win_rate:>8.1%}"
                f"{pnl_str:>10}"
                f"{trading_zone}"
            )

    lines.append("")
    verdict_icon = "✓" if report.verdict == "CALIBRATED" else "⚠" if report.verdict == "WEAKLY_CALIBRATED" else "✗"
    lines.append(f"Monotonicity score: {report.monotonicity_score:.2f} (Spearman)")
    lines.append(f"Verdict: {verdict_icon} {report.verdict}")
    lines.append(f"  → {report.recommendation}")

    if report.regime_breakdown:
        lines.append("")
        lines.append("Regime breakdown:")
        for regime, win_rate in sorted(report.regime_breakdown.items()):
            flag = "  ← below 50%" if win_rate < 0.50 else ""
            lines.append(f"  {regime:<12} {win_rate:.1%}{flag}")

    return "\n".join(lines)
```

### Step 4: Run tests

```bash
pytest tests/test_calibration_analyzer.py -v 2>&1 | tail -15
```
Expected: all tests pass.

### Step 5: Commit

```bash
git add src/analysis/calibration_analyzer.py tests/test_calibration_analyzer.py
git commit -m "feat(calibration): add format_report() terminal output"
```

---

## Task 4: CLI __main__ Block

**Files:**
- Modify: `src/analysis/calibration_analyzer.py`

### Step 1: Add the __main__ block

Append at the **bottom** of `src/analysis/calibration_analyzer.py`:

```python
# ─── CLI entrypoint ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    import sys
    from pathlib import Path

    # Allow running from project root without installing
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.core.database import Database

    parser = argparse.ArgumentParser(
        description='Calibration Analyzer — confidence vs win rate diagnostic'
    )
    parser.add_argument('--symbol',     default=None, help='e.g. BTC/USDT')
    parser.add_argument('--interval',   default=None, help='e.g. 15m')
    parser.add_argument('--min-trades', type=int, default=MIN_TRADES_DEFAULT,
                        help=f'Minimum trades required (default: {MIN_TRADES_DEFAULT})')
    parser.add_argument('--db',         default='data/trading.db',
                        help='Path to trading.db (default: data/trading.db)')
    args = parser.parse_args()

    db = Database(args.db)
    analyzer = CalibrationAnalyzer(db)

    if args.symbol or args.interval:
        report = analyzer.analyze(args.symbol, args.interval, args.min_trades)
        print(analyzer.format_report(report))
    else:
        reports = analyzer.analyze_all(args.min_trades)
        if not reports:
            print("No closed trades found in database.")
        for report in reports:
            print(analyzer.format_report(report))
            print()
```

### Step 2: Verify CLI runs without error (even with 1 trade in DB)

```bash
cd /home/development1/Desktop/Ai-Trade-Bot
python -m src.analysis.calibration_analyzer
```
Expected output (with current 1-trade DB):
```
Calibration Report — BTC/USDT @ 15m
══════════════════════════════════════
  INSUFFICIENT DATA: 1/50 trades needed (49 more required)
  ...
```
No crash, clean message.

### Step 3: Test symbol filter

```bash
python -m src.analysis.calibration_analyzer --symbol BTC/USDT --interval 15m --min-trades 1
```
Expected: Shows report for the 1 existing trade with actual bucket data.

### Step 4: Commit

```bash
git add src/analysis/calibration_analyzer.py
git commit -m "feat(calibration): add CLI __main__ entrypoint with argparse"
```

---

## Task 5: Export from __init__.py

**Files:**
- Modify: `src/analysis/__init__.py`

### Step 1: Add export

Open `src/analysis/__init__.py` and add:

```python
from src.analysis.calibration_analyzer import (
    CalibrationAnalyzer,
    CalibrationReport,
    BucketStats,
)

__all__ = [
    'CalibrationAnalyzer',
    'CalibrationReport',
    'BucketStats',
]
```

### Step 2: Verify import works from project root

```bash
cd /home/development1/Desktop/Ai-Trade-Bot
python -c "from src.analysis import CalibrationAnalyzer; print('OK')"
```
Expected: `OK`

### Step 3: Run full test suite

```bash
pytest tests/test_calibration_analyzer.py -v
```
Expected: All tests pass.

### Step 4: Commit

```bash
git add src/analysis/__init__.py
git commit -m "feat(calibration): export CalibrationAnalyzer from src.analysis"
```

---

## Final Verification

```bash
# 1. Full test suite for this module
pytest tests/test_calibration_analyzer.py -v

# 2. Lint check
ruff check src/analysis/calibration_analyzer.py

# 3. CLI smoke test
python -m src.analysis.calibration_analyzer

# 4. Import check (used by retraining engine)
python -c "
from src.analysis import CalibrationAnalyzer
from src.core.database import Database
db = Database('data/trading.db')
report = CalibrationAnalyzer(db).analyze('BTC/USDT', '15m')
print(f'Verdict: {report.verdict}')
print(f'Trades: {report.total_trades}')
"
```

Expected output for current DB (1 trade):
```
Verdict: INSUFFICIENT_DATA
Trades: 1
```

---

## What This Enables Next

Once 200+ trades accumulate, run:
```bash
python -m src.analysis.calibration_analyzer
```

And get the first empirical answer to: **"Is our 80% confidence gate meaningful?"**

If `CALIBRATED` → expand data, add Fyers.
If `MISCALIBRATED` → fix ensemble weights first.
