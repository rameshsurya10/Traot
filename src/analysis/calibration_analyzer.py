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
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr
from scipy.stats import ConstantInputWarning

logger = logging.getLogger(__name__)

# Confidence buckets — fixed, aligned with trading gate at 80%
BUCKETS = [
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),   # <- TRADING ZONE starts here
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
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
        min_trades: int = MIN_TRADES_DEFAULT
    ) -> CalibrationReport:
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

    def analyze_all(self, min_trades: int = MIN_TRADES_DEFAULT) -> List[CalibrationReport]:
        """Run analyze() for every (symbol, interval) combination in the DB."""
        pairs = self._distinct_pairs()
        return [self.analyze(sym, ivl, min_trades) for sym, ivl in pairs]

    def format_report(self, report: 'CalibrationReport') -> str:
        """
        Convert a CalibrationReport into a human-readable terminal string.

        Args:
            report: CalibrationReport produced by analyze().

        Returns:
            Multi-line string suitable for printing to a terminal.
        """
        # Verdict symbol mapping
        verdict_symbols = {
            'CALIBRATED':        '\u2713',
            'WEAKLY_CALIBRATED': '~',
            'MISCALIBRATED':     '\u2717',
            'INSUFFICIENT_DATA': '?',
        }
        verdict_sym = verdict_symbols.get(report.verdict, '?')

        lines: List[str] = []

        # Header
        lines.append(f"Calibration Report \u2014 {report.symbol} @ {report.interval}")
        lines.append('\u2550' * 38)

        # Overall stats
        lines.append(
            f"Total trades: {report.total_trades}  |  "
            f"Overall win rate: {report.overall_win_rate * 100:.1f}%"
        )

        # Paper vs live
        paper = report.paper_vs_live.get('paper', {'trades': 0, 'win_rate': 0.0})
        live  = report.paper_vs_live.get('live',  {'trades': 0, 'win_rate': 0.0})
        lines.append(
            f"Paper: {paper['trades']} trades ({paper['win_rate'] * 100:.1f}%)  |  "
            f"Live: {live['trades']} trades ({live['win_rate'] * 100:.1f}%)"
        )
        lines.append('')

        # Bucket table header
        lines.append(f"{'Confidence':<12} {'Trades':>7}   {'Win%':<8}  {'Avg P&L'}")
        lines.append('\u2500' * 41)

        # Bucket rows — trading zone on 80–90% and 90–100%
        for bucket in report.buckets:
            # Readable label: 90-100% (not 90-101%)
            high_label = 100 if bucket.conf_high >= 1.0 else int(bucket.conf_high * 100)
            low_label  = int(bucket.conf_low * 100)
            conf_label = f"{low_label}\u2013{high_label}%"

            is_trading_zone = bucket.conf_low >= 0.80

            if bucket.trade_count == 0:
                win_str = '(no data)'
                pnl_str = '(no data)'
            else:
                win_str = f"{bucket.win_rate * 100:.1f}%"
                pnl_val = bucket.avg_pnl   # already stored as percent (e.g., 0.5 = 0.5%)
                sign    = '+' if pnl_val > 0 else ''
                pnl_str = f"{sign}{pnl_val:.2f}%"

            zone_marker = '   \u2190 TRADING ZONE' if is_trading_zone else ''

            lines.append(
                f"{conf_label:<12} {bucket.trade_count:>7}   {win_str:<8}  {pnl_str}{zone_marker}"
            )

        lines.append('')

        # Monotonicity score
        lines.append(f"Monotonicity score: {report.monotonicity_score:.2f} (Spearman)")

        # Verdict line
        lines.append(f"Verdict: {verdict_sym} {report.verdict} \u2014 {report.recommendation}")

        # Regime breakdown (sorted by win rate descending)
        if report.regime_breakdown:
            lines.append('')
            lines.append("Regime breakdown:")
            sorted_regimes = sorted(
                report.regime_breakdown.items(),
                key=lambda kv: kv[1],
                reverse=True
            )
            for regime, win_rate in sorted_regimes:
                # Count trades per regime from paper_vs_live is not available per-regime;
                # the DB query was regime-level only — show win rate with a placeholder
                # for trade count (not stored on the report dataclass).
                lines.append(f"  {regime}:  {win_rate * 100:.1f}% win")

        return '\n'.join(lines)

    # ─── Private helpers ─────────────────────────────────────────────────────

    def _fetch_outcomes(self, symbol: Optional[str], interval: Optional[str]) -> List[Dict]:
        """Fetch closed trade outcomes from DB."""
        with self._db.connection() as conn:
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

    def _compute_buckets(self, outcomes: List[Dict]) -> List[BucketStats]:
        """Assign outcomes to confidence buckets and compute per-bucket stats."""
        result = []
        for low, high in BUCKETS:
            in_bucket = [
                o for o in outcomes
                if low <= o['predicted_confidence'] < high
            ]
            if not in_bucket:
                result.append(BucketStats(
                    conf_low=low, conf_high=min(high, 1.0),
                    trade_count=0, win_count=0,
                    win_rate=0.0, avg_pnl=0.0, avg_confidence=0.0
                ))
                continue
            wins = sum(1 for o in in_bucket if o['was_correct'])
            pnls = [o['pnl_percent'] for o in in_bucket if o['pnl_percent'] is not None]
            confs = [o['predicted_confidence'] for o in in_bucket]
            result.append(BucketStats(
                conf_low=low,
                conf_high=min(high, 1.0),
                trade_count=len(in_bucket),
                win_count=wins,
                win_rate=round(wins / len(in_bucket), 4),
                avg_pnl=round(float(np.mean(pnls)), 4) if pnls else 0.0,
                avg_confidence=round(float(np.mean(confs)), 4)
            ))
        return result

    def _empty_buckets(self) -> List[BucketStats]:
        """Return 5 zero-filled bucket stats."""
        return [
            BucketStats(conf_low=low, conf_high=min(high, 1.0),
                        trade_count=0, win_count=0,
                        win_rate=0.0, avg_pnl=0.0, avg_confidence=0.0)
            for low, high in BUCKETS
        ]

    def _spearman_score(self, buckets: List[BucketStats]) -> float:
        """Spearman rank correlation of avg_confidence vs win_rate.

        Uses non-empty buckets only. Returns 0.0 if fewer than 3 non-empty
        buckets (2-point Spearman is always ±1, statistically meaningless).
        """
        non_empty = [b for b in buckets if b.trade_count > 0]
        if len(non_empty) < 3:
            return 0.0
        x = [b.avg_confidence for b in non_empty]
        y = [b.win_rate for b in non_empty]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            result = spearmanr(x, y)
        score = result.statistic if hasattr(result, 'statistic') else result.correlation
        return float(score) if not np.isnan(score) else 0.0

    def _verdict(self, score: float) -> Tuple[str, str]:
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
        regime_wins: Dict[str, int] = defaultdict(int)
        regime_total: Dict[str, int] = defaultdict(int)
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

if __name__ == '__main__':
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description='Run calibration analysis: does confidence correlate with win rate?'
    )
    parser.add_argument('--symbol',     default=None,              help="Trading pair, e.g. BTC/USDT (default: all)")
    parser.add_argument('--interval',   default=None,              help="Timeframe, e.g. 15m (default: all)")
    parser.add_argument('--min-trades', default=50,   type=int,    help="Minimum closed trades needed (default: 50)")
    parser.add_argument('--db',         default='data/trading.db', help="Path to SQLite database (default: data/trading.db)")
    args = parser.parse_args()

    if args.min_trades < 1:
        print("Error: --min-trades must be >= 1", file=sys.stderr)
        sys.exit(1)

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: database not found at '{db_path}'", file=sys.stderr)
        sys.exit(1)

    # Import here to avoid circular imports when run as module
    from src.core.database import Database

    db = Database(str(db_path))
    analyzer = CalibrationAnalyzer(db)

    if args.symbol or args.interval:
        reports = [analyzer.analyze(args.symbol, args.interval, args.min_trades)]
    else:
        reports = analyzer.analyze_all(args.min_trades)

    if not reports:
        print("No closed trades found in the database.")
        sys.exit(0)

    for report in reports:
        print(analyzer.format_report(report))
        print()
