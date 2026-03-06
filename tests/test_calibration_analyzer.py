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
    db.record_trade_outcome(
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
        from src.analysis.calibration_analyzer import CalibrationReport
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


class TestCalibrationAnalyzerAll:
    def test_analyze_all_returns_one_report_per_pair(self, temp_db):
        """analyze_all() returns exactly one report per distinct (symbol, interval) pair."""
        from src.analysis.calibration_analyzer import CalibrationAnalyzer
        for _ in range(15):
            _insert_outcome(temp_db, symbol='BTC/USDT', interval='15m',
                            confidence=0.85, was_correct=1)
        for _ in range(15):
            _insert_outcome(temp_db, symbol='ETH/USDT', interval='1h',
                            confidence=0.75, was_correct=1)
        analyzer = CalibrationAnalyzer(temp_db)
        reports = analyzer.analyze_all(min_trades=10)
        symbols = {(r.symbol, r.interval) for r in reports}
        assert ('BTC/USDT', '15m') in symbols
        assert ('ETH/USDT', '1h') in symbols
        assert len(reports) == 2

    def test_analyze_all_empty_db_returns_empty_list(self, temp_db):
        """analyze_all() on an empty DB returns an empty list (no pairs exist)."""
        from src.analysis.calibration_analyzer import CalibrationAnalyzer
        analyzer = CalibrationAnalyzer(temp_db)
        reports = analyzer.analyze_all()
        assert reports == []


class TestFormatReport:
    """Tests for CalibrationAnalyzer.format_report() — terminal string output."""

    def _monotonic_report(self, temp_db):
        """Build a CALIBRATED report with data spread across multiple buckets."""
        from src.analysis.calibration_analyzer import CalibrationAnalyzer
        # Strictly increasing win rate: 40%, 50%, 55%, 60%, 70%
        scenarios = [
            (0.55, [(1, 4), (0, 6)]),
            (0.65, [(1, 5), (0, 5)]),
            (0.75, [(1, 11), (0, 9)]),
            (0.85, [(1, 12), (0, 8)]),
            (0.95, [(1, 7), (0, 3)]),
        ]
        for conf, outcomes in scenarios:
            for correct, count in outcomes:
                for _ in range(count):
                    _insert_outcome(temp_db, symbol='BTC/USDT', interval='15m',
                                    confidence=conf, was_correct=correct)
        return CalibrationAnalyzer(temp_db).analyze('BTC/USDT', '15m', min_trades=10)

    def test_format_report_contains_symbol_and_interval(self, temp_db):
        """Output must contain the symbol and interval from the report."""
        from src.analysis.calibration_analyzer import CalibrationAnalyzer
        report = self._monotonic_report(temp_db)
        output = CalibrationAnalyzer(temp_db).format_report(report)
        assert 'BTC/USDT' in output
        assert '15m' in output

    def test_format_report_shows_verdict(self, temp_db):
        """A CALIBRATED report's output must contain the word CALIBRATED."""
        from src.analysis.calibration_analyzer import CalibrationAnalyzer
        report = self._monotonic_report(temp_db)
        output = CalibrationAnalyzer(temp_db).format_report(report)
        assert 'CALIBRATED' in output

    def test_format_report_empty_bucket_shows_no_data(self, temp_db):
        """A bucket with zero trades must show '(no data)' in the output."""
        from src.analysis.calibration_analyzer import CalibrationAnalyzer
        # All trades in 80-90% bucket only — other buckets will be empty
        for _ in range(60):
            _insert_outcome(temp_db, symbol='BTC/USDT', interval='15m',
                            confidence=0.85, was_correct=1)
        report = CalibrationAnalyzer(temp_db).analyze('BTC/USDT', '15m', min_trades=10)
        output = CalibrationAnalyzer(temp_db).format_report(report)
        assert '(no data)' in output

    def test_format_report_trading_zone_marker(self, temp_db):
        """The '← TRADING ZONE' marker must appear in the output."""
        from src.analysis.calibration_analyzer import CalibrationAnalyzer
        report = self._monotonic_report(temp_db)
        output = CalibrationAnalyzer(temp_db).format_report(report)
        assert '\u2190 TRADING ZONE' in output

    def test_format_report_insufficient_data(self, temp_db):
        """An INSUFFICIENT_DATA report renders without raising an exception."""
        from src.analysis.calibration_analyzer import CalibrationAnalyzer
        # Empty DB — zero trades, well below any min_trades
        report = CalibrationAnalyzer(temp_db).analyze('BTC/USDT', '15m', min_trades=50)
        assert report.verdict == 'INSUFFICIENT_DATA'
        output = CalibrationAnalyzer(temp_db).format_report(report)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_format_report_positive_pnl_has_plus_prefix(self, temp_db):
        """Positive Avg P&L values must show correct value with '+' prefix (not * 100)."""
        from src.analysis.calibration_analyzer import CalibrationAnalyzer
        # 60 winning trades at 85% confidence with pnl=0.5 (meaning 0.5%)
        for _ in range(60):
            _insert_outcome(temp_db, symbol='BTC/USDT', interval='15m',
                            confidence=0.85, was_correct=1, pnl=0.5)
        report = CalibrationAnalyzer(temp_db).analyze('BTC/USDT', '15m', min_trades=10)
        output = CalibrationAnalyzer(temp_db).format_report(report)
        # pnl=0.5 is already a percent — should display as +0.50%, NOT +50.00%
        assert '+0.50%' in output

    def test_format_report_regime_sort_order(self, temp_db):
        """Regime breakdown must appear sorted by win rate descending."""
        from src.analysis.calibration_analyzer import CalibrationAnalyzer
        # TRENDING: 20 wins / 30 total = 66.7%
        for _ in range(20):
            _insert_outcome(temp_db, confidence=0.82, was_correct=1, regime='TRENDING')
        for _ in range(10):
            _insert_outcome(temp_db, confidence=0.82, was_correct=0, regime='TRENDING')
        # CHOPPY: 10 wins / 30 total = 33.3%
        for _ in range(10):
            _insert_outcome(temp_db, confidence=0.82, was_correct=1, regime='CHOPPY')
        for _ in range(20):
            _insert_outcome(temp_db, confidence=0.82, was_correct=0, regime='CHOPPY')
        report = CalibrationAnalyzer(temp_db).analyze('BTC/USDT', '15m', min_trades=10)
        output = CalibrationAnalyzer(temp_db).format_report(report)
        trending_pos = output.index('TRENDING')
        choppy_pos = output.index('CHOPPY')
        assert trending_pos < choppy_pos, "Higher win rate (TRENDING) should appear before CHOPPY"
