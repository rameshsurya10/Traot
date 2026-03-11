"""
Traot — Daily Trading Report (Email + PDF)
===========================================
Sends a detailed 24-hour report every day at a configured time (default 8 AM IST).

Covers the previous calendar day (00:00 to 23:59):
  - Overview metrics, signals breakdown, trade results
  - Per-symbol performance, learning activity, confidence tracking
  - Retraining events, AI commentary (what happened / improve / outlook)

Config (config.yaml):
    daily_report:
      enabled: true
      send_hour: 8
      timezone_offset: 5.5

SMTP credentials from environment variables only:
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_RECIPIENT
"""

import html as html_mod
import io
import logging
import os
import smtplib
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        HRFlowable,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    _REPORTLAB_AVAILABLE = True
except ImportError:
    _REPORTLAB_AVAILABLE = False

logger = logging.getLogger(__name__)

APP_NAME = "Traot"
APP_VERSION = "1.0"


class DailyReportScheduler:
    """Schedules and sends daily trading report emails."""

    def __init__(self, config: Dict, db_path: str = "data/trading.db"):
        report_cfg = config.get('daily_report', {})
        self._enabled = report_cfg.get('enabled', True)
        self._send_hour = report_cfg.get('send_hour', 8)
        self._tz_offset = report_cfg.get('timezone_offset', 5.5)
        self._db_path = db_path

        # SMTP from env vars only (never config — security)
        self._smtp_host = os.environ.get('SMTP_HOST', 'smtp.gmail.com')
        self._smtp_port = int(os.environ.get('SMTP_PORT', '587'))
        self._smtp_user = os.environ.get('SMTP_USER', '')
        self._smtp_password = os.environ.get('SMTP_PASSWORD', '')
        self._recipient = os.environ.get('SMTP_RECIPIENT', self._smtp_user)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_sent_date: Optional[str] = None

        if not self._smtp_user or not self._smtp_password:
            if self._enabled:
                logger.warning("Daily report enabled but SMTP_USER/SMTP_PASSWORD not set")
                self._enabled = False

    @property
    def is_available(self) -> bool:
        return self._enabled and bool(self._smtp_user)

    def start(self):
        if not self._enabled:
            logger.info("Daily report disabled — skipping")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._scheduler_loop, name="daily-report", daemon=True,
        )
        self._thread.start()
        logger.info(
            f"Daily report scheduler started "
            f"(send at {self._send_hour:02d}:00 UTC+{self._tz_offset})"
        )

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("Daily report scheduler stopped")

    def _scheduler_loop(self):
        while not self._stop_event.is_set():
            try:
                tz = timezone(timedelta(hours=self._tz_offset))
                now = datetime.now(tz)
                today_str = now.strftime('%Y-%m-%d')
                if now.hour == self._send_hour and self._last_sent_date != today_str:
                    logger.info("Sending daily report...")
                    yesterday = now - timedelta(days=1)
                    self._send_report(yesterday.strftime('%Y-%m-%d'))
                    self._last_sent_date = today_str
            except Exception as e:
                logger.error(f"Daily report scheduler error: {e}")
            self._stop_event.wait(60)

    def send_now(self, date_str: Optional[str] = None, recipient: Optional[str] = None):
        if not self._smtp_user or not self._smtp_password:
            logger.error("Cannot send report: SMTP credentials not configured")
            return
        if date_str is None:
            tz = timezone(timedelta(hours=self._tz_offset))
            yesterday = datetime.now(tz) - timedelta(days=1)
            date_str = yesterday.strftime('%Y-%m-%d')
        self._send_report(date_str, recipient_override=recipient)

    def _send_report(self, date_str: str, recipient_override: Optional[str] = None):
        recipient = recipient_override or self._recipient
        try:
            data = self._query_all_data(date_str)
            commentary = self._generate_commentary(data, date_str)
            html = self._build_html(data, commentary, date_str)
            text = self._build_plain_text(data, commentary, date_str)
            subject = f"{APP_NAME} — Daily Report {date_str}"

            msg = MIMEMultipart('mixed')
            msg['Subject'] = subject
            msg['From'] = self._smtp_user
            msg['To'] = recipient

            body = MIMEMultipart('alternative')
            body.attach(MIMEText(text, 'plain', 'utf-8'))
            body.attach(MIMEText(html, 'html', 'utf-8'))
            msg.attach(body)

            if _REPORTLAB_AVAILABLE:
                pdf_bytes = self._build_pdf(data, commentary, date_str)
                pdf_part = MIMEBase('application', 'pdf')
                pdf_part.set_payload(pdf_bytes)
                encoders.encode_base64(pdf_part)
                pdf_part.add_header(
                    'Content-Disposition',
                    f'attachment; filename="traot_report_{date_str}.pdf"',
                )
                msg.attach(pdf_part)
            else:
                logger.info("reportlab not installed — sending HTML-only email (no PDF)")

            with smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=30) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(self._smtp_user, self._smtp_password)
                server.send_message(msg)

            logger.info(f"Daily report sent to {recipient} for {date_str}")
        except Exception as e:
            logger.error(f"Failed to send daily report: {e}", exc_info=True)

    # =========================================================================
    # Data collection
    # =========================================================================

    def _query_all_data(self, date_str: str) -> Dict[str, Any]:
        start = f"{date_str}T00:00:00"
        end = f"{date_str}T23:59:59"
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            return {
                'overview': self._q_overview(conn, start, end),
                'signals': self._q_signals(conn, start, end),
                'trades': self._q_trades(conn, start, end),
                'trade_list': self._q_trade_list(conn, start, end),
                'per_symbol': self._q_per_symbol(conn, start, end),
                'learning': self._q_learning(conn, start, end),
                'confidence': self._q_confidence(conn, start, end),
                'retraining': self._q_retraining(conn, start, end),
            }
        finally:
            conn.close()

    def _q_overview(self, conn, start, end) -> Dict:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*), COUNT(DISTINCT symbol) FROM candles "
            "WHERE datetime >= ? AND datetime <= ?", (start, end),
        )
        candles, symbols = cur.fetchone()
        cur.execute(
            "SELECT COUNT(*) FROM signals WHERE datetime >= ? AND datetime <= ?",
            (start, end),
        )
        signals = cur.fetchone()[0]
        cur.execute(
            "SELECT COUNT(*) FROM trade_outcomes WHERE entry_time >= ? AND entry_time <= ? "
            "AND COALESCE(is_replay, 0) = 0",
            (start, end),
        )
        trades = cur.fetchone()[0]
        return {'candles': candles, 'symbols': symbols, 'signals': signals, 'trades': trades}

    def _q_signals(self, conn, start, end) -> Dict:
        cur = conn.cursor()
        cur.execute(
            "SELECT AVG(confidence), MAX(confidence), MIN(confidence) "
            "FROM signals WHERE datetime >= ? AND datetime <= ?", (start, end),
        )
        row = cur.fetchone()
        avg_c, max_c, min_c = (row[0] or 0), (row[1] or 0), (row[2] or 0)
        cur.execute(
            "SELECT COUNT(*) FROM signals "
            "WHERE datetime >= ? AND datetime <= ? AND confidence >= 0.80",
            (start, end),
        )
        above_gate = cur.fetchone()[0]
        cur.execute(
            "SELECT signal_type, COUNT(*) FROM signals "
            "WHERE datetime >= ? AND datetime <= ? GROUP BY signal_type",
            (start, end),
        )
        tc = {r[0]: r[1] for r in cur.fetchall()}
        return {
            'avg_conf': avg_c, 'max_conf': max_c, 'min_conf': min_c,
            'above_gate': above_gate,
            'buys': tc.get('BUY', 0),
            'sells': tc.get('SELL', 0),
            'holds': tc.get('HOLD', 0) + tc.get('NEUTRAL', 0),
        }

    def _q_trades(self, conn, start, end) -> Dict:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*), "
            "  SUM(CASE WHEN was_correct=1 THEN 1 ELSE 0 END), "
            "  AVG(pnl_percent), SUM(pnl_percent), AVG(predicted_confidence) "
            "FROM trade_outcomes "
            "WHERE entry_time >= ? AND entry_time <= ? AND was_correct IS NOT NULL "
            "AND COALESCE(is_replay, 0) = 0",
            (start, end),
        )
        row = cur.fetchone()
        closed, wins = (row[0] or 0), (row[1] or 0)
        avg_pnl, total_pnl, avg_conf = (row[2] or 0), (row[3] or 0), (row[4] or 0)
        cur.execute(
            "SELECT COUNT(*) FROM trade_outcomes "
            "WHERE entry_time >= ? AND entry_time <= ? AND was_correct IS NULL "
            "AND COALESCE(is_replay, 0) = 0",
            (start, end),
        )
        still_open = cur.fetchone()[0]
        return {
            'closed': closed, 'wins': wins, 'losses': closed - wins,
            'avg_pnl': avg_pnl, 'total_pnl': total_pnl,
            'avg_conf': avg_conf, 'still_open': still_open,
            'win_rate': (wins / closed * 100) if closed > 0 else 0,
        }

    def _q_trade_list(self, conn, start, end) -> List[Dict]:
        cur = conn.cursor()
        cur.execute(
            "SELECT symbol, interval, predicted_direction, predicted_confidence, "
            "  entry_price, exit_price, entry_time, exit_time, "
            "  was_correct, pnl_percent, regime, strategy_name, closed_by "
            "FROM trade_outcomes WHERE entry_time >= ? AND entry_time <= ? "
            "AND COALESCE(is_replay, 0) = 0 "
            "ORDER BY entry_time ASC", (start, end),
        )
        return [dict(r) for r in cur.fetchall()]

    def _q_per_symbol(self, conn, start, end) -> List[Dict]:
        cur = conn.cursor()
        cur.execute(
            "SELECT symbol, COUNT(*) as cnt, "
            "  SUM(CASE WHEN was_correct=1 THEN 1 ELSE 0 END) as wins, "
            "  AVG(pnl_percent) as avg_pnl, SUM(pnl_percent) as total_pnl, "
            "  AVG(predicted_confidence) as avg_conf, MAX(predicted_confidence) as max_conf "
            "FROM trade_outcomes "
            "WHERE entry_time >= ? AND entry_time <= ? AND was_correct IS NOT NULL "
            "AND COALESCE(is_replay, 0) = 0 "
            "GROUP BY symbol ORDER BY total_pnl DESC", (start, end),
        )
        return [dict(r) for r in cur.fetchall()]

    def _q_learning(self, conn, start, end) -> Dict:
        cur = conn.cursor()
        cur.execute(
            "SELECT mode, COUNT(*) FROM learning_states "
            "WHERE entered_at >= ? AND entered_at <= ? GROUP BY mode", (start, end),
        )
        mc = {r[0]: r[1] for r in cur.fetchall()}
        cur.execute(
            "SELECT symbol, mode, confidence_score, entered_at FROM learning_states "
            "WHERE entered_at >= ? AND entered_at <= ? ORDER BY entered_at DESC LIMIT 10",
            (start, end),
        )
        return {
            'trading_count': mc.get('TRADING', 0),
            'learning_count': mc.get('LEARNING', 0),
            'transitions': [dict(r) for r in cur.fetchall()],
        }

    def _q_confidence(self, conn, start, end) -> List[Dict]:
        cur = conn.cursor()
        cur.execute(
            "SELECT symbol, AVG(confidence_score) as avg_c, "
            "  MAX(confidence_score) as max_c, MIN(confidence_score) as min_c, "
            "  COUNT(*) as samples "
            "FROM confidence_history WHERE timestamp >= ? AND timestamp <= ? "
            "GROUP BY symbol", (start, end),
        )
        return [dict(r) for r in cur.fetchall()]

    def _q_retraining(self, conn, start, end) -> List[Dict]:
        cur = conn.cursor()
        cur.execute(
            "SELECT symbol, interval, trigger_reason, status, "
            "  validation_accuracy, duration_seconds, completed_at "
            "FROM retraining_history "
            "WHERE triggered_at >= ? AND triggered_at <= ? ORDER BY triggered_at DESC",
            (start, end),
        )
        return [dict(r) for r in cur.fetchall()]

    # =========================================================================
    # AI Commentary
    # =========================================================================

    def _generate_commentary(self, data: Dict, date_str: str) -> Dict[str, str]:
        trades = data['trades']
        signals = data['signals']
        overview = data['overview']
        learning = data['learning']
        confidence = data['confidence']
        retraining = data['retraining']

        # --- What Happened ---
        parts = []
        if overview['candles'] == 0:
            parts.append(
                "No market data was received today. The bot may have been offline "
                "or data feeds were down."
            )
        else:
            parts.append(
                f"Processed {overview['candles']:,} candles across "
                f"{overview['symbols']} symbol(s)."
            )
        if signals['above_gate'] > 0:
            ratio = signals['above_gate'] / max(overview['signals'], 1) * 100
            parts.append(
                f"{signals['above_gate']} signal(s) crossed the 80% confidence gate "
                f"({ratio:.0f}% of {overview['signals']} total)."
            )
        elif overview['signals'] > 0:
            parts.append(
                f"Generated {overview['signals']} signals but none crossed the 80% gate. "
                f"Max confidence: {signals['max_conf']:.1%}."
            )
        if trades['closed'] > 0:
            parts.append(
                f"Closed {trades['closed']} trade(s) — {trades['win_rate']:.0f}% win rate, "
                f"{trades['total_pnl']:+.3f}% total P&L."
            )
        elif trades['still_open'] > 0:
            parts.append(f"{trades['still_open']} trade(s) still open (awaiting exit).")
        if learning['trading_count'] + learning['learning_count'] > 0:
            parts.append(
                f"Mode transitions: {learning['trading_count']} TRADING, "
                f"{learning['learning_count']} LEARNING."
            )
        what_happened = ' '.join(parts)

        # --- What Should Improve ---
        parts = []
        if trades['closed'] > 0 and trades['win_rate'] < 50:
            parts.append(
                f"Win rate is {trades['win_rate']:.0f}% (below breakeven). "
                "Consider tightening the confidence gate or reviewing regime filtering."
            )
        if trades['closed'] > 0 and trades['avg_pnl'] < 0:
            parts.append(
                f"Average P&L per trade is {trades['avg_pnl']:+.3f}%. "
                "Position sizing or stop-loss levels may need adjustment."
            )
        if signals['above_gate'] == 0 and overview['signals'] > 0:
            parts.append(
                "No signals passed the 80% gate. If calibrated correctly, "
                "market was too uncertain. More features or data sources could help."
            )
        if confidence:
            avgs = [c['avg_c'] for c in confidence if c['avg_c']]
            if avgs and max(avgs) < 0.6:
                parts.append(
                    "Average confidence below 60% across all symbols — "
                    "consider retraining with more recent data."
                )
        for r in retraining:
            if r.get('status') == 'failed':
                parts.append(f"Retraining failed for {r['symbol']}@{r['interval']}.")
        if not parts:
            parts.append("No critical issues detected. Continue monitoring calibration and trends.")
        what_to_improve = ' '.join(parts)

        # --- Future Outlook ---
        parts = []
        if trades['closed'] > 0 and trades['win_rate'] >= 55 and trades['total_pnl'] > 0:
            parts.append(
                "Positive momentum — profitable with good win rate. "
                "If sustained over the week, consider increasing position sizes."
            )
        elif trades['closed'] == 0:
            parts.append(
                "No trades closed today. Watch confidence trends over the next 24-48h "
                "to gauge model alignment."
            )
        else:
            parts.append(
                "Mixed results. Let the learning system accumulate more outcomes "
                "before strategy changes. Review after 3-5 trading days."
            )
        successful = [r for r in retraining if r.get('status') == 'completed']
        if successful:
            accs = [r['validation_accuracy'] for r in successful if r.get('validation_accuracy')]
            if accs:
                parts.append(
                    f"{len(successful)} model(s) retrained "
                    f"(avg accuracy: {sum(accs)/len(accs):.1%}). "
                    "Expect improved predictions next session."
                )
        future_outlook = ' '.join(parts)

        return {
            'what_happened': what_happened,
            'what_to_improve': what_to_improve,
            'future_outlook': future_outlook,
        }

    # =========================================================================
    # HTML helpers
    # =========================================================================

    @staticmethod
    def _pnl_color(val: float) -> str:
        return '#16a34a' if val >= 0 else '#dc2626'

    @staticmethod
    def _wr_color(val: float) -> str:
        if val >= 55:
            return '#16a34a'
        return '#d97706' if val >= 45 else '#dc2626'

    @staticmethod
    def _mode_color(mode: str) -> str:
        return '#16a34a' if mode == 'TRADING' else '#d97706'

    # =========================================================================
    # HTML email — clean light theme, fully structured
    # =========================================================================

    def _build_html(self, data: Dict, commentary: Dict, date_str: str) -> str:
        esc = html_mod.escape
        o = data['overview']
        s = data['signals']
        t = data['trades']
        per_sym = data['per_symbol']
        learn = data['learning']
        conf = data['confidence']
        retrain = data['retraining']
        trade_list = data['trade_list']
        gen_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

        # Common styles
        CARD = 'background:#ffffff;border:1px solid #e5e7eb;border-radius:8px;padding:16px 20px;margin-bottom:16px'
        HEADING = 'margin:0 0 12px;font-size:14px;color:#374151;text-transform:uppercase;letter-spacing:1px;border-bottom:2px solid #e5e7eb;padding-bottom:8px'
        ROW_LABEL = 'padding:7px 0;color:#6b7280;font-size:13px'
        ROW_VALUE = 'padding:7px 0;text-align:right;font-weight:600;font-size:13px'
        TH = 'padding:8px 10px;text-align:left;color:#6b7280;font-weight:500;font-size:12px;text-transform:uppercase;letter-spacing:0.5px'
        TD = 'padding:7px 10px;font-size:13px;border-top:1px solid #f3f4f6'
        EMPTY = '<tr><td colspan="{cols}" style="padding:14px;text-align:center;color:#9ca3af;font-style:italic">{msg}</td></tr>'

        # --- Per-symbol rows ---
        sym_rows = ''
        for r in per_sym:
            wr = (r['wins'] / r['cnt'] * 100) if r['cnt'] > 0 else 0
            tp = r['total_pnl'] or 0
            sym_rows += (
                f'<tr>'
                f'<td style="{TD};font-weight:600">{esc(str(r["symbol"]))}</td>'
                f'<td style="{TD};text-align:center">{r["cnt"]}</td>'
                f'<td style="{TD};text-align:center;color:{self._wr_color(wr)};font-weight:600">{wr:.0f}%</td>'
                f'<td style="{TD};text-align:right;color:{self._pnl_color(tp)};font-weight:600">{tp:+.3f}%</td>'
                f'<td style="{TD};text-align:center">{r["avg_conf"] or 0:.1%}</td>'
                f'<td style="{TD};text-align:center">{r["max_conf"] or 0:.1%}</td>'
                f'</tr>'
            )
        if not per_sym:
            sym_rows = EMPTY.format(cols=6, msg='No closed trades today.')

        # --- Trade log rows (top 15 in email, full list in PDF) ---
        trade_rows = ''
        display_trades = trade_list[:15]
        for tr in display_trades:
            _raw_et = tr['entry_time'] or ''
            try:
                entry_t = datetime.fromisoformat(_raw_et).strftime('%H:%M:%S')
            except Exception:
                entry_t = _raw_et[-8:] if len(_raw_et) >= 8 else _raw_et
            tf_str = str(tr.get('interval') or '')
            result = 'WIN' if tr['was_correct'] == 1 else ('LOSS' if tr['was_correct'] == 0 else 'OPEN')
            r_color = '#16a34a' if result == 'WIN' else ('#dc2626' if result == 'LOSS' else '#6b7280')
            pnl_str = f"{tr['pnl_percent']:+.3f}%" if tr['pnl_percent'] is not None else '—'
            pnl_c = self._pnl_color(tr['pnl_percent']) if tr['pnl_percent'] is not None else '#6b7280'
            conf_str = f"{tr['predicted_confidence']:.0%}" if tr['predicted_confidence'] else '—'
            ep = f"{tr['entry_price']:.2f}" if tr['entry_price'] else '—'
            xp = f"{tr['exit_price']:.2f}" if tr['exit_price'] else '—'
            trade_rows += (
                f'<tr>'
                f'<td style="{TD};color:#6b7280;font-size:12px">{esc(entry_t)}</td>'
                f'<td style="{TD};color:#6b7280;font-size:12px;text-align:center">{esc(tf_str)}</td>'
                f'<td style="{TD};font-weight:600">{esc(str(tr["symbol"] or ""))}</td>'
                f'<td style="{TD};text-align:center">{esc(str(tr["predicted_direction"] or ""))}</td>'
                f'<td style="{TD};text-align:right">{ep}</td>'
                f'<td style="{TD};text-align:right">{xp}</td>'
                f'<td style="{TD};text-align:right;color:{pnl_c}">{pnl_str}</td>'
                f'<td style="{TD};text-align:center">{conf_str}</td>'
                f'<td style="{TD};text-align:center;color:{r_color};font-weight:600">{result}</td>'
                f'</tr>'
            )
        if not trade_list:
            trade_rows = EMPTY.format(cols=9, msg='No trades today.')
        extra_note = ''
        if len(trade_list) > 15:
            extra_note = f'<p style="color:#6b7280;font-size:12px;margin:8px 0 0;text-align:center">Showing 15 of {len(trade_list)} trades. Full list in PDF attachment.</p>'

        # --- Learning transitions ---
        trans_rows = ''
        for tr in learn['transitions']:
            trans_rows += (
                f'<tr>'
                f'<td style="{TD};color:#6b7280;font-size:12px">{esc(str(tr["entered_at"])[-8:])}</td>'
                f'<td style="{TD}">{esc(str(tr["symbol"]))}</td>'
                f'<td style="{TD};color:{self._mode_color(tr["mode"])};font-weight:600">{esc(str(tr["mode"]))}</td>'
                f'<td style="{TD};text-align:right">{tr["confidence_score"]:.1%}</td>'
                f'</tr>'
            )
        if not learn['transitions']:
            trans_rows = EMPTY.format(cols=4, msg='No transitions today.')

        # --- Confidence rows ---
        conf_rows = ''
        for c in conf:
            conf_rows += (
                f'<tr>'
                f'<td style="{TD}">{esc(str(c["symbol"]))}</td>'
                f'<td style="{TD};text-align:center">{c["avg_c"]:.1%}</td>'
                f'<td style="{TD};text-align:center;font-weight:600">{c["max_c"]:.1%}</td>'
                f'<td style="{TD};text-align:center">{c["min_c"]:.1%}</td>'
                f'<td style="{TD};text-align:center;color:#6b7280">{c["samples"]}</td>'
                f'</tr>'
            )
        if not conf:
            conf_rows = EMPTY.format(cols=5, msg='No confidence data today.')

        # --- Retraining rows ---
        ret_rows = ''
        for r in retrain:
            sc = '#16a34a' if r['status'] == 'completed' else '#dc2626'
            acc = f"{r['validation_accuracy']:.1%}" if r.get('validation_accuracy') else '—'
            dur = f"{r['duration_seconds']:.0f}s" if r.get('duration_seconds') else '—'
            ret_rows += (
                f'<tr>'
                f'<td style="{TD}">{esc(str(r["symbol"]))}@{esc(str(r["interval"]))}</td>'
                f'<td style="{TD};color:#6b7280">{esc(str(r["trigger_reason"]))}</td>'
                f'<td style="{TD};color:{sc};font-weight:600">{esc(str(r["status"]))}</td>'
                f'<td style="{TD};text-align:center">{acc}</td>'
                f'<td style="{TD};text-align:right;color:#6b7280">{dur}</td>'
                f'</tr>'
            )
        if not retrain:
            ret_rows = EMPTY.format(cols=5, msg='No retraining events today.')

        # Escape commentary for HTML
        c_happened = esc(commentary['what_happened'])
        c_improve = esc(commentary['what_to_improve'])
        c_outlook = esc(commentary['future_outlook'])

        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f3f4f6;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:#1f2937">
<div style="max-width:700px;margin:0 auto;padding:24px 16px">

  <!-- Header -->
  <div style="text-align:center;padding:20px 0 16px;margin-bottom:24px;border-bottom:3px solid #2563eb">
    <h1 style="margin:0;font-size:24px;color:#1e3a5f;letter-spacing:1px">{APP_NAME}</h1>
    <p style="margin:6px 0 0;color:#6b7280;font-size:13px">Daily Report &mdash; {date_str}</p>
  </div>

  <!-- 1. Overview -->
  <div style="{CARD}">
    <h2 style="{HEADING}">1. Overview</h2>
    <table style="width:100%;border-collapse:collapse">
      <tr><td style="{ROW_LABEL}">Active Symbols</td><td style="{ROW_VALUE}">{o['symbols']}</td></tr>
      <tr><td style="{ROW_LABEL}">Candles Received</td><td style="{ROW_VALUE}">{o['candles']:,}</td></tr>
      <tr><td style="{ROW_LABEL}">Signals Generated</td><td style="{ROW_VALUE}">{o['signals']}</td></tr>
      <tr><td style="{ROW_LABEL}">Trades Executed</td><td style="{ROW_VALUE}">{o['trades']}</td></tr>
    </table>
  </div>

  <!-- 2. Signals -->
  <div style="{CARD}">
    <h2 style="{HEADING}">2. Signals</h2>
    <table style="width:100%;border-collapse:collapse">
      <tr><td style="{ROW_LABEL}">BUY / SELL / HOLD</td><td style="{ROW_VALUE}">{s['buys']} / {s['sells']} / {s['holds']}</td></tr>
      <tr><td style="{ROW_LABEL}">Avg Confidence</td><td style="{ROW_VALUE}">{s['avg_conf']:.1%}</td></tr>
      <tr><td style="{ROW_LABEL}">Max Confidence</td><td style="{ROW_VALUE};font-weight:700">{s['max_conf']:.1%}</td></tr>
      <tr><td style="{ROW_LABEL}">Min Confidence</td><td style="{ROW_VALUE}">{s['min_conf']:.1%}</td></tr>
      <tr><td style="{ROW_LABEL}">Above 80% Gate</td><td style="{ROW_VALUE};color:{'#16a34a' if s['above_gate'] > 0 else '#d97706'}">{s['above_gate']}</td></tr>
    </table>
  </div>

  <!-- 3. Trade Results -->
  <div style="{CARD}">
    <h2 style="{HEADING}">3. Trade Results</h2>
    <table style="width:100%;border-collapse:collapse">
      <tr><td style="{ROW_LABEL}">Closed Trades</td><td style="{ROW_VALUE}">{t['closed']}  (W:{t['wins']} / L:{t['losses']})</td></tr>
      <tr><td style="{ROW_LABEL}">Win Rate</td><td style="{ROW_VALUE};color:{self._wr_color(t['win_rate'])}">{t['win_rate']:.1f}%</td></tr>
      <tr><td style="{ROW_LABEL}">Avg P&amp;L / Trade</td><td style="{ROW_VALUE};color:{self._pnl_color(t['avg_pnl'])}">{t['avg_pnl']:+.3f}%</td></tr>
      <tr><td style="{ROW_LABEL}">Total P&amp;L</td><td style="{ROW_VALUE};color:{self._pnl_color(t['total_pnl'])};font-size:16px">{t['total_pnl']:+.3f}%</td></tr>
      <tr><td style="{ROW_LABEL}">Avg Trade Confidence</td><td style="{ROW_VALUE}">{t['avg_conf']:.1%}</td></tr>
      <tr><td style="{ROW_LABEL}">Still Open</td><td style="{ROW_VALUE}">{t['still_open']}</td></tr>
    </table>
  </div>

  <!-- 4. Per Symbol -->
  <div style="{CARD}">
    <h2 style="{HEADING}">4. Per Symbol Breakdown</h2>
    <table style="width:100%;border-collapse:collapse">
      <tr style="border-bottom:2px solid #e5e7eb">
        <th style="{TH}">Symbol</th><th style="{TH};text-align:center">Trades</th>
        <th style="{TH};text-align:center">Win Rate</th><th style="{TH};text-align:right">P&amp;L</th>
        <th style="{TH};text-align:center">Avg Conf</th><th style="{TH};text-align:center">Max Conf</th>
      </tr>
      {sym_rows}
    </table>
  </div>

  <!-- 5. Trade Log -->
  <div style="{CARD}">
    <h2 style="{HEADING}">5. Trade Log</h2>
    <table style="width:100%;border-collapse:collapse">
      <tr style="border-bottom:2px solid #e5e7eb">
        <th style="{TH}">Time</th><th style="{TH};text-align:center">TF</th><th style="{TH}">Symbol</th>
        <th style="{TH};text-align:center">Dir</th><th style="{TH};text-align:right">Entry</th>
        <th style="{TH};text-align:right">Exit</th><th style="{TH};text-align:right">P&amp;L</th>
        <th style="{TH};text-align:center">Conf</th><th style="{TH};text-align:center">Result</th>
      </tr>
      {trade_rows}
    </table>
    {extra_note}
  </div>

  <!-- 6. Learning Activity -->
  <div style="{CARD}">
    <h2 style="{HEADING}">6. Learning Activity</h2>
    <p style="margin:0 0 10px;color:#6b7280;font-size:13px">
      <span style="color:#16a34a;font-weight:600">TRADING = {learn['trading_count']}</span> &nbsp;|&nbsp;
      <span style="color:#d97706;font-weight:600">LEARNING = {learn['learning_count']}</span>
    </p>
    <table style="width:100%;border-collapse:collapse">
      <tr style="border-bottom:2px solid #e5e7eb">
        <th style="{TH}">Time</th><th style="{TH}">Symbol</th>
        <th style="{TH}">Mode</th><th style="{TH};text-align:right">Confidence</th>
      </tr>
      {trans_rows}
    </table>
  </div>

  <!-- 7. Confidence Tracking -->
  <div style="{CARD}">
    <h2 style="{HEADING}">7. Confidence Tracking</h2>
    <table style="width:100%;border-collapse:collapse">
      <tr style="border-bottom:2px solid #e5e7eb">
        <th style="{TH}">Symbol</th><th style="{TH};text-align:center">Avg</th>
        <th style="{TH};text-align:center">Max</th><th style="{TH};text-align:center">Min</th>
        <th style="{TH};text-align:center">Samples</th>
      </tr>
      {conf_rows}
    </table>
  </div>

  <!-- 8. Retraining Events -->
  <div style="{CARD}">
    <h2 style="{HEADING}">8. Retraining Events ({len(retrain)})</h2>
    <table style="width:100%;border-collapse:collapse">
      <tr style="border-bottom:2px solid #e5e7eb">
        <th style="{TH}">Model</th><th style="{TH}">Trigger</th>
        <th style="{TH}">Status</th><th style="{TH};text-align:center">Accuracy</th>
        <th style="{TH};text-align:right">Duration</th>
      </tr>
      {ret_rows}
    </table>
  </div>

  <!-- 9. AI Daily Review -->
  <div style="background:#eff6ff;border:1px solid #bfdbfe;border-left:4px solid #2563eb;border-radius:8px;padding:18px 20px;margin-bottom:16px">
    <h2 style="margin:0 0 14px;font-size:14px;color:#1e40af;text-transform:uppercase;letter-spacing:1px">9. AI Daily Review</h2>

    <h3 style="margin:0 0 4px;font-size:12px;color:#1e40af;text-transform:uppercase;letter-spacing:0.5px">What Happened</h3>
    <p style="margin:0 0 14px;font-size:13px;color:#374151;line-height:1.6">{c_happened}</p>

    <h3 style="margin:0 0 4px;font-size:12px;color:#b45309;text-transform:uppercase;letter-spacing:0.5px">What Should Improve</h3>
    <p style="margin:0 0 14px;font-size:13px;color:#374151;line-height:1.6">{c_improve}</p>

    <h3 style="margin:0 0 4px;font-size:12px;color:#047857;text-transform:uppercase;letter-spacing:0.5px">Future Outlook</h3>
    <p style="margin:0;font-size:13px;color:#374151;line-height:1.6">{c_outlook}</p>
  </div>

  <!-- Footer -->
  <div style="text-align:center;padding:14px 0;border-top:1px solid #d1d5db;margin-top:8px;color:#9ca3af;font-size:11px">
    <p style="margin:0">Full trade log attached as PDF</p>
    <p style="margin:4px 0 0">Generated {gen_time} UTC &nbsp;|&nbsp; {APP_NAME} v{APP_VERSION}</p>
  </div>

</div>
</body></html>"""

    # =========================================================================
    # Plain text fallback
    # =========================================================================

    def _build_plain_text(self, data: Dict, commentary: Dict, date_str: str) -> str:
        o, s, t = data['overview'], data['signals'], data['trades']
        lines = [
            f"{'=' * 50}",
            f"  {APP_NAME} — DAILY REPORT: {date_str}",
            f"{'=' * 50}",
            "",
            "1. OVERVIEW",
            f"   Active Symbols:    {o['symbols']}",
            f"   Candles Received:  {o['candles']:,}",
            f"   Signals Generated: {o['signals']}",
            f"   Trades Executed:   {o['trades']}",
            "",
            "2. SIGNALS",
            f"   BUY: {s['buys']}  |  SELL: {s['sells']}  |  HOLD: {s['holds']}",
            f"   Confidence: avg={s['avg_conf']:.1%}  max={s['max_conf']:.1%}  min={s['min_conf']:.1%}",
            f"   Above 80% Gate: {s['above_gate']}",
            "",
            "3. TRADE RESULTS",
            f"   Closed: {t['closed']} (W:{t['wins']} / L:{t['losses']})",
            f"   Win Rate: {t['win_rate']:.1f}%",
            f"   Avg P&L: {t['avg_pnl']:+.3f}%  |  Total P&L: {t['total_pnl']:+.3f}%",
            f"   Avg Confidence: {t['avg_conf']:.1%}",
            f"   Still Open: {t['still_open']}",
            "",
            "4. PER SYMBOL",
        ]
        for r in data['per_symbol']:
            wr = (r['wins'] / r['cnt'] * 100) if r['cnt'] > 0 else 0
            lines.append(
                f"   {r['symbol']}: {r['cnt']} trades, WR={wr:.0f}%, "
                f"P&L={r['total_pnl'] or 0:+.3f}%"
            )
        if not data['per_symbol']:
            lines.append("   No closed trades today.")
        lines += [
            "",
            "5. TRADE LOG",
        ]
        for tr in data['trade_list'][:20]:
            entry_t = (tr['entry_time'] or '')[-8:]
            result = 'WIN' if tr['was_correct'] == 1 else ('LOSS' if tr['was_correct'] == 0 else 'OPEN')
            pnl_s = f"{tr['pnl_percent']:+.3f}%" if tr['pnl_percent'] is not None else '—'
            lines.append(f"   {entry_t}  {tr['symbol']}  {tr['predicted_direction']}  {pnl_s}  {result}")
        if not data['trade_list']:
            lines.append("   No trades today.")
        lines += [
            "",
            "6. LEARNING",
            f"   TRADING={data['learning']['trading_count']}, LEARNING={data['learning']['learning_count']}",
            "",
            f"{'─' * 50}",
            "  AI DAILY REVIEW",
            f"{'─' * 50}",
            "",
            "WHAT HAPPENED:",
            f"   {commentary['what_happened']}",
            "",
            "WHAT SHOULD IMPROVE:",
            f"   {commentary['what_to_improve']}",
            "",
            "FUTURE OUTLOOK:",
            f"   {commentary['future_outlook']}",
        ]
        return '\n'.join(lines)

    # =========================================================================
    # PDF — fully structured with all sections
    # =========================================================================

    def _build_pdf(self, data: Dict, commentary: Dict, date_str: str) -> bytes:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            leftMargin=14 * mm, rightMargin=14 * mm,
            topMargin=14 * mm, bottomMargin=14 * mm,
        )

        styles = getSampleStyleSheet()
        NAVY = colors.HexColor('#1e3a5f')
        BLUE = colors.HexColor('#2563eb')
        GRAY = colors.HexColor('#6b7280')
        DARK = colors.HexColor('#374151')
        GREEN = colors.HexColor('#16a34a')
        RED = colors.HexColor('#dc2626')
        AMBER = colors.HexColor('#d97706')
        LIGHT_BG = colors.HexColor('#f9fafb')
        BORDER = colors.HexColor('#d1d5db')
        HEADER_BG = colors.HexColor('#1e3a5f')
        BLUE_BG = colors.HexColor('#eff6ff')

        title_s = ParagraphStyle('T', parent=styles['Title'], fontSize=20, spaceAfter=4, textColor=NAVY)
        sub_s = ParagraphStyle('Sub', parent=styles['Normal'], fontSize=10, textColor=GRAY, spaceAfter=16, alignment=1)
        h2_s = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=13, spaceBefore=18, spaceAfter=8, textColor=NAVY)
        body_s = ParagraphStyle('B', parent=styles['BodyText'], fontSize=9, leading=14, textColor=DARK)
        small_s = ParagraphStyle('Sm', parent=styles['Normal'], fontSize=8, textColor=GRAY, alignment=1)
        label_s = ParagraphStyle('Lbl', parent=styles['Normal'], fontSize=9, textColor=GRAY)
        val_s = ParagraphStyle('Val', parent=styles['Normal'], fontSize=9, textColor=DARK, fontName='Helvetica-Bold')

        def make_table(header: List[str], rows: List[List[str]], col_widths=None) -> Table:
            data_rows = [header] + rows
            tbl = Table(data_rows, colWidths=col_widths)
            style = [
                ('BACKGROUND', (0, 0), (-1, 0), HEADER_BG),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ]
            tbl.setStyle(TableStyle(style))
            return tbl

        elements = []

        # Title
        elements.append(Paragraph(f"{APP_NAME} — Daily Report", title_s))
        elements.append(Paragraph(f"Date: {date_str}", sub_s))
        elements.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=12))

        # 1. Overview
        o, s, t = data['overview'], data['signals'], data['trades']
        elements.append(Paragraph("1. Overview", h2_s))
        summary = [
            ['Metric', 'Value', 'Metric', 'Value'],
            ['Symbols', str(o['symbols']), 'Signals', str(o['signals'])],
            ['Candles', f"{o['candles']:,}", 'Above 80% Gate', str(s['above_gate'])],
            ['Closed Trades', str(t['closed']), 'Win Rate', f"{t['win_rate']:.1f}%"],
            ['Total P&L', f"{t['total_pnl']:+.3f}%", 'Still Open', str(t['still_open'])],
        ]
        st = Table(summary, colWidths=[85, 75, 90, 75])
        st.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), BLUE_BG),
            ('TEXTCOLOR', (0, 0), (-1, 0), NAVY),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('ALIGN', (3, 0), (3, -1), 'RIGHT'),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        elements.append(st)

        # 2. Signals
        elements.append(Paragraph("2. Signals", h2_s))
        sig_data = [
            ['BUY', 'SELL', 'HOLD', 'Avg Conf', 'Max Conf', 'Min Conf'],
            [str(s['buys']), str(s['sells']), str(s['holds']),
             f"{s['avg_conf']:.1%}", f"{s['max_conf']:.1%}", f"{s['min_conf']:.1%}"],
        ]
        elements.append(make_table(sig_data[0], sig_data[1:], [50, 50, 50, 60, 60, 60]))

        # 3. Trade Results
        elements.append(Paragraph("3. Trade Results", h2_s))
        tr_data = [
            ['Closed', 'Wins', 'Losses', 'Win Rate', 'Avg P&L', 'Total P&L', 'Avg Conf'],
            [str(t['closed']), str(t['wins']), str(t['losses']),
             f"{t['win_rate']:.1f}%", f"{t['avg_pnl']:+.3f}%",
             f"{t['total_pnl']:+.3f}%", f"{t['avg_conf']:.1%}"],
        ]
        elements.append(make_table(tr_data[0], tr_data[1:], [45, 40, 45, 52, 55, 58, 52]))

        # 4. Per Symbol
        elements.append(Paragraph("4. Per Symbol Breakdown", h2_s))
        if data['per_symbol']:
            ps_header = ['Symbol', 'Trades', 'Wins', 'Win Rate', 'Total P&L', 'Avg Conf']
            ps_rows = []
            for r in data['per_symbol']:
                wr = (r['wins'] / r['cnt'] * 100) if r['cnt'] > 0 else 0
                ps_rows.append([
                    str(r['symbol']), str(r['cnt']), str(r['wins']),
                    f"{wr:.0f}%", f"{r['total_pnl'] or 0:+.3f}%", f"{r['avg_conf'] or 0:.1%}",
                ])
            elements.append(make_table(ps_header, ps_rows, [80, 45, 40, 52, 60, 52]))
        else:
            elements.append(Paragraph("No closed trades today.", body_s))

        # 5. Trade Log (full list in PDF)
        elements.append(Paragraph("5. Trade Log", h2_s))
        if data['trade_list']:
            t_header = ['Time', 'TF', 'Symbol', 'Dir', 'Entry', 'Exit', 'P&L%', 'Conf', 'Result']
            t_rows = []
            for tr in data['trade_list']:
                _raw_et = tr['entry_time'] or ''
                try:
                    entry_t = datetime.fromisoformat(_raw_et).strftime('%H:%M:%S')
                except Exception:
                    entry_t = _raw_et[-8:] if len(_raw_et) >= 8 else _raw_et
                tf_s = str(tr.get('interval') or '')
                result = 'WIN' if tr['was_correct'] == 1 else ('LOSS' if tr['was_correct'] == 0 else 'OPEN')
                pnl_s = f"{tr['pnl_percent']:+.3f}" if tr['pnl_percent'] is not None else '-'
                conf_s = f"{tr['predicted_confidence']:.0%}" if tr['predicted_confidence'] else '-'
                ep = f"{tr['entry_price']:.2f}" if tr['entry_price'] else '-'
                xp = f"{tr['exit_price']:.2f}" if tr['exit_price'] else '-'
                t_rows.append([entry_t, tf_s, str(tr['symbol'] or ''), str(tr['predicted_direction'] or ''),
                               ep, xp, pnl_s, conf_s, result])

            # colWidths: Time, TF, Symbol, Dir, Entry, Exit, P&L%, Conf, Result
            tbl = Table([t_header] + t_rows, colWidths=[46, 26, 56, 28, 48, 48, 40, 34, 34])
            t_style = [
                ('BACKGROUND', (0, 0), (-1, 0), HEADER_BG),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 7.5),
                ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ('ALIGN', (1, 0), (1, -1), 'CENTER'),   # TF col centred
                ('ALIGN', (4, 0), (7, -1), 'RIGHT'),    # Entry/Exit/P&L%/Conf right-aligned
                ('ALIGN', (8, 0), (8, -1), 'CENTER'),   # Result centred
            ]
            # Color WIN/LOSS/P&L
            for i, tr in enumerate(data['trade_list'], start=1):
                if tr['was_correct'] == 1:
                    t_style.append(('TEXTCOLOR', (8, i), (8, i), GREEN))
                elif tr['was_correct'] == 0:
                    t_style.append(('TEXTCOLOR', (8, i), (8, i), RED))
                if tr['pnl_percent'] is not None:
                    pc = GREEN if tr['pnl_percent'] >= 0 else RED
                    t_style.append(('TEXTCOLOR', (6, i), (6, i), pc))
            tbl.setStyle(TableStyle(t_style))
            elements.append(tbl)
        else:
            elements.append(Paragraph("No trades recorded today.", body_s))

        # 6. Learning Activity
        elements.append(Paragraph("6. Learning Activity", h2_s))
        lc = data['learning']
        elements.append(Paragraph(
            f"TRADING = {lc['trading_count']} &nbsp;&nbsp;|&nbsp;&nbsp; LEARNING = {lc['learning_count']}",
            body_s,
        ))
        if lc['transitions']:
            lt_header = ['Time', 'Symbol', 'Mode', 'Confidence']
            lt_rows = [[
                str(tr['entered_at'])[-8:], str(tr['symbol']),
                str(tr['mode']), f"{tr['confidence_score']:.1%}",
            ] for tr in lc['transitions']]
            elements.append(make_table(lt_header, lt_rows, [55, 75, 60, 60]))

        # 7. Confidence Tracking
        elements.append(Paragraph("7. Confidence Tracking", h2_s))
        if data['confidence']:
            c_header = ['Symbol', 'Avg', 'Max', 'Min', 'Samples']
            c_rows = [[
                str(c['symbol']), f"{c['avg_c']:.1%}", f"{c['max_c']:.1%}",
                f"{c['min_c']:.1%}", str(c['samples']),
            ] for c in data['confidence']]
            elements.append(make_table(c_header, c_rows, [80, 50, 50, 50, 50]))
        else:
            elements.append(Paragraph("No confidence data today.", body_s))

        # 8. Retraining Events
        elements.append(Paragraph(f"8. Retraining Events ({len(data['retraining'])})", h2_s))
        if data['retraining']:
            r_header = ['Model', 'Trigger', 'Status', 'Accuracy', 'Duration']
            r_rows = []
            for r in data['retraining']:
                acc = f"{r['validation_accuracy']:.1%}" if r.get('validation_accuracy') else '-'
                dur = f"{r['duration_seconds']:.0f}s" if r.get('duration_seconds') else '-'
                r_rows.append([
                    f"{r['symbol']}@{r['interval']}", str(r['trigger_reason']),
                    str(r['status']), acc, dur,
                ])
            elements.append(make_table(r_header, r_rows, [75, 70, 55, 50, 50]))
        else:
            elements.append(Paragraph("No retraining events today.", body_s))

        # 9. AI Daily Review
        elements.append(Paragraph("9. AI Daily Review", h2_s))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=BLUE, spaceAfter=8))

        ch_s = ParagraphStyle('CH', parent=body_s, fontSize=10, textColor=NAVY, fontName='Helvetica-Bold', spaceBefore=8, spaceAfter=2)
        ci_s = ParagraphStyle('CI', parent=ch_s, textColor=AMBER)
        co_s = ParagraphStyle('CO', parent=ch_s, textColor=GREEN)

        elements.append(Paragraph("What Happened", ch_s))
        elements.append(Paragraph(html_mod.escape(commentary['what_happened']), body_s))
        elements.append(Paragraph("What Should Improve", ci_s))
        elements.append(Paragraph(html_mod.escape(commentary['what_to_improve']), body_s))
        elements.append(Paragraph("Future Outlook", co_s))
        elements.append(Paragraph(html_mod.escape(commentary['future_outlook']), body_s))

        # Footer
        elements.append(Spacer(1, 20))
        gen_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        elements.append(Paragraph(
            f"Generated {gen_time} UTC | {APP_NAME} v{APP_VERSION}",
            small_s,
        ))

        doc.build(elements)
        pdf_data = buf.getvalue()
        buf.close()
        return pdf_data
