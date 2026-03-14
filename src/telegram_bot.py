"""
Traot Telegram Bot — Full Command System
=========================================
Remote monitoring and control of the trading bot via Telegram.

Commands:
  /start, /help       — Welcome & command reference
  /status              — Bot status, uptime, open trades
  /profit              — Total P&L, win rate
  /daily               — Today's P&L
  /weekly              — This week's P&L
  /monthly             — This month's P&L
  /balance             — Portfolio value
  /performance         — Per-pair stats
  /trades              — Recent trade history
  /confidence          — Confidence levels per pair
  /learning            — Retraining history & model status
  /start_trading       — Resume trading
  /stop_trading        — Pause trading
  /forceexit <symbol>  — Close specific trade
  /forceexit_all       — Close all open trades
  /report              — Send daily report email now
"""

import asyncio
import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Optional

from telegram import Update, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 4096


def authorized_only(func):
    """Decorator: reject commands from unauthorized users."""
    @wraps(func)
    async def wrapper(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            logger.warning(
                f"Unauthorized access attempt from chat_id={update.effective_chat.id}"
            )
            return
        return await func(self, update, context)
    return wrapper


class TelegramBot:
    """
    Telegram command handler for Traot trading bot.

    Receives a reference to the live trading runner and database
    to serve real-time data via commands.
    """

    def __init__(
        self,
        token: str,
        chat_id: str,
        runner=None,
        database=None,
        config: dict = None,
    ):
        self._token = token
        self._chat_id = str(chat_id)
        self._runner = runner
        self._config = config or {}
        self._app: Optional[Application] = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
        self._db_path = self._config.get('database', {}).get('path', 'data/trading.db')

    def _get_db_connection(self) -> sqlite3.Connection:
        """Get a read-only database connection with WAL mode."""
        conn = self._get_db_connection()
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _is_authorized(self, update: Update) -> bool:
        """Check if the message sender is the authorized user."""
        return str(update.effective_chat.id) == self._chat_id

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def start(self):
        """Start the Telegram bot in a background thread."""
        if self._running:
            logger.warning("Telegram bot already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_bot, daemon=True, name="TelegramBot"
        )
        self._thread.start()
        logger.info("Telegram bot started (polling mode)")

    def stop(self):
        """Stop the Telegram bot gracefully."""
        self._running = False
        if self._loop and self._loop.is_running():
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception as e:
                logger.debug(f"Telegram bot stop signal: {e}")
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._app = None
        self._loop = None
        logger.info("Telegram bot stopped")

    def _run_bot(self):
        """Run the bot's polling loop in a new asyncio event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._async_main())
        except Exception as e:
            logger.error(f"Telegram bot error: {e}")
        finally:
            loop.close()
            self._loop = None

    async def _async_main(self):
        """Build and run the Application."""
        builder = Application.builder().token(self._token)
        self._app = builder.build()

        # Register commands
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("help", self._cmd_help))
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("profit", self._cmd_profit))
        self._app.add_handler(CommandHandler("daily", self._cmd_daily))
        self._app.add_handler(CommandHandler("weekly", self._cmd_weekly))
        self._app.add_handler(CommandHandler("monthly", self._cmd_monthly))
        self._app.add_handler(CommandHandler("balance", self._cmd_balance))
        self._app.add_handler(CommandHandler("performance", self._cmd_performance))
        self._app.add_handler(CommandHandler("trades", self._cmd_trades))
        self._app.add_handler(CommandHandler("confidence", self._cmd_confidence))
        self._app.add_handler(CommandHandler("learning", self._cmd_learning))
        self._app.add_handler(CommandHandler("start_trading", self._cmd_start_trading))
        self._app.add_handler(CommandHandler("stop_trading", self._cmd_stop_trading))
        self._app.add_handler(CommandHandler("forceexit", self._cmd_forceexit))
        self._app.add_handler(CommandHandler("forceexit_all", self._cmd_forceexit_all))
        self._app.add_handler(CommandHandler("report", self._cmd_report))

        # Unknown command handler
        self._app.add_handler(
            MessageHandler(filters.COMMAND, self._cmd_unknown)
        )

        # Set command menu in Telegram
        await self._app.bot.set_my_commands([
            BotCommand("status", "Bot status & open trades"),
            BotCommand("profit", "Total P&L & win rate"),
            BotCommand("daily", "Today's P&L"),
            BotCommand("weekly", "This week's P&L"),
            BotCommand("monthly", "This month's P&L"),
            BotCommand("balance", "Portfolio value"),
            BotCommand("performance", "Per-pair stats"),
            BotCommand("trades", "Recent trade history"),
            BotCommand("confidence", "Confidence levels"),
            BotCommand("learning", "Model & retraining info"),
            BotCommand("start_trading", "Resume trading"),
            BotCommand("stop_trading", "Pause trading"),
            BotCommand("forceexit", "Close a trade"),
            BotCommand("forceexit_all", "Close all trades"),
            BotCommand("report", "Send daily report email"),
            BotCommand("help", "Command reference"),
        ])

        # Send startup message
        try:
            await self._app.bot.send_message(
                chat_id=self._chat_id,
                text="\u2705 Traot Trading Bot is online.\nType /help for commands.",
            )
        except Exception as e:
            logger.error(f"Failed to send startup message: {e}")

        # Start polling
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(
            drop_pending_updates=True,
            allowed_updates=["message"],
        )

        # Keep alive until stopped
        while self._running:
            await asyncio.sleep(1)

        # Graceful shutdown
        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()

    # =========================================================================
    # NOTIFICATION METHODS (called by runner/notifier)
    # =========================================================================

    async def _send_message(self, text: str, parse_mode: str = "HTML"):
        """Send a message to the authorized user, splitting if too long."""
        if not self._app or not self._app.bot:
            return
        try:
            # Split long messages at newline boundaries
            if len(text) <= MAX_MESSAGE_LENGTH:
                await self._app.bot.send_message(
                    chat_id=self._chat_id, text=text, parse_mode=parse_mode,
                )
            else:
                chunks = self._split_message(text)
                for chunk in chunks:
                    await self._app.bot.send_message(
                        chat_id=self._chat_id, text=chunk, parse_mode=parse_mode,
                    )
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

    @staticmethod
    def _split_message(text: str, max_len: int = MAX_MESSAGE_LENGTH) -> list:
        """Split a long message at newline boundaries."""
        chunks = []
        while len(text) > max_len:
            split_at = text.rfind('\n', 0, max_len)
            if split_at == -1:
                split_at = max_len
            chunks.append(text[:split_at])
            text = text[split_at:].lstrip('\n')
        if text:
            chunks.append(text)
        return chunks

    def send_notification(self, text: str):
        """Thread-safe notification sender (called from trading threads)."""
        if not self._app or not self._loop:
            return
        try:
            if self._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._send_message(text), self._loop
                )
        except Exception as e:
            logger.error(f"Telegram notification failed: {e}")

    def notify_trade_opened(self, trade: dict):
        """Send trade opened notification."""
        direction = trade.get('predicted_direction', trade.get('direction', 'UNKNOWN'))
        symbol = trade.get('symbol', '?')
        entry = trade.get('entry_price', 0)
        sl = trade.get('stop_loss', 0)
        tp = trade.get('take_profit', 0)
        conf = trade.get('predicted_confidence', trade.get('confidence', 0))
        interval = trade.get('interval', '')

        emoji = "\U0001f7e2" if direction == 'BUY' else "\U0001f534"
        now_utc = datetime.now(timezone.utc).strftime('%H:%M:%S')

        lines = [
            f"{emoji} <b>Trade Opened</b>",
            "",
            f"<b>Pair:</b> {symbol}",
            f"<b>Direction:</b> {direction}",
            f"<b>Entry:</b> ${entry:,.4f}",
        ]
        if sl:
            lines.append(f"<b>Stop Loss:</b> ${sl:,.4f}")
        if tp:
            lines.append(f"<b>Take Profit:</b> ${tp:,.4f}")
        lines.append(f"<b>Confidence:</b> {conf:.1%}")
        if interval:
            lines.append(f"<b>Timeframe:</b> {interval}")
        lines.append(f"<b>Time:</b> {now_utc} UTC")

        self.send_notification('\n'.join(lines))

    def notify_trade_closed(self, trade: dict):
        """Send trade closed notification."""
        symbol = trade.get('symbol', '?')
        pnl = trade.get('pnl_percent', 0) or 0
        was_correct = trade.get('was_correct')
        direction = trade.get('predicted_direction', trade.get('direction', '?'))
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        closed_by = trade.get('closed_by', '')
        duration = trade.get('duration', '')
        now_utc = datetime.now(timezone.utc).strftime('%H:%M:%S')

        # Freqtrade-style emoji
        if pnl >= 5.0:
            emoji = "\U0001f680"  # rocket for big wins
        elif pnl > 0:
            emoji = "\u2705"     # checkmark for small wins
        elif closed_by and 'stop' in closed_by.lower():
            emoji = "\u26a0\ufe0f"  # warning for stop-loss
        else:
            emoji = "\u274c"     # cross for losses

        outcome = "WIN" if was_correct == 1 else ("LOSS" if was_correct == 0 else "CLOSED")

        lines = [
            f"{emoji} <b>Trade Closed — {outcome}</b>",
            "",
            f"<b>Pair:</b> {symbol}",
            f"<b>Direction:</b> {direction}",
            f"<b>P&L:</b> {pnl:+.3f}%",
        ]
        if entry_price:
            lines.append(f"<b>Entry:</b> ${entry_price:,.4f}")
        if exit_price:
            lines.append(f"<b>Exit:</b> ${exit_price:,.4f}")
        if closed_by:
            lines.append(f"<b>Exit Reason:</b> {closed_by}")
        if duration:
            lines.append(f"<b>Duration:</b> {duration}")
        lines.append(f"<b>Time:</b> {now_utc} UTC")

        self.send_notification('\n'.join(lines))

    def notify_daily_summary(self, summary: dict):
        """Send daily P&L summary."""
        total_pnl = summary.get('total_pnl', 0)
        trade_count = summary.get('trade_count', 0)
        win_rate = summary.get('win_rate', 0)
        winners = summary.get('winners', 0)
        losers = summary.get('losers', 0)

        emoji = "\U0001f4c8" if total_pnl >= 0 else "\U0001f4c9"
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        msg = (
            f"{emoji} <b>Daily Summary</b>\n\n"
            f"<b>P&L:</b> {total_pnl:+.2f}%\n"
            f"<b>Trades:</b> {trade_count}\n"
            f"<b>Win Rate:</b> {win_rate:.0%}\n"
            f"<b>W/L:</b> {winners}/{losers}\n"
            f"<b>Date:</b> {today}"
        )
        self.send_notification(msg)

    # =========================================================================
    # COMMAND HANDLERS
    # =========================================================================

    @authorized_only
    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Welcome message."""
        await update.message.reply_text(
            "<b>Traot Trading Bot</b>\n\n"
            "AI-powered multi-currency trading with continuous learning.\n\n"
            "Type /help for the full command list.",
            parse_mode="HTML",
        )

    @authorized_only
    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Full command reference."""
        await update.message.reply_text(
            "<b>Traot Commands</b>\n\n"
            "<b>Monitoring:</b>\n"
            "/status — Bot status & open trades\n"
            "/profit — Total P&L & win rate\n"
            "/daily — Today's P&L\n"
            "/weekly — This week's P&L\n"
            "/monthly — This month's P&L\n"
            "/balance — Portfolio value\n"
            "/performance — Per-pair stats\n"
            "/trades — Recent trade history\n"
            "/confidence — Confidence levels\n"
            "/learning — Model & retraining info\n\n"
            "<b>Control:</b>\n"
            "/start_trading — Resume trading\n"
            "/stop_trading — Pause trading\n"
            "/forceexit &lt;SYMBOL&gt; — Close a trade\n"
            "/forceexit_all — Close all trades\n"
            "/report — Send daily report email now",
            parse_mode="HTML",
        )

    @authorized_only
    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Bot status, uptime, open trades."""
        if not self._runner:
            await update.message.reply_text("Runner not connected.")
            return

        try:
            status = self._runner.get_status()
            mode = status.get('mode', '?')
            state = status.get('status', '?')
            uptime = status.get('uptime', 'N/A')
            symbols = status.get('symbols', [])
            signals = status.get('total_signals', 0)
            orders = status.get('total_orders', 0)
            errors = status.get('errors', 0)
            portfolio = status.get('portfolio', {})
            learning = status.get('continuous_learning', {})

            # Format uptime to hours:minutes
            if uptime and uptime != 'N/A':
                uptime = uptime.split('.')[0]

            # State emoji
            state_emoji = {
                'running': '\u2705', 'paused': '\u23f8\ufe0f',
                'starting': '\u23f3', 'stopped': '\u26d4',
                'error': '\u274c',
            }.get(state, '\u2753')

            msg = (
                f"{state_emoji} <b>Traot Status</b>\n\n"
                f"<b>State:</b> {state}\n"
                f"<b>Mode:</b> {mode}\n"
                f"<b>Uptime:</b> {uptime}\n"
                f"<b>Symbols:</b> {', '.join(symbols)}\n"
                f"<b>Signals:</b> {signals}\n"
                f"<b>Orders:</b> {orders}\n"
                f"<b>Errors:</b> {errors}\n"
            )

            if portfolio:
                total_val = portfolio.get('total_value', 0)
                positions = portfolio.get('position_count', 0)
                unrealized = portfolio.get('unrealized_pnl', 0)
                msg += (
                    f"\n<b>Portfolio:</b>\n"
                    f"  Value: ${total_val:,.2f}\n"
                    f"  Positions: {positions}\n"
                    f"  Unrealized P&L: ${unrealized:,.2f}\n"
                )

            if learning:
                mode_info = learning.get('mode', '?')
                conf = learning.get('confidence', 0)
                msg += (
                    f"\n<b>Learning:</b>\n"
                    f"  Mode: {mode_info}\n"
                    f"  Confidence: {conf:.1%}\n"
                )

            # Show open holdings
            try:
                holdings = self._runner.get_holdings()
                if holdings:
                    msg += "\n<b>Open Positions:</b>\n"
                    for h in holdings[:10]:
                        sym = h.get('symbol', '?')
                        qty = h.get('quantity', 0)
                        pnl_pct = h.get('unrealized_pnl_pct', 0)
                        pnl_emoji = "\U0001f7e2" if pnl_pct >= 0 else "\U0001f534"
                        msg += f"  {pnl_emoji} {sym}: {qty:.4f} ({pnl_pct:+.2f}%)\n"
            except Exception:
                pass

            await self._send_long_message(update, msg)

        except Exception as e:
            logger.error(f"/status error: {e}", exc_info=True)
            await update.message.reply_text("Error fetching status. Check server logs.")

    @authorized_only
    async def _cmd_profit(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Total P&L, win rate, trade count."""
        try:
            stats = self._get_trade_stats()
            total_pnl = stats['total_pnl']
            win_rate = stats['win_rate']
            total = stats['total_trades']
            winners = stats['winners']
            losers = stats['losers']
            avg_pnl = stats['avg_pnl']
            best_trade = stats.get('best_trade', 0)
            worst_trade = stats.get('worst_trade', 0)

            if total == 0:
                await update.message.reply_text("No closed trades yet.")
                return

            emoji = "\U0001f4c8" if total_pnl >= 0 else "\U0001f4c9"
            msg = (
                f"{emoji} <b>Profit Summary</b>\n\n"
                f"<b>Total P&L:</b> {total_pnl:+.3f}%\n"
                f"<b>Win Rate:</b> {win_rate:.0%}\n"
                f"<b>Trades:</b> {total}\n"
                f"<b>W/L:</b> {winners}/{losers}\n"
                f"<b>Avg P&L:</b> {avg_pnl:+.3f}%\n"
                f"<b>Best:</b> {best_trade:+.3f}%\n"
                f"<b>Worst:</b> {worst_trade:+.3f}%"
            )
            await update.message.reply_text(msg, parse_mode="HTML")

        except Exception as e:
            logger.error(f"/profit error: {e}", exc_info=True)
            await update.message.reply_text("An error occurred. Check server logs.")

    @authorized_only
    async def _cmd_daily(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Today's P&L breakdown."""
        try:
            stats = self._get_period_stats(days=1, label="Today")
            await update.message.reply_text(stats, parse_mode="HTML")
        except Exception as e:
            logger.error(f"/daily error: {e}", exc_info=True)
            await update.message.reply_text("An error occurred. Check server logs.")

    @authorized_only
    async def _cmd_weekly(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """This week's P&L."""
        try:
            stats = self._get_period_stats(days=7, label="This Week")
            await update.message.reply_text(stats, parse_mode="HTML")
        except Exception as e:
            logger.error(f"/weekly error: {e}", exc_info=True)
            await update.message.reply_text("An error occurred. Check server logs.")

    @authorized_only
    async def _cmd_monthly(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """This month's P&L."""
        try:
            stats = self._get_period_stats(days=30, label="This Month")
            await update.message.reply_text(stats, parse_mode="HTML")
        except Exception as e:
            logger.error(f"/monthly error: {e}", exc_info=True)
            await update.message.reply_text("An error occurred. Check server logs.")

    @authorized_only
    async def _cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Portfolio value and positions."""
        if not self._runner:
            await update.message.reply_text("Runner not connected.")
            return

        try:
            portfolio = getattr(self._runner, '_portfolio', None)
            if not portfolio:
                await update.message.reply_text("Portfolio not available.")
                return

            summary = portfolio.get_summary()
            total = summary.get('total_value', 0)
            cash = summary.get('cash', 0)
            positions = summary.get('position_count', 0)
            unrealized = summary.get('unrealized_pnl', 0)
            realized = summary.get('realized_pnl', 0)
            total_return = summary.get('total_return_pct', 0)

            msg = (
                f"\U0001f4b0 <b>Balance</b>\n\n"
                f"<b>Total Value:</b> ${total:,.2f}\n"
                f"<b>Cash:</b> ${cash:,.2f}\n"
                f"<b>Positions:</b> {positions}\n"
                f"<b>Unrealized P&L:</b> ${unrealized:,.2f}\n"
                f"<b>Realized P&L:</b> ${realized:,.2f}\n"
                f"<b>Total Return:</b> {total_return:+.2f}%\n"
            )

            holdings = self._runner.get_holdings()
            if holdings:
                msg += "\n<b>Holdings:</b>\n"
                for h in holdings[:10]:
                    sym = h.get('symbol', '?')
                    qty = h.get('quantity', 0)
                    val = h.get('holdings_value', 0)
                    pnl_pct = h.get('unrealized_pnl_pct', 0)
                    pnl_emoji = "\U0001f7e2" if pnl_pct >= 0 else "\U0001f534"
                    msg += f"  {pnl_emoji} {sym}: {qty:.4f} (${val:,.2f}, {pnl_pct:+.2f}%)\n"

                if len(holdings) > 10:
                    msg += f"  ... and {len(holdings) - 10} more\n"

            await self._send_long_message(update, msg)

        except Exception as e:
            logger.error(f"/balance error: {e}", exc_info=True)
            await update.message.reply_text("An error occurred. Check server logs.")

    @authorized_only
    async def _cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Per-pair win rate and profit from database."""
        try:
            conn = self._get_db_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT symbol,
                           COUNT(*) as cnt,
                           SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as wins,
                           AVG(pnl_percent) as avg_pnl,
                           SUM(pnl_percent) as total_pnl,
                           AVG(predicted_confidence) as avg_conf
                    FROM trade_outcomes
                    WHERE was_correct IS NOT NULL
                      AND COALESCE(is_paper_trade, 0) = 0
                      AND COALESCE(is_replay, 0) = 0
                    GROUP BY symbol
                    ORDER BY total_pnl DESC
                ''')
                rows = cursor.fetchall()
            finally:
                conn.close()

            if not rows:
                await update.message.reply_text("No closed trades yet.")
                return

            msg = "\U0001f4ca <b>Per-Pair Performance</b>\n\n"
            for r in rows:
                symbol = r['symbol']
                cnt = r['cnt']
                wins = r['wins']
                wr = (wins / cnt * 100) if cnt > 0 else 0
                total_pnl = r['total_pnl'] or 0
                avg_conf = r['avg_conf'] or 0

                emoji = "\u2705" if total_pnl >= 0 else "\u274c"
                msg += (
                    f"{emoji} <b>{symbol}</b>\n"
                    f"  Trades: {cnt} | WR: {wr:.0f}% | P&L: {total_pnl:+.3f}%\n"
                    f"  Avg Conf: {avg_conf:.1%}\n\n"
                )

            await self._send_long_message(update, msg)

        except Exception as e:
            logger.error(f"/performance error: {e}", exc_info=True)
            await update.message.reply_text("An error occurred. Check server logs.")

    @authorized_only
    async def _cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Recent 10 trades with details."""
        try:
            conn = self._get_db_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT symbol, predicted_direction, pnl_percent, was_correct,
                           entry_price, exit_price, entry_time, exit_time,
                           interval, predicted_confidence, closed_by
                    FROM trade_outcomes
                    WHERE COALESCE(is_paper_trade, 0) = 0
                      AND COALESCE(is_replay, 0) = 0
                    ORDER BY COALESCE(exit_time, entry_time) DESC
                    LIMIT 10
                ''')
                trades = [dict(row) for row in cursor.fetchall()]
            finally:
                conn.close()

            if not trades:
                await update.message.reply_text("No trades found.")
                return

            msg = "\U0001f4dc <b>Recent Trades</b>\n\n"
            for t in trades:
                sym = t.get('symbol', '?')
                direction = t.get('predicted_direction', '?')
                pnl = t.get('pnl_percent')
                was_correct = t.get('was_correct')
                entry = t.get('entry_price') or 0
                exit_p = t.get('exit_price')
                entry_time = t.get('entry_time') or '?'
                interval = t.get('interval') or ''
                conf = t.get('predicted_confidence') or 0
                closed_by = t.get('closed_by') or ''

                # Format time
                if isinstance(entry_time, str) and len(entry_time) >= 16:
                    time_display = entry_time[5:16]  # MM-DD HH:MM
                else:
                    time_display = str(entry_time)

                # Status
                if was_correct is None:
                    outcome_emoji = "\u23f3"  # hourglass = still open
                    status = "OPEN"
                    pnl_str = "—"
                elif was_correct:
                    outcome_emoji = "\u2705"
                    status = "WIN"
                    pnl_str = f"{pnl:+.3f}%" if pnl is not None else "—"
                else:
                    outcome_emoji = "\u274c"
                    status = "LOSS"
                    pnl_str = f"{pnl:+.3f}%" if pnl is not None else "—"

                msg += (
                    f"{outcome_emoji} <b>{sym}</b> {direction} [{interval}]\n"
                    f"  Entry: ${entry:,.2f}"
                )
                if exit_p:
                    msg += f" → ${exit_p:,.2f}"
                msg += f"\n  P&L: {pnl_str} | Conf: {conf:.0%}"
                if closed_by:
                    msg += f" | {closed_by}"
                msg += f"\n  {time_display}\n\n"

            await self._send_long_message(update, msg)

        except Exception as e:
            logger.error(f"/trades error: {e}", exc_info=True)
            await update.message.reply_text("An error occurred. Check server logs.")

    @authorized_only
    async def _cmd_confidence(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Current confidence levels per pair."""
        try:
            conn = self._get_db_connection()
            try:
                cursor = conn.cursor()
                # Get latest confidence per symbol+interval
                cursor.execute('''
                    SELECT ch.symbol, ch.confidence_score, ch.mode, ch.timestamp, ch.interval
                    FROM confidence_history ch
                    INNER JOIN (
                        SELECT symbol, interval, MAX(timestamp) as max_ts
                        FROM confidence_history
                        GROUP BY symbol, interval
                    ) latest ON ch.symbol = latest.symbol
                        AND ch.interval = latest.interval
                        AND ch.timestamp = latest.max_ts
                    ORDER BY ch.confidence_score DESC
                ''')
                rows = cursor.fetchall()
            finally:
                conn.close()

            if not rows:
                await update.message.reply_text("No confidence data yet.")
                return

            msg = "\U0001f4ca <b>Confidence Levels</b>\n\n"
            for row in rows:
                symbol = row['symbol']
                conf = row['confidence_score']
                mode = row['mode']
                ts = row['timestamp']
                interval = row['interval'] or ''

                if isinstance(ts, str) and len(ts) >= 16:
                    ts = ts[5:16]

                bar = self._confidence_bar(conf)
                mode_emoji = "\U0001f7e2" if mode == 'TRADING' else "\U0001f7e1"
                msg += (
                    f"{mode_emoji} <b>{symbol}</b> [{interval}]\n"
                    f"  {conf:.1%} {bar}\n"
                    f"  Mode: {mode} | Updated: {ts}\n\n"
                )

            await self._send_long_message(update, msg)

        except Exception as e:
            logger.error(f"/confidence error: {e}", exc_info=True)
            await update.message.reply_text("An error occurred. Check server logs.")

    @authorized_only
    async def _cmd_learning(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Retraining history and model status."""
        try:
            conn = self._get_db_connection()
            try:
                cursor = conn.cursor()

                # Get latest learning state per symbol
                cursor.execute('''
                    SELECT ls.symbol, ls.interval, ls.mode, ls.confidence_score, ls.entered_at
                    FROM learning_states ls
                    INNER JOIN (
                        SELECT symbol, interval, MAX(entered_at) as max_at
                        FROM learning_states
                        GROUP BY symbol, interval
                    ) latest ON ls.symbol = latest.symbol
                        AND ls.interval = latest.interval
                        AND ls.entered_at = latest.max_at
                    ORDER BY ls.symbol
                ''')
                states = cursor.fetchall()

                # Get recent retraining events
                cursor.execute('''
                    SELECT symbol, interval, trigger_reason, status,
                           validation_accuracy, duration_seconds, triggered_at
                    FROM retraining_history
                    ORDER BY triggered_at DESC
                    LIMIT 10
                ''')
                retraining = cursor.fetchall()
            finally:
                conn.close()

            msg = "\U0001f9e0 <b>Learning Status</b>\n\n"

            if states:
                msg += "<b>Current Modes:</b>\n"
                for s in states:
                    mode = s['mode']
                    mode_emoji = "\U0001f7e2" if mode == 'TRADING' else "\U0001f7e1"
                    msg += (
                        f"  {mode_emoji} {s['symbol']} [{s['interval']}]: "
                        f"{mode} ({s['confidence_score']:.1%})\n"
                    )
                msg += "\n"

            if retraining:
                msg += "<b>Recent Retraining:</b>\n"
                for r in retraining:
                    status = r['status']
                    s_emoji = "\u2705" if status == 'completed' else "\u274c"
                    acc = f"{r['validation_accuracy']:.1%}" if r['validation_accuracy'] else "—"
                    dur = f"{r['duration_seconds']:.0f}s" if r['duration_seconds'] else "—"
                    triggered = r['triggered_at'] or '?'
                    if isinstance(triggered, str) and len(triggered) >= 16:
                        triggered = triggered[5:16]

                    msg += (
                        f"  {s_emoji} {r['symbol']}@{r['interval']}\n"
                        f"    {r['trigger_reason']} | Acc: {acc} | {dur}\n"
                        f"    {triggered}\n\n"
                    )
            elif not states:
                msg += "No learning data yet."

            await self._send_long_message(update, msg)

        except Exception as e:
            logger.error(f"/learning error: {e}", exc_info=True)
            await update.message.reply_text("An error occurred. Check server logs.")

    @authorized_only
    async def _cmd_start_trading(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Resume trading."""
        if not self._runner:
            await update.message.reply_text("Runner not connected.")
            return

        try:
            from src.live_trading.runner import RunnerStatus
            current = self._runner.status
            if current == RunnerStatus.PAUSED:
                self._runner.resume()
                await update.message.reply_text(
                    "\u25b6\ufe0f Trading resumed."
                )
            elif current == RunnerStatus.RUNNING:
                await update.message.reply_text(
                    "Trading is already running."
                )
            else:
                await update.message.reply_text(
                    f"Cannot resume from state: {current.value}"
                )
        except Exception as e:
            logger.error(f"/start_trading error: {e}", exc_info=True)
            await update.message.reply_text("An error occurred. Check server logs.")

    @authorized_only
    async def _cmd_stop_trading(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Pause trading (keeps bot running)."""
        if not self._runner:
            await update.message.reply_text("Runner not connected.")
            return

        try:
            from src.live_trading.runner import RunnerStatus
            current = self._runner.status
            if current == RunnerStatus.RUNNING:
                self._runner.pause()
                await update.message.reply_text(
                    "\u23f8\ufe0f Trading paused. Bot still running.\n"
                    "Use /start_trading to resume."
                )
            elif current == RunnerStatus.PAUSED:
                await update.message.reply_text(
                    "Trading is already paused."
                )
            else:
                await update.message.reply_text(
                    f"Cannot pause from state: {current.value}"
                )
        except Exception as e:
            logger.error(f"/stop_trading error: {e}", exc_info=True)
            await update.message.reply_text("An error occurred. Check server logs.")

    @authorized_only
    async def _cmd_forceexit(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Close a specific trade by symbol."""
        if not self._runner:
            await update.message.reply_text("Runner not connected.")
            return

        args = context.args
        if not args:
            # Show open positions to help user
            try:
                holdings = self._runner.get_holdings()
                if holdings:
                    symbols = [h.get('symbol', '?') for h in holdings]
                    await update.message.reply_text(
                        "Usage: /forceexit BTC/USDT\n\n"
                        f"<b>Open positions:</b>\n" +
                        "\n".join(f"  • {s}" for s in symbols),
                        parse_mode="HTML",
                    )
                else:
                    await update.message.reply_text("No open positions.")
            except Exception:
                await update.message.reply_text(
                    "Usage: /forceexit BTC/USDT\nSpecify the symbol to close."
                )
            return

        symbol = args[0].upper()
        try:
            holdings = self._runner.get_holdings()
            position = next(
                (h for h in holdings if h.get('symbol') == symbol), None
            )
            if not position:
                await update.message.reply_text(
                    f"No open position for {symbol}."
                )
                return

            if hasattr(self._runner, 'force_exit'):
                self._runner.force_exit(symbol)
                await update.message.reply_text(
                    f"\u26a0\ufe0f Force exit requested for {symbol}."
                )
            else:
                await update.message.reply_text(
                    f"\u26a0\ufe0f Force exit not yet implemented.\n"
                    f"Please close {symbol} manually on the exchange."
                )

        except Exception as e:
            logger.error(f"/forceexit error: {e}", exc_info=True)
            await update.message.reply_text("An error occurred. Check server logs.")

    @authorized_only
    async def _cmd_forceexit_all(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Close all open trades."""
        if not self._runner:
            await update.message.reply_text("Runner not connected.")
            return

        try:
            holdings = self._runner.get_holdings()
            if not holdings:
                await update.message.reply_text("No open positions.")
                return

            if not hasattr(self._runner, 'force_exit'):
                symbols = [h.get('symbol', '?') for h in holdings]
                await update.message.reply_text(
                    "\u26a0\ufe0f Force exit not yet implemented.\n"
                    f"Please close manually: {', '.join(symbols)}"
                )
                return

            closed = 0
            failed = 0
            for h in holdings:
                symbol = h.get('symbol')
                if symbol:
                    try:
                        self._runner.force_exit(symbol)
                        closed += 1
                    except Exception as e:
                        logger.error(f"Force exit failed for {symbol}: {e}")
                        failed += 1

            msg = f"\u26a0\ufe0f Force exit requested for {closed} position(s)."
            if failed:
                msg += f"\n{failed} position(s) failed to close. Check logs."
            await update.message.reply_text(msg)

        except Exception as e:
            logger.error(f"/forceexit_all error: {e}", exc_info=True)
            await update.message.reply_text("An error occurred. Check server logs.")

    @authorized_only
    async def _cmd_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Trigger the daily email report immediately."""
        if not self._runner:
            await update.message.reply_text("Runner not connected.")
            return

        try:
            report_scheduler = getattr(self._runner, '_daily_report', None)
            if not report_scheduler:
                await update.message.reply_text(
                    "Daily report scheduler not available.\n"
                    "Check SMTP_USER and SMTP_PASSWORD in .env"
                )
                return

            if not report_scheduler.is_available:
                await update.message.reply_text(
                    "Daily report disabled — SMTP credentials not configured.\n"
                    "Set SMTP_USER and SMTP_PASSWORD in .env"
                )
                return

            # Parse optional date argument (e.g., /report 2026-03-12)
            date_str = None
            if context.args:
                raw_date = context.args[0]
                try:
                    datetime.strptime(raw_date, '%Y-%m-%d')
                    date_str = raw_date
                except ValueError:
                    await update.message.reply_text(
                        "Invalid date format. Use YYYY-MM-DD.\n"
                        "Example: /report 2026-03-12"
                    )
                    return

            await update.message.reply_text("Generating and sending daily report...")

            import concurrent.futures
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                await loop.run_in_executor(
                    pool, lambda: report_scheduler.send_now(date_str=date_str)
                )

            await update.message.reply_text(
                "\u2705 Daily report sent to your email.\n"
                "Check your inbox (and spam folder)."
            )
        except Exception as e:
            logger.error(f"/report error: {e}", exc_info=True)
            await update.message.reply_text("An error occurred. Check server logs.")

    @authorized_only
    async def _cmd_unknown(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle unknown commands."""
        await update.message.reply_text(
            "Unknown command. Type /help for available commands."
        )

    # =========================================================================
    # DATA HELPERS
    # =========================================================================

    def _get_trade_stats(self, days: int = None) -> dict:
        """Get aggregated trade stats from database."""
        empty = {
            'total_pnl': 0, 'win_rate': 0, 'total_trades': 0,
            'winners': 0, 'losers': 0, 'avg_pnl': 0,
            'best_trade': 0, 'worst_trade': 0,
        }

        conn = self._get_db_connection()
        try:
            cursor = conn.cursor()
            if days:
                cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
                cursor.execute('''
                    SELECT pnl_percent, was_correct
                    FROM trade_outcomes
                    WHERE exit_time >= ?
                      AND was_correct IS NOT NULL
                      AND COALESCE(is_paper_trade, 0) = 0
                      AND COALESCE(is_replay, 0) = 0
                    ORDER BY exit_time DESC
                ''', (cutoff,))
            else:
                cursor.execute('''
                    SELECT pnl_percent, was_correct
                    FROM trade_outcomes
                    WHERE was_correct IS NOT NULL
                      AND COALESCE(is_paper_trade, 0) = 0
                      AND COALESCE(is_replay, 0) = 0
                    ORDER BY exit_time DESC
                ''')

            rows = cursor.fetchall()
        finally:
            conn.close()

        if not rows:
            return empty

        winners = sum(1 for r in rows if r['was_correct'])
        losers = len(rows) - winners
        pnls = [r['pnl_percent'] for r in rows if r['pnl_percent'] is not None]
        total_pnl = sum(pnls)
        avg_pnl = total_pnl / len(pnls) if pnls else 0
        best = max(pnls) if pnls else 0
        worst = min(pnls) if pnls else 0

        return {
            'total_pnl': total_pnl,
            'win_rate': winners / len(rows) if rows else 0,
            'total_trades': len(rows),
            'winners': winners,
            'losers': losers,
            'avg_pnl': avg_pnl,
            'best_trade': best,
            'worst_trade': worst,
        }

    def _get_period_stats(self, days: int, label: str) -> str:
        """Get P&L stats for a period, formatted as HTML."""
        stats = self._get_trade_stats(days=days)
        total_pnl = stats['total_pnl']
        win_rate = stats['win_rate']
        total = stats['total_trades']
        winners = stats['winners']
        losers = stats['losers']
        avg_pnl = stats['avg_pnl']

        emoji = "\U0001f4c8" if total_pnl >= 0 else "\U0001f4c9"

        if total == 0:
            return f"<b>{label}</b>\n\nNo trades in this period."

        return (
            f"{emoji} <b>{label}</b>\n\n"
            f"<b>P&L:</b> {total_pnl:+.3f}%\n"
            f"<b>Win Rate:</b> {win_rate:.0%}\n"
            f"<b>Trades:</b> {total}\n"
            f"<b>W/L:</b> {winners}/{losers}\n"
            f"<b>Avg P&L:</b> {avg_pnl:+.3f}%"
        )

    async def _send_long_message(self, update: Update, text: str):
        """Send a message, splitting if it exceeds Telegram's limit."""
        if len(text) <= MAX_MESSAGE_LENGTH:
            await update.message.reply_text(text, parse_mode="HTML")
        else:
            chunks = self._split_message(text)
            for chunk in chunks:
                await update.message.reply_text(chunk, parse_mode="HTML")

    @staticmethod
    def _confidence_bar(conf: float) -> str:
        """Render a visual confidence bar."""
        filled = max(0, min(10, int(conf * 10)))
        empty = 10 - filled
        return "\u2588" * filled + "\u2591" * empty
