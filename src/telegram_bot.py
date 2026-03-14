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
import threading
from datetime import datetime, timedelta
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


def authorized_only(func):
    """Decorator: reject commands from unauthorized users."""
    @wraps(func)
    async def wrapper(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            logger.warning(
                f"Unauthorized access attempt from chat_id={update.effective_chat.id}"
            )
            return  # Silent ignore — no response to prevent enumeration
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
        self._database = database
        self._config = config or {}
        self._app: Optional[Application] = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False

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
                text="Traot Trading Bot is online.\nType /help for commands.",
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
        """Send a message to the authorized user."""
        if not self._app or not self._app.bot:
            return
        try:
            await self._app.bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode=parse_mode,
            )
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

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
        direction = trade.get('direction', 'UNKNOWN')
        symbol = trade.get('symbol', '?')
        entry = trade.get('entry_price', 0)
        sl = trade.get('stop_loss', 0)
        tp = trade.get('take_profit', 0)
        conf = trade.get('confidence', 0)

        emoji = "\U0001f7e2" if direction == 'BUY' else "\U0001f534"
        msg = (
            f"{emoji} <b>Trade Opened</b>\n\n"
            f"<b>Pair:</b> {symbol}\n"
            f"<b>Direction:</b> {direction}\n"
            f"<b>Entry:</b> ${entry:,.4f}\n"
            f"<b>Stop Loss:</b> ${sl:,.4f}\n"
            f"<b>Take Profit:</b> ${tp:,.4f}\n"
            f"<b>Confidence:</b> {conf:.1%}\n"
            f"<b>Time:</b> {datetime.utcnow().strftime('%H:%M:%S')} UTC"
        )
        self.send_notification(msg)

    def notify_trade_closed(self, trade: dict):
        """Send trade closed notification."""
        symbol = trade.get('symbol', '?')
        pnl = trade.get('pnl_percent', 0)
        outcome = trade.get('outcome', 'UNKNOWN')
        duration = trade.get('duration', '?')

        emoji = "\u2705" if pnl >= 0 else "\u274c"
        msg = (
            f"{emoji} <b>Trade Closed</b>\n\n"
            f"<b>Pair:</b> {symbol}\n"
            f"<b>Outcome:</b> {outcome}\n"
            f"<b>P&L:</b> {pnl:+.2f}%\n"
            f"<b>Duration:</b> {duration}\n"
            f"<b>Time:</b> {datetime.utcnow().strftime('%H:%M:%S')} UTC"
        )
        self.send_notification(msg)

    def notify_daily_summary(self, summary: dict):
        """Send daily P&L summary."""
        total_pnl = summary.get('total_pnl', 0)
        trade_count = summary.get('trade_count', 0)
        win_rate = summary.get('win_rate', 0)
        winners = summary.get('winners', 0)
        losers = summary.get('losers', 0)

        emoji = "\U0001f4c8" if total_pnl >= 0 else "\U0001f4c9"
        msg = (
            f"{emoji} <b>Daily Summary</b>\n\n"
            f"<b>P&L:</b> {total_pnl:+.2f}%\n"
            f"<b>Trades:</b> {trade_count}\n"
            f"<b>Win Rate:</b> {win_rate:.0%}\n"
            f"<b>W/L:</b> {winners}/{losers}\n"
            f"<b>Date:</b> {datetime.utcnow().strftime('%Y-%m-%d')}"
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

            # Truncate uptime to hours:minutes
            if uptime and uptime != 'N/A':
                uptime = uptime.split('.')[0]

            msg = (
                f"<b>Traot Status</b>\n\n"
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
                msg += (
                    f"\n<b>Portfolio:</b>\n"
                    f"  Value: ${total_val:,.2f}\n"
                    f"  Positions: {positions}\n"
                )

            if learning:
                mode_info = learning.get('mode', '?')
                conf = learning.get('confidence', 0)
                msg += (
                    f"\n<b>Learning:</b>\n"
                    f"  Mode: {mode_info}\n"
                    f"  Confidence: {conf:.1%}\n"
                )

            await update.message.reply_text(msg, parse_mode="HTML")

        except Exception as e:
            logger.error(f"/status error: {e}")
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

            emoji = "\U0001f4c8" if total_pnl >= 0 else "\U0001f4c9"
            msg = (
                f"{emoji} <b>Profit Summary</b>\n\n"
                f"<b>Total P&L:</b> {total_pnl:+.2f}%\n"
                f"<b>Win Rate:</b> {win_rate:.0%}\n"
                f"<b>Trades:</b> {total}\n"
                f"<b>W/L:</b> {winners}/{losers}\n"
                f"<b>Avg P&L:</b> {avg_pnl:+.2f}%"
            )
            await update.message.reply_text(msg, parse_mode="HTML")

        except Exception as e:
            logger.error(f"/profit error: {e}")
            await update.message.reply_text("An error occurred. Check server logs.")

    @authorized_only
    async def _cmd_daily(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Today's P&L breakdown."""
        try:
            stats = self._get_period_stats(days=1, label="Today")
            await update.message.reply_text(stats, parse_mode="HTML")
        except Exception as e:
            logger.error(f"/daily error: {e}")
            await update.message.reply_text("An error occurred. Check server logs.")

    @authorized_only
    async def _cmd_weekly(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """This week's P&L."""
        try:
            stats = self._get_period_stats(days=7, label="This Week")
            await update.message.reply_text(stats, parse_mode="HTML")
        except Exception as e:
            logger.error(f"/weekly error: {e}")
            await update.message.reply_text("An error occurred. Check server logs.")

    @authorized_only
    async def _cmd_monthly(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """This month's P&L."""
        try:
            stats = self._get_period_stats(days=30, label="This Month")
            await update.message.reply_text(stats, parse_mode="HTML")
        except Exception as e:
            logger.error(f"/monthly error: {e}")
            await update.message.reply_text("An error occurred. Check server logs.")

    @authorized_only
    async def _cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Portfolio value and positions."""
        if not self._runner or not self._runner._portfolio:
            await update.message.reply_text("Portfolio not available.")
            return

        try:
            summary = self._runner._portfolio.get_summary()
            total = summary.get('total_value', 0)
            cash = summary.get('cash', 0)
            positions = summary.get('position_count', 0)
            holdings = self._runner.get_holdings()

            msg = (
                f"<b>Balance</b>\n\n"
                f"<b>Total Value:</b> ${total:,.2f}\n"
                f"<b>Cash:</b> ${cash:,.2f}\n"
                f"<b>Positions:</b> {positions}\n"
            )

            if holdings:
                msg += "\n<b>Holdings:</b>\n"
                for h in holdings[:10]:
                    sym = h.get('symbol', '?')
                    qty = h.get('quantity', 0)
                    val = h.get('value', 0)
                    pnl = h.get('unrealized_pnl', 0)
                    msg += f"  {sym}: {qty:.4f} (${val:,.2f}, {pnl:+.2f}%)\n"

            await update.message.reply_text(msg, parse_mode="HTML")

        except Exception as e:
            logger.error(f"/balance error: {e}")
            await update.message.reply_text("An error occurred. Check server logs.")

    @authorized_only
    async def _cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Per-pair win rate and profit."""
        if not self._database:
            await update.message.reply_text("Database not connected.")
            return

        try:
            symbols = self._get_symbols()
            if not symbols:
                await update.message.reply_text("No symbols configured.")
                return

            msg = "<b>Per-Pair Performance</b>\n\n"

            for symbol in symbols:
                outcomes = self._database.get_recent_outcomes(symbol, limit=1000)
                if not outcomes:
                    msg += f"<b>{symbol}:</b> No trades\n"
                    continue

                wins = sum(1 for o in outcomes if o.get('was_correct'))
                total = len(outcomes)
                wr = wins / total if total > 0 else 0
                pnl_vals = [
                    o.get('pnl_percent', 0) for o in outcomes
                    if o.get('pnl_percent') is not None
                ]
                total_pnl = sum(pnl_vals)

                emoji = "\u2705" if total_pnl >= 0 else "\u274c"
                msg += (
                    f"{emoji} <b>{symbol}:</b> "
                    f"WR {wr:.0%} | "
                    f"P&L {total_pnl:+.2f}% | "
                    f"{total} trades\n"
                )

            await update.message.reply_text(msg, parse_mode="HTML")

        except Exception as e:
            logger.error(f"/performance error: {e}")
            await update.message.reply_text("An error occurred. Check server logs.")

    @authorized_only
    async def _cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Recent 10 trades."""
        if not self._database:
            await update.message.reply_text("Database not connected.")
            return

        try:
            trades = self._get_all_recent_trades(limit=10)
            if not trades:
                await update.message.reply_text("No trades found.")
                return

            msg = "<b>Recent Trades</b>\n\n"
            for t in trades:
                sym = t.get('symbol', '?')
                direction = t.get('direction', '?')
                pnl = t.get('pnl_percent', 0) or 0
                outcome = "\u2705" if t.get('was_correct') else "\u274c"
                entry_time = t.get('entry_time', '?')
                if isinstance(entry_time, str) and len(entry_time) > 16:
                    entry_time = entry_time[:16]

                msg += (
                    f"{outcome} {sym} {direction} "
                    f"{pnl:+.2f}% ({entry_time})\n"
                )

            await update.message.reply_text(msg, parse_mode="HTML")

        except Exception as e:
            logger.error(f"/trades error: {e}")
            await update.message.reply_text("An error occurred. Check server logs.")

    @authorized_only
    async def _cmd_confidence(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Current confidence levels per pair."""
        if not self._database:
            await update.message.reply_text("Database not connected.")
            return

        try:
            symbols = self._get_symbols()
            msg = "<b>Confidence Levels</b>\n\n"

            for symbol in symbols:
                # Get latest confidence from history
                with self._database.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT confidence_score, mode, timestamp
                        FROM confidence_history
                        WHERE symbol = ?
                        ORDER BY timestamp DESC
                        LIMIT 1
                    ''', (symbol,))
                    row = cursor.fetchone()

                if row:
                    conf = row['confidence_score']
                    mode = row['mode']
                    ts = row['timestamp']
                    if isinstance(ts, str) and len(ts) > 16:
                        ts = ts[:16]

                    bar = self._confidence_bar(conf)
                    msg += f"<b>{symbol}:</b> {conf:.1%} {bar} ({mode}) @ {ts}\n"
                else:
                    msg += f"<b>{symbol}:</b> No data\n"

            await update.message.reply_text(msg, parse_mode="HTML")

        except Exception as e:
            logger.error(f"/confidence error: {e}")
            await update.message.reply_text("An error occurred. Check server logs.")

    @authorized_only
    async def _cmd_learning(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Retraining history and model status."""
        if not self._database:
            await update.message.reply_text("Database not connected.")
            return

        try:
            symbols = self._get_symbols()
            msg = "<b>Learning Status</b>\n\n"

            for symbol in symbols:
                history = self._database.get_retraining_history(symbol, limit=3)
                if not history:
                    msg += f"<b>{symbol}:</b> No retraining yet\n\n"
                    continue

                msg += f"<b>{symbol}:</b>\n"
                for h in history:
                    triggered = h.get('triggered_at', '?')
                    if isinstance(triggered, str) and len(triggered) > 16:
                        triggered = triggered[:16]
                    reason = h.get('trigger_reason', '?')
                    status = h.get('status', '?')
                    acc_before = h.get('accuracy_before')
                    acc_after = h.get('accuracy_after')

                    acc_str = ""
                    if acc_before is not None and acc_after is not None:
                        emoji = "\U0001f4c8" if acc_after > acc_before else "\U0001f4c9"
                        acc_str = f" {emoji} {acc_before:.1%} -> {acc_after:.1%}"

                    msg += f"  {triggered} | {reason} | {status}{acc_str}\n"
                msg += "\n"

            await update.message.reply_text(msg, parse_mode="HTML")

        except Exception as e:
            logger.error(f"/learning error: {e}")
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
            logger.error(f"/start_trading error: {e}")
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
            logger.error(f"/stop_trading error: {e}")
            await update.message.reply_text("An error occurred. Check server logs.")

    @authorized_only
    async def _cmd_forceexit(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Close a specific trade by symbol."""
        if not self._runner:
            await update.message.reply_text("Runner not connected.")
            return

        args = context.args
        if not args:
            await update.message.reply_text(
                "Usage: /forceexit BTC/USDT\n"
                "Specify the symbol to close."
            )
            return

        symbol = args[0].upper()
        try:
            # Check if we have a position in this symbol
            holdings = self._runner.get_holdings()
            position = next(
                (h for h in holdings if h.get('symbol') == symbol), None
            )
            if not position:
                await update.message.reply_text(
                    f"No open position for {symbol}."
                )
                return

            # Request market sell via the runner
            if hasattr(self._runner, 'force_exit'):
                self._runner.force_exit(symbol)
                await update.message.reply_text(
                    f"\u26a0\ufe0f Force exit requested for {symbol}."
                )
            else:
                await update.message.reply_text(
                    "Force exit not yet implemented in runner."
                )

        except Exception as e:
            logger.error(f"/forceexit error: {e}")
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

            closed = 0
            failed = 0
            for h in holdings:
                symbol = h.get('symbol')
                if symbol and hasattr(self._runner, 'force_exit'):
                    try:
                        self._runner.force_exit(symbol)
                        closed += 1
                    except Exception as e:
                        logger.error(f"Force exit failed for {symbol}: {e}")
                        failed += 1

            if closed > 0:
                await update.message.reply_text(
                    f"\u26a0\ufe0f Force exit requested for {closed} positions."
                )
            else:
                await update.message.reply_text(
                    "Force exit not yet implemented in runner."
                )

        except Exception as e:
            logger.error(f"/forceexit_all error: {e}")
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

            # Run in thread to avoid blocking the bot
            import concurrent.futures
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                await loop.run_in_executor(
                    pool, lambda: report_scheduler.send_now(date_str=date_str)
                )

            await update.message.reply_text(
                "Daily report sent to your email.\n"
                "Check your inbox (and spam folder)."
            )
        except Exception as e:
            logger.error(f"/report error: {e}")
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

    def _get_symbols(self) -> list:
        """Get configured trading symbols."""
        if self._runner and hasattr(self._runner, '_symbols'):
            return list(self._runner._symbols.keys())
        if self._config:
            return self._config.get('symbols', [])
        return []

    def _get_trade_stats(self, days: int = None) -> dict:
        """Get aggregated trade stats from database."""
        if not self._database:
            return {
                'total_pnl': 0, 'win_rate': 0, 'total_trades': 0,
                'winners': 0, 'losers': 0, 'avg_pnl': 0,
            }

        cutoff = None
        if days:
            cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        with self._database.connection() as conn:
            cursor = conn.cursor()
            if cutoff:
                cursor.execute('''
                    SELECT pnl_percent, was_correct, direction, symbol
                    FROM trade_outcomes
                    WHERE exit_time >= ?
                      AND COALESCE(is_paper_trade, 0) = 0
                      AND COALESCE(is_replay, 0) = 0
                    ORDER BY exit_time DESC
                ''', (cutoff,))
            else:
                cursor.execute('''
                    SELECT pnl_percent, was_correct, direction, symbol
                    FROM trade_outcomes
                    WHERE COALESCE(is_paper_trade, 0) = 0
                      AND COALESCE(is_replay, 0) = 0
                    ORDER BY exit_time DESC
                ''')

            rows = cursor.fetchall()

        if not rows:
            return {
                'total_pnl': 0, 'win_rate': 0, 'total_trades': 0,
                'winners': 0, 'losers': 0, 'avg_pnl': 0,
            }

        winners = sum(1 for r in rows if r['was_correct'])
        losers = len(rows) - winners
        pnls = [r['pnl_percent'] for r in rows if r['pnl_percent'] is not None]
        total_pnl = sum(pnls)
        avg_pnl = total_pnl / len(pnls) if pnls else 0

        return {
            'total_pnl': total_pnl,
            'win_rate': winners / len(rows) if rows else 0,
            'total_trades': len(rows),
            'winners': winners,
            'losers': losers,
            'avg_pnl': avg_pnl,
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
            f"<b>P&L:</b> {total_pnl:+.2f}%\n"
            f"<b>Win Rate:</b> {win_rate:.0%}\n"
            f"<b>Trades:</b> {total}\n"
            f"<b>W/L:</b> {winners}/{losers}\n"
            f"<b>Avg P&L:</b> {avg_pnl:+.2f}%"
        )

    def _get_all_recent_trades(self, limit: int = 10) -> list:
        """Get recent trades across all symbols."""
        if not self._database:
            return []

        with self._database.connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT symbol, direction, pnl_percent, was_correct,
                       entry_time, exit_time, interval
                FROM trade_outcomes
                WHERE COALESCE(is_paper_trade, 0) = 0
                  AND COALESCE(is_replay, 0) = 0
                ORDER BY exit_time DESC
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def _confidence_bar(conf: float) -> str:
        """Render a visual confidence bar."""
        filled = int(conf * 10)
        empty = 10 - filled
        return "\u2588" * filled + "\u2591" * empty
