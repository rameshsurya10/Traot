# Telegram Bot Design - Traot Trading Bot

## Summary

Full Telegram command system for Traot, enabling remote monitoring and control of the live trading bot from any device. Single authorized user (chat ID auth). Notifications on trades + daily summary.

## Architecture

**Approach 1: Separate module with shared bot instance**

- New file: `src/telegram_bot.py` — command handler (TelegramBot class)
- Existing: `src/notifier.py` — outbound alerts (unchanged, receives shared bot ref)
- Runner wires both together at startup

## Auth Model

Single user only. Every incoming message/command checked against `TELEGRAM_CHAT_ID` from .env. Unauthorized messages silently ignored (no response to prevent enumeration).

## Commands

| Command | Description | Data Source |
|---------|-------------|-------------|
| `/start` | Welcome + command list | Static |
| `/help` | Full command reference | Static |
| `/status` | Bot status, uptime, open trades, confidence | `runner.get_status()` |
| `/profit` | Total P&L, win rate, trade count | `database` trade_outcomes |
| `/daily` | Today's P&L breakdown | `database` trade_outcomes |
| `/weekly` | This week's P&L | `database` trade_outcomes |
| `/monthly` | This month's P&L | `database` trade_outcomes |
| `/balance` | Portfolio value, positions | `runner._portfolio` |
| `/performance` | Per-pair win rate & profit | `database` trade_outcomes |
| `/trades` | Recent 10 trades | `database` trade_outcomes |
| `/confidence` | Current confidence per pair | `database` confidence_history |
| `/learning` | Retraining history, model age | `database` retraining_history |
| `/start_trading` | Resume trading (unpause) | `runner.resume()` |
| `/stop_trading` | Pause trading (keeps bot alive) | `runner.pause()` |
| `/forceexit` | Close specific trade (by symbol) | `runner` order execution |
| `/forceexit_all` | Close all open trades | `runner` order execution |

## Notifications (Automatic)

- **Trade opened**: symbol, direction, entry price, SL/TP, confidence
- **Trade closed**: symbol, P&L, duration, outcome
- **Daily summary**: sent at configured hour (default 8:00 IST), total P&L, win rate, trade count

## Config

```yaml
# config.yaml additions
notifications:
  telegram:
    enabled: true
    bot_token_env: TELEGRAM_BOT_TOKEN    # Read from .env
    chat_id_env: TELEGRAM_CHAT_ID        # Read from .env
    notify_on_trade: true
    notify_daily_summary: true
```

```env
# .env additions
TELEGRAM_BOT_TOKEN=8543659480:AAELXHHYxRoiVAPMRJBh7Xu9fNKqp6YduBE
TELEGRAM_CHAT_ID=1226859879
```

## Dependencies

- `python-telegram-bot>=20.0` (async, already in requirements)

## VPS Deployment

- Bot runs in same process as trading runner (daemon thread)
- No extra ports needed (Telegram uses polling, not webhooks)
- Survives via existing nohup/systemd setup
