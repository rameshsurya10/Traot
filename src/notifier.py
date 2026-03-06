"""
Notifier - Desktop, Sound, and External Notifications
=====================================================
Sends alerts when signals are generated.
Works even when browser/dashboard is closed!
"""

import logging
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from plyer import notification as desktop_notification
    PLYER_AVAILABLE = True
except ImportError:
    PLYER_AVAILABLE = False
    logger.warning("plyer not installed - desktop notifications disabled")

try:
    import telegram
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False


class Notifier:
    """
    Multi-channel notification system.

    Channels:
    - Desktop notifications (always works, even with browser closed)
    - Sound alerts
    - Telegram messages (optional)
    - Console logging (always on)

    WORKS INDEPENDENTLY OF DASHBOARD!
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize notifier."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Notification settings
        notif_config = self.config['notifications']
        self.desktop_enabled = notif_config.get('desktop', True)
        self.sound_enabled = notif_config.get('sound', True)
        self.sound_file = notif_config.get('sound_file')

        # Telegram
        telegram_config = notif_config.get('telegram', {})
        self.telegram_enabled = telegram_config.get('enabled', False)
        self.telegram_token = telegram_config.get('bot_token', '')
        self.telegram_chat_id = telegram_config.get('chat_id', '')

        # Stats
        self._notifications_sent = 0

        logger.info("Notifier initialized")

    def on_signal(self, signal: dict):
        """
        Handle incoming trading signal.

        Sends notifications through all enabled channels.
        """
        try:
            # Format message
            direction = signal.get('signal', 'UNKNOWN')
            strength = signal.get('strength', 'UNKNOWN')
            price = signal.get('price', 0)
            confidence = signal.get('confidence', 0)

            title = f"🚨 {strength} {direction} Signal"
            short_message = f"${price:,.2f} | Confidence: {confidence:.1%}"

            full_message = self._format_full_message(signal)

            # Send through all channels (in parallel)
            threads = []

            if self.desktop_enabled:
                t = threading.Thread(
                    target=self._send_desktop,
                    args=(title, short_message)
                )
                threads.append(t)

            if self.sound_enabled:
                t = threading.Thread(target=self._play_sound)
                threads.append(t)

            if self.telegram_enabled:
                t = threading.Thread(
                    target=self._send_telegram,
                    args=(full_message,)
                )
                threads.append(t)

            # Always log to console
            self._log_signal(signal)

            # Start all threads
            for t in threads:
                t.start()

            # Wait for completion
            for t in threads:
                t.join(timeout=10)

            self._notifications_sent += 1

        except Exception as e:
            logger.error(f"Notification error: {e}")

    def _format_full_message(self, signal: dict) -> str:
        """Format full signal message."""
        direction = signal.get('signal', 'UNKNOWN')
        strength = signal.get('strength', 'UNKNOWN')
        price = signal.get('price', 0)
        confidence = signal.get('confidence', 0)
        stop_loss = signal.get('stop_loss')
        take_profit = signal.get('take_profit')
        risk_per_trade = signal.get('risk_per_trade', 0.02)
        risk_reward = signal.get('risk_reward_ratio', 2.0)

        emoji = '🟢' if direction == 'BUY' else '🔴'

        message = f"""
{emoji} {strength} {direction} SIGNAL

📊 Price: ${price:,.2f}
🎯 Confidence: {confidence:.1%}
"""

        if stop_loss and take_profit:
            message += f"""
📍 Suggested Levels:
   Stop Loss: ${stop_loss:,.2f}
   Take Profit: ${take_profit:,.2f}
   Risk:Reward = 1:{risk_reward:.1f}
"""

        message += f"""
⚠️ Risk only {risk_per_trade:.0%} of capital
⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
        return message.strip()

    def _send_desktop(self, title: str, message: str):
        """Send desktop notification."""
        if not PLYER_AVAILABLE:
            # Fallback: try system-specific methods
            self._send_desktop_fallback(title, message)
            return

        try:
            desktop_notification.notify(
                title=title,
                message=message,
                app_name="Traot",
                timeout=30
            )
            logger.debug("Desktop notification sent")
        except Exception as e:
            logger.error(f"Desktop notification failed: {e}")
            self._send_desktop_fallback(title, message)

    @staticmethod
    def _sanitize_for_shell(text: str) -> str:
        """Sanitize text to prevent command injection."""
        # Remove or escape dangerous characters
        dangerous_chars = ['`', '$', '\\', '"', "'", ';', '|', '&', '\n', '\r', '(', ')', '{', '}', '[', ']', '<', '>']
        sanitized = text
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, ' ')
        # Limit length to prevent buffer issues
        return sanitized[:200]

    def _send_desktop_fallback(self, title: str, message: str):
        """Fallback desktop notification using system commands."""
        try:
            import platform
            system = platform.system()

            # Sanitize all inputs to prevent command injection
            title_safe = self._sanitize_for_shell(title)
            message_safe = self._sanitize_for_shell(message)

            if system == "Linux":
                # Linux notify-send is safe with list arguments
                subprocess.run(
                    ['notify-send', title_safe, message_safe],
                    timeout=5,
                    capture_output=True
                )
            elif system == "Darwin":  # macOS
                # Use osascript with sanitized input
                script = f'display notification "{message_safe}" with title "{title_safe}"'
                subprocess.run(
                    ['osascript', '-e', script],
                    timeout=5,
                    capture_output=True,
                    check=False
                )
            elif system == "Windows":
                # Windows toast notification via PowerShell with sanitized input
                ps_script = f'''
                [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
                $template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
                $textNodes = $template.GetElementsByTagName("text")
                $textNodes.Item(0).AppendChild($template.CreateTextNode("{title_safe}")) | Out-Null
                $textNodes.Item(1).AppendChild($template.CreateTextNode("{message_safe}")) | Out-Null
                $toast = [Windows.UI.Notifications.ToastNotification]::new($template)
                [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Traot").Show($toast)
                '''
                subprocess.run(
                    ['powershell', '-Command', ps_script],
                    timeout=5,
                    capture_output=True
                )

            logger.debug("Desktop notification sent (fallback)")

        except Exception as e:
            logger.error(f"Desktop fallback failed: {e}")

    def _validate_sound_path(self, path: str) -> Optional[str]:
        """Validate sound file path to prevent path traversal attacks."""
        if not path:
            return None

        try:
            sound_path = Path(path).resolve()
            # Only allow files with audio extensions
            allowed_extensions = {'.wav', '.mp3', '.ogg', '.oga', '.aiff', '.aif', '.m4a'}
            if sound_path.suffix.lower() not in allowed_extensions:
                logger.warning(f"Sound file has invalid extension: {sound_path.suffix}")
                return None
            # Check file exists
            if not sound_path.exists():
                return None
            return str(sound_path)
        except Exception as e:
            logger.warning(f"Invalid sound path: {e}")
            return None

    def _play_sound(self):
        """Play alert sound."""
        try:
            import platform
            system = platform.system()

            # Check and validate custom sound file
            sound_path = self._validate_sound_path(self.sound_file)

            if system == "Linux":
                if sound_path:
                    subprocess.run(
                        ['paplay', sound_path],
                        timeout=5,
                        capture_output=True
                    )
                else:
                    # System beep
                    subprocess.run(
                        ['paplay', '/usr/share/sounds/freedesktop/stereo/complete.oga'],
                        timeout=5,
                        capture_output=True
                    )

            elif system == "Darwin":  # macOS
                if sound_path:
                    subprocess.run(
                        ['afplay', sound_path],
                        timeout=5,
                        capture_output=True
                    )
                else:
                    subprocess.run(
                        ['afplay', '/System/Library/Sounds/Glass.aiff'],
                        timeout=5,
                        capture_output=True
                    )

            elif system == "Windows":
                if sound_path:
                    import winsound
                    winsound.PlaySound(sound_path, winsound.SND_FILENAME)
                else:
                    import winsound
                    winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)

            logger.debug("Sound alert played")

        except Exception as e:
            logger.debug(f"Sound alert failed (non-critical): {e}")

    def _send_telegram(self, message: str):
        """Send Telegram message."""
        if not TELEGRAM_AVAILABLE:
            logger.warning("python-telegram-bot not installed")
            return

        if not self.telegram_token or not self.telegram_chat_id:
            logger.warning("Telegram credentials not configured")
            return

        try:
            import asyncio

            async def send():
                bot = telegram.Bot(token=self.telegram_token)
                await bot.send_message(
                    chat_id=self.telegram_chat_id,
                    text=message,
                    parse_mode='HTML'
                )

            asyncio.run(send())
            logger.debug("Telegram message sent")

        except Exception as e:
            logger.error(f"Telegram failed: {e}")

    def _log_signal(self, signal: dict):
        """Log signal to console with formatting."""
        direction = signal.get('signal', 'UNKNOWN')
        strength = signal.get('strength', 'UNKNOWN')
        price = signal.get('price', 0)
        confidence = signal.get('confidence', 0)
        stop_loss = signal.get('stop_loss')
        take_profit = signal.get('take_profit')

        # Color codes for terminal
        RED = '\033[91m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BOLD = '\033[1m'
        RESET = '\033[0m'

        color = GREEN if direction == 'BUY' else RED

        print(f"\n{'='*60}")
        print(f"{BOLD}{color}🚨 {strength} {direction} SIGNAL 🚨{RESET}")
        print(f"{'='*60}")
        print(f"📊 Price:      ${price:,.2f}")
        print(f"🎯 Confidence: {confidence:.1%}")

        if stop_loss and take_profit:
            print(f"\n{YELLOW}📍 Suggested Levels:{RESET}")
            print(f"   🛑 Stop Loss:   ${stop_loss:,.2f}")
            print(f"   ✅ Take Profit: ${take_profit:,.2f}")

        print(f"\n⏰ Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"{'='*60}\n")

    def test_notifications(self):
        """Test all notification channels."""
        test_signal = {
            'signal': 'BUY',
            'strength': 'STRONG',
            'price': 50000,
            'confidence': 0.68,
            'stop_loss': 48000,
            'take_profit': 54000,
            'risk_per_trade': 0.02,
            'risk_reward_ratio': 2.0
        }

        print("Testing notifications...")
        self.on_signal(test_signal)
        print("Test complete!")

    def get_status(self) -> dict:
        """Get notifier status."""
        return {
            'desktop_enabled': self.desktop_enabled,
            'sound_enabled': self.sound_enabled,
            'telegram_enabled': self.telegram_enabled,
            'telegram_configured': bool(self.telegram_token and self.telegram_chat_id),
            'notifications_sent': self._notifications_sent
        }


# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    notifier = Notifier()
    print("Status:", notifier.get_status())
    print("\nRunning notification test...")
    notifier.test_notifications()
