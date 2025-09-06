"""
Notification Manager for Solana Trading Bot 2025

Handles notifications via Telegram, Discord, and other channels.
"""

import asyncio
import aiohttp
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from config.settings import Config, NotificationConfig

class NotificationManager:
    """Manages notifications across multiple channels"""
    
    def __init__(self, config: Config):
        self.config = config
        self.notification_config = config.notification_config
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Notification queues
        self.telegram_queue = []
        self.discord_queue = []
        self.email_queue = []
        
        # Rate limiting
        self.rate_limits = {
            "telegram": {"last_sent": 0.0, "min_interval": 1},  # 1 second between messages
            "discord": {"last_sent": 0.0, "min_interval": 1},
            "email": {"last_sent": 0.0, "min_interval": 60}  # 1 minute between emails
        }
    
    async def _ensure_session(self):
        """Ensure session is initialized"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        assert self.session is not None
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def send_notification(self, message: str, level: str = "INFO", 
                              channels: Optional[List[str]] = None) -> bool:
        """Send notification to specified channels"""
        try:
            if channels is None:
                channels = self._get_default_channels()
            
            success = True
            
            for channel in channels:
                try:
                    if channel == "telegram" and self.notification_config.telegram_enabled:
                        await self._send_telegram(message, level)
                    elif channel == "discord" and self.notification_config.discord_enabled:
                        await self._send_discord(message, level)
                    elif channel == "email" and self.notification_config.email_enabled:
                        await self._send_email(message, level)
                    elif channel == "browser" and self.notification_config.browser_notifications:
                        await self._send_browser_notification(message, level)
                except Exception as e:
                    print(f"Error sending {channel} notification: {e}")
                    success = False
            
            return success
            
        except Exception as e:
            print(f"Error in send_notification: {e}")
            return False
    
    def _get_default_channels(self) -> List[str]:
        """Get default notification channels based on config"""
        channels = []
        
        if self.notification_config.telegram_enabled:
            channels.append("telegram")
        
        if self.notification_config.discord_enabled:
            channels.append("discord")
        
        if self.notification_config.email_enabled:
            channels.append("email")
        
        if self.notification_config.browser_notifications:
            channels.append("browser")
        
        return channels
    
    async def _send_telegram(self, message: str, level: str) -> bool:
        """Send notification via Telegram"""
        try:
            if not self.config.api_config.telegram_bot_token:
                return False
            
            # Check rate limit
            if not self._check_rate_limit("telegram"):
                return False
            
            # Format message
            formatted_message = self._format_telegram_message(message, level)
            
            # Send message
            url = f"https://api.telegram.org/bot{self.config.api_config.telegram_bot_token}/sendMessage"
            
            payload = {
                "text": formatted_message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            }
            
            if self.session is None:
                await self._ensure_session()
            assert self.session is not None
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("ok", False)
                
                return False
                
        except Exception as e:
            print(f"Error sending Telegram notification: {e}")
            return False
    
    async def _send_discord(self, message: str, level: str) -> bool:
        """Send notification via Discord webhook"""
        try:
            if not self.config.api_config.discord_webhook:
                return False
            
            # Check rate limit
            if not self._check_rate_limit("discord"):
                return False
            
            # Format message
            formatted_message = self._format_discord_message(message, level)
            
            # Send webhook
            if self.session is None:
                await self._ensure_session()
            assert self.session is not None
            async with self.session.post(
                self.config.api_config.discord_webhook,
                json=formatted_message
            ) as response:
                return response.status == 204
                
        except Exception as e:
            print(f"Error sending Discord notification: {e}")
            return False
    
    async def _send_email(self, message: str, level: str) -> bool:
        """Send notification via email"""
        try:
            # Check rate limit
            if not self._check_rate_limit("email"):
                return False
            
            # This would implement email sending
            # For now, just log the email
            print(f"Email notification ({level}): {message}")
            return True
            
        except Exception as e:
            print(f"Error sending email notification: {e}")
            return False
    
    async def _send_browser_notification(self, message: str, level: str) -> bool:
        """Send browser notification"""
        try:
            # This would trigger browser notifications
            # For now, just log the notification
            print(f"Browser notification ({level}): {message}")
            return True
            
        except Exception as e:
            print(f"Error sending browser notification: {e}")
            return False
    
    def _format_telegram_message(self, message: str, level: str) -> str:
        """Format message for Telegram"""
        try:
            # Add emoji based on level
            emoji_map = {
                "INFO": "ℹ️",
                "WARNING": "⚠️",
                "ERROR": "❌",
                "SUCCESS": "✅",
                "TRADE": "💰",
                "SIGNAL": "🎯"
            }
            
            emoji = emoji_map.get(level, "ℹ️")
            
            # Format message
            formatted = f"{emoji} <b>{level}</b>\n\n{message}"
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted += f"\n\n<code>{timestamp}</code>"
            
            return formatted
            
        except Exception as e:
            print(f"Error formatting Telegram message: {e}")
            return message
    
    def _format_discord_message(self, message: str, level: str) -> Dict[str, Any]:
        """Format message for Discord"""
        try:
            # Color mapping for Discord embeds
            color_map = {
                "INFO": 0x3498db,      # Blue
                "WARNING": 0xf39c12,   # Orange
                "ERROR": 0xe74c3c,     # Red
                "SUCCESS": 0x2ecc71,   # Green
                "TRADE": 0x9b59b6,     # Purple
                "SIGNAL": 0xe67e22     # Dark Orange
            }
            
            color = color_map.get(level, 0x95a5a6)  # Gray default
            
            # Create embed
            embed = {
                "title": f"{level} Notification",
                "description": message,
                "color": color,
                "timestamp": datetime.now().isoformat(),
                "footer": {
                    "text": "Solana Trading Bot 2025"
                }
            }
            
            return {
                "embeds": [embed]
            }
            
        except Exception as e:
            print(f"Error formatting Discord message: {e}")
            return {"content": message}
    
    def _check_rate_limit(self, channel: str) -> bool:
        """Check rate limit for notification channel"""
        try:
            import time
            current_time = time.time()
            rate_limit = self.rate_limits[channel]
            
            if current_time - rate_limit["last_sent"] < rate_limit["min_interval"]:
                return False
            
            rate_limit["last_sent"] = current_time
            return True
            
        except Exception as e:
            print(f"Error checking rate limit: {e}")
            return True  # Allow if rate limit check fails
    
    async def send_trade_notification(self, trade_data: Dict[str, Any]) -> bool:
        """Send trade-specific notification"""
        try:
            # Format trade message
            message = self._format_trade_message(trade_data)
            
            # Send notification
            return await self.send_notification(message, "TRADE", ["telegram", "discord"])
            
        except Exception as e:
            print(f"Error sending trade notification: {e}")
            return False
    
    def _format_trade_message(self, trade_data: Dict[str, Any]) -> str:
        """Format trade data into notification message"""
        try:
            action = trade_data.get("action", "UNKNOWN").upper()
            token_symbol = trade_data.get("token_symbol", "UNKNOWN")
            quantity = trade_data.get("quantity", 0)
            price = trade_data.get("price", 0)
            pnl = trade_data.get("pnl", 0)
            
            message = f"<b>Trade Executed</b>\n\n"
            message += f"Action: {action}\n"
            message += f"Token: {token_symbol}\n"
            message += f"Quantity: {quantity:.6f}\n"
            message += f"Price: ${price:.6f}\n"
            
            if pnl != 0:
                pnl_emoji = "📈" if pnl > 0 else "📉"
                message += f"P&L: {pnl_emoji} ${pnl:.2f}\n"
            
            return message
            
        except Exception as e:
            print(f"Error formatting trade message: {e}")
            return "Trade executed"
    
    async def send_signal_notification(self, signal_data: Dict[str, Any]) -> bool:
        """Send signal-specific notification"""
        try:
            # Format signal message
            message = self._format_signal_message(signal_data)
            
            # Send notification
            return await self.send_notification(message, "SIGNAL", ["telegram", "discord"])
            
        except Exception as e:
            print(f"Error sending signal notification: {e}")
            return False
    
    def _format_signal_message(self, signal_data: Dict[str, Any]) -> str:
        """Format signal data into notification message"""
        try:
            action = signal_data.get("action", "UNKNOWN").upper()
            token_symbol = signal_data.get("token_symbol", "UNKNOWN")
            confidence = signal_data.get("confidence", 0)
            source = signal_data.get("source", "UNKNOWN")
            
            message = f"<b>Trading Signal</b>\n\n"
            message += f"Action: {action}\n"
            message += f"Token: {token_symbol}\n"
            message += f"Confidence: {confidence:.1%}\n"
            message += f"Source: {source}\n"
            
            return message
            
        except Exception as e:
            print(f"Error formatting signal message: {e}")
            return "Trading signal generated"
    
    async def send_error_notification(self, error: str, context: str = "") -> bool:
        """Send error notification"""
        try:
            message = f"<b>Error Occurred</b>\n\n"
            message += f"Error: {error}\n"
            
            if context:
                message += f"Context: {context}\n"
            
            return await self.send_notification(message, "ERROR", ["telegram", "discord"])
            
        except Exception as e:
            print(f"Error sending error notification: {e}")
            return False
    
    async def send_performance_notification(self, performance_data: Dict[str, Any]) -> bool:
        """Send performance notification"""
        try:
            message = self._format_performance_message(performance_data)
            return await self.send_notification(message, "INFO", ["telegram", "discord"])
            
        except Exception as e:
            print(f"Error sending performance notification: {e}")
            return False
    
    def _format_performance_message(self, performance_data: Dict[str, Any]) -> str:
        """Format performance data into notification message"""
        try:
            message = f"<b>Performance Update</b>\n\n"
            
            if "total_trades" in performance_data:
                message += f"Total Trades: {performance_data['total_trades']}\n"
            
            if "win_rate" in performance_data:
                message += f"Win Rate: {performance_data['win_rate']:.1%}\n"
            
            if "total_profit" in performance_data:
                profit = performance_data["total_profit"]
                profit_emoji = "📈" if profit > 0 else "📉"
                message += f"Total Profit: {profit_emoji} ${profit:.2f}\n"
            
            if "daily_pnl" in performance_data:
                daily_pnl = performance_data["daily_pnl"]
                daily_emoji = "📈" if daily_pnl > 0 else "📉"
                message += f"Daily P&L: {daily_emoji} ${daily_pnl:.2f}\n"
            
            return message
            
        except Exception as e:
            print(f"Error formatting performance message: {e}")
            return "Performance update"
    
    async def send_startup_notification(self) -> bool:
        """Send startup notification"""
        try:
            message = "🚀 <b>Solana Trading Bot Started</b>\n\n"
            message += "Bot is now running and monitoring the market."
            
            return await self.send_notification(message, "SUCCESS", ["telegram", "discord"])
            
        except Exception as e:
            print(f"Error sending startup notification: {e}")
            return False
    
    async def send_shutdown_notification(self) -> bool:
        """Send shutdown notification"""
        try:
            message = "🛑 <b>Solana Trading Bot Stopped</b>\n\n"
            message += "Bot has been shut down."
            
            return await self.send_notification(message, "WARNING", ["telegram", "discord"])
            
        except Exception as e:
            print(f"Error sending shutdown notification: {e}")
            return False 