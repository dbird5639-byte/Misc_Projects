"""
Notification utilities for AI-Powered Solana Meme Coin Sniper
"""

import asyncio
import aiohttp
import json
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from .logger import get_logger

logger = get_logger(__name__)


class NotificationManager:
    """
    Manages notifications across multiple channels
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = True
        
        # Notification channels
        self.telegram_enabled = config.get("telegram_enabled", False)
        self.telegram_bot_token = config.get("telegram_bot_token")
        self.telegram_chat_id = config.get("telegram_chat_id")
        
        self.discord_enabled = config.get("discord_enabled", False)
        self.discord_webhook_url = config.get("discord_webhook_url")
        
        self.browser_notifications = config.get("browser_notifications", True)
        
        self.email_enabled = config.get("email_enabled", False)
        self.email_smtp_server = config.get("email_smtp_server", "smtp.gmail.com")
        self.email_smtp_port = config.get("email_smtp_port", 587)
        self.email_username = config.get("email_username")
        self.email_password = config.get("email_password")
        self.email_recipients = config.get("email_recipients", [])
        
        # Rate limiting
        self.rate_limit = config.get("rate_limit", 10)  # messages per minute
        self.last_notification_time = 0
        self.notification_count = 0
        self.rate_limit_window = 60  # seconds
        
        # Message queue for rate limiting
        self.message_queue: List[Dict[str, Any]] = []
        self.queue_processor_task = None
        
        # Initialize session
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self):
        """Initialize notification manager"""
        try:
            # Create aiohttp session
            self.session = aiohttp.ClientSession()
            
            # Start queue processor
            self.queue_processor_task = asyncio.create_task(self._process_message_queue())
            
            # Test notifications
            await self._test_notifications()
            
            logger.info("Notification manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize notification manager: {e}")
            return False
    
    async def _test_notifications(self):
        """Test notification channels"""
        test_message = "🔔 AI Sniper notification system initialized successfully"
        
        if self.telegram_enabled and self.telegram_bot_token and self.telegram_chat_id:
            try:
                await self._send_telegram_message(test_message)
                logger.info("Telegram notifications working")
            except Exception as e:
                logger.warning(f"Telegram notifications failed: {e}")
        
        if self.discord_enabled and self.discord_webhook_url:
            try:
                await self._send_discord_message("AI Sniper Test", test_message, "normal")
                logger.info("Discord notifications working")
            except Exception as e:
                logger.warning(f"Discord notifications failed: {e}")
        
        if self.email_enabled:
            try:
                await self._send_email("AI Sniper Initialized", test_message)
                logger.info("Email notifications working")
            except Exception as e:
                logger.warning(f"Email notifications failed: {e}")
    
    async def send_notification(self, title: str, message: str, priority: str = "normal", 
                              channels: Optional[List[str]] = None, **kwargs):
        """
        Send notification across configured channels
        
        Args:
            title: Notification title
            message: Notification message
            priority: Priority level (low, normal, high, urgent)
            channels: Specific channels to use (telegram, discord, email, browser)
            **kwargs: Additional data for the notification
        """
        if not self.enabled:
            return
        
        # Check rate limiting
        if not self._check_rate_limit():
            logger.warning("Rate limit exceeded, queuing notification")
            self.message_queue.append({
                "title": title,
                "message": message,
                "priority": priority,
                "channels": channels,
                "kwargs": kwargs,
                "timestamp": datetime.now()
            })
            return
        
        # Determine channels to use
        if channels is None:
            channels = []
            if self.telegram_enabled:
                channels.append("telegram")
            if self.discord_enabled:
                channels.append("discord")
            if self.email_enabled:
                channels.append("email")
            if self.browser_notifications:
                channels.append("browser")
        
        # Send notifications
        tasks = []
        
        if "telegram" in channels and self.telegram_enabled:
            tasks.append(self._send_telegram_message(f"**{title}**\n\n{message}"))
        
        if "discord" in channels and self.discord_enabled:
            tasks.append(self._send_discord_message(title, message, priority))
        
        if "email" in channels and self.email_enabled:
            tasks.append(self._send_email(title, message, priority))
        
        if "browser" in channels and self.browser_notifications:
            tasks.append(self._send_browser_notification(title, message))
        
        # Execute all notifications concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Notification failed: {result}")
                else:
                    logger.debug(f"Notification sent successfully")
    
    async def _send_telegram_message(self, message: str) -> bool:
        """Send message via Telegram"""
        if not self.telegram_bot_token or not self.telegram_chat_id or not self.session:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            data = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            async with self.session.post(url, json=data) as response:
                if response.status == 200:
                    return True
                else:
                    logger.error(f"Telegram API error: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
    
    async def _send_discord_message(self, title: str, message: str, priority: str = "normal") -> bool:
        """Send message via Discord webhook"""
        if not self.discord_webhook_url or not self.session:
            return False
        
        try:
            # Choose color based on priority
            color_map = {
                "low": 0x00ff00,      # Green
                "normal": 0x0099ff,   # Blue
                "high": 0xff9900,     # Orange
                "urgent": 0xff0000    # Red
            }
            color = color_map.get(priority, 0x0099ff)
            
            embed = {
                "title": title,
                "description": message,
                "color": color,
                "timestamp": datetime.now().isoformat(),
                "footer": {
                    "text": "AI-Powered Solana Sniper"
                }
            }
            
            data = {"embeds": [embed]}
            
            async with self.session.post(self.discord_webhook_url, json=data) as response:
                if response.status == 204:
                    return True
                else:
                    logger.error(f"Discord webhook error: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending Discord message: {e}")
            return False
    
    async def _send_email(self, subject: str, message: str, priority: str = "normal") -> bool:
        """Send email notification"""
        if not self.email_username or not self.email_password or not self.email_recipients:
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_username
            msg['To'] = ", ".join(self.email_recipients)
            msg['Subject'] = f"[AI Sniper] {subject}"
            
            # Add priority header
            priority_map = {
                "low": "Low",
                "normal": "Normal", 
                "high": "High",
                "urgent": "Urgent"
            }
            msg['X-Priority'] = priority_map.get(priority, "Normal")
            
            # Add body
            body = f"""
            AI-Powered Solana Meme Coin Sniper Notification
            
            Priority: {priority.upper()}
            Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            {message}
            
            ---
            This is an automated notification from your AI trading system.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.email_smtp_server, self.email_smtp_port) as server:
                server.starttls()
                server.login(self.email_username, self.email_password)
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    async def _send_browser_notification(self, title: str, message: str) -> bool:
        """Send browser notification (placeholder for web interface)"""
        # This would typically integrate with a web interface
        # For now, just log the notification
        logger.info(f"Browser notification: {title} - {message}")
        return True
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        current_time = time.time()
        
        # Reset counter if window has passed
        if current_time - self.last_notification_time > self.rate_limit_window:
            self.notification_count = 0
            self.last_notification_time = current_time
        
        # Check if we're under the limit
        if self.notification_count < self.rate_limit:
            self.notification_count += 1
            return True
        
        return False
    
    async def _process_message_queue(self):
        """Process queued messages with rate limiting"""
        while True:
            try:
                if self.message_queue:
                    # Process one message per second to respect rate limits
                    message_data = self.message_queue.pop(0)
                    
                    # Wait if we're rate limited
                    while not self._check_rate_limit():
                        await asyncio.sleep(1)
                    
                    # Send the notification
                    await self.send_notification(
                        message_data["title"],
                        message_data["message"],
                        message_data["priority"],
                        message_data["channels"],
                        **message_data["kwargs"]
                    )
                
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing message queue: {e}")
                await asyncio.sleep(5)
    
    async def send_trade_notification(self, trade_type: str, token_address: str, 
                                    amount: float, price: float, profit_loss: Optional[float] = None):
        """Send trade-specific notification"""
        emoji_map = {
            "buy": "💰",
            "sell": "💸",
            "profit": "📈",
            "loss": "📉"
        }
        
        emoji = emoji_map.get(trade_type, "📊")
        
        if profit_loss is not None:
            title = f"{emoji} Trade {trade_type.title()}"
            message = f"""
            Token: {token_address}
            Amount: {amount:.6f}
            Price: ${price:.8f}
            P&L: ${profit_loss:.2f}
            """
        else:
            title = f"{emoji} {trade_type.title()} Order"
            message = f"""
            Token: {token_address}
            Amount: {amount:.6f}
            Price: ${price:.8f}
            """
        
        await self.send_notification(title, message, priority="high")
    
    async def send_alert_notification(self, alert_type: str, description: str, severity: str = "normal"):
        """Send alert notification"""
        emoji_map = {
            "error": "❌",
            "warning": "⚠️",
            "success": "✅",
            "info": "ℹ️"
        }
        
        emoji = emoji_map.get(alert_type, "🔔")
        priority = "urgent" if severity == "high" else "normal"
        
        title = f"{emoji} {alert_type.title()} Alert"
        message = f"{description}\n\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        await self.send_notification(title, message, priority=priority)
    
    async def send_performance_notification(self, metrics: Dict[str, Any]):
        """Send performance metrics notification"""
        title = "📊 Performance Update"
        
        message = f"""
        AI Sniper Performance Metrics:
        
        Total Opportunities: {metrics.get('total_opportunities', 0)}
        Successful Trades: {metrics.get('successful_trades', 0)}
        Failed Trades: {metrics.get('failed_trades', 0)}
        Success Rate: {metrics.get('success_rate', 0):.1%}
        Total Profit: ${metrics.get('total_profit', 0):.2f}
        AI Accuracy: {metrics.get('ai_accuracy', 0):.1%}
        
        Runtime: {metrics.get('runtime', 'Unknown')}
        """
        
        await self.send_notification(title, message, priority="normal")
    
    async def send_system_status_notification(self, status: Dict[str, Any]):
        """Send system status notification"""
        title = "🖥️ System Status"
        
        message = f"""
        System Health:
        
        CPU Usage: {status.get('cpu_usage', 0):.1f}%
        Memory Usage: {status.get('memory_usage', 0):.1f}%
        Disk Usage: {status.get('disk_usage', 0):.1f}%
        Network Latency: {status.get('network_latency', 0):.0f}ms
        Error Rate: {status.get('error_rate', 0):.2%}
        Uptime: {status.get('uptime', 0):.0f}s
        
        Active Tasks: {status.get('active_tasks', 0)}
        Pending Tasks: {status.get('pending_tasks', 0)}
        """
        
        await self.send_notification(title, message, priority="low")
    
    async def close(self):
        """Close notification manager"""
        if self.queue_processor_task:
            self.queue_processor_task.cancel()
            try:
                await self.queue_processor_task
            except asyncio.CancelledError:
                pass
        
        if self.session:
            await self.session.close()
        
        logger.info("Notification manager closed")


# Convenience functions for quick notifications
async def send_quick_notification(message: str, channels: Optional[List[str]] = None):
    """Send a quick notification"""
    manager = NotificationManager({})
    await manager.send_notification("AI Sniper", message, channels=channels)


async def send_error_notification(error: str, context: str = ""):
    """Send error notification"""
    manager = NotificationManager({})
    await manager.send_alert_notification("error", f"{context}\n\nError: {error}", "high")


async def send_success_notification(message: str):
    """Send success notification"""
    manager = NotificationManager({})
    await manager.send_alert_notification("success", message, "normal") 