"""
Notification utilities for AI Market Maker & Liquidation Monitor
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from ..utils.logger import get_logger

logger = get_logger(__name__)


class NotificationManager:
    """Manages notifications across multiple channels"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = True
        self.is_running = False
        
        # Notification settings
        self.telegram_enabled = config.get("telegram_enabled", False)
        self.telegram_bot_token = config.get("telegram_bot_token", "")
        self.telegram_chat_id = config.get("telegram_chat_id", "")
        
        self.discord_enabled = config.get("discord_enabled", False)
        self.discord_webhook_url = config.get("discord_webhook_url", "")
        
        self.email_enabled = config.get("email_enabled", False)
        self.email_smtp_server = config.get("email_smtp_server", "")
        self.email_smtp_port = config.get("email_smtp_port", 587)
        self.email_username = config.get("email_username", "")
        self.email_password = config.get("email_password", "")
        self.email_recipients = config.get("email_recipients", [])
        
        self.browser_notifications = config.get("browser_notifications", True)
        self.rate_limit = config.get("rate_limit", 10)  # messages per minute
        
        # Rate limiting
        self.last_notification_time = 0
        self.notification_count = 0
        self.rate_limit_window = 60  # seconds
        
        # Session for HTTP requests
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self):
        """Initialize the notification manager"""
        try:
            logger.info("Initializing Notification Manager...")
            
            # Create HTTP session
            self.session = aiohttp.ClientSession()
            
            # Test notification channels
            await self._test_channels()
            
            self.is_running = True
            logger.info("Notification Manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Notification Manager: {e}")
            return False
    
    async def _test_channels(self):
        """Test notification channels"""
        try:
            if self.telegram_enabled and self.telegram_bot_token:
                await self._test_telegram()
            
            if self.discord_enabled and self.discord_webhook_url:
                await self._test_discord()
                
        except Exception as e:
            logger.error(f"Error testing notification channels: {e}")
    
    async def _test_telegram(self):
        """Test Telegram connection"""
        try:
            if not self.session:
                logger.warning("Session not initialized")
                return
                
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/getMe"
            async with self.session.get(url) as response:
                if response.status == 200:
                    logger.info("Telegram connection successful")
                else:
                    logger.warning("Telegram connection failed")
                    
        except Exception as e:
            logger.error(f"Error testing Telegram: {e}")
    
    async def _test_discord(self):
        """Test Discord webhook"""
        try:
            if not self.session:
                logger.warning("Session not initialized")
                return
                
            test_payload = {
                "content": "🤖 AI Market Maker notification test"
            }
            
            async with self.session.post(self.discord_webhook_url, json=test_payload) as response:
                if response.status == 204:
                    logger.info("Discord webhook test successful")
                else:
                    logger.warning("Discord webhook test failed")
                    
        except Exception as e:
            logger.error(f"Error testing Discord: {e}")
    
    async def send_notification(self, title: str, message: str, level: str = "info"):
        """Send notification across all enabled channels"""
        try:
            # Check rate limiting
            if not self._check_rate_limit():
                logger.warning("Rate limit exceeded, skipping notification")
                return
            
            # Send to all enabled channels
            tasks = []
            
            if self.telegram_enabled:
                tasks.append(self._send_telegram(title, message))
            
            if self.discord_enabled:
                tasks.append(self._send_discord(title, message, level))
            
            if self.email_enabled:
                tasks.append(self._send_email(title, message))
            
            if self.browser_notifications:
                tasks.append(self._send_browser_notification(title, message))
            
            # Execute all notifications concurrently
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update rate limiting
            self._update_rate_limit()
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    async def _send_telegram(self, title: str, message: str):
        """Send Telegram notification"""
        try:
            if not self.telegram_bot_token or not self.telegram_chat_id:
                return
            
            if not self.session:
                logger.warning("Session not initialized")
                return
            
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": f"*{title}*\n\n{message}",
                "parse_mode": "Markdown"
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Telegram notification failed: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")
    
    async def _send_discord(self, title: str, message: str, level: str = "info"):
        """Send Discord notification"""
        try:
            if not self.discord_webhook_url:
                return
            
            if not self.session:
                logger.warning("Session not initialized")
                return
            
            # Choose emoji based on level
            emoji_map = {
                "info": "ℹ️",
                "warning": "⚠️",
                "error": "❌",
                "success": "✅"
            }
            emoji = emoji_map.get(level, "ℹ️")
            
            payload = {
                "embeds": [{
                    "title": f"{emoji} {title}",
                    "description": message,
                    "color": self._get_discord_color(level),
                    "timestamp": datetime.now().isoformat()
                }]
            }
            
            async with self.session.post(self.discord_webhook_url, json=payload) as response:
                if response.status != 204:
                    logger.error(f"Discord notification failed: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error sending Discord notification: {e}")
    
    async def _send_email(self, title: str, message: str):
        """Send email notification (placeholder)"""
        try:
            # This would use smtplib to send emails
            # For now, just log the email
            logger.info(f"Email notification: {title} - {message}")
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
    
    async def _send_browser_notification(self, title: str, message: str):
        """Send browser notification (placeholder)"""
        try:
            # This would use a web framework to send browser notifications
            # For now, just log the notification
            logger.info(f"Browser notification: {title} - {message}")
            
        except Exception as e:
            logger.error(f"Error sending browser notification: {e}")
    
    def _get_discord_color(self, level: str) -> int:
        """Get Discord embed color based on level"""
        color_map = {
            "info": 0x3498db,    # Blue
            "warning": 0xf39c12,  # Orange
            "error": 0xe74c3c,    # Red
            "success": 0x2ecc71   # Green
        }
        return color_map.get(level, 0x3498db)
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        current_time = time.time()
        
        # Reset counter if window has passed
        if current_time - self.last_notification_time > self.rate_limit_window:
            self.notification_count = 0
            self.last_notification_time = current_time
        
        # Check if we're under the limit
        return self.notification_count < self.rate_limit
    
    def _update_rate_limit(self):
        """Update rate limiting counters"""
        self.notification_count += 1
    
    async def send_liquidation_alert(self, symbol: str, side: str, size: float, price: float):
        """Send liquidation alert"""
        title = "🚨 Liquidation Alert"
        message = f"**{symbol}** {side.upper()} liquidation\nSize: ${size:,.0f}\nPrice: ${price:,.2f}"
        
        await self.send_notification(title, message, "warning")
    
    async def send_prediction_alert(self, symbol: str, side: str, probability: float, confidence: float):
        """Send prediction alert"""
        title = "🎯 Liquidation Prediction"
        message = f"**{symbol}** {side.upper()} prediction\nProbability: {probability:.1%}\nConfidence: {confidence:.1%}"
        
        await self.send_notification(title, message, "info")
    
    async def send_position_alert(self, trader: str, symbol: str, side: str, size: float):
        """Send large position alert"""
        title = "💰 Large Position Detected"
        message = f"Trader: {trader[:8]}...\n**{symbol}** {side.upper()}\nSize: ${size:,.0f}"
        
        await self.send_notification(title, message, "info")
    
    async def send_system_alert(self, title: str, message: str, level: str = "info"):
        """Send system alert"""
        await self.send_notification(f"🤖 {title}", message, level)
    
    async def stop(self):
        """Stop the notification manager"""
        logger.info("Stopping Notification Manager...")
        self.is_running = False
        
        # Close session
        if self.session:
            await self.session.close()
        
        logger.info("Notification Manager stopped")


def create_notification_manager(config: Dict[str, Any]) -> NotificationManager:
    """Create notification manager instance"""
    return NotificationManager(config) 