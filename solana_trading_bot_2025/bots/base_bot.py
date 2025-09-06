"""
Base Bot Class for Solana Trading Bot 2025

This module provides the foundation for both sniper and copy bots.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import logging

from config.settings import Config
from utils.logger import setup_logger
from utils.notifications import NotificationManager
from risk_management.risk_manager import RiskManager

@dataclass
class TokenInfo:
    """Token information data structure"""
    address: str
    name: str
    symbol: str
    price: float
    volume_24h: float
    liquidity: float
    market_cap: float
    launch_time: datetime
    dex: str
    pair_address: str

@dataclass
class TradeSignal:
    """Trade signal data structure"""
    token_address: str
    action: str  # "buy" or "sell"
    price: float
    quantity: float
    confidence: float
    source: str
    timestamp: datetime
    metadata: Dict[str, Any]

class BaseBot(ABC):
    """Abstract base class for trading bots"""
    
    def __init__(self, config: Config, bot_name: str):
        self.config = config
        self.bot_name = bot_name
        self.logger = setup_logger(f"{bot_name}_bot")
        self.notification_manager = NotificationManager(config)
        self.risk_manager = RiskManager(config.risk_config)
        
        # Bot state
        self.is_running = False
        self.start_time = None
        self.stats = {
            "tokens_scanned": 0,
            "signals_generated": 0,
            "trades_executed": 0,
            "total_profit": 0.0,
            "success_rate": 0.0
        }
        
        # Performance tracking
        self.performance_history = []
        self.error_count = 0
        self.last_error = None
    
    async def start(self):
        """Start the bot"""
        self.logger.info(f"Starting {self.bot_name} bot...")
        self.is_running = True
        self.start_time = datetime.now()
        
        try:
            await self.notification_manager.send_notification(
                f"🚀 {self.bot_name} bot started successfully"
            )
            
            await self.run()
            
        except Exception as e:
            self.logger.error(f"Error in {self.bot_name} bot: {e}")
            self.error_count += 1
            self.last_error = str(e)
            
            await self.notification_manager.send_notification(
                f"❌ {self.bot_name} bot error: {e}"
            )
            
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the bot"""
        self.logger.info(f"Stopping {self.bot_name} bot...")
        self.is_running = False
        
        # Generate final report
        await self.generate_performance_report()
        
        await self.notification_manager.send_notification(
            f"🛑 {self.bot_name} bot stopped"
        )
    
    @abstractmethod
    async def run(self):
        """Main bot loop - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    async def process_signal(self, signal: TradeSignal) -> bool:
        """Process a trade signal - must be implemented by subclasses"""
        pass
    
    async def validate_token(self, token: TokenInfo) -> bool:
        """Validate if a token meets trading criteria"""
        try:
            # Basic validation
            if not token.address or not token.symbol:
                return False
            
            # Volume check
            if token.volume_24h < self.config.sniper_config.min_volume:
                return False
            
            # Liquidity check
            if token.liquidity < self.config.sniper_config.min_liquidity:
                return False
            
            # Market cap check (optional)
            if token.market_cap < 1000:  # Minimum $1000 market cap
                return False
            
            # Additional validation can be added here
            # - Contract verification
            # - Honeypot detection
            # - Rug pull indicators
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating token {token.address}: {e}")
            return False
    
    async def calculate_position_size(self, token: TokenInfo, confidence: float) -> float:
        """Calculate position size based on risk management rules"""
        try:
            # Base position size from config
            base_size = self.config.sniper_config.max_position_size
            
            # Adjust based on confidence
            adjusted_size = base_size * confidence
            
            # Apply risk management rules
            final_size = self.risk_manager.calculate_position_size(
                token.price, adjusted_size
            )
            
            return final_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    async def execute_trade(self, signal: TradeSignal) -> bool:
        """Execute a trade based on signal"""
        try:
            # Check if we should trade
            if not self.risk_manager.should_trade(signal):
                self.logger.info(f"Trade rejected by risk manager: {signal.token_address}")
                return False
            
            # Execute the trade (implementation depends on DEX)
            # This is a placeholder - actual implementation would use Jupiter API
            self.logger.info(f"Executing trade: {signal.action} {signal.quantity} of {signal.token_address}")
            
            # Update statistics
            self.stats["trades_executed"] += 1
            
            # Record trade
            await self.record_trade(signal)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return False
    
    async def record_trade(self, signal: TradeSignal):
        """Record trade for performance tracking"""
        trade_record = {
            "timestamp": signal.timestamp,
            "token_address": signal.token_address,
            "action": signal.action,
            "price": signal.price,
            "quantity": signal.quantity,
            "confidence": signal.confidence,
            "source": signal.source
        }
        
        self.performance_history.append(trade_record)
    
    async def generate_performance_report(self):
        """Generate performance report"""
        if not self.performance_history:
            return
        
        total_trades = len(self.performance_history)
        profitable_trades = len([t for t in self.performance_history if t.get("profit", 0) > 0])
        success_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        report = f"""
📊 {self.bot_name} Bot Performance Report

Runtime: {datetime.now() - self.start_time if self.start_time else 'N/A'}
Total Trades: {total_trades}
Profitable Trades: {profitable_trades}
Success Rate: {success_rate:.2%}
Total Profit: ${self.stats['total_profit']:.2f}
Tokens Scanned: {self.stats['tokens_scanned']}
Signals Generated: {self.stats['signals_generated']}
Errors: {self.error_count}
        """
        
        self.logger.info(report)
        await self.notification_manager.send_notification(report)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        return {
            "bot_name": self.bot_name,
            "is_running": self.is_running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "stats": self.stats,
            "error_count": self.error_count,
            "last_error": self.last_error
        }
    
    async def handle_error(self, error: Exception, context: str = ""):
        """Handle errors gracefully"""
        self.error_count += 1
        self.last_error = str(error)
        
        error_msg = f"Error in {self.bot_name} bot"
        if context:
            error_msg += f" ({context})"
        error_msg += f": {error}"
        
        self.logger.error(error_msg)
        
        # Send notification for critical errors
        if self.error_count <= 3:  # Only notify for first few errors
            await self.notification_manager.send_notification(f"❌ {error_msg}")
    
    async def health_check(self) -> bool:
        """Perform health check"""
        try:
            # Check if bot is responsive
            if not self.is_running:
                return False
            
            # Check error rate
            if self.error_count > 10:
                return False
            
            # Additional health checks can be added here
            # - API connectivity
            # - Database connection
            # - Memory usage
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False 