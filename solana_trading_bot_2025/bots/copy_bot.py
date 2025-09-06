"""
Copy Bot for Solana Trading Bot 2025

This bot tracks trading activities of influential traders and copies their trades.
"""

import asyncio
import aiohttp
from typing import List, Optional, Dict, Any, Set
from datetime import datetime, timedelta
import json
from dataclasses import dataclass

from bots.base_bot import BaseBot, TokenInfo, TradeSignal
from data.market_data import MarketDataManager
from utils.browser_automation import BrowserAutomation

@dataclass
class TraderInfo:
    """Information about a tracked trader"""
    address: str
    name: str
    success_rate: float
    total_trades: int
    last_trade_time: datetime
    total_volume: float

@dataclass
class CopyTrade:
    """Copy trade information"""
    original_trader: str
    original_trade_hash: str
    token_address: str
    action: str
    price: float
    quantity: float
    timestamp: datetime
    delay: int  # seconds delay from original trade

class CopyBot(BaseBot):
    """Bot that copies trades from influential traders"""
    
    def __init__(self, config):
        super().__init__(config, "Copy")
        self.market_data = MarketDataManager(config)
        self.browser_automation = BrowserAutomation()
        
        # Copy-specific state
        self.tracked_traders: Dict[str, TraderInfo] = {}
        self.recent_trades: List[CopyTrade] = []
        self.copied_trades: Set[str] = set()  # Track original trade hashes
        
        # Performance tracking
        self.trader_performance = {}
        self.copy_success_rate = 0.0
        
        # Initialize tracked traders
        self._initialize_tracked_traders()
    
    def _initialize_tracked_traders(self):
        """Initialize the list of tracked traders"""
        follow_list = self.config.copy_config.follow_list or []
        
        for address in follow_list:
            self.tracked_traders[address] = TraderInfo(
                address=address,
                name=f"Trader_{address[:8]}",
                success_rate=0.0,
                total_trades=0,
                last_trade_time=datetime.now(),
                total_volume=0.0
            )
    
    async def run(self):
        """Main copy bot loop"""
        self.logger.info("Starting copy bot main loop...")
        
        while self.is_running:
            try:
                # Monitor tracked traders
                await self.monitor_traders()
                
                # Process copy trades
                await self.process_copy_trades()
                
                # Update trader performance
                await self.update_trader_performance()
                
                # Wait for next cycle
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                await self.handle_error(e, "main loop")
                await asyncio.sleep(30)  # Wait longer on error
    
    async def monitor_traders(self):
        """Monitor tracked traders for new trades"""
        try:
            for address, trader in self.tracked_traders.items():
                # Get recent trades for this trader
                recent_trades = await self.get_trader_recent_trades(address)
                
                for trade in recent_trades:
                    # Check if we've already copied this trade
                    if trade["hash"] in self.copied_trades:
                        continue
                    
                    # Validate trade
                    if await self.validate_copy_trade(trade, trader):
                        # Create copy trade
                        copy_trade = await self.create_copy_trade(trade, trader)
                        if copy_trade:
                            self.recent_trades.append(copy_trade)
                            self.copied_trades.add(trade["hash"])
                            
                            # Mark as copied to avoid duplicates
                            self.copied_trades.add(trade["hash"])
                
        except Exception as e:
            await self.handle_error(e, "monitor_traders")
    
    async def get_trader_recent_trades(self, address: str) -> List[Dict[str, Any]]:
        """Get recent trades for a specific trader"""
        try:
            # Get trades from multiple sources
            trades = []
            
            # Source 1: Solscan API
            solscan_trades = await self.get_solscan_trades(address)
            trades.extend(solscan_trades)
            
            # Source 2: Birdeye API
            birdeye_trades = await self.get_birdeye_trades(address)
            trades.extend(birdeye_trades)
            
            # Source 3: Jupiter API (if available)
            jupiter_trades = await self.get_jupiter_trades(address)
            trades.extend(jupiter_trades)
            
            # Remove duplicates and sort by timestamp
            unique_trades = {}
            for trade in trades:
                if trade["hash"] not in unique_trades:
                    unique_trades[trade["hash"]] = trade
            
            sorted_trades = sorted(
                unique_trades.values(),
                key=lambda x: x["timestamp"],
                reverse=True
            )
            
            # Return only recent trades (last hour)
            cutoff_time = datetime.now() - timedelta(hours=1)
            recent_trades = [
                trade for trade in sorted_trades
                if trade["timestamp"] > cutoff_time
            ]
            
            return recent_trades
            
        except Exception as e:
            self.logger.error(f"Error getting trades for {address}: {e}")
            return []
    
    async def get_solscan_trades(self, address: str) -> List[Dict[str, Any]]:
        """Get trades from Solscan API"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://public-api.solscan.io/account/transactions"
                params = {
                    "account": address,
                    "limit": 50
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        trades = []
                        
                        for tx in data.get("data", []):
                            # Parse transaction to identify trades
                            trade_info = await self.parse_solscan_transaction(tx)
                            if trade_info:
                                trades.append(trade_info)
                        
                        return trades
                    
        except Exception as e:
            self.logger.error(f"Error getting Solscan trades: {e}")
        
        return []
    
    async def get_birdeye_trades(self, address: str) -> List[Dict[str, Any]]:
        """Get trades from Birdeye API"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"X-API-KEY": self.config.api_config.birdeye_api}
                url = f"https://public-api.birdeye.so/public/portfolio"
                params = {"wallet": address}
                
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        trades = []
                        
                        # Parse portfolio data for recent trades
                        # This is a simplified version - actual implementation would be more complex
                        
                        return trades
                    
        except Exception as e:
            self.logger.error(f"Error getting Birdeye trades: {e}")
        
        return []
    
    async def get_jupiter_trades(self, address: str) -> List[Dict[str, Any]]:
        """Get trades from Jupiter API"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://price.jup.ag/v4/price"
                # Jupiter doesn't provide direct trade history, so this would need
                # to be implemented differently or use alternative sources
                
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting Jupiter trades: {e}")
        
        return []
    
    async def parse_solscan_transaction(self, tx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse Solscan transaction to identify trades"""
        try:
            # This is a simplified parser - actual implementation would be more complex
            # and would need to identify specific DEX interactions
            
            # Check if transaction involves token swaps
            if "tokenTransfers" in tx:
                for transfer in tx["tokenTransfers"]:
                    # Identify if this is a trade
                    if self.is_trade_transaction(transfer):
                        return {
                            "hash": tx["txHash"],
                            "timestamp": datetime.fromtimestamp(tx["blockTime"]),
                            "token_address": transfer.get("tokenAddress"),
                            "action": "buy" if transfer.get("type") == "in" else "sell",
                            "quantity": float(transfer.get("amount", 0)),
                            "price": 0.0,  # Would need to calculate from transaction
                            "trader": tx.get("signer", "")
                        }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error parsing transaction: {e}")
            return None
    
    def is_trade_transaction(self, transfer: Dict[str, Any]) -> bool:
        """Determine if a transaction is a trade"""
        # This is a simplified check - actual implementation would be more sophisticated
        # and would check for specific DEX program interactions
        
        # Check if it involves known DEX programs
        dex_programs = [
            "JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB",  # Jupiter
            "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM",  # Raydium
            "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"   # Orca
        ]
        
        return any(program in str(transfer) for program in dex_programs)
    
    async def validate_copy_trade(self, trade: Dict[str, Any], trader: TraderInfo) -> bool:
        """Validate if a trade should be copied"""
        try:
            # Check minimum trade size
            if trade["quantity"] < self.config.copy_config.min_trade_size:
                return False
            
            # Check trader success rate
            if trader.success_rate < 0.5:  # Only copy from traders with >50% success rate
                return False
            
            # Check if trade is recent enough
            if datetime.now() - trade["timestamp"] > timedelta(minutes=5):
                return False
            
            # Additional validation can be added here
            # - Token safety checks
            # - Liquidity checks
            # - Volume checks
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating copy trade: {e}")
            return False
    
    async def create_copy_trade(self, trade: Dict[str, Any], trader: TraderInfo) -> Optional[CopyTrade]:
        """Create a copy trade from original trade"""
        try:
            # Calculate copy quantity based on percentage
            copy_quantity = trade["quantity"] * self.config.copy_config.copy_percentage
            
            # Calculate delay
            delay = int((datetime.now() - trade["timestamp"]).total_seconds())
            
            # Check if delay is within limits
            if delay > self.config.copy_config.max_delay:
                return None
            
            copy_trade = CopyTrade(
                original_trader=trader.address,
                original_trade_hash=trade["hash"],
                token_address=trade["token_address"],
                action=trade["action"],
                price=trade["price"],
                quantity=copy_quantity,
                timestamp=datetime.now(),
                delay=delay
            )
            
            return copy_trade
            
        except Exception as e:
            self.logger.error(f"Error creating copy trade: {e}")
            return None
    
    async def process_copy_trades(self):
        """Process copy trades"""
        try:
            for copy_trade in self.recent_trades[:]:  # Copy list to avoid modification
                # Check if we should execute this copy trade
                if await self.should_execute_copy_trade(copy_trade):
                    # Generate signal
                    signal = await self.generate_copy_signal(copy_trade)
                    if signal:
                        await self.process_signal(signal)
                
                # Remove processed trade
                self.recent_trades.remove(copy_trade)
                
        except Exception as e:
            await self.handle_error(e, "process_copy_trades")
    
    async def should_execute_copy_trade(self, copy_trade: CopyTrade) -> bool:
        """Determine if a copy trade should be executed"""
        try:
            # Check if delay is acceptable
            if copy_trade.delay > self.config.copy_config.max_delay:
                return False
            
            # Check if we have enough balance
            if not await self.check_balance(copy_trade):
                return False
            
            # Additional checks can be added here
            # - Market conditions
            # - Risk management rules
            # - Portfolio limits
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking copy trade execution: {e}")
            return False
    
    async def check_balance(self, copy_trade: CopyTrade) -> bool:
        """Check if we have sufficient balance for the trade"""
        try:
            # This would check actual wallet balance
            # For now, return True as placeholder
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking balance: {e}")
            return False
    
    async def generate_copy_signal(self, copy_trade: CopyTrade) -> Optional[TradeSignal]:
        """Generate trading signal from copy trade"""
        try:
            # Get current token data
            token_data = await self.market_data.get_token_data(copy_trade.token_address)
            if not token_data:
                return None
            
            # Calculate confidence based on trader performance
            trader = self.tracked_traders.get(copy_trade.original_trader)
            confidence = trader.success_rate if trader else 0.5
            
            signal = TradeSignal(
                token_address=copy_trade.token_address,
                action=copy_trade.action,
                price=copy_trade.price,
                quantity=copy_trade.quantity,
                confidence=confidence,
                source="copy_bot",
                timestamp=copy_trade.timestamp,
                metadata={
                    "original_trader": copy_trade.original_trader,
                    "original_trade_hash": copy_trade.original_trade_hash,
                    "delay_seconds": copy_trade.delay,
                    "trader_success_rate": confidence
                }
            )
            
            self.stats["signals_generated"] += 1
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating copy signal: {e}")
            return None
    
    async def process_signal(self, signal: TradeSignal) -> bool:
        """Process a copy trading signal"""
        try:
            self.logger.info(f"Processing copy signal: {signal.action} {signal.quantity} of {signal.token_address}")
            
            # Send notification
            await self.notification_manager.send_notification(
                f"📋 Copy Signal: {signal.action.upper()} {signal.metadata['original_trader'][:8]}\n"
                f"Token: {signal.token_address[:8]}...\n"
                f"Quantity: {signal.quantity:.6f}\n"
                f"Delay: {signal.metadata['delay_seconds']}s\n"
                f"Trader Success Rate: {signal.metadata['trader_success_rate']:.1%}"
            )
            
            # Execute trade if auto-trading is enabled
            if self.config.copy_config.enabled:
                return await self.execute_trade(signal)
            else:
                self.logger.info("Copy trading disabled, signal logged for manual review")
                return True
                
        except Exception as e:
            await self.handle_error(e, "process_signal")
            return False
    
    async def update_trader_performance(self):
        """Update trader performance statistics"""
        try:
            for address, trader in self.tracked_traders.items():
                # Get recent performance data
                performance = await self.get_trader_performance(address)
                if performance:
                    trader.success_rate = performance.get("success_rate", 0.0)
                    trader.total_trades = performance.get("total_trades", 0)
                    trader.total_volume = performance.get("total_volume", 0.0)
                    
        except Exception as e:
            await self.handle_error(e, "update_trader_performance")
    
    async def get_trader_performance(self, address: str) -> Optional[Dict[str, Any]]:
        """Get performance data for a trader"""
        try:
            # This would fetch actual performance data from APIs
            # For now, return placeholder data
            return {
                "success_rate": 0.65,
                "total_trades": 100,
                "total_volume": 10000.0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting trader performance: {e}")
            return None
    
    async def get_copy_stats(self) -> Dict[str, Any]:
        """Get copy-specific statistics"""
        return {
            **self.get_status(),
            "tracked_traders_count": len(self.tracked_traders),
            "recent_trades_count": len(self.recent_trades),
            "copied_trades_count": len(self.copied_trades),
            "copy_success_rate": self.copy_success_rate,
            "trader_performance": {
                addr: {
                    "success_rate": trader.success_rate,
                    "total_trades": trader.total_trades,
                    "total_volume": trader.total_volume
                }
                for addr, trader in self.tracked_traders.items()
            }
        } 