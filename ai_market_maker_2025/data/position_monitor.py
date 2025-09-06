"""
Position Monitor - Tracks large trader positions across exchanges
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict

from ..config.settings import get_settings, get_settings_manager
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Position:
    """Represents a trading position"""
    trader_address: str
    exchange: str
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    leverage: float
    liquidation_price: Optional[float]
    timestamp: datetime
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class TraderProfile:
    """Represents a trader's profile"""
    address: str
    total_value: float
    total_pnl: float
    positions: List[Position]
    risk_level: str  # 'low', 'medium', 'high'
    trading_volume_24h: float
    last_active: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class PositionMonitor:
    """
    Monitors positions of large traders across multiple exchanges
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = True
        self.is_running = False
        
        # Settings
        self.track_top_traders = config.get("track_top_traders", 500)
        self.min_position_size = config.get("min_position_size", 1_000_000)
        self.update_interval = config.get("update_interval", 60)  # seconds
        self.max_workers = config.get("max_workers", 10)
        
        # Data storage
        self.traders: Dict[str, TraderProfile] = {}
        self.positions: Dict[str, Position] = {}
        self.exchange_data: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "last_update": None,
            "processing_time": 0
        }
        
        # Session for HTTP requests
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Task management
        self.monitor_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize the position monitor"""
        try:
            logger.info("Initializing Position Monitor...")
            
            # Create HTTP session
            self.session = aiohttp.ClientSession()
            
            # Start monitoring tasks
            self.monitor_task = asyncio.create_task(self._monitor_loop())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self.is_running = True
            logger.info("Position Monitor initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Position Monitor: {e}")
            return False
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Get top traders from all exchanges
                await self._update_top_traders()
                
                # Update positions for tracked traders
                await self._update_positions()
                
                # Analyze position changes
                await self._analyze_position_changes()
                
                # Update metrics
                self.metrics["processing_time"] = time.time() - start_time
                self.metrics["last_update"] = datetime.now()
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                await asyncio.sleep(30)  # Wait before retrying
    
    async def _update_top_traders(self):
        """Update list of top traders"""
        try:
            settings = get_settings()
            settings_manager = get_settings_manager()
            enabled_exchanges = settings_manager.get_enabled_exchanges()
            
            all_traders = []
            
            # Gather traders from all exchanges concurrently
            tasks = []
            for exchange in enabled_exchanges:
                task = asyncio.create_task(self._get_exchange_traders(exchange))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error getting traders from {enabled_exchanges[i]}: {result}")
                elif isinstance(result, list):
                    all_traders.extend(result)
            
            # Sort by total value and take top traders
            all_traders.sort(key=lambda x: x["total_value"], reverse=True)
            top_traders = all_traders[:self.track_top_traders]
            
            # Update trader profiles
            for trader_data in top_traders:
                address = trader_data["address"]
                if address not in self.traders:
                    self.traders[address] = TraderProfile(
                        address=address,
                        total_value=trader_data["total_value"],
                        total_pnl=trader_data.get("total_pnl", 0),
                        positions=[],
                        risk_level=self._calculate_risk_level(trader_data),
                        trading_volume_24h=trader_data.get("trading_volume_24h", 0),
                        last_active=datetime.now()
                    )
                else:
                    # Update existing trader
                    self.traders[address].total_value = trader_data["total_value"]
                    self.traders[address].total_pnl = trader_data.get("total_pnl", 0)
                    self.traders[address].trading_volume_24h = trader_data.get("trading_volume_24h", 0)
                    self.traders[address].last_active = datetime.now()
            
            logger.info(f"Updated {len(self.traders)} trader profiles")
            
        except Exception as e:
            logger.error(f"Error updating top traders: {e}")
    
    async def _get_exchange_traders(self, exchange: str) -> List[Dict[str, Any]]:
        """Get top traders from specific exchange"""
        try:
            if exchange == "hyperliquid":
                return await self._get_hyperliquid_traders()
            elif exchange == "binance":
                return await self._get_binance_traders()
            elif exchange == "bybit":
                return await self._get_bybit_traders()
            else:
                logger.warning(f"Unsupported exchange: {exchange}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting traders from {exchange}: {e}")
            return []
    
    async def _get_hyperliquid_traders(self) -> List[Dict[str, Any]]:
        """Get top traders from Hyperliquid"""
        try:
            if not self.session:
                logger.warning("Session not initialized")
                return []
            
            # Hyperliquid API endpoint for top traders
            url = "https://api.hyperliquid.xyz/info"
            payload = {
                "type": "allMids",
                "user": "all"
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    traders = []
                    for user_data in data:
                        if user_data.get("marginSummary"):
                            margin = user_data["marginSummary"]
                            total_value = float(margin.get("accountValue", 0))
                            
                            if total_value >= self.min_position_size:
                                traders.append({
                                    "address": user_data["user"],
                                    "total_value": total_value,
                                    "total_pnl": float(margin.get("unrealizedPnl", 0)),
                                    "trading_volume_24h": float(margin.get("volume24h", 0))
                                })
                    
                    return traders
                else:
                    logger.error(f"Hyperliquid API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting Hyperliquid traders: {e}")
            return []
    
    async def _get_binance_traders(self) -> List[Dict[str, Any]]:
        """Get top traders from Binance (placeholder)"""
        # Binance doesn't provide public trader data
        # This would require private API access or alternative data sources
        return []
    
    async def _get_bybit_traders(self) -> List[Dict[str, Any]]:
        """Get top traders from Bybit (placeholder)"""
        # Bybit doesn't provide public trader data
        # This would require private API access or alternative data sources
        return []
    
    async def _update_positions(self):
        """Update positions for tracked traders"""
        try:
            # Get positions for all tracked traders concurrently
            tasks = []
            for address in self.traders.keys():
                task = asyncio.create_task(self._get_trader_positions(address))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                address = list(self.traders.keys())[i]
                if isinstance(result, Exception):
                    logger.error(f"Error getting positions for {address}: {result}")
                elif isinstance(result, list):
                    self.traders[address].positions = result
            
            logger.info(f"Updated positions for {len(self.traders)} traders")
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    async def _get_trader_positions(self, address: str) -> List[Position]:
        """Get positions for specific trader"""
        try:
            settings = get_settings()
            settings_manager = get_settings_manager()
            enabled_exchanges = settings_manager.get_enabled_exchanges()
            
            all_positions = []
            
            # Get positions from all exchanges
            for exchange in enabled_exchanges:
                if exchange == "hyperliquid":
                    positions = await self._get_hyperliquid_positions(address)
                    all_positions.extend(positions)
                # Add other exchanges as needed
            
            return all_positions
            
        except Exception as e:
            logger.error(f"Error getting positions for {address}: {e}")
            return []
    
    async def _get_hyperliquid_positions(self, address: str) -> List[Position]:
        """Get positions from Hyperliquid for specific trader"""
        try:
            if not self.session:
                logger.warning("Session not initialized")
                return []
            
            url = "https://api.hyperliquid.xyz/info"
            payload = {
                "type": "clearinghouseState",
                "user": address
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    positions = []
                    for asset_data in data.get("assetPositions", []):
                        if float(asset_data.get("position", 0)) != 0:
                            position = Position(
                                trader_address=address,
                                exchange="hyperliquid",
                                symbol=asset_data["coin"],
                                side="long" if float(asset_data["position"]) > 0 else "short",
                                size=abs(float(asset_data["position"])),
                                entry_price=float(asset_data.get("entryPx", 0)),
                                current_price=float(asset_data.get("markPx", 0)),
                                unrealized_pnl=float(asset_data.get("unrealizedPnl", 0)),
                                leverage=float(asset_data.get("leverage", 1)),
                                liquidation_price=float(asset_data.get("liquidationPx", 0)) if asset_data.get("liquidationPx") else None,
                                timestamp=datetime.now(),
                                last_updated=datetime.now()
                            )
                            positions.append(position)
                    
                    return positions
                else:
                    logger.error(f"Hyperliquid API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting Hyperliquid positions: {e}")
            return []
    
    async def _analyze_position_changes(self):
        """Analyze position changes and detect unusual activity"""
        try:
            for address, trader in self.traders.items():
                # Check for large position changes
                for position in trader.positions:
                    if position.size > self.min_position_size:
                        # Check if this is a new large position
                        position_key = f"{address}_{position.symbol}_{position.side}"
                        
                        if position_key not in self.positions:
                            # New large position detected
                            logger.info(f"New large position detected: {address} {position.symbol} {position.side} ${position.size:,.0f}")
                            
                            # Emit event for other components
                            await self._emit_position_event("new_large_position", {
                                "trader": address,
                                "position": position.to_dict()
                            })
                        
                        # Update stored position
                        self.positions[position_key] = position
                
                # Check for high leverage positions
                high_leverage_positions = [
                    p for p in trader.positions 
                    if p.leverage > 10 and p.size > self.min_position_size
                ]
                
                if high_leverage_positions:
                    logger.warning(f"High leverage positions detected for {address}: {len(high_leverage_positions)} positions")
                    
                    await self._emit_position_event("high_leverage_detected", {
                        "trader": address,
                        "positions": [p.to_dict() for p in high_leverage_positions]
                    })
            
        except Exception as e:
            logger.error(f"Error analyzing position changes: {e}")
    
    def _calculate_risk_level(self, trader_data: Dict[str, Any]) -> str:
        """Calculate risk level for trader"""
        total_value = trader_data.get("total_value", 0)
        total_pnl = trader_data.get("total_pnl", 0)
        volume_24h = trader_data.get("trading_volume_24h", 0)
        
        # Simple risk calculation
        if total_value > 10_000_000 and abs(total_pnl) > total_value * 0.1:
            return "high"
        elif total_value > 1_000_000 and abs(total_pnl) > total_value * 0.05:
            return "medium"
        else:
            return "low"
    
    async def _emit_position_event(self, event_type: str, data: Dict[str, Any]):
        """Emit position event for other components"""
        # This would typically emit to an event bus or notification system
        logger.info(f"Position event: {event_type} - {data}")
    
    async def _cleanup_loop(self):
        """Cleanup old data periodically"""
        while self.is_running:
            try:
                # Remove old positions (older than 1 hour)
                cutoff_time = datetime.now() - timedelta(hours=1)
                old_positions = [
                    key for key, position in self.positions.items()
                    if position.last_updated < cutoff_time
                ]
                
                for key in old_positions:
                    del self.positions[key]
                
                if old_positions:
                    logger.info(f"Cleaned up {len(old_positions)} old positions")
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)
    
    def get_top_traders(self, limit: int = 100) -> List[TraderProfile]:
        """Get top traders by total value"""
        sorted_traders = sorted(
            self.traders.values(),
            key=lambda x: x.total_value,
            reverse=True
        )
        return sorted_traders[:limit]
    
    def get_trader_positions(self, address: str) -> List[Position]:
        """Get positions for specific trader"""
        if address in self.traders:
            return self.traders[address].positions
        return []
    
    def get_large_positions(self, min_size: Optional[float] = None) -> List[Position]:
        """Get all large positions"""
        if min_size is None:
            min_size = self.min_position_size
        
        large_positions = []
        for trader in self.traders.values():
            for position in trader.positions:
                if min_size is not None and position.size >= min_size:
                    large_positions.append(position)
        
        return large_positions
    
    def get_high_leverage_positions(self, min_leverage: float = 10) -> List[Position]:
        """Get high leverage positions"""
        high_leverage = []
        for trader in self.traders.values():
            for position in trader.positions:
                if position.leverage >= min_leverage:
                    high_leverage.append(position)
        
        return high_leverage
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.metrics,
            "total_traders": len(self.traders),
            "total_positions": len(self.positions),
            "large_positions": len(self.get_large_positions()),
            "high_leverage_positions": len(self.get_high_leverage_positions())
        }
    
    async def stop(self):
        """Stop the position monitor"""
        logger.info("Stopping Position Monitor...")
        self.is_running = False
        
        # Cancel tasks
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close session
        if self.session:
            await self.session.close()
        
        logger.info("Position Monitor stopped")


def create_position_monitor(config: Dict[str, Any]) -> PositionMonitor:
    """Create position monitor instance"""
    return PositionMonitor(config) 