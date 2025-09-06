"""
Market Maker Tracker - AI agent for tracking market maker activities and strategies
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

from ..config.settings import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MarketMakerActivity:
    """Represents market maker activity"""
    market_maker_id: str
    exchange: str
    symbol: str
    activity_type: str  # 'liquidity_provision', 'arbitrage', 'market_making'
    side: str  # 'buy', 'sell', 'both'
    volume: float
    price_impact: float
    timestamp: datetime
    duration: int  # seconds
    profit_loss: Optional[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class MarketMakerStrategy:
    """Represents a detected market maker strategy"""
    strategy_type: str  # 'grid_trading', 'mean_reversion', 'momentum', 'arbitrage'
    market_maker_id: str
    symbols: List[str]
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    confidence: float
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class LiquidityEvent:
    """Represents a liquidity event"""
    event_type: str  # 'liquidity_added', 'liquidity_removed', 'spread_widening'
    symbol: str
    exchange: str
    impact: float  # price impact
    volume: float
    duration: int  # seconds
    market_makers_involved: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class MarketMakerTracker:
    """
    AI agent for tracking market maker activities and strategies
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = True
        self.is_running = False
        
        # Settings
        self.track_top_market_makers = config.get("track_top_market_makers", 50)
        self.min_activity_volume = config.get("min_activity_volume", 100_000)
        self.tracking_interval = config.get("tracking_interval", 60)  # seconds
        self.strategy_detection_enabled = config.get("strategy_detection_enabled", True)
        self.liquidity_monitoring_enabled = config.get("liquidity_monitoring_enabled", True)
        
        # AI Models
        self.strategy_classifier = None
        self.liquidity_predictor = None
        self.scaler: Optional[StandardScaler] = None
        
        # Data storage
        self.market_maker_activities: List[MarketMakerActivity] = []
        self.detected_strategies: List[MarketMakerStrategy] = []
        self.liquidity_events: List[LiquidityEvent] = []
        self.market_maker_profiles: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self.metrics = {
            "total_activities_tracked": 0,
            "strategies_detected": 0,
            "liquidity_events_detected": 0,
            "accuracy": 0.0,
            "processing_time": 0.0,
            "last_update": None
        }
        
        # Task management
        self.tracker_task: Optional[asyncio.Task] = None
        self.strategy_task: Optional[asyncio.Task] = None
        self.liquidity_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize the market maker tracker"""
        try:
            logger.info("Initializing Market Maker Tracker...")
            
            # Initialize AI models
            await self._initialize_models()
            
            # Load historical data
            await self._load_historical_data()
            
            # Start tracking tasks
            self.tracker_task = asyncio.create_task(self._tracking_loop())
            if self.strategy_detection_enabled:
                self.strategy_task = asyncio.create_task(self._strategy_detection_loop())
            if self.liquidity_monitoring_enabled:
                self.liquidity_task = asyncio.create_task(self._liquidity_monitoring_loop())
            
            self.is_running = True
            logger.info("Market Maker Tracker initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Market Maker Tracker: {e}")
            return False
    
    async def _initialize_models(self):
        """Initialize AI models"""
        try:
            # Initialize strategy classifier
            self.strategy_classifier = LinearRegression()
            
            # Initialize liquidity predictor
            self.liquidity_predictor = LinearRegression()
            
            # Initialize scaler
            self.scaler = StandardScaler()
            
            logger.info("AI models initialized for market maker tracking")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    async def _tracking_loop(self):
        """Main tracking loop"""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Track market maker activities
                activities = await self._track_market_maker_activities()
                self.market_maker_activities.extend(activities)
                
                # Update market maker profiles
                await self._update_market_maker_profiles()
                
                # Update metrics
                self.metrics["total_activities_tracked"] += len(activities)
                self.metrics["processing_time"] = time.time() - start_time
                self.metrics["last_update"] = datetime.now()
                
                # Wait for next tracking cycle
                await asyncio.sleep(self.tracking_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in tracking loop: {e}")
                await asyncio.sleep(30)
    
    async def _strategy_detection_loop(self):
        """Strategy detection loop"""
        while self.is_running:
            try:
                # Detect market maker strategies
                strategies = await self._detect_strategies()
                self.detected_strategies.extend(strategies)
                
                # Update strategy performance
                await self._update_strategy_performance()
                
                # Wait for next strategy detection cycle (every 15 minutes)
                await asyncio.sleep(900)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in strategy detection loop: {e}")
                await asyncio.sleep(300)
    
    async def _liquidity_monitoring_loop(self):
        """Liquidity monitoring loop"""
        while self.is_running:
            try:
                # Monitor liquidity events
                events = await self._monitor_liquidity_events()
                self.liquidity_events.extend(events)
                
                # Predict liquidity changes
                await self._predict_liquidity_changes()
                
                # Wait for next liquidity monitoring cycle (every 30 seconds)
                await asyncio.sleep(30)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in liquidity monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _track_market_maker_activities(self) -> List[MarketMakerActivity]:
        """Track market maker activities"""
        activities = []
        
        try:
            # Get market maker data from exchanges
            exchange_data = await self._get_exchange_market_maker_data()
            
            for exchange, data in exchange_data.items():
                for activity_data in data:
                    if activity_data.get("volume", 0) >= self.min_activity_volume:
                        activity = MarketMakerActivity(
                            market_maker_id=activity_data.get("market_maker_id"),
                            exchange=exchange,
                            symbol=activity_data.get("symbol"),
                            activity_type=activity_data.get("activity_type", "unknown"),
                            side=activity_data.get("side", "both"),
                            volume=activity_data.get("volume", 0),
                            price_impact=activity_data.get("price_impact", 0),
                            timestamp=datetime.now(),
                            duration=activity_data.get("duration", 0),
                            profit_loss=activity_data.get("profit_loss")
                        )
                        activities.append(activity)
            
            logger.info(f"Tracked {len(activities)} market maker activities")
            
        except Exception as e:
            logger.error(f"Error tracking market maker activities: {e}")
        
        return activities
    
    async def _get_exchange_market_maker_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get market maker data from exchanges"""
        try:
            settings = get_settings()
            enabled_exchanges = [name for name, config in settings.exchanges.items() if config.enabled]
            
            exchange_data = {}
            
            # Gather data from all exchanges concurrently
            tasks = []
            for exchange in enabled_exchanges:
                task = asyncio.create_task(self._get_exchange_data(exchange))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error getting data from {enabled_exchanges[i]}: {result}")
                elif isinstance(result, list):
                    exchange_data[enabled_exchanges[i]] = result
            
            return exchange_data
            
        except Exception as e:
            logger.error(f"Error getting exchange market maker data: {e}")
            return {}
    
    async def _get_exchange_data(self, exchange: str) -> List[Dict[str, Any]]:
        """Get market maker data from specific exchange"""
        try:
            if exchange == "hyperliquid":
                return await self._get_hyperliquid_market_maker_data()
            elif exchange == "binance":
                return await self._get_binance_market_maker_data()
            elif exchange == "bybit":
                return await self._get_bybit_market_maker_data()
            else:
                return []
        except Exception as e:
            logger.error(f"Error getting data from {exchange}: {e}")
            return []
    
    async def _get_hyperliquid_market_maker_data(self) -> List[Dict[str, Any]]:
        """Get market maker data from Hyperliquid"""
        # Implementation would connect to Hyperliquid API
        # For now, return mock data
        return []
    
    async def _get_binance_market_maker_data(self) -> List[Dict[str, Any]]:
        """Get market maker data from Binance"""
        # Implementation would connect to Binance API
        # For now, return mock data
        return []
    
    async def _get_bybit_market_maker_data(self) -> List[Dict[str, Any]]:
        """Get market maker data from Bybit"""
        # Implementation would connect to Bybit API
        # For now, return mock data
        return []
    
    async def _update_market_maker_profiles(self):
        """Update market maker profiles"""
        try:
            # Group activities by market maker
            market_maker_activities = {}
            for activity in self.market_maker_activities[-1000:]:  # Last 1000 activities
                mm_id = activity.market_maker_id
                if mm_id not in market_maker_activities:
                    market_maker_activities[mm_id] = []
                market_maker_activities[mm_id].append(activity)
            
            # Update profiles
            for mm_id, activities in market_maker_activities.items():
                profile = self._calculate_market_maker_profile(mm_id, activities)
                self.market_maker_profiles[mm_id] = profile
            
            logger.info(f"Updated {len(self.market_maker_profiles)} market maker profiles")
            
        except Exception as e:
            logger.error(f"Error updating market maker profiles: {e}")
    
    def _calculate_market_maker_profile(self, mm_id: str, activities: List[MarketMakerActivity]) -> Dict[str, Any]:
        """Calculate market maker profile from activities"""
        if not activities:
            return {}
        
        total_volume = sum(a.volume for a in activities)
        avg_price_impact = np.mean([a.price_impact for a in activities])
        activity_frequency = len(activities) / 24  # activities per hour
        
        preferred_symbols = {}
        for activity in activities:
            symbol = activity.symbol
            preferred_symbols[symbol] = preferred_symbols.get(symbol, 0) + activity.volume
        
        preferred_symbols = dict(sorted(preferred_symbols.items(), key=lambda x: x[1], reverse=True)[:5])
        
        return {
            "total_volume": total_volume,
            "avg_price_impact": avg_price_impact,
            "activity_frequency": activity_frequency,
            "preferred_symbols": preferred_symbols,
            "last_active": max(a.timestamp for a in activities),
            "total_activities": len(activities)
        }
    
    async def _detect_strategies(self) -> List[MarketMakerStrategy]:
        """Detect market maker strategies"""
        strategies = []
        
        try:
            # Analyze recent activities for patterns
            recent_activities = [a for a in self.market_maker_activities 
                               if a.timestamp > datetime.now() - timedelta(hours=1)]
            
            # Group by market maker
            mm_activities = {}
            for activity in recent_activities:
                mm_id = activity.market_maker_id
                if mm_id not in mm_activities:
                    mm_activities[mm_id] = []
                mm_activities[mm_id].append(activity)
            
            # Detect strategies for each market maker
            for mm_id, activities in mm_activities.items():
                if len(activities) >= 5:  # Need minimum activities for pattern detection
                    strategy = self._detect_market_maker_strategy(mm_id, activities)
                    if strategy:
                        strategies.append(strategy)
            
            self.metrics["strategies_detected"] += len(strategies)
            logger.info(f"Detected {len(strategies)} market maker strategies")
            
        except Exception as e:
            logger.error(f"Error detecting strategies: {e}")
        
        return strategies
    
    def _detect_market_maker_strategy(self, mm_id: str, activities: List[MarketMakerActivity]) -> Optional[MarketMakerStrategy]:
        """Detect strategy for a specific market maker"""
        try:
            # Analyze activity patterns
            buy_activities = [a for a in activities if a.side == "buy"]
            sell_activities = [a for a in activities if a.side == "sell"]
            
            # Calculate strategy indicators
            buy_volume = sum(a.volume for a in buy_activities)
            sell_volume = sum(a.volume for a in sell_activities)
            total_volume = buy_volume + sell_volume
            
            if total_volume == 0:
                return None
            
            buy_ratio = buy_volume / total_volume
            avg_price_impact = np.mean([a.price_impact for a in activities])
            activity_frequency = len(activities) / 60  # activities per minute
            
            # Determine strategy type
            strategy_type = self._classify_strategy(buy_ratio, avg_price_impact, activity_frequency)
            
            if strategy_type:
                return MarketMakerStrategy(
                    strategy_type=strategy_type,
                    market_maker_id=mm_id,
                    symbols=list(set(a.symbol for a in activities)),
                    parameters={
                        "buy_ratio": buy_ratio,
                        "avg_price_impact": avg_price_impact,
                        "activity_frequency": activity_frequency
                    },
                    performance_metrics=self._calculate_strategy_performance(activities),
                    confidence=self._calculate_strategy_confidence(activities),
                    last_updated=datetime.now()
                )
            
        except Exception as e:
            logger.error(f"Error detecting strategy for {mm_id}: {e}")
        
        return None
    
    def _classify_strategy(self, buy_ratio: float, price_impact: float, frequency: float) -> Optional[str]:
        """Classify market maker strategy based on indicators"""
        if frequency > 10:  # High frequency
            if abs(buy_ratio - 0.5) < 0.1:  # Balanced
                return "grid_trading"
            else:
                return "momentum"
        elif price_impact < 0.001:  # Low price impact
            return "arbitrage"
        elif abs(buy_ratio - 0.5) < 0.2:  # Relatively balanced
            return "mean_reversion"
        else:
            return "directional"
    
    def _calculate_strategy_performance(self, activities: List[MarketMakerActivity]) -> Dict[str, float]:
        """Calculate performance metrics for a strategy"""
        if not activities:
            return {}
        
        total_volume = sum(a.volume for a in activities)
        avg_price_impact = np.mean([a.price_impact for a in activities])
        
        # Calculate profit/loss if available
        profits = [a.profit_loss for a in activities if a.profit_loss is not None]
        total_pnl = sum(profits) if profits else 0
        
        return {
            "total_volume": total_volume,
            "avg_price_impact": avg_price_impact,
            "total_pnl": total_pnl,
            "profit_rate": total_pnl / total_volume if total_volume > 0 else 0
        }
    
    def _calculate_strategy_confidence(self, activities: List[MarketMakerActivity]) -> float:
        """Calculate confidence in strategy detection"""
        if len(activities) < 5:
            return 0.3
        
        # Higher confidence with more activities and consistent patterns
        volume_consistency = 1 - np.std([a.volume for a in activities]) / np.mean([a.volume for a in activities])
        time_consistency = 1 - np.std([a.timestamp.timestamp() for a in activities]) / 3600  # 1 hour
        
        return min(0.9, (len(activities) / 20) * 0.5 + volume_consistency * 0.3 + time_consistency * 0.2)
    
    async def _update_strategy_performance(self):
        """Update performance of detected strategies"""
        try:
            for strategy in self.detected_strategies:
                # Get recent activities for this market maker
                recent_activities = [a for a in self.market_maker_activities 
                                   if a.market_maker_id == strategy.market_maker_id and
                                   a.timestamp > strategy.last_updated]
                
                if recent_activities:
                    # Update performance metrics
                    strategy.performance_metrics = self._calculate_strategy_performance(recent_activities)
                    strategy.last_updated = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating strategy performance: {e}")
    
    async def _monitor_liquidity_events(self) -> List[LiquidityEvent]:
        """Monitor liquidity events"""
        events = []
        
        try:
            # Analyze recent activities for liquidity events
            recent_activities = [a for a in self.market_maker_activities 
                               if a.timestamp > datetime.now() - timedelta(minutes=5)]
            
            # Detect liquidity additions/removals
            liquidity_changes = self._detect_liquidity_changes(recent_activities)
            events.extend(liquidity_changes)
            
            # Detect spread widening
            spread_events = self._detect_spread_widening(recent_activities)
            events.extend(spread_events)
            
            self.metrics["liquidity_events_detected"] += len(events)
            logger.info(f"Detected {len(events)} liquidity events")
            
        except Exception as e:
            logger.error(f"Error monitoring liquidity events: {e}")
        
        return events
    
    def _detect_liquidity_changes(self, activities: List[MarketMakerActivity]) -> List[LiquidityEvent]:
        """Detect liquidity addition/removal events"""
        events = []
        
        try:
            # Group by symbol and time window
            symbol_activities = {}
            for activity in activities:
                symbol = activity.symbol
                if symbol not in symbol_activities:
                    symbol_activities[symbol] = []
                symbol_activities[symbol].append(activity)
            
            for symbol, symbol_acts in symbol_activities.items():
                # Calculate net liquidity change
                net_volume = sum(a.volume for a in symbol_acts)
                
                if abs(net_volume) > self.min_activity_volume:
                    event = LiquidityEvent(
                        event_type="liquidity_added" if net_volume > 0 else "liquidity_removed",
                        symbol=symbol,
                        exchange=symbol_acts[0].exchange,
                        impact=np.mean([a.price_impact for a in symbol_acts]),
                        volume=abs(net_volume),
                        duration=300,  # 5 minutes
                        market_makers_involved=list(set(a.market_maker_id for a in symbol_acts)),
                        timestamp=datetime.now()
                    )
                    events.append(event)
            
        except Exception as e:
            logger.error(f"Error detecting liquidity changes: {e}")
        
        return events
    
    def _detect_spread_widening(self, activities: List[MarketMakerActivity]) -> List[LiquidityEvent]:
        """Detect spread widening events"""
        events = []
        
        try:
            # This would analyze bid-ask spreads
            # For now, return empty list
            pass
            
        except Exception as e:
            logger.error(f"Error detecting spread widening: {e}")
        
        return events
    
    async def _predict_liquidity_changes(self):
        """Predict future liquidity changes"""
        try:
            # Use AI model to predict liquidity changes
            # This would analyze historical patterns and current market conditions
            pass
            
        except Exception as e:
            logger.error(f"Error predicting liquidity changes: {e}")
    
    async def _load_historical_data(self):
        """Load historical market maker data"""
        try:
            # Load from database or file
            logger.info("Loading historical market maker data...")
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    def get_top_market_makers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top market makers by volume"""
        sorted_mms = sorted(self.market_maker_profiles.items(), 
                           key=lambda x: x[1].get("total_volume", 0), reverse=True)
        return [{"id": mm_id, **profile} for mm_id, profile in sorted_mms[:limit]]
    
    def get_recent_strategies(self, hours: int = 24) -> List[MarketMakerStrategy]:
        """Get recent strategies"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [s for s in self.detected_strategies if s.last_updated > cutoff_time]
    
    def get_liquidity_events(self, hours: int = 24) -> List[LiquidityEvent]:
        """Get recent liquidity events"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [e for e in self.liquidity_events if e.timestamp > cutoff_time]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics
    
    async def stop(self):
        """Stop the market maker tracker"""
        self.is_running = False
        
        if self.tracker_task:
            self.tracker_task.cancel()
        
        if self.strategy_task:
            self.strategy_task.cancel()
        
        if self.liquidity_task:
            self.liquidity_task.cancel()
        
        logger.info("Market Maker Tracker stopped")


def create_market_maker_tracker(config: Dict[str, Any]) -> MarketMakerTracker:
    """Create a new market maker tracker instance"""
    return MarketMakerTracker(config) 