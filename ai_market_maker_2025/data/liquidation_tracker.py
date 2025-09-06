"""
Liquidation Tracker - Monitors and predicts liquidation events
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
import numpy as np

from ..config.settings import get_settings, get_settings_manager
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LiquidationEvent:
    """Represents a liquidation event"""
    exchange: str
    symbol: str
    trader_address: str
    side: str  # 'long' or 'short'
    size: float
    price: float
    liquidation_price: float
    timestamp: datetime
    leverage: float
    pnl: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class LiquidationPrediction:
    """Represents a liquidation prediction"""
    symbol: str
    side: str
    probability: float
    estimated_time: datetime
    total_size: float
    affected_traders: int
    confidence: float
    factors: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class LiquidationCluster:
    """Represents a cluster of liquidation events"""
    symbol: str
    side: str
    events: List[LiquidationEvent]
    total_size: float
    start_time: datetime
    end_time: datetime
    cascade_probability: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class LiquidationTracker:
    """
    Tracks liquidation events and predicts future liquidations
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = True
        self.is_running = False
        
        # Settings
        self.min_liquidation_size = config.get("min_liquidation_size", 100_000)
        self.prediction_horizon = config.get("prediction_horizon", 3600)  # 1 hour
        self.cluster_threshold = config.get("cluster_threshold", 5)  # events
        self.update_interval = config.get("update_interval", 30)  # seconds
        
        # Data storage
        self.liquidation_events: List[LiquidationEvent] = []
        self.predictions: List[LiquidationPrediction] = []
        self.clusters: List[LiquidationCluster] = []
        self.market_data: Dict[str, Dict[str, Any]] = {}
        
        # Historical data for analysis
        self.historical_events: List[LiquidationEvent] = []
        self.patterns: Dict[str, Any] = {}
        
        # Performance metrics
        self.metrics = {
            "total_events": 0,
            "predictions_made": 0,
            "predictions_correct": 0,
            "accuracy": 0.0,
            "last_update": None
        }
        
        # Session for HTTP requests
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Task management
        self.tracker_task: Optional[asyncio.Task] = None
        self.prediction_task: Optional[asyncio.Task] = None
        self.analysis_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize the liquidation tracker"""
        try:
            logger.info("Initializing Liquidation Tracker...")
            
            # Create HTTP session
            self.session = aiohttp.ClientSession()
            
            # Load historical data
            await self._load_historical_data()
            
            # Start tracking tasks
            self.tracker_task = asyncio.create_task(self._tracking_loop())
            self.prediction_task = asyncio.create_task(self._prediction_loop())
            self.analysis_task = asyncio.create_task(self._analysis_loop())
            
            self.is_running = True
            logger.info("Liquidation Tracker initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Liquidation Tracker: {e}")
            return False
    
    async def _tracking_loop(self):
        """Main tracking loop for liquidation events"""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Get liquidation events from all exchanges
                await self._update_liquidation_events()
                
                # Detect liquidation clusters
                await self._detect_clusters()
                
                # Update metrics
                self.metrics["last_update"] = datetime.now()
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in tracking loop: {e}")
                await asyncio.sleep(30)
    
    async def _prediction_loop(self):
        """Prediction loop for future liquidations"""
        while self.is_running:
            try:
                # Generate liquidation predictions
                await self._generate_predictions()
                
                # Validate previous predictions
                await self._validate_predictions()
                
                # Wait for next prediction cycle
                await asyncio.sleep(300)  # 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
                await asyncio.sleep(60)
    
    async def _analysis_loop(self):
        """Analysis loop for pattern recognition"""
        while self.is_running:
            try:
                # Analyze liquidation patterns
                await self._analyze_patterns()
                
                # Update market correlations
                await self._update_correlations()
                
                # Wait for next analysis cycle
                await asyncio.sleep(1800)  # 30 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                await asyncio.sleep(300)
    
    async def _update_liquidation_events(self):
        """Update liquidation events from all exchanges"""
        try:
            settings = get_settings()
            settings_manager = get_settings_manager()
            enabled_exchanges = settings_manager.get_enabled_exchanges()
            
            all_events = []
            
            # Gather events from all exchanges concurrently
            tasks = []
            for exchange in enabled_exchanges:
                task = asyncio.create_task(self._get_exchange_liquidations(exchange))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error getting liquidations from {enabled_exchanges[i]}: {result}")
                elif isinstance(result, list):
                    all_events.extend(result)
            
            # Filter events by size
            filtered_events = [
                event for event in all_events
                if event.size >= self.min_liquidation_size
            ]
            
            # Add new events to storage
            for event in filtered_events:
                self.liquidation_events.append(event)
                self.historical_events.append(event)
                
                # Log significant liquidations
                if event.size > 1_000_000:  # $1M+
                    logger.warning(f"Large liquidation: {event.symbol} {event.side} ${event.size:,.0f} at ${event.price:,.2f}")
                
                # Emit event for other components
                await self._emit_liquidation_event("liquidation_detected", event.to_dict())
            
            # Keep only recent events in memory
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.liquidation_events = [
                event for event in self.liquidation_events
                if event.timestamp > cutoff_time
            ]
            
            self.metrics["total_events"] = len(self.historical_events)
            logger.info(f"Updated liquidation events: {len(filtered_events)} new events")
            
        except Exception as e:
            logger.error(f"Error updating liquidation events: {e}")
    
    async def _get_exchange_liquidations(self, exchange: str) -> List[LiquidationEvent]:
        """Get liquidation events from specific exchange"""
        try:
            if exchange == "hyperliquid":
                return await self._get_hyperliquid_liquidations()
            elif exchange == "binance":
                return await self._get_binance_liquidations()
            elif exchange == "bybit":
                return await self._get_bybit_liquidations()
            else:
                logger.warning(f"Unsupported exchange: {exchange}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting liquidations from {exchange}: {e}")
            return []
    
    async def _get_hyperliquid_liquidations(self) -> List[LiquidationEvent]:
        """Get liquidation events from Hyperliquid"""
        try:
            if not self.session:
                logger.warning("Session not initialized")
                return []
            
            # Hyperliquid liquidation endpoint
            url = "https://api.hyperliquid.xyz/info"
            payload = {
                "type": "liquidations",
                "user": "all"
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    events = []
                    for liquidation in data:
                        event = LiquidationEvent(
                            exchange="hyperliquid",
                            symbol=liquidation.get("coin", "UNKNOWN"),
                            trader_address=liquidation.get("user", "UNKNOWN"),
                            side=liquidation.get("side", "unknown"),
                            size=float(liquidation.get("size", 0)),
                            price=float(liquidation.get("price", 0)),
                            liquidation_price=float(liquidation.get("liquidationPrice", 0)),
                            timestamp=datetime.fromtimestamp(liquidation.get("timestamp", time.time())),
                            leverage=float(liquidation.get("leverage", 1)),
                            pnl=float(liquidation.get("pnl", 0))
                        )
                        events.append(event)
                    
                    return events
                else:
                    logger.error(f"Hyperliquid API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting Hyperliquid liquidations: {e}")
            return []
    
    async def _get_binance_liquidations(self) -> List[LiquidationEvent]:
        """Get liquidation events from Binance (placeholder)"""
        # Binance liquidation data would require private API access
        return []
    
    async def _get_bybit_liquidations(self) -> List[LiquidationEvent]:
        """Get liquidation events from Bybit (placeholder)"""
        # Bybit liquidation data would require private API access
        return []
    
    async def _detect_clusters(self):
        """Detect clusters of liquidation events"""
        try:
            # Group events by symbol and side
            grouped_events = defaultdict(list)
            for event in self.liquidation_events:
                key = f"{event.symbol}_{event.side}"
                grouped_events[key].append(event)
            
            # Detect clusters
            for key, events in grouped_events.items():
                if len(events) >= self.cluster_threshold:
                    # Sort events by timestamp
                    events.sort(key=lambda x: x.timestamp)
                    
                    # Check if events are within time window
                    time_window = timedelta(minutes=30)
                    cluster_events = []
                    
                    for i, event in enumerate(events):
                        cluster_events = [event]
                        
                        # Check subsequent events
                        for j in range(i + 1, len(events)):
                            if events[j].timestamp - event.timestamp <= time_window:
                                cluster_events.append(events[j])
                            else:
                                break
                        
                        # If cluster is large enough, create cluster object
                        if len(cluster_events) >= self.cluster_threshold:
                            symbol, side = key.split("_")
                            cluster = LiquidationCluster(
                                symbol=symbol,
                                side=side,
                                events=cluster_events,
                                total_size=sum(e.size for e in cluster_events),
                                start_time=cluster_events[0].timestamp,
                                end_time=cluster_events[-1].timestamp,
                                cascade_probability=self._calculate_cascade_probability(cluster_events)
                            )
                            
                            self.clusters.append(cluster)
                            
                            logger.warning(f"Liquidation cluster detected: {symbol} {side} {len(cluster_events)} events ${cluster.total_size:,.0f}")
                            
                            # Emit cluster event
                            await self._emit_liquidation_event("cluster_detected", cluster.to_dict())
                            break
            
            # Keep only recent clusters
            cutoff_time = datetime.now() - timedelta(hours=6)
            self.clusters = [
                cluster for cluster in self.clusters
                if cluster.end_time > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Error detecting clusters: {e}")
    
    async def _generate_predictions(self):
        """Generate liquidation predictions"""
        try:
            # Clear old predictions
            self.predictions.clear()
            
            # Analyze current market conditions
            market_conditions = await self._analyze_market_conditions()
            
            # Generate predictions for each symbol
            for symbol, conditions in market_conditions.items():
                for side in ["long", "short"]:
                    prediction = await self._predict_liquidation(symbol, side, conditions)
                    if prediction and prediction.probability > 0.3:  # 30% threshold
                        self.predictions.append(prediction)
                        
                        logger.info(f"Liquidation prediction: {symbol} {side} {prediction.probability:.1%} probability")
                        
                        # Emit prediction event
                        await self._emit_liquidation_event("prediction_generated", prediction.to_dict())
            
            self.metrics["predictions_made"] = len(self.predictions)
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
    
    async def _predict_liquidation(self, symbol: str, side: str, conditions: Dict[str, Any]) -> Optional[LiquidationPrediction]:
        """Predict liquidation for specific symbol and side"""
        try:
            # Get historical data for this symbol/side
            historical_events = [
                event for event in self.historical_events
                if event.symbol == symbol and event.side == side
            ]
            
            if not historical_events:
                return None
            
            # Calculate base probability from historical frequency
            recent_events = [
                event for event in historical_events
                if event.timestamp > datetime.now() - timedelta(hours=24)
            ]
            
            base_probability = len(recent_events) / 24  # events per hour
            
            # Adjust probability based on market conditions
            volatility_factor = conditions.get("volatility", 1.0)
            volume_factor = conditions.get("volume_ratio", 1.0)
            funding_factor = conditions.get("funding_rate", 0.0)
            
            adjusted_probability = base_probability * volatility_factor * volume_factor
            
            # Apply funding rate adjustment
            if side == "long" and funding_factor > 0.01:  # High positive funding
                adjusted_probability *= 1.5
            elif side == "short" and funding_factor < -0.01:  # High negative funding
                adjusted_probability *= 1.5
            
            # Cap probability at 95%
            adjusted_probability = min(adjusted_probability, 0.95)
            
            if adjusted_probability > 0.1:  # 10% minimum threshold
                return LiquidationPrediction(
                    symbol=symbol,
                    side=side,
                    probability=adjusted_probability,
                    estimated_time=datetime.now() + timedelta(minutes=30),
                    total_size=conditions.get("open_interest", 0),
                    affected_traders=len(recent_events),
                    confidence=min(adjusted_probability * 2, 1.0),
                    factors=[
                        "historical_frequency",
                        "volatility",
                        "volume",
                        "funding_rate"
                    ]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error predicting liquidation for {symbol} {side}: {e}")
            return None
    
    async def _analyze_market_conditions(self) -> Dict[str, Dict[str, Any]]:
        """Analyze current market conditions"""
        try:
            conditions = {}
            
            # Get market data for all symbols
            symbols = self._get_tracked_symbols()
            
            for symbol in symbols:
                market_data = await self._get_market_data(symbol)
                
                if market_data:
                    conditions[symbol] = {
                        "volatility": market_data.get("volatility", 1.0),
                        "volume_ratio": market_data.get("volume_ratio", 1.0),
                        "funding_rate": market_data.get("funding_rate", 0.0),
                        "open_interest": market_data.get("open_interest", 0),
                        "price_change_24h": market_data.get("price_change_24h", 0)
                    }
            
            return conditions
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return {}
    
    async def _get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get market data for specific symbol"""
        try:
            # This would fetch real market data from exchanges
            # For now, return placeholder data
            return {
                "volatility": np.random.uniform(0.5, 2.0),
                "volume_ratio": np.random.uniform(0.5, 3.0),
                "funding_rate": np.random.uniform(-0.01, 0.01),
                "open_interest": np.random.uniform(1000000, 10000000),
                "price_change_24h": np.random.uniform(-0.1, 0.1)
            }
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def _get_tracked_symbols(self) -> List[str]:
        """Get list of tracked symbols"""
        symbols = set()
        for event in self.liquidation_events:
            symbols.add(event.symbol)
        return list(symbols)
    
    def _calculate_cascade_probability(self, events: List[LiquidationEvent]) -> float:
        """Calculate probability of liquidation cascade"""
        try:
            if len(events) < 2:
                return 0.0
            
            # Calculate time intervals between events
            intervals = []
            for i in range(1, len(events)):
                interval = (events[i].timestamp - events[i-1].timestamp).total_seconds()
                intervals.append(interval)
            
            # If intervals are decreasing, cascade probability is high
            if len(intervals) >= 2:
                decreasing_intervals = sum(1 for i in range(1, len(intervals)) if intervals[i] < intervals[i-1])
                cascade_probability = decreasing_intervals / (len(intervals) - 1)
                return min(cascade_probability, 1.0)
            
            return 0.5  # Default probability
            
        except Exception as e:
            logger.error(f"Error calculating cascade probability: {e}")
            return 0.0
    
    async def _validate_predictions(self):
        """Validate previous predictions"""
        try:
            current_time = datetime.now()
            validated_predictions = []
            
            for prediction in self.predictions:
                if current_time > prediction.estimated_time:
                    # Check if liquidation occurred
                    occurred = await self._check_prediction_occurred(prediction)
                    
                    if occurred:
                        self.metrics["predictions_correct"] += 1
                    
                    # Remove expired prediction
                    continue
                
                validated_predictions.append(prediction)
            
            self.predictions = validated_predictions
            
            # Update accuracy
            if self.metrics["predictions_made"] > 0:
                self.metrics["accuracy"] = self.metrics["predictions_correct"] / self.metrics["predictions_made"]
            
        except Exception as e:
            logger.error(f"Error validating predictions: {e}")
    
    async def _check_prediction_occurred(self, prediction: LiquidationPrediction) -> bool:
        """Check if a prediction occurred"""
        try:
            # Check for liquidation events in the predicted time window
            window_start = prediction.estimated_time - timedelta(minutes=15)
            window_end = prediction.estimated_time + timedelta(minutes=15)
            
            matching_events = [
                event for event in self.liquidation_events
                if (event.symbol == prediction.symbol and 
                    event.side == prediction.side and
                    window_start <= event.timestamp <= window_end)
            ]
            
            return len(matching_events) > 0
            
        except Exception as e:
            logger.error(f"Error checking prediction occurrence: {e}")
            return False
    
    async def _analyze_patterns(self):
        """Analyze liquidation patterns"""
        try:
            # Analyze patterns by symbol
            symbol_patterns = defaultdict(list)
            for event in self.historical_events:
                symbol_patterns[event.symbol].append(event)
            
            for symbol, events in symbol_patterns.items():
                # Calculate average liquidation size
                avg_size = sum(e.size for e in events) / len(events)
                
                # Calculate time patterns
                timestamps = [e.timestamp for e in events]
                time_diffs = []
                for i in range(1, len(timestamps)):
                    diff = (timestamps[i] - timestamps[i-1]).total_seconds()
                    time_diffs.append(diff)
                
                avg_interval = sum(time_diffs) / len(time_diffs) if time_diffs else 0
                
                self.patterns[symbol] = {
                    "avg_size": avg_size,
                    "avg_interval": avg_interval,
                    "total_events": len(events),
                    "last_event": max(timestamps) if timestamps else None
                }
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
    
    async def _update_correlations(self):
        """Update market correlations"""
        try:
            # This would calculate correlations between different assets
            # and their liquidation patterns
            pass
        except Exception as e:
            logger.error(f"Error updating correlations: {e}")
    
    async def _emit_liquidation_event(self, event_type: str, data: Dict[str, Any]):
        """Emit liquidation event for other components"""
        # This would typically emit to an event bus or notification system
        logger.info(f"Liquidation event: {event_type} - {data}")
    
    async def _load_historical_data(self):
        """Load historical liquidation data"""
        try:
            # This would load from database or file
            # For now, start with empty data
            logger.info("No historical data loaded - starting fresh")
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    def get_recent_events(self, hours: int = 24) -> List[LiquidationEvent]:
        """Get recent liquidation events"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            event for event in self.liquidation_events
            if event.timestamp > cutoff_time
        ]
    
    def get_active_predictions(self) -> List[LiquidationPrediction]:
        """Get active liquidation predictions"""
        current_time = datetime.now()
        return [
            prediction for prediction in self.predictions
            if prediction.estimated_time > current_time
        ]
    
    def get_recent_clusters(self, hours: int = 6) -> List[LiquidationCluster]:
        """Get recent liquidation clusters"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            cluster for cluster in self.clusters
            if cluster.end_time > cutoff_time
        ]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.metrics,
            "active_predictions": len(self.get_active_predictions()),
            "recent_events": len(self.get_recent_events()),
            "recent_clusters": len(self.get_recent_clusters())
        }
    
    async def stop(self):
        """Stop the liquidation tracker"""
        logger.info("Stopping Liquidation Tracker...")
        self.is_running = False
        
        # Cancel tasks
        for task in [self.tracker_task, self.prediction_task, self.analysis_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close session
        if self.session:
            await self.session.close()
        
        logger.info("Liquidation Tracker stopped")


def create_liquidation_tracker(config: Dict[str, Any]) -> LiquidationTracker:
    """Create liquidation tracker instance"""
    return LiquidationTracker(config) 