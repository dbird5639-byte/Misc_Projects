"""
Signal Generator - AI agent for generating trading signals based on multiple data sources
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px

from ..config.settings import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradingSignal:
    """Represents a trading signal"""
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold', 'strong_buy', 'strong_sell'
    confidence: float  # 0-1 scale
    strength: float  # 0-1 scale
    entry_price: Optional[float]
    target_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    timeframe: str  # 'short', 'medium', 'long'
    reasoning: List[str]
    data_sources: List[str]
    timestamp: datetime
    expires_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class SignalMetrics:
    """Represents signal performance metrics"""
    total_signals: int
    successful_signals: int
    win_rate: float
    avg_profit: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class MarketCondition:
    """Represents market conditions for signal generation"""
    symbol: str
    trend: str  # 'bullish', 'bearish', 'sideways'
    volatility: float
    volume: float
    momentum: float
    support_level: Optional[float]
    resistance_level: Optional[float]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class SignalGenerator:
    """
    AI agent for generating trading signals based on multiple data sources
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = True
        self.is_running = False
        
        # Settings
        self.min_confidence = config.get("min_confidence", 0.6)
        self.min_strength = config.get("min_strength", 0.5)
        self.signal_lifetime = config.get("signal_lifetime", 3600)  # 1 hour
        self.update_interval = config.get("update_interval", 300)  # 5 minutes
        self.max_signals_per_symbol = config.get("max_signals_per_symbol", 3)
        
        # AI Models
        self.signal_classifier: Optional[RandomForestClassifier] = None
        self.price_predictor: Optional[GradientBoostingRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        
        # Data storage
        self.active_signals: List[TradingSignal] = []
        self.signal_history: List[TradingSignal] = []
        self.market_conditions: Dict[str, MarketCondition] = {}
        self.signal_metrics: List[SignalMetrics] = []
        
        # Performance metrics
        self.metrics = {
            "total_signals_generated": 0,
            "active_signals": 0,
            "accuracy": 0.0,
            "processing_time": 0.0,
            "last_update": None
        }
        
        # Task management
        self.generator_task: Optional[asyncio.Task] = None
        self.validation_task: Optional[asyncio.Task] = None
        self.optimization_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize the signal generator"""
        try:
            logger.info("Initializing Signal Generator...")
            
            # Initialize AI models
            await self._initialize_models()
            
            # Load historical data
            await self._load_historical_data()
            
            # Start signal generation tasks
            self.generator_task = asyncio.create_task(self._signal_generation_loop())
            self.validation_task = asyncio.create_task(self._signal_validation_loop())
            self.optimization_task = asyncio.create_task(self._model_optimization_loop())
            
            self.is_running = True
            logger.info("Signal Generator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Signal Generator: {e}")
            return False
    
    async def _initialize_models(self):
        """Initialize AI models"""
        try:
            # Initialize signal classifier
            self.signal_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Initialize price predictor
            self.price_predictor = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
            
            # Initialize scaler
            self.scaler = StandardScaler()
            
            logger.info("AI models initialized for signal generation")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    async def _signal_generation_loop(self):
        """Main signal generation loop"""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Update market conditions
                await self._update_market_conditions()
                
                # Generate signals for all symbols
                new_signals = await self._generate_signals()
                self.active_signals.extend(new_signals)
                
                # Clean up expired signals
                await self._cleanup_expired_signals()
                
                # Update metrics
                self.metrics["total_signals_generated"] += len(new_signals)
                self.metrics["active_signals"] = len(self.active_signals)
                self.metrics["processing_time"] = time.time() - start_time
                self.metrics["last_update"] = datetime.now()
                
                # Wait for next signal generation cycle
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in signal generation loop: {e}")
                await asyncio.sleep(60)
    
    async def _signal_validation_loop(self):
        """Signal validation loop"""
        while self.is_running:
            try:
                # Validate active signals
                await self._validate_active_signals()
                
                # Update signal performance
                await self._update_signal_performance()
                
                # Wait for next validation cycle (every 10 minutes)
                await asyncio.sleep(600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in signal validation loop: {e}")
                await asyncio.sleep(300)
    
    async def _model_optimization_loop(self):
        """Model optimization loop"""
        while self.is_running:
            try:
                # Retrain models with new data
                await self._retrain_models()
                
                # Optimize model parameters
                await self._optimize_model_parameters()
                
                # Wait for next optimization cycle (every 6 hours)
                await asyncio.sleep(21600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in model optimization loop: {e}")
                await asyncio.sleep(3600)
    
    async def _update_market_conditions(self):
        """Update market conditions for all symbols"""
        try:
            symbols = self._get_tracked_symbols()
            
            for symbol in symbols:
                condition = await self._analyze_market_condition(symbol)
                if condition:
                    self.market_conditions[symbol] = condition
            
            logger.info(f"Updated market conditions for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Error updating market conditions: {e}")
    
    def _get_tracked_symbols(self) -> List[str]:
        """Get list of tracked symbols"""
        # This would come from configuration or data sources
        return ["BTC", "ETH", "SOL", "AVAX", "MATIC"]
    
    async def _analyze_market_condition(self, symbol: str) -> Optional[MarketCondition]:
        """Analyze market condition for a symbol"""
        try:
            # Get market data
            market_data = await self._get_market_data(symbol)
            
            if not market_data:
                return None
            
            # Calculate technical indicators
            trend = self._calculate_trend(market_data)
            volatility = self._calculate_volatility(market_data)
            volume = market_data.get("volume_24h", 0)
            momentum = self._calculate_momentum(market_data)
            support, resistance = self._calculate_support_resistance(market_data)
            
            return MarketCondition(
                symbol=symbol,
                trend=trend,
                volatility=volatility,
                volume=volume,
                momentum=momentum,
                support_level=support,
                resistance_level=resistance,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing market condition for {symbol}: {e}")
            return None
    
    async def _get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get market data for a symbol"""
        try:
            # This would fetch from exchange APIs
            # For now, return mock data
            return {
                "price": 45000 + np.random.normal(0, 1000),
                "volume_24h": 1000000 + np.random.normal(0, 100000),
                "price_change_24h": np.random.normal(0, 0.05),
                "high_24h": 47000,
                "low_24h": 43000
            }
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def _calculate_trend(self, market_data: Dict[str, Any]) -> str:
        """Calculate market trend"""
        try:
            price_change = market_data.get("price_change_24h", 0)
            
            if price_change > 0.02:
                return "bullish"
            elif price_change < -0.02:
                return "bearish"
            else:
                return "sideways"
                
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return "sideways"
    
    def _calculate_volatility(self, market_data: Dict[str, Any]) -> float:
        """Calculate market volatility"""
        try:
            high = market_data.get("high_24h", 0)
            low = market_data.get("low_24h", 0)
            price = market_data.get("price", 0)
            
            if price > 0:
                return (high - low) / price
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.0
    
    def _calculate_momentum(self, market_data: Dict[str, Any]) -> float:
        """Calculate market momentum"""
        try:
            return market_data.get("price_change_24h", 0)
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return 0.0
    
    def _calculate_support_resistance(self, market_data: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        """Calculate support and resistance levels"""
        try:
            high = market_data.get("high_24h", 0)
            low = market_data.get("low_24h", 0)
            price = market_data.get("price", 0)
            
            # Simple support/resistance calculation
            support = low * 0.99
            resistance = high * 1.01
            
            return support, resistance
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return None, None
    
    async def _generate_signals(self) -> List[TradingSignal]:
        """Generate trading signals"""
        signals = []
        
        try:
            for symbol, condition in self.market_conditions.items():
                # Check if we already have max signals for this symbol
                existing_signals = [s for s in self.active_signals if s.symbol == symbol]
                if len(existing_signals) >= self.max_signals_per_symbol:
                    continue
                
                # Generate signal for this symbol
                signal = await self._generate_signal_for_symbol(symbol, condition)
                if signal:
                    signals.append(signal)
            
            logger.info(f"Generated {len(signals)} new signals")
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
        
        return signals
    
    async def _generate_signal_for_symbol(self, symbol: str, condition: MarketCondition) -> Optional[TradingSignal]:
        """Generate signal for a specific symbol"""
        try:
            # Prepare features for signal generation
            features = self._prepare_signal_features(condition)
            
            # Generate signal using AI model
            signal_type, confidence = await self._predict_signal_type(features)
            
            # Check if signal meets minimum criteria
            if confidence < self.min_confidence:
                return None
            
            # Calculate signal strength
            strength = self._calculate_signal_strength(condition, confidence)
            
            if strength < self.min_strength:
                return None
            
            # Generate price targets
            entry_price, target_price, stop_loss, take_profit = self._calculate_price_targets(symbol, condition, signal_type)
            
            # Determine timeframe
            timeframe = self._determine_timeframe(condition, signal_type)
            
            # Generate reasoning
            reasoning = self._generate_signal_reasoning(condition, signal_type, confidence)
            
            # Determine data sources
            data_sources = self._determine_data_sources(condition)
            
            # Create signal
            signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                strength=strength,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timeframe=timeframe,
                reasoning=reasoning,
                data_sources=data_sources,
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(seconds=self.signal_lifetime)
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def _prepare_signal_features(self, condition: MarketCondition) -> List[float]:
        """Prepare features for signal generation"""
        try:
            features = [
                1.0 if condition.trend == "bullish" else (-1.0 if condition.trend == "bearish" else 0.0),
                condition.volatility,
                condition.volume / 1000000,  # Normalize volume
                condition.momentum,
                1.0 if condition.support_level else 0.0,
                1.0 if condition.resistance_level else 0.0
            ]
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing signal features: {e}")
            return [0.0] * 6
    
    async def _predict_signal_type(self, features: List[float]) -> Tuple[str, float]:
        """Predict signal type using AI model"""
        try:
            # This would use the trained model
            # For now, use simple logic
            trend_score = features[0]
            volatility = features[1]
            momentum = features[3]
            
            # Simple signal logic
            if trend_score > 0.5 and momentum > 0.02:
                return "strong_buy", 0.8
            elif trend_score > 0.2:
                return "buy", 0.7
            elif trend_score < -0.5 and momentum < -0.02:
                return "strong_sell", 0.8
            elif trend_score < -0.2:
                return "sell", 0.7
            else:
                return "hold", 0.5
                
        except Exception as e:
            logger.error(f"Error predicting signal type: {e}")
            return "hold", 0.5
    
    def _calculate_signal_strength(self, condition: MarketCondition, confidence: float) -> float:
        """Calculate signal strength"""
        try:
            # Base strength on confidence and market conditions
            base_strength = confidence
            
            # Adjust based on volatility
            volatility_factor = 1.0 - condition.volatility  # Lower volatility = higher strength
            
            # Adjust based on volume
            volume_factor = min(condition.volume / 1000000, 1.0)  # Normalize to $1M
            
            # Adjust based on momentum
            momentum_factor = abs(condition.momentum) * 10  # Scale momentum
            
            # Combine factors
            strength = base_strength * 0.5 + volatility_factor * 0.2 + volume_factor * 0.2 + momentum_factor * 0.1
            
            return min(strength, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            return 0.5
    
    def _calculate_price_targets(self, symbol: str, condition: MarketCondition, signal_type: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Calculate price targets for signal"""
        try:
            # Get current price
            current_price = 45000  # This would be fetched from market data
            
            if signal_type in ["buy", "strong_buy"]:
                entry_price = current_price
                target_price = current_price * 1.05  # 5% target
                stop_loss = current_price * 0.97  # 3% stop loss
                take_profit = current_price * 1.08  # 8% take profit
            elif signal_type in ["sell", "strong_sell"]:
                entry_price = current_price
                target_price = current_price * 0.95  # 5% target
                stop_loss = current_price * 1.03  # 3% stop loss
                take_profit = current_price * 0.92  # 8% take profit
            else:
                return None, None, None, None
            
            return entry_price, target_price, stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Error calculating price targets: {e}")
            return None, None, None, None
    
    def _determine_timeframe(self, condition: MarketCondition, signal_type: str) -> str:
        """Determine signal timeframe"""
        try:
            if signal_type in ["strong_buy", "strong_sell"]:
                return "short"
            elif signal_type in ["buy", "sell"]:
                return "medium"
            else:
                return "long"
                
        except Exception as e:
            logger.error(f"Error determining timeframe: {e}")
            return "medium"
    
    def _generate_signal_reasoning(self, condition: MarketCondition, signal_type: str, confidence: float) -> List[str]:
        """Generate reasoning for signal"""
        reasoning = []
        
        try:
            if condition.trend != "sideways":
                reasoning.append(f"Market trend is {condition.trend}")
            
            if condition.momentum > 0.02:
                reasoning.append("Strong positive momentum")
            elif condition.momentum < -0.02:
                reasoning.append("Strong negative momentum")
            
            if condition.volatility < 0.03:
                reasoning.append("Low volatility environment")
            
            if condition.volume > 500000:
                reasoning.append("High trading volume")
            
            reasoning.append(f"Signal confidence: {confidence:.1%}")
            
        except Exception as e:
            logger.error(f"Error generating signal reasoning: {e}")
            reasoning.append("Signal generated based on market analysis")
        
        return reasoning
    
    def _determine_data_sources(self, condition: MarketCondition) -> List[str]:
        """Determine data sources used for signal"""
        sources = ["market_data", "technical_analysis"]
        
        if condition.support_level or condition.resistance_level:
            sources.append("support_resistance")
        
        if condition.volume > 1000000:
            sources.append("volume_analysis")
        
        return sources
    
    async def _cleanup_expired_signals(self):
        """Clean up expired signals"""
        try:
            current_time = datetime.now()
            expired_signals = [s for s in self.active_signals if s.expires_at < current_time]
            
            for signal in expired_signals:
                self.active_signals.remove(signal)
                self.signal_history.append(signal)
            
            if expired_signals:
                logger.info(f"Cleaned up {len(expired_signals)} expired signals")
                
        except Exception as e:
            logger.error(f"Error cleaning up expired signals: {e}")
    
    async def _validate_active_signals(self):
        """Validate active signals"""
        try:
            for signal in self.active_signals:
                # Check if signal is still valid
                if not await self._is_signal_valid(signal):
                    signal.signal_type = "hold"
                    signal.confidence = 0.5
                    signal.strength = 0.3
                    
        except Exception as e:
            logger.error(f"Error validating active signals: {e}")
    
    async def _is_signal_valid(self, signal: TradingSignal) -> bool:
        """Check if a signal is still valid"""
        try:
            # Get current market condition
            condition = self.market_conditions.get(signal.symbol)
            if not condition:
                return False
            
            # Check if market conditions have changed significantly
            if signal.signal_type in ["buy", "strong_buy"] and condition.trend == "bearish":
                return False
            elif signal.signal_type in ["sell", "strong_sell"] and condition.trend == "bullish":
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking signal validity: {e}")
            return False
    
    async def _update_signal_performance(self):
        """Update signal performance metrics"""
        try:
            # Calculate performance metrics
            total_signals = len(self.signal_history)
            if total_signals == 0:
                return
            
            # This would calculate actual performance based on executed trades
            # For now, use mock data
            successful_signals = int(total_signals * 0.6)  # 60% success rate
            
            metrics = SignalMetrics(
                total_signals=total_signals,
                successful_signals=successful_signals,
                win_rate=successful_signals / total_signals,
                avg_profit=0.05,  # 5% average profit
                avg_loss=0.03,    # 3% average loss
                profit_factor=1.67,  # 5% / 3%
                max_drawdown=0.15,   # 15% max drawdown
                sharpe_ratio=1.2,    # 1.2 Sharpe ratio
                timestamp=datetime.now()
            )
            
            self.signal_metrics.append(metrics)
            
        except Exception as e:
            logger.error(f"Error updating signal performance: {e}")
    
    async def _retrain_models(self):
        """Retrain AI models with new data"""
        try:
            # This would retrain models with new historical data
            logger.info("Retraining signal generation models...")
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
    
    async def _optimize_model_parameters(self):
        """Optimize model parameters"""
        try:
            # This would optimize model hyperparameters
            logger.info("Optimizing model parameters...")
            
        except Exception as e:
            logger.error(f"Error optimizing model parameters: {e}")
    
    async def _load_historical_data(self):
        """Load historical signal data"""
        try:
            # Load from database or file
            logger.info("Loading historical signal data...")
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    def get_active_signals(self) -> List[TradingSignal]:
        """Get active trading signals"""
        return self.active_signals
    
    def get_signals_for_symbol(self, symbol: str) -> List[TradingSignal]:
        """Get signals for a specific symbol"""
        return [s for s in self.active_signals if s.symbol == symbol]
    
    def get_signal_history(self, days: int = 30) -> List[TradingSignal]:
        """Get signal history"""
        cutoff_time = datetime.now() - timedelta(days=days)
        return [s for s in self.signal_history if s.timestamp > cutoff_time]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics
    
    async def stop(self):
        """Stop the signal generator"""
        self.is_running = False
        
        if self.generator_task:
            self.generator_task.cancel()
        
        if self.validation_task:
            self.validation_task.cancel()
        
        if self.optimization_task:
            self.optimization_task.cancel()
        
        logger.info("Signal Generator stopped")


def create_signal_generator(config: Dict[str, Any]) -> SignalGenerator:
    """Create a new signal generator instance"""
    return SignalGenerator(config) 