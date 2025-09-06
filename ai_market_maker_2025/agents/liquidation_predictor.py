"""
Liquidation Predictor - AI agent for predicting liquidation events
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

from ..config.settings import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LiquidationSignal:
    """Represents a liquidation prediction signal"""
    symbol: str
    side: str  # 'long' or 'short'
    probability: float
    confidence: float
    estimated_time: datetime
    factors: List[str]
    risk_level: str  # 'low', 'medium', 'high'
    recommended_action: str
    stop_loss: Optional[float]
    take_profit: Optional[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class MarketCondition:
    """Represents market conditions for analysis"""
    symbol: str
    price: float
    volume_24h: float
    volatility: float
    funding_rate: float
    open_interest: float
    long_short_ratio: float
    liquidation_ratio: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class LiquidationPredictor:
    """
    AI agent for predicting liquidation events
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = True
        self.is_running = False
        
        # Settings
        self.min_probability = config.get("min_probability", 0.3)
        self.prediction_horizon = config.get("prediction_horizon", 3600)  # 1 hour
        self.update_interval = config.get("update_interval", 300)  # 5 minutes
        self.model_retrain_interval = config.get("model_retrain_interval", 86400)  # 24 hours
        
        # AI Model
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_columns: List[str] = []
        self.is_trained = False
        
        # Data storage
        self.historical_data: List[Dict[str, Any]] = []
        self.predictions: List[LiquidationSignal] = []
        self.market_conditions: Dict[str, MarketCondition] = {}
        
        # Performance metrics
        self.metrics = {
            "total_predictions": 0,
            "correct_predictions": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "last_training": None,
            "model_version": 1
        }
        
        # Task management
        self.predictor_task: Optional[asyncio.Task] = None
        self.training_task: Optional[asyncio.Task] = None
        self.analysis_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize the liquidation predictor"""
        try:
            logger.info("Initializing Liquidation Predictor...")
            
            # Initialize AI model
            await self._initialize_model()
            
            # Load historical data
            await self._load_historical_data()
            
            # Train initial model
            await self._train_model()
            
            # Start prediction tasks
            self.predictor_task = asyncio.create_task(self._prediction_loop())
            self.training_task = asyncio.create_task(self._training_loop())
            self.analysis_task = asyncio.create_task(self._analysis_loop())
            
            self.is_running = True
            logger.info("Liquidation Predictor initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Liquidation Predictor: {e}")
            return False
    
    async def _initialize_model(self):
        """Initialize the AI model"""
        try:
            # Initialize Random Forest classifier
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Initialize scaler
            self.scaler = StandardScaler()
            
            # Define feature columns
            self.feature_columns = [
                'price_change_1h',
                'price_change_24h',
                'volume_change_1h',
                'volume_change_24h',
                'volatility',
                'funding_rate',
                'open_interest_change',
                'long_short_ratio',
                'liquidation_ratio',
                'market_cap',
                'hour_of_day',
                'day_of_week'
            ]
            
            logger.info("AI model initialized")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
    
    async def _prediction_loop(self):
        """Main prediction loop"""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Update market conditions
                await self._update_market_conditions()
                
                # Generate predictions for all symbols
                await self._generate_predictions()
                
                # Validate previous predictions
                await self._validate_predictions()
                
                # Wait for next prediction cycle
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
                await asyncio.sleep(60)
    
    async def _training_loop(self):
        """Model training loop"""
        while self.is_running:
            try:
                # Retrain model periodically
                await self._train_model()
                
                # Wait for next training cycle
                await asyncio.sleep(self.model_retrain_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in training loop: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retrying
    
    async def _analysis_loop(self):
        """Analysis loop for model performance"""
        while self.is_running:
            try:
                # Analyze model performance
                await self._analyze_performance()
                
                # Update feature importance
                await self._update_feature_importance()
                
                # Wait for next analysis cycle
                await asyncio.sleep(3600)  # 1 hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                await asyncio.sleep(1800)  # 30 minutes
    
    async def _update_market_conditions(self):
        """Update market conditions for all symbols"""
        try:
            symbols = self._get_tracked_symbols()
            
            for symbol in symbols:
                market_data = await self._fetch_market_data(symbol)
                
                if market_data:
                    condition = MarketCondition(
                        symbol=symbol,
                        price=market_data.get("price", 0),
                        volume_24h=market_data.get("volume_24h", 0),
                        volatility=market_data.get("volatility", 0),
                        funding_rate=market_data.get("funding_rate", 0),
                        open_interest=market_data.get("open_interest", 0),
                        long_short_ratio=market_data.get("long_short_ratio", 1.0),
                        liquidation_ratio=market_data.get("liquidation_ratio", 0),
                        timestamp=datetime.now()
                    )
                    
                    self.market_conditions[symbol] = condition
            
            logger.debug(f"Updated market conditions for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Error updating market conditions: {e}")
    
    async def _fetch_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch market data for specific symbol"""
        try:
            # This would fetch real market data from exchanges
            # For now, return simulated data
            return {
                "price": np.random.uniform(100, 50000),
                "volume_24h": np.random.uniform(1000000, 100000000),
                "volatility": np.random.uniform(0.1, 0.5),
                "funding_rate": np.random.uniform(-0.01, 0.01),
                "open_interest": np.random.uniform(1000000, 10000000),
                "long_short_ratio": np.random.uniform(0.5, 2.0),
                "liquidation_ratio": np.random.uniform(0, 0.1)
            }
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None
    
    def _get_tracked_symbols(self) -> List[str]:
        """Get list of tracked symbols"""
        # This would return symbols from configuration or data sources
        return ["BTC", "ETH", "SOL", "BNB", "ADA", "DOT", "LINK", "UNI"]
    
    async def _generate_predictions(self):
        """Generate liquidation predictions for all symbols"""
        try:
            if not self.is_trained:
                logger.warning("Model not trained yet, skipping predictions")
                return
            
            # Clear old predictions
            self.predictions.clear()
            
            # Generate predictions for each symbol
            for symbol in self._get_tracked_symbols():
                if symbol in self.market_conditions:
                    # Predict for both long and short
                    for side in ["long", "short"]:
                        prediction = await self._predict_liquidation(symbol, side)
                        
                        if prediction and prediction.probability >= self.min_probability:
                            self.predictions.append(prediction)
                            
                            logger.info(f"Liquidation prediction: {symbol} {side} {prediction.probability:.1%} probability")
                            
                            # Emit prediction event
                            await self._emit_prediction_event("prediction_generated", prediction.to_dict())
            
            self.metrics["total_predictions"] = len(self.predictions)
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
    
    async def _predict_liquidation(self, symbol: str, side: str) -> Optional[LiquidationSignal]:
        """Predict liquidation for specific symbol and side"""
        try:
            if symbol not in self.market_conditions:
                return None
            
            if not self.model or not self.scaler:
                logger.warning("Model or scaler not initialized")
                return None
            
            condition = self.market_conditions[symbol]
            
            # Prepare features for prediction
            features = self._prepare_features(condition, side)
            
            if features is None:
                return None
            
            # Make prediction
            features_scaled: np.ndarray = np.array(self.scaler.transform([features]))
            probability = self.model.predict_proba(features_scaled)[0][1]  # Probability of liquidation
            
            # Calculate confidence based on model certainty
            confidence = self._calculate_confidence(features_scaled)
            
            # Determine risk level
            risk_level = self._determine_risk_level(probability, confidence)
            
            # Generate recommendations
            recommended_action = self._generate_recommendation(probability, confidence, risk_level)
            
            # Calculate stop loss and take profit
            stop_loss, take_profit = self._calculate_risk_levels(condition, side, probability)
            
            # Create prediction signal
            signal = LiquidationSignal(
                symbol=symbol,
                side=side,
                probability=probability,
                confidence=confidence,
                estimated_time=datetime.now() + timedelta(minutes=30),
                factors=self._identify_factors(features),
                risk_level=risk_level,
                recommended_action=recommended_action,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error predicting liquidation for {symbol} {side}: {e}")
            return None
    
    def _prepare_features(self, condition: MarketCondition, side: str) -> Optional[List[float]]:
        """Prepare features for model prediction"""
        try:
            # Calculate feature values
            features = []
            
            # Price changes (would need historical data)
            features.extend([0.0, 0.0])  # price_change_1h, price_change_24h
            
            # Volume changes (would need historical data)
            features.extend([0.0, 0.0])  # volume_change_1h, volume_change_24h
            
            # Current market conditions
            features.append(condition.volatility)
            features.append(condition.funding_rate)
            features.append(0.0)  # open_interest_change
            
            # Long/short ratio
            if side == "long":
                features.append(condition.long_short_ratio)
            else:
                features.append(1.0 / condition.long_short_ratio)
            
            # Liquidation ratio
            features.append(condition.liquidation_ratio)
            
            # Market cap (placeholder)
            features.append(condition.price * condition.volume_24h / 1000000)
            
            # Time features
            now = datetime.now()
            features.append(now.hour / 24.0)  # hour_of_day
            features.append(now.weekday() / 7.0)  # day_of_week
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None
    
    def _calculate_confidence(self, features_scaled: np.ndarray) -> float:
        """Calculate prediction confidence"""
        try:
            if not self.model:
                return 0.5
            
            # Use model's prediction probabilities to calculate confidence
            probabilities = self.model.predict_proba(features_scaled)[0]
            max_prob = max(probabilities)
            min_prob = min(probabilities)
            
            # Confidence is based on how certain the model is
            confidence = (max_prob - min_prob) * 2  # Scale to 0-1
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _determine_risk_level(self, probability: float, confidence: float) -> str:
        """Determine risk level based on probability and confidence"""
        if probability > 0.7 and confidence > 0.8:
            return "high"
        elif probability > 0.5 and confidence > 0.6:
            return "medium"
        else:
            return "low"
    
    def _generate_recommendation(self, probability: float, confidence: float, risk_level: str) -> str:
        """Generate trading recommendation"""
        if risk_level == "high" and probability > 0.7:
            return "strong_sell" if probability > 0.8 else "sell"
        elif risk_level == "medium" and probability > 0.5:
            return "hold"
        else:
            return "buy"
    
    def _calculate_risk_levels(self, condition: MarketCondition, side: str, probability: float) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels"""
        try:
            current_price = condition.price
            
            # Stop loss based on volatility and probability
            stop_loss_pct = condition.volatility * (1 + probability)
            stop_loss = current_price * (1 - stop_loss_pct) if side == "long" else current_price * (1 + stop_loss_pct)
            
            # Take profit based on risk-reward ratio
            take_profit_pct = stop_loss_pct * 2  # 2:1 risk-reward ratio
            take_profit = current_price * (1 + take_profit_pct) if side == "long" else current_price * (1 - take_profit_pct)
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Error calculating risk levels: {e}")
            return None, None
    
    def _identify_factors(self, features: List[float]) -> List[str]:
        """Identify key factors contributing to prediction"""
        try:
            factors = []
            
            # Check which features are most important
            if self.model and hasattr(self.model, 'feature_importances_') and len(self.model.feature_importances_) == len(features):
                # Get top 3 most important features
                feature_importance = list(enumerate(self.model.feature_importances_))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                top_features = feature_importance[:3]
                
                for idx, importance in top_features:
                    if idx < len(self.feature_columns):
                        factor_name = self.feature_columns[idx]
                        if features[idx] > 0.5:  # High value
                            factors.append(f"high_{factor_name}")
                        elif features[idx] < -0.5:  # Low value
                            factors.append(f"low_{factor_name}")
            
            return factors[:5]  # Limit to 5 factors
            
        except Exception as e:
            logger.error(f"Error identifying factors: {e}")
            return ["market_volatility", "funding_rate"]
    
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
                        self.metrics["correct_predictions"] += 1
                    
                    # Remove expired prediction
                    continue
                
                validated_predictions.append(prediction)
            
            self.predictions = validated_predictions
            
            # Update accuracy metrics
            if self.metrics["total_predictions"] > 0:
                self.metrics["accuracy"] = self.metrics["correct_predictions"] / self.metrics["total_predictions"]
            
        except Exception as e:
            logger.error(f"Error validating predictions: {e}")
    
    async def _check_prediction_occurred(self, prediction: LiquidationSignal) -> bool:
        """Check if a prediction occurred"""
        try:
            # This would check actual liquidation events
            # For now, return random result for demonstration
            return np.random.random() < prediction.probability
            
        except Exception as e:
            logger.error(f"Error checking prediction occurrence: {e}")
            return False
    
    async def _train_model(self):
        """Train the AI model"""
        try:
            if len(self.historical_data) < 100:
                logger.warning("Insufficient data for training")
                return
            
            if not self.model or not self.scaler:
                logger.warning("Model or scaler not initialized")
                return
            
            # Prepare training data
            X, y = self._prepare_training_data()
            
            if len(X) < 50:
                logger.warning("Insufficient training samples")
                return
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            
            # Update metrics
            self.metrics["last_training"] = datetime.now()
            self.metrics["model_version"] += 1
            self.is_trained = True
            
            logger.info(f"Model trained successfully with {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
    
    def _prepare_training_data(self) -> Tuple[List[List[float]], List[int]]:
        """Prepare training data from historical data"""
        try:
            X = []
            y = []
            
            for data_point in self.historical_data:
                features = data_point.get("features", [])
                label = data_point.get("liquidation_occurred", 0)
                
                if len(features) == len(self.feature_columns):
                    X.append(features)
                    y.append(label)
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return [], []
    
    async def _analyze_performance(self):
        """Analyze model performance"""
        try:
            if not self.is_trained:
                return
            
            # Calculate performance metrics
            total = self.metrics["total_predictions"]
            correct = self.metrics["correct_predictions"]
            
            if total > 0:
                accuracy = correct / total
                precision = correct / total if total > 0 else 0
                recall = correct / total if total > 0 else 0
                
                if precision + recall > 0:
                    f1_score = 2 * (precision * recall) / (precision + recall)
                else:
                    f1_score = 0
                
                self.metrics.update({
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score
                })
                
                logger.info(f"Model performance - Accuracy: {accuracy:.2%}, F1: {f1_score:.2%}")
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
    
    async def _update_feature_importance(self):
        """Update feature importance analysis"""
        try:
            if not self.is_trained or not self.model or not hasattr(self.model, 'feature_importances_'):
                return
            
            # Log feature importance
            feature_importance = list(zip(self.feature_columns, self.model.feature_importances_))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            logger.info("Top 5 most important features:")
            for feature, importance in feature_importance[:5]:
                logger.info(f"  {feature}: {importance:.3f}")
            
        except Exception as e:
            logger.error(f"Error updating feature importance: {e}")
    
    async def _load_historical_data(self):
        """Load historical data for training"""
        try:
            # This would load from database or file
            # For now, create some synthetic data
            self.historical_data = []
            
            # Generate synthetic training data
            for i in range(1000):
                features = np.random.random(len(self.feature_columns))
                liquidation_occurred = np.random.random() < 0.3  # 30% liquidation rate
                
                self.historical_data.append({
                    "features": features.tolist(),
                    "liquidation_occurred": 1 if liquidation_occurred else 0,
                    "timestamp": datetime.now() - timedelta(hours=np.random.randint(1, 168))
                })
            
            logger.info(f"Loaded {len(self.historical_data)} historical data points")
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    async def _emit_prediction_event(self, event_type: str, data: Dict[str, Any]):
        """Emit prediction event for other components"""
        # This would typically emit to an event bus or notification system
        logger.info(f"Prediction event: {event_type} - {data}")
    
    def get_active_predictions(self) -> List[LiquidationSignal]:
        """Get active liquidation predictions"""
        current_time = datetime.now()
        return [
            prediction for prediction in self.predictions
            if prediction.estimated_time > current_time
        ]
    
    def get_high_risk_predictions(self) -> List[LiquidationSignal]:
        """Get high-risk liquidation predictions"""
        return [
            prediction for prediction in self.predictions
            if prediction.risk_level == "high" and prediction.probability > 0.7
        ]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.metrics,
            "active_predictions": len(self.get_active_predictions()),
            "high_risk_predictions": len(self.get_high_risk_predictions()),
            "is_trained": self.is_trained
        }
    
    async def stop(self):
        """Stop the liquidation predictor"""
        logger.info("Stopping Liquidation Predictor...")
        self.is_running = False
        
        # Cancel tasks
        for task in [self.predictor_task, self.training_task, self.analysis_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Liquidation Predictor stopped")


def create_liquidation_predictor(config: Dict[str, Any]) -> LiquidationPredictor:
    """Create liquidation predictor instance"""
    return LiquidationPredictor(config) 