"""
Ensemble Predictor - AI agent that combines multiple models for better predictions
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
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import plotly.graph_objects as go
import plotly.express as px

from ..config.settings import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EnsemblePrediction:
    """Represents an ensemble prediction"""
    symbol: str
    prediction_type: str  # 'price', 'direction', 'volatility', 'liquidation'
    prediction_value: float
    confidence: float
    model_weights: Dict[str, float]
    individual_predictions: Dict[str, float]
    ensemble_method: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ModelPerformance:
    """Represents model performance metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mse: float
    mae: float
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class EnsembleConfig:
    """Represents ensemble configuration"""
    models: List[str]
    weights: Dict[str, float]
    voting_method: str  # 'hard', 'soft', 'weighted'
    update_frequency: int  # seconds
    min_confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class EnsemblePredictor:
    """
    AI agent that combines multiple models for better predictions
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = True
        self.is_running = False
        
        # Settings
        self.prediction_types = config.get("prediction_types", ["price", "direction", "volatility"])
        self.update_interval = config.get("update_interval", 300)  # 5 minutes
        self.model_retrain_interval = config.get("model_retrain_interval", 86400)  # 24 hours
        self.ensemble_method = config.get("ensemble_method", "weighted")
        
        # AI Models
        self.ensemble_models: Dict[str, Any] = {}
        self.individual_models: Dict[str, Dict[str, Any]] = {}
        self.scaler: Optional[StandardScaler] = None
        
        # Data storage
        self.predictions: List[EnsemblePrediction] = []
        self.model_performances: Dict[str, ModelPerformance] = {}
        self.ensemble_configs: Dict[str, EnsembleConfig] = {}
        self.historical_data: Dict[str, List[Dict[str, Any]]] = {}
        
        # Performance metrics
        self.metrics = {
            "total_predictions": 0,
            "ensemble_accuracy": 0.0,
            "individual_accuracies": {},
            "processing_time": 0.0,
            "last_update": None
        }
        
        # Task management
        self.predictor_task: Optional[asyncio.Task] = None
        self.training_task: Optional[asyncio.Task] = None
        self.optimization_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize the ensemble predictor"""
        try:
            logger.info("Initializing Ensemble Predictor...")
            
            # Initialize AI models
            await self._initialize_models()
            
            # Load historical data
            await self._load_historical_data()
            
            # Train initial models
            await self._train_all_models()
            
            # Start prediction tasks
            self.predictor_task = asyncio.create_task(self._prediction_loop())
            self.training_task = asyncio.create_task(self._training_loop())
            self.optimization_task = asyncio.create_task(self._optimization_loop())
            
            self.is_running = True
            logger.info("Ensemble Predictor initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Ensemble Predictor: {e}")
            return False
    
    async def _initialize_models(self):
        """Initialize AI models"""
        try:
            # Initialize scaler
            self.scaler = StandardScaler()
            
            # Initialize ensemble configurations
            for pred_type in self.prediction_types:
                config = await self._create_ensemble_config(pred_type)
                self.ensemble_configs[pred_type] = config
                
                # Initialize individual models
                self.individual_models[pred_type] = await self._create_individual_models(pred_type)
                
                # Initialize ensemble model
                self.ensemble_models[pred_type] = await self._create_ensemble_model(pred_type, config)
            
            logger.info(f"Initialized ensemble models for {len(self.prediction_types)} prediction types")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    async def _create_ensemble_config(self, prediction_type: str) -> EnsembleConfig:
        """Create ensemble configuration for a prediction type"""
        try:
            if prediction_type == "price":
                models = ["linear_regression", "random_forest", "svr"]
                weights = {"linear_regression": 0.3, "random_forest": 0.4, "svr": 0.3}
            elif prediction_type == "direction":
                models = ["logistic_regression", "random_forest", "svc"]
                weights = {"logistic_regression": 0.3, "random_forest": 0.4, "svc": 0.3}
            elif prediction_type == "volatility":
                models = ["linear_regression", "random_forest"]
                weights = {"linear_regression": 0.4, "random_forest": 0.6}
            else:
                models = ["linear_regression", "random_forest"]
                weights = {"linear_regression": 0.5, "random_forest": 0.5}
            
            return EnsembleConfig(
                models=models,
                weights=weights,
                voting_method="weighted",
                update_frequency=300,
                min_confidence=0.6
            )
            
        except Exception as e:
            logger.error(f"Error creating ensemble config for {prediction_type}: {e}")
            return EnsembleConfig(
                models=["linear_regression"],
                weights={"linear_regression": 1.0},
                voting_method="hard",
                update_frequency=300,
                min_confidence=0.6
            )
    
    async def _create_individual_models(self, prediction_type: str) -> Dict[str, Any]:
        """Create individual models for a prediction type"""
        models = {}
        
        try:
            if prediction_type == "price":
                models["linear_regression"] = LinearRegression()
                models["random_forest"] = RandomForestRegressor(n_estimators=100, random_state=42)
                models["svr"] = SVR(kernel='rbf', C=1.0, gamma='scale')
            elif prediction_type == "direction":
                models["logistic_regression"] = LogisticRegression(random_state=42)
                models["random_forest"] = RandomForestClassifier(n_estimators=100, random_state=42)
                models["svc"] = SVC(kernel='rbf', probability=True, random_state=42)
            elif prediction_type == "volatility":
                models["linear_regression"] = LinearRegression()
                models["random_forest"] = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                models["linear_regression"] = LinearRegression()
                models["random_forest"] = RandomForestRegressor(n_estimators=100, random_state=42)
            
            logger.info(f"Created {len(models)} individual models for {prediction_type}")
            
        except Exception as e:
            logger.error(f"Error creating individual models for {prediction_type}: {e}")
        
        return models
    
    async def _create_ensemble_model(self, prediction_type: str, config: EnsembleConfig) -> Any:
        """Create ensemble model for a prediction type"""
        try:
            individual_models = self.individual_models[prediction_type]
            
            if prediction_type == "direction":
                # Classification ensemble
                estimators = [(name, model) for name, model in individual_models.items() 
                            if name in config.models]
                ensemble = VotingClassifier(
                    estimators=estimators,
                    voting=config.voting_method,
                    weights=[config.weights.get(name, 1.0) for name, _ in estimators]
                )
            else:
                # Regression ensemble
                estimators = [(name, model) for name, model in individual_models.items() 
                            if name in config.models]
                ensemble = VotingRegressor(
                    estimators=estimators,
                    weights=[config.weights.get(name, 1.0) for name, _ in estimators]
                )
            
            return ensemble
            
        except Exception as e:
            logger.error(f"Error creating ensemble model for {prediction_type}: {e}")
            return None
    
    async def _prediction_loop(self):
        """Main prediction loop"""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Generate ensemble predictions
                predictions = await self._generate_ensemble_predictions()
                self.predictions.extend(predictions)
                
                # Update model performances
                await self._update_model_performances()
                
                # Update metrics
                self.metrics["total_predictions"] += len(predictions)
                self.metrics["processing_time"] = time.time() - start_time
                self.metrics["last_update"] = datetime.now()
                
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
                # Retrain all models
                await self._train_all_models()
                
                # Wait for next training cycle
                await asyncio.sleep(self.model_retrain_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in training loop: {e}")
                await asyncio.sleep(3600)
    
    async def _optimization_loop(self):
        """Model optimization loop"""
        while self.is_running:
            try:
                # Optimize ensemble weights
                await self._optimize_ensemble_weights()
                
                # Wait for next optimization cycle (every 6 hours)
                await asyncio.sleep(21600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(3600)
    
    async def _generate_ensemble_predictions(self) -> List[EnsemblePrediction]:
        """Generate ensemble predictions"""
        predictions = []
        
        try:
            symbols = self._get_tracked_symbols()
            
            for symbol in symbols:
                for pred_type in self.prediction_types:
                    prediction = await self._generate_prediction(symbol, pred_type)
                    if prediction:
                        predictions.append(prediction)
            
            logger.info(f"Generated {len(predictions)} ensemble predictions")
            
        except Exception as e:
            logger.error(f"Error generating ensemble predictions: {e}")
        
        return predictions
    
    async def _generate_prediction(self, symbol: str, prediction_type: str) -> Optional[EnsemblePrediction]:
        """Generate prediction for a specific symbol and type"""
        try:
            # Get features for prediction
            features = await self._get_prediction_features(symbol, prediction_type)
            
            if not features:
                return None
            
            # Get individual model predictions
            individual_predictions = {}
            for model_name, model in self.individual_models[prediction_type].items():
                try:
                    if hasattr(model, 'predict_proba'):
                        # Classification model
                        pred_proba = model.predict_proba([features])[0]
                        individual_predictions[model_name] = pred_proba[1]  # Probability of positive class
                    else:
                        # Regression model
                        pred = model.predict([features])[0]
                        individual_predictions[model_name] = pred
                except Exception as e:
                    logger.error(f"Error getting prediction from {model_name}: {e}")
                    individual_predictions[model_name] = 0.0
            
            # Get ensemble prediction
            ensemble_model = self.ensemble_models[prediction_type]
            if ensemble_model:
                try:
                    if hasattr(ensemble_model, 'predict_proba'):
                        # Classification ensemble
                        ensemble_pred_proba = ensemble_model.predict_proba([features])[0]
                        prediction_value = ensemble_pred_proba[1]
                    else:
                        # Regression ensemble
                        prediction_value = ensemble_model.predict([features])[0]
                except Exception as e:
                    logger.error(f"Error getting ensemble prediction: {e}")
                    # Use weighted average of individual predictions
                    config = self.ensemble_configs[prediction_type]
                    weights = [config.weights.get(name, 1.0) for name in individual_predictions.keys()]
                    total_weight = sum(weights)
                    if total_weight > 0:
                        prediction_value = sum(pred * weight for pred, weight in zip(individual_predictions.values(), weights)) / total_weight
                    else:
                        prediction_value = np.mean(list(individual_predictions.values()))
            else:
                prediction_value = np.mean(list(individual_predictions.values()))
            
            # Calculate confidence
            confidence = self._calculate_prediction_confidence(individual_predictions, prediction_type)
            
            # Get model weights
            config = self.ensemble_configs[prediction_type]
            model_weights = {name: config.weights.get(name, 1.0) for name in individual_predictions.keys()}
            
            # Create prediction
            prediction = EnsemblePrediction(
                symbol=symbol,
                prediction_type=prediction_type,
                prediction_value=prediction_value,
                confidence=confidence,
                model_weights=model_weights,
                individual_predictions=individual_predictions,
                ensemble_method=config.voting_method,
                timestamp=datetime.now()
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating prediction for {symbol} {prediction_type}: {e}")
            return None
    
    async def _get_prediction_features(self, symbol: str, prediction_type: str) -> Optional[List[float]]:
        """Get features for prediction"""
        try:
            # This would fetch actual market data
            # For now, return mock features
            features = [
                np.random.normal(0, 1),  # Price change
                np.random.normal(0, 1),  # Volume change
                np.random.normal(0, 1),  # Volatility
                np.random.normal(0, 1),  # Momentum
                np.random.normal(0, 1),  # RSI
                np.random.normal(0, 1),  # MACD
            ]
            
            return features
            
        except Exception as e:
            logger.error(f"Error getting prediction features: {e}")
            return None
    
    def _calculate_prediction_confidence(self, individual_predictions: Dict[str, float], prediction_type: str) -> float:
        """Calculate confidence in ensemble prediction"""
        try:
            if not individual_predictions:
                return 0.0
            
            # Calculate agreement between models
            predictions = list(individual_predictions.values())
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            
            # Higher confidence with lower standard deviation (more agreement)
            agreement_score = 1.0 / (1.0 + std_pred)
            
            # Higher confidence with more models
            model_count_score = min(len(predictions) / 3, 1.0)
            
            # Combine scores
            confidence = (agreement_score * 0.7 + model_count_score * 0.3)
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating prediction confidence: {e}")
            return 0.5
    
    async def _train_all_models(self):
        """Train all models"""
        try:
            logger.info("Training all ensemble models...")
            
            for pred_type in self.prediction_types:
                await self._train_models_for_type(pred_type)
            
            logger.info("All models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training all models: {e}")
    
    async def _train_models_for_type(self, prediction_type: str):
        """Train models for a specific prediction type"""
        try:
            # Get training data
            X_train, y_train = await self._get_training_data(prediction_type)
            
            if len(X_train) < 10:  # Need minimum data
                logger.warning(f"Insufficient training data for {prediction_type}")
                return
            
            # Train individual models
            for model_name, model in self.individual_models[prediction_type].items():
                try:
                    model.fit(X_train, y_train)
                    logger.info(f"Trained {model_name} for {prediction_type}")
                except Exception as e:
                    logger.error(f"Error training {model_name} for {prediction_type}: {e}")
            
            # Train ensemble model
            ensemble_model = self.ensemble_models[prediction_type]
            if ensemble_model:
                try:
                    ensemble_model.fit(X_train, y_train)
                    logger.info(f"Trained ensemble model for {prediction_type}")
                except Exception as e:
                    logger.error(f"Error training ensemble model for {prediction_type}: {e}")
            
        except Exception as e:
            logger.error(f"Error training models for {prediction_type}: {e}")
    
    async def _get_training_data(self, prediction_type: str) -> Tuple[List[List[float]], List[float]]:
        """Get training data for a prediction type"""
        try:
            # This would load actual historical data
            # For now, generate mock data
            n_samples = 1000
            n_features = 6
            
            X_train = [np.random.normal(0, 1, n_features).tolist() for _ in range(n_samples)]
            
            if prediction_type == "direction":
                y_train = np.random.choice([0, 1], n_samples).tolist()
            else:
                y_train = np.random.normal(0, 1, n_samples).tolist()
            
            return X_train, y_train
            
        except Exception as e:
            logger.error(f"Error getting training data for {prediction_type}: {e}")
            return [], []
    
    async def _update_model_performances(self):
        """Update model performance metrics"""
        try:
            for pred_type in self.prediction_types:
                for model_name, model in self.individual_models[pred_type].items():
                    performance = await self._calculate_model_performance(model_name, pred_type)
                    self.model_performances[f"{pred_type}_{model_name}"] = performance
            
        except Exception as e:
            logger.error(f"Error updating model performances: {e}")
    
    async def _calculate_model_performance(self, model_name: str, prediction_type: str) -> ModelPerformance:
        """Calculate performance metrics for a model"""
        try:
            # This would calculate actual performance metrics
            # For now, use mock data
            return ModelPerformance(
                model_name=model_name,
                accuracy=0.7 + np.random.normal(0, 0.1),
                precision=0.65 + np.random.normal(0, 0.1),
                recall=0.7 + np.random.normal(0, 0.1),
                f1_score=0.67 + np.random.normal(0, 0.1),
                mse=0.1 + np.random.normal(0, 0.05),
                mae=0.08 + np.random.normal(0, 0.03),
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance for {model_name}: {e}")
            return ModelPerformance(
                model_name=model_name,
                accuracy=0.5,
                precision=0.5,
                recall=0.5,
                f1_score=0.5,
                mse=0.5,
                mae=0.5,
                last_updated=datetime.now()
            )
    
    async def _optimize_ensemble_weights(self):
        """Optimize ensemble weights based on performance"""
        try:
            logger.info("Optimizing ensemble weights...")
            
            for pred_type in self.prediction_types:
                config = self.ensemble_configs[pred_type]
                
                # Get performance scores for each model
                performances = {}
                for model_name in config.models:
                    perf_key = f"{pred_type}_{model_name}"
                    if perf_key in self.model_performances:
                        performances[model_name] = self.model_performances[perf_key].accuracy
                
                if performances:
                    # Update weights based on performance
                    total_performance = sum(performances.values())
                    if total_performance > 0:
                        new_weights = {name: perf / total_performance for name, perf in performances.items()}
                        config.weights = new_weights
                        
                        # Update ensemble model
                        self.ensemble_models[pred_type] = await self._create_ensemble_model(pred_type, config)
                        
                        logger.info(f"Updated weights for {pred_type}: {new_weights}")
            
        except Exception as e:
            logger.error(f"Error optimizing ensemble weights: {e}")
    
    def _get_tracked_symbols(self) -> List[str]:
        """Get list of tracked symbols"""
        return ["BTC", "ETH", "SOL", "AVAX", "MATIC"]
    
    async def _load_historical_data(self):
        """Load historical data"""
        try:
            # Load from database or file
            logger.info("Loading historical data for ensemble models...")
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    def get_recent_predictions(self, hours: int = 24) -> List[EnsemblePrediction]:
        """Get recent predictions"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [p for p in self.predictions if p.timestamp > cutoff_time]
    
    def get_predictions_for_symbol(self, symbol: str) -> List[EnsemblePrediction]:
        """Get predictions for a specific symbol"""
        return [p for p in self.predictions if p.symbol == symbol]
    
    def get_model_performances(self) -> Dict[str, ModelPerformance]:
        """Get model performance metrics"""
        return self.model_performances
    
    def get_ensemble_configs(self) -> Dict[str, EnsembleConfig]:
        """Get ensemble configurations"""
        return self.ensemble_configs
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics
    
    async def stop(self):
        """Stop the ensemble predictor"""
        self.is_running = False
        
        if self.predictor_task:
            self.predictor_task.cancel()
        
        if self.training_task:
            self.training_task.cancel()
        
        if self.optimization_task:
            self.optimization_task.cancel()
        
        logger.info("Ensemble Predictor stopped")


def create_ensemble_predictor(config: Dict[str, Any]) -> EnsemblePredictor:
    """Create a new ensemble predictor instance"""
    return EnsemblePredictor(config) 