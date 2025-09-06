"""
Machine Learning Model Factory
Build complex, evolving machine learning models from many small predictive signals

Based on Jim Simons' approach of building sophisticated models that combine
many small edges rather than relying on a single "magic formula".
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib
import json

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of machine learning models"""
    LINEAR = "linear"
    RIDGE = "ridge"
    LASSO = "lasso"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    SVM = "svm"
    NEURAL_NET = "neural_net"
    ENSEMBLE = "ensemble"


class FeatureType(Enum):
    """Types of features"""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    MICROSTRUCTURE = "microstructure"
    SENTIMENT = "sentiment"
    MACRO = "macro"
    CROSS_ASSET = "cross_asset"


@dataclass
class ModelConfig:
    """Model configuration"""
    model_type: ModelType
    hyperparameters: Dict[str, Any]
    feature_types: List[FeatureType]
    target_variable: str
    prediction_horizon: int
    validation_method: str
    ensemble_method: Optional[str] = None


@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    train_score: float
    validation_score: float
    test_score: float
    mse: float
    mae: float
    r2: float
    sharpe_ratio: float
    max_drawdown: float
    feature_importance: Dict[str, float]
    prediction_errors: List[float]
    training_time: float


@dataclass
class Model:
    """Machine learning model"""
    name: str
    model_type: ModelType
    model_object: Any
    config: ModelConfig
    scaler: Optional[Any] = None
    feature_names: List[str]
    performance: Optional[ModelPerformance] = None
    created_at: datetime
    last_updated: datetime


class ModelFactory:
    """
    Machine learning model factory
    
    Implements Jim Simons' approach of building complex, evolving models
    from many small predictive signals rather than searching for a single
    magic formula.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the model factory"""
        self.config = config or self._default_config()
        self.models = {}
        self.model_templates = self._create_model_templates()
        
        logger.info("Model Factory initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            "default_validation": "time_series_split",
            "n_splits": 5,
            "test_size": 0.2,
            "random_state": 42,
            "n_jobs": -1,
            "early_stopping": True,
            "feature_selection": True,
            "ensemble_methods": ["voting", "stacking", "blending"],
            "performance_metrics": ["mse", "mae", "r2", "sharpe", "drawdown"]
        }
    
    def _create_model_templates(self) -> Dict[ModelType, Dict]:
        """Create model templates with default hyperparameters"""
        return {
            ModelType.LINEAR: {
                "model_class": LinearRegression,
                "hyperparameters": {}
            },
            ModelType.RIDGE: {
                "model_class": Ridge,
                "hyperparameters": {
                    "alpha": 1.0,
                    "random_state": self.config["random_state"]
                }
            },
            ModelType.LASSO: {
                "model_class": Lasso,
                "hyperparameters": {
                    "alpha": 1.0,
                    "random_state": self.config["random_state"]
                }
            },
            ModelType.RANDOM_FOREST: {
                "model_class": RandomForestRegressor,
                "hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "random_state": self.config["random_state"],
                    "n_jobs": self.config["n_jobs"]
                }
            },
            ModelType.GRADIENT_BOOSTING: {
                "model_class": GradientBoostingRegressor,
                "hyperparameters": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "random_state": self.config["random_state"]
                }
            },
            ModelType.XGBOOST: {
                "model_class": xgb.XGBRegressor,
                "hyperparameters": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": self.config["random_state"],
                    "n_jobs": self.config["n_jobs"]
                }
            },
            ModelType.LIGHTGBM: {
                "model_class": lgb.LGBMRegressor,
                "hyperparameters": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": self.config["random_state"],
                    "n_jobs": self.config["n_jobs"]
                }
            },
            ModelType.CATBOOST: {
                "model_class": CatBoostRegressor,
                "hyperparameters": {
                    "iterations": 100,
                    "learning_rate": 0.1,
                    "depth": 6,
                    "random_state": self.config["random_state"],
                    "verbose": False
                }
            },
            ModelType.SVM: {
                "model_class": SVR,
                "hyperparameters": {
                    "kernel": "rbf",
                    "C": 1.0,
                    "epsilon": 0.1
                }
            },
            ModelType.NEURAL_NET: {
                "model_class": MLPRegressor,
                "hyperparameters": {
                    "hidden_layer_sizes": (100, 50),
                    "activation": "relu",
                    "solver": "adam",
                    "alpha": 0.0001,
                    "learning_rate": "adaptive",
                    "max_iter": 500,
                    "random_state": self.config["random_state"]
                }
            }
        }
    
    def create_model(
        self,
        model_type: Union[ModelType, str],
        features: pd.DataFrame,
        target: pd.Series,
        config: Optional[ModelConfig] = None,
        name: Optional[str] = None
    ) -> Model:
        """
        Create a machine learning model
        
        Args:
            model_type: Type of model to create
            features: Feature matrix
            target: Target variable
            config: Model configuration
            name: Model name
            
        Returns:
            Trained model
        """
        if isinstance(model_type, str):
            model_type = ModelType(model_type)
        
        if name is None:
            name = f"{model_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Creating {model_type.value} model: {name}")
        
        # Create default config if not provided
        if config is None:
            config = ModelConfig(
                model_type=model_type,
                hyperparameters=self.model_templates[model_type]["hyperparameters"],
                feature_types=[FeatureType.TECHNICAL],
                target_variable=target.name if hasattr(target, 'name') else 'target',
                prediction_horizon=1,
                validation_method=self.config["default_validation"]
            )
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self._prepare_data(
            features, target, config
        )
        
        # Create and train model
        model_object, scaler = self._create_and_train_model(
            model_type, X_train, y_train, config
        )
        
        # Evaluate model
        performance = self._evaluate_model(
            model_object, scaler, X_train, X_val, X_test, y_train, y_val, y_test, name
        )
        
        # Create model object
        model = Model(
            name=name,
            model_type=model_type,
            model_object=model_object,
            config=config,
            scaler=scaler,
            feature_names=features.columns.tolist(),
            performance=performance,
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        # Store model
        self.models[name] = model
        
        return model
    
    def create_ensemble_model(
        self,
        base_models: List[Union[ModelType, str]],
        features: pd.DataFrame,
        target: pd.Series,
        ensemble_method: str = "voting",
        name: Optional[str] = None
    ) -> Model:
        """
        Create an ensemble model combining multiple base models
        
        Args:
            base_models: List of base model types
            features: Feature matrix
            target: Target variable
            ensemble_method: Ensemble method (voting, stacking, blending)
            name: Model name
            
        Returns:
            Ensemble model
        """
        if name is None:
            name = f"ensemble_{ensemble_method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Creating ensemble model: {name}")
        
        # Create base models
        base_model_objects = []
        for i, model_type in enumerate(base_models):
            base_name = f"{name}_base_{i}"
            base_model = self.create_model(model_type, features, target, name=base_name)
            base_model_objects.append(base_model)
        
        # Create ensemble
        if ensemble_method == "voting":
            ensemble_model = self._create_voting_ensemble(base_model_objects)
        elif ensemble_method == "stacking":
            ensemble_model = self._create_stacking_ensemble(base_model_objects, features, target)
        elif ensemble_method == "blending":
            ensemble_model = self._create_blending_ensemble(base_model_objects, features, target)
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")
        
        # Create ensemble config
        config = ModelConfig(
            model_type=ModelType.ENSEMBLE,
            hyperparameters={"ensemble_method": ensemble_method},
            feature_types=[FeatureType.TECHNICAL],
            target_variable=target.name if hasattr(target, 'name') else 'target',
            prediction_horizon=1,
            validation_method=self.config["default_validation"],
            ensemble_method=ensemble_method
        )
        
        # Evaluate ensemble
        X_train, X_val, X_test, y_train, y_val, y_test = self._prepare_data(
            features, target, config
        )
        
        performance = self._evaluate_ensemble(
            ensemble_model, X_train, X_val, X_test, y_train, y_val, y_test, name
        )
        
        # Create ensemble model object
        ensemble = Model(
            name=name,
            model_type=ModelType.ENSEMBLE,
            model_object=ensemble_model,
            config=config,
            scaler=None,  # Ensemble handles scaling internally
            feature_names=features.columns.tolist(),
            performance=performance,
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        self.models[name] = ensemble
        return ensemble
    
    def _prepare_data(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        config: ModelConfig
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training, validation, and testing"""
        
        # Align features and target
        aligned_data = pd.concat([features, target], axis=1).dropna()
        features_aligned = aligned_data.iloc[:, :-1]
        target_aligned = aligned_data.iloc[:, -1]
        
        # Split data
        if config.validation_method == "time_series_split":
            # Time series split
            n_samples = len(features_aligned)
            train_size = int(n_samples * 0.6)
            val_size = int(n_samples * 0.2)
            
            X_train = features_aligned.iloc[:train_size]
            y_train = target_aligned.iloc[:train_size]
            
            X_val = features_aligned.iloc[train_size:train_size + val_size]
            y_val = target_aligned.iloc[train_size:train_size + val_size]
            
            X_test = features_aligned.iloc[train_size + val_size:]
            y_test = target_aligned.iloc[train_size + val_size:]
        else:
            # Random split
            from sklearn.model_selection import train_test_split
            
            X_temp, X_test, y_temp, y_test = train_test_split(
                features_aligned, target_aligned,
                test_size=self.config["test_size"],
                random_state=self.config["random_state"]
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=0.25,
                random_state=self.config["random_state"]
            )
        
        return X_train.values, X_val.values, X_test.values, y_train.values, y_val.values, y_test.values
    
    def _create_and_train_model(
        self,
        model_type: ModelType,
        X_train: np.ndarray,
        y_train: np.ndarray,
        config: ModelConfig
    ) -> Tuple[Any, Optional[Any]]:
        """Create and train a model"""
        
        template = self.model_templates[model_type]
        model_class = template["model_class"]
        hyperparameters = {**template["hyperparameters"], **config.hyperparameters}
        
        # Create model
        model = model_class(**hyperparameters)
        
        # Scale features for certain models
        scaler = None
        if model_type in [ModelType.SVM, ModelType.NEURAL_NET, ModelType.RIDGE, ModelType.LASSO]:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
        else:
            X_train_scaled = X_train
        
        # Train model
        start_time = datetime.now()
        model.fit(X_train_scaled, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Model trained in {training_time:.2f} seconds")
        
        return model, scaler
    
    def _evaluate_model(
        self,
        model: Any,
        scaler: Optional[Any],
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        model_name: str
    ) -> ModelPerformance:
        """Evaluate model performance"""
        
        # Make predictions
        if scaler is not None:
            X_train_scaled = scaler.transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
            X_test_scaled = X_test
        
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_score = r2_score(y_train, y_train_pred)
        val_score = r2_score(y_val, y_val_pred)
        test_score = r2_score(y_test, y_test_pred)
        
        mse = mean_squared_error(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)
        
        # Calculate financial metrics
        returns = pd.Series(y_test_pred) - pd.Series(y_test)
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Calculate max drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Feature importance
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(range(len(model.feature_importances_)), model.feature_importances_))
        elif hasattr(model, 'coef_'):
            feature_importance = dict(zip(range(len(model.coef_)), np.abs(model.coef_)))
        
        # Prediction errors
        prediction_errors = (y_test_pred - y_test).tolist()
        
        return ModelPerformance(
            model_name=model_name,
            train_score=train_score,
            validation_score=val_score,
            test_score=test_score,
            mse=mse,
            mae=mae,
            r2=r2,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            feature_importance=feature_importance,
            prediction_errors=prediction_errors,
            training_time=0.0  # Will be set by caller
        )
    
    def _create_voting_ensemble(self, base_models: List[Model]) -> Any:
        """Create voting ensemble"""
        from sklearn.ensemble import VotingRegressor
        
        estimators = [(model.name, model.model_object) for model in base_models]
        ensemble = VotingRegressor(estimators=estimators, n_jobs=self.config["n_jobs"])
        
        return ensemble
    
    def _create_stacking_ensemble(
        self,
        base_models: List[Model],
        features: pd.DataFrame,
        target: pd.Series
    ) -> Any:
        """Create stacking ensemble"""
        from sklearn.ensemble import StackingRegressor
        from sklearn.linear_model import LinearRegression
        
        estimators = [(model.name, model.model_object) for model in base_models]
        ensemble = StackingRegressor(
            estimators=estimators,
            final_estimator=LinearRegression(),
            n_jobs=self.config["n_jobs"]
        )
        
        return ensemble
    
    def _create_blending_ensemble(
        self,
        base_models: List[Model],
        features: pd.DataFrame,
        target: pd.Series
    ) -> Any:
        """Create blending ensemble"""
        
        class BlendingEnsemble:
            def __init__(self, base_models, weights=None):
                self.base_models = base_models
                self.weights = weights or [1/len(base_models)] * len(base_models)
                self.meta_model = LinearRegression()
            
            def fit(self, X, y):
                # Get base predictions
                base_predictions = []
                for model in self.base_models:
                    pred = model.model_object.predict(X)
                    base_predictions.append(pred)
                
                # Train meta-model
                meta_features = np.column_stack(base_predictions)
                self.meta_model.fit(meta_features, y)
                return self
            
            def predict(self, X):
                # Get base predictions
                base_predictions = []
                for model in self.base_models:
                    pred = model.model_object.predict(X)
                    base_predictions.append(pred)
                
                # Meta-prediction
                meta_features = np.column_stack(base_predictions)
                return self.meta_model.predict(meta_features)
        
        return BlendingEnsemble(base_models)
    
    def _evaluate_ensemble(
        self,
        ensemble: Any,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        model_name: str
    ) -> ModelPerformance:
        """Evaluate ensemble model performance"""
        
        # Train ensemble
        start_time = datetime.now()
        ensemble.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Make predictions
        y_train_pred = ensemble.predict(X_train)
        y_val_pred = ensemble.predict(X_val)
        y_test_pred = ensemble.predict(X_test)
        
        # Calculate metrics
        train_score = r2_score(y_train, y_train_pred)
        val_score = r2_score(y_val, y_val_pred)
        test_score = r2_score(y_test, y_test_pred)
        
        mse = mean_squared_error(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)
        
        # Calculate financial metrics
        returns = pd.Series(y_test_pred) - pd.Series(y_test)
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Calculate max drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Feature importance (not applicable for ensemble)
        feature_importance = {}
        
        # Prediction errors
        prediction_errors = (y_test_pred - y_test).tolist()
        
        return ModelPerformance(
            model_name=model_name,
            train_score=train_score,
            validation_score=val_score,
            test_score=test_score,
            mse=mse,
            mae=mae,
            r2=r2,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            feature_importance=feature_importance,
            prediction_errors=prediction_errors,
            training_time=training_time
        )
    
    def get_model(self, name: str) -> Optional[Model]:
        """Get a model by name"""
        return self.models.get(name)
    
    def list_models(self) -> List[str]:
        """List all model names"""
        return list(self.models.keys())
    
    def compare_models(self, model_names: List[str]) -> pd.DataFrame:
        """Compare multiple models"""
        comparison_data = []
        
        for name in model_names:
            model = self.get_model(name)
            if model and model.performance:
                comparison_data.append({
                    'Model': name,
                    'Type': model.model_type.value,
                    'Train Score': model.performance.train_score,
                    'Validation Score': model.performance.validation_score,
                    'Test Score': model.performance.test_score,
                    'MSE': model.performance.mse,
                    'MAE': model.performance.mae,
                    'R²': model.performance.r2,
                    'Sharpe Ratio': model.performance.sharpe_ratio,
                    'Max Drawdown': model.performance.max_drawdown,
                    'Training Time': model.performance.training_time
                })
        
        return pd.DataFrame(comparison_data)
    
    def save_model(self, name: str, filepath: str):
        """Save model to file"""
        model = self.get_model(name)
        if model:
            joblib.dump(model, filepath)
            logger.info(f"Model {name} saved to {filepath}")
    
    def load_model(self, filepath: str) -> Model:
        """Load model from file"""
        model = joblib.load(filepath)
        self.models[model.name] = model
        logger.info(f"Model {model.name} loaded from {filepath}")
        return model
    
    def generate_report(self, model_names: Optional[List[str]] = None) -> str:
        """Generate comprehensive model report"""
        
        if model_names is None:
            model_names = self.list_models()
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Factory Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background: #e8f4f8; border-radius: 3px; }
                .model { margin: 10px 0; padding: 10px; background: #f9f9f9; border-radius: 3px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Model Factory Report</h1>
                <p>Comprehensive analysis of machine learning models</p>
            </div>
        """
        
        # Model comparison
        comparison_df = self.compare_models(model_names)
        if not comparison_df.empty:
            html += '<div class="section"><h2>Model Comparison</h2>'
            html += comparison_df.to_html()
            html += '</div>'
        
        # Individual model details
        for name in model_names:
            model = self.get_model(name)
            if model and model.performance:
                html += f'<div class="section"><h2>{name}</h2>'
                html += f'<div class="model">'
                html += f'<p><strong>Type:</strong> {model.model_type.value}</p>'
                html += f'<p><strong>Created:</strong> {model.created_at}</p>'
                html += f'<p><strong>Features:</strong> {len(model.feature_names)}</p>'
                
                if model.performance:
                    html += f'<div class="metric"><strong>Test Score:</strong> {model.performance.test_score:.3f}</div>'
                    html += f'<div class="metric"><strong>Sharpe Ratio:</strong> {model.performance.sharpe_ratio:.3f}</div>'
                    html += f'<div class="metric"><strong>Max Drawdown:</strong> {model.performance.max_drawdown:.3f}</div>'
                    html += f'<div class="metric"><strong>MSE:</strong> {model.performance.mse:.6f}</div>'
                
                html += '</div></div>'
        
        html += """
        </body>
        </html>
        """
        
        return html


# Example usage
if __name__ == "__main__":
    # Initialize model factory
    factory = ModelFactory()
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Generate features
    features = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Generate target (non-linear relationship)
    target = pd.Series(
        0.1 * features['feature_0'] + 
        0.2 * features['feature_1']**2 + 
        0.3 * features['feature_2'] * features['feature_3'] +
        np.random.randn(n_samples) * 0.1,
        name='target'
    )
    
    # Create different models
    models = []
    
    # Linear model
    linear_model = factory.create_model(
        ModelType.LINEAR, features, target, name="linear_model"
    )
    models.append(linear_model)
    
    # Random Forest
    rf_model = factory.create_model(
        ModelType.RANDOM_FOREST, features, target, name="random_forest_model"
    )
    models.append(rf_model)
    
    # XGBoost
    xgb_model = factory.create_model(
        ModelType.XGBOOST, features, target, name="xgboost_model"
    )
    models.append(xgb_model)
    
    # Neural Network
    nn_model = factory.create_model(
        ModelType.NEURAL_NET, features, target, name="neural_net_model"
    )
    models.append(nn_model)
    
    # Create ensemble
    ensemble_model = factory.create_ensemble_model(
        [ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.NEURAL_NET],
        features, target, ensemble_method="voting", name="ensemble_model"
    )
    models.append(ensemble_model)
    
    # Compare models
    comparison = factory.compare_models([m.name for m in models])
    print("Model Comparison:")
    print(comparison)
    
    # Show best model
    best_model = comparison.loc[comparison['Test Score'].idxmax()]
    print(f"\nBest Model: {best_model['Model']}")
    print(f"Test Score: {best_model['Test Score']:.3f}")
    print(f"Sharpe Ratio: {best_model['Sharpe Ratio']:.3f}")
    
    # Generate report
    report = factory.generate_report()
    with open("model_factory_report.html", "w") as f:
        f.write(report)
    print("\nReport saved to model_factory_report.html") 