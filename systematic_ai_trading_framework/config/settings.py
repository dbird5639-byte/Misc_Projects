"""
Configuration settings for the Systematic AI Trading Framework.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pydantic import BaseSettings, Field


@dataclass
class AIModelConfig:
    """Configuration for AI models."""
    name: str
    type: str = "ollama"  # ollama, openai, anthropic, local
    temperature: float = 0.1
    max_tokens: int = 2048
    model_path: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class StrategyConfig:
    """Configuration for trading strategies."""
    name: str
    enabled: bool = True
    parameters: Dict = field(default_factory=dict)
    risk_management: Dict = field(default_factory=dict)
    symbols: List[str] = field(default_factory=list)
    timeframe: str = "1h"


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_portfolio_risk: float = 0.02  # 2% max portfolio risk
    max_strategy_risk: float = 0.01   # 1% max strategy risk
    max_correlation: float = 0.7      # Max correlation between strategies
    stop_loss_pct: float = 0.05       # 5% stop loss
    take_profit_pct: float = 0.15     # 15% take profit
    max_drawdown: float = 0.20        # 20% max drawdown


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    start_date: str = "2020-01-01"
    end_date: str = "2024-01-01"
    initial_capital: float = 100000.0
    commission: float = 0.001         # 0.1% commission
    slippage: float = 0.0005          # 0.05% slippage
    walk_forward_periods: int = 12
    min_trades: int = 30
    min_sharpe: float = 1.0
    max_drawdown: float = 0.20


@dataclass
class DataConfig:
    """Data configuration."""
    data_sources: List[str] = field(default_factory=lambda: ["yfinance", "alpha_vantage"])
    symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "IWM", "GLD", "TLT"])
    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h", "1d"])
    cache_duration: int = 3600        # 1 hour cache
    max_retries: int = 3
    retry_delay: int = 5


class Settings(BaseSettings):
    """Main settings class for the framework."""
    
    # Framework settings
    framework_name: str = "Systematic AI Trading Framework"
    version: str = "1.0.0"
    log_level: str = "INFO"
    log_file: str = "logs/framework.log"
    
    # AI Configuration
    default_model: str = "deepseek-coder:6.7b"
    ai_models: Dict[str, AIModelConfig] = field(default_factory=dict)
    
    # Research settings
    research_interval: int = 3600     # 1 hour
    research_sources: List[str] = field(default_factory=lambda: [
        "youtube", "twitter", "reddit", "news", "academic"
    ])
    max_research_results: int = 100
    
    # Backtesting settings
    backtest_interval: int = 1800     # 30 minutes
    backtest_config: BacktestConfig = field(default_factory=BacktestConfig)
    
    # Implementation settings
    implementation_interval: int = 300  # 5 minutes
    auto_deploy: bool = True
    deployment_threshold: float = 1.5  # Minimum Sharpe ratio for deployment
    
    # Monitoring settings
    monitoring_interval: int = 60     # 1 minute
    performance_update_interval: int = 300  # 5 minutes
    
    # Risk management
    risk_config: RiskConfig = field(default_factory=RiskConfig)
    
    # Data configuration
    data_config: DataConfig = field(default_factory=DataConfig)
    
    # Strategy configuration
    strategies: Dict[str, StrategyConfig] = field(default_factory=dict)
    
    # Web dashboard
    enable_dashboard: bool = True
    dashboard_port: int = 8080
    dashboard_host: str = "0.0.0.0"
    
    # Notifications
    enable_notifications: bool = True
    notification_channels: List[str] = field(default_factory=lambda: ["email", "slack"])
    
    # Database
    database_url: str = "sqlite:///data/framework.db"
    redis_url: str = "redis://localhost:6379"
    
    # API Keys (load from environment)
    yfinance_api_key: Optional[str] = None
    alpha_vantage_api_key: Optional[str] = Field(None, env="ALPHA_VANTAGE_API_KEY")
    polygon_api_key: Optional[str] = Field(None, env="POLYGON_API_KEY")
    quandl_api_key: Optional[str] = Field(None, env="QUANDL_API_KEY")
    
    # Exchange API keys
    exchange_api_keys: Dict[str, Dict[str, str]] = field(default_factory=dict)
    
    # Notification API keys
    twilio_account_sid: Optional[str] = Field(None, env="TWILIO_ACCOUNT_SID")
    twilio_auth_token: Optional[str] = Field(None, env="TWILIO_AUTH_TOKEN")
    slack_webhook_url: Optional[str] = Field(None, env="SLACK_WEBHOOK_URL")
    telegram_bot_token: Optional[str] = Field(None, env="TELEGRAM_BOT_TOKEN")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        
        # Load configuration from file if provided
        if config_path and os.path.exists(config_path):
            self._load_from_file(config_path)
        
        # Set up default AI models
        self._setup_default_models()
        
        # Set up default strategies
        self._setup_default_strategies()
        
        # Create necessary directories
        self._create_directories()
    
    def _load_from_file(self, config_path: str):
        """Load configuration from JSON file."""
        import json
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Update settings with file data
        for key, value in config_data.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def _setup_default_models(self):
        """Set up default AI model configurations."""
        if not self.ai_models:
            self.ai_models = {
                "deepseek-coder:6.7b": AIModelConfig(
                    name="deepseek-coder:6.7b",
                    type="ollama",
                    temperature=0.1,
                    max_tokens=2048
                ),
                "llama2:7b": AIModelConfig(
                    name="llama2:7b",
                    type="ollama",
                    temperature=0.2,
                    max_tokens=1024
                ),
                "claude-3-sonnet": AIModelConfig(
                    name="claude-3-sonnet",
                    type="anthropic",
                    temperature=0.1,
                    max_tokens=2048
                )
            }
    
    def _setup_default_strategies(self):
        """Set up default strategy configurations."""
        if not self.strategies:
            self.strategies = {
                "momentum_strategy": StrategyConfig(
                    name="momentum_strategy",
                    enabled=True,
                    parameters={
                        "lookback_period": 20,
                        "threshold": 0.02,
                        "rsi_period": 14,
                        "rsi_oversold": 30,
                        "rsi_overbought": 70
                    },
                    risk_management={
                        "max_position_size": 0.1,
                        "stop_loss": 0.05,
                        "take_profit": 0.15
                    },
                    symbols=["SPY", "QQQ", "IWM"],
                    timeframe="1h"
                ),
                "mean_reversion_strategy": StrategyConfig(
                    name="mean_reversion_strategy",
                    enabled=True,
                    parameters={
                        "bb_period": 20,
                        "bb_std": 2,
                        "rsi_period": 14,
                        "rsi_oversold": 30,
                        "rsi_overbought": 70
                    },
                    risk_management={
                        "max_position_size": 0.1,
                        "stop_loss": 0.05,
                        "take_profit": 0.15
                    },
                    symbols=["SPY", "QQQ", "IWM"],
                    timeframe="1h"
                ),
                "regime_detection_strategy": StrategyConfig(
                    name="regime_detection_strategy",
                    enabled=True,
                    parameters={
                        "volatility_period": 20,
                        "correlation_period": 60,
                        "regime_threshold": 0.7
                    },
                    risk_management={
                        "max_position_size": 0.05,
                        "stop_loss": 0.03,
                        "take_profit": 0.10
                    },
                    symbols=["SPY", "QQQ", "IWM", "GLD", "TLT"],
                    timeframe="1d"
                )
            }
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            "logs",
            "data",
            "backtests",
            "strategies",
            "models",
            "reports"
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    def get_model_config(self, model_name: str) -> Optional[AIModelConfig]:
        """Get AI model configuration by name."""
        return self.ai_models.get(model_name)
    
    def get_strategy_config(self, strategy_name: str) -> Optional[StrategyConfig]:
        """Get strategy configuration by name."""
        return self.strategies.get(strategy_name)
    
    def update_strategy_config(self, strategy_name: str, config: StrategyConfig):
        """Update strategy configuration."""
        self.strategies[strategy_name] = config
    
    def get_enabled_strategies(self) -> List[str]:
        """Get list of enabled strategy names."""
        return [
            name for name, config in self.strategies.items()
            if config.enabled
        ]
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        # Check required API keys
        if "alpha_vantage" in self.data_config.data_sources and not self.alpha_vantage_api_key:
            raise ValueError("Alpha Vantage API key required")
        
        if "polygon" in self.data_config.data_sources and not self.polygon_api_key:
            raise ValueError("Polygon API key required")
        
        # Check AI model availability
        if not self.ai_models:
            raise ValueError("No AI models configured")
        
        # Check strategy configuration
        if not self.strategies:
            raise ValueError("No strategies configured")
        
        return True


# Global settings instance
settings = Settings() 