"""
Configuration settings for Interactive Brokers Trading Bot
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class IBConfig:
    """Interactive Brokers connection configuration"""
    host: str = "127.0.0.1"
    port: int = 7497  # TWS Paper Trading port (7496 for live)
    client_id: int = 1
    timeout: int = 20
    retry_attempts: int = 3

@dataclass
class TradingConfig:
    """Trading configuration settings"""
    max_position_size: float = 10000.0  # Maximum position size in USD
    max_portfolio_exposure: float = 0.2  # Maximum 20% of portfolio in single position
    stop_loss_percentage: float = 0.05  # 5% stop loss
    take_profit_percentage: float = 0.10  # 10% take profit
    max_drawdown: float = 0.15  # Maximum 15% drawdown
    correlation_limit: float = 0.7  # Maximum correlation between positions

@dataclass
class StrategyConfig:
    """Strategy configuration settings"""
    momentum_lookback: int = 20  # Days for momentum calculation
    mean_reversion_lookback: int = 50  # Days for mean reversion
    rsi_period: int = 14  # RSI calculation period
    rsi_oversold: float = 30.0  # RSI oversold threshold
    rsi_overbought: float = 70.0  # RSI overbought threshold
    volume_threshold: float = 1.5  # Volume spike threshold

@dataclass
class RiskConfig:
    """Risk management configuration"""
    position_sizing_method: str = "kelly"  # kelly, fixed, volatility
    max_positions: int = 10  # Maximum number of concurrent positions
    sector_limits: Optional[Dict[str, float]] = None  # Sector exposure limits
    volatility_lookback: int = 30  # Days for volatility calculation
    var_confidence: float = 0.95  # Value at Risk confidence level

class Settings:
    """Main settings class"""
    
    def __init__(self):
        # Load environment variables
        self.load_environment()
        
        # Initialize configurations
        self.ib = IBConfig()
        self.trading = TradingConfig()
        self.strategy = StrategyConfig()
        self.risk = RiskConfig()
        
        # Set sector limits
        self.risk.sector_limits = {
            "technology": 0.3,
            "healthcare": 0.2,
            "financial": 0.2,
            "consumer": 0.15,
            "energy": 0.1,
            "other": 0.05
        }
        
        # Trading instruments
        self.instruments = {
            "stocks": True,
            "futures": False,
            "options": False  # Recommended to keep False initially
        }
        
        # Market data settings
        self.market_data = {
            "real_time": True,
            "historical_days": 100,
            "update_interval": 1,  # seconds
            "subscriptions": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        }
        
        # Logging configuration
        self.logging = {
            "level": "INFO",
            "file": "trading_bot.log",
            "max_size": "10MB",
            "backup_count": 5
        }
        
        # Performance tracking
        self.performance = {
            "track_metrics": True,
            "save_trades": True,
            "calculate_sharpe": True,
            "benchmark": "SPY"
        }
    
    def load_environment(self):
        """Load settings from environment variables"""
        # IB Connection
        if os.getenv("IB_HOST"):
            self.ib_host = os.getenv("IB_HOST")
        if os.getenv("IB_PORT"):
            port_value = os.getenv("IB_PORT")
            if port_value:
                self.ib_port = int(port_value)
        if os.getenv("IB_CLIENT_ID"):
            client_id_value = os.getenv("IB_CLIENT_ID")
            if client_id_value:
                self.ib_client_id = int(client_id_value)
    
    def get_instrument_config(self, instrument_type: str) -> Dict[str, Any]:
        """Get configuration for specific instrument type"""
        if instrument_type == "stocks":
            return {
                "leverage": 4.0,  # IB offers up to 4x leverage on stocks
                "margin_requirement": 0.25,
                "commission": 0.005,  # $0.005 per share
                "min_tick": 0.01
            }
        elif instrument_type == "futures":
            return {
                "leverage": 20.0,
                "margin_requirement": 0.05,
                "commission": 2.50,  # Per contract
                "min_tick": 0.25
            }
        elif instrument_type == "options":
            return {
                "leverage": 100.0,
                "margin_requirement": 0.10,
                "commission": 0.65,  # Per contract
                "min_tick": 0.01
            }
        else:
            return {}
    
    def validate_settings(self) -> List[str]:
        """Validate configuration settings"""
        errors = []
        
        # Validate IB connection
        if not (1024 <= self.ib.port <= 65535):
            errors.append("Invalid IB port number")
        
        # Validate trading parameters
        if self.trading.max_position_size <= 0:
            errors.append("Max position size must be positive")
        
        if not (0 < self.trading.max_portfolio_exposure <= 1):
            errors.append("Max portfolio exposure must be between 0 and 1")
        
        if not (0 < self.trading.stop_loss_percentage < 1):
            errors.append("Stop loss percentage must be between 0 and 1")
        
        # Validate strategy parameters
        if self.strategy.momentum_lookback <= 0:
            errors.append("Momentum lookback must be positive")
        
        if not (0 < self.strategy.rsi_oversold < self.strategy.rsi_overbought < 100):
            errors.append("Invalid RSI thresholds")
        
        return errors

# Global settings instance
settings = Settings() 