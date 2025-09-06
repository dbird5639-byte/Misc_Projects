"""
Configuration settings for Harvard Algorithmic Trading System
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TradingConfig:
    """Main configuration class for trading system"""
    
    # API Configuration
    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
    ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    
    # Data Configuration
    DATA_SOURCE = os.getenv("DATA_SOURCE", "alpaca")  # alpaca, yfinance, ccxt
    DEFAULT_TIMEFRAME = "1D"
    LOOKBACK_PERIOD = 252  # One year of trading days
    
    # Risk Management
    MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", "0.02"))  # 2% per position
    MAX_PORTFOLIO_RISK = float(os.getenv("MAX_PORTFOLIO_RISK", "0.06"))  # 6% total risk
    STOP_LOSS_PERCENTAGE = float(os.getenv("STOP_LOSS_PERCENTAGE", "0.05"))  # 5% stop loss
    TAKE_PROFIT_PERCENTAGE = float(os.getenv("TAKE_PROFIT_PERCENTAGE", "0.10"))  # 10% take profit
    
    # Strategy Parameters
    MOMENTUM_LOOKBACK = 20
    MEAN_REVERSION_LOOKBACK = 50
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    
    # Backtesting
    BACKTEST_START_DATE = "2022-01-01"
    BACKTEST_END_DATE = "2023-12-31"
    INITIAL_CAPITAL = 100000
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = "logs/trading_system.log"
    
    # AI Tools Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///trading_data.db")
    
    @classmethod
    def get_strategy_config(cls, strategy_name: str) -> Dict[str, Any]:
        """Get configuration for specific strategy"""
        configs = {
            "momentum": {
                "lookback_period": cls.MOMENTUM_LOOKBACK,
                "threshold": 0.02,
                "rsi_period": cls.RSI_PERIOD,
                "rsi_overbought": cls.RSI_OVERBOUGHT,
                "rsi_oversold": cls.RSI_OVERSOLD
            },
            "mean_reversion": {
                "lookback_period": cls.MEAN_REVERSION_LOOKBACK,
                "std_dev_threshold": 2.0,
                "rsi_period": cls.RSI_PERIOD,
                "rsi_overbought": cls.RSI_OVERBOUGHT,
                "rsi_oversold": cls.RSI_OVERSOLD
            }
        }
        return configs.get(strategy_name, {})
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        required_keys = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"]
        missing_keys = [key for key in required_keys if not getattr(cls, key)]
        
        if missing_keys:
            print(f"Missing required configuration: {missing_keys}")
            return False
        
        return True

# Global config instance
config = TradingConfig() 