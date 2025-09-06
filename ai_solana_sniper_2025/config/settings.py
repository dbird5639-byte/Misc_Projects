"""
Configuration settings for AI-Powered Solana Meme Coin Sniper
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "ai_models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for directory in [CONFIG_DIR, DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)


class AIConfig(BaseModel):
    """AI model configuration"""
    local_models: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    cloud_models: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    agent_config: Dict[str, Any] = Field(default_factory=dict)


class TradingConfig(BaseModel):
    """Trading configuration"""
    sniper: Dict[str, Any] = Field(default_factory=dict)
    risk_management: Dict[str, Any] = Field(default_factory=dict)
    notifications: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class Settings:
    """Main settings class"""
    
    # API Keys
    solana_rpc_url: str = "https://api.mainnet-beta.solana.com"
    jupiter_api_key: Optional[str] = None
    birdeye_api_key: Optional[str] = None
    dexscreener_api_key: Optional[str] = None
    
    # AI Configuration
    ai_config: Optional[AIConfig] = None
    
    # Trading Configuration
    trading_config: Optional[TradingConfig] = None
    
    # System Configuration
    debug_mode: bool = False
    log_level: str = "INFO"
    max_workers: int = 4
    
    def __post_init__(self):
        if self.ai_config is None:
            self.ai_config = self._load_ai_config()
        if self.trading_config is None:
            self.trading_config = self._load_trading_config()
    
    def _load_ai_config(self) -> AIConfig:
        """Load AI configuration from file"""
        config_file = CONFIG_DIR / "ai_config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                return AIConfig(**config_data)
        return AIConfig()
    
    def _load_trading_config(self) -> TradingConfig:
        """Load trading configuration from file"""
        config_file = CONFIG_DIR / "trading_config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                return TradingConfig(**config_data)
        return TradingConfig()
    
    def save_ai_config(self):
        """Save AI configuration to file"""
        if not self.ai_config:
            return
        config_file = CONFIG_DIR / "ai_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.ai_config.dict(), f, indent=2)
    
    def save_trading_config(self):
        """Save trading configuration to file"""
        if not self.trading_config:
            return
        config_file = CONFIG_DIR / "trading_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.trading_config.dict(), f, indent=2)


# Default configuration values
DEFAULT_AI_CONFIG = {
    "local_models": {
        "gemma": {
            "enabled": True,
            "model_path": str(MODELS_DIR / "gemma-2b"),
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.9
        },
        "llama": {
            "enabled": False,
            "model_path": str(MODELS_DIR / "llama-7b"),
            "max_tokens": 4096,
            "temperature": 0.8,
            "top_p": 0.9
        }
    },
    "cloud_models": {
        "openai": {
            "enabled": True,
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": "gpt-4",
            "max_tokens": 1000,
            "temperature": 0.7
        },
        "anthropic": {
            "enabled": False,
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": "claude-3-sonnet",
            "max_tokens": 1000,
            "temperature": 0.7
        }
    },
    "agent_config": {
        "decision_threshold": 0.7,
        "max_analysis_time": 30,
        "parallel_processing": True,
        "ensemble_weighting": {
            "local_models": 0.4,
            "cloud_models": 0.6
        }
    }
}

DEFAULT_TRADING_CONFIG = {
    "sniper": {
        "enabled": True,
        "max_position_size": 0.05,
        "min_volume": 500,
        "min_liquidity": 2000,
        "auto_trade": False,
        "scan_interval": 3,
        "max_slippage": 0.05,
        "gas_limit": 300000
    },
    "risk_management": {
        "max_portfolio_risk": 0.03,
        "stop_loss_percentage": 0.15,
        "take_profit_percentage": 0.5,
        "max_positions": 5,
        "max_daily_loss": 0.1,
        "circuit_breaker_threshold": 0.2
    },
    "notifications": {
        "telegram_enabled": True,
        "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN"),
        "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID"),
        "discord_enabled": True,
        "discord_webhook_url": os.getenv("DISCORD_WEBHOOK_URL"),
        "browser_notifications": True,
        "email_enabled": False,
        "email_smtp_server": "smtp.gmail.com",
        "email_smtp_port": 587,
        "email_username": os.getenv("EMAIL_USERNAME"),
        "email_password": os.getenv("EMAIL_PASSWORD")
    }
}


def create_default_configs():
    """Create default configuration files if they don't exist"""
    
    # Create AI config
    ai_config_file = CONFIG_DIR / "ai_config.json"
    if not ai_config_file.exists():
        with open(ai_config_file, 'w') as f:
            json.dump(DEFAULT_AI_CONFIG, f, indent=2)
        print(f"Created default AI config: {ai_config_file}")
    
    # Create trading config
    trading_config_file = CONFIG_DIR / "trading_config.json"
    if not trading_config_file.exists():
        with open(trading_config_file, 'w') as f:
            json.dump(DEFAULT_TRADING_CONFIG, f, indent=2)
        print(f"Created default trading config: {trading_config_file}")


def get_settings() -> Settings:
    """Get application settings"""
    return Settings()


# Environment variables
def get_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable with fallback"""
    return os.getenv(key, default)


# Logging configuration
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(LOGS_DIR / "ai_sniper.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False
        }
    }
}


if __name__ == "__main__":
    # Create default configs when run directly
    create_default_configs()
    print("Configuration setup complete!") 