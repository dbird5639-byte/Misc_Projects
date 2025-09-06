"""
Configuration settings for AI Market Maker & Liquidation Monitor
"""

import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from pydantic import BaseModel, Field
from datetime import datetime
import logging

# Base directory
BASE_DIR = Path(__file__).parent.parent


class ExchangeConfig(BaseModel):
    """Exchange configuration"""
    name: str
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    enabled: bool = True
    rate_limit: int = 100  # requests per minute
    timeout: int = 30  # seconds


class LiquidationStrategyConfig(BaseModel):
    """Liquidation strategy configuration"""
    enabled: bool = True
    min_position_size: float = 1_000_000  # $1M minimum
    max_risk_per_trade: float = 0.02  # 2% max risk
    stop_loss_percentage: float = 0.05  # 5% stop loss
    take_profit_percentage: float = 0.10  # 10% take profit
    min_liquidation_probability: float = 0.7  # 70% confidence
    max_leverage: float = 10.0  # Maximum leverage
    position_timeout: int = 3600  # 1 hour max position hold


class MarketMakerStrategyConfig(BaseModel):
    """Market maker strategy configuration"""
    enabled: bool = True
    track_top_traders: int = 500
    min_volume_threshold: float = 1_000_000  # $1M minimum volume
    position_change_threshold: float = 0.1  # 10% position change
    correlation_threshold: float = 0.8  # 80% correlation
    max_positions_per_asset: int = 5
    rebalance_interval: int = 300  # 5 minutes


class AIConfig(BaseModel):
    """AI model configuration"""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    model_name: str = "gpt-4"
    max_tokens: int = 2000
    temperature: float = 0.1
    enable_streaming: bool = True
    cache_responses: bool = True
    max_concurrent_requests: int = 10


class NotificationConfig(BaseModel):
    """Notification configuration"""
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    
    discord_enabled: bool = False
    discord_webhook_url: str = ""
    
    email_enabled: bool = False
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_recipients: List[str] = []
    
    browser_notifications: bool = True
    rate_limit: int = 10  # messages per minute


class DatabaseConfig(BaseModel):
    """Database configuration"""
    type: str = "sqlite"  # sqlite, postgresql, mysql
    host: str = "localhost"
    port: int = 5432
    database: str = "market_maker.db"
    username: str = ""
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20


class WebConfig(BaseModel):
    """Web dashboard configuration"""
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    secret_key: str = "market-maker-secret-key"
    cors_enabled: bool = True
    ssl_enabled: bool = False
    ssl_cert: str = ""
    ssl_key: str = ""


class SystemConfig(BaseModel):
    """System configuration"""
    debug_mode: bool = False
    log_level: str = "INFO"
    max_workers: int = 10
    data_retention_days: int = 30
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    max_memory_usage: float = 0.8  # 80% of available memory
    cpu_threshold: float = 0.9  # 90% CPU usage threshold


class Settings(BaseModel):
    """Main settings class"""
    # Core configurations
    exchanges: Dict[str, ExchangeConfig] = {}
    liquidation_strategy: LiquidationStrategyConfig = LiquidationStrategyConfig()
    market_maker_strategy: MarketMakerStrategyConfig = MarketMakerStrategyConfig()
    ai_config: AIConfig = AIConfig()
    notifications: NotificationConfig = NotificationConfig()
    database: DatabaseConfig = DatabaseConfig()
    web: WebConfig = WebConfig()
    system: SystemConfig = SystemConfig()
    
    # Runtime settings
    start_time: Optional[datetime] = None
    version: str = "1.0.0"
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SettingsManager:
    """Manages application settings"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or str(BASE_DIR / "config" / "settings.json")
        self.settings = self._load_settings()
        self._setup_logging()
    
    def _load_settings(self) -> Settings:
        """Load settings from file or create defaults"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                return Settings(**config_data)
            else:
                # Create default settings
                settings = self._create_default_settings()
                self.save_settings(settings)
                return settings
        except Exception as e:
            print(f"Error loading settings: {e}")
            return self._create_default_settings()
    
    def _create_default_settings(self) -> Settings:
        """Create default settings"""
        return Settings(
            exchanges={
                "hyperliquid": ExchangeConfig(
                    name="hyperliquid",
                    testnet=True,
                    enabled=True
                ),
                "binance": ExchangeConfig(
                    name="binance",
                    testnet=True,
                    enabled=True
                ),
                "bybit": ExchangeConfig(
                    name="bybit",
                    testnet=True,
                    enabled=True
                )
            },
            liquidation_strategy=LiquidationStrategyConfig(),
            market_maker_strategy=MarketMakerStrategyConfig(),
            ai_config=AIConfig(),
            notifications=NotificationConfig(),
            database=DatabaseConfig(),
            web=WebConfig(),
            system=SystemConfig()
        )
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.settings.system.log_level.upper())
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(BASE_DIR / "logs" / "market_maker.log"),
                logging.StreamHandler()
            ]
        )
    
    def save_settings(self, settings: Optional[Settings] = None):
        """Save settings to file"""
        if settings is None:
            settings = self.settings
        
        # Ensure config directory exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            json.dump(settings.dict(), f, indent=2, default=str)
    
    def update_settings(self, updates: Dict[str, Any]):
        """Update settings with new values"""
        for key, value in updates.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
        
        self.save_settings()
    
    def get_exchange_config(self, exchange_name: str) -> Optional[ExchangeConfig]:
        """Get configuration for specific exchange"""
        return self.settings.exchanges.get(exchange_name)
    
    def get_enabled_exchanges(self) -> List[str]:
        """Get list of enabled exchanges"""
        return [
            name for name, config in self.settings.exchanges.items()
            if config.enabled
        ]
    
    def validate_settings(self) -> List[str]:
        """Validate settings and return list of errors"""
        errors = []
        
        # Check required API keys
        for name, config in self.settings.exchanges.items():
            if config.enabled and not config.testnet:
                if not config.api_key or not config.api_secret:
                    errors.append(f"Missing API credentials for {name}")
        
        # Check AI configuration
        if not self.settings.ai_config.openai_api_key and not self.settings.ai_config.anthropic_api_key:
            errors.append("Missing AI API key (OpenAI or Anthropic)")
        
        # Check notification configuration
        if self.settings.notifications.telegram_enabled:
            if not self.settings.notifications.telegram_bot_token or not self.settings.notifications.telegram_chat_id:
                errors.append("Missing Telegram configuration")
        
        if self.settings.notifications.discord_enabled:
            if not self.settings.notifications.discord_webhook_url:
                errors.append("Missing Discord webhook URL")
        
        return errors


# Global settings instance
_settings_manager: Optional[SettingsManager] = None


def get_settings() -> Settings:
    """Get global settings instance"""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager.settings


def get_settings_manager() -> SettingsManager:
    """Get settings manager instance"""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager


def create_default_configs():
    """Create default configuration files"""
    # Create config directory
    config_dir = BASE_DIR / "config"
    config_dir.mkdir(exist_ok=True)
    
    # Create logs directory
    logs_dir = BASE_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Create data directory
    data_dir = BASE_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Initialize settings manager (this will create default settings)
    get_settings_manager()


if __name__ == "__main__":
    # Test settings
    create_default_configs()
    settings = get_settings()
    print("Settings loaded successfully!")
    print(f"Enabled exchanges: {get_settings_manager().get_enabled_exchanges()}")
    
    # Validate settings
    errors = get_settings_manager().validate_settings()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration is valid!") 