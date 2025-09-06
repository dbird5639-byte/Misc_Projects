"""
Configuration settings for Solana Trading Bot 2025
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class APIConfig:
    """API configuration settings"""
    solana_rpc: str
    jupiter_api: str
    birdeye_api: str
    dexscreener_api: str
    telegram_bot_token: str
    discord_webhook: str

@dataclass
class SniperBotConfig:
    """Sniper bot configuration"""
    enabled: bool = True
    max_position_size: float = 0.1
    min_volume: float = 1000
    min_liquidity: float = 5000
    auto_trade: bool = False
    scan_interval: int = 5  # seconds
    max_tokens_per_scan: int = 100

@dataclass
class CopyBotConfig:
    """Copy bot configuration"""
    enabled: bool = True
    follow_list: Optional[List[str]] = None
    copy_percentage: float = 0.5
    max_delay: int = 30  # seconds
    min_trade_size: float = 0.01  # SOL

@dataclass
class RiskManagementConfig:
    """Risk management configuration"""
    max_portfolio_risk: float = 0.05
    stop_loss_percentage: float = 0.1
    take_profit_percentage: float = 0.3
    max_positions: int = 10
    max_correlation: float = 0.7

@dataclass
class NotificationConfig:
    """Notification configuration"""
    telegram_enabled: bool = True
    discord_enabled: bool = True
    email_enabled: bool = False
    browser_notifications: bool = True

class Config:
    """Main configuration class"""
    
    def __init__(self, config_path: str = "config/"):
        self.config_path = Path(config_path)
        self.api_config = self._load_api_config()
        self.sniper_config = self._load_sniper_config()
        self.copy_config = self._load_copy_config()
        self.risk_config = self._load_risk_config()
        self.notification_config = self._load_notification_config()
    
    def _load_api_config(self) -> APIConfig:
        """Load API configuration from file"""
        api_file = self.config_path / "api_keys.json"
        
        if api_file.exists():
            with open(api_file, 'r') as f:
                data = json.load(f)
        else:
            # Default values - should be overridden
            data = {
                "solana_rpc": os.getenv("SOLANA_RPC", "https://api.mainnet-beta.solana.com"),
                "jupiter_api": os.getenv("JUPITER_API", ""),
                "birdeye_api": os.getenv("BIRDEYE_API", ""),
                "dexscreener_api": os.getenv("DEXSCREENER_API", ""),
                "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
                "discord_webhook": os.getenv("DISCORD_WEBHOOK", "")
            }
        
        return APIConfig(**data)
    
    def _load_sniper_config(self) -> SniperBotConfig:
        """Load sniper bot configuration"""
        config_file = self.config_path / "trading_config.json"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                data = json.load(f)
                sniper_data = data.get("sniper_bot", {})
        else:
            sniper_data = {}
        
        return SniperBotConfig(**sniper_data)
    
    def _load_copy_config(self) -> CopyBotConfig:
        """Load copy bot configuration"""
        config_file = self.config_path / "trading_config.json"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                data = json.load(f)
                copy_data = data.get("copy_bot", {})
        else:
            copy_data = {}
        
        return CopyBotConfig(**copy_data)
    
    def _load_risk_config(self) -> RiskManagementConfig:
        """Load risk management configuration"""
        config_file = self.config_path / "trading_config.json"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                data = json.load(f)
                risk_data = data.get("risk_management", {})
        else:
            risk_data = {}
        
        return RiskManagementConfig(**risk_data)
    
    def _load_notification_config(self) -> NotificationConfig:
        """Load notification configuration"""
        config_file = self.config_path / "trading_config.json"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                data = json.load(f)
                notification_data = data.get("notifications", {})
        else:
            notification_data = {}
        
        return NotificationConfig(**notification_data)
    
    def save_config(self):
        """Save current configuration to file"""
        config_data = {
            "sniper_bot": {
                "enabled": self.sniper_config.enabled,
                "max_position_size": self.sniper_config.max_position_size,
                "min_volume": self.sniper_config.min_volume,
                "min_liquidity": self.sniper_config.min_liquidity,
                "auto_trade": self.sniper_config.auto_trade,
                "scan_interval": self.sniper_config.scan_interval,
                "max_tokens_per_scan": self.sniper_config.max_tokens_per_scan
            },
            "copy_bot": {
                "enabled": self.copy_config.enabled,
                "follow_list": self.copy_config.follow_list or [],
                "copy_percentage": self.copy_config.copy_percentage,
                "max_delay": self.copy_config.max_delay,
                "min_trade_size": self.copy_config.min_trade_size
            },
            "risk_management": {
                "max_portfolio_risk": self.risk_config.max_portfolio_risk,
                "stop_loss_percentage": self.risk_config.stop_loss_percentage,
                "take_profit_percentage": self.risk_config.take_profit_percentage,
                "max_positions": self.risk_config.max_positions,
                "max_correlation": self.risk_config.max_correlation
            },
            "notifications": {
                "telegram_enabled": self.notification_config.telegram_enabled,
                "discord_enabled": self.notification_config.discord_enabled,
                "email_enabled": self.notification_config.email_enabled,
                "browser_notifications": self.notification_config.browser_notifications
            }
        }
        
        config_file = self.config_path / "trading_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=4)
    
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Check API keys
        if not self.api_config.solana_rpc:
            errors.append("Solana RPC URL is required")
        
        if not self.api_config.jupiter_api:
            errors.append("Jupiter API key is required")
        
        if not self.api_config.birdeye_api:
            errors.append("BirdEye API key is required")
        
        # Check trading parameters
        if self.sniper_config.max_position_size <= 0 or self.sniper_config.max_position_size > 1:
            errors.append("Max position size must be between 0 and 1")
        
        if self.risk_config.max_portfolio_risk <= 0 or self.risk_config.max_portfolio_risk > 1:
            errors.append("Max portfolio risk must be between 0 and 1")
        
        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True

# Global configuration instance
config = Config() 