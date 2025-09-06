"""
Configuration settings for AI Ad Exchange Project
"""

import os
from pathlib import Path
from typing import Dict, List

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ADS_DIR = DATA_DIR / "ads"
ANALYTICS_DIR = DATA_DIR / "analytics"
LOGS_DIR = DATA_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, ADS_DIR, ANALYTICS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Settings
API_SETTINGS = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": True,
    "title": "AI Ad Exchange API",
    "version": "1.0.0"
}

# Security Settings
SECURITY_SETTINGS = {
    "secret_key": os.getenv("SECRET_KEY", "your-secret-key-change-this"),
    "algorithm": "HS256",
    "access_token_expire_minutes": 30,
    "hmac_secret": os.getenv("HMAC_SECRET", "your-hmac-secret-change-this")
}

# Database Settings
DATABASE_SETTINGS = {
    "url": os.getenv("DATABASE_URL", "sqlite:///./ad_exchange.db"),
    "echo": False
}

# Redis Settings
REDIS_SETTINGS = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", 6379)),
    "db": int(os.getenv("REDIS_DB", 0))
}

# Ad Exchange Settings
EXCHANGE_SETTINGS = {
    "min_bid": 0.01,  # Minimum bid amount in USD
    "max_bid": 1000.0,  # Maximum bid amount in USD
    "commission_rate": 0.15,  # 15% commission on transactions
    "min_impression_duration": 5,  # Minimum seconds for impression
    "max_ads_per_streamer": 10,  # Maximum ads per streamer
    "ad_rotation_interval": 30,  # Seconds between ad rotations
}

# Publisher Settings
PUBLISHER_SETTINGS = {
    "min_followers": 100,  # Minimum followers for approval
    "min_stream_hours": 10,  # Minimum streaming hours per month
    "payout_threshold": 50.0,  # Minimum payout amount in USD
    "payout_schedule": "weekly",  # weekly, monthly
}

# Advertiser Settings
ADVERTISER_SETTINGS = {
    "min_budget": 100.0,  # Minimum campaign budget in USD
    "max_budget": 100000.0,  # Maximum campaign budget in USD
    "auto_approval_threshold": 1000.0,  # Auto-approve campaigns under this amount
    "verification_required": True,  # Require advertiser verification
}

# AI Model Settings
AI_SETTINGS = {
    "prediction_model": "linear_regression",  # linear_regression, random_forest, neural_network
    "optimization_algorithm": "genetic_algorithm",  # genetic_algorithm, bayesian_optimization
    "training_data_days": 30,  # Days of historical data to use for training
    "prediction_horizon": 7,  # Days to predict into the future
    "confidence_threshold": 0.7,  # Minimum confidence for AI recommendations
}

# Analytics Settings
ANALYTICS_SETTINGS = {
    "track_impressions": True,
    "track_clicks": True,
    "track_conversions": True,
    "track_engagement": True,
    "retention_days": 90,  # Days to retain analytics data
    "real_time_updates": True,
}

# Platform Integration Settings
PLATFORM_SETTINGS = {
    "obs_integration": True,
    "twitch_integration": False,
    "youtube_integration": False,
    "facebook_integration": False,
    "twitter_integration": False,
}

# Logging Settings
LOGGING_SETTINGS = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "ad_exchange.log",
    "max_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
}

# Environment variables
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = ENVIRONMENT == "development" 