"""
Configuration settings for AI Video Income Generator Project
"""

import os
from typing import Dict, List, Any

# Project Settings
PROJECT_NAME = "AI Video Income Generator"
PROJECT_VERSION = "1.0.0"
PROJECT_DESCRIPTION = "Generate passive income through AI-powered video clipping"

# API Keys and External Services
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Video Processing Settings
VIDEO_CONFIG = {
    "min_clip_duration": 300,  # 5 minutes in seconds
    "max_clip_duration": 7200,  # 2 hours in seconds
    "target_clip_duration": 1800,  # 30 minutes target
    "quality_settings": {
        "resolution": "1080p",
        "fps": 30,
        "bitrate": "5000k",
        "audio_bitrate": "128k"
    },
    "supported_formats": ["mp4", "avi", "mov", "mkv"],
    "output_format": "mp4"
}

# AI Analysis Settings
AI_CONFIG = {
    "content_analysis": {
        "min_engagement_score": 0.7,
        "key_topics": [
            "programming", "coding", "development",
            "business", "entrepreneurship", "marketing",
            "technology", "AI", "machine learning",
            "productivity", "tips", "tutorials"
        ],
        "sentiment_threshold": 0.6
    },
    "segment_detection": {
        "min_segment_length": 60,  # 1 minute minimum
        "max_segment_length": 7200,  # 2 hours maximum
        "overlap_threshold": 0.1,  # 10% overlap allowed
        "quality_threshold": 0.8
    },
    "title_generation": {
        "max_title_length": 100,
        "include_keywords": True,
        "seo_optimized": True,
        "language": "en"
    }
}

# YouTube Upload Settings
YOUTUBE_CONFIG = {
    "upload_settings": {
        "privacy": "public",  # public, unlisted, private
        "category": "Education",  # YouTube category
        "default_language": "en",
        "auto_chapters": True,
        "auto_cards": True
    },
    "metadata_template": {
        "description_template": """
{title}

🔗 Original Video: {original_url}
👨‍💻 Creator: {creator_name}

{description}

#programming #coding #development #tutorial #tips
        """,
        "tags_template": [
            "programming", "coding", "development", "tutorial",
            "tips", "technology", "software", "learning"
        ]
    },
    "upload_limits": {
        "daily_uploads": 10,
        "hourly_uploads": 2,
        "file_size_limit": "128GB"
    }
}

# Payment and Revenue Settings
PAYMENT_CONFIG = {
    "payout_thresholds": {
        "10000": 69,  # $69 for 10,000 views
        "25000": 150,  # $150 for 25,000 views
        "50000": 300,  # $300 for 50,000 views
        "100000": 600  # $600 for 100,000 views
    },
    "payment_methods": ["crypto", "paypal", "bank_transfer"],
    "crypto_settings": {
        "preferred_currency": "USDT",
        "supported_currencies": ["BTC", "ETH", "USDT", "USDC"],
        "minimum_payout": 50
    },
    "tracking": {
        "view_update_interval": 3600,  # 1 hour
        "payout_check_interval": 86400,  # 24 hours
        "reporting_frequency": "weekly"
    }
}

# Creator Program Settings
CREATOR_CONFIG = {
    "supported_creators": [
        {
            "name": "Moon Dev",
            "channel_id": "UCexample",
            "program_active": True,
            "guidelines": [
                "No YouTube Shorts",
                "Full-length videos only",
                "Include original video link",
                "Follow community guidelines"
            ],
            "contact_info": {
                "discord": "moon_dev_community",
                "email": "contact@moondev.com"
            }
        }
    ],
    "content_guidelines": {
        "allowed_content": [
            "Educational tutorials",
            "Programming tips",
            "Development insights",
            "Technology discussions"
        ],
        "prohibited_content": [
            "Copyright violations",
            "Low-quality content",
            "Spam or clickbait",
            "Inappropriate material"
        ]
    }
}

# File Paths
DATA_DIR = "data"
SOURCE_VIDEOS_DIR = os.path.join(DATA_DIR, "source_videos")
GENERATED_CLIPS_DIR = os.path.join(DATA_DIR, "generated_clips")
ANALYTICS_DIR = os.path.join(DATA_DIR, "analytics")
PAYMENTS_DIR = os.path.join(DATA_DIR, "payments")
TEMPLATES_DIR = "templates"
THUMBNAILS_DIR = os.path.join(TEMPLATES_DIR, "thumbnails")

# Database Settings
DATABASE_CONFIG = {
    "type": "sqlite",
    "path": os.path.join(DATA_DIR, "video_income.db"),
    "backup_interval": 24  # hours
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "logs/video_income.log"
}

# Feature Flags
FEATURES = {
    "ai_content_analysis": True,
    "automated_clipping": True,
    "thumbnail_generation": True,
    "youtube_upload": True,
    "payment_tracking": True,
    "analytics_dashboard": True
}

# Rate Limiting
RATE_LIMITS = {
    "youtube_api_calls_per_minute": 60,
    "ai_requests_per_hour": 100,
    "upload_attempts_per_hour": 10,
    "payment_checks_per_day": 24
}

# Security Settings
SECURITY_CONFIG = {
    "encrypt_api_keys": True,
    "require_authentication": False,
    "session_timeout": 3600,  # seconds
    "max_file_size": 128 * 1024 * 1024 * 1024  # 128GB
}

# Performance Settings
PERFORMANCE_CONFIG = {
    "max_concurrent_uploads": 3,
    "video_processing_threads": 4,
    "ai_processing_batch_size": 10,
    "cache_duration": 86400  # 24 hours
} 