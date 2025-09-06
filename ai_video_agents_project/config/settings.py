"""
Configuration settings for AI Video Agents Project
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_VIDEOS_DIR = DATA_DIR / "input_videos"
OUTPUT_CLIPS_DIR = DATA_DIR / "output_clips"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"

# Create directories if they don't exist
for directory in [DATA_DIR, INPUT_VIDEOS_DIR, OUTPUT_CLIPS_DIR, TRANSCRIPTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# AI Model Settings
AI_MODELS = {
    "deepseek": {
        "name": "DeepSeek",
        "api_base": "https://api.deepseek.com",
        "model": "deepseek-chat"
    },
    "grok": {
        "name": "Grok",
        "api_base": "https://api.grok.ai",
        "model": "grok-beta"
    },
    "llama": {
        "name": "Llama",
        "api_base": "https://api.llama.ai",
        "model": "llama-2-70b"
    }
}

# Default AI model to use
DEFAULT_AI_MODEL = "deepseek"

# Video Processing Settings
VIDEO_SETTINGS = {
    "max_clip_duration": 60,  # seconds
    "min_clip_duration": 10,  # seconds
    "output_format": "mp4",
    "video_quality": "720p",
    "fps": 30
}

# Content Generation Settings
CONTENT_SETTINGS = {
    "max_summary_length": 200,
    "title_max_length": 100,
    "description_max_length": 500,
    "tags_max_count": 10
}

# Platform Settings
PLATFORMS = {
    "youtube": {
        "name": "YouTube",
        "max_title_length": 100,
        "max_description_length": 5000,
        "supported_formats": ["mp4", "avi", "mov"]
    },
    "tiktok": {
        "name": "TikTok",
        "max_duration": 60,
        "supported_formats": ["mp4"]
    },
    "instagram": {
        "name": "Instagram",
        "max_duration": 60,
        "supported_formats": ["mp4"]
    }
}

# Environment variables (load from .env file if available)
API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY", ""),
    "deepseek": os.getenv("DEEPSEEK_API_KEY", ""),
    "grok": os.getenv("GROK_API_KEY", ""),
    "youtube": os.getenv("YOUTUBE_API_KEY", "")
}

# Logging settings
LOGGING = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": PROJECT_ROOT / "logs" / "app.log"
}

# Create logs directory
(PROJECT_ROOT / "logs").mkdir(exist_ok=True) 