"""
Configuration settings for AI Phone Agent Project
"""

import os
from pathlib import Path
from typing import Dict, List

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONVERSATIONS_DIR = DATA_DIR / "conversations"
RECORDINGS_DIR = DATA_DIR / "recordings"
LOGS_DIR = DATA_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, CONVERSATIONS_DIR, RECORDINGS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Settings
API_SETTINGS = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": True,
    "title": "AI Phone Agent API",
    "version": "1.0.0"
}

# OpenAI Settings
OPENAI_SETTINGS = {
    "api_key": os.getenv("OPENAI_API_KEY", "your-openai-api-key"),
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 150,
    "voice_model": "tts-1",
    "voice": "alloy",  # alloy, echo, fable, onyx, nova, shimmer
    "speech_rate": 1.0
}

# Twilio Settings
TWILIO_SETTINGS = {
    "account_sid": os.getenv("TWILIO_ACCOUNT_SID", "your-twilio-account-sid"),
    "auth_token": os.getenv("TWILIO_AUTH_TOKEN", "your-twilio-auth-token"),
    "phone_number": os.getenv("TWILIO_PHONE_NUMBER", "+1234567890"),
    "webhook_url": os.getenv("WEBHOOK_URL", "https://your-domain.com/webhook")
}

# Voice Processing Settings
VOICE_SETTINGS = {
    "sample_rate": 16000,
    "chunk_size": 1024,
    "channels": 1,
    "format": "int16",
    "vad_mode": 3,  # Voice Activity Detection mode (0-3)
    "silence_threshold": 0.5,
    "speech_threshold": 0.8,
    "max_silence_duration": 2.0,  # seconds
    "min_speech_duration": 0.5,   # seconds
}

# Conversation Settings
CONVERSATION_SETTINGS = {
    "max_conversation_length": 300,  # seconds
    "max_turns": 20,
    "context_window": 10,  # number of previous exchanges to remember
    "greeting_message": "Hello! Thank you for calling. How can I help you today?",
    "goodbye_message": "Thank you for calling. Have a great day!",
    "fallback_message": "I'm sorry, I didn't understand that. Could you please repeat?",
    "escalation_threshold": 3,  # number of fallbacks before escalation
}

# Knowledge Base Settings
KNOWLEDGE_SETTINGS = {
    "max_results": 5,
    "similarity_threshold": 0.7,
    "cache_size": 1000,
    "update_interval": 3600,  # seconds
    "categories": [
        "product_info",
        "pricing",
        "support",
        "policies",
        "general"
    ]
}

# Testing Mode Settings
TESTING_SETTINGS = {
    "enabled": True,
    "simulate_phone_calls": True,
    "terminal_interface": True,
    "mock_audio": True,
    "test_conversations": [
        "What are your business hours?",
        "How much does your product cost?",
        "I need help with my order",
        "What is your return policy?"
    ]
}

# Audio Settings
AUDIO_SETTINGS = {
    "input_device": None,  # Auto-detect
    "output_device": None,  # Auto-detect
    "buffer_size": 4096,
    "latency": 0.1,  # seconds
    "noise_reduction": True,
    "echo_cancellation": True,
    "automatic_gain_control": True
}

# Logging Settings
LOGGING_SETTINGS = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "phone_agent.log",
    "max_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
    "console_output": True
}

# Security Settings
SECURITY_SETTINGS = {
    "encrypt_conversations": True,
    "encryption_key": os.getenv("ENCRYPTION_KEY", "your-encryption-key"),
    "max_file_size": 50 * 1024 * 1024,  # 50MB
    "allowed_file_types": [".wav", ".mp3", ".m4a"],
    "rate_limiting": {
        "calls_per_minute": 10,
        "calls_per_hour": 100,
        "calls_per_day": 1000
    }
}

# Analytics Settings
ANALYTICS_SETTINGS = {
    "track_calls": True,
    "track_conversations": True,
    "track_satisfaction": True,
    "retention_days": 90,
    "metrics": [
        "call_duration",
        "conversation_turns",
        "escalation_rate",
        "satisfaction_score",
        "common_questions"
    ]
}

# Environment variables
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = ENVIRONMENT == "development" 