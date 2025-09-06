"""
Configuration settings for AI Business Builder Project
"""

import os
from typing import Dict, List, Any

# Project Settings
PROJECT_NAME = "AI Business Builder"
PROJECT_VERSION = "1.0.0"
PROJECT_DESCRIPTION = "Build profitable businesses using AI tools"

# API Keys and External Services
# Note: These should be set as environment variables in production
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# App Store Analysis Settings
APP_STORE_CATEGORIES = [
    "Business",
    "Education", 
    "Entertainment",
    "Finance",
    "Food & Drink",
    "Games",
    "Health & Fitness",
    "Lifestyle",
    "Medical",
    "Music",
    "Navigation",
    "News",
    "Photo & Video",
    "Productivity",
    "Reference",
    "Shopping",
    "Social Networking",
    "Sports",
    "Travel",
    "Utilities",
    "Weather"
]

# Business Categories for Analysis
BUSINESS_CATEGORIES = {
    "mobile_apps": {
        "description": "Mobile applications for iOS and Android",
        "subcategories": [
            "Language Learning",
            "Study & Education", 
            "Health & Fitness",
            "Productivity",
            "Entertainment",
            "Finance",
            "Social Networking"
        ]
    },
    "web_applications": {
        "description": "Web-based applications and services",
        "subcategories": [
            "Content Creation",
            "E-commerce",
            "Analytics",
            "Project Management",
            "Communication",
            "Creative Tools"
        ]
    },
    "ai_services": {
        "description": "AI-powered services and APIs",
        "subcategories": [
            "Content Generation",
            "Image Processing",
            "Text Analysis",
            "Recommendation Systems",
            "Automation",
            "Predictive Analytics"
        ]
    }
}

# Market Research Settings
MARKET_RESEARCH_CONFIG = {
    "min_market_size": 1000000,  # Minimum market size in USD
    "max_competition": 50,  # Maximum number of direct competitors
    "min_rating": 4.0,  # Minimum app store rating to consider
    "min_reviews": 100,  # Minimum number of reviews
    "target_audience_sizes": {
        "niche": 10000,
        "small": 100000,
        "medium": 1000000,
        "large": 10000000
    }
}

# Revenue Models
REVENUE_MODELS = {
    "saas": {
        "name": "Software as a Service",
        "description": "Subscription-based software",
        "pricing_strategies": [
            "Freemium",
            "Tiered Pricing",
            "Usage-based",
            "Enterprise"
        ]
    },
    "mobile_app": {
        "name": "Mobile App",
        "description": "App store and in-app purchases",
        "pricing_strategies": [
            "One-time Purchase",
            "In-app Purchases",
            "Subscription",
            "Advertising"
        ]
    },
    "api_service": {
        "name": "API Service",
        "description": "API-based services",
        "pricing_strategies": [
            "Per Request",
            "Monthly Tiers",
            "Usage-based",
            "Enterprise"
        ]
    }
}

# Development Settings
DEVELOPMENT_CONFIG = {
    "default_language": "python",
    "supported_languages": ["python", "javascript", "typescript", "react", "vue"],
    "deployment_platforms": [
        "Vercel",
        "Netlify", 
        "Heroku",
        "AWS",
        "Google Cloud",
        "Azure"
    ],
    "ai_tools": {
        "code_generation": ["GitHub Copilot", "Cursor", "CodeWhisperer"],
        "design": ["Midjourney", "DALL-E", "Stable Diffusion"],
        "content": ["ChatGPT", "Claude", "Bard"],
        "analytics": ["Google Analytics", "Mixpanel", "Amplitude"]
    }
}

# Success Metrics
SUCCESS_METRICS = {
    "user_metrics": [
        "Daily Active Users",
        "Monthly Active Users", 
        "User Retention Rate",
        "User Acquisition Cost",
        "Session Duration"
    ],
    "business_metrics": [
        "Monthly Recurring Revenue",
        "Customer Lifetime Value",
        "Churn Rate",
        "Conversion Rate",
        "Profit Margin"
    ],
    "technical_metrics": [
        "App Performance",
        "Error Rate",
        "Response Time",
        "Uptime",
        "Security Score"
    ]
}

# File Paths
DATA_DIR = "data"
TEMPLATES_DIR = "templates"
ANALYSIS_DIR = os.path.join(DATA_DIR, "app_analysis")
RESEARCH_DIR = os.path.join(DATA_DIR, "market_research")
METRICS_DIR = os.path.join(DATA_DIR, "business_metrics")

# Database Settings (if using)
DATABASE_CONFIG = {
    "type": "sqlite",  # or "postgresql", "mysql"
    "path": os.path.join(DATA_DIR, "business_builder.db"),
    "backup_interval": 24  # hours
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "logs/business_builder.log"
}

# Feature Flags
FEATURES = {
    "ai_analysis": True,
    "market_research": True,
    "idea_generation": True,
    "project_templates": True,
    "deployment_automation": True,
    "analytics_dashboard": True
}

# Rate Limiting
RATE_LIMITS = {
    "api_calls_per_minute": 60,
    "ai_requests_per_hour": 100,
    "market_research_per_day": 50
}

# Security Settings
SECURITY_CONFIG = {
    "encrypt_api_keys": True,
    "require_authentication": False,
    "session_timeout": 3600,  # seconds
    "max_file_size": 10 * 1024 * 1024  # 10MB
} 