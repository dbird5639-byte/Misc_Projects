"""
Logger utilities for AI Market Maker & Liquidation Monitor
"""

import logging
import sys
from typing import Optional
from pathlib import Path
import structlog
from datetime import datetime

# Base directory
BASE_DIR = Path(__file__).parent.parent


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    # Create logs directory
    logs_dir = BASE_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Set log level
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logs_dir / "market_maker.log")
        ]
    )


def get_logger(name: str) -> logging.Logger:
    """Get logger instance"""
    return structlog.get_logger(name)


def log_performance_metrics(metrics: dict, component: str):
    """Log performance metrics"""
    logger = get_logger(f"{component}.metrics")
    logger.info("Performance metrics", **metrics)


def log_trade_event(event_type: str, data: dict):
    """Log trade event"""
    logger = get_logger("trading.events")
    logger.info(f"Trade event: {event_type}", **data)


def log_liquidation_event(event_type: str, data: dict):
    """Log liquidation event"""
    logger = get_logger("liquidation.events")
    logger.warning(f"Liquidation event: {event_type}", **data)


def log_prediction_event(event_type: str, data: dict):
    """Log prediction event"""
    logger = get_logger("prediction.events")
    logger.info(f"Prediction event: {event_type}", **data)


def log_system_event(event_type: str, data: dict):
    """Log system event"""
    logger = get_logger("system.events")
    logger.info(f"System event: {event_type}", **data)


def log_error(error: Exception, context: str = ""):
    """Log error with context"""
    logger = get_logger("system.errors")
    logger.error(f"Error in {context}: {str(error)}", exc_info=True)


def log_warning(message: str, context: str = ""):
    """Log warning with context"""
    logger = get_logger("system.warnings")
    logger.warning(f"Warning in {context}: {message}")


def log_info(message: str, context: str = ""):
    """Log info with context"""
    logger = get_logger("system.info")
    logger.info(f"Info in {context}: {message}")


def log_debug(message: str, context: str = ""):
    """Log debug with context"""
    logger = get_logger("system.debug")
    logger.debug(f"Debug in {context}: {message}")


# Convenience functions for quick logging
def log_startup(component: str):
    """Log component startup"""
    log_info(f"Starting {component}", component)


def log_shutdown(component: str):
    """Log component shutdown"""
    log_info(f"Stopping {component}", component)


def log_initialization(component: str, success: bool):
    """Log component initialization"""
    if success:
        log_info(f"{component} initialized successfully", component)
    else:
        log_error(Exception(f"Failed to initialize {component}"), component)


def log_configuration(config: dict, component: str):
    """Log configuration"""
    log_debug(f"Configuration loaded for {component}", component)
    log_debug(f"Config: {config}", component) 