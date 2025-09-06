"""
Logging utilities for the trading bot

Provides centralized logging configuration and utilities.
"""

import os
import sys
from datetime import datetime
from typing import Optional, Union

try:
    from loguru import logger as loguru_logger  # type: ignore
    _LOGURU = True
    logger = loguru_logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    _LOGURU = False

def setup_logging(log_file: Optional[str] = None, 
                 log_level: str = "INFO",
                 max_size: str = "10MB",
                 backup_count: int = 5):
    """
    Setup logging configuration
    
    Args:
        log_file: Log file path (optional)
        log_level: Logging level
        max_size: Maximum log file size
        backup_count: Number of backup files to keep
    """
    try:
        if _LOGURU:
            # All loguru-specific code inside this block
            logger.remove()  # type: ignore
            
            # Add console logger
            logger.add(  # type: ignore
                sys.stdout,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                       "<level>{level: <8}</level> | "
                       "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                       "<level>{message}</level>",
                level=log_level,
                colorize=True
            )
            
            # Add file logger if specified
            if log_file:
                # Create log directory if it doesn't exist
                log_dir = os.path.dirname(log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                
                logger.add(  # type: ignore
                    log_file,
                    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                           "{name}:{function}:{line} | {message}",
                    level=log_level,
                    rotation=max_size,
                    retention=backup_count,
                    compression="zip"
                )
            
            logger.info("Logging setup completed")
        else:
            # Standard logging fallback
            logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
            logger.info("Standard logging setup completed")
        
    except Exception as e:
        print(f"Error setting up logging: {e}")

def get_logger(name: str):
    """
    Get a logger instance for a specific module
    
    Args:
        name: Module name
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)  # type: ignore

def log_trade(trade_data: dict):
    """
    Log trade information
    
    Args:
        trade_data: Trade data dictionary
    """
    try:
        logger.info(f"TRADE: {trade_data}")
    except Exception as e:
        logger.error(f"Error logging trade: {e}")

def log_signal(signal_data: dict):
    """
    Log trading signal
    
    Args:
        signal_data: Signal data dictionary
    """
    try:
        logger.info(f"SIGNAL: {signal_data}")
    except Exception as e:
        logger.error(f"Error logging signal: {e}")

def log_error(error: Exception, context: str = ""):
    """
    Log error with context
    
    Args:
        error: Exception object
        context: Additional context information
    """
    try:
        if context:
            logger.error(f"ERROR in {context}: {error}")
        else:
            logger.error(f"ERROR: {error}")
    except Exception as e:
        print(f"Error logging error: {e}")

def log_performance(metrics: dict):
    """
    Log performance metrics
    
    Args:
        metrics: Performance metrics dictionary
    """
    try:
        logger.info(f"PERFORMANCE: {metrics}")
    except Exception as e:
        logger.error(f"Error logging performance: {e}")

def log_risk_violation(violation: dict):
    """
    Log risk limit violation
    
    Args:
        violation: Risk violation data
    """
    try:
        logger.warning(f"RISK VIOLATION: {violation}")
    except Exception as e:
        logger.error(f"Error logging risk violation: {e}")

def create_log_filename(prefix: str = "trading_bot") -> str:
    """
    Create a log filename with timestamp
    
    Args:
        prefix: Filename prefix
        
    Returns:
        Log filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"logs/{prefix}_{timestamp}.log"

def setup_structured_logging():
    """Setup structured logging for better analysis"""
    try:
        # Add structured logging format
        logger.add(  # type: ignore
            "logs/structured.json",
            format="{time} | {level} | {extra}",
            serialize=True,
            rotation="1 day",
            retention="30 days"
        )
        
        logger.info("Structured logging setup completed")
        
    except Exception as e:
        logger.error(f"Error setting up structured logging: {e}")

def log_market_data(symbol: str, data: dict):
    """
    Log market data updates
    
    Args:
        symbol: Stock symbol
        data: Market data
    """
    try:
        logger.debug(f"MARKET DATA {symbol}: {data}")
    except Exception as e:
        logger.error(f"Error logging market data: {e}")

def log_connection_status(status: str, details: str = ""):
    """
    Log connection status
    
    Args:
        status: Connection status
        details: Additional details
    """
    try:
        if details:
            logger.info(f"CONNECTION {status}: {details}")
        else:
            logger.info(f"CONNECTION {status}")
    except Exception as e:
        logger.error(f"Error logging connection status: {e}")

def log_strategy_event(strategy_name: str, event: str, data: Optional[dict] = None):
    """
    Log strategy events
    
    Args:
        strategy_name: Name of the strategy
        event: Event type
        data: Event data
    """
    try:
        if data:
            logger.info(f"STRATEGY {strategy_name} {event}: {data}")
        else:
            logger.info(f"STRATEGY {strategy_name} {event}")
    except Exception as e:
        logger.error(f"Error logging strategy event: {e}")

def log_portfolio_update(portfolio_data: dict):
    """
    Log portfolio updates
    
    Args:
        portfolio_data: Portfolio data
    """
    try:
        logger.info(f"PORTFOLIO UPDATE: {portfolio_data}")
    except Exception as e:
        logger.error(f"Error logging portfolio update: {e}")

def log_order_update(order_id: int, status: str, details: str = ""):
    """
    Log order updates
    
    Args:
        order_id: Order ID
        status: Order status
        details: Additional details
    """
    try:
        if details:
            logger.info(f"ORDER {order_id} {status}: {details}")
        else:
            logger.info(f"ORDER {order_id} {status}")
    except Exception as e:
        logger.error(f"Error logging order update: {e}")

def log_risk_check(risk_data: dict):
    """
    Log risk management checks
    
    Args:
        risk_data: Risk check data
    """
    try:
        logger.info(f"RISK CHECK: {risk_data}")
    except Exception as e:
        logger.error(f"Error logging risk check: {e}")

def log_system_event(event: str, data: Optional[dict] = None):
    """
    Log system events
    
    Args:
        event: Event type
        data: Event data
    """
    try:
        if data:
            logger.info(f"SYSTEM {event}: {data}")
        else:
            logger.info(f"SYSTEM {event}")
    except Exception as e:
        logger.error(f"Error logging system event: {e}")

def log_startup_info(config: dict):
    """
    Log startup information
    
    Args:
        config: Configuration data
    """
    try:
        logger.info("=" * 60)
        logger.info("TRADING BOT STARTUP")
        logger.info("=" * 60)
        logger.info(f"Configuration: {config}")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Error logging startup info: {e}")

def log_shutdown_info():
    """Log shutdown information"""
    try:
        logger.info("=" * 60)
        logger.info("TRADING BOT SHUTDOWN")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Error logging shutdown info: {e}") 