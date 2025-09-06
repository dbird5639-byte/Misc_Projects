"""
Logging utilities for AI-Powered Solana Meme Coin Sniper
"""

import logging
import logging.config
import structlog
from typing import Optional
from pathlib import Path
import sys

from config.settings import LOG_CONFIG, LOGS_DIR


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Setup structured logging for the application
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional custom log file path
    """
    
    # Ensure logs directory exists
    LOGS_DIR.mkdir(exist_ok=True)
    
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
    
    # Update log config with custom file if provided
    log_config = LOG_CONFIG.copy()
    if log_file:
        log_config["handlers"]["file"]["filename"] = str(Path(log_file))
    
    # Set log level
    log_config["loggers"][""]["level"] = level.upper()
    log_config["handlers"]["console"]["level"] = level.upper()
    
    # Apply configuration
    logging.config.dictConfig(log_config)
    
    # Set root logger level
    logging.getLogger().setLevel(getattr(logging, level.upper()))


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


def setup_console_logging(level: str = "INFO") -> None:
    """
    Setup console-only logging for development
    
    Args:
        level: Logging level
    """
    # Configure basic console logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_colored_console_logging(level: str = "INFO") -> None:
    """
    Setup colored console logging for better readability
    
    Args:
        level: Logging level
    """
    # Create console handler with colored formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Create colored formatter
    formatter = ColoredFormatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers and add colored handler
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.addHandler(console_handler)


class PerformanceLogger:
    """Logger for performance metrics and timing"""
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = get_logger(logger_name)
    
    def log_timing(self, operation: str, duration: float, **kwargs):
        """Log timing information for an operation"""
        self.logger.info(
            "Operation timing",
            operation=operation,
            duration_ms=duration * 1000,
            **kwargs
        )
    
    def log_performance_metric(self, metric_name: str, value: float, **kwargs):
        """Log a performance metric"""
        self.logger.info(
            "Performance metric",
            metric_name=metric_name,
            value=value,
            **kwargs
        )
    
    def log_error_rate(self, operation: str, error_count: int, total_count: int, **kwargs):
        """Log error rate for an operation"""
        error_rate = error_count / max(total_count, 1)
        self.logger.warning(
            "Error rate",
            operation=operation,
            error_count=error_count,
            total_count=total_count,
            error_rate=error_rate,
            **kwargs
        )


class TradeLogger:
    """Logger specifically for trading operations"""
    
    def __init__(self):
        self.logger = get_logger("trading")
    
    def log_trade_opportunity(self, token_address: str, confidence: float, **kwargs):
        """Log a trading opportunity"""
        self.logger.info(
            "Trading opportunity detected",
            token_address=token_address,
            confidence=confidence,
            **kwargs
        )
    
    def log_trade_execution(self, token_address: str, action: str, amount: float, **kwargs):
        """Log trade execution"""
        self.logger.info(
            "Trade executed",
            token_address=token_address,
            action=action,
            amount=amount,
            **kwargs
        )
    
    def log_trade_result(self, token_address: str, profit_loss: float, success: bool, **kwargs):
        """Log trade result"""
        level = "info" if success else "warning"
        self.logger.log(
            getattr(logging, level.upper()),
            "Trade result",
            token_address=token_address,
            profit_loss=profit_loss,
            success=success,
            **kwargs
        )
    
    def log_risk_assessment(self, token_address: str, risk_score: float, risk_factors: list, **kwargs):
        """Log risk assessment"""
        self.logger.info(
            "Risk assessment",
            token_address=token_address,
            risk_score=risk_score,
            risk_factors=risk_factors,
            **kwargs
        )


class AILogger:
    """Logger specifically for AI operations"""
    
    def __init__(self):
        self.logger = get_logger("ai")
    
    def log_model_request(self, model_name: str, prompt_length: int, **kwargs):
        """Log AI model request"""
        self.logger.info(
            "AI model request",
            model_name=model_name,
            prompt_length=prompt_length,
            **kwargs
        )
    
    def log_model_response(self, model_name: str, response_time: float, confidence: float, **kwargs):
        """Log AI model response"""
        self.logger.info(
            "AI model response",
            model_name=model_name,
            response_time_ms=response_time * 1000,
            confidence=confidence,
            **kwargs
        )
    
    def log_model_error(self, model_name: str, error: str, **kwargs):
        """Log AI model error"""
        self.logger.error(
            "AI model error",
            model_name=model_name,
            error=error,
            **kwargs
        )
    
    def log_decision_making(self, decision_type: str, confidence: float, reasoning: str, **kwargs):
        """Log AI decision making"""
        self.logger.info(
            "AI decision",
            decision_type=decision_type,
            confidence=confidence,
            reasoning=reasoning[:200] + "..." if len(reasoning) > 200 else reasoning,
            **kwargs
        )


# Convenience functions for common logging patterns
def log_function_call(func_name: str, **kwargs):
    """Log a function call with parameters"""
    logger = get_logger("function_calls")
    logger.debug("Function called", function=func_name, **kwargs)


def log_api_call(api_name: str, endpoint: str, response_time: float, status_code: int = None, **kwargs):
    """Log an API call"""
    logger = get_logger("api_calls")
    level = "info" if status_code and status_code < 400 else "warning"
    logger.log(
        getattr(logging, level.upper()),
        "API call",
        api_name=api_name,
        endpoint=endpoint,
        response_time_ms=response_time * 1000,
        status_code=status_code,
        **kwargs
    )


def log_system_event(event_type: str, description: str, **kwargs):
    """Log a system event"""
    logger = get_logger("system_events")
    logger.info(
        "System event",
        event_type=event_type,
        description=description,
        **kwargs
    )


def log_security_event(event_type: str, severity: str, description: str, **kwargs):
    """Log a security event"""
    logger = get_logger("security")
    level = severity.lower()
    logger.log(
        getattr(logging, level.upper()),
        "Security event",
        event_type=event_type,
        severity=severity,
        description=description,
        **kwargs
    )


# Context managers for timing operations
import time
from contextlib import contextmanager


@contextmanager
def log_operation_timing(operation_name: str, logger_name: str = "timing"):
    """Context manager for logging operation timing"""
    logger = get_logger(logger_name)
    start_time = time.time()
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.info(
            "Operation completed",
            operation=operation_name,
            duration_ms=duration * 1000
        )


@contextmanager
def log_error_context(operation_name: str, logger_name: str = "errors"):
    """Context manager for logging errors with context"""
    logger = get_logger(logger_name)
    
    try:
        yield
    except Exception as e:
        logger.error(
            "Operation failed",
            operation=operation_name,
            error=str(e),
            error_type=type(e).__name__
        )
        raise 