"""
Logging utility for Solana Trading Bot 2025

Provides centralized logging with different levels and outputs.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json

def setup_logger(name: str, log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with console and file output
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

class TradingLogger:
    """Specialized logger for trading operations"""
    
    def __init__(self, name: str, log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Set up different loggers for different purposes
        self.general_logger = self._setup_general_logger()
        self.trade_logger = self._setup_trade_logger()
        self.error_logger = self._setup_error_logger()
        self.performance_logger = self._setup_performance_logger()
    
    def _setup_general_logger(self) -> logging.Logger:
        """Set up general purpose logger"""
        log_file = self.log_dir / f"{self.name}_general.log"
        return setup_logger(f"{self.name}_general", "INFO", str(log_file))
    
    def _setup_trade_logger(self) -> logging.Logger:
        """Set up trade-specific logger"""
        log_file = self.log_dir / f"{self.name}_trades.log"
        return setup_logger(f"{self.name}_trades", "INFO", str(log_file))
    
    def _setup_error_logger(self) -> logging.Logger:
        """Set up error logger"""
        log_file = self.log_dir / f"{self.name}_errors.log"
        return setup_logger(f"{self.name}_errors", "ERROR", str(log_file))
    
    def _setup_performance_logger(self) -> logging.Logger:
        """Set up performance logger"""
        log_file = self.log_dir / f"{self.name}_performance.log"
        return setup_logger(f"{self.name}_performance", "INFO", str(log_file))
    
    def log_trade(self, trade_data: dict):
        """Log trade information"""
        try:
            trade_log = {
                "timestamp": datetime.now().isoformat(),
                "type": "trade",
                "data": trade_data
            }
            
            self.trade_logger.info(json.dumps(trade_log))
            
        except Exception as e:
            self.error_logger.error(f"Error logging trade: {e}")
    
    def log_signal(self, signal_data: dict):
        """Log trading signal"""
        try:
            signal_log = {
                "timestamp": datetime.now().isoformat(),
                "type": "signal",
                "data": signal_data
            }
            
            self.trade_logger.info(json.dumps(signal_log))
            
        except Exception as e:
            self.error_logger.error(f"Error logging signal: {e}")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error with context"""
        try:
            error_log = {
                "timestamp": datetime.now().isoformat(),
                "type": "error",
                "error": str(error),
                "error_type": type(error).__name__,
                "context": context
            }
            
            self.error_logger.error(json.dumps(error_log))
            
        except Exception as e:
            print(f"Error logging error: {e}")
    
    def log_performance(self, performance_data: dict):
        """Log performance metrics"""
        try:
            performance_log = {
                "timestamp": datetime.now().isoformat(),
                "type": "performance",
                "data": performance_data
            }
            
            self.performance_logger.info(json.dumps(performance_log))
            
        except Exception as e:
            self.error_logger.error(f"Error logging performance: {e}")
    
    def log_info(self, message: str):
        """Log general information"""
        self.general_logger.info(message)
    
    def log_warning(self, message: str):
        """Log warning"""
        self.general_logger.warning(message)
    
    def log_debug(self, message: str):
        """Log debug information"""
        self.general_logger.debug(message)

class StructuredLogger:
    """Logger that outputs structured JSON logs"""
    
    def __init__(self, name: str, log_file: Optional[str] = None):
        self.name = name
        self.logger = setup_logger(name, "INFO", log_file)
    
    def log(self, level: str, message: str, **kwargs):
        """Log structured message"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": level.upper(),
                "logger": self.name,
                "message": message,
                **kwargs
            }
            
            log_message = json.dumps(log_entry)
            
            if level.upper() == "DEBUG":
                self.logger.debug(log_message)
            elif level.upper() == "INFO":
                self.logger.info(log_message)
            elif level.upper() == "WARNING":
                self.logger.warning(log_message)
            elif level.upper() == "ERROR":
                self.logger.error(log_message)
            elif level.upper() == "CRITICAL":
                self.logger.critical(log_message)
            
        except Exception as e:
            print(f"Error in structured logging: {e}")
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.log("DEBUG", message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.log("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.log("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.log("ERROR", message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.log("CRITICAL", message, **kwargs)

# Convenience function for quick logging setup
def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Get a logger with default settings"""
    return setup_logger(name, "INFO", log_file)

# Global logger instance
def get_trading_logger(name: str) -> TradingLogger:
    """Get a trading logger instance"""
    return TradingLogger(name)

def get_structured_logger(name: str, log_file: Optional[str] = None) -> StructuredLogger:
    """Get a structured logger instance"""
    return StructuredLogger(name, log_file) 