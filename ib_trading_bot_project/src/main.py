"""
Main entry point for Interactive Brokers Trading Bot

This module initializes the trading bot, establishes connections,
and manages the overall trading process.
"""

import sys
import os
import signal
import time
from typing import Optional, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from src.connection.ib_connector import IBConnector
from src.trading.order_manager import OrderManager
from src.trading.position_manager import PositionManager
from src.trading.risk_manager import RiskManager
from src.strategies.momentum_strategy import MomentumStrategy
from src.strategies.mean_reversion_strategy import MeanReversionStrategy
from src.utils.logger import setup_logging, logger

class TradingBot:
    """
    Main trading bot class that orchestrates all components
    """
    
    def __init__(self):
        """Initialize the trading bot"""
        self.connector: Optional[IBConnector] = None
        self.order_manager: Optional[OrderManager] = None
        self.position_manager: Optional[PositionManager] = None
        self.risk_manager: Optional[RiskManager] = None
        self.strategies: Dict[str, Any] = {}
        self.running = False
        
        # Setup logging
        setup_logging()
        
        # Validate settings
        self._validate_configuration()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _validate_configuration(self):
        """Validate configuration settings"""
        errors = settings.validate_settings()
        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            sys.exit(1)
        
        logger.info("Configuration validation passed")
    
    def initialize(self):
        """Initialize all trading bot components"""
        try:
            logger.info("Initializing trading bot...")
            
            # Initialize IB connection
            self.connector = IBConnector(
                host=settings.ib.host,
                port=settings.ib.port,
                client_id=settings.ib.client_id
            )
            
            # Initialize trading components
            self.order_manager = OrderManager(self.connector)
            self.position_manager = PositionManager(self.connector)
            self.risk_manager = RiskManager(
                self.position_manager,
                settings.risk
            )
            
            # Initialize strategies
            self._initialize_strategies()
            
            logger.info("Trading bot initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize trading bot: {e}")
            raise
    
    def _initialize_strategies(self):
        """Initialize trading strategies"""
        logger.info("Initializing trading strategies...")
        
        # Initialize momentum strategy
        if settings.instruments["stocks"]:
            self.strategies["momentum"] = MomentumStrategy(
                self.order_manager,
                self.position_manager,
                self.risk_manager
            )
            
            self.strategies["mean_reversion"] = MeanReversionStrategy(
                self.order_manager,
                self.position_manager,
                self.risk_manager
            )
        
        logger.info(f"Initialized {len(self.strategies)} strategies")
    
    def connect(self):
        """Establish connection to Interactive Brokers"""
        try:
            logger.info("Connecting to Interactive Brokers...")
            
            if not self.connector:
                raise RuntimeError("Connector not initialized")
            
            # Connect to IB
            success = self.connector.connect()
            
            if success:
                logger.info("Successfully connected to Interactive Brokers")
                
                # Subscribe to market data
                self._subscribe_market_data()
                
            else:
                logger.error("Failed to connect to Interactive Brokers")
                return False
                
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
        
        return True
    
    def _subscribe_market_data(self):
        """Subscribe to market data for configured symbols"""
        try:
            if not self.connector:
                logger.error("Connector not initialized")
                return
                
            symbols = settings.market_data["subscriptions"]
            logger.info(f"Subscribing to market data for {len(symbols)} symbols")
            
            for symbol in symbols:
                self.connector.subscribe_market_data(symbol)
                time.sleep(0.1)  # Rate limiting
            
            logger.info("Market data subscriptions completed")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to market data: {e}")
    
    def start(self):
        """Start the trading bot"""
        try:
            logger.info("Starting trading bot...")
            
            if not self.connector or not self.connector.is_connected():
                logger.error("Not connected to Interactive Brokers")
                return False
            
            self.running = True
            
            # Start strategies
            for name, strategy in self.strategies.items():
                logger.info(f"Starting strategy: {name}")
                strategy.start()
            
            logger.info("Trading bot started successfully")
            
            # Main trading loop
            self._trading_loop()
            
        except Exception as e:
            logger.error(f"Error starting trading bot: {e}")
            return False
    
    def _trading_loop(self):
        """Main trading loop"""
        logger.info("Entering main trading loop...")
        
        try:
            while self.running:
                # Check connection status
                if not self.connector or not self.connector.is_connected():
                    logger.warning("Connection lost, attempting to reconnect...")
                    if not self.connect():
                        logger.error("Failed to reconnect, stopping bot")
                        break
                
                # Update positions and risk metrics
                if self.position_manager:
                    self.position_manager.update_positions()
                if self.risk_manager:
                    self.risk_manager.check_risk_limits()
                
                # Run strategy analysis
                for name, strategy in self.strategies.items():
                    if strategy.is_running():
                        strategy.analyze_and_trade()
                
                # Sleep for update interval
                time.sleep(settings.market_data["update_interval"])
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the trading bot"""
        logger.info("Stopping trading bot...")
        
        self.running = False
        
        # Stop strategies
        for name, strategy in self.strategies.items():
            logger.info(f"Stopping strategy: {name}")
            strategy.stop()
        
        # Close connection
        if self.connector:
            self.connector.disconnect()
        
        logger.info("Trading bot stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle system signals"""
        logger.info(f"Received signal {signum}")
        self.stop()
        sys.exit(0)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        return {
            "running": self.running,
            "connected": self.connector.is_connected() if self.connector else False,
            "strategies": {
                name: strategy.get_status() 
                for name, strategy in self.strategies.items()
            },
            "positions": self.position_manager.get_positions() if self.position_manager else [],
            "risk_metrics": self.risk_manager.get_risk_metrics() if self.risk_manager else {}
        }

def main():
    """Main entry point"""
    print("=" * 60)
    print("Interactive Brokers Trading Bot")
    print("=" * 60)
    print("⚠️  DISCLAIMER: This is not financial advice.")
    print("   Trading involves significant risk and potential losses.")
    print("=" * 60)
    
    # Create and run trading bot
    bot = TradingBot()
    
    try:
        # Initialize bot
        bot.initialize()
        
        # Connect to IB
        if not bot.connect():
            logger.error("Failed to connect to Interactive Brokers")
            return 1
        
        # Start trading
        bot.start()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 