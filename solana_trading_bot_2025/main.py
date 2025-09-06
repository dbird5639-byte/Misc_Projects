"""
Main entry point for Solana Trading Bot 2025

Handles bot initialization, command-line interface, and main execution loop.
"""

import asyncio
import argparse
import signal
import sys
from typing import Optional
import time

from config.settings import Config
from bots.sniper_bot import SniperBot
from bots.copy_bot import CopyBot
from utils.logger import setup_logger, get_trading_logger
from utils.notifications import NotificationManager
from web.dashboard import start_dashboard

class TradingBotManager:
    """Manages the trading bots"""
    
    def __init__(self):
        self.config = Config()
        self.logger = setup_logger("main")
        self.trading_logger = get_trading_logger("main")
        self.notification_manager = NotificationManager(self.config)
        
        # Bot instances
        self.sniper_bot: Optional[SniperBot] = None
        self.copy_bot: Optional[CopyBot] = None
        
        # Control flags
        self.running = False
        self.shutdown_requested = False
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def start(self, bot_type: str = "both"):
        """Start the trading bots"""
        try:
            self.logger.info("Starting Solana Trading Bot 2025...")
            
            # Validate configuration
            if not self.config.validate_config():
                self.logger.error("Configuration validation failed")
                return False
            
            # Send startup notification
            await self.notification_manager.send_startup_notification()
            
            # Start bots based on type
            if bot_type in ["sniper", "both"] and self.config.sniper_config.enabled:
                await self._start_sniper_bot()
            
            if bot_type in ["copy", "both"] and self.config.copy_config.enabled:
                await self._start_copy_bot()
            
            self.running = True
            
            # Main loop
            await self._main_loop()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting bot: {e}")
            await self.notification_manager.send_error_notification(str(e))
            return False
    
    async def _start_sniper_bot(self):
        """Start the sniper bot"""
        try:
            self.logger.info("Starting sniper bot...")
            self.sniper_bot = SniperBot(self.config)
            
            # Start sniper bot in background
            asyncio.create_task(self.sniper_bot.start())
            
            self.logger.info("Sniper bot started successfully")
            
        except Exception as e:
            self.logger.error(f"Error starting sniper bot: {e}")
            raise
    
    async def _start_copy_bot(self):
        """Start the copy bot"""
        try:
            self.logger.info("Starting copy bot...")
            self.copy_bot = CopyBot(self.config)
            
            # Start copy bot in background
            asyncio.create_task(self.copy_bot.start())
            
            self.logger.info("Copy bot started successfully")
            
        except Exception as e:
            self.logger.error(f"Error starting copy bot: {e}")
            raise
    
    async def _main_loop(self):
        """Main execution loop"""
        try:
            self.logger.info("Entering main loop...")
            
            while self.running and not self.shutdown_requested:
                try:
                    # Health check
                    await self._health_check()
                    
                    # Performance monitoring
                    await self._performance_monitoring()
                    
                    # Wait before next iteration
                    await asyncio.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(10)
            
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            await self.stop()
    
    async def _health_check(self):
        """Perform health checks on bots"""
        try:
            # Check sniper bot health
            if self.sniper_bot:
                sniper_healthy = await self.sniper_bot.health_check()
                if not sniper_healthy:
                    self.logger.warning("Sniper bot health check failed")
            
            # Check copy bot health
            if self.copy_bot:
                copy_healthy = await self.copy_bot.health_check()
                if not copy_healthy:
                    self.logger.warning("Copy bot health check failed")
            
        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
    
    async def _performance_monitoring(self):
        """Monitor bot performance"""
        try:
            # Get performance data
            performance_data = {}
            
            if self.sniper_bot:
                sniper_stats = self.sniper_bot.get_sniper_stats()
                performance_data["sniper_bot"] = sniper_stats
            
            if self.copy_bot:
                copy_stats = self.copy_bot.get_copy_stats()
                performance_data["copy_bot"] = copy_stats
            
            # Log performance data
            self.trading_logger.log_performance(performance_data)
            
            # Send performance notification (hourly)
            current_hour = time.localtime().tm_hour
            if current_hour % 1 == 0 and time.localtime().tm_min == 0:  # Every hour
                await self.notification_manager.send_performance_notification(performance_data)
            
        except Exception as e:
            self.logger.error(f"Error in performance monitoring: {e}")
    
    async def stop(self):
        """Stop all bots"""
        try:
            self.logger.info("Stopping trading bots...")
            self.running = False
            
            # Stop sniper bot
            if self.sniper_bot:
                await self.sniper_bot.stop()
            
            # Stop copy bot
            if self.copy_bot:
                await self.copy_bot.stop()
            
            # Send shutdown notification
            await self.notification_manager.send_shutdown_notification()
            
            self.logger.info("All bots stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping bots: {e}")
    
    def get_status(self) -> dict:
        """Get current bot status"""
        status = {
            "running": self.running,
            "shutdown_requested": self.shutdown_requested,
            "bots": {}
        }
        
        if self.sniper_bot:
            status["bots"]["sniper"] = self.sniper_bot.get_status()
        
        if self.copy_bot:
            status["bots"]["copy"] = self.copy_bot.get_status()
        
        return status

async def start_dashboard_server(port: int = 8000):
    """Start the web dashboard server"""
    try:
        await start_dashboard(port)
    except Exception as e:
        print(f"Error starting dashboard: {e}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Solana Trading Bot 2025")
    
    parser.add_argument(
        "--bot",
        choices=["sniper", "copy", "both"],
        default="both",
        help="Which bot to run (default: both)"
    )
    
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Start web dashboard"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Dashboard port (default: 8000)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    return parser.parse_args()

async def main():
    """Main function"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Create bot manager
        manager = TradingBotManager()
        
        # Start dashboard if requested
        if args.dashboard:
            dashboard_task = asyncio.create_task(start_dashboard_server(args.port))
        
        # Start trading bots
        success = await manager.start(args.bot)
        
        if not success:
            print("Failed to start trading bots")
            return 1
        
        # Wait for shutdown
        while not manager.shutdown_requested:
            await asyncio.sleep(1)
        
        # Stop bots
        await manager.stop()
        
        return 0
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        return 0
    except Exception as e:
        print(f"Error in main: {e}")
        return 1

if __name__ == "__main__":
    try:
        # Run the main function
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1) 