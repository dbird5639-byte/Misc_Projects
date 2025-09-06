#!/usr/bin/env python3
"""
Systematic AI Trading Framework - Main Entry Point

Implements the RBI (Research, Backtest, Implement) methodology enhanced with local AI agents
for systematic strategy development and execution.
"""

import asyncio
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import Settings
from agents.model_factory import ModelFactory
from agents.research_agent import ResearchAgent
from agents.backtest_agent import BacktestAgent
from agents.package_agent import PackageAgent
from data.market_data import MarketDataManager
from backtesting.backtest_engine import BacktestEngine
from execution.portfolio_manager import PortfolioManager
from execution.risk_manager import RiskManager
from utils.logger import setup_logger
from utils.notifications import NotificationManager
from web.dashboard import start_dashboard


class SystematicTradingFramework:
    """
    Main framework class that orchestrates the RBI system with AI agents.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the systematic trading framework."""
        self.settings = Settings(config_path)
        self.logger = setup_logger("framework", self.settings.log_level)
        
        # Initialize components
        self.model_factory = ModelFactory(self.settings)
        self.research_agent = ResearchAgent(self.model_factory, self.settings)
        self.backtest_agent = BacktestAgent(self.model_factory, self.settings)
        self.package_agent = PackageAgent(self.settings)
        
        # Data and execution components
        self.market_data = MarketDataManager(self.settings)
        self.backtest_engine = BacktestEngine(self.settings)
        self.portfolio_manager = PortfolioManager(self.settings)
        self.risk_manager = RiskManager(self.settings)
        self.notifications = NotificationManager(self.settings)
        
        # State tracking
        self.is_running = False
        self.active_strategies = {}
        self.discovered_strategies = []
        self.backtest_results = {}
        
        self.logger.info("Systematic AI Trading Framework initialized")
    
    async def start(self):
        """Start the framework and begin the RBI cycle."""
        try:
            self.logger.info("Starting Systematic AI Trading Framework...")
            
            # Verify system health
            await self._verify_system_health()
            
            # Start background tasks
            self.is_running = True
            tasks = [
                self._research_cycle(),
                self._backtest_cycle(),
                self._implementation_cycle(),
                self._monitoring_cycle()
            ]
            
            # Start web dashboard
            if self.settings.enable_dashboard:
                dashboard_task = asyncio.create_task(self._start_dashboard())
                tasks.append(dashboard_task)
            
            # Run all cycles concurrently
            await asyncio.gather(*tasks)
            
        except KeyboardInterrupt:
            self.logger.info("Shutdown signal received")
        except Exception as e:
            self.logger.error(f"Framework error: {e}")
            await self.notifications.send_alert(f"Framework error: {e}")
        finally:
            await self.shutdown()
    
    async def _verify_system_health(self):
        """Verify all system components are healthy."""
        self.logger.info("Verifying system health...")
        
        # Check AI models
        models_ok = await self.model_factory.verify_models()
        if not models_ok:
            raise RuntimeError("AI models not available")
        
        # Check data sources
        data_ok = await self.market_data.verify_connections()
        if not data_ok:
            raise RuntimeError("Market data connections failed")
        
        # Check package dependencies
        packages_ok = await self.package_agent.verify_packages()
        if not packages_ok:
            raise RuntimeError("Required packages not available")
        
        self.logger.info("System health check passed")
    
    async def _research_cycle(self):
        """Continuous research cycle for discovering new strategies."""
        self.logger.info("Starting research cycle...")
        
        while self.is_running:
            try:
                # Discover new strategies
                new_strategies = await self.research_agent.discover_strategies()
                
                for strategy in new_strategies:
                    if strategy not in self.discovered_strategies:
                        self.discovered_strategies.append(strategy)
                        self.logger.info(f"Discovered new strategy: {strategy.name}")
                        
                        # Send notification
                        await self.notifications.send_alert(
                            f"New strategy discovered: {strategy.name} "
                            f"(Confidence: {strategy.confidence:.2f})"
                        )
                
                # Wait before next research cycle
                await asyncio.sleep(self.settings.research_interval)
                
            except Exception as e:
                self.logger.error(f"Research cycle error: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _backtest_cycle(self):
        """Continuous backtesting cycle for validating strategies."""
        self.logger.info("Starting backtesting cycle...")
        
        while self.is_running:
            try:
                # Get strategies to backtest
                strategies_to_test = [
                    s for s in self.discovered_strategies 
                    if s.name not in self.backtest_results
                ]
                
                for strategy in strategies_to_test:
                    self.logger.info(f"Backtesting strategy: {strategy.name}")
                    
                    # Run backtest
                    results = await self.backtest_agent.backtest_strategy(strategy)
                    self.backtest_results[strategy.name] = results
                    
                    # Evaluate results
                    if results.is_profitable and results.meets_criteria:
                        self.logger.info(f"Strategy {strategy.name} passed backtesting")
                        await self.notifications.send_alert(
                            f"Strategy {strategy.name} passed backtesting "
                            f"(Sharpe: {results.sharpe_ratio:.2f})"
                        )
                    else:
                        self.logger.info(f"Strategy {strategy.name} failed backtesting")
                
                # Wait before next backtest cycle
                await asyncio.sleep(self.settings.backtest_interval)
                
            except Exception as e:
                self.logger.error(f"Backtesting cycle error: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _implementation_cycle(self):
        """Implementation cycle for deploying successful strategies."""
        self.logger.info("Starting implementation cycle...")
        
        while self.is_running:
            try:
                # Find strategies ready for implementation
                ready_strategies = [
                    name for name, results in self.backtest_results.items()
                    if results.is_profitable and results.meets_criteria 
                    and name not in self.active_strategies
                ]
                
                for strategy_name in ready_strategies:
                    self.logger.info(f"Implementing strategy: {strategy_name}")
                    
                    # Get strategy and results
                    strategy = next(s for s in self.discovered_strategies if s.name == strategy_name)
                    results = self.backtest_results[strategy_name]
                    
                    # Deploy strategy
                    success = await self._deploy_strategy(strategy, results)
                    
                    if success:
                        self.active_strategies[strategy_name] = strategy
                        await self.notifications.send_alert(
                            f"Strategy {strategy_name} deployed successfully"
                        )
                    else:
                        self.logger.error(f"Failed to deploy strategy: {strategy_name}")
                
                # Wait before next implementation cycle
                await asyncio.sleep(self.settings.implementation_interval)
                
            except Exception as e:
                self.logger.error(f"Implementation cycle error: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _monitoring_cycle(self):
        """Continuous monitoring of active strategies."""
        self.logger.info("Starting monitoring cycle...")
        
        while self.is_running:
            try:
                # Monitor active strategies
                for strategy_name, strategy in self.active_strategies.items():
                    # Check strategy performance
                    performance = await self.portfolio_manager.get_strategy_performance(strategy_name)
                    
                    # Check risk metrics
                    risk_status = await self.risk_manager.check_strategy_risk(strategy_name)
                    
                    # Take action if needed
                    if risk_status.requires_action:
                        await self._handle_risk_event(strategy_name, risk_status)
                    
                    # Log performance
                    self.logger.info(
                        f"Strategy {strategy_name}: "
                        f"P&L={performance.pnl:.2f}, "
                        f"Risk={risk_status.risk_level}"
                    )
                
                # Wait before next monitoring cycle
                await asyncio.sleep(self.settings.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring cycle error: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _deploy_strategy(self, strategy, backtest_results):
        """Deploy a strategy to live trading."""
        try:
            # Configure strategy with backtest parameters
            strategy.configure(backtest_results.optimal_parameters)
            
            # Set up risk management
            await self.risk_manager.setup_strategy_risk(strategy.name, backtest_results)
            
            # Initialize portfolio position
            await self.portfolio_manager.initialize_strategy(strategy.name, strategy)
            
            # Start strategy execution
            await strategy.start()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Strategy deployment failed: {e}")
            return False
    
    async def _handle_risk_event(self, strategy_name, risk_status):
        """Handle risk events for active strategies."""
        if risk_status.risk_level == "HIGH":
            # Close positions
            await self.portfolio_manager.close_strategy_positions(strategy_name)
            await self.notifications.send_alert(f"Risk event: Closed positions for {strategy_name}")
            
        elif risk_status.risk_level == "MEDIUM":
            # Reduce position size
            await self.portfolio_manager.reduce_strategy_exposure(strategy_name, 0.5)
            await self.notifications.send_alert(f"Risk event: Reduced exposure for {strategy_name}")
    
    async def _start_dashboard(self):
        """Start the web dashboard."""
        try:
            await start_dashboard(self.settings.dashboard_port)
        except Exception as e:
            self.logger.error(f"Dashboard error: {e}")
    
    async def shutdown(self):
        """Gracefully shutdown the framework."""
        self.logger.info("Shutting down framework...")
        
        self.is_running = False
        
        # Close active strategies
        for strategy_name in list(self.active_strategies.keys()):
            await self.portfolio_manager.close_strategy_positions(strategy_name)
        
        # Close connections
        await self.market_data.close()
        await self.portfolio_manager.close()
        
        self.logger.info("Framework shutdown complete")
    
    def get_status(self) -> Dict:
        """Get current framework status."""
        return {
            "is_running": self.is_running,
            "active_strategies": len(self.active_strategies),
            "discovered_strategies": len(self.discovered_strategies),
            "backtest_results": len(self.backtest_results),
            "strategies": {
                "active": list(self.active_strategies.keys()),
                "discovered": [s.name for s in self.discovered_strategies],
                "backtested": list(self.backtest_results.keys())
            }
        }


async def main():
    """Main entry point."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Systematic AI Trading Framework")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--research-only", action="store_true", help="Run only research cycle")
    parser.add_argument("--backtest-only", action="store_true", help="Run only backtesting")
    parser.add_argument("--deploy", action="store_true", help="Deploy strategies immediately")
    args = parser.parse_args()
    
    # Initialize framework
    framework = SystematicTradingFramework(args.config)
    
    # Set up signal handlers
    def signal_handler(signum, frame):
        framework.logger.info(f"Received signal {signum}")
        asyncio.create_task(framework.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run framework
    if args.research_only:
        await framework._research_cycle()
    elif args.backtest_only:
        await framework._backtest_cycle()
    elif args.deploy:
        # Deploy specific strategies
        pass
    else:
        await framework.start()


if __name__ == "__main__":
    asyncio.run(main()) 