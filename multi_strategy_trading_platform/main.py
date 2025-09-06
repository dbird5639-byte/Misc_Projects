"""
Multi-Strategy Algorithmic Trading Platform
Main Entry Point

Built on the wisdom of Jacob Amaral and Kevin Davy
Quality over quantity - building robust trading systems
"""

import asyncio
import logging
from typing import Dict, List, Optional
from pathlib import Path

# Import platform components
from strategy_development.straten_generator import StratenGenerator
from testing_framework.multiwalk_tester import MultiWalkTester
from execution_engine.portfolio_manager import PortfolioManager
from performance_analytics.metrics_calculator import MetricsCalculator
from risk_management.risk_manager import RiskManager
from data_management.historical_data import HistoricalDataManager
from web.dashboard import create_dashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiStrategyTradingPlatform:
    """
    Main platform class that orchestrates all components
    
    Implements the methodologies of:
    - Jacob Amaral: Straten tool for strategy generation
    - Kevin Davy: MultiWalk for strategy testing
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the trading platform"""
        self.config = self._load_config(config_path)
        
        # Initialize core components
        self.straten_generator = StratenGenerator()
        self.multiwalk_tester = MultiWalkTester()
        self.portfolio_manager = PortfolioManager()
        self.metrics_calculator = MetricsCalculator()
        self.risk_manager = RiskManager()
        self.data_manager = HistoricalDataManager()
        
        # Platform state
        self.strategies = {}
        self.test_results = {}
        self.portfolio_performance = {}
        
        logger.info("Multi-Strategy Trading Platform initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load platform configuration"""
        # Default configuration
        config = {
            "data_dir": "data",
            "strategies_dir": "strategies",
            "results_dir": "results",
            "max_drawdown": 0.15,  # Kevin's 2x drawdown rule
            "min_return_drawdown_ratio": 2.0,
            "default_markets": ["ES", "NQ", "YM", "CL", "GC"],
            "default_timeframes": ["daily", "4h", "1h"],
            "risk_per_trade": 0.02,  # 2% risk per trade
            "max_correlation": 0.7,  # Maximum strategy correlation
            "backtest_years": 10,
            "walk_forward_periods": 12
        }
        
        # Load from file if provided
        if config_path and Path(config_path).exists():
            import json
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
        
        return config
    
    def generate_strategies(self, strategy_configs: List[Dict]) -> Dict[str, any]:
        """
        Generate strategies using Jacob's Straten-inspired approach
        
        Args:
            strategy_configs: List of strategy configurations
            
        Returns:
            Dictionary of generated strategies
        """
        logger.info("Generating strategies using Straten-inspired approach")
        
        strategies = {}
        for config in strategy_configs:
            try:
                strategy = self.straten_generator.generate_strategy(
                    indicators=config["indicators"],
                    logic=config["logic"],
                    language=config.get("language", "python"),
                    strategy_name=config.get("name"),
                    custom_parameters=config.get("parameters")
                )
                
                strategies[strategy.name] = strategy
                logger.info(f"Generated strategy: {strategy.name}")
                
                # Save strategy files
                self.straten_generator.save_strategy(
                    strategy, 
                    self.config["strategies_dir"]
                )
                
            except Exception as e:
                logger.error(f"Failed to generate strategy {config}: {e}")
        
        self.strategies.update(strategies)
        return strategies
    
    def test_strategies(self, strategies: Dict[str, any], markets: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Test strategies using Kevin's MultiWalk-inspired approach
        
        Args:
            strategies: Dictionary of strategies to test
            markets: Markets to test on (defaults to config)
            
        Returns:
            Dictionary of test results
        """
        logger.info("Testing strategies using MultiWalk-inspired approach")
        
        if markets is None:
            markets = self.config["default_markets"]
        
        results = {}
        for strategy_name, strategy in strategies.items():
            try:
                # Test across multiple markets and timeframes
                test_result = self.multiwalk_tester.test_strategy(
                    strategy=strategy,
                    markets=markets,
                    timeframes=self.config["default_timeframes"],
                    years=self.config["backtest_years"],
                    walk_forward_periods=self.config["walk_forward_periods"]
                )
                
                results[strategy_name] = test_result
                logger.info(f"Tested strategy: {strategy_name}")
                
                # Apply Kevin's 2x drawdown rule
                if test_result.total_return / test_result.max_drawdown < self.config["min_return_drawdown_ratio"]:
                    logger.warning(f"Strategy {strategy_name} fails 2x drawdown rule")
                
            except Exception as e:
                logger.error(f"Failed to test strategy {strategy_name}: {e}")
        
        self.test_results.update(results)
        return results
    
    def build_portfolio(self, strategies: Dict[str, any], results: Dict[str, any]) -> Dict[str, any]:
        """
        Build multi-strategy portfolio with proper risk management
        
        Args:
            strategies: Dictionary of strategies
            results: Dictionary of test results
            
        Returns:
            Portfolio configuration
        """
        logger.info("Building multi-strategy portfolio")
        
        # Filter strategies that meet performance criteria
        qualified_strategies = {}
        for strategy_name, result in results.items():
            if (result.total_return / result.max_drawdown >= self.config["min_return_drawdown_ratio"] and
                result.max_drawdown <= self.config["max_drawdown"]):
                qualified_strategies[strategy_name] = strategies[strategy_name]
        
        logger.info(f"Qualified strategies: {len(qualified_strategies)}")
        
        # Calculate optimal allocations
        portfolio_config = self.portfolio_manager.build_portfolio(
            strategies=qualified_strategies,
            results=results,
            max_correlation=self.config["max_correlation"],
            risk_per_trade=self.config["risk_per_trade"]
        )
        
        return portfolio_config
    
    def run_backtest(self, portfolio_config: Dict[str, any]) -> Dict[str, any]:
        """
        Run comprehensive backtest of the portfolio
        
        Args:
            portfolio_config: Portfolio configuration
            
        Returns:
            Backtest results
        """
        logger.info("Running portfolio backtest")
        
        # Load historical data
        data = self.data_manager.load_data(
            symbols=portfolio_config["symbols"],
            start_date=portfolio_config["start_date"],
            end_date=portfolio_config["end_date"]
        )
        
        # Run backtest
        backtest_results = self.portfolio_manager.run_backtest(
            portfolio_config=portfolio_config,
            data=data
        )
        
        # Calculate performance metrics
        performance = self.metrics_calculator.calculate_portfolio_metrics(
            equity_curve=backtest_results["equity_curve"],
            trades=backtest_results["trades"],
            positions=backtest_results["positions"]
        )
        
        self.portfolio_performance = performance
        return performance
    
    def start_live_trading(self, portfolio_config: Dict[str, any]) -> bool:
        """
        Start live trading with the portfolio
        
        Args:
            portfolio_config: Portfolio configuration
            
        Returns:
            Success status
        """
        logger.info("Starting live trading")
        
        try:
            # Initialize risk management
            self.risk_manager.initialize(
                max_drawdown=self.config["max_drawdown"],
                risk_per_trade=self.config["risk_per_trade"],
                max_correlation=self.config["max_correlation"]
            )
            
            # Start portfolio manager
            success = self.portfolio_manager.start_live_trading(
                portfolio_config=portfolio_config,
                risk_manager=self.risk_manager
            )
            
            if success:
                logger.info("Live trading started successfully")
            else:
                logger.error("Failed to start live trading")
            
            return success
            
        except Exception as e:
            logger.error(f"Error starting live trading: {e}")
            return False
    
    def create_dashboard(self, port: int = 8000) -> None:
        """
        Create web dashboard for monitoring
        
        Args:
            port: Dashboard port
        """
        logger.info(f"Creating web dashboard on port {port}")
        
        dashboard = create_dashboard(
            strategies=self.strategies,
            test_results=self.test_results,
            portfolio_performance=self.portfolio_performance,
            config=self.config
        )
        
        # Start dashboard
        import uvicorn
        uvicorn.run(dashboard, host="0.0.0.0", port=port)
    
    def generate_report(self, output_path: str = "trading_report.html") -> str:
        """
        Generate comprehensive trading report
        
        Args:
            output_path: Output file path
            
        Returns:
            Report file path
        """
        logger.info("Generating trading report")
        
        report_content = self._generate_report_content()
        
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Report saved to {output_path}")
        return output_path
    
    def _generate_report_content(self) -> str:
        """Generate HTML report content"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Multi-Strategy Trading Platform Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background: #e8f4f8; border-radius: 3px; }
                .strategy { margin: 10px 0; padding: 10px; background: #f9f9f9; border-left: 4px solid #007acc; }
                .warning { color: #ff6b35; }
                .success { color: #28a745; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Multi-Strategy Trading Platform Report</h1>
                <p>Built on the wisdom of Jacob Amaral and Kevin Davy</p>
                <p><strong>Quality over quantity - building robust trading systems</strong></p>
            </div>
        """
        
        # Add strategies section
        if self.strategies:
            html += '<div class="section"><h2>Generated Strategies</h2>'
            for name, strategy in self.strategies.items():
                html += f'''
                <div class="strategy">
                    <h3>{strategy.name}</h3>
                    <p><strong>Type:</strong> {strategy.type.value}</p>
                    <p><strong>Description:</strong> {strategy.description}</p>
                    <p><strong>Indicators:</strong> {", ".join([ind.name for ind in strategy.indicators])}</p>
                    <p><strong>Logic:</strong> {strategy.logic}</p>
                </div>
                '''
            html += '</div>'
        
        # Add test results section
        if self.test_results:
            html += '<div class="section"><h2>Strategy Test Results</h2>'
            for name, result in self.test_results.items():
                ratio = result.total_return / result.max_drawdown if result.max_drawdown > 0 else 0
                status_class = "success" if ratio >= 2.0 else "warning"
                html += f'''
                <div class="strategy">
                    <h3>{name}</h3>
                    <div class="metric"><strong>Total Return:</strong> {result.total_return:.2%}</div>
                    <div class="metric"><strong>Max Drawdown:</strong> {result.max_drawdown:.2%}</div>
                    <div class="metric"><strong>Sharpe Ratio:</strong> {result.sharpe_ratio:.2f}</div>
                    <div class="metric {status_class}"><strong>Return/Drawdown:</strong> {ratio:.2f}</div>
                </div>
                '''
            html += '</div>'
        
        # Add portfolio performance section
        if self.portfolio_performance:
            html += '<div class="section"><h2>Portfolio Performance</h2>'
            perf = self.portfolio_performance
            html += f'''
            <div class="metric"><strong>Total Return:</strong> {perf.get('total_return', 0):.2%}</div>
            <div class="metric"><strong>Max Drawdown:</strong> {perf.get('max_drawdown', 0):.2%}</div>
            <div class="metric"><strong>Sharpe Ratio:</strong> {perf.get('sharpe_ratio', 0):.2f}</div>
            <div class="metric"><strong>Win Rate:</strong> {perf.get('win_rate', 0):.2%}</div>
            <div class="metric"><strong>Profit Factor:</strong> {perf.get('profit_factor', 0):.2f}</div>
            '''
            html += '</div>'
        
        html += """
        </body>
        </html>
        """
        
        return html


def main():
    """Main entry point with example usage"""
    print("🚀 Multi-Strategy Algorithmic Trading Platform")
    print("Built on the wisdom of Jacob Amaral and Kevin Davy")
    print("Quality over quantity - building robust trading systems\n")
    
    # Initialize platform
    platform = MultiStrategyTradingPlatform()
    
    # Example strategy configurations
    strategy_configs = [
        {
            "name": "mean_reversion_bb_rsi",
            "indicators": ["bollinger_bands", "rsi"],
            "logic": "mean_reversion",
            "language": "python",
            "parameters": {
                "bollinger_bands": {"period": 20, "std_dev": 2},
                "rsi": {"period": 14}
            }
        },
        {
            "name": "trend_following_sma_macd",
            "indicators": ["sma", "macd"],
            "logic": "trend_following",
            "language": "python",
            "parameters": {
                "sma": {"period": 50},
                "macd": {"fast": 12, "slow": 26, "signal": 9}
            }
        },
        {
            "name": "momentum_rsi_cci",
            "indicators": ["rsi", "cci"],
            "logic": "momentum",
            "language": "python",
            "parameters": {
                "rsi": {"period": 14},
                "cci": {"period": 20}
            }
        }
    ]
    
    try:
        # Generate strategies using Jacob's approach
        print("📊 Generating strategies using Straten-inspired approach...")
        strategies = platform.generate_strategies(strategy_configs)
        print(f"✅ Generated {len(strategies)} strategies\n")
        
        # Test strategies using Kevin's approach
        print("🧪 Testing strategies using MultiWalk-inspired approach...")
        results = platform.test_strategies(strategies)
        print(f"✅ Tested {len(results)} strategies\n")
        
        # Build portfolio
        print("🏗️ Building multi-strategy portfolio...")
        portfolio_config = platform.build_portfolio(strategies, results)
        print("✅ Portfolio built\n")
        
        # Run backtest
        print("📈 Running portfolio backtest...")
        performance = platform.run_backtest(portfolio_config)
        print("✅ Backtest completed\n")
        
        # Generate report
        print("📋 Generating trading report...")
        report_path = platform.generate_report()
        print(f"✅ Report saved to {report_path}\n")
        
        # Show performance summary
        print("📊 Performance Summary:")
        print(f"Total Return: {performance.get('total_return', 0):.2%}")
        print(f"Max Drawdown: {performance.get('max_drawdown', 0):.2%}")
        print(f"Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
        print(f"Win Rate: {performance.get('win_rate', 0):.2%}")
        print(f"Profit Factor: {performance.get('profit_factor', 0):.2f}")
        
        # Check Kevin's 2x drawdown rule
        ratio = performance.get('total_return', 0) / performance.get('max_drawdown', 1) if performance.get('max_drawdown', 0) > 0 else 0
        if ratio >= 2.0:
            print(f"✅ Passes Kevin's 2x drawdown rule: {ratio:.2f}")
        else:
            print(f"⚠️ Fails Kevin's 2x drawdown rule: {ratio:.2f}")
        
        print("\n🎯 Platform ready for live trading!")
        print("Use platform.start_live_trading(portfolio_config) to begin")
        print("Use platform.create_dashboard() to start web interface")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 