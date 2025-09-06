"""
Harvard Algorithmic Trading System - Main Entry Point

This is the main entry point for the RBI (Research, Backtest, Implement) 
algorithmic trading system.
"""

import sys
import os
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import config
from research.strategy_research import StrategyResearcher
from backtesting.backtest_engine import BacktestEngine, MomentumStrategy, MeanReversionStrategy
from implementation.live_trading_system import LiveTradingSystem, LiveMomentumStrategy, LiveMeanReversionStrategy
from strategies.strategy_factory import StrategyFactory
from risk_management.risk_manager import RiskManager, RiskLevel
from data.market_data_manager import MarketDataManager, YahooFinanceSource
from ai_tools.ai_assistant import AIAssistant

class HarvardAlgoTradingSystem:
    """Main system class implementing the RBI methodology"""
    
    def __init__(self):
        self.researcher = StrategyResearcher()
        self.backtest_engine = BacktestEngine()
        self.strategy_factory = StrategyFactory()
        self.risk_manager = RiskManager()
        self.data_manager = MarketDataManager()
        self.ai_assistant = AIAssistant()
        
        # Initialize data sources
        self.data_manager.add_data_source(YahooFinanceSource())
        
        print("=" * 60)
        print("Harvard Algorithmic Trading System")
        print("RBI Methodology Implementation")
        print("=" * 60)
    
    def run_research_phase(self, symbols: Optional[List] = None) -> Dict[str, Any]:
        """Execute Research phase of RBI system"""
        print("\n🔍 PHASE 1: RESEARCH")
        print("=" * 40)
        
        if symbols is None:
            symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY"]
        
        # Analyze strategies
        print("Analyzing trading strategies...")
        momentum_analysis = self.researcher.analyze_momentum_strategy()
        mean_reversion_analysis = self.researcher.analyze_mean_reversion_strategy()
        
        # Analyze market inefficiencies
        print("Identifying market inefficiencies...")
        inefficiencies = self.researcher.analyze_market_inefficiencies()
        
        # Research market behavior for symbols
        print("Researching market behavior...")
        market_research = {}
        for symbol in symbols:
            analysis = self.researcher.research_market_behavior(symbol)
            if analysis:
                market_research[symbol] = analysis
        
        # Generate research report
        print("Generating research report...")
        self.researcher.save_research_report("research_report.md")
        
        research_results = {
            "strategies_analyzed": [momentum_analysis, mean_reversion_analysis],
            "market_inefficiencies": inefficiencies,
            "market_research": market_research,
            "report_generated": "research_report.md"
        }
        
        print("✅ Research phase completed")
        return research_results
    
    def run_backtest_phase(self, symbols: Optional[List] = None) -> Dict[str, Any]:
        """Execute Backtest phase of RBI system"""
        print("\n📊 PHASE 2: BACKTEST")
        print("=" * 40)
        
        if symbols is None:
            symbols = ["AAPL", "GOOGL", "MSFT"]
        
        # Get historical data
        print("Downloading historical data...")
        market_data = {}
        for symbol in symbols:
            data = self.data_manager.get_historical_data(
                symbol, "2022-01-01", "2023-12-31"
            )
            if not data.empty:
                market_data[symbol] = data
                print(f"  {symbol}: {len(data)} data points")
        
        if not market_data:
            print("❌ No market data available for backtesting")
            return {}
        
        # Create strategies
        print("Creating strategies for backtesting...")
        momentum_config = self.strategy_factory.get_strategy_config_template("momentum")
        mean_reversion_config = self.strategy_factory.get_strategy_config_template("mean_reversion")
        
        momentum_strategy = MomentumStrategy(
            lookback_period=20, threshold=0.02
        )
        mean_reversion_strategy = MeanReversionStrategy(
            lookback_period=50, std_dev_threshold=2.0
        )
        
        # Run backtests
        print("Running backtests...")
        results = []
        
        for symbol in symbols:
            if symbol in market_data:
                print(f"  Backtesting {symbol}...")
                
                # Test momentum strategy
                momentum_result = self.backtest_engine.run_backtest(
                    momentum_strategy, market_data[symbol], symbol
                )
                results.append(momentum_result)
                
                # Test mean reversion strategy
                mean_reversion_result = self.backtest_engine.run_backtest(
                    mean_reversion_strategy, market_data[symbol], symbol
                )
                results.append(mean_reversion_result)
        
        # Generate backtest report
        print("Generating backtest report...")
        report = self.backtest_engine.generate_backtest_report(results)
        
        with open("backtest_report.md", "w") as f:
            f.write(report)
        
        backtest_results = {
            "symbols_tested": symbols,
            "strategies_tested": ["momentum", "mean_reversion"],
            "results": results,
            "report_generated": "backtest_report.md"
        }
        
        print("✅ Backtest phase completed")
        return backtest_results
    
    def run_implementation_phase(self, symbols: Optional[List] = None) -> Dict[str, Any]:
        """Execute Implement phase of RBI system"""
        print("\n🚀 PHASE 3: IMPLEMENT")
        print("=" * 40)
        
        if symbols is None:
            symbols = ["AAPL", "GOOGL", "MSFT"]
        
        # Validate configuration
        if not config.validate_config():
            print("❌ Configuration validation failed")
            print("Please set up your API keys in config/settings.py")
            return {}
        
        # Initialize live trading system
        print("Initializing live trading system...")
        trading_system = LiveTradingSystem(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
            config.ALPACA_BASE_URL
        )
        
        # Add strategies
        print("Adding trading strategies...")
        momentum_strategy = LiveMomentumStrategy(symbols, 20, 0.02)
        mean_reversion_strategy = LiveMeanReversionStrategy(symbols, 50, 2.0)
        
        trading_system.add_strategy(momentum_strategy)
        trading_system.add_strategy(mean_reversion_strategy)
        
        # Get system status
        status = trading_system.get_system_status()
        print("System Status:")
        print(f"  Account Value: ${status['account'].get('portfolio_value', 0):,.2f}")
        print(f"  Cash: ${status['account'].get('cash', 0):,.2f}")
        print(f"  Strategies: {status['strategies']}")
        
        implementation_results = {
            "system_initialized": True,
            "strategies_added": status['strategies'],
            "account_status": status['account'],
            "ready_for_trading": True
        }
        
        print("✅ Implementation phase completed")
        print("⚠️  System is ready for live trading")
        print("⚠️  Use with caution - this involves real money")
        
        return implementation_results
    
    def run_full_rbi_cycle(self, symbols: Optional[List] = None) -> Dict[str, Any]:
        """Run complete RBI cycle"""
        print("\n🔄 RUNNING COMPLETE RBI CYCLE")
        print("=" * 50)
        
        results = {}
        
        # Phase 1: Research
        research_results = self.run_research_phase(symbols)
        results["research"] = research_results
        
        # Phase 2: Backtest
        backtest_results = self.run_backtest_phase(symbols)
        results["backtest"] = backtest_results
        
        # Phase 3: Implement
        implementation_results = self.run_implementation_phase(symbols)
        results["implementation"] = implementation_results
        
        # Generate final report
        self._generate_final_report(results)
        
        print("\n🎉 RBI CYCLE COMPLETED SUCCESSFULLY!")
        print("Check the generated reports for detailed results.")
        
        return results
    
    def _generate_final_report(self, results: Dict[str, Any]):
        """Generate final comprehensive report"""
        report = f"""# Harvard Algorithmic Trading System - Final Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report summarizes the complete RBI (Research, Backtest, Implement) cycle
for the Harvard Algorithmic Trading System.

## Research Phase Results

- **Strategies Analyzed**: {len(results.get('research', {}).get('strategies_analyzed', []))}
- **Market Inefficiencies Identified**: {len(results.get('research', {}).get('market_inefficiencies', []))}
- **Symbols Researched**: {len(results.get('research', {}).get('market_research', {}))}

## Backtest Phase Results

- **Symbols Tested**: {len(results.get('backtest', {}).get('symbols_tested', []))}
- **Strategies Tested**: {len(results.get('backtest', {}).get('strategies_tested', []))}
- **Total Backtests**: {len(results.get('backtest', {}).get('results', []))}

## Implementation Phase Results

- **System Initialized**: {results.get('implementation', {}).get('system_initialized', False)}
- **Strategies Added**: {results.get('implementation', {}).get('strategies_added', [])}
- **Ready for Trading**: {results.get('implementation', {}).get('ready_for_trading', False)}

## Recommendations

1. **Start Small**: Begin with paper trading to validate strategies
2. **Monitor Closely**: Keep close watch on system performance
3. **Risk Management**: Always use proper risk controls
4. **Continuous Improvement**: Regularly review and optimize strategies

## Important Disclaimers

- This is not financial advice
- Past performance does not guarantee future results
- Trading involves significant risk of loss
- Always test thoroughly before using real money

## Next Steps

1. Review the detailed reports in research_report.md and backtest_report.md
2. Set up proper monitoring and alerting systems
3. Start with small position sizes
4. Monitor performance and adjust as needed
"""
        
        with open("final_report.md", "w") as f:
            f.write(report)
        
        print("Final report generated: final_report.md")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Harvard Algorithmic Trading System")
    parser.add_argument("--phase", choices=["research", "backtest", "implement", "full"], 
                       default="full", help="RBI phase to run")
    parser.add_argument("--symbols", nargs="+", 
                       default=["AAPL", "GOOGL", "MSFT"], 
                       help="Trading symbols to analyze")
    parser.add_argument("--config", type=str, help="Configuration file path")
    
    args = parser.parse_args()
    
    # Initialize system
    system = HarvardAlgoTradingSystem()
    
    # Run specified phase
    if args.phase == "research":
        results = system.run_research_phase(args.symbols)
    elif args.phase == "backtest":
        results = system.run_backtest_phase(args.symbols)
    elif args.phase == "implement":
        results = system.run_implementation_phase(args.symbols)
    else:  # full
        results = system.run_full_rbi_cycle(args.symbols)
    
    print(f"\n✅ {args.phase.upper()} phase completed successfully!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 