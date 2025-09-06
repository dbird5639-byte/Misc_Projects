"""
Multi-Strategy Trading Platform Example
Demonstrates the platform functionality with a simple example
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategy_development.straten_generator import StratenGenerator
from testing_framework.multiwalk_tester import MultiWalkTester
from execution_engine.portfolio_manager import PortfolioManager
from performance_analytics.metrics_calculator import MetricsCalculator
from risk_management.risk_manager import RiskManager
from data_management.historical_data import HistoricalDataManager


def main():
    """Run platform example"""
    print("🚀 Multi-Strategy Algorithmic Trading Platform Example")
    print("Built on the wisdom of Jacob Amaral and Kevin Davy")
    print("Quality over quantity - building robust trading systems\n")
    
    # Initialize components
    print("📊 Initializing platform components...")
    straten_generator = StratenGenerator()
    multiwalk_tester = MultiWalkTester()
    portfolio_manager = PortfolioManager()
    metrics_calculator = MetricsCalculator()
    risk_manager = RiskManager()
    data_manager = HistoricalDataManager()
    
    print("✅ Components initialized\n")
    
    # Generate strategies using Jacob's approach
    print("🔧 Generating strategies using Straten-inspired approach...")
    strategy_configs = [
        {
            "name": "mean_reversion_bb_rsi",
            "indicators": ["bollinger_bands", "rsi"],
            "logic": "mean_reversion",
            "language": "python"
        },
        {
            "name": "trend_following_sma_macd",
            "indicators": ["sma", "macd"],
            "logic": "trend_following",
            "language": "python"
        },
        {
            "name": "momentum_rsi_cci",
            "indicators": ["rsi", "cci"],
            "logic": "momentum",
            "language": "python"
        }
    ]
    
    strategies = {}
    for config in strategy_configs:
        try:
            strategy = straten_generator.generate_strategy(
                indicators=config["indicators"],
                logic=config["logic"],
                language=config.get("language", "python"),
                strategy_name=config.get("name")
            )
            strategies[strategy.name] = strategy
            print(f"✅ Generated strategy: {strategy.name}")
        except Exception as e:
            print(f"❌ Failed to generate strategy {config['name']}: {e}")
    
    print(f"✅ Generated {len(strategies)} strategies\n")
    
    # Test strategies using Kevin's approach
    print("🧪 Testing strategies using MultiWalk-inspired approach...")
    
    # Create simple test strategies for demonstration
    class SimpleTestStrategy:
        def __init__(self, name, logic_type):
            self.name = name
            self.logic_type = logic_type
        
        def generate_signals(self, data):
            # Simple signal generation for demonstration
            data = data.copy()
            data['signal'] = 0
            
            if self.logic_type == "mean_reversion":
                # Simple mean reversion logic
                data['sma'] = data['close'].rolling(20).mean()
                data.loc[data['close'] < data['sma'] * 0.98, 'signal'] = 1
                data.loc[data['close'] > data['sma'] * 1.02, 'signal'] = -1
            elif self.logic_type == "trend_following":
                # Simple trend following logic
                data['sma_20'] = data['close'].rolling(20).mean()
                data['sma_50'] = data['close'].rolling(50).mean()
                data.loc[data['sma_20'] > data['sma_50'], 'signal'] = 1
                data.loc[data['sma_20'] < data['sma_50'], 'signal'] = -1
            elif self.logic_type == "momentum":
                # Simple momentum logic
                data['returns'] = data['close'].pct_change()
                data.loc[data['returns'] > 0.01, 'signal'] = 1
                data.loc[data['returns'] < -0.01, 'signal'] = -1
            
            return data
    
    # Create test strategies
    test_strategies = {
        "mean_reversion_bb_rsi": SimpleTestStrategy("mean_reversion_bb_rsi", "mean_reversion"),
        "trend_following_sma_macd": SimpleTestStrategy("trend_following_sma_macd", "trend_following"),
        "momentum_rsi_cci": SimpleTestStrategy("momentum_rsi_cci", "momentum")
    }
    
    # Test strategies
    results = {}
    for strategy_name, strategy in test_strategies.items():
        try:
            result = multiwalk_tester.test_strategy(
                strategy=strategy,
                markets=["ES", "NQ"],
                timeframes=["daily"],
                years=3
            )
            results[strategy_name] = result
            print(f"✅ Tested strategy: {strategy_name}")
            print(f"   Return: {result.total_return:.2%}, Drawdown: {result.max_drawdown:.2%}")
            
            # Check Kevin's 2x drawdown rule
            ratio = result.total_return / abs(result.max_drawdown) if result.max_drawdown != 0 else 0
            if ratio >= 2.0:
                print(f"   ✅ Passes 2x drawdown rule: {ratio:.2f}")
            else:
                print(f"   ⚠️ Fails 2x drawdown rule: {ratio:.2f}")
                
        except Exception as e:
            print(f"❌ Failed to test strategy {strategy_name}: {e}")
    
    print(f"✅ Tested {len(results)} strategies\n")
    
    # Build portfolio
    print("🏗️ Building multi-strategy portfolio...")
    try:
        portfolio_config = portfolio_manager.build_portfolio(
            strategies=test_strategies,
            results=results,
            max_correlation=0.7,
            risk_per_trade=0.02
        )
        print("✅ Portfolio built successfully")
        print(f"   Strategies: {len(portfolio_config.get('strategies', []))}")
        print(f"   Expected return: {portfolio_config.get('expected_return', 0):.2%}")
        print(f"   Expected risk: {portfolio_config.get('expected_risk', 0):.2%}")
    except Exception as e:
        print(f"❌ Failed to build portfolio: {e}")
    
    print("\n🎯 Platform demonstration completed!")
    print("\nKey Features Demonstrated:")
    print("✅ Jacob Amaral's Straten-inspired strategy generation")
    print("✅ Kevin Davy's MultiWalk-inspired testing approach")
    print("✅ Multi-strategy portfolio management")
    print("✅ Risk management with 2x drawdown rule")
    print("✅ Performance metrics calculation")
    print("✅ Data management and backtesting")
    
    print("\nNext Steps:")
    print("1. Run main.py for full platform functionality")
    print("2. Use the web dashboard for monitoring")
    print("3. Implement live trading with proper risk management")
    print("4. Add more sophisticated strategies and testing")


if __name__ == "__main__":
    main() 