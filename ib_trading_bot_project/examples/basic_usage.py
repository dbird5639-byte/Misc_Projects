"""
Basic Usage Example

Demonstrates how to use the Interactive Brokers Trading Bot
with basic configuration and strategy execution.
"""

import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from src.connection.ib_connector import IBConnector
from src.trading.order_manager import OrderManager
from src.trading.position_manager import PositionManager
from src.trading.risk_manager import RiskManager
from src.strategies.momentum_strategy import MomentumStrategy
from src.strategies.mean_reversion_strategy import MeanReversionStrategy
from src.utils.logger import setup_logging

def main():
    """Main example function"""
    print("=" * 60)
    print("Interactive Brokers Trading Bot - Basic Usage Example")
    print("=" * 60)
    print("⚠️  DISCLAIMER: This is not financial advice.")
    print("   Trading involves significant risk and potential losses.")
    print("=" * 60)
    
    # Setup logging
    setup_logging("logs/basic_usage.log", "INFO")
    
    try:
        # Initialize components
        print("Initializing trading bot components...")
        
        # Create IB connector
        connector = IBConnector(
            host=settings.ib.host,
            port=settings.ib.port,
            client_id=settings.ib.client_id
        )
        
        # Create trading components
        order_manager = OrderManager(connector)
        position_manager = PositionManager(connector)
        risk_manager = RiskManager(position_manager, settings.risk)
        
        print("✓ Components initialized")
        
        # Connect to Interactive Brokers
        print("\nConnecting to Interactive Brokers...")
        if connector.connect():
            print("✓ Connected to Interactive Brokers")
        else:
            print("✗ Failed to connect to Interactive Brokers")
            print("  Make sure TWS or IB Gateway is running")
            return
        
        # Subscribe to market data
        print("\nSubscribing to market data...")
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        for symbol in symbols:
            connector.subscribe_market_data(symbol)
            print(f"  ✓ Subscribed to {symbol}")
        
        # Create strategies
        print("\nInitializing trading strategies...")
        
        # Momentum strategy
        momentum_config = {
            "symbols": symbols,
            "lookback_period": 20,
            "momentum_threshold": 0.02,
            "volume_threshold": 1.5,
            "stop_loss": 0.05,
            "take_profit": 0.10,
            "analysis_interval": 60
        }
        
        momentum_strategy = MomentumStrategy(
            order_manager, position_manager, risk_manager, momentum_config
        )
        
        # Mean reversion strategy
        mean_reversion_config = {
            "symbols": symbols,
            "lookback_period": 50,
            "std_dev_threshold": 2.0,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "stop_loss": 0.03,
            "take_profit": 0.08,
            "analysis_interval": 60
        }
        
        mean_reversion_strategy = MeanReversionStrategy(
            order_manager, position_manager, risk_manager, mean_reversion_config
        )
        
        print("✓ Strategies initialized")
        
        # Start strategies
        print("\nStarting trading strategies...")
        momentum_strategy.start()
        mean_reversion_strategy.start()
        print("✓ Strategies started")
        
        # Main monitoring loop
        print("\nStarting monitoring loop...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                # Update positions
                position_manager.update_positions()
                
                # Check risk limits
                violations = risk_manager.check_risk_limits()
                if violations:
                    print(f"⚠️  Risk violations detected: {violations}")
                
                # Get status
                momentum_status = momentum_strategy.get_status()
                mean_reversion_status = mean_reversion_strategy.get_status()
                
                # Print status every 5 minutes
                if int(time.time()) % 300 == 0:
                    print("\n" + "=" * 40)
                    print("TRADING BOT STATUS")
                    print("=" * 40)
                    
                    # Portfolio summary
                    portfolio_summary = position_manager.get_position_summary()
                    print(f"Total Positions: {portfolio_summary['total_positions']}")
                    print(f"Total Value: {portfolio_summary['total_market_value']:.2f}")
                    print(f"Total P&L: {portfolio_summary['total_pnl']:.2f}")
                    
                    # Strategy status
                    print(f"\nMomentum Strategy:")
                    print(f"  Running: {momentum_status['running']}")
                    print(f"  Signals Generated: {momentum_status['signals_generated']}")
                    print(f"  Trades Executed: {momentum_status['trades_executed']}")
                    
                    print(f"\nMean Reversion Strategy:")
                    print(f"  Running: {mean_reversion_status['running']}")
                    print(f"  Signals Generated: {mean_reversion_status['signals_generated']}")
                    print(f"  Trades Executed: {mean_reversion_status['trades_executed']}")
                    
                    # Risk metrics
                    risk_metrics = risk_manager.get_risk_metrics()
                    if risk_metrics:
                        print(f"\nRisk Metrics:")
                        print(f"  Portfolio Value: {risk_metrics.portfolio_value:.2f}")
                        print(f"  Total Exposure: {risk_metrics.total_exposure:.2f}")
                        print(f"  Position Concentration: {risk_metrics.position_concentration:.2f}%")
                    
                    print("=" * 40)
                
                time.sleep(10)  # Check every 10 seconds
                
        except KeyboardInterrupt:
            print("\n\nStopping trading bot...")
        
        # Stop strategies
        print("Stopping strategies...")
        momentum_strategy.stop()
        mean_reversion_strategy.stop()
        print("✓ Strategies stopped")
        
        # Disconnect
        print("Disconnecting from Interactive Brokers...")
        connector.disconnect()
        print("✓ Disconnected")
        
        print("\nTrading bot stopped successfully")
        
    except Exception as e:
        print(f"Error in main: {e}")
        return 1
    
    return 0

def example_manual_trade():
    """Example of manual trade execution"""
    print("\n" + "=" * 40)
    print("MANUAL TRADE EXAMPLE")
    print("=" * 40)
    
    try:
        # Initialize components
        connector = IBConnector()
        order_manager = OrderManager(connector)
        position_manager = PositionManager(connector)
        risk_manager = RiskManager(position_manager, settings.risk)
        
        # Connect
        if not connector.connect():
            print("Failed to connect")
            return
        
        # Subscribe to market data
        connector.subscribe_market_data("AAPL")
        time.sleep(2)  # Wait for data
        
        # Get current price
        market_data = connector.get_market_data("AAPL")
        if market_data:
            current_price = market_data.get("last_price", 0)
            print(f"Current AAPL price: ${current_price:.2f}")
            
            # Calculate position size
            quantity = risk_manager.calculate_position_size("AAPL", current_price, "fixed")
            print(f"Recommended position size: {quantity} shares")
            
            # Check if trade is allowed
            if risk_manager.should_trade("AAPL", "BUY", quantity, current_price):
                print("Trade approved by risk manager")
                
                # Place order (commented out for safety)
                # order_id = order_manager.place_market_order("AAPL", "BUY", quantity)
                # print(f"Order placed: {order_id}")
            else:
                print("Trade rejected by risk manager")
        
        # Disconnect
        connector.disconnect()
        
    except Exception as e:
        print(f"Error in manual trade example: {e}")

def example_strategy_analysis():
    """Example of strategy analysis"""
    print("\n" + "=" * 40)
    print("STRATEGY ANALYSIS EXAMPLE")
    print("=" * 40)
    
    try:
        # Initialize components
        connector = IBConnector()
        order_manager = OrderManager(connector)
        position_manager = PositionManager(connector)
        risk_manager = RiskManager(position_manager, settings.risk)
        
        # Create strategy
        config = {
            "symbols": ["AAPL", "GOOGL"],
            "lookback_period": 20,
            "momentum_threshold": 0.02
        }
        
        strategy = MomentumStrategy(order_manager, position_manager, risk_manager, config)
        
        # Simulate market data
        print("Simulating market analysis...")
        
        # Analyze market
        analysis = strategy.analyze_market()
        print(f"Analysis results: {analysis}")
        
        # Generate signals
        signals = strategy.generate_signals(analysis)
        print(f"Generated {len(signals)} signals")
        
        for signal in signals:
            print(f"  {signal.symbol}: {signal.action} {signal.quantity} shares at ${signal.price:.2f}")
        
    except Exception as e:
        print(f"Error in strategy analysis example: {e}")

if __name__ == "__main__":
    # Run main example
    exit_code = main()
    
    # Run additional examples
    if exit_code == 0:
        example_manual_trade()
        example_strategy_analysis()
    
    sys.exit(exit_code) 