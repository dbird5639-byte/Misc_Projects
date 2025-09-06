"""
Main entry point for AI Market Maker & Liquidation Monitor 2025
Enhanced version with comprehensive AI agents and advanced features
"""

import asyncio
import argparse
import signal
import sys
from typing import Dict, Any, List
import logging
from datetime import datetime, timedelta

from config.settings import get_settings, create_default_configs
from data.position_monitor import create_position_monitor
from data.liquidation_tracker import create_liquidation_tracker
from agents.liquidation_predictor import create_liquidation_predictor
from agents.position_analyzer import create_position_analyzer
from agents.market_maker_tracker import create_market_maker_tracker
from agents.risk_manager import create_risk_manager
from agents.signal_generator import create_signal_generator
from agents.ensemble_predictor import create_ensemble_predictor
from utils.logger import setup_logging
from utils.notifications import NotificationManager
from web.dashboard import start_dashboard

# Global variables for graceful shutdown
components = {
    "position_monitor": None,
    "liquidation_tracker": None,
    "liquidation_predictor": None,
    "position_analyzer": None,
    "market_maker_tracker": None,
    "risk_manager": None,
    "signal_generator": None,
    "ensemble_predictor": None,
    "notification_manager": None,
    "dashboard_task": None
}

# System status
system_status = {
    "start_time": None,
    "is_running": False,
    "active_components": 0,
    "errors": 0,
    "warnings": 0
}


async def main():
    """Main application entry point"""
    global components, system_status
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI Market Maker & Liquidation Monitor 2025")
    parser.add_argument("--mode", choices=["monitor", "trading", "analysis", "backtest", "dashboard", "all"], 
                       default="all", help="Operation mode")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-notifications", action="store_true", help="Disable notifications")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable web dashboard")
    parser.add_argument("--backtest-config", type=str, help="Path to backtest configuration")
    
    args = parser.parse_args()
    
    try:
        # Create default configs if they don't exist
        create_default_configs()
        
        # Load settings
        settings = get_settings()
        settings.system.debug_mode = args.debug
        
        # Setup logging
        setup_logging(settings.system.log_level)
        logger = logging.getLogger(__name__)
        
        logger.info("🚀 Starting AI Market Maker & Liquidation Monitor 2025")
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Debug: {args.debug}")
        logger.info(f"Version: {settings.version}")
        
        # Initialize system status
        system_status["start_time"] = datetime.now()
        system_status["is_running"] = True
        
        # Initialize notification manager
        if not args.no_notifications:
            components["notification_manager"] = NotificationManager(settings.notifications.dict())
            await components["notification_manager"].initialize()
            await components["notification_manager"].send_notification(
                "🚀 AI Market Maker Starting",
                f"Mode: {args.mode}\nDebug: {args.debug}\nVersion: {settings.version}"
            )
        
        # Initialize components based on mode
        await initialize_components(args.mode, settings, args)
        
        # Start components based on mode
        tasks = await start_components(args.mode, args)
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        if components["notification_manager"]:
            await components["notification_manager"].send_notification(
                "❌ AI Market Maker Error",
                f"Error: {str(e)}"
            )
    finally:
        await shutdown()


async def initialize_components(mode: str, settings, args):
    """Initialize components based on mode"""
    logger = logging.getLogger(__name__)
    
    try:
        if mode in ["monitor", "all"]:
            logger.info("Initializing monitoring components...")
            
            # Position Monitor
            logger.info("Initializing Position Monitor...")
            components["position_monitor"] = create_position_monitor(settings.market_maker_strategy.dict())
            await components["position_monitor"].initialize()
            
            # Liquidation Tracker
            logger.info("Initializing Liquidation Tracker...")
            components["liquidation_tracker"] = create_liquidation_tracker(settings.liquidation_strategy.dict())
            await components["liquidation_tracker"].initialize()
            
            # Position Analyzer
            logger.info("Initializing Position Analyzer...")
            components["position_analyzer"] = create_position_analyzer(settings.ai_config.dict())
            await components["position_analyzer"].initialize()
            
            # Market Maker Tracker
            logger.info("Initializing Market Maker Tracker...")
            components["market_maker_tracker"] = create_market_maker_tracker(settings.ai_config.dict())
            await components["market_maker_tracker"].initialize()
        
        if mode in ["analysis", "all"]:
            logger.info("Initializing analysis components...")
            
            # Liquidation Predictor
            logger.info("Initializing Liquidation Predictor...")
            components["liquidation_predictor"] = create_liquidation_predictor(settings.ai_config.dict())
            await components["liquidation_predictor"].initialize()
            
            # Ensemble Predictor
            logger.info("Initializing Ensemble Predictor...")
            components["ensemble_predictor"] = create_ensemble_predictor(settings.ai_config.dict())
            await components["ensemble_predictor"].initialize()
        
        if mode in ["trading", "all"]:
            logger.info("Initializing trading components...")
            
            # Risk Manager
            logger.info("Initializing Risk Manager...")
            components["risk_manager"] = create_risk_manager(settings.ai_config.dict())
            await components["risk_manager"].initialize()
            
            # Signal Generator
            logger.info("Initializing Signal Generator...")
            components["signal_generator"] = create_signal_generator(settings.ai_config.dict())
            await components["signal_generator"].initialize()
        
        if mode == "dashboard" or (mode == "all" and not args.no_dashboard):
            logger.info("Initializing Web Dashboard...")
            # Dashboard will be started as a separate task
        
        system_status["active_components"] = len([c for c in components.values() if c is not None])
        logger.info(f"Initialized {system_status['active_components']} components")
        
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        raise


async def start_components(mode: str, args) -> List[asyncio.Task]:
    """Start components based on mode"""
    logger = logging.getLogger(__name__)
    tasks = []
    
    try:
        if mode == "monitor":
            logger.info("Starting monitoring mode...")
            tasks.extend([
                asyncio.create_task(components["position_monitor"]._monitor_loop()),
                asyncio.create_task(components["liquidation_tracker"]._tracking_loop()),
                asyncio.create_task(components["position_analyzer"]._analysis_loop()),
                asyncio.create_task(components["market_maker_tracker"]._tracking_loop()),
                asyncio.create_task(monitor_system())
            ])
            
        elif mode == "trading":
            logger.info("Starting trading mode...")
            tasks.extend([
                asyncio.create_task(components["risk_manager"]._risk_assessment_loop()),
                asyncio.create_task(components["signal_generator"]._signal_generation_loop()),
                asyncio.create_task(monitor_trading_system())
            ])
            
        elif mode == "analysis":
            logger.info("Starting analysis mode...")
            tasks.extend([
                asyncio.create_task(components["liquidation_predictor"]._prediction_loop()),
                asyncio.create_task(components["ensemble_predictor"]._prediction_loop()),
                asyncio.create_task(interactive_analysis_loop())
            ])
            
        elif mode == "backtest":
            logger.info("Starting backtest mode...")
            tasks.append(asyncio.create_task(run_backtest_mode(args.backtest_config)))
            
        elif mode == "dashboard":
            logger.info("Starting dashboard mode...")
            tasks.append(asyncio.create_task(start_dashboard()))
            
        elif mode == "all":
            logger.info("Starting all components...")
            
            # Start all monitoring components
            if components["position_monitor"]:
                tasks.append(asyncio.create_task(components["position_monitor"]._monitor_loop()))
            if components["liquidation_tracker"]:
                tasks.append(asyncio.create_task(components["liquidation_tracker"]._tracking_loop()))
            if components["position_analyzer"]:
                tasks.append(asyncio.create_task(components["position_analyzer"]._analysis_loop()))
            if components["market_maker_tracker"]:
                tasks.append(asyncio.create_task(components["market_maker_tracker"]._tracking_loop()))
            
            # Start all analysis components
            if components["liquidation_predictor"]:
                tasks.append(asyncio.create_task(components["liquidation_predictor"]._prediction_loop()))
            if components["ensemble_predictor"]:
                tasks.append(asyncio.create_task(components["ensemble_predictor"]._prediction_loop()))
            
            # Start all trading components
            if components["risk_manager"]:
                tasks.append(asyncio.create_task(components["risk_manager"]._risk_assessment_loop()))
            if components["signal_generator"]:
                tasks.append(asyncio.create_task(components["signal_generator"]._signal_generation_loop()))
            
            # Start system monitoring
            tasks.append(asyncio.create_task(monitor_system()))
            
            # Start dashboard if enabled
            if not args.no_dashboard:
                tasks.append(asyncio.create_task(start_dashboard()))
        
        logger.info(f"Started {len(tasks)} tasks")
        
    except Exception as e:
        logger.error(f"Error starting components: {e}")
    
    return tasks


async def monitor_system():
    """Monitor all system components"""
    logger = logging.getLogger(__name__)
    
    while system_status["is_running"]:
        try:
            # Get status from all components
            status_updates = []
            
            if components["position_monitor"]:
                metrics = components["position_monitor"].get_performance_metrics()
                status_updates.append(f"Position Monitor: {metrics.get('total_requests', 0)} requests")
            
            if components["liquidation_tracker"]:
                metrics = components["liquidation_tracker"].get_performance_metrics()
                status_updates.append(f"Liquidation Tracker: {metrics.get('total_events', 0)} events")
            
            if components["liquidation_predictor"]:
                metrics = components["liquidation_predictor"].get_performance_metrics()
                status_updates.append(f"Liquidation Predictor: {metrics.get('accuracy', 0):.1%} accuracy")
            
            if components["position_analyzer"]:
                metrics = components["position_analyzer"].get_performance_metrics()
                status_updates.append(f"Position Analyzer: {metrics.get('patterns_detected', 0)} patterns")
            
            if components["market_maker_tracker"]:
                metrics = components["market_maker_tracker"].get_performance_metrics()
                status_updates.append(f"Market Maker Tracker: {metrics.get('total_activities_tracked', 0)} activities")
            
            if components["risk_manager"]:
                metrics = components["risk_manager"].get_performance_metrics()
                status_updates.append(f"Risk Manager: {metrics.get('total_risk_assessments', 0)} assessments")
            
            if components["signal_generator"]:
                metrics = components["signal_generator"].get_performance_metrics()
                status_updates.append(f"Signal Generator: {metrics.get('total_signals_generated', 0)} signals")
            
            if components["ensemble_predictor"]:
                metrics = components["ensemble_predictor"].get_performance_metrics()
                status_updates.append(f"Ensemble Predictor: {metrics.get('total_predictions', 0)} predictions")
            
            # Log status every 5 minutes
            if status_updates:
                logger.info(f"System Status: {' | '.join(status_updates)}")
            
            # Update system status
            system_status["active_components"] = len([c for c in components.values() if c is not None])
            
            await asyncio.sleep(300)  # 5 minutes
            
        except Exception as e:
            logger.error(f"Error in system monitor: {e}")
            system_status["errors"] += 1
            await asyncio.sleep(60)


async def monitor_trading_system():
    """Monitor trading system"""
    logger = logging.getLogger(__name__)
    
    while system_status["is_running"]:
        try:
            logger.info("Trading system monitoring...")
            
            # Check risk levels
            if components["risk_manager"]:
                risk_metrics = components["risk_manager"].get_current_risk_metrics()
                if risk_metrics and risk_metrics.var_95 < -0.05:  # High risk
                    logger.warning("High portfolio risk detected!")
                    if components["notification_manager"]:
                        await components["notification_manager"].send_notification(
                            "⚠️ High Risk Alert",
                            f"Portfolio VaR: {risk_metrics.var_95:.2%}"
                        )
            
            # Check active signals
            if components["signal_generator"]:
                active_signals = components["signal_generator"].get_active_signals()
                high_confidence_signals = [s for s in active_signals if s.confidence > 0.8]
                if high_confidence_signals:
                    logger.info(f"Found {len(high_confidence_signals)} high-confidence signals")
            
            await asyncio.sleep(60)
            
        except Exception as e:
            logger.error(f"Error in trading monitor: {e}")
            system_status["errors"] += 1
            await asyncio.sleep(30)


async def interactive_analysis_loop():
    """Interactive analysis loop"""
    logger = logging.getLogger(__name__)
    
    print("\n🤖 AI Market Maker Analysis Mode")
    print("Type 'help' for commands, 'quit' to exit\n")
    
    while system_status["is_running"]:
        try:
            # Get user input
            user_input = input("Analysis: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            elif user_input.lower() == 'help':
                print_analysis_help()
                continue
            elif user_input.lower() == 'status':
                print_analysis_status()
                continue
            elif user_input.lower() == 'predictions':
                print_predictions()
                continue
            elif user_input.lower() == 'positions':
                print_positions()
                continue
            elif user_input.lower() == 'signals':
                print_signals()
                continue
            elif user_input.lower() == 'risk':
                print_risk_metrics()
                continue
            elif user_input.lower() == 'patterns':
                print_patterns()
                continue
            elif user_input.lower() == 'market_makers':
                print_market_maker_activities()
                continue
            else:
            response = await process_analysis_request(user_input)
                print(f"Response: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            logger.error(f"Error in analysis loop: {e}")
            print(f"Error: {e}")


async def run_backtest_mode(config_path: str):
    """Run backtest mode"""
    logger = logging.getLogger(__name__)
    
    try:
        from backtesting.backtest_engine import run_backtest, create_backtest_config
        
        if config_path:
            # Load config from file
            import json
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Create backtest config
            config = create_backtest_config(
                start_date=datetime.fromisoformat(config_data["start_date"]),
                end_date=datetime.fromisoformat(config_data["end_date"]),
                initial_capital=config_data.get("initial_capital", 100000),
                symbols=config_data.get("symbols", ["BTC", "ETH"]),
                strategies=config_data.get("strategies", ["sma_crossover"])
            )
        else:
            # Use default config
            config = create_backtest_config(
                start_date=datetime.now() - timedelta(days=90),
                end_date=datetime.now(),
                initial_capital=100000,
                symbols=["BTC", "ETH", "SOL"],
                strategies=["sma_crossover", "rsi"]
            )
        
        logger.info("Starting backtest...")
        results = await run_backtest(config)
        
        if results:
            logger.info("Backtest completed successfully!")
            logger.info(f"Total Return: {results.total_return:.2%}")
            logger.info(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
            logger.info(f"Max Drawdown: {results.max_drawdown:.2%}")
            logger.info(f"Win Rate: {results.win_rate:.2%}")
            
            # Save results
            results.save_results("backtest_results.json")
            logger.info("Results saved to backtest_results.json")
        else:
            logger.error("Backtest failed!")
            
    except Exception as e:
        logger.error(f"Error in backtest mode: {e}")


async def process_analysis_request(request: str) -> str:
    """Process analysis request"""
    try:
        request_lower = request.lower()
        
        if "liquidation" in request_lower:
            if components["liquidation_predictor"]:
                predictions = components["liquidation_predictor"].get_active_predictions()
                return f"Found {len(predictions)} active liquidation predictions"
            else:
                return "Liquidation predictor not available"
        
        elif "position" in request_lower:
            if components["position_monitor"]:
                traders = components["position_monitor"].get_top_traders(10)
                return f"Tracking {len(traders)} top traders"
            else:
                return "Position monitor not available"
        
        elif "signal" in request_lower:
            if components["signal_generator"]:
                signals = components["signal_generator"].get_active_signals()
                return f"Found {len(signals)} active trading signals"
            else:
                return "Signal generator not available"
        
        elif "risk" in request_lower:
            if components["risk_manager"]:
                metrics = components["risk_manager"].get_current_risk_metrics()
                if metrics:
                    return f"Current VaR: {metrics.var_95:.2%}, Max Drawdown: {metrics.max_drawdown:.2%}"
                else:
                    return "No risk metrics available"
            else:
                return "Risk manager not available"
        
        else:
            return "Unknown request. Type 'help' for available commands."
            
    except Exception as e:
        return f"Error processing request: {e}"


def print_analysis_help():
    """Print analysis help"""
    print("\nAvailable commands:")
    print("  status      - Show system status")
    print("  predictions - Show liquidation predictions")
    print("  positions   - Show position data")
    print("  signals     - Show trading signals")
    print("  risk        - Show risk metrics")
    print("  patterns    - Show detected patterns")
    print("  market_makers - Show market maker activities")
    print("  help        - Show this help")
    print("  quit        - Exit analysis mode")
    print()


def print_analysis_status():
    """Print analysis status"""
    print(f"\nSystem Status:")
    print(f"  Active Components: {system_status['active_components']}")
    print(f"  Errors: {system_status['errors']}")
    print(f"  Warnings: {system_status['warnings']}")
    print(f"  Uptime: {datetime.now() - system_status['start_time']}")
    print()


def print_predictions():
    """Print liquidation predictions"""
    if components["liquidation_predictor"]:
        predictions = components["liquidation_predictor"].get_active_predictions()
        print(f"\nActive Liquidation Predictions ({len(predictions)}):")
        for pred in predictions[:5]:  # Show top 5
            print(f"  {pred.symbol} {pred.side}: {pred.probability:.1%} confidence")
    else:
        print("Liquidation predictor not available")


def print_positions():
    """Print position data"""
    if components["position_monitor"]:
        traders = components["position_monitor"].get_top_traders(5)
        print(f"\nTop Traders ({len(traders)}):")
        for trader in traders:
            print(f"  {trader.address}: ${trader.total_value:,.0f}")
    else:
        print("Position monitor not available")


def print_signals():
    """Print trading signals"""
    if components["signal_generator"]:
        signals = components["signal_generator"].get_active_signals()
        print(f"\nActive Trading Signals ({len(signals)}):")
        for signal in signals[:5]:  # Show top 5
            print(f"  {signal.symbol} {signal.signal_type}: {signal.confidence:.1%} confidence")
    else:
        print("Signal generator not available")


def print_risk_metrics():
    """Print risk metrics"""
    if components["risk_manager"]:
        metrics = components["risk_manager"].get_current_risk_metrics()
        if metrics:
            print(f"\nRisk Metrics:")
            print(f"  VaR (95%): {metrics.var_95:.2%}")
            print(f"  VaR (99%): {metrics.var_99:.2%}")
            print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
            print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        else:
            print("No risk metrics available")
    else:
        print("Risk manager not available")


def print_patterns():
    """Print detected patterns"""
    if components["position_analyzer"]:
        patterns = components["position_analyzer"].get_recent_patterns(24)
        print(f"\nRecent Patterns ({len(patterns)}):")
        for pattern in patterns[:5]:  # Show top 5
            print(f"  {pattern.pattern_type} in {pattern.traders_involved[0]}: {pattern.confidence:.1%} confidence")
    else:
        print("Position analyzer not available")


def print_market_maker_activities():
    """Print market maker activities"""
    if components["market_maker_tracker"]:
        activities = components["market_maker_tracker"].get_liquidity_events(24)
        print(f"\nRecent Market Maker Activities ({len(activities)}):")
        for activity in activities[:5]:  # Show top 5
            print(f"  {activity.event_type} in {activity.symbol}: ${activity.volume:,.0f}")
    else:
        print("Market maker tracker not available")


async def shutdown():
    """Graceful shutdown"""
    logger = logging.getLogger(__name__)
    
    logger.info("🛑 Shutting down AI Market Maker...")
    system_status["is_running"] = False
    
    # Stop all components
    for name, component in components.items():
        if component and hasattr(component, 'stop'):
            try:
                await component.stop()
                logger.info(f"Stopped {name}")
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")
    
    # Send final notification
    if components["notification_manager"]:
        try:
            uptime = datetime.now() - system_status["start_time"]
            await components["notification_manager"].send_notification(
                "🛑 AI Market Maker Stopped",
                f"Uptime: {uptime}\nActive Components: {system_status['active_components']}\nErrors: {system_status['errors']}"
            )
        except Exception as e:
            logger.error(f"Error sending final notification: {e}")
    
    logger.info("✅ AI Market Maker shutdown complete")


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {signum}, initiating shutdown...")
    asyncio.create_task(shutdown())


if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the main application
        asyncio.run(main())