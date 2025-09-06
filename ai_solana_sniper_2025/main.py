"""
Main entry point for AI-Powered Solana Meme Coin Sniper
"""

import asyncio
import argparse
import signal
import sys
from typing import Dict, Any
import logging

from config.settings import get_settings, create_default_configs
from agents.sniper_agent import SniperAgent
from agents.chat_agent import ChatAgent
from agents.focus_agent import FocusAgent
from agents.model_factory import ModelFactory
from utils.logger import setup_logging
from utils.notifications import NotificationManager

# Global variables for graceful shutdown
sniper_agent = None
chat_agent = None
focus_agent = None
notification_manager = None


async def main():
    """Main application entry point"""
    global sniper_agent, chat_agent, focus_agent, notification_manager
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI-Powered Solana Meme Coin Sniper")
    parser.add_argument("--mode", choices=["sniper", "analysis", "all", "chat"], 
                       default="all", help="Operation mode")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-notifications", action="store_true", help="Disable notifications")
    
    args = parser.parse_args()
    
    try:
        # Create default configs if they don't exist
        create_default_configs()
        
        # Load settings
        settings = get_settings()
        settings.debug_mode = args.debug
        
        # Setup logging
        setup_logging(settings.log_level)
        logger = logging.getLogger(__name__)
        
        logger.info("🚀 Starting AI-Powered Solana Meme Coin Sniper")
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Debug: {args.debug}")
        
        # Initialize notification manager
        if not args.no_notifications:
            notifications_config = settings.trading_config.notifications if settings.trading_config else {}
            notification_manager = NotificationManager(notifications_config)
            await notification_manager.send_notification(
                "🚀 AI Sniper Starting",
                f"Mode: {args.mode}\nDebug: {args.debug}"
            )
        
        # Initialize AI model factory
        logger.info("Initializing AI Model Factory...")
        ai_config_dict = settings.ai_config.model_dump() if settings.ai_config else {}
        model_factory = ModelFactory(ai_config_dict)
        await model_factory.initialize_models()
        
        available_models = model_factory.get_available_models()
        logger.info(f"Available AI models: {available_models}")
        
        # Initialize agents based on mode
        if args.mode in ["sniper", "all"]:
            logger.info("Initializing Sniper Agent...")
            trading_config_dict = settings.trading_config.model_dump() if settings.trading_config else {}
            sniper_agent = SniperAgent(settings.trading_config.sniper if settings.trading_config else {})
            await sniper_agent.initialize(ai_config_dict, trading_config_dict)
        
        if args.mode in ["analysis", "all"]:
            logger.info("Initializing Chat Agent...")
            chat_agent = ChatAgent(settings.ai_config.agent_config if settings.ai_config else {})
            await chat_agent.initialize(model_factory)
        
        if args.mode in ["all"]:
            logger.info("Initializing Focus Agent...")
            focus_agent = FocusAgent(settings.ai_config.agent_config if settings.ai_config else {})
            await focus_agent.initialize()
        
        # Start agents based on mode
        tasks = []
        
        if args.mode == "sniper":
            logger.info("Starting Sniper Agent only...")
            if sniper_agent:
                tasks.append(asyncio.create_task(sniper_agent.start()))
            
        elif args.mode == "analysis":
            logger.info("Starting Chat Agent only...")
            # For analysis mode, start an interactive chat loop
            if chat_agent:
                tasks.append(asyncio.create_task(interactive_chat_loop(chat_agent)))
            
        elif args.mode == "chat":
            logger.info("Starting interactive chat mode...")
            if chat_agent:
                tasks.append(asyncio.create_task(interactive_chat_loop(chat_agent)))
            
        elif args.mode == "all":
            logger.info("Starting all agents...")
            if sniper_agent:
                tasks.append(asyncio.create_task(sniper_agent.start()))
            if focus_agent:
                tasks.append(asyncio.create_task(focus_agent.start()))
            tasks.append(asyncio.create_task(monitor_agents()))
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        if notification_manager:
            await notification_manager.send_notification(
                "❌ AI Sniper Error",
                f"Error: {str(e)}"
            )
    finally:
        await shutdown()


async def interactive_chat_loop(chat_agent: ChatAgent):
    """Interactive chat loop for manual analysis"""
    logger = logging.getLogger(__name__)
    
    print("\n🤖 AI-Powered Solana Trading Assistant")
    print("Type 'help' for commands, 'quit' to exit\n")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            elif user_input.lower() == 'help':
                print_help()
                continue
            elif user_input.lower() == 'status':
                print_status(chat_agent)
                continue
            elif user_input.lower() == 'clear':
                print("\n" * 50)
                continue
            elif not user_input:
                continue
            
            # Process user input
            print("🤖 AI: Analyzing...")
            response = await chat_agent.add_user_message(user_input)
            print(f"🤖 AI: {response}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            logger.error(f"Error in chat loop: {e}")
            print(f"❌ Error: {e}\n")


async def monitor_agents():
    """Monitor all agents and provide status updates"""
    logger = logging.getLogger(__name__)
    
    while True:
        try:
            # Get status from all agents
            status_updates = []
            
            if sniper_agent:
                sniper_status = sniper_agent.get_status()
                status_updates.append(f"Sniper: {'🟢 Running' if sniper_status['is_running'] else '🔴 Stopped'}")
            
            if focus_agent:
                focus_metrics = focus_agent.get_performance_metrics()
                status_updates.append(f"Focus: {focus_metrics['success_rate']:.1%} success rate")
            
            if chat_agent:
                chat_metrics = chat_agent.get_performance_metrics()
                status_updates.append(f"Chat: {chat_metrics['success_rate']:.1%} success rate")
            
            # Log status every 5 minutes
            if status_updates:
                logger.info(f"Agent Status: {' | '.join(status_updates)}")
            
            await asyncio.sleep(300)  # 5 minutes
            
        except Exception as e:
            logger.error(f"Error in agent monitor: {e}")
            await asyncio.sleep(60)


def print_help():
    """Print help information"""
    print("\n📚 Available Commands:")
    print("  help          - Show this help message")
    print("  status        - Show agent status")
    print("  clear         - Clear the screen")
    print("  quit/exit/q   - Exit the application")
    print("\n💡 Example Analysis Requests:")
    print("  'Analyze this token: SOL/USDC price $150, volume $1M, liquidity $500K'")
    print("  'What are the current market conditions for Solana meme coins?'")
    print("  'Assess the risk of this token: [token_address]'")
    print("  'Review my trading strategy: [strategy_details]'")
    print()


def print_status(chat_agent: ChatAgent):
    """Print agent status"""
    print("\n📊 Agent Status:")
    
    if chat_agent:
        metrics = chat_agent.get_performance_metrics()
        print(f"  Chat Agent:")
        print(f"    Total Requests: {metrics['total_requests']}")
        print(f"    Success Rate: {metrics['success_rate']:.1%}")
        print(f"    Avg Response Time: {metrics['avg_response_time']:.2f}s")
        print(f"    Conversation Length: {metrics['conversation_length']} messages")
    
    print()


async def shutdown():
    """Graceful shutdown"""
    logger = logging.getLogger(__name__)
    logger.info("🛑 Shutting down AI Sniper...")
    
    # Stop all agents
    if sniper_agent and sniper_agent.is_running:
        await sniper_agent.stop()
    
    if focus_agent and focus_agent.is_running:
        await focus_agent.stop()
    
    # Send shutdown notification
    if notification_manager:
        await notification_manager.send_notification(
            "🛑 AI Sniper Shutdown",
            "Application stopped gracefully"
        )
    
    logger.info("✅ Shutdown complete")


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {signum}, initiating shutdown...")
    asyncio.create_task(shutdown())


if __name__ == "__main__":
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the main application
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1) 