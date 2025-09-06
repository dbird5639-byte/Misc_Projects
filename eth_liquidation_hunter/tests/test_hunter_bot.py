import pytest
import asyncio
from bot.hunter_bot import HunterBot
from hunter.position_tracker import PositionTracker
from hunter.liquidation_detector import LiquidationDetector
from hunter.cascade_analyzer import CascadeAnalyzer
from risk.risk_manager import RiskManager
from data.trade_executor import TradeExecutor
from api.exchange_api import ExchangeAPI
from config.settings import get_config

@pytest.mark.asyncio
async def test_hunterbot_init_and_run():
    config = get_config()
    api = ExchangeAPI(config)
    position_tracker = PositionTracker(config)
    liquidation_detector = LiquidationDetector(config)
    cascade_analyzer = CascadeAnalyzer(config)
    risk_manager = RiskManager(config)
    trade_executor = TradeExecutor(config, api)
    bot = HunterBot(config, position_tracker, liquidation_detector, cascade_analyzer, risk_manager, trade_executor)
    await bot.initialize()
    # Run only one loop iteration for test
    bot.is_running = False
    await bot.run()
    await bot.stop() 