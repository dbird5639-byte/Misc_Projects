import pytest
import asyncio
from bots.dragonfly import DragonflyBot
from data.position_aggregator import PositionAggregator
from data.ppm_tracker import PPMTracker
from data.trade_executor import TradeExecutor
from api.hyperliquid_api import HyperliquidAPI
from config.settings import get_config

@pytest.mark.asyncio
async def test_dragonflybot_init_and_run():
    config = get_config()
    api = HyperliquidAPI(config)
    position_aggregator = PositionAggregator(config)
    ppm_tracker = PPMTracker(config)
    trade_executor = TradeExecutor(config, api)
    bot = DragonflyBot(config, position_aggregator, ppm_tracker, trade_executor)
    await bot.initialize()
    # Run only one loop iteration for test
    bot.is_running = False
    await bot.run()
    await bot.stop() 