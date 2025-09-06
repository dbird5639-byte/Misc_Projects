import pytest
import asyncio
from bot.liquidation_bot import LiquidationBot
from data.liquidation_feed import LiquidationFeed
from data.orderbook_feed import OrderbookFeed
from risk.risk_manager import RiskManager
from data.trade_executor import TradeExecutor
from api.exchange_api import ExchangeAPI
from config.settings import get_config

@pytest.mark.asyncio
async def test_liquidationbot_init_and_run():
    config = get_config()
    api = ExchangeAPI(config)
    liquidation_feed = LiquidationFeed(config)
    orderbook_feed = OrderbookFeed(config)
    risk_manager = RiskManager(config)
    trade_executor = TradeExecutor(config, api)
    bot = LiquidationBot(config, liquidation_feed, orderbook_feed, risk_manager, trade_executor)
    await bot.initialize()
    # Run only one loop iteration for test
    bot.is_running = False
    await bot.run()
    await bot.stop() 