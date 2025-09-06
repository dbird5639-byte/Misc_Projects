from .base_bot import BaseBot
import asyncio

class LiquidationBot(BaseBot):
    """
    LiquidationBot enters trades when large liquidations occur, analyzes order book for optimal entry,
    and manages risk with stop loss, take profit, and trend/volatility filters.
    """
    def __init__(self, config, liquidation_feed, orderbook_feed, risk_manager, trade_executor):
        super().__init__(config)
        self.liquidation_feed = liquidation_feed
        self.orderbook_feed = orderbook_feed
        self.risk_manager = risk_manager
        self.trade_executor = trade_executor
        self.current_position = None

    async def initialize(self):
        print("[LiquidationBot] Initializing...")
        await self.liquidation_feed.initialize()
        await self.orderbook_feed.initialize()
        await self.risk_manager.initialize()
        await self.trade_executor.initialize()
        self.is_running = True

    async def run(self):
        print("[LiquidationBot] Running main loop...")
        while self.is_running:
            # 1. Check for large liquidation events
            signal = await self.liquidation_feed.get_liquidation_signal()
            # 2. Analyze order book for entry
            entry_price = await self.orderbook_feed.get_optimal_entry(signal)
            # 3. Risk checks
            if await self.risk_manager.should_enter(signal, entry_price):
                await self.trade_executor.enter_position(signal, entry_price)
                self.current_position = (signal, entry_price)
            # 4. Monitor and exit if needed
            if self.current_position and await self.risk_manager.should_exit(self.current_position):
                await self.trade_executor.exit_position()
                self.current_position = None
            # 5. Sleep or wait for next tick
            await asyncio.sleep(self.config.get('loop_interval', 5))

    async def stop(self):
        print("[LiquidationBot] Stopping...")
        self.is_running = False
        await self.trade_executor.exit_position() 