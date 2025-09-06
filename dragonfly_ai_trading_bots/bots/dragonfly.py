from .base_bot import BaseBot
import asyncio

class DragonflyBot(BaseBot):
    """
    DragonflyBot trades in the direction of the majority, tracks all positions,
    focuses on liquidation opportunities, and uses PPM (Profit Per Minute) for exits.
    """
    def __init__(self, config, position_aggregator, ppm_tracker, trade_executor):
        super().__init__(config)
        self.position_aggregator = position_aggregator
        self.ppm_tracker = ppm_tracker
        self.trade_executor = trade_executor
        self.current_position = None

    async def initialize(self):
        print("[DragonflyBot] Initializing...")
        await self.position_aggregator.initialize()
        await self.ppm_tracker.initialize()
        await self.trade_executor.initialize()
        self.is_running = True

    async def run(self):
        print("[DragonflyBot] Running main loop...")
        while self.is_running:
            # 1. Aggregate positions
            majority_side = await self.position_aggregator.get_majority_side()
            # 2. Check for liquidation opportunities
            liquidation_signal = await self.position_aggregator.detect_liquidation_opportunity()
            # 3. Decide whether to enter/exit
            if not self.current_position and majority_side:
                await self.trade_executor.enter_position(majority_side)
                self.current_position = majority_side
            # 4. Track PPM and exit if target met
            ppm = await self.ppm_tracker.get_current_ppm()
            if self.current_position and ppm is not None and ppm > self.config.get('ppm_exit', 0.1):
                await self.trade_executor.exit_position()
                self.current_position = None
            # 5. Sleep or wait for next tick
            await asyncio.sleep(self.config.get('loop_interval', 5))

    async def stop(self):
        print("[DragonflyBot] Stopping...")
        self.is_running = False
        await self.trade_executor.exit_position() 