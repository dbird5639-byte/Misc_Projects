from .base_bot import BaseBot
import asyncio

class HunterBot(BaseBot):
    """
    HunterBot tracks large public positions, detects liquidation risk, analyzes cascade potential,
    and executes risk-managed trades around liquidation events.
    """
    def __init__(self, config, position_tracker, liquidation_detector, cascade_analyzer, risk_manager, trade_executor):
        super().__init__(config)
        self.position_tracker = position_tracker
        self.liquidation_detector = liquidation_detector
        self.cascade_analyzer = cascade_analyzer
        self.risk_manager = risk_manager
        self.trade_executor = trade_executor
        self.current_position = None

    async def initialize(self):
        print("[HunterBot] Initializing...")
        await self.position_tracker.initialize()
        await self.liquidation_detector.initialize()
        await self.cascade_analyzer.initialize()
        await self.risk_manager.initialize()
        await self.trade_executor.initialize()
        self.is_running = True

    async def run(self):
        print("[HunterBot] Running main loop...")
        while self.is_running:
            # 1. Track large positions
            positions = await self.position_tracker.get_tracked_positions()
            # 2. Detect liquidation risk
            risk_signals = await self.liquidation_detector.detect_liquidation_risk(positions)
            # 3. Analyze cascade potential
            cascade_risk = await self.cascade_analyzer.analyze(risk_signals)
            # 4. Risk checks and trade execution
            if await self.risk_manager.should_enter(risk_signals, cascade_risk):
                await self.trade_executor.enter_position(risk_signals, cascade_risk)
                self.current_position = (risk_signals, cascade_risk)
            # 5. Monitor and exit if needed
            if self.current_position and await self.risk_manager.should_exit(self.current_position):
                await self.trade_executor.exit_position()
                self.current_position = None
            await asyncio.sleep(self.config.get('loop_interval', 5))

    async def stop(self):
        print("[HunterBot] Stopping...")
        self.is_running = False
        await self.trade_executor.exit_position() 