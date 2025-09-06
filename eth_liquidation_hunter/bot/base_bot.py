import abc

class BaseBot(abc.ABC):
    def __init__(self, config):
        self.config = config
        self.is_running = False

    @abc.abstractmethod
    async def initialize(self):
        pass

    @abc.abstractmethod
    async def run(self):
        pass

    @abc.abstractmethod
    async def stop(self):
        pass

    def position_size(self, balance, risk_per_trade):
        # Stub for position sizing logic
        return min(balance * risk_per_trade, self.config.get('max_position_size', 100))

    def stop_loss(self, entry_price):
        # Stub for stop-loss logic
        return entry_price * (1 - self.config.get('stop_loss_pct', 0.01)) 