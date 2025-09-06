CONFIG = {
    'api_key': 'your_exchange_api_key',
    'api_secret': 'your_exchange_api_secret',
    'position_size': 1,
    'max_position_size': 10,
    'stop_loss_pct': 0.01,
    'take_profit_pct': 0.02,
    'leverage': 5,
    'liquidation_threshold': 10000000,  # $10M
    'loop_interval': 5,
}

def get_config():
    return CONFIG 