CONFIG = {
    'api_key': 'your_hyperliquid_api_key',
    'api_secret': 'your_hyperliquid_api_secret',
    'position_size': 1,
    'max_position_size': 10,
    'stop_loss_pct': 0.01,
    'ppm_exit': 0.1,
    'loop_interval': 5,
    'risk_per_trade': 0.01,
}

def get_config():
    return CONFIG 