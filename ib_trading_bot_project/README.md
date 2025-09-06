# Interactive Brokers Trading Bot

A Python-based automated trading bot for Interactive Brokers (IB) that supports stocks, futures, and options trading with comprehensive risk management and strategy implementation.

## ⚠️ Important Disclaimers

- **This is not financial advice**
- The creator is not a financial advisor or guru
- Results will vary—you cannot expect the same results as others
- Trading involves significant risk and potential losses
- Always test thoroughly before using real money

## Features

- **Multi-Instrument Support**: Stocks, futures, and options trading
- **Real-time Market Data**: Live price feeds and market information
- **Order Management**: Automated order placement and management
- **Risk Management**: Position sizing, stop-loss, and portfolio protection
- **Strategy Framework**: Modular strategy implementation
- **Performance Monitoring**: Real-time tracking and reporting
- **Connection Management**: Robust IB TWS connection handling

## Prerequisites

- Interactive Brokers account
- Trader Workstation (TWS) or IB Gateway installed and running
- Python 3.8+
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure your IB credentials in `config/settings.py`
4. Start TWS or IB Gateway
5. Run the bot: `python src/main.py`

## Project Structure

```
ib_trading_bot_project/
├── config/
│   ├── settings.py          # Configuration settings
│   └── strategies.json      # Strategy configurations
├── src/
│   ├── main.py              # Main application entry point
│   ├── connection/
│   │   ├── ib_connector.py  # IB connection management
│   │   └── market_data.py   # Market data handling
│   ├── trading/
│   │   ├── order_manager.py # Order management system
│   │   ├── position_manager.py # Position tracking
│   │   └── risk_manager.py  # Risk management
│   ├── strategies/
│   │   ├── base_strategy.py # Base strategy class
│   │   ├── momentum_strategy.py # Example strategy
│   │   └── mean_reversion_strategy.py # Example strategy
│   └── utils/
│       ├── logger.py        # Logging utilities
│       └── helpers.py       # Helper functions
├── examples/
│   └── basic_usage.py       # Basic usage examples
├── tests/
│   └── test_strategies.py   # Strategy testing
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Quick Start

```python
from src.connection.ib_connector import IBConnector
from src.trading.order_manager import OrderManager
from src.strategies.momentum_strategy import MomentumStrategy

# Initialize connection
connector = IBConnector()
connector.connect()

# Create order manager
order_manager = OrderManager(connector)

# Create and run strategy
strategy = MomentumStrategy(order_manager)
strategy.run()
```

## Configuration

Edit `config/settings.py` to configure:
- IB connection settings
- Trading parameters
- Risk management rules
- Strategy settings

## Risk Management

The bot includes comprehensive risk management:
- Position size limits
- Stop-loss orders
- Portfolio exposure limits
- Maximum drawdown protection
- Correlation-based position limits

## Strategies

### Available Strategies
- **Momentum Strategy**: Follows price momentum with trend confirmation
- **Mean Reversion**: Trades against extreme price movements
- **Custom Strategies**: Extend base strategy class for custom logic

### Creating Custom Strategies

```python
from src.strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def analyze_market(self, data):
        # Your analysis logic here
        pass
    
    def generate_signals(self, analysis):
        # Your signal generation logic here
        pass
```

## Monitoring and Logging

The bot provides comprehensive logging and monitoring:
- Trade execution logs
- Performance metrics
- Error tracking
- Real-time alerts

## Testing

Run tests with:
```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is for educational purposes only. Use at your own risk.

## Support

For issues and questions, please check the documentation or create an issue in the repository. 