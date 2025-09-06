# Liquidation Signal Trading Bot

## Vision
A systematic, data-driven trading bot that capitalizes on large liquidation events in crypto markets. The bot enters trades when significant liquidations occur, using robust risk management and backtesting to ensure disciplined, repeatable performance.

## Key Features
- **Liquidation-Driven Trading:** Enters trades on large liquidations, exploiting short-term price inefficiencies.
- **Order Book Analysis:** Finds optimal entry points using real-time order book data.
- **Risk Management:** Hard stop loss, take profit, leverage control, and trend/volatility filters.
- **Backtesting:** Validates all logic with historical liquidation data.
- **Performance Tracking:** Logs all trades, calculates P&L, and provides analytics.
- **Web Dashboard:** Live monitoring of signals, trades, and performance.
- **Configurable:** All thresholds, risk parameters, and filters are easily adjustable.

## Architecture
```
liquidation_signal_trading_bot/
├── bot/                  # Bot logic
├── data/                 # Liquidation and order book feeds
├── risk/                 # Risk management and filters
├── backtesting/          # Backtesting engine
├── api/                  # Exchange connector
├── web/                  # Dashboard and web interface
├── config/               # Settings and strategy configs
├── tests/                # Unit and integration tests
├── main.py               # CLI entry point
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

## Quick Start
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure your settings and API keys in `config/`**
3. **Run the system:**
   ```bash
   python main.py --mode live
   ```
4. **Access the dashboard:**
   - Visit `http://localhost:5000` for live monitoring

## Strategy Overview
- **Signal:** Enter trade when liquidation > threshold (e.g., $10M)
- **Entry:** Avoid if price is too far from SMA; use order book for best entry
- **Risk:** Hard stop loss, take profit, leverage control, trend/volatility filter
- **Backtest:** Validate all logic with historical data before live trading

## Contributing
Contributions are welcome! See the `CONTRIBUTING.md` for guidelines.

## License
MIT License 