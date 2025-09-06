# Dragonfly AI Trading Bots

## Vision
A next-generation, AI-powered trading bot platform that leverages voice-driven development, real-time market aggregation, and advanced risk management. Inspired by the Dragonfly Project, this system enables anyone to rapidly build, backtest, and deploy sophisticated trading bots using natural language and AI tools.

## Key Features
- **AI Voice Input:** Build bots by speaking requirements; AI generates and deploys code.
- **Dragonfly Bot:** Trades with the majority, tracks all positions, focuses on liquidations, and uses PPM (Profit Per Minute) for exits.
- **PPM Metric:** All strategies and trades are evaluated by Profit Per Minute, not just P&L.
- **Real-Time Position Aggregation:** Continuously monitors and aggregates all market positions (e.g., Hyperliquid).
- **Risk Management:** Tiny position sizes, strict stop-loss, adaptive scaling.
- **Backtesting:** RBI (Research, Backtest, Implement) framework ensures all strategies are validated before live trading.
- **Web Dashboard:** Live monitoring of bots, PPM, positions, and risk.
- **Rapid Iteration:** Voice-driven, AI-assisted development for fast prototyping and deployment.

## Architecture
```
dragonfly_ai_trading_bots/
├── ai_voice/           # Voice-to-code and command parsing
├── bots/               # Bot logic (Dragonfly, base classes)
├── data/               # Position aggregation, market feeds, PPM tracking
├── framework/          # Utilities, risk management, RBI pipeline
├── backtesting/        # Backtesting engine
├── api/                # Exchange connectors (e.g., Hyperliquid)
├── web/                # Dashboard and web interface
├── config/             # Settings and strategy configs
├── tests/              # Unit and integration tests
├── main.py             # CLI entry point
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
```

## Quick Start
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure your settings and API keys in `config/`**
3. **Run the system:**
   ```bash
   python main.py --mode all
   ```
4. **Access the dashboard:**
   - Visit `http://localhost:5000` for live monitoring

## RBI Framework
- **Research:** Analyze market data, develop hypotheses, and design strategies.
- **Backtest:** Validate strategies using historical data and the PPM metric.
- **Implement:** Deploy bots with small position sizes, monitor, and scale up as performance is proven.

## Contributing
Contributions are welcome! See the `CONTRIBUTING.md` for guidelines.

## License
MIT License 