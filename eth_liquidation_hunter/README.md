# ETH Liquidation Hunter

## Vision
A systematic, data-driven bot platform for tracking, analyzing, and trading around large public Ethereum positions at risk of liquidation. The system leverages on-chain and exchange data to detect liquidation risk, analyze cascade potential, and execute risk-managed strategies.

## Key Features
- **Position Tracking:** Monitor large public positions (e.g., Maker Vaults, CEX whales) in real time.
- **Liquidation Risk Detection:** Alert and act when positions approach critical liquidation thresholds.
- **Cascade Analysis:** Assess the risk and potential impact of large liquidations on the broader market.
- **Automated Trading:** Execute strategies to profit from or hedge against liquidation events.
- **Risk Management:** Strict position sizing, stop loss, and compliance checks.
- **Backtesting:** Validate all logic with historical data before live deployment.
- **Web Dashboard:** Live monitoring of tracked positions, liquidation risk, and bot actions.
- **Compliance:** Consider regulatory and ethical implications of liquidation hunting.

## Architecture
```
eth_liquidation_hunter/
├── hunter/                  # Position tracking, liquidation detection, cascade analysis
├── bot/                     # Bot logic
├── data/                    # On-chain and exchange data feeds
├── risk/                    # Risk management and compliance
├── backtesting/             # Backtesting engine
├── api/                     # API connectors
├── web/                     # Dashboard and web interface
├── config/                  # Settings and strategy configs
├── tests/                   # Unit and integration tests
├── main.py                  # CLI entry point
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
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
- **Track:** Monitor large public positions (on-chain, CEX, DEX)
- **Detect:** Alert and act when positions approach liquidation
- **Analyze:** Assess cascade risk and market impact
- **Trade:** Execute risk-managed strategies around liquidation events
- **Backtest:** Validate all logic with historical data before live trading

## Contributing
Contributions are welcome! See the `CONTRIBUTING.md` for guidelines.

## License
MIT License 