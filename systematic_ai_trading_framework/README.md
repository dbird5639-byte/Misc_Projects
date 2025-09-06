# Systematic AI Trading Framework

A comprehensive algorithmic trading system that implements the RBI (Research, Backtest, Implement) methodology enhanced with local AI agents for systematic strategy development and execution.

## 🎯 Core Philosophy

This framework follows the principle that **AI should enhance systematic trading processes rather than attempt direct price prediction**. Instead of trying to predict market movements, we use AI to:

- **Research**: Continuously discover and analyze new trading strategies
- **Backtest**: Automatically validate strategies against historical data
- **Implement**: Deploy only proven strategies with proper risk management

## 🏗️ Architecture

```
systematic_ai_trading_framework/
├── agents/                 # AI Agent System
│   ├── research_agent.py   # Discovers new strategies
│   ├── backtest_agent.py   # Validates strategies
│   ├── package_agent.py    # Manages dependencies
│   └── model_factory.py    # AI model management
├── strategies/             # Trading Strategies
│   ├── base_strategy.py    # Strategy base class
│   ├── momentum_strategy.py
│   ├── mean_reversion_strategy.py
│   └── regime_detection_strategy.py
├── data/                   # Data Management
│   ├── market_data.py      # Market data feeds
│   ├── data_processor.py   # Data preprocessing
│   └── feature_engineer.py # Feature engineering
├── backtesting/            # Backtesting Engine
│   ├── backtest_engine.py  # Core backtesting logic
│   ├── performance.py      # Performance metrics
│   └── walk_forward.py     # Walk-forward analysis
├── execution/              # Trade Execution
│   ├── order_manager.py    # Order management
│   ├── risk_manager.py     # Risk controls
│   └── portfolio_manager.py # Portfolio tracking
├── config/                 # Configuration
│   ├── settings.py         # Main settings
│   ├── models.json         # AI model configs
│   └── strategies.json     # Strategy configs
├── utils/                  # Utilities
│   ├── logger.py           # Logging system
│   ├── notifications.py    # Alert system
│   └── database.py         # Data persistence
├── web/                    # Web Dashboard
│   ├── dashboard.py        # Flask dashboard
│   └── templates/          # HTML templates
├── tests/                  # Testing
│   ├── test_agents.py      # Agent tests
│   ├── test_strategies.py  # Strategy tests
│   └── test_backtesting.py # Backtesting tests
├── main.py                 # Main entry point
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 🚀 Key Features

### AI Agent System
- **Local AI Processing**: Uses Ollama for local AI models (no API costs)
- **Research Agent**: Continuously scans for new trading ideas
- **Backtest Agent**: Automatically validates strategies
- **Package Agent**: Manages system dependencies
- **Model Factory**: Handles multiple AI models

### Systematic Trading
- **RBI Methodology**: Research → Backtest → Implement workflow
- **Multi-Strategy Support**: Framework for various strategy types
- **Risk Management**: Built-in position sizing and risk controls
- **Performance Tracking**: Comprehensive metrics and analysis

### Advanced Backtesting
- **Walk-Forward Analysis**: Out-of-sample testing
- **Monte Carlo Simulation**: Risk assessment
- **Regime Detection**: Market condition analysis
- **Performance Metrics**: Sharpe ratio, drawdown, etc.

### Real-Time Execution
- **Multi-Exchange Support**: Connect to various exchanges
- **Order Management**: Smart order routing
- **Portfolio Tracking**: Real-time position monitoring
- **Alert System**: Notifications for key events

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd systematic_ai_trading_framework
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up local AI models** (Ollama):
```bash
# Install Ollama (https://ollama.ai)
ollama pull deepseek-coder:6.7b
ollama pull llama2:7b
```

4. **Configure settings**:
```bash
cp config/settings.py.example config/settings.py
# Edit config/settings.py with your preferences
```

## 📊 Usage

### Basic Usage

```python
from systematic_ai_trading_framework.main import SystematicTradingFramework

# Initialize the framework
framework = SystematicTradingFramework()

# Start the RBI process
framework.start_research_cycle()

# Run backtesting on discovered strategies
framework.run_backtesting()

# Deploy successful strategies
framework.deploy_strategies()
```

### AI Agent Interaction

```python
from systematic_ai_trading_framework.agents.research_agent import ResearchAgent

# Initialize research agent
research_agent = ResearchAgent()

# Feed new trading idea
idea = "Momentum strategy based on RSI divergence"
strategy = research_agent.analyze_idea(idea)

# Get strategy details
print(f"Strategy: {strategy.name}")
print(f"Confidence: {strategy.confidence}")
print(f"Expected Sharpe: {strategy.expected_sharpe}")
```

### Strategy Development

```python
from systematic_ai_trading_framework.strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self, name, parameters):
        super().__init__(name, parameters)
    
    def generate_signals(self, data):
        # Implement your strategy logic
        signals = self.calculate_signals(data)
        return signals
    
    def calculate_signals(self, data):
        # Your signal generation logic
        pass
```

## 🔧 Configuration

### AI Model Configuration (`config/models.json`)

```json
{
  "default_model": "deepseek-coder:6.7b",
  "models": {
    "deepseek-coder:6.7b": {
      "type": "ollama",
      "temperature": 0.1,
      "max_tokens": 2048
    },
    "llama2:7b": {
      "type": "ollama", 
      "temperature": 0.2,
      "max_tokens": 1024
    }
  }
}
```

### Strategy Configuration (`config/strategies.json`)

```json
{
  "momentum_strategy": {
    "enabled": true,
    "parameters": {
      "lookback_period": 20,
      "threshold": 0.02
    },
    "risk_management": {
      "max_position_size": 0.1,
      "stop_loss": 0.05,
      "take_profit": 0.15
    }
  }
}
```

## 📈 Performance Metrics

The framework tracks comprehensive performance metrics:

- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Calmar Ratio**: Annual return / Maximum drawdown
- **Sortino Ratio**: Downside deviation-adjusted returns

## 🔒 Risk Management

Built-in risk management features:

- **Position Sizing**: Kelly criterion and fixed fractional sizing
- **Stop Losses**: Dynamic and static stop loss mechanisms
- **Portfolio Limits**: Maximum exposure per strategy
- **Correlation Analysis**: Diversification monitoring
- **Regime Detection**: Market condition-based position sizing

## 🌐 Web Dashboard

Access the web dashboard for real-time monitoring:

```bash
python web/dashboard.py
```

Features:
- Real-time strategy performance
- Portfolio overview
- Trade history
- Risk metrics
- AI agent status

## 🧪 Testing

Run the test suite:

```bash
python -m pytest tests/
```

Test coverage includes:
- AI agent functionality
- Strategy backtesting
- Risk management
- Data processing
- Performance calculations

## 📚 Strategy Examples

### Momentum Strategy
```python
class MomentumStrategy(BaseStrategy):
    def calculate_signals(self, data):
        # Calculate momentum indicators
        rsi = self.calculate_rsi(data, period=14)
        macd = self.calculate_macd(data)
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        signals[(rsi < 30) & (macd > 0)] = 1  # Buy signal
        signals[(rsi > 70) & (macd < 0)] = -1 # Sell signal
        
        return signals
```

### Mean Reversion Strategy
```python
class MeanReversionStrategy(BaseStrategy):
    def calculate_signals(self, data):
        # Calculate Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data)
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        signals[data['close'] < bb_lower] = 1   # Buy signal
        signals[data['close'] > bb_upper] = -1  # Sell signal
        
        return signals
```

## 🤖 AI Agent Capabilities

### Research Agent
- **Idea Extraction**: Analyzes videos, articles, and text
- **Strategy Formulation**: Converts ideas into executable strategies
- **Market Analysis**: Identifies relevant market conditions
- **Literature Review**: Scans academic and industry research

### Backtest Agent
- **Historical Testing**: Tests strategies on historical data
- **Performance Evaluation**: Calculates comprehensive metrics
- **Parameter Optimization**: Finds optimal strategy parameters
- **Robustness Testing**: Validates strategy stability

### Package Agent
- **Dependency Management**: Ensures all required packages are available
- **Environment Setup**: Configures trading environment
- **Data Source Management**: Manages market data connections
- **System Health**: Monitors system performance

## 🔄 Workflow

1. **Research Phase**: AI agents continuously scan for new trading ideas
2. **Analysis Phase**: Ideas are converted into testable strategies
3. **Backtesting Phase**: Strategies are validated on historical data
4. **Selection Phase**: Only proven strategies are selected for deployment
5. **Implementation Phase**: Strategies are deployed with risk management
6. **Monitoring Phase**: Continuous performance tracking and optimization

## 📊 Example Output

```
=== Systematic AI Trading Framework ===

🤖 AI Agents Status:
✅ Research Agent: Active (scanning for new ideas)
✅ Backtest Agent: Active (testing 3 strategies)
✅ Package Agent: Active (dependencies up to date)

📈 Strategy Performance:
Momentum Strategy: Sharpe=1.85, Win Rate=62%, Max DD=8.2%
Mean Reversion: Sharpe=1.42, Win Rate=58%, Max DD=12.1%
Regime Detection: Sharpe=2.01, Win Rate=65%, Max DD=6.8%

💰 Portfolio Status:
Total Value: $125,430
Daily P&L: +$2,340 (+1.9%)
Open Positions: 8
Risk Level: Medium

🔍 Recent Discoveries:
- New momentum strategy from YouTube analysis
- Mean reversion opportunity in tech sector
- Regime change detected in crypto markets
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the example strategies

---

**Remember**: The edge comes from systematic, automated processes—not from trying to outguess the market. Use AI to scale your research and testing, and let only the best strategies reach live trading. 