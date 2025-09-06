"""
Backtesting Engine - RBI System Phase 2

This module implements the Backtest phase of the RBI system,
validating strategies against historical data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import yfinance as yf
from abc import ABC, abstractmethod

@dataclass
class Trade:
    """Data class for individual trades"""
    entry_date: datetime
    exit_date: datetime
    symbol: str
    entry_price: float
    exit_price: float
    quantity: int
    side: str  # "BUY" or "SELL"
    pnl: float
    return_pct: float

@dataclass
class BacktestResult:
    """Data class for backtest results"""
    strategy_name: str
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_return: float
    profit_factor: float
    trades: List[Trade]
    equity_curve: pd.Series

class Strategy(ABC):
    """Abstract base class for trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.positions = {}
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from market data"""
        pass
    
    def calculate_position_size(self, price: float, capital: float) -> int:
        """Calculate position size based on risk management"""
        # Simple 2% risk per trade
        risk_amount = capital * 0.02
        return int(risk_amount / price)

class MomentumStrategy(Strategy):
    """Momentum trading strategy"""
    
    def __init__(self, lookback_period: int = 20, threshold: float = 0.02):
        super().__init__("Momentum Strategy")
        self.lookback_period = lookback_period
        self.threshold = threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate momentum signals"""
        signals = pd.Series(0, index=data.index)
        
        # Calculate momentum
        momentum = data['Close'].pct_change(self.lookback_period)
        
        # Generate signals
        signals[momentum > self.threshold] = 1   # Buy signal
        signals[momentum < -self.threshold] = -1 # Sell signal
        
        return signals

class MeanReversionStrategy(Strategy):
    """Mean reversion trading strategy"""
    
    def __init__(self, lookback_period: int = 50, std_dev_threshold: float = 2.0):
        super().__init__("Mean Reversion Strategy")
        self.lookback_period = lookback_period
        self.std_dev_threshold = std_dev_threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate mean reversion signals"""
        signals = pd.Series(0, index=data.index)
        
        # Calculate rolling mean and standard deviation
        rolling_mean = data['Close'].rolling(self.lookback_period).mean()
        rolling_std = data['Close'].rolling(self.lookback_period).std()
        
        # Calculate z-score
        z_score = (data['Close'] - rolling_mean) / rolling_std
        
        # Generate signals
        signals[z_score < -self.std_dev_threshold] = 1   # Buy signal (oversold)
        signals[z_score > self.std_dev_threshold] = -1   # Sell signal (overbought)
        
        return signals

class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
    
    def run_backtest(self, strategy: Strategy, data: pd.DataFrame, 
                    symbol: str) -> BacktestResult:
        """Run backtest for a strategy"""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Process each day
        for i, (date, row) in enumerate(data.iterrows()):
            signal = signals.iloc[i]
            price = float(row['Close'])
            
            # Execute trades based on signals
            if signal == 1 and symbol not in self.positions:  # Buy signal
                quantity = strategy.calculate_position_size(price, float(self.current_capital))
                if quantity > 0:
                    self.positions[symbol] = {
                        'quantity': quantity,
                        'entry_price': price,
                        'entry_date': date
                    }
                    self.current_capital -= quantity * price
            
            elif signal == -1 and symbol in self.positions:  # Sell signal
                position = self.positions[symbol]
                exit_price = price
                pnl = (exit_price - position['entry_price']) * position['quantity']
                
                # Record trade
                try:
                    timestamp = pd.Timestamp(str(date)).to_pydatetime()
                    if timestamp is None or str(timestamp) == 'NaT':
                        timestamp = datetime.now()
                except:
                    timestamp = datetime.now()
                
                # Type assertion to ensure timestamp is datetime
                assert isinstance(timestamp, datetime)
                
                trade = Trade(
                    entry_date=position['entry_date'],
                    exit_date=timestamp,
                    symbol=symbol,
                    entry_price=position['entry_price'],
                    exit_price=exit_price,
                    quantity=position['quantity'],
                    side="BUY",
                    pnl=pnl,
                    return_pct=(exit_price - position['entry_price']) / position['entry_price']
                )
                self.trades.append(trade)
                
                # Update capital
                self.current_capital += position['quantity'] * exit_price
                del self.positions[symbol]
            
            # Calculate current equity
            current_equity = self.current_capital
            for pos_symbol, position in self.positions.items():
                current_equity += position['quantity'] * price
            
            self.equity_curve.append(current_equity)
        
        # Close any remaining positions
        if symbol in self.positions:
            position = self.positions[symbol]
            exit_price = float(data['Close'].iloc[-1])
            pnl = (exit_price - position['entry_price']) * position['quantity']
            
            try:
                timestamp = pd.Timestamp(str(data.index[-1])).to_pydatetime()
                if timestamp is None or str(timestamp) == 'NaT':
                    timestamp = datetime.now()
            except:
                timestamp = datetime.now()
            
            # Type assertion to ensure timestamp is datetime
            assert isinstance(timestamp, datetime)
            
            trade = Trade(
                entry_date=position['entry_date'],
                exit_date=timestamp,
                symbol=symbol,
                entry_price=position['entry_price'],
                exit_price=exit_price,
                quantity=position['quantity'],
                side="BUY",
                pnl=pnl,
                return_pct=(exit_price - position['entry_price']) / position['entry_price']
            )
            self.trades.append(trade)
        
        # Calculate performance metrics
        return self._calculate_performance_metrics(strategy.name)
    
    def _calculate_performance_metrics(self, strategy_name: str) -> BacktestResult:
        """Calculate performance metrics from backtest results"""
        if not self.trades:
            return BacktestResult(
                strategy_name=strategy_name,
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_trades=0,
                avg_trade_return=0.0,
                profit_factor=0.0,
                trades=[],
                equity_curve=pd.Series()
            )
        
        # Calculate returns
        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital
        
        # Calculate trade statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        avg_trade_return = np.mean([t.return_pct for t in self.trades])
        
        # Calculate profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate Sharpe ratio (simplified)
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Calculate max drawdown
        equity_series = pd.Series(self.equity_curve)
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()
        
        # Annualized return
        days = len(self.equity_curve)
        annualized_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        
        return BacktestResult(
            strategy_name=strategy_name,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(self.trades),
            avg_trade_return=float(avg_trade_return),
            profit_factor=profit_factor,
            trades=self.trades,
            equity_curve=pd.Series(self.equity_curve)
        )
    
    def generate_backtest_report(self, results: List[BacktestResult]) -> str:
        """Generate backtest report"""
        report = "# Backtest Results Report\n\n"
        report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for result in results:
            report += f"## {result.strategy_name}\n\n"
            report += f"- **Total Return**: {result.total_return:.2%}\n"
            report += f"- **Annualized Return**: {result.annualized_return:.2%}\n"
            report += f"- **Sharpe Ratio**: {result.sharpe_ratio:.2f}\n"
            report += f"- **Max Drawdown**: {result.max_drawdown:.2%}\n"
            report += f"- **Win Rate**: {result.win_rate:.1%}\n"
            report += f"- **Total Trades**: {result.total_trades}\n"
            report += f"- **Avg Trade Return**: {result.avg_trade_return:.2%}\n"
            report += f"- **Profit Factor**: {result.profit_factor:.2f}\n\n"
        
        return report

def main():
    """Main function for running backtests"""
    # Download sample data
    data = yf.download("AAPL", start="2022-01-01", end="2023-12-31")
    
    if data is None or data.empty:
        print("Failed to download data")
        return
    
    # Initialize strategies
    momentum_strategy = MomentumStrategy(lookback_period=20, threshold=0.02)
    mean_reversion_strategy = MeanReversionStrategy(lookback_period=50, std_dev_threshold=2.0)
    
    # Initialize backtest engine
    engine = BacktestEngine(initial_capital=100000)
    
    # Run backtests
    results = []
    
    momentum_result = engine.run_backtest(momentum_strategy, data, "AAPL")
    results.append(momentum_result)
    
    mean_reversion_result = engine.run_backtest(mean_reversion_strategy, data, "AAPL")
    results.append(mean_reversion_result)
    
    # Generate report
    report = engine.generate_backtest_report(results)
    
    # Save report
    with open("backtest_report.md", "w") as f:
        f.write(report)
    
    print("Backtest completed. Check backtest_report.md for results.")

if __name__ == "__main__":
    main() 