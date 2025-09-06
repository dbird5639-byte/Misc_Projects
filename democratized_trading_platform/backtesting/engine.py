"""
Advanced Backtesting Engine for Democratized Trading Platform
Makes sophisticated backtesting accessible to everyone, not just quants.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import logging

from .data_manager import DataManager
from .performance_analyzer import PerformanceAnalyzer
from .risk_manager import RiskManager


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005   # 0.05%
    data_source: str = "yfinance"
    symbols: List[str] = None
    benchmark: str = "SPY"
    risk_free_rate: float = 0.02
    walk_forward_periods: int = 12
    min_trades: int = 30
    max_drawdown_limit: float = 0.20


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    equity_curve: pd.Series
    trade_log: pd.DataFrame
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    walk_forward_results: Optional[Dict[str, Any]] = None


class BacktestEngine:
    """
    Advanced backtesting engine designed for accessibility and education.
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_manager = DataManager(config.data_source)
        self.performance_analyzer = PerformanceAnalyzer(config.risk_free_rate)
        self.risk_manager = RiskManager(config.max_drawdown_limit)
        
        # Backtest state
        self.current_capital = config.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
        self.logger.info("Backtest Engine initialized")
    
    def run_backtest(self, strategy_func: Callable, strategy_params: Dict[str, Any] = None) -> BacktestResult:
        """
        Run a complete backtest for a strategy.
        
        Args:
            strategy_func: Function that generates trading signals
            strategy_params: Parameters for the strategy
            
        Returns:
            Comprehensive backtest results
        """
        self.logger.info(f"Starting backtest for strategy: {strategy_func.__name__}")
        
        try:
            # Get historical data
            data = self._get_historical_data()
            
            # Initialize backtest
            self._initialize_backtest()
            
            # Run backtest
            self._execute_backtest(data, strategy_func, strategy_params)
            
            # Calculate results
            results = self._calculate_results(strategy_func.__name__)
            
            # Run walk-forward analysis if requested
            if self.config.walk_forward_periods > 1:
                results.walk_forward_results = self._run_walk_forward_analysis(
                    data, strategy_func, strategy_params
                )
            
            self.logger.info(f"Backtest completed: {results.total_return:.2%} return")
            return results
            
        except Exception as e:
            self.logger.error(f"Backtest error: {e}")
            raise
    
    def _get_historical_data(self) -> pd.DataFrame:
        """Get historical market data."""
        symbols = self.config.symbols or ["SPY"]
        
        all_data = {}
        for symbol in symbols:
            data = self.data_manager.get_data(
                symbol=symbol,
                start_date=self.config.start_date,
                end_date=self.config.end_date
            )
            all_data[symbol] = data
        
        # Combine data if multiple symbols
        if len(all_data) == 1:
            return list(all_data.values())[0]
        else:
            return self._combine_multi_symbol_data(all_data)
    
    def _combine_multi_symbol_data(self, all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine data from multiple symbols."""
        # For simplicity, use the first symbol's data
        # In a real implementation, you'd combine them based on strategy needs
        return list(all_data.values())[0]
    
    def _initialize_backtest(self):
        """Initialize backtest state."""
        self.current_capital = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
    
    def _execute_backtest(
        self, 
        data: pd.DataFrame, 
        strategy_func: Callable, 
        strategy_params: Dict[str, Any]
    ):
        """Execute the backtest."""
        for timestamp, row in data.iterrows():
            # Update current market data
            current_data = data.loc[:timestamp]
            
            # Generate signals
            signals = strategy_func(current_data, strategy_params or {})
            current_signal = signals.iloc[-1] if len(signals) > 0 else 0
            
            # Process existing positions
            self._process_positions(row, timestamp)
            
            # Process new signals
            if current_signal != 0:
                self._process_signals(current_signal, row, timestamp)
            
            # Record equity
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': self.current_capital + self._calculate_positions_value(row)
            })
        
        # Close any remaining positions
        if self.positions:
            self._close_all_positions(data.iloc[-1], data.index[-1])
    
    def _process_positions(self, current_data: pd.Series, timestamp: datetime):
        """Process existing positions and check for exits."""
        positions_to_remove = []
        
        for symbol, position in self.positions.items():
            # Update position with current price
            current_price = current_data['close']
            position['current_price'] = current_price
            position['current_time'] = timestamp
            
            # Check if position should be closed
            should_exit, exit_reason = self.risk_manager.should_exit_position(position)
            
            if should_exit:
                # Close position
                exit_price = current_price * (1 - self.config.slippage) if position['side'] == 'long' else current_price * (1 + self.config.slippage)
                
                # Calculate P&L
                if position['side'] == 'long':
                    pnl = (exit_price - position['entry_price']) * position['quantity']
                else:
                    pnl = (position['entry_price'] - exit_price) * position['quantity']
                
                # Apply commission
                commission_cost = exit_price * position['quantity'] * self.config.commission
                pnl -= commission_cost
                
                # Update capital
                self.current_capital += pnl
                
                # Record trade
                trade = {
                    'entry_time': position['entry_time'],
                    'exit_time': timestamp,
                    'symbol': symbol,
                    'side': position['side'],
                    'quantity': position['quantity'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': pnl / (position['entry_price'] * position['quantity']),
                    'commission': commission_cost,
                    'exit_reason': exit_reason
                }
                self.trades.append(trade)
                
                # Mark for removal
                positions_to_remove.append(symbol)
        
        # Remove closed positions
        for symbol in positions_to_remove:
            del self.positions[symbol]
    
    def _process_signals(self, signal: float, current_data: pd.Series, timestamp: datetime):
        """Process new trading signals."""
        symbol = current_data.name if hasattr(current_data, 'name') else 'UNKNOWN'
        
        # Determine signal type
        if signal > 0:  # Buy signal
            side = 'long'
            entry_price = current_data['close'] * (1 + self.config.slippage)
        else:  # Sell signal
            side = 'short'
            entry_price = current_data['close'] * (1 - self.config.slippage)
        
        # Calculate position size using risk management
        position_size = self.risk_manager.calculate_position_size(
            self.current_capital, entry_price, signal
        )
        quantity = position_size / entry_price
        
        # Check if we have enough capital
        required_capital = position_size + (position_size * self.config.commission)
        if required_capital > self.current_capital:
            # Reduce position size to fit available capital
            quantity = self.current_capital / (entry_price * (1 + self.config.commission))
            position_size = quantity * entry_price
        
        if quantity > 0:
            # Apply commission
            commission_cost = position_size * self.config.commission
            self.current_capital -= commission_cost
            
            # Create position
            position = {
                'side': side,
                'quantity': quantity,
                'entry_price': entry_price,
                'entry_time': timestamp,
                'current_price': entry_price,
                'current_time': timestamp
            }
            
            self.positions[symbol] = position
    
    def _close_all_positions(self, final_data: pd.Series, timestamp: datetime):
        """Close all remaining positions at the end of backtest."""
        for symbol, position in self.positions.items():
            # Close position at final price
            final_price = final_data['close']
            exit_price = final_price * (1 - self.config.slippage) if position['side'] == 'long' else final_price * (1 + self.config.slippage)
            
            # Calculate P&L
            if position['side'] == 'long':
                pnl = (exit_price - position['entry_price']) * position['quantity']
            else:
                pnl = (position['entry_price'] - exit_price) * position['quantity']
            
            # Apply commission
            commission_cost = exit_price * position['quantity'] * self.config.commission
            pnl -= commission_cost
            
            # Update capital
            self.current_capital += pnl
            
            # Record trade
            trade = {
                'entry_time': position['entry_time'],
                'exit_time': timestamp,
                'symbol': symbol,
                'side': position['side'],
                'quantity': position['quantity'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_pct': pnl / (position['entry_price'] * position['quantity']),
                'commission': commission_cost,
                'exit_reason': 'end_of_backtest'
            }
            self.trades.append(trade)
    
    def _calculate_positions_value(self, current_data: pd.Series) -> float:
        """Calculate current value of all positions."""
        total_value = 0.0
        current_price = current_data['close']
        
        for position in self.positions.values():
            if position['side'] == 'long':
                total_value += position['quantity'] * current_price
            else:  # short
                total_value += position['quantity'] * (2 * position['entry_price'] - current_price)
        
        return total_value
    
    def _calculate_results(self, strategy_name: str) -> BacktestResult:
        """Calculate comprehensive backtest results."""
        
        # Basic metrics
        total_return = (self.current_capital - self.config.initial_capital) / self.config.initial_capital
        
        # Calculate time period
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)
        years = (end_date - start_date).days / 365.25
        
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Create equity curve DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate performance metrics
        performance_metrics = self.performance_analyzer.calculate_metrics(
            equity_df['equity'], self.trades
        )
        
        # Calculate risk metrics
        risk_metrics = self.risk_manager.calculate_risk_metrics(
            equity_df['equity'], self.trades
        )
        
        # Create trade log
        trade_log = pd.DataFrame(self.trades)
        
        return BacktestResult(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.config.initial_capital,
            final_capital=self.current_capital,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=performance_metrics['sharpe_ratio'],
            sortino_ratio=performance_metrics['sortino_ratio'],
            max_drawdown=risk_metrics['max_drawdown'],
            calmar_ratio=annualized_return / risk_metrics['max_drawdown'] if risk_metrics['max_drawdown'] != 0 else 0,
            win_rate=performance_metrics['win_rate'],
            profit_factor=performance_metrics['profit_factor'],
            total_trades=len(self.trades),
            avg_trade_duration=self._calculate_avg_trade_duration(),
            equity_curve=equity_df['equity'],
            trade_log=trade_log,
            performance_metrics=performance_metrics,
            risk_metrics=risk_metrics
        )
    
    def _calculate_avg_trade_duration(self) -> float:
        """Calculate average trade duration in days."""
        if not self.trades:
            return 0.0
        
        durations = []
        for trade in self.trades:
            duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / (24 * 3600)
            durations.append(duration)
        
        return np.mean(durations)
    
    def _run_walk_forward_analysis(
        self, 
        data: pd.DataFrame, 
        strategy_func: Callable, 
        strategy_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run walk-forward analysis to prevent overfitting."""
        self.logger.info("Running walk-forward analysis...")
        
        # Split data into training and testing periods
        periods = self._create_walk_forward_periods(data)
        
        results = []
        for i, (train_data, test_data) in enumerate(periods):
            # Train on training data
            train_result = self._run_period_backtest(train_data, strategy_func, strategy_params)
            
            # Test on out-of-sample data
            test_result = self._run_period_backtest(test_data, strategy_func, strategy_params)
            
            results.append({
                'period': i,
                'train_sharpe': train_result.sharpe_ratio,
                'test_sharpe': test_result.sharpe_ratio,
                'train_return': train_result.total_return,
                'test_return': test_result.total_return,
                'stability': test_result.sharpe_ratio / train_result.sharpe_ratio if train_result.sharpe_ratio > 0 else 0
            })
        
        # Calculate walk-forward metrics
        avg_train_sharpe = np.mean([r['train_sharpe'] for r in results])
        avg_test_sharpe = np.mean([r['test_sharpe'] for r in results])
        stability_score = avg_test_sharpe / avg_train_sharpe if avg_train_sharpe > 0 else 0
        
        return {
            'periods': results,
            'avg_train_sharpe': avg_train_sharpe,
            'avg_test_sharpe': avg_test_sharpe,
            'stability_score': stability_score,
            'is_stable': stability_score > 0.7  # 70% stability threshold
        }
    
    def _create_walk_forward_periods(self, data: pd.DataFrame) -> List[tuple]:
        """Create walk-forward analysis periods."""
        periods = []
        total_length = len(data)
        period_length = total_length // self.config.walk_forward_periods
        
        for i in range(self.config.walk_forward_periods - 1):
            train_end = (i + 1) * period_length
            test_end = (i + 2) * period_length
            
            train_data = data.iloc[:train_end]
            test_data = data.iloc[train_end:test_end]
            
            periods.append((train_data, test_data))
        
        return periods
    
    def _run_period_backtest(
        self, 
        data: pd.DataFrame, 
        strategy_func: Callable, 
        strategy_params: Dict[str, Any]
    ) -> BacktestResult:
        """Run backtest for a specific period."""
        # Create temporary config for this period
        temp_config = BacktestConfig(
            start_date=data.index[0].strftime('%Y-%m-%d'),
            end_date=data.index[-1].strftime('%Y-%m-%d'),
            initial_capital=self.config.initial_capital,
            commission=self.config.commission,
            slippage=self.config.slippage
        )
        
        # Create temporary engine
        temp_engine = BacktestEngine(temp_config)
        
        # Run backtest
        return temp_engine.run_backtest(strategy_func, strategy_params)
    
    def generate_report(self, result: BacktestResult) -> str:
        """Generate a comprehensive backtest report."""
        report = f"""
# Backtest Report: {result.strategy_name}

## Summary
- **Period**: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}
- **Initial Capital**: ${result.initial_capital:,.2f}
- **Final Capital**: ${result.final_capital:,.2f}
- **Total Return**: {result.total_return:.2%}
- **Annualized Return**: {result.annualized_return:.2%}

## Performance Metrics
- **Sharpe Ratio**: {result.sharpe_ratio:.3f}
- **Sortino Ratio**: {result.sortino_ratio:.3f}
- **Calmar Ratio**: {result.calmar_ratio:.3f}
- **Win Rate**: {result.win_rate:.2%}
- **Profit Factor**: {result.profit_factor:.3f}

## Risk Metrics
- **Maximum Drawdown**: {result.max_drawdown:.2%}
- **Total Trades**: {result.total_trades}
- **Average Trade Duration**: {result.avg_trade_duration:.1f} days

## Trade Analysis
- **Average Win**: ${result.performance_metrics.get('avg_win', 0):.2f}
- **Average Loss**: ${result.performance_metrics.get('avg_loss', 0):.2f}
- **Largest Win**: ${result.performance_metrics.get('largest_win', 0):.2f}
- **Largest Loss**: ${result.performance_metrics.get('largest_loss', 0):.2f}

## Walk-Forward Analysis
"""
        
        if result.walk_forward_results:
            wf = result.walk_forward_results
            report += f"""
- **Average Train Sharpe**: {wf['avg_train_sharpe']:.3f}
- **Average Test Sharpe**: {wf['avg_test_sharpe']:.3f}
- **Stability Score**: {wf['stability_score']:.3f}
- **Strategy Stability**: {'Stable' if wf['is_stable'] else 'Unstable'}
"""
        
        report += """
## Recommendations
"""
        
        if result.sharpe_ratio < 1.0:
            report += "- Consider improving risk-adjusted returns\n"
        
        if result.max_drawdown > 0.20:
            report += "- Implement stricter risk management\n"
        
        if result.win_rate < 0.50:
            report += "- Focus on improving win rate or profit factor\n"
        
        if result.walk_forward_results and not result.walk_forward_results['is_stable']:
            report += "- Strategy may be overfitted, consider simplifying\n"
        
        return report


# Example strategy functions
def momentum_strategy(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Simple momentum strategy for beginners."""
    lookback = params.get('lookback', 20)
    threshold = params.get('threshold', 0.02)
    
    data = data.copy()
    data['returns'] = data['close'].pct_change()
    data['momentum'] = data['returns'].rolling(lookback).mean()
    
    signals = pd.Series(0, index=data.index)
    signals[data['momentum'] > threshold] = 1
    signals[data['momentum'] < -threshold] = -1
    
    return signals


def mean_reversion_strategy(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Mean reversion strategy using Bollinger Bands."""
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2)
    
    data = data.copy()
    
    # Calculate Bollinger Bands
    bb_middle = data['close'].rolling(bb_period).mean()
    bb_std_dev = data['close'].rolling(bb_period).std()
    bb_upper = bb_middle + (bb_std_dev * bb_std)
    bb_lower = bb_middle - (bb_std_dev * bb_std)
    
    signals = pd.Series(0, index=data.index)
    signals[data['close'] < bb_lower] = 1
    signals[data['close'] > bb_upper] = -1
    
    return signals


# Example usage
if __name__ == "__main__":
    # Create backtest configuration
    config = BacktestConfig(
        start_date="2020-01-01",
        end_date="2024-01-01",
        initial_capital=100000.0,
        symbols=["SPY"]
    )
    
    # Create backtest engine
    engine = BacktestEngine(config)
    
    # Run backtest for momentum strategy
    momentum_params = {"lookback": 20, "threshold": 0.02}
    momentum_result = engine.run_backtest(momentum_strategy, momentum_params)
    
    # Generate report
    report = engine.generate_report(momentum_result)
    print(report) 