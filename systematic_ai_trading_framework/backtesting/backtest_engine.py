"""
Backtesting Engine for the Systematic AI Trading Framework.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

from strategies.base_strategy import BaseStrategy
from config.settings import Settings
from utils.logger import setup_logger


@dataclass
class Trade:
    """Represents a completed trade."""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: str  # 'long', 'short'
    quantity: float
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    strategy_name: str
    exit_reason: str
    metadata: Dict[str, Any]


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    equity_curve: pd.Series
    trade_log: pd.DataFrame
    daily_returns: pd.Series
    metadata: Dict[str, Any]


class BacktestEngine:
    """
    Engine for backtesting trading strategies on historical data.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = setup_logger("backtest_engine", settings.log_level)
        
        # Default parameters
        self.default_commission = 0.001  # 0.1%
        self.default_slippage = 0.0005   # 0.05%
        
        self.logger.info("Backtest Engine initialized")
    
    async def run_backtest(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        initial_capital: float = 100000.0,
        commission: Optional[float] = None,
        slippage: Optional[float] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Run a complete backtest for a strategy.
        
        Args:
            strategy: Trading strategy to backtest
            data: Historical market data
            initial_capital: Starting capital
            commission: Commission rate (default from settings)
            slippage: Slippage rate (default from settings)
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Dictionary containing backtest results
        """
        self.logger.info(f"Starting backtest for strategy: {strategy.name}")
        
        # Set defaults
        commission = commission or self.settings.backtest_config.commission
        slippage = slippage or self.settings.backtest_config.slippage
        
        # Filter data by date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        if len(data) == 0:
            raise ValueError("No data available for backtest period")
        
        # Initialize backtest state
        capital = initial_capital
        positions = {}
        trades = []
        equity_curve = []
        
        # Reset strategy
        strategy.reset()
        
        # Run backtest
        for timestamp, row in data.iterrows():
            # Update current market data
            current_data = data.loc[:timestamp]
            
            # Generate signals
            signals = strategy.calculate_signals(current_data)
            current_signal = signals.iloc[-1] if len(signals) > 0 else 0
            
            # Process existing positions
            capital, positions, new_trades = self._process_positions(
                positions, row, timestamp, capital, commission, slippage, strategy
            )
            trades.extend(new_trades)
            
            # Process new signals
            if current_signal != 0:
                capital, positions, new_trades = self._process_signals(
                    current_signal, row, timestamp, capital, positions, 
                    commission, slippage, strategy
                )
                trades.extend(new_trades)
            
            # Record equity
            equity_curve.append({
                'timestamp': timestamp,
                'equity': capital + self._calculate_positions_value(positions, row)
            })
        
        # Close any remaining positions
        if positions:
            capital, positions, final_trades = self._close_all_positions(
                positions, data.iloc[-1], data.index[-1], capital, commission, slippage, strategy
            )
            trades.extend(final_trades)
        
        # Calculate results
        results = self._calculate_results(
            strategy.name, data.index[0], data.index[-1], initial_capital, 
            capital, trades, equity_curve
        )
        
        self.logger.info(f"Backtest completed for {strategy.name}: {results['total_return']:.2%} return")
        
        return results
    
    def _process_positions(
        self,
        positions: Dict[str, Dict],
        current_data: pd.Series,
        timestamp: datetime,
        capital: float,
        commission: float,
        slippage: float,
        strategy: BaseStrategy
    ) -> Tuple[float, Dict, List[Trade]]:
        """Process existing positions and check for exits."""
        trades = []
        positions_to_remove = []
        
        for symbol, position in positions.items():
            # Update position with current price
            current_price = current_data['close']
            position['current_price'] = current_price
            position['current_time'] = timestamp
            
            # Check if position should be closed
            should_exit, exit_reason = strategy.should_exit_position(
                self._dict_to_position(position), pd.DataFrame([current_data])
            )
            
            if should_exit:
                # Close position
                exit_price = current_price * (1 - slippage) if position['side'] == 'long' else current_price * (1 + slippage)
                
                # Calculate P&L
                if position['side'] == 'long':
                    pnl = (exit_price - position['entry_price']) * position['quantity']
                else:
                    pnl = (position['entry_price'] - exit_price) * position['quantity']
                
                # Apply commission
                commission_cost = exit_price * position['quantity'] * commission
                pnl -= commission_cost
                
                # Update capital
                capital += pnl
                
                # Record trade
                trade = Trade(
                    entry_time=position['entry_time'],
                    exit_time=timestamp,
                    symbol=symbol,
                    side=position['side'],
                    quantity=position['quantity'],
                    entry_price=position['entry_price'],
                    exit_price=exit_price,
                    pnl=pnl,
                    pnl_pct=pnl / (position['entry_price'] * position['quantity']),
                    commission=commission_cost,
                    slippage=slippage,
                    strategy_name=strategy.name,
                    exit_reason=exit_reason,
                    metadata={}
                )
                trades.append(trade)
                
                # Mark for removal
                positions_to_remove.append(symbol)
        
        # Remove closed positions
        for symbol in positions_to_remove:
            del positions[symbol]
        
        return capital, positions, trades
    
    def _process_signals(
        self,
        signal: float,
        current_data: pd.Series,
        timestamp: datetime,
        capital: float,
        positions: Dict[str, Dict],
        commission: float,
        slippage: float,
        strategy: BaseStrategy
    ) -> Tuple[float, Dict, List[Trade]]:
        """Process new trading signals."""
        trades = []
        symbol = current_data.name if hasattr(current_data, 'name') else 'UNKNOWN'
        
        # Determine signal type
        if signal > 0:  # Buy signal
            side = 'long'
            entry_price = current_data['close'] * (1 + slippage)
        else:  # Sell signal
            side = 'short'
            entry_price = current_data['close'] * (1 - slippage)
        
        # Calculate position size
        signal_strength = abs(signal)
        position_size = strategy.calculate_position_size(signal_strength, capital)
        quantity = position_size / entry_price
        
        # Check if we have enough capital
        required_capital = position_size + (position_size * commission)
        if required_capital > capital:
            # Reduce position size to fit available capital
            quantity = capital / (entry_price * (1 + commission))
            position_size = quantity * entry_price
        
        if quantity > 0:
            # Apply commission
            commission_cost = position_size * commission
            capital -= commission_cost
            
            # Create position
            position = {
                'side': side,
                'quantity': quantity,
                'entry_price': entry_price,
                'entry_time': timestamp,
                'current_price': entry_price,
                'current_time': timestamp
            }
            
            positions[symbol] = position
            
            # Record trade entry
            trade = Trade(
                entry_time=timestamp,
                exit_time=timestamp,  # Will be updated when closed
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                exit_price=entry_price,  # Will be updated when closed
                pnl=0.0,
                pnl_pct=0.0,
                commission=commission_cost,
                slippage=slippage,
                strategy_name=strategy.name,
                exit_reason='open',
                metadata={'signal_strength': signal_strength}
            )
            trades.append(trade)
        
        return capital, positions, trades
    
    def _close_all_positions(
        self,
        positions: Dict[str, Dict],
        final_data: pd.Series,
        timestamp: datetime,
        capital: float,
        commission: float,
        slippage: float,
        strategy: BaseStrategy
    ) -> Tuple[float, Dict, List[Trade]]:
        """Close all remaining positions at the end of backtest."""
        trades = []
        
        for symbol, position in positions.items():
            # Close position at final price
            final_price = final_data['close']
            exit_price = final_price * (1 - slippage) if position['side'] == 'long' else final_price * (1 + slippage)
            
            # Calculate P&L
            if position['side'] == 'long':
                pnl = (exit_price - position['entry_price']) * position['quantity']
            else:
                pnl = (position['entry_price'] - exit_price) * position['quantity']
            
            # Apply commission
            commission_cost = exit_price * position['quantity'] * commission
            pnl -= commission_cost
            
            # Update capital
            capital += pnl
            
            # Record trade
            trade = Trade(
                entry_time=position['entry_time'],
                exit_time=timestamp,
                symbol=symbol,
                side=position['side'],
                quantity=position['quantity'],
                entry_price=position['entry_price'],
                exit_price=exit_price,
                pnl=pnl,
                pnl_pct=pnl / (position['entry_price'] * position['quantity']),
                commission=commission_cost,
                slippage=slippage,
                strategy_name=strategy.name,
                exit_reason='end_of_backtest',
                metadata={}
            )
            trades.append(trade)
        
        return capital, {}, trades
    
    def _calculate_positions_value(self, positions: Dict[str, Dict], current_data: pd.Series) -> float:
        """Calculate current value of all positions."""
        total_value = 0.0
        current_price = current_data['close']
        
        for position in positions.values():
            if position['side'] == 'long':
                total_value += position['quantity'] * current_price
            else:  # short
                total_value += position['quantity'] * (2 * position['entry_price'] - current_price)
        
        return total_value
    
    def _dict_to_position(self, position_dict: Dict) -> 'Position':
        """Convert position dictionary to Position object."""
        from strategies.base_strategy import Position
        return Position(
            symbol=position_dict.get('symbol', ''),
            side=position_dict['side'],
            quantity=position_dict['quantity'],
            entry_price=position_dict['entry_price'],
            entry_time=position_dict['entry_time'],
            current_price=position_dict['current_price'],
            current_time=position_dict['current_time'],
            pnl=0.0,
            pnl_pct=0.0,
            metadata={}
        )
    
    def _calculate_results(
        self,
        strategy_name: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float,
        final_capital: float,
        trades: List[Trade],
        equity_curve: List[Dict]
    ) -> Dict[str, Any]:
        """Calculate comprehensive backtest results."""
        
        # Basic metrics
        total_return = (final_capital - initial_capital) / initial_capital
        total_trades = len([t for t in trades if t.exit_reason != 'open'])
        winning_trades = len([t for t in trades if t.pnl > 0 and t.exit_reason != 'open'])
        losing_trades = len([t for t in trades if t.pnl < 0 and t.exit_reason != 'open'])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate average win/loss
        winning_pnls = [t.pnl for t in trades if t.pnl > 0 and t.exit_reason != 'open']
        losing_pnls = [t.pnl for t in trades if t.pnl < 0 and t.exit_reason != 'open']
        
        avg_win = np.mean(winning_pnls) if winning_pnls else 0
        avg_loss = np.mean(losing_pnls) if losing_pnls else 0
        
        # Profit factor
        gross_profit = sum(winning_pnls)
        gross_loss = abs(sum(losing_pnls))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Create equity curve DataFrame
        equity_df = pd.DataFrame(equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate drawdown
        equity_df['peak'] = equity_df['equity'].expanding().max()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].min()
        
        # Calculate returns
        equity_df['returns'] = equity_df['equity'].pct_change()
        daily_returns = equity_df['returns'].resample('D').sum()
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        sortino_ratio = self._calculate_sortino_ratio(daily_returns)
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Create trade log
        trade_log = pd.DataFrame([
            {
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'symbol': t.symbol,
                'side': t.side,
                'quantity': t.quantity,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'commission': t.commission,
                'exit_reason': t.exit_reason
            }
            for t in trades if t.exit_reason != 'open'
        ])
        
        return {
            'strategy_name': strategy_name,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'equity_curve': equity_df['equity'],
            'trade_log': trade_log,
            'daily_returns': daily_returns,
            'metadata': {
                'total_commission': sum(t.commission for t in trades),
                'total_slippage': sum(t.slippage * t.quantity * t.exit_price for t in trades if t.exit_reason != 'open'),
                'avg_trade_duration': self._calculate_avg_trade_duration(trades)
            }
        }
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        return excess_returns.mean() / downside_returns.std() * np.sqrt(252)
    
    def _calculate_avg_trade_duration(self, trades: List[Trade]) -> float:
        """Calculate average trade duration in days."""
        durations = []
        for trade in trades:
            if trade.exit_reason != 'open':
                duration = (trade.exit_time - trade.entry_time).total_seconds() / (24 * 3600)
                durations.append(duration)
        
        return np.mean(durations) if durations else 0.0
    
    async def run_parameter_optimization(
        self,
        strategy_class,
        data: pd.DataFrame,
        parameter_ranges: Dict[str, List],
        initial_capital: float = 100000.0,
        optimization_metric: str = 'sharpe_ratio'
    ) -> Dict[str, Any]:
        """
        Run parameter optimization for a strategy.
        
        Args:
            strategy_class: Strategy class to optimize
            data: Historical market data
            parameter_ranges: Dictionary of parameter ranges to test
            initial_capital: Starting capital
            optimization_metric: Metric to optimize for
            
        Returns:
            Optimization results
        """
        self.logger.info(f"Starting parameter optimization for {strategy_class.__name__}")
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations(parameter_ranges)
        
        results = []
        best_result = None
        best_metric = float('-inf')
        
        # Test each parameter combination
        for i, params in enumerate(param_combinations):
            try:
                # Create strategy with parameters
                strategy = strategy_class(f"{strategy_class.__name__}_opt_{i}", params)
                
                # Run backtest
                backtest_result = await self.run_backtest(
                    strategy, data, initial_capital
                )
                
                # Get optimization metric
                metric_value = backtest_result.get(optimization_metric, 0)
                
                result = {
                    'parameters': params,
                    'metric': metric_value,
                    'backtest_result': backtest_result
                }
                results.append(result)
                
                # Update best result
                if metric_value > best_metric:
                    best_metric = metric_value
                    best_result = result
                
                self.logger.info(f"Parameter set {i+1}/{len(param_combinations)}: {optimization_metric} = {metric_value:.3f}")
                
            except Exception as e:
                self.logger.error(f"Error testing parameter set {i}: {e}")
        
        return {
            'best_parameters': best_result['parameters'] if best_result else {},
            'best_metric': best_metric,
            'all_results': results,
            'parameter_ranges': parameter_ranges,
            'optimization_metric': optimization_metric
        }
    
    def _generate_parameter_combinations(self, parameter_ranges: Dict[str, List]) -> List[Dict]:
        """Generate all combinations of parameters."""
        import itertools
        
        # Get all parameter names and their values
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        # Convert to list of dictionaries
        result = []
        for combo in combinations:
            param_dict = dict(zip(param_names, combo))
            result.append(param_dict)
        
        return result 