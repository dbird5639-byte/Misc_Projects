"""
Multi-Strategy Portfolio Manager
Inspired by Kevin Davy's approach to multi-strategy trading

This module manages multiple strategies in a single account with:
- Position netting across strategies
- Correlation analysis and management
- Optimal risk allocation
- Performance attribution
- Real-time portfolio monitoring
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class PositionType(Enum):
    """Types of positions"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Position:
    """Position information"""
    symbol: str
    size: float
    entry_price: float
    current_price: float
    entry_time: datetime
    strategy: str
    position_type: PositionType
    unrealized_pnl: float
    realized_pnl: float = 0.0


@dataclass
class StrategyAllocation:
    """Strategy allocation configuration"""
    strategy_name: str
    allocation: float  # Percentage of portfolio
    max_allocation: float
    risk_per_trade: float
    correlation_threshold: float
    enabled: bool = True


@dataclass
class PortfolioConfig:
    """Portfolio configuration"""
    strategies: List[StrategyAllocation]
    total_capital: float
    max_drawdown: float
    max_correlation: float
    rebalance_frequency: str  # daily, weekly, monthly
    risk_per_trade: float
    position_sizing_method: str  # kelly, fixed, volatility


class PortfolioManager:
    """
    Multi-strategy portfolio manager
    
    Implements Kevin Davy's approach to managing multiple strategies
    in a single account with proper position netting and risk management.
    """
    
    def __init__(self, config: Optional[PortfolioConfig] = None):
        """Initialize portfolio manager"""
        self.config = config or self._default_config()
        
        # Portfolio state
        self.positions = {}  # symbol -> Position
        self.strategy_positions = {}  # strategy -> {symbol -> Position}
        self.equity_curve = []
        self.trades = []
        self.performance_history = []
        
        # Risk management
        self.current_drawdown = 0.0
        self.peak_equity = 0.0
        self.total_pnl = 0.0
        
        # Correlation tracking
        self.strategy_returns = {}
        self.correlation_matrix = pd.DataFrame()
        
        # Execution state
        self.is_trading = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("Portfolio Manager initialized")
    
    def _default_config(self) -> PortfolioConfig:
        """Create default portfolio configuration"""
        return PortfolioConfig(
            strategies=[],
            total_capital=100000.0,
            max_drawdown=0.15,
            max_correlation=0.7,
            rebalance_frequency="daily",
            risk_per_trade=0.02,
            position_sizing_method="kelly"
        )
    
    def add_strategy(
        self,
        strategy_name: str,
        allocation: float,
        max_allocation: float = 0.25,
        risk_per_trade: float = 0.02,
        correlation_threshold: float = 0.7
    ) -> None:
        """
        Add a strategy to the portfolio
        
        Args:
            strategy_name: Name of the strategy
            allocation: Target allocation percentage
            max_allocation: Maximum allocation percentage
            risk_per_trade: Risk per trade percentage
            correlation_threshold: Maximum correlation threshold
        """
        strategy = StrategyAllocation(
            strategy_name=strategy_name,
            allocation=allocation,
            max_allocation=max_allocation,
            risk_per_trade=risk_per_trade,
            correlation_threshold=correlation_threshold
        )
        
        self.config.strategies.append(strategy)
        self.strategy_positions[strategy_name] = {}
        
        logger.info(f"Added strategy: {strategy_name} with {allocation:.1%} allocation")
    
    def build_portfolio(
        self,
        strategies: Dict[str, Any],
        results: Dict[str, Any],
        max_correlation: float = 0.7,
        risk_per_trade: float = 0.02
    ) -> Dict[str, Any]:
        """
        Build optimal portfolio from strategies and test results
        
        Args:
            strategies: Dictionary of strategies
            results: Dictionary of test results
            max_correlation: Maximum allowed correlation
            risk_per_trade: Risk per trade percentage
            
        Returns:
            Portfolio configuration
        """
        logger.info("Building optimal portfolio")
        
        # Calculate strategy correlations
        correlation_matrix = self._calculate_strategy_correlations(results)
        
        # Filter strategies by correlation
        qualified_strategies = self._filter_by_correlation(
            strategies, correlation_matrix, max_correlation
        )
        
        # Calculate optimal allocations
        allocations = self._calculate_optimal_allocations(
            qualified_strategies, results, risk_per_trade
        )
        
        # Create portfolio configuration
        portfolio_config = {
            "strategies": qualified_strategies,
            "allocations": allocations,
            "correlation_matrix": correlation_matrix,
            "expected_return": self._calculate_expected_return(allocations, results),
            "expected_risk": self._calculate_expected_risk(allocations, results),
            "diversification_score": self._calculate_diversification_score(correlation_matrix)
        }
        
        logger.info(f"Portfolio built with {len(qualified_strategies)} strategies")
        logger.info(f"Expected return: {portfolio_config['expected_return']:.2%}")
        logger.info(f"Expected risk: {portfolio_config['expected_risk']:.2%}")
        
        return portfolio_config
    
    def _calculate_strategy_correlations(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Calculate correlation matrix between strategies"""
        # Extract equity curves
        equity_curves = {}
        for strategy_name, result in results.items():
            if hasattr(result, 'equity_curve') and len(result.equity_curve) > 0:
                equity_curves[strategy_name] = result.equity_curve
        
        if not equity_curves:
            return pd.DataFrame()
        
        # Align equity curves
        aligned_equity = pd.DataFrame(equity_curves).dropna()
        
        # Calculate returns
        returns = aligned_equity.pct_change().dropna()
        
        # Calculate correlation matrix
        correlation_matrix = returns.corr()
        
        return correlation_matrix
    
    def _filter_by_correlation(
        self,
        strategies: Dict[str, Any],
        correlation_matrix: pd.DataFrame,
        max_correlation: float
    ) -> List[str]:
        """Filter strategies by correlation threshold"""
        if correlation_matrix.empty:
            return list(strategies.keys())
        
        qualified_strategies = []
        used_strategies = set()
        
        # Sort strategies by Sharpe ratio (assuming it's available)
        strategy_scores = []
        for strategy_name in strategies.keys():
            if strategy_name in correlation_matrix.index:
                # Use a simple score based on correlation and performance
                score = 0
                if hasattr(strategies[strategy_name], 'sharpe_ratio'):
                    score += strategies[strategy_name].sharpe_ratio
                strategy_scores.append((strategy_name, score))
        
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        
        for strategy_name, score in strategy_scores:
            # Check correlation with already selected strategies
            max_corr = 0
            for used_strategy in used_strategies:
                if strategy_name in correlation_matrix.index and used_strategy in correlation_matrix.columns:
                    corr = abs(correlation_matrix.loc[strategy_name, used_strategy])
                    max_corr = max(max_corr, corr)
            
            if max_corr <= max_correlation:
                qualified_strategies.append(strategy_name)
                used_strategies.add(strategy_name)
                logger.info(f"Strategy {strategy_name} qualified (max correlation: {max_corr:.3f})")
            else:
                logger.info(f"Strategy {strategy_name} rejected (max correlation: {max_corr:.3f})")
        
        return qualified_strategies
    
    def _calculate_optimal_allocations(
        self,
        strategies: List[str],
        results: Dict[str, Any],
        risk_per_trade: float
    ) -> Dict[str, float]:
        """Calculate optimal strategy allocations"""
        # Simple equal allocation for now
        # In practice, this would use modern portfolio theory or Kelly criterion
        
        n_strategies = len(strategies)
        if n_strategies == 0:
            return {}
        
        # Equal allocation
        allocation = 1.0 / n_strategies
        
        allocations = {}
        for strategy in strategies:
            allocations[strategy] = allocation
        
        return allocations
    
    def _calculate_expected_return(self, allocations: Dict[str, float], results: Dict[str, Any]) -> float:
        """Calculate expected portfolio return"""
        expected_return = 0.0
        
        for strategy, allocation in allocations.items():
            if strategy in results and hasattr(results[strategy], 'total_return'):
                expected_return += allocation * results[strategy].total_return
        
        return expected_return
    
    def _calculate_expected_risk(self, allocations: Dict[str, float], results: Dict[str, Any]) -> float:
        """Calculate expected portfolio risk"""
        # Simplified risk calculation
        expected_risk = 0.0
        
        for strategy, allocation in allocations.items():
            if strategy in results and hasattr(results[strategy], 'max_drawdown'):
                expected_risk += allocation * abs(results[strategy].max_drawdown)
        
        return expected_risk
    
    def _calculate_diversification_score(self, correlation_matrix: pd.DataFrame) -> float:
        """Calculate portfolio diversification score"""
        if correlation_matrix.empty:
            return 1.0
        
        # Average correlation (lower is better)
        n_strategies = len(correlation_matrix)
        if n_strategies <= 1:
            return 1.0
        
        # Calculate average correlation excluding diagonal
        total_corr = 0
        count = 0
        
        for i in range(n_strategies):
            for j in range(i + 1, n_strategies):
                total_corr += abs(correlation_matrix.iloc[i, j])
                count += 1
        
        avg_correlation = total_corr / count if count > 0 else 0
        
        # Diversification score (1 - avg_correlation)
        diversification_score = 1 - avg_correlation
        
        return max(0, diversification_score)
    
    def run_backtest(
        self,
        portfolio_config: Dict[str, Any],
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Run backtest on the portfolio
        
        Args:
            portfolio_config: Portfolio configuration
            data: Market data for each symbol
            
        Returns:
            Backtest results
        """
        logger.info("Running portfolio backtest")
        
        # Initialize backtest
        self._initialize_backtest(portfolio_config)
        
        # Get all dates
        all_dates = set()
        for symbol_data in data.values():
            all_dates.update(symbol_data.index)
        
        dates = sorted(list(all_dates))
        
        # Run backtest day by day
        for date in dates:
            self._process_backtest_day(date, portfolio_config, data)
        
        # Calculate final results
        results = self._calculate_backtest_results()
        
        logger.info("Portfolio backtest completed")
        return results
    
    def _initialize_backtest(self, portfolio_config: Dict[str, Any]) -> None:
        """Initialize backtest state"""
        self.positions = {}
        self.strategy_positions = {}
        self.equity_curve = []
        self.trades = []
        self.performance_history = []
        self.current_drawdown = 0.0
        self.peak_equity = self.config.total_capital
        self.total_pnl = 0.0
    
    def _process_backtest_day(
        self,
        date: datetime,
        portfolio_config: Dict[str, Any],
        data: Dict[str, pd.DataFrame]
    ) -> None:
        """Process a single day in the backtest"""
        # Update positions with current prices
        self._update_positions(date, data)
        
        # Generate signals for each strategy
        for strategy_name in portfolio_config["strategies"]:
            if strategy_name in data:
                self._process_strategy_signals(strategy_name, date, data[strategy_name])
        
        # Net positions across strategies
        self._net_positions()
        
        # Update portfolio metrics
        self._update_portfolio_metrics(date)
    
    def _update_positions(self, date: datetime, data: Dict[str, pd.DataFrame]) -> None:
        """Update position prices"""
        for symbol, position in self.positions.items():
            if symbol in data and date in data[symbol].index:
                current_price = data[symbol].loc[date, 'close']
                position.current_price = current_price
                
                # Calculate unrealized PnL
                if position.position_type == PositionType.LONG:
                    position.unrealized_pnl = (current_price - position.entry_price) * position.size
                else:
                    position.unrealized_pnl = (position.entry_price - current_price) * position.size
    
    def _process_strategy_signals(
        self,
        strategy_name: str,
        date: datetime,
        data: pd.DataFrame
    ) -> None:
        """Process signals for a single strategy"""
        if date not in data.index:
            return
        
        # Generate signals (simplified)
        signal = self._generate_signal(data, date)
        
        if signal != 0:
            # Execute trade
            self._execute_trade(strategy_name, signal, data.loc[date, 'close'], date)
    
    def _generate_signal(self, data: pd.DataFrame, date: datetime) -> int:
        """Generate trading signal (simplified)"""
        # Simple moving average crossover
        if len(data) < 50:
            return 0
        
        current_idx = data.index.get_loc(date)
        if current_idx < 50:
            return 0
        
        sma_20 = data['close'].rolling(20).mean().iloc[current_idx]
        sma_50 = data['close'].rolling(50).mean().iloc[current_idx]
        
        if sma_20 > sma_50:
            return 1  # Buy signal
        elif sma_20 < sma_50:
            return -1  # Sell signal
        
        return 0
    
    def _execute_trade(
        self,
        strategy_name: str,
        signal: int,
        price: float,
        date: datetime
    ) -> None:
        """Execute a trade"""
        symbol = f"{strategy_name}_symbol"  # Simplified symbol naming
        
        # Calculate position size
        allocation = self._get_strategy_allocation(strategy_name)
        position_size = self._calculate_position_size(allocation, price)
        
        if signal == 1:  # Buy
            if symbol in self.positions and self.positions[symbol].position_type == PositionType.SHORT:
                # Close short position
                self._close_position(symbol, price, date)
            
            # Open long position
            position = Position(
                symbol=symbol,
                size=position_size,
                entry_price=price,
                current_price=price,
                entry_time=date,
                strategy=strategy_name,
                position_type=PositionType.LONG,
                unrealized_pnl=0.0
            )
            
            self.positions[symbol] = position
            self.strategy_positions[strategy_name][symbol] = position
            
        elif signal == -1:  # Sell
            if symbol in self.positions and self.positions[symbol].position_type == PositionType.LONG:
                # Close long position
                self._close_position(symbol, price, date)
            
            # Open short position
            position = Position(
                symbol=symbol,
                size=position_size,
                entry_price=price,
                current_price=price,
                entry_time=date,
                strategy=strategy_name,
                position_type=PositionType.SHORT,
                unrealized_pnl=0.0
            )
            
            self.positions[symbol] = position
            self.strategy_positions[strategy_name][symbol] = position
    
    def _close_position(self, symbol: str, price: float, date: datetime) -> None:
        """Close an existing position"""
        position = self.positions[symbol]
        
        # Calculate realized PnL
        if position.position_type == PositionType.LONG:
            realized_pnl = (price - position.entry_price) * position.size
        else:
            realized_pnl = (position.entry_price - price) * position.size
        
        # Record trade
        trade = {
            'entry_date': position.entry_time,
            'exit_date': date,
            'symbol': symbol,
            'strategy': position.strategy,
            'entry_price': position.entry_price,
            'exit_price': price,
            'size': position.size,
            'position_type': position.position_type.value,
            'realized_pnl': realized_pnl,
            'return': realized_pnl / (position.entry_price * position.size)
        }
        
        self.trades.append(trade)
        self.total_pnl += realized_pnl
        
        # Remove position
        del self.positions[symbol]
        if position.strategy in self.strategy_positions:
            del self.strategy_positions[position.strategy][symbol]
    
    def _net_positions(self) -> None:
        """Net positions across strategies (Kevin's approach)"""
        # Group positions by symbol
        symbol_positions = {}
        
        for position in self.positions.values():
            if position.symbol not in symbol_positions:
                symbol_positions[position.symbol] = []
            symbol_positions[position.symbol].append(position)
        
        # Net positions for each symbol
        for symbol, positions in symbol_positions.items():
            if len(positions) > 1:
                # Calculate net position
                net_size = 0
                net_entry_price = 0
                total_value = 0
                
                for position in positions:
                    if position.position_type == PositionType.LONG:
                        net_size += position.size
                        total_value += position.size * position.entry_price
                    else:
                        net_size -= position.size
                        total_value += position.size * position.entry_price
                
                if total_value != 0:
                    net_entry_price = total_value / abs(net_size)
                
                # Create net position
                if net_size != 0:
                    net_position_type = PositionType.LONG if net_size > 0 else PositionType.SHORT
                    
                    net_position = Position(
                        symbol=symbol,
                        size=abs(net_size),
                        entry_price=net_entry_price,
                        current_price=positions[0].current_price,
                        entry_time=positions[0].entry_time,
                        strategy="NET",
                        position_type=net_position_type,
                        unrealized_pnl=0.0
                    )
                    
                    # Replace individual positions with net position
                    self.positions[symbol] = net_position
                    
                    # Clear strategy positions
                    for position in positions:
                        if position.strategy in self.strategy_positions:
                            if symbol in self.strategy_positions[position.strategy]:
                                del self.strategy_positions[position.strategy][symbol]
    
    def _update_portfolio_metrics(self, date: datetime) -> None:
        """Update portfolio performance metrics"""
        # Calculate total portfolio value
        total_value = self.config.total_capital + self.total_pnl
        
        # Add unrealized PnL
        for position in self.positions.values():
            total_value += position.unrealized_pnl
        
        # Update peak equity
        if total_value > self.peak_equity:
            self.peak_equity = total_value
        
        # Calculate drawdown
        if self.peak_equity > 0:
            self.current_drawdown = (total_value - self.peak_equity) / self.peak_equity
        
        # Record equity curve
        self.equity_curve.append({
            'date': date,
            'equity': total_value,
            'drawdown': self.current_drawdown,
            'total_pnl': self.total_pnl
        })
    
    def _get_strategy_allocation(self, strategy_name: str) -> float:
        """Get strategy allocation"""
        for strategy in self.config.strategies:
            if strategy.strategy_name == strategy_name:
                return strategy.allocation
        return 0.0
    
    def _calculate_position_size(self, allocation: float, price: float) -> float:
        """Calculate position size based on allocation and risk"""
        capital = self.config.total_capital
        position_value = capital * allocation
        position_size = position_value / price
        
        return position_size
    
    def _calculate_backtest_results(self) -> Dict[str, Any]:
        """Calculate final backtest results"""
        if not self.equity_curve:
            return {}
        
        # Convert to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('date', inplace=True)
        
        # Calculate returns
        returns = equity_df['equity'].pct_change().dropna()
        
        # Calculate metrics
        total_return = (equity_df['equity'].iloc[-1] - equity_df['equity'].iloc[0]) / equity_df['equity'].iloc[0]
        max_drawdown = equity_df['drawdown'].min()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Trade metrics
        if self.trades:
            trade_pnls = [trade['realized_pnl'] for trade in self.trades]
            winning_trades = [p for p in trade_pnls if p > 0]
            losing_trades = [p for p in trade_pnls if p < 0]
            
            win_rate = len(winning_trades) / len(trade_pnls) if trade_pnls else 0
            avg_trade = np.mean(trade_pnls) if trade_pnls else 0
            profit_factor = sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else float('inf')
        else:
            win_rate = 0
            avg_trade = 0
            profit_factor = 0
        
        return {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": len(self.trades),
            "avg_trade": avg_trade,
            "equity_curve": equity_df['equity'],
            "trades": self.trades,
            "positions": self.positions
        }
    
    def start_live_trading(
        self,
        portfolio_config: Dict[str, Any],
        risk_manager: Any
    ) -> bool:
        """
        Start live trading
        
        Args:
            portfolio_config: Portfolio configuration
            risk_manager: Risk manager instance
            
        Returns:
            Success status
        """
        logger.info("Starting live trading")
        
        try:
            self.is_trading = True
            
            # Initialize live trading components
            self._initialize_live_trading(portfolio_config, risk_manager)
            
            # Start trading loop
            asyncio.create_task(self._trading_loop(portfolio_config, risk_manager))
            
            logger.info("Live trading started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start live trading: {e}")
            self.is_trading = False
            return False
    
    def _initialize_live_trading(self, portfolio_config: Dict[str, Any], risk_manager: Any) -> None:
        """Initialize live trading components"""
        # Initialize real-time data feeds
        # Initialize order execution
        # Initialize risk monitoring
        pass
    
    async def _trading_loop(self, portfolio_config: Dict[str, Any], risk_manager: Any) -> None:
        """Main trading loop"""
        while self.is_trading:
            try:
                # Check risk limits
                if not risk_manager.check_risk_limits(self):
                    logger.warning("Risk limits exceeded, stopping trading")
                    self.is_trading = False
                    break
                
                # Process market data
                # Generate signals
                # Execute trades
                # Update positions
                
                # Wait for next iteration
                await asyncio.sleep(1)  # 1 second delay
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    def stop_trading(self) -> None:
        """Stop live trading"""
        logger.info("Stopping live trading")
        self.is_trading = False
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary"""
        total_value = self.config.total_capital + self.total_pnl
        
        # Add unrealized PnL
        for position in self.positions.values():
            total_value += position.unrealized_pnl
        
        return {
            "total_value": total_value,
            "total_pnl": self.total_pnl,
            "current_drawdown": self.current_drawdown,
            "peak_equity": self.peak_equity,
            "total_trades": len(self.trades),
            "open_positions": len(self.positions),
            "strategies": list(self.strategy_positions.keys())
        }


# Example usage
if __name__ == "__main__":
    # Create portfolio manager
    portfolio = PortfolioManager()
    
    # Add strategies
    portfolio.add_strategy("strategy_1", allocation=0.25)
    portfolio.add_strategy("strategy_2", allocation=0.25)
    portfolio.add_strategy("strategy_3", allocation=0.25)
    portfolio.add_strategy("strategy_4", allocation=0.25)
    
    # Build portfolio
    strategies = {"strategy_1": None, "strategy_2": None, "strategy_3": None, "strategy_4": None}
    results = {"strategy_1": None, "strategy_2": None, "strategy_3": None, "strategy_4": None}
    
    portfolio_config = portfolio.build_portfolio(strategies, results)
    
    print("Portfolio built successfully")
    print(f"Strategies: {portfolio_config['strategies']}")
    print(f"Expected return: {portfolio_config['expected_return']:.2%}")
    print(f"Expected risk: {portfolio_config['expected_risk']:.2%}") 