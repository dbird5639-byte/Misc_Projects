"""
Backtesting Engine for AI Market Maker & Liquidation Monitor
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

from ..config.settings import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Trade:
    """Represents a trade in backtesting"""
    symbol: str
    side: str  # 'buy' or 'sell'
    entry_price: float
    exit_price: Optional[float]
    size: float
    entry_time: datetime
    exit_time: Optional[datetime]
    pnl: Optional[float]
    commission: float
    slippage: float
    strategy: str
    signal_confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class BacktestResult:
    """Represents backtest results"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    max_win: float
    max_loss: float
    avg_trade_duration: float
    total_commission: float
    total_slippage: float
    start_date: datetime
    end_date: datetime
    trades: List[Trade]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class BacktestConfig:
    """Represents backtest configuration"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    commission_rate: float
    slippage_rate: float
    symbols: List[str]
    strategies: List[str]
    risk_management: Dict[str, Any]
    data_source: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class BacktestEngine:
    """
    Comprehensive backtesting engine for AI Market Maker strategies
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.is_running = False
        
        # Data storage
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.portfolio = {
            "cash": config.initial_capital,
            "positions": {},
            "trades": [],
            "equity_curve": []
        }
        
        # Performance tracking
        self.current_equity = config.initial_capital
        self.peak_equity = config.initial_capital
        self.max_drawdown = 0.0
        
        # Results
        self.results: Optional[BacktestResult] = None
        
    async def initialize(self):
        """Initialize the backtest engine"""
        try:
            logger.info("Initializing Backtest Engine...")
            
            # Load market data
            await self._load_market_data()
            
            # Validate data
            if not self._validate_data():
                raise ValueError("Invalid or insufficient market data")
            
            logger.info("Backtest Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Backtest Engine: {e}")
            return False
    
    async def _load_market_data(self):
        """Load market data for backtesting"""
        try:
            logger.info("Loading market data...")
            
            for symbol in self.config.symbols:
                # Load data from file or API
                data = await self._load_symbol_data(symbol)
                if data is not None:
                    self.market_data[symbol] = data
                    logger.info(f"Loaded {len(data)} records for {symbol}")
            
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
    
    async def _load_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load data for a specific symbol"""
        try:
            # This would load from actual data source
            # For now, generate mock data
            start_date = self.config.start_date
            end_date = self.config.end_date
            
            # Generate daily data
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Generate price data with some trend and volatility
            np.random.seed(42)  # For reproducible results
            returns = np.random.normal(0.001, 0.02, len(date_range))
            prices = 100 * np.exp(np.cumsum(returns))
            
            # Add some trend
            trend = np.linspace(0, 0.1, len(date_range))
            prices = prices * (1 + trend)
            
            # Create DataFrame
            data = pd.DataFrame({
                'date': date_range,
                'open': prices * (1 + np.random.normal(0, 0.005, len(date_range))),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(date_range)))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(date_range)))),
                'close': prices,
                'volume': np.random.lognormal(10, 0.5, len(date_range))
            })
            
            data.set_index('date', inplace=True)
            return data
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return None
    
    def _validate_data(self) -> bool:
        """Validate market data"""
        try:
            if not self.market_data:
                return False
            
            for symbol, data in self.market_data.items():
                if len(data) < 100:  # Need minimum data points
                    logger.warning(f"Insufficient data for {symbol}: {len(data)} records")
                    return False
                
                # Check for required columns
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in data.columns for col in required_columns):
                    logger.error(f"Missing required columns for {symbol}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return False
    
    async def run_backtest(self) -> BacktestResult:
        """Run the backtest"""
        try:
            logger.info("Starting backtest...")
            self.is_running = True
            
            # Initialize portfolio
            self._initialize_portfolio()
            
            # Get date range
            start_date = self.config.start_date
            end_date = self.config.end_date
            
            # Run simulation
            current_date = start_date
            while current_date <= end_date and self.is_running:
                # Process market data for current date
                await self._process_date(current_date)
                
                # Update equity curve
                self._update_equity_curve(current_date)
                
                # Move to next date
                current_date += timedelta(days=1)
            
            # Close all positions at end
            await self._close_all_positions(end_date)
            
            # Calculate results
            self.results = self._calculate_results()
            
            logger.info("Backtest completed successfully")
            return self.results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return None
    
    def _initialize_portfolio(self):
        """Initialize portfolio"""
        self.portfolio = {
            "cash": self.config.initial_capital,
            "positions": {},
            "trades": [],
            "equity_curve": []
        }
        self.current_equity = self.config.initial_capital
        self.peak_equity = self.config.initial_capital
        self.max_drawdown = 0.0
    
    async def _process_date(self, date: datetime):
        """Process market data for a specific date"""
        try:
            for symbol in self.config.symbols:
                if symbol in self.market_data:
                    data = self.market_data[symbol]
                    if date in data.index:
                        # Get market data for this date
                        market_data = data.loc[date]
                        
                        # Generate signals
                        signals = await self._generate_signals(symbol, market_data, date)
                        
                        # Execute trades based on signals
                        for signal in signals:
                            await self._execute_trade(signal, date)
                        
                        # Update existing positions
                        await self._update_positions(symbol, market_data, date)
            
        except Exception as e:
            logger.error(f"Error processing date {date}: {e}")
    
    async def _generate_signals(self, symbol: str, market_data: pd.Series, date: datetime) -> List[Dict[str, Any]]:
        """Generate trading signals"""
        signals = []
        
        try:
            # Get historical data for analysis
            historical_data = self._get_historical_data(symbol, date, days=30)
            
            if len(historical_data) < 20:  # Need minimum data for analysis
                return signals
            
            # Simple moving average strategy
            sma_short = historical_data['close'].rolling(window=10).mean().iloc[-1]
            sma_long = historical_data['close'].rolling(window=20).mean().iloc[-1]
            current_price = market_data['close']
            
            # Generate signals
            if sma_short > sma_long and current_price > sma_short:
                # Buy signal
                signals.append({
                    'symbol': symbol,
                    'side': 'buy',
                    'price': current_price,
                    'size': self._calculate_position_size(symbol, current_price),
                    'confidence': 0.7,
                    'strategy': 'sma_crossover'
                })
            elif sma_short < sma_long and current_price < sma_short:
                # Sell signal
                signals.append({
                    'symbol': symbol,
                    'side': 'sell',
                    'price': current_price,
                    'size': self._calculate_position_size(symbol, current_price),
                    'confidence': 0.7,
                    'strategy': 'sma_crossover'
                })
            
            # RSI strategy
            rsi = self._calculate_rsi(historical_data['close'])
            if rsi < 30:
                signals.append({
                    'symbol': symbol,
                    'side': 'buy',
                    'price': current_price,
                    'size': self._calculate_position_size(symbol, current_price),
                    'confidence': 0.6,
                    'strategy': 'rsi_oversold'
                })
            elif rsi > 70:
                signals.append({
                    'symbol': symbol,
                    'side': 'sell',
                    'price': current_price,
                    'size': self._calculate_position_size(symbol, current_price),
                    'confidence': 0.6,
                    'strategy': 'rsi_overbought'
                })
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
        
        return signals
    
    def _get_historical_data(self, symbol: str, date: datetime, days: int) -> pd.DataFrame:
        """Get historical data for analysis"""
        try:
            data = self.market_data[symbol]
            end_date = date
            start_date = end_date - timedelta(days=days)
            
            return data[(data.index >= start_date) & (data.index <= end_date)]
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0
    
    def _calculate_position_size(self, symbol: str, price: float) -> float:
        """Calculate position size based on risk management"""
        try:
            # Simple position sizing: 2% of portfolio per trade
            risk_per_trade = self.current_equity * 0.02
            position_size = risk_per_trade / price
            
            # Check if we have enough cash
            required_cash = position_size * price
            if required_cash > self.portfolio['cash']:
                position_size = self.portfolio['cash'] / price
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    async def _execute_trade(self, signal: Dict[str, Any], date: datetime):
        """Execute a trade based on signal"""
        try:
            symbol = signal['symbol']
            side = signal['side']
            price = signal['price']
            size = signal['size']
            
            if side == 'buy':
                # Check if we have enough cash
                required_cash = size * price
                if required_cash > self.portfolio['cash']:
                    return
                
                # Calculate costs
                commission = required_cash * self.config.commission_rate
                slippage = required_cash * self.config.slippage_rate
                total_cost = required_cash + commission + slippage
                
                # Execute trade
                self.portfolio['cash'] -= total_cost
                
                if symbol not in self.portfolio['positions']:
                    self.portfolio['positions'][symbol] = {
                        'size': size,
                        'entry_price': price,
                        'entry_time': date
                    }
                else:
                    # Average down/up
                    current_pos = self.portfolio['positions'][symbol]
                    total_size = current_pos['size'] + size
                    avg_price = ((current_pos['size'] * current_pos['entry_price']) + (size * price)) / total_size
                    current_pos['size'] = total_size
                    current_pos['entry_price'] = avg_price
                
                # Record trade
                trade = Trade(
                    symbol=symbol,
                    side=side,
                    entry_price=price,
                    exit_price=None,
                    size=size,
                    entry_time=date,
                    exit_time=None,
                    pnl=None,
                    commission=commission,
                    slippage=slippage,
                    strategy=signal['strategy'],
                    signal_confidence=signal['confidence']
                )
                self.portfolio['trades'].append(trade)
                
            elif side == 'sell':
                # Check if we have position to sell
                if symbol not in self.portfolio['positions'] or self.portfolio['positions'][symbol]['size'] <= 0:
                    return
                
                position = self.portfolio['positions'][symbol]
                sell_size = min(size, position['size'])
                
                # Calculate proceeds
                proceeds = sell_size * price
                commission = proceeds * self.config.commission_rate
                slippage = proceeds * self.config.slippage_rate
                net_proceeds = proceeds - commission - slippage
                
                # Execute trade
                self.portfolio['cash'] += net_proceeds
                position['size'] -= sell_size
                
                # Calculate PnL
                pnl = (price - position['entry_price']) * sell_size - commission - slippage
                
                # Record trade
                trade = Trade(
                    symbol=symbol,
                    side=side,
                    entry_price=position['entry_price'],
                    exit_price=price,
                    size=sell_size,
                    entry_time=position['entry_time'],
                    exit_time=date,
                    pnl=pnl,
                    commission=commission,
                    slippage=slippage,
                    strategy=signal['strategy'],
                    signal_confidence=signal['confidence']
                )
                self.portfolio['trades'].append(trade)
                
                # Remove position if fully sold
                if position['size'] <= 0:
                    del self.portfolio['positions'][symbol]
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    async def _update_positions(self, symbol: str, market_data: pd.Series, date: datetime):
        """Update existing positions"""
        try:
            if symbol in self.portfolio['positions']:
                position = self.portfolio['positions'][symbol]
                current_price = market_data['close']
                
                # Check stop loss (5% loss)
                stop_loss_price = position['entry_price'] * 0.95
                if current_price <= stop_loss_price:
                    # Close position at stop loss
                    await self._execute_trade({
                        'symbol': symbol,
                        'side': 'sell',
                        'price': stop_loss_price,
                        'size': position['size'],
                        'confidence': 1.0,
                        'strategy': 'stop_loss'
                    }, date)
                
                # Check take profit (10% gain)
                take_profit_price = position['entry_price'] * 1.10
                if current_price >= take_profit_price:
                    # Close position at take profit
                    await self._execute_trade({
                        'symbol': symbol,
                        'side': 'sell',
                        'price': take_profit_price,
                        'size': position['size'],
                        'confidence': 1.0,
                        'strategy': 'take_profit'
                    }, date)
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def _update_equity_curve(self, date: datetime):
        """Update equity curve"""
        try:
            # Calculate current portfolio value
            portfolio_value = self.portfolio['cash']
            
            for symbol, position in self.portfolio['positions'].items():
                if symbol in self.market_data and date in self.market_data[symbol].index:
                    current_price = self.market_data[symbol].loc[date]['close']
                    position_value = position['size'] * current_price
                    portfolio_value += position_value
            
            self.current_equity = portfolio_value
            
            # Update peak equity
            if self.current_equity > self.peak_equity:
                self.peak_equity = self.current_equity
            
            # Calculate drawdown
            drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
            
            # Record equity curve
            self.portfolio['equity_curve'].append({
                'date': date,
                'equity': self.current_equity,
                'drawdown': drawdown
            })
            
        except Exception as e:
            logger.error(f"Error updating equity curve: {e}")
    
    async def _close_all_positions(self, date: datetime):
        """Close all positions at end of backtest"""
        try:
            for symbol in list(self.portfolio['positions'].keys()):
                position = self.portfolio['positions'][symbol]
                
                if symbol in self.market_data and date in self.market_data[symbol].index:
                    current_price = self.market_data[symbol].loc[date]['close']
                else:
                    # Use last available price
                    data = self.market_data[symbol]
                    current_price = data['close'].iloc[-1]
                
                # Close position
                await self._execute_trade({
                    'symbol': symbol,
                    'side': 'sell',
                    'price': current_price,
                    'size': position['size'],
                    'confidence': 1.0,
                    'strategy': 'end_of_backtest'
                }, date)
            
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest results"""
        try:
            trades = self.portfolio['trades']
            
            if not trades:
                return self._create_empty_results()
            
            # Basic statistics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t.pnl and t.pnl > 0])
            losing_trades = len([t for t in trades if t.pnl and t.pnl < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # PnL statistics
            total_pnl = sum(t.pnl for t in trades if t.pnl)
            total_return = (self.current_equity - self.config.initial_capital) / self.config.initial_capital
            
            # Calculate returns for Sharpe ratio
            equity_curve = pd.DataFrame(self.portfolio['equity_curve'])
            if len(equity_curve) > 1:
                equity_curve['returns'] = equity_curve['equity'].pct_change()
                returns = equity_curve['returns'].dropna()
                
                if len(returns) > 0:
                    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                    
                    # Sortino ratio
                    downside_returns = returns[returns < 0]
                    sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
                else:
                    sharpe_ratio = 0
                    sortino_ratio = 0
            else:
                sharpe_ratio = 0
                sortino_ratio = 0
            
            # Calmar ratio
            calmar_ratio = total_return / self.max_drawdown if self.max_drawdown > 0 else 0
            
            # Profit factor
            gross_profit = sum(t.pnl for t in trades if t.pnl and t.pnl > 0)
            gross_loss = abs(sum(t.pnl for t in trades if t.pnl and t.pnl < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Average win/loss
            wins = [t.pnl for t in trades if t.pnl and t.pnl > 0]
            losses = [t.pnl for t in trades if t.pnl and t.pnl < 0]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            max_win = max(wins) if wins else 0
            max_loss = min(losses) if losses else 0
            
            # Average trade duration
            durations = []
            for trade in trades:
                if trade.exit_time:
                    duration = (trade.exit_time - trade.entry_time).total_seconds() / 86400  # days
                    durations.append(duration)
            
            avg_trade_duration = np.mean(durations) if durations else 0
            
            # Total costs
            total_commission = sum(t.commission for t in trades)
            total_slippage = sum(t.slippage for t in trades)
            
            return BacktestResult(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                total_return=total_return,
                max_drawdown=self.max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                profit_factor=profit_factor,
                avg_win=avg_win,
                avg_loss=avg_loss,
                max_win=max_win,
                max_loss=max_loss,
                avg_trade_duration=avg_trade_duration,
                total_commission=total_commission,
                total_slippage=total_slippage,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                trades=trades
            )
            
        except Exception as e:
            logger.error(f"Error calculating results: {e}")
            return self._create_empty_results()
    
    def _create_empty_results(self) -> BacktestResult:
        """Create empty results"""
        return BacktestResult(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            total_return=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            max_win=0.0,
            max_loss=0.0,
            avg_trade_duration=0.0,
            total_commission=0.0,
            total_slippage=0.0,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            trades=[]
        )
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive backtest report"""
        try:
            if not self.results:
                return {"error": "No backtest results available"}
            
            # Create equity curve chart
            equity_chart = self._create_equity_chart()
            
            # Create drawdown chart
            drawdown_chart = self._create_drawdown_chart()
            
            # Create trade distribution chart
            trade_dist_chart = self._create_trade_distribution_chart()
            
            # Create monthly returns chart
            monthly_returns_chart = self._create_monthly_returns_chart()
            
            return {
                "summary": {
                    "total_return": f"{self.results.total_return:.2%}",
                    "sharpe_ratio": f"{self.results.sharpe_ratio:.2f}",
                    "max_drawdown": f"{self.results.max_drawdown:.2%}",
                    "win_rate": f"{self.results.win_rate:.2%}",
                    "profit_factor": f"{self.results.profit_factor:.2f}",
                    "total_trades": self.results.total_trades
                },
                "charts": {
                    "equity_curve": equity_chart,
                    "drawdown": drawdown_chart,
                    "trade_distribution": trade_dist_chart,
                    "monthly_returns": monthly_returns_chart
                },
                "detailed_results": self.results.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {"error": str(e)}
    
    def _create_equity_chart(self) -> Dict[str, Any]:
        """Create equity curve chart"""
        try:
            equity_curve = pd.DataFrame(self.portfolio['equity_curve'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_curve['date'],
                y=equity_curve['equity'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#00ff88', width=2)
            ))
            
            fig.update_layout(
                title='Portfolio Equity Curve',
                xaxis_title='Date',
                yaxis_title='Portfolio Value ($)',
                template='plotly_dark',
                height=400
            )
            
            return json.loads(fig.to_json())
            
        except Exception as e:
            logger.error(f"Error creating equity chart: {e}")
            return {}
    
    def _create_drawdown_chart(self) -> Dict[str, Any]:
        """Create drawdown chart"""
        try:
            equity_curve = pd.DataFrame(self.portfolio['equity_curve'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_curve['date'],
                y=equity_curve['drawdown'] * 100,
                mode='lines',
                name='Drawdown',
                line=dict(color='#ff4444', width=2),
                fill='tonexty'
            ))
            
            fig.update_layout(
                title='Portfolio Drawdown',
                xaxis_title='Date',
                yaxis_title='Drawdown (%)',
                template='plotly_dark',
                height=300
            )
            
            return json.loads(fig.to_json())
            
        except Exception as e:
            logger.error(f"Error creating drawdown chart: {e}")
            return {}
    
    def _create_trade_distribution_chart(self) -> Dict[str, Any]:
        """Create trade distribution chart"""
        try:
            pnls = [t.pnl for t in self.results.trades if t.pnl]
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=pnls,
                nbinsx=20,
                name='Trade PnL Distribution',
                marker_color='#ff8800'
            ))
            
            fig.update_layout(
                title='Trade PnL Distribution',
                xaxis_title='PnL ($)',
                yaxis_title='Frequency',
                template='plotly_dark',
                height=300
            )
            
            return json.loads(fig.to_json())
            
        except Exception as e:
            logger.error(f"Error creating trade distribution chart: {e}")
            return {}
    
    def _create_monthly_returns_chart(self) -> Dict[str, Any]:
        """Create monthly returns chart"""
        try:
            equity_curve = pd.DataFrame(self.portfolio['equity_curve'])
            equity_curve['returns'] = equity_curve['equity'].pct_change()
            equity_curve['month'] = equity_curve['date'].dt.to_period('M')
            
            monthly_returns = equity_curve.groupby('month')['returns'].sum()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=monthly_returns.index.astype(str),
                y=monthly_returns.values * 100,
                name='Monthly Returns',
                marker_color=['#00ff88' if x > 0 else '#ff4444' for x in monthly_returns.values]
            ))
            
            fig.update_layout(
                title='Monthly Returns',
                xaxis_title='Month',
                yaxis_title='Return (%)',
                template='plotly_dark',
                height=300
            )
            
            return json.loads(fig.to_json())
            
        except Exception as e:
            logger.error(f"Error creating monthly returns chart: {e}")
            return {}
    
    def save_results(self, filename: str):
        """Save backtest results to file"""
        try:
            if self.results:
                report = self.generate_report()
                
                with open(filename, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                
                logger.info(f"Backtest results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")


async def run_backtest(config: BacktestConfig) -> Optional[BacktestResult]:
    """Run a backtest with given configuration"""
    try:
        engine = BacktestEngine(config)
        
        if await engine.initialize():
            return await engine.run_backtest()
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return None


def create_backtest_config(
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 100000,
    symbols: List[str] = None,
    strategies: List[str] = None
) -> BacktestConfig:
    """Create a backtest configuration"""
    if symbols is None:
        symbols = ["BTC", "ETH", "SOL"]
    
    if strategies is None:
        strategies = ["sma_crossover", "rsi", "momentum"]
    
    return BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        commission_rate=0.001,  # 0.1%
        slippage_rate=0.0005,   # 0.05%
        symbols=symbols,
        strategies=strategies,
        risk_management={
            "max_position_size": 0.02,  # 2% per trade
            "stop_loss": 0.05,          # 5% stop loss
            "take_profit": 0.10,        # 10% take profit
            "max_drawdown": 0.20        # 20% max drawdown
        },
        data_source="mock"
    ) 