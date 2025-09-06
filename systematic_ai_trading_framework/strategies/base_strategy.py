"""
Base Strategy Class for the Systematic AI Trading Framework.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

from utils.logger import setup_logger


@dataclass
class Signal:
    """Represents a trading signal."""
    timestamp: datetime
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float   # 0-1 confidence
    price: float
    quantity: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    side: str  # 'long', 'short'
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float
    current_time: datetime
    pnl: float
    pnl_pct: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    This class provides the foundation for implementing systematic trading strategies
    with standardized interfaces for signal generation, risk management, and execution.
    """
    
    def __init__(self, name: str, parameters: Dict[str, Any]):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters dictionary
        """
        self.name = name
        self.parameters = parameters
        self.logger = setup_logger(f"strategy.{name}", "INFO")
        
        # Strategy state
        self.is_active = False
        self.positions = {}
        self.signals = []
        self.equity_curve = []
        self.trade_history = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = 0.0
        
        # Risk management
        self.max_position_size = parameters.get('max_position_size', 0.1)
        self.stop_loss = parameters.get('stop_loss', 0.05)
        self.take_profit = parameters.get('take_profit', 0.15)
        self.max_drawdown_limit = parameters.get('max_drawdown_limit', 0.20)
        
        # Data storage
        self.data_buffer = {}
        self.lookback_period = parameters.get('lookback_period', 100)
        
        self.logger.info(f"Strategy {name} initialized with parameters: {parameters}")
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from market data.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Series with signals: 1 (buy), -1 (sell), 0 (hold)
        """
        pass
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate trading signals with additional processing.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Series with processed signals
        """
        # Generate raw signals
        raw_signals = self.generate_signals(data)
        
        # Apply filters and enhancements
        filtered_signals = self._apply_signal_filters(raw_signals, data)
        enhanced_signals = self._enhance_signals(filtered_signals, data)
        
        return enhanced_signals
    
    def _apply_signal_filters(self, signals: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Apply filters to raw signals."""
        filtered_signals = signals.copy()
        
        # Volatility filter
        if 'volatility_filter' in self.parameters:
            volatility = data['close'].pct_change().rolling(20).std()
            volatility_threshold = self.parameters.get('volatility_threshold', 0.02)
            filtered_signals[volatility < volatility_threshold] = 0
        
        # Volume filter
        if 'volume_filter' in self.parameters and 'volume' in data.columns:
            volume_ma = data['volume'].rolling(20).mean()
            volume_threshold = self.parameters.get('volume_threshold', 0.5)
            filtered_signals[data['volume'] < volume_ma * volume_threshold] = 0
        
        # Trend filter
        if 'trend_filter' in self.parameters:
            trend_period = self.parameters.get('trend_period', 50)
            trend_ma = data['close'].rolling(trend_period).mean()
            filtered_signals[data['close'] < trend_ma] = 0
        
        return filtered_signals
    
    def _enhance_signals(self, signals: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Enhance signals with additional logic."""
        enhanced_signals = signals.copy()
        
        # Signal strength based on indicators
        if 'rsi' in data.columns:
            rsi = data['rsi']
            # Stronger signals when RSI is more extreme
            enhanced_signals[(signals == 1) & (rsi < 30)] *= 1.2
            enhanced_signals[(signals == -1) & (rsi > 70)] *= 1.2
        
        # Momentum confirmation
        if 'momentum' in data.columns:
            momentum = data['momentum']
            enhanced_signals[(signals == 1) & (momentum > 0)] *= 1.1
            enhanced_signals[(signals == -1) & (momentum < 0)] *= 1.1
        
        return enhanced_signals
    
    def calculate_position_size(self, signal_strength: float, available_capital: float) -> float:
        """
        Calculate position size based on signal strength and risk management.
        
        Args:
            signal_strength: Signal strength (0-1)
            available_capital: Available capital for trading
            
        Returns:
            Position size in currency units
        """
        # Base position size
        base_size = available_capital * self.max_position_size
        
        # Adjust for signal strength
        adjusted_size = base_size * signal_strength
        
        # Apply Kelly criterion if enabled
        if self.parameters.get('use_kelly', False):
            win_rate = self.winning_trades / max(self.total_trades, 1)
            avg_win = self.parameters.get('avg_win', 0.02)
            avg_loss = self.parameters.get('avg_loss', 0.01)
            
            if avg_loss > 0:
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
                adjusted_size *= kelly_fraction
        
        return adjusted_size
    
    def should_exit_position(self, position: Position, current_data: pd.DataFrame) -> Tuple[bool, str]:
        """
        Determine if a position should be exited.
        
        Args:
            position: Current position
            current_data: Latest market data
            
        Returns:
            Tuple of (should_exit, reason)
        """
        current_price = current_data['close'].iloc[-1]
        
        # Stop loss check
        if position.side == 'long':
            stop_loss_price = position.entry_price * (1 - self.stop_loss)
            if current_price <= stop_loss_price:
                return True, "stop_loss"
            
            # Take profit check
            take_profit_price = position.entry_price * (1 + self.take_profit)
            if current_price >= take_profit_price:
                return True, "take_profit"
        
        elif position.side == 'short':
            stop_loss_price = position.entry_price * (1 + self.stop_loss)
            if current_price >= stop_loss_price:
                return True, "stop_loss"
            
            # Take profit check
            take_profit_price = position.entry_price * (1 - self.take_profit)
            if current_price <= take_profit_price:
                return True, "take_profit"
        
        # Time-based exit
        if 'max_hold_days' in self.parameters:
            hold_days = (datetime.now() - position.entry_time).days
            if hold_days >= self.parameters['max_hold_days']:
                return True, "time_exit"
        
        # Signal-based exit
        if 'exit_on_signal_reversal' in self.parameters:
            # Check if signal has reversed
            pass
        
        return False, ""
    
    def update_position(self, symbol: str, current_price: float, current_time: datetime):
        """Update position with current market data."""
        if symbol in self.positions:
            position = self.positions[symbol]
            position.current_price = current_price
            position.current_time = current_time
            
            # Calculate P&L
            if position.side == 'long':
                position.pnl = (current_price - position.entry_price) * position.quantity
                position.pnl_pct = (current_price - position.entry_price) / position.entry_price
            else:  # short
                position.pnl = (position.entry_price - current_price) * position.quantity
                position.pnl_pct = (position.entry_price - current_price) / position.entry_price
    
    def record_trade(self, trade_data: Dict[str, Any]):
        """Record a completed trade."""
        self.trade_history.append(trade_data)
        self.total_trades += 1
        
        # Update performance metrics
        pnl = trade_data.get('pnl', 0)
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
        
        # Update equity curve
        self.equity_curve.append({
            'timestamp': trade_data.get('exit_time', datetime.now()),
            'equity': self.total_pnl,
            'trade_pnl': pnl
        })
        
        # Update max drawdown
        current_equity = self.total_pnl
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        drawdown = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate and return performance metrics."""
        if self.total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_trade_pnl': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
        
        win_rate = self.winning_trades / self.total_trades
        avg_trade_pnl = self.total_pnl / self.total_trades
        
        # Calculate Sharpe ratio
        if len(self.equity_curve) > 1:
            returns = [trade['trade_pnl'] for trade in self.equity_curve]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0.0
        
        return {
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'avg_trade_pnl': avg_trade_pnl,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd - signal_line
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def start(self):
        """Start the strategy."""
        self.is_active = True
        self.logger.info(f"Strategy {self.name} started")
    
    def stop(self):
        """Stop the strategy."""
        self.is_active = False
        self.logger.info(f"Strategy {self.name} stopped")
    
    def reset(self):
        """Reset strategy state."""
        self.positions = {}
        self.signals = []
        self.equity_curve = []
        self.trade_history = []
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = 0.0
        self.logger.info(f"Strategy {self.name} reset")
    
    def configure(self, parameters: Dict[str, Any]):
        """Update strategy parameters."""
        self.parameters.update(parameters)
        self.logger.info(f"Strategy {self.name} reconfigured with parameters: {parameters}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current strategy status."""
        return {
            'name': self.name,
            'is_active': self.is_active,
            'parameters': self.parameters,
            'positions': len(self.positions),
            'total_trades': self.total_trades,
            'total_pnl': self.total_pnl,
            'performance': self.get_performance_metrics()
        } 