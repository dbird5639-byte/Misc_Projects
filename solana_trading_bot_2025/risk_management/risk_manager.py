"""
Risk Manager for Solana Trading Bot 2025

Manages trading risks, position sizing, and portfolio limits.
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import math

from bots.base_bot import TradeSignal
from config.settings import RiskManagementConfig

class RiskManager:
    """Manages trading risks and position sizing"""
    
    def __init__(self, config: RiskManagementConfig):
        self.config = config
        self.positions = {}  # Current open positions
        self.trade_history = []  # Historical trades
        self.portfolio_value = 0.0  # Current portfolio value
        self.max_risk_per_trade = 0.02  # 2% max risk per trade
        
        # Risk metrics
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.win_rate = 0.0
        
        # Position tracking
        self.total_positions = 0
        self.open_positions = 0
        self.max_positions_reached = False
    
    def should_trade(self, signal: TradeSignal) -> bool:
        """Determine if a trade should be executed based on risk management rules"""
        try:
            # Check portfolio risk limits
            if not self._check_portfolio_risk():
                return False
            
            # Check position limits
            if not self._check_position_limits(signal):
                return False
            
            # Check correlation limits
            if not self._check_correlation_limits(signal):
                return False
            
            # Check daily loss limits
            if not self._check_daily_loss_limits():
                return False
            
            # Check drawdown limits
            if not self._check_drawdown_limits():
                return False
            
            return True
            
        except Exception as e:
            print(f"Error in should_trade: {e}")
            return False
    
    def calculate_position_size(self, price: float, base_size: float) -> float:
        """Calculate position size based on risk management rules"""
        try:
            # Kelly Criterion for position sizing
            kelly_size = self._calculate_kelly_position_size(price, base_size)
            
            # Volatility-adjusted position sizing
            volatility_size = self._calculate_volatility_position_size(price, base_size)
            
            # Use the smaller of the two for conservative approach
            position_size = min(kelly_size, volatility_size)
            
            # Apply portfolio limits
            max_portfolio_size = self.portfolio_value * self.config.max_portfolio_risk
            position_size = min(position_size, max_portfolio_size)
            
            # Apply minimum position size
            min_position = self.portfolio_value * 0.001  # 0.1% minimum
            position_size = max(position_size, min_position)
            
            return position_size
            
        except Exception as e:
            print(f"Error calculating position size: {e}")
            return 0.0
    
    def _check_portfolio_risk(self) -> bool:
        """Check if portfolio risk is within limits"""
        try:
            # Calculate current portfolio risk
            total_risk = sum(pos.get("risk", 0) for pos in self.positions.values())
            
            # Check against maximum portfolio risk
            max_risk = self.portfolio_value * self.config.max_portfolio_risk
            
            return total_risk <= max_risk
            
        except Exception as e:
            print(f"Error checking portfolio risk: {e}")
            return False
    
    def _check_position_limits(self, signal: TradeSignal) -> bool:
        """Check position limits"""
        try:
            # Check maximum number of positions
            if self.open_positions >= self.config.max_positions:
                return False
            
            # Check if we already have a position in this token
            if signal.token_address in self.positions:
                return False
            
            # Check position size limits
            position_value = signal.price * signal.quantity
            max_position_value = self.portfolio_value * 0.1  # Max 10% per position
            
            return position_value <= max_position_value
            
        except Exception as e:
            print(f"Error checking position limits: {e}")
            return False
    
    def _check_correlation_limits(self, signal: TradeSignal) -> bool:
        """Check correlation limits to avoid over-concentration"""
        try:
            # This is a simplified correlation check
            # In practice, you would calculate actual correlations between tokens
            
            # Check if we have too many positions in similar tokens
            # For now, just check the number of open positions
            return self.open_positions < self.config.max_positions
            
        except Exception as e:
            print(f"Error checking correlation limits: {e}")
            return False
    
    def _check_daily_loss_limits(self) -> bool:
        """Check daily loss limits"""
        try:
            # Calculate daily P&L
            daily_loss_limit = self.portfolio_value * 0.05  # 5% daily loss limit
            
            return self.daily_pnl >= -daily_loss_limit
            
        except Exception as e:
            print(f"Error checking daily loss limits: {e}")
            return False
    
    def _check_drawdown_limits(self) -> bool:
        """Check drawdown limits"""
        try:
            # Maximum drawdown limit
            max_drawdown_limit = 0.2  # 20% maximum drawdown
            
            return self.max_drawdown <= max_drawdown_limit
            
        except Exception as e:
            print(f"Error checking drawdown limits: {e}")
            return False
    
    def _calculate_kelly_position_size(self, price: float, base_size: float) -> float:
        """Calculate position size using Kelly Criterion"""
        try:
            # Kelly Criterion: f = (bp - q) / b
            # where f = fraction of bankroll to bet
            # b = odds received on bet
            # p = probability of winning
            # q = probability of losing (1 - p)
            
            # For trading, we'll use a simplified version
            win_rate = self.win_rate if self.win_rate > 0 else 0.5
            avg_win = 0.1  # 10% average win
            avg_loss = 0.05  # 5% average loss
            
            # Calculate Kelly fraction
            b = avg_win / avg_loss  # odds ratio
            p = win_rate
            q = 1 - p
            
            kelly_fraction = (b * p - q) / b
            
            # Apply Kelly fraction to base size
            kelly_size = base_size * kelly_fraction
            
            # Cap at reasonable levels
            max_kelly = self.portfolio_value * 0.25  # Max 25% Kelly
            kelly_size = min(kelly_size, max_kelly)
            
            return max(kelly_size, 0.0)
            
        except Exception as e:
            print(f"Error calculating Kelly position size: {e}")
            return base_size * 0.1  # Conservative fallback
    
    def _calculate_volatility_position_size(self, price: float, base_size: float) -> float:
        """Calculate position size based on volatility"""
        try:
            # Simplified volatility calculation
            # In practice, you would use actual volatility data
            
            # Assume higher volatility = smaller position
            volatility_factor = 0.5  # Placeholder - would be calculated from actual data
            
            volatility_size = base_size * volatility_factor
            
            return volatility_size
            
        except Exception as e:
            print(f"Error calculating volatility position size: {e}")
            return base_size * 0.5  # Conservative fallback
    
    def add_position(self, signal: TradeSignal, position_size: float):
        """Add a new position to tracking"""
        try:
            position = {
                "token_address": signal.token_address,
                "entry_price": signal.price,
                "quantity": position_size / signal.price,
                "entry_time": signal.timestamp,
                "risk": position_size * 0.05,  # Assume 5% risk per position
                "stop_loss": signal.price * (1 - self.config.stop_loss_percentage),
                "take_profit": signal.price * (1 + self.config.take_profit_percentage),
                "status": "open"
            }
            
            self.positions[signal.token_address] = position
            self.open_positions += 1
            self.total_positions += 1
            
        except Exception as e:
            print(f"Error adding position: {e}")
    
    def close_position(self, token_address: str, exit_price: float, exit_time: datetime):
        """Close a position and record P&L"""
        try:
            if token_address not in self.positions:
                return
            
            position = self.positions[token_address]
            
            # Calculate P&L
            entry_price = position["entry_price"]
            quantity = position["quantity"]
            
            if position["status"] == "open":
                pnl = (exit_price - entry_price) * quantity
                
                # Record trade
                trade_record = {
                    "token_address": token_address,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "quantity": quantity,
                    "pnl": pnl,
                    "entry_time": position["entry_time"],
                    "exit_time": exit_time,
                    "duration": (exit_time - position["entry_time"]).total_seconds() / 3600  # hours
                }
                
                self.trade_history.append(trade_record)
                
                # Update metrics
                self.daily_pnl += pnl
                self._update_risk_metrics()
                
                # Remove position
                del self.positions[token_address]
                self.open_positions -= 1
                
        except Exception as e:
            print(f"Error closing position: {e}")
    
    def _update_risk_metrics(self):
        """Update risk metrics based on trade history"""
        try:
            if not self.trade_history:
                return
            
            # Calculate win rate
            winning_trades = [t for t in self.trade_history if t["pnl"] > 0]
            self.win_rate = len(winning_trades) / len(self.trade_history)
            
            # Calculate Sharpe ratio (simplified)
            if len(self.trade_history) > 1:
                returns = [t["pnl"] for t in self.trade_history]
                avg_return = sum(returns) / len(returns)
                std_return = math.sqrt(sum((r - avg_return) ** 2 for r in returns) / len(returns))
                
                if std_return > 0:
                    self.sharpe_ratio = avg_return / std_return
            
            # Calculate maximum drawdown
            cumulative_pnl = 0
            peak_pnl = 0
            max_dd = 0
            
            for trade in self.trade_history:
                cumulative_pnl += trade["pnl"]
                peak_pnl = max(peak_pnl, cumulative_pnl)
                drawdown = peak_pnl - cumulative_pnl
                max_dd = max(max_dd, drawdown)
            
            self.max_drawdown = max_dd / self.portfolio_value if self.portfolio_value > 0 else 0
            
        except Exception as e:
            print(f"Error updating risk metrics: {e}")
    
    def get_stop_loss_price(self, token_address: str) -> Optional[float]:
        """Get stop loss price for a position"""
        try:
            if token_address in self.positions:
                return self.positions[token_address]["stop_loss"]
            return None
            
        except Exception as e:
            print(f"Error getting stop loss price: {e}")
            return None
    
    def get_take_profit_price(self, token_address: str) -> Optional[float]:
        """Get take profit price for a position"""
        try:
            if token_address in self.positions:
                return self.positions[token_address]["take_profit"]
            return None
            
        except Exception as e:
            print(f"Error getting take profit price: {e}")
            return None
    
    def should_close_position(self, token_address: str, current_price: float) -> bool:
        """Determine if a position should be closed"""
        try:
            if token_address not in self.positions:
                return False
            
            position = self.positions[token_address]
            
            # Check stop loss
            if current_price <= position["stop_loss"]:
                return True
            
            # Check take profit
            if current_price >= position["take_profit"]:
                return True
            
            # Additional exit conditions can be added here
            # - Time-based exits
            # - Technical indicator exits
            # - News-based exits
            
            return False
            
        except Exception as e:
            print(f"Error checking position close: {e}")
            return False
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk management summary"""
        try:
            return {
                "portfolio_value": self.portfolio_value,
                "open_positions": self.open_positions,
                "total_positions": self.total_positions,
                "daily_pnl": self.daily_pnl,
                "max_drawdown": self.max_drawdown,
                "sharpe_ratio": self.sharpe_ratio,
                "win_rate": self.win_rate,
                "positions": self.positions,
                "risk_limits": {
                    "max_portfolio_risk": self.config.max_portfolio_risk,
                    "max_positions": self.config.max_positions,
                    "stop_loss_percentage": self.config.stop_loss_percentage,
                    "take_profit_percentage": self.config.take_profit_percentage
                }
            }
            
        except Exception as e:
            print(f"Error getting risk summary: {e}")
            return {}
    
    def reset_daily_metrics(self):
        """Reset daily metrics (call at start of new day)"""
        self.daily_pnl = 0.0
    
    def update_portfolio_value(self, new_value: float):
        """Update portfolio value"""
        self.portfolio_value = new_value 