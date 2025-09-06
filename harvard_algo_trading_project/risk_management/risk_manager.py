"""
Risk Management System

Comprehensive risk management for algorithmic trading,
including position sizing, stop losses, and portfolio risk controls.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import math

class RiskLevel(Enum):
    """Risk tolerance levels"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

@dataclass
class Position:
    """Position data structure"""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    entry_time: datetime
    side: str  # "long" or "short"
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

@dataclass
class RiskMetrics:
    """Risk metrics for portfolio"""
    total_exposure: float
    portfolio_value: float
    leverage_ratio: float
    var_95: float  # Value at Risk (95% confidence)
    max_drawdown: float
    sharpe_ratio: float
    beta: float
    correlation: float

class PositionSizer:
    """Position sizing algorithms"""
    
    def __init__(self, risk_level: RiskLevel = RiskLevel.MODERATE):
        self.risk_level = risk_level
        self.risk_per_trade = self._get_risk_per_trade()
    
    def _get_risk_per_trade(self) -> float:
        """Get risk per trade based on risk level"""
        risk_levels = {
            RiskLevel.CONSERVATIVE: 0.01,  # 1% per trade
            RiskLevel.MODERATE: 0.02,      # 2% per trade
            RiskLevel.AGGRESSIVE: 0.05     # 5% per trade
        }
        return risk_levels.get(self.risk_level, 0.02)
    
    def kelly_criterion(self, win_rate: float, avg_win: float, 
                       avg_loss: float) -> float:
        """Calculate position size using Kelly Criterion"""
        if avg_loss == 0:
            return 0.0
        
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # Apply fractional Kelly (more conservative)
        return max(0.0, min(kelly_fraction * 0.25, self.risk_per_trade))
    
    def volatility_based(self, price: float, volatility: float, 
                        account_value: float) -> int:
        """Calculate position size based on volatility"""
        # Higher volatility = smaller position
        volatility_factor = 1.0 / (1.0 + volatility)
        risk_amount = account_value * self.risk_per_trade * volatility_factor
        
        return int(risk_amount / price)
    
    def equal_weight(self, account_value: float, num_positions: int, 
                    price: float) -> int:
        """Equal weight position sizing"""
        position_value = account_value / num_positions
        return int(position_value / price)
    
    def risk_parity(self, volatilities: Dict[str, float], 
                   account_value: float, prices: Dict[str, float]) -> Dict[str, int]:
        """Risk parity position sizing"""
        # Equal risk contribution from each position
        total_risk = sum(1.0 / vol for vol in volatilities.values())
        risk_per_position = account_value * self.risk_per_trade / total_risk
        
        positions = {}
        for symbol, volatility in volatilities.items():
            if symbol in prices and volatility > 0:
                position_value = risk_per_position * volatility
                positions[symbol] = int(position_value / prices[symbol])
        
        return positions

class StopLossManager:
    """Stop loss and take profit management"""
    
    def __init__(self, stop_loss_pct: float = 0.05, 
                 take_profit_pct: float = 0.10,
                 trailing_stop: bool = True):
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop = trailing_stop
        self.trailing_stops = {}  # Track trailing stops per position
    
    def should_stop_loss(self, position: Position) -> bool:
        """Check if stop loss should be triggered"""
        if position.side == "long":
            loss_pct = (position.entry_price - position.current_price) / position.entry_price
            return loss_pct >= self.stop_loss_pct
        else:  # short
            loss_pct = (position.current_price - position.entry_price) / position.entry_price
            return loss_pct >= self.stop_loss_pct
    
    def should_take_profit(self, position: Position) -> bool:
        """Check if take profit should be triggered"""
        if position.side == "long":
            profit_pct = (position.current_price - position.entry_price) / position.entry_price
            return profit_pct >= self.take_profit_pct
        else:  # short
            profit_pct = (position.entry_price - position.current_price) / position.entry_price
            return profit_pct >= self.take_profit_pct
    
    def update_trailing_stop(self, position: Position):
        """Update trailing stop for a position"""
        if not self.trailing_stop:
            return
        
        position_key = f"{position.symbol}_{position.side}"
        
        if position.side == "long":
            # For long positions, trail below the high
            if position_key not in self.trailing_stops:
                self.trailing_stops[position_key] = position.entry_price * (1 - self.stop_loss_pct)
            else:
                # Update trailing stop if price moves higher
                new_stop = position.current_price * (1 - self.stop_loss_pct)
                if new_stop > self.trailing_stops[position_key]:
                    self.trailing_stops[position_key] = new_stop
        
        else:  # short
            # For short positions, trail above the low
            if position_key not in self.trailing_stops:
                self.trailing_stops[position_key] = position.entry_price * (1 + self.stop_loss_pct)
            else:
                # Update trailing stop if price moves lower
                new_stop = position.current_price * (1 + self.stop_loss_pct)
                if new_stop < self.trailing_stops[position_key]:
                    self.trailing_stops[position_key] = new_stop
    
    def check_trailing_stop(self, position: Position) -> bool:
        """Check if trailing stop is triggered"""
        if not self.trailing_stop:
            return False
        
        position_key = f"{position.symbol}_{position.side}"
        
        if position_key not in self.trailing_stops:
            return False
        
        if position.side == "long":
            return position.current_price <= self.trailing_stops[position_key]
        else:  # short
            return position.current_price >= self.trailing_stops[position_key]

class PortfolioRiskManager:
    """Portfolio-level risk management"""
    
    def __init__(self, max_portfolio_risk: float = 0.06,
                 max_correlation: float = 0.7,
                 max_sector_exposure: float = 0.3):
        self.max_portfolio_risk = max_portfolio_risk
        self.max_correlation = max_correlation
        self.max_sector_exposure = max_sector_exposure
        self.position_history = []
    
    def calculate_portfolio_metrics(self, positions: List[Position], 
                                  account_value: float) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        if not positions:
            return RiskMetrics(
                total_exposure=0.0,
                portfolio_value=account_value,
                leverage_ratio=0.0,
                var_95=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                beta=1.0,
                correlation=0.0
            )
        
        # Calculate total exposure
        total_exposure = sum(abs(pos.quantity * pos.current_price) for pos in positions)
        leverage_ratio = total_exposure / account_value
        
        # Calculate Value at Risk (simplified)
        position_returns = [pos.unrealized_pnl / (pos.quantity * pos.entry_price) for pos in positions]
        if position_returns:
            var_95 = np.percentile(position_returns, 5)
        else:
            var_95 = 0.0
        
        # Calculate Sharpe ratio (simplified)
        if position_returns:
            returns_mean = np.mean(position_returns)
            returns_std = np.std(position_returns)
            sharpe_ratio = returns_mean / returns_std if returns_std > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Calculate max drawdown from position history
        max_drawdown = self._calculate_max_drawdown()
        
        return RiskMetrics(
            total_exposure=total_exposure,
            portfolio_value=account_value,
            leverage_ratio=leverage_ratio,
            var_95=float(var_95),
            max_drawdown=max_drawdown,
            sharpe_ratio=float(sharpe_ratio),
            beta=1.0,  # Simplified
            correlation=0.0  # Simplified
        )
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from position history"""
        if not self.position_history:
            return 0.0
        
        # Calculate cumulative P&L
        cumulative_pnl = []
        running_pnl = 0.0
        
        for pnl in self.position_history:
            running_pnl += pnl
            cumulative_pnl.append(running_pnl)
        
        if not cumulative_pnl:
            return 0.0
        
        # Calculate drawdown
        peak = pd.Series(cumulative_pnl).expanding().max()
        drawdown = (pd.Series(cumulative_pnl) - peak) / peak.abs()
        
        return drawdown.min()
    
    def check_risk_limits(self, positions: List[Position], 
                         account_value: float) -> Dict[str, bool]:
        """Check if portfolio violates risk limits"""
        metrics = self.calculate_portfolio_metrics(positions, account_value)
        
        risk_checks = {
            "leverage_ok": metrics.leverage_ratio <= 2.0,  # Max 2x leverage
            "var_ok": abs(metrics.var_95) <= self.max_portfolio_risk,
            "exposure_ok": metrics.total_exposure <= account_value * 1.5,
            "drawdown_ok": abs(metrics.max_drawdown) <= 0.20  # Max 20% drawdown
        }
        
        return risk_checks
    
    def should_reduce_risk(self, positions: List[Position], 
                          account_value: float) -> bool:
        """Determine if risk should be reduced"""
        risk_checks = self.check_risk_limits(positions, account_value)
        return not all(risk_checks.values())
    
    def suggest_position_adjustments(self, positions: List[Position], 
                                   account_value: float) -> List[Dict[str, Any]]:
        """Suggest position adjustments to reduce risk"""
        adjustments = []
        metrics = self.calculate_portfolio_metrics(positions, account_value)
        
        # Check leverage
        if metrics.leverage_ratio > 2.0:
            reduction_needed = (metrics.leverage_ratio - 2.0) / metrics.leverage_ratio
            adjustments.append({
                "type": "reduce_leverage",
                "reduction_pct": reduction_needed,
                "reason": "Leverage too high"
            })
        
        # Check individual position sizes
        for position in positions:
            position_value = abs(position.quantity * position.current_price)
            position_pct = position_value / account_value
            
            if position_pct > 0.1:  # Max 10% per position
                adjustments.append({
                    "type": "reduce_position",
                    "symbol": position.symbol,
                    "current_pct": position_pct,
                    "target_pct": 0.05,
                    "reason": "Position too large"
                })
        
        return adjustments

class RiskManager:
    """Main risk management class"""
    
    def __init__(self, risk_level: RiskLevel = RiskLevel.MODERATE,
                 stop_loss_pct: float = 0.05,
                 take_profit_pct: float = 0.10,
                 max_portfolio_risk: float = 0.06):
        
        self.position_sizer = PositionSizer(risk_level)
        self.stop_loss_manager = StopLossManager(stop_loss_pct, take_profit_pct)
        self.portfolio_risk_manager = PortfolioRiskManager(max_portfolio_risk)
        
        self.risk_level = risk_level
        self.positions = []
        self.account_value = 0.0
    
    def calculate_position_size(self, symbol: str, price: float, 
                              strategy: str = "kelly") -> int:
        """Calculate position size for a new trade"""
        if strategy == "kelly":
            # Simplified Kelly - would need historical data for proper calculation
            return int(self.position_sizer.kelly_criterion(0.55, 0.02, 0.01) * self.account_value / price)
        elif strategy == "volatility":
            # Would need volatility data
            return self.position_sizer.volatility_based(price, 0.02, self.account_value)
        else:
            # Default to equal weight
            return self.position_sizer.equal_weight(self.account_value, len(self.positions) + 1, price)
    
    def should_trade(self, symbol: str, side: str, quantity: int, 
                    price: float) -> bool:
        """Check if a trade should be executed based on risk rules"""
        # Check if we already have a position in this symbol
        existing_position = next((p for p in self.positions if p.symbol == symbol), None)
        
        if existing_position and existing_position.side == side:
            return False  # Don't double down
        
        # Check portfolio risk limits
        if self.portfolio_risk_manager.should_reduce_risk(self.positions, self.account_value):
            return False
        
        # Check position size limits
        position_value = quantity * price
        if position_value > self.account_value * 0.1:  # Max 10% per position
            return False
        
        return True
    
    def update_position(self, symbol: str, current_price: float):
        """Update position with current price"""
        for position in self.positions:
            if position.symbol == symbol:
                position.current_price = current_price
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                
                # Update trailing stop
                self.stop_loss_manager.update_trailing_stop(position)
                break
    
    def check_exit_signals(self, symbol: str, current_price: float) -> List[str]:
        """Check for exit signals"""
        exit_signals = []
        
        for position in self.positions:
            if position.symbol == symbol:
                # Update position price
                self.update_position(symbol, current_price)
                
                # Check stop loss
                if self.stop_loss_manager.should_stop_loss(position):
                    exit_signals.append("stop_loss")
                
                # Check take profit
                if self.stop_loss_manager.should_take_profit(position):
                    exit_signals.append("take_profit")
                
                # Check trailing stop
                if self.stop_loss_manager.check_trailing_stop(position):
                    exit_signals.append("trailing_stop")
                
                break
        
        return exit_signals
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        metrics = self.portfolio_risk_manager.calculate_portfolio_metrics(
            self.positions, self.account_value
        )
        
        risk_checks = self.portfolio_risk_manager.check_risk_limits(
            self.positions, self.account_value
        )
        
        adjustments = self.portfolio_risk_manager.suggest_position_adjustments(
            self.positions, self.account_value
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "account_value": self.account_value,
            "num_positions": len(self.positions),
            "risk_metrics": {
                "total_exposure": metrics.total_exposure,
                "leverage_ratio": metrics.leverage_ratio,
                "var_95": metrics.var_95,
                "max_drawdown": metrics.max_drawdown,
                "sharpe_ratio": metrics.sharpe_ratio
            },
            "risk_checks": risk_checks,
            "suggested_adjustments": adjustments,
            "risk_level": self.risk_level.value
        }

def main():
    """Main function for testing risk management"""
    # Initialize risk manager
    risk_manager = RiskManager(
        risk_level=RiskLevel.MODERATE,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        max_portfolio_risk=0.06
    )
    
    # Set account value
    risk_manager.account_value = 100000
    
    # Test position sizing
    position_size = risk_manager.calculate_position_size("AAPL", 150.0)
    print(f"Position size for AAPL: {position_size} shares")
    
    # Test risk report
    report = risk_manager.get_risk_report()
    print("\nRisk Report:")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    import json
    main() 