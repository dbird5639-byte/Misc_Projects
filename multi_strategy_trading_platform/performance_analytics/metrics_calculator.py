"""
Performance Analytics - Metrics Calculator
Calculate comprehensive performance metrics for trading strategies and portfolios
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate comprehensive performance metrics"""
    
    def __init__(self):
        """Initialize metrics calculator"""
        pass
    
    def calculate_portfolio_metrics(
        self,
        equity_curve: pd.Series,
        trades: List[Dict],
        positions: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Calculate portfolio performance metrics"""
        if equity_curve.empty:
            return self._empty_metrics()
        
        # Basic metrics
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
        returns = equity_curve.pct_change().dropna()
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade metrics
        trade_metrics = self._calculate_trade_metrics(trades)
        
        return {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "win_rate": trade_metrics["win_rate"],
            "profit_factor": trade_metrics["profit_factor"],
            "total_trades": trade_metrics["total_trades"],
            "avg_trade": trade_metrics["avg_trade"],
            "avg_win": trade_metrics["avg_win"],
            "avg_loss": trade_metrics["avg_loss"],
            "max_consecutive_losses": trade_metrics["max_consecutive_losses"]
        }
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        return drawdown.min()
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if returns.std() == 0:
            return 0
        return (returns.mean() - risk_free_rate/252) / returns.std() * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0 or negative_returns.std() == 0:
            return 0
        return (returns.mean() - risk_free_rate/252) / negative_returns.std() * np.sqrt(252)
    
    def _calculate_trade_metrics(self, trades: List[Dict]) -> Dict[str, Any]:
        """Calculate trade-based metrics"""
        if not trades:
            return self._empty_trade_metrics()
        
        pnls = [trade.get('realized_pnl', 0) for trade in trades]
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_trade = np.mean(pnls) if pnls else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        for pnl in pnls:
            if pnl < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        return {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": len(trades),
            "avg_trade": avg_trade,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_consecutive_losses": max_consecutive_losses
        }
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics"""
        return {
            "total_return": 0,
            "max_drawdown": 0,
            "sharpe_ratio": 0,
            "sortino_ratio": 0,
            "calmar_ratio": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "total_trades": 0,
            "avg_trade": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "max_consecutive_losses": 0
        }
    
    def _empty_trade_metrics(self) -> Dict[str, Any]:
        """Return empty trade metrics"""
        return {
            "win_rate": 0,
            "profit_factor": 0,
            "total_trades": 0,
            "avg_trade": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "max_consecutive_losses": 0
        } 