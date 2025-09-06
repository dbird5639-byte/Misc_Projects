"""
Risk Management Module
Implement Kevin Davy's risk management principles including 2x drawdown rule
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """Risk management for multi-strategy portfolios"""
    
    def __init__(self):
        """Initialize risk manager"""
        self.max_drawdown = 0.15
        self.risk_per_trade = 0.02
        self.max_correlation = 0.7
        self.position_limits = {}
        self.risk_metrics = {}
        
    def initialize(
        self,
        max_drawdown: float = 0.15,
        risk_per_trade: float = 0.02,
        max_correlation: float = 0.7
    ) -> None:
        """Initialize risk parameters"""
        self.max_drawdown = max_drawdown
        self.risk_per_trade = risk_per_trade
        self.max_correlation = max_correlation
        
        logger.info(f"Risk manager initialized: max_dd={max_drawdown}, risk_per_trade={risk_per_trade}")
    
    def check_risk_limits(self, portfolio: Any) -> bool:
        """Check if portfolio is within risk limits"""
        summary = portfolio.get_portfolio_summary()
        
        # Check drawdown
        if summary["current_drawdown"] < -self.max_drawdown:
            logger.warning(f"Drawdown limit exceeded: {summary['current_drawdown']:.2%}")
            return False
        
        return True
    
    def calculate_position_size(
        self,
        capital: float,
        risk_per_trade: float,
        stop_loss_pct: float
    ) -> float:
        """Calculate position size based on risk"""
        risk_amount = capital * risk_per_trade
        position_size = risk_amount / stop_loss_pct
        return position_size 