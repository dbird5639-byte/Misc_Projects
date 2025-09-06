"""
Risk Manager

Manages portfolio risk, position sizing, and risk limits.
"""

import time
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
try:
    from loguru import logger  # type: ignore
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Risk metrics data structure"""
    portfolio_value: float
    total_exposure: float
    max_drawdown: float
    var_95: float  # Value at Risk (95% confidence)
    sharpe_ratio: float
    position_concentration: float
    sector_exposure: Dict[str, float]
    timestamp: float

class RiskManager:
    """
    Manages portfolio risk and position sizing
    """
    
    def __init__(self, position_manager, risk_config):
        """
        Initialize risk manager
        
        Args:
            position_manager: Position manager instance
            risk_config: Risk configuration settings
        """
        self.position_manager = position_manager
        self.risk_config = risk_config
        self._lock = threading.Lock()
        self._risk_metrics: Optional[RiskMetrics] = None
        self._last_check = 0.0
        
        # Risk limits
        self.max_position_size = risk_config.max_position_size
        self.max_portfolio_exposure = risk_config.max_portfolio_exposure
        self.max_drawdown = risk_config.max_drawdown
        self.max_positions = risk_config.max_positions
        self.sector_limits = risk_config.sector_limits or {}
    
    def check_risk_limits(self) -> Dict[str, Any]:
        """
        Check all risk limits and return violations
        
        Returns:
            Dictionary of risk limit violations
        """
        try:
            violations = {}
            
            # Update risk metrics
            self._update_risk_metrics()
            
            if not self._risk_metrics:
                return violations
            
            # Check position size limits
            position_violations = self._check_position_limits()
            if position_violations:
                violations["position_limits"] = position_violations
            
            # Check portfolio exposure
            exposure_violations = self._check_portfolio_exposure()
            if exposure_violations:
                violations["portfolio_exposure"] = exposure_violations
            
            # Check drawdown
            drawdown_violations = self._check_drawdown()
            if drawdown_violations:
                violations["drawdown"] = drawdown_violations
            
            # Check sector limits
            sector_violations = self._check_sector_limits()
            if sector_violations:
                violations["sector_limits"] = sector_violations
            
            # Check position count
            count_violations = self._check_position_count()
            if count_violations:
                violations["position_count"] = count_violations
            
            if violations:
                logger.warning(f"Risk limit violations detected: {violations}")
            
            return violations
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return {}
    
    def _update_risk_metrics(self):
        """Update risk metrics"""
        try:
            summary = self.position_manager.get_position_summary()
            metrics = self.position_manager.calculate_portfolio_metrics()
            
            # Calculate sector exposure (simplified)
            sector_exposure = self._calculate_sector_exposure()
            
            # Calculate VaR (simplified)
            var_95 = self._calculate_var()
            
            # Calculate Sharpe ratio (simplified)
            sharpe_ratio = self._calculate_sharpe_ratio()
            
            self._risk_metrics = RiskMetrics(
                portfolio_value=summary["total_market_value"],
                total_exposure=summary["total_market_value"],
                max_drawdown=metrics.get("max_drawdown", 0.0),
                var_95=var_95,
                sharpe_ratio=sharpe_ratio,
                position_concentration=metrics.get("position_concentration", 0.0),
                sector_exposure=sector_exposure,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
    
    def _check_position_limits(self) -> List[str]:
        """Check individual position size limits"""
        violations = []
        
        positions = self.position_manager.get_positions()
        for position in positions:
            if abs(position.market_value) > self.max_position_size:
                violations.append(
                    f"Position {position.symbol} exceeds size limit: "
                    f"${position.market_value:.2f} > ${self.max_position_size:.2f}"
                )
        
        return violations
    
    def _check_portfolio_exposure(self) -> List[str]:
        """Check total portfolio exposure"""
        violations = []
        
        if not self._risk_metrics:
            return violations
        
        portfolio_value = self._risk_metrics.portfolio_value
        if portfolio_value > 0:
            exposure_ratio = self._risk_metrics.total_exposure / portfolio_value
            if exposure_ratio > self.max_portfolio_exposure:
                violations.append(
                    f"Portfolio exposure exceeds limit: "
                    f"{exposure_ratio:.2%} > {self.max_portfolio_exposure:.2%}"
                )
        
        return violations
    
    def _check_drawdown(self) -> List[str]:
        """Check maximum drawdown"""
        violations = []
        
        if not self._risk_metrics:
            return violations
        
        if self._risk_metrics.max_drawdown > self.max_drawdown:
            violations.append(
                f"Maximum drawdown exceeded: "
                f"{self._risk_metrics.max_drawdown:.2%} > {self.max_drawdown:.2%}"
            )
        
        return violations
    
    def _check_sector_limits(self) -> List[str]:
        """Check sector exposure limits"""
        violations = []
        
        if not self._risk_metrics or not self.sector_limits:
            return violations
        
        for sector, limit in self.sector_limits.items():
            if sector in self._risk_metrics.sector_exposure:
                exposure = self._risk_metrics.sector_exposure[sector]
                if exposure > limit:
                    violations.append(
                        f"Sector {sector} exposure exceeds limit: "
                        f"{exposure:.2%} > {limit:.2%}"
                    )
        
        return violations
    
    def _check_position_count(self) -> List[str]:
        """Check maximum number of positions"""
        violations = []
        
        positions = self.position_manager.get_positions()
        if len(positions) > self.max_positions:
            violations.append(
                f"Position count exceeds limit: "
                f"{len(positions)} > {self.max_positions}"
            )
        
        return violations
    
    def _calculate_sector_exposure(self) -> Dict[str, float]:
        """Calculate sector exposure (simplified)"""
        # In real implementation, would use sector classification data
        sector_exposure = {}
        positions = self.position_manager.get_positions()
        
        if not positions:
            return sector_exposure
        
        total_value = sum(abs(pos.market_value) for pos in positions)
        if total_value == 0:
            return sector_exposure
        
        # Simplified sector assignment
        for position in positions:
            # Mock sector assignment based on symbol
            if position.symbol in ["AAPL", "GOOGL", "MSFT", "TSLA"]:
                sector = "technology"
            elif position.symbol in ["JNJ", "PFE", "UNH"]:
                sector = "healthcare"
            elif position.symbol in ["JPM", "BAC", "WFC"]:
                sector = "financial"
            else:
                sector = "other"
            
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + abs(position.market_value) / total_value
        
        return sector_exposure
    
    def _calculate_var(self) -> float:
        """Calculate Value at Risk (simplified)"""
        # In real implementation, would use historical data and statistical methods
        positions = self.position_manager.get_positions()
        if not positions:
            return 0.0
        
        # Simplified VaR calculation
        total_value = sum(abs(pos.market_value) for pos in positions)
        return total_value * 0.02  # Assume 2% daily VaR
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio (simplified)"""
        # In real implementation, would use historical returns and risk-free rate
        metrics = self.position_manager.calculate_portfolio_metrics()
        pnl_percentage = metrics.get("pnl_percentage", 0.0)
        
        # Simplified Sharpe calculation
        if pnl_percentage > 0:
            return pnl_percentage / 10.0  # Assume 10% volatility
        else:
            return 0.0
    
    def calculate_position_size(self, symbol: str, price: float, 
                              strategy: str = "kelly") -> int:
        """
        Calculate optimal position size
        
        Args:
            symbol: Stock symbol
            price: Current price
            strategy: Position sizing strategy
            
        Returns:
            Recommended position size in shares
        """
        try:
            if strategy == "kelly":
                return self._kelly_position_size(symbol, price)
            elif strategy == "fixed":
                return self._fixed_position_size(price)
            elif strategy == "volatility":
                return self._volatility_position_size(symbol, price)
            else:
                logger.warning(f"Unknown position sizing strategy: {strategy}")
                return self._fixed_position_size(price)
                
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def _kelly_position_size(self, symbol: str, price: float) -> int:
        """Calculate position size using Kelly Criterion"""
        # In real implementation, would use historical win rate and odds
        # Simplified Kelly calculation
        win_rate = 0.6  # Assume 60% win rate
        avg_win = 0.05  # Assume 5% average win
        avg_loss = 0.03  # Assume 3% average loss
        
        if avg_loss > 0:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0.0, min(kelly_fraction, 0.25))  # Cap at 25%
        else:
            kelly_fraction = 0.1  # Default to 10%
        
        portfolio_value = self._risk_metrics.portfolio_value if self._risk_metrics else 100000
        position_value = portfolio_value * kelly_fraction
        
        return int(position_value / price)
    
    def _fixed_position_size(self, price: float) -> int:
        """Calculate fixed position size"""
        position_value = self.max_position_size * 0.1  # 10% of max position size
        return int(position_value / price)
    
    def _volatility_position_size(self, symbol: str, price: float) -> int:
        """Calculate position size based on volatility"""
        # In real implementation, would use historical volatility
        # Simplified volatility-based sizing
        volatility = 0.02  # Assume 2% daily volatility
        
        if volatility > 0:
            position_value = self.max_position_size * (0.02 / volatility)
        else:
            position_value = self.max_position_size * 0.1
        
        return int(position_value / price)
    
    def get_risk_metrics(self) -> Optional[RiskMetrics]:
        """Get current risk metrics"""
        return self._risk_metrics
    
    def should_trade(self, symbol: str, action: str, quantity: int, 
                    price: float) -> bool:
        """
        Check if a trade should be allowed based on risk limits
        
        Args:
            symbol: Stock symbol
            action: 'BUY' or 'SELL'
            quantity: Number of shares
            price: Price per share
            
        Returns:
            True if trade should be allowed
        """
        try:
            # Check position size limit
            position_value = abs(quantity * price)
            if position_value > self.max_position_size:
                logger.warning(f"Position size {position_value} exceeds limit {self.max_position_size}")
                return False
            
            # Check portfolio exposure
            if self._risk_metrics:
                new_exposure = self._risk_metrics.total_exposure + position_value
                if new_exposure > self._risk_metrics.portfolio_value * self.max_portfolio_exposure:
                    logger.warning("Trade would exceed portfolio exposure limit")
                    return False
            
            # Check position count
            current_positions = len(self.position_manager.get_positions())
            if action == "BUY" and not self.position_manager.has_position(symbol):
                if current_positions >= self.max_positions:
                    logger.warning("Trade would exceed maximum position count")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking trade approval: {e}")
            return False
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk management summary"""
        if not self._risk_metrics:
            return {}
        
        return {
            "portfolio_value": self._risk_metrics.portfolio_value,
            "total_exposure": self._risk_metrics.total_exposure,
            "max_drawdown": self._risk_metrics.max_drawdown,
            "var_95": self._risk_metrics.var_95,
            "sharpe_ratio": self._risk_metrics.sharpe_ratio,
            "position_concentration": self._risk_metrics.position_concentration,
            "sector_exposure": self._risk_metrics.sector_exposure,
            "risk_limits": {
                "max_position_size": self.max_position_size,
                "max_portfolio_exposure": self.max_portfolio_exposure,
                "max_drawdown": self.max_drawdown,
                "max_positions": self.max_positions
            }
        } 