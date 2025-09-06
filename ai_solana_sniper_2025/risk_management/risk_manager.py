"""
Risk Manager for AI-Powered Solana Meme Coin Sniper
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import math

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RiskAssessment:
    """Represents a risk assessment"""
    token_address: str
    risk_score: float  # 0.0 (low risk) to 1.0 (high risk)
    risk_level: str  # "low", "medium", "high", "extreme"
    risk_factors: List[str]
    position_size_recommendation: float  # Percentage of portfolio
    max_loss_amount: float
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    assessment_timestamp: Optional[datetime] = None


@dataclass
class PortfolioRisk:
    """Represents portfolio risk metrics"""
    total_value: float
    total_risk: float
    max_drawdown: float
    var_95: float  # Value at Risk (95% confidence)
    sharpe_ratio: float
    correlation_matrix: Dict[str, Dict[str, float]]
    risk_timestamp: Optional[datetime] = None


class RiskManager:
    """
    Manages risk for trading operations
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        
        # Risk parameters
        self.max_portfolio_risk = config.get("max_portfolio_risk", 0.03)  # 3%
        self.max_position_risk = config.get("max_position_risk", 0.01)   # 1%
        self.stop_loss_percentage = config.get("stop_loss_percentage", 0.15)  # 15%
        self.take_profit_percentage = config.get("take_profit_percentage", 0.5)  # 50%
        self.max_positions = config.get("max_positions", 5)
        self.max_daily_loss = config.get("max_daily_loss", 0.1)  # 10%
        self.circuit_breaker_threshold = config.get("circuit_breaker_threshold", 0.2)  # 20%
        
        # Kelly Criterion parameters
        self.kelly_enabled = config.get("kelly_enabled", True)
        self.kelly_fraction = config.get("kelly_fraction", 0.25)  # Use 25% of Kelly
        
        # Volatility parameters
        self.volatility_lookback = config.get("volatility_lookback", 24)  # hours
        self.volatility_threshold = config.get("volatility_threshold", 0.5)  # 50%
        
        # State management
        self.portfolio_value = 0.0
        self.daily_pnl = 0.0
        self.daily_start_value = 0.0
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.risk_assessments: Dict[str, RiskAssessment] = {}
        self.trade_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        
        # Circuit breaker state
        self.circuit_breaker_active = False
        self.circuit_breaker_triggered_at = None
        
    async def initialize(self):
        """Initialize the risk manager"""
        try:
            # Reset daily metrics
            self.daily_start_value = self.portfolio_value
            self.daily_pnl = 0.0
            
            logger.info("Risk manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize risk manager: {e}")
            return False
    
    async def assess_token_risk(self, token_address: str, token_data: Dict[str, Any], 
                              market_data: Dict[str, Any]) -> RiskAssessment:
        """Assess risk for a specific token"""
        try:
            logger.info(f"Assessing risk for token: {token_address}")
            
            # Calculate risk factors
            risk_factors = []
            risk_score = 0.0
            
            # Liquidity risk
            liquidity = token_data.get("liquidity", 0)
            if liquidity < 1000:
                risk_factors.append("very_low_liquidity")
                risk_score += 0.3
            elif liquidity < 10000:
                risk_factors.append("low_liquidity")
                risk_score += 0.2
            elif liquidity < 50000:
                risk_factors.append("moderate_liquidity")
                risk_score += 0.1
            
            # Volume risk
            volume_24h = token_data.get("volume_24h", 0)
            if volume_24h < 100:
                risk_factors.append("very_low_volume")
                risk_score += 0.25
            elif volume_24h < 1000:
                risk_factors.append("low_volume")
                risk_score += 0.15
            elif volume_24h < 10000:
                risk_factors.append("moderate_volume")
                risk_score += 0.1
            
            # Market cap risk
            market_cap = token_data.get("market_cap", 0)
            if market_cap < 10000:
                risk_factors.append("micro_cap")
                risk_score += 0.2
            elif market_cap < 100000:
                risk_factors.append("small_cap")
                risk_score += 0.1
            
            # Volatility risk
            volatility = self._calculate_volatility(token_data.get("price_history", []))
            if volatility > 0.8:
                risk_factors.append("extreme_volatility")
                risk_score += 0.3
            elif volatility > 0.5:
                risk_factors.append("high_volatility")
                risk_score += 0.2
            elif volatility > 0.3:
                risk_factors.append("moderate_volatility")
                risk_score += 0.1
            
            # Age risk
            launch_time = token_data.get("launch_time", time.time())
            age_hours = (time.time() - launch_time) / 3600
            if age_hours < 1:
                risk_factors.append("very_new")
                risk_score += 0.25
            elif age_hours < 24:
                risk_factors.append("new")
                risk_score += 0.15
            elif age_hours < 168:  # 1 week
                risk_factors.append("recent")
                risk_score += 0.1
            
            # Market sentiment risk
            sentiment = market_data.get("sentiment", "neutral")
            if sentiment == "bearish":
                risk_factors.append("bearish_market")
                risk_score += 0.1
            
            # Normalize risk score
            risk_score = min(1.0, risk_score)
            
            # Determine risk level
            if risk_score >= 0.8:
                risk_level = "extreme"
            elif risk_score >= 0.6:
                risk_level = "high"
            elif risk_score >= 0.4:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            # Calculate position size recommendation
            position_size = self._calculate_position_size(risk_score, token_data)
            
            # Calculate max loss amount
            max_loss_amount = self.portfolio_value * self.max_position_risk
            
            # Calculate stop loss and take profit prices
            current_price = token_data.get("price", 0)
            stop_loss_price = current_price * (1 - self.stop_loss_percentage) if current_price > 0 else None
            take_profit_price = current_price * (1 + self.take_profit_percentage) if current_price > 0 else None
            
            assessment = RiskAssessment(
                token_address=token_address,
                risk_score=risk_score,
                risk_level=risk_level,
                risk_factors=risk_factors,
                position_size_recommendation=position_size,
                max_loss_amount=max_loss_amount,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                assessment_timestamp=datetime.now()
            )
            
            # Cache assessment
            self.risk_assessments[token_address] = assessment
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing token risk: {e}")
            return self._create_default_risk_assessment(token_address)
    
    def _calculate_volatility(self, price_history: List[Dict[str, Any]]) -> float:
        """Calculate price volatility"""
        if len(price_history) < 2:
            return 0.5  # Default moderate volatility
        
        try:
            prices = [float(entry.get("price", 0)) for entry in price_history]
            if len(prices) < 2:
                return 0.5
            
            # Calculate returns
            returns = []
            for i in range(1, len(prices)):
                if prices[i-1] > 0:
                    returns.append((prices[i] - prices[i-1]) / prices[i-1])
            
            if not returns:
                return 0.5
            
            # Calculate standard deviation
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            volatility = variance ** 0.5
            
            return min(1.0, volatility)
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.5
    
    def _calculate_position_size(self, risk_score: float, token_data: Dict[str, Any]) -> float:
        """Calculate recommended position size"""
        try:
            # Base position size based on risk
            base_size = self.max_position_risk * (1 - risk_score)
            
            # Apply Kelly Criterion if enabled
            if self.kelly_enabled:
                kelly_size = self._calculate_kelly_criterion(token_data)
                base_size = min(base_size, kelly_size * self.kelly_fraction)
            
            # Adjust for portfolio concentration
            if len(self.positions) >= self.max_positions:
                base_size *= 0.5  # Reduce size if too many positions
            
            # Adjust for daily loss limit
            daily_loss_ratio = abs(self.daily_pnl) / max(self.daily_start_value, 1)
            if daily_loss_ratio > self.max_daily_loss * 0.5:
                base_size *= 0.5  # Reduce size if approaching daily loss limit
            
            # Ensure minimum and maximum bounds
            base_size = max(0.001, min(base_size, self.max_position_risk))
            
            return base_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return self.max_position_risk * 0.5
    
    def _calculate_kelly_criterion(self, token_data: Dict[str, Any]) -> float:
        """Calculate Kelly Criterion position size"""
        try:
            # Get historical win rate and average win/loss
            win_rate = self._calculate_win_rate()
            avg_win = self._calculate_average_win()
            avg_loss = self._calculate_average_loss()
            
            if avg_loss == 0:
                return 0.0
            
            # Kelly formula: f = (bp - q) / b
            # where b = odds received, p = probability of win, q = probability of loss
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Ensure positive and reasonable
            return max(0.0, min(kelly_fraction, 0.25))
            
        except Exception as e:
            logger.error(f"Error calculating Kelly Criterion: {e}")
            return 0.0
    
    def _calculate_win_rate(self) -> float:
        """Calculate historical win rate"""
        if self.total_trades == 0:
            return 0.5  # Default 50% win rate
        
        return self.winning_trades / self.total_trades
    
    def _calculate_average_win(self) -> float:
        """Calculate average winning trade"""
        winning_trades = [trade for trade in self.trade_history if trade.get("pnl", 0) > 0]
        
        if not winning_trades:
            return 0.1  # Default 10% average win
        
        total_win = sum(trade.get("pnl", 0) for trade in winning_trades)
        return total_win / len(winning_trades)
    
    def _calculate_average_loss(self) -> float:
        """Calculate average losing trade"""
        losing_trades = [trade for trade in self.trade_history if trade.get("pnl", 0) < 0]
        
        if not losing_trades:
            return 0.1  # Default 10% average loss
        
        total_loss = abs(sum(trade.get("pnl", 0) for trade in losing_trades))
        return total_loss / len(losing_trades)
    
    async def check_portfolio_risk(self) -> PortfolioRisk:
        """Check overall portfolio risk"""
        try:
            # Calculate portfolio metrics
            total_value = self.portfolio_value
            total_risk = self._calculate_total_portfolio_risk()
            max_drawdown = self._calculate_max_drawdown()
            var_95 = self._calculate_value_at_risk(0.95)
            sharpe_ratio = self._calculate_sharpe_ratio()
            correlation_matrix = self._calculate_correlation_matrix()
            
            portfolio_risk = PortfolioRisk(
                total_value=total_value,
                total_risk=total_risk,
                max_drawdown=max_drawdown,
                var_95=var_95,
                sharpe_ratio=sharpe_ratio,
                correlation_matrix=correlation_matrix,
                risk_timestamp=datetime.now()
            )
            
            # Check circuit breaker
            await self._check_circuit_breaker(portfolio_risk)
            
            return portfolio_risk
            
        except Exception as e:
            logger.error(f"Error checking portfolio risk: {e}")
            return self._create_default_portfolio_risk()
    
    def _calculate_total_portfolio_risk(self) -> float:
        """Calculate total portfolio risk"""
        try:
            total_risk = 0.0
            
            for position in self.positions.values():
                position_value = position.get("value", 0)
                position_risk = position.get("risk_score", 0.5)
                total_risk += (position_value / self.portfolio_value) * position_risk
            
            return min(1.0, total_risk)
            
        except Exception as e:
            logger.error(f"Error calculating total portfolio risk: {e}")
            return 0.5
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        return self.max_drawdown
    
    def _calculate_value_at_risk(self, confidence: float) -> float:
        """Calculate Value at Risk"""
        try:
            if not self.trade_history:
                return self.portfolio_value * 0.1  # Default 10% VaR
            
            # Calculate returns
            returns = []
            for trade in self.trade_history:
                if trade.get("portfolio_value_before", 0) > 0:
                    return_val = (trade.get("portfolio_value_after", 0) - trade.get("portfolio_value_before", 0)) / trade.get("portfolio_value_before", 0)
                    returns.append(return_val)
            
            if not returns:
                return self.portfolio_value * 0.1
            
            # Sort returns and find VaR
            returns.sort()
            var_index = int(len(returns) * (1 - confidence))
            var_return = returns[var_index] if var_index < len(returns) else returns[-1]
            
            return abs(var_return) * self.portfolio_value
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return self.portfolio_value * 0.1
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        try:
            if not self.trade_history:
                return 0.0
            
            # Calculate returns
            returns = []
            for trade in self.trade_history:
                if trade.get("portfolio_value_before", 0) > 0:
                    return_val = (trade.get("portfolio_value_after", 0) - trade.get("portfolio_value_before", 0)) / trade.get("portfolio_value_before", 0)
                    returns.append(return_val)
            
            if not returns:
                return 0.0
            
            # Calculate mean and standard deviation
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            std_dev = variance ** 0.5
            
            if std_dev == 0:
                return 0.0
            
            # Assume risk-free rate of 0 for simplicity
            sharpe_ratio = mean_return / std_dev
            
            return sharpe_ratio
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix between positions"""
        try:
            correlation_matrix = {}
            
            position_addresses = list(self.positions.keys())
            
            for i, addr1 in enumerate(position_addresses):
                correlation_matrix[addr1] = {}
                for j, addr2 in enumerate(position_addresses):
                    if i == j:
                        correlation_matrix[addr1][addr2] = 1.0
                    else:
                        # Calculate correlation based on price movements
                        correlation = self._calculate_position_correlation(addr1, addr2)
                        correlation_matrix[addr1][addr2] = correlation
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return {}
    
    def _calculate_position_correlation(self, addr1: str, addr2: str) -> float:
        """Calculate correlation between two positions"""
        try:
            # This would typically use historical price data
            # For now, return a default correlation
            return 0.3  # Moderate correlation
            
        except Exception as e:
            logger.error(f"Error calculating position correlation: {e}")
            return 0.0
    
    async def _check_circuit_breaker(self, portfolio_risk: PortfolioRisk):
        """Check if circuit breaker should be triggered"""
        try:
            # Check daily loss limit
            daily_loss_ratio = abs(self.daily_pnl) / max(self.daily_start_value, 1)
            
            if daily_loss_ratio >= self.circuit_breaker_threshold:
                if not self.circuit_breaker_active:
                    self.circuit_breaker_active = True
                    self.circuit_breaker_triggered_at = datetime.now()
                    logger.warning(f"Circuit breaker triggered: Daily loss {daily_loss_ratio:.2%}")
            
            # Check if circuit breaker should be reset (next day)
            if self.circuit_breaker_active and self.circuit_breaker_triggered_at:
                if datetime.now().date() > self.circuit_breaker_triggered_at.date():
                    self.circuit_breaker_active = False
                    self.circuit_breaker_triggered_at = None
                    logger.info("Circuit breaker reset for new day")
                    
        except Exception as e:
            logger.error(f"Error checking circuit breaker: {e}")
    
    async def update_position(self, token_address: str, amount: float, price: float, action: str):
        """Update position in risk manager"""
        try:
            if action == "buy":
                if token_address in self.positions:
                    # Update existing position
                    position = self.positions[token_address]
                    total_amount = position.get("amount", 0) + amount
                    total_value = (position.get("amount", 0) * position.get("average_price", 0)) + (amount * price)
                    position["average_price"] = total_value / total_amount
                    position["amount"] = total_amount
                    position["current_price"] = price
                    position["value"] = total_amount * price
                    position["last_updated"] = datetime.now()
                else:
                    # Create new position
                    self.positions[token_address] = {
                        "amount": amount,
                        "average_price": price,
                        "current_price": price,
                        "value": amount * price,
                        "entry_time": datetime.now(),
                        "last_updated": datetime.now(),
                        "risk_score": 0.5  # Default risk score
                    }
            
            elif action == "sell":
                if token_address in self.positions:
                    position = self.positions[token_address]
                    position["amount"] -= amount
                    position["current_price"] = price
                    position["value"] = position["amount"] * price
                    position["last_updated"] = datetime.now()
                    
                    # Remove position if amount is 0
                    if position["amount"] <= 0:
                        del self.positions[token_address]
                        
        except Exception as e:
            logger.error(f"Error updating position: {e}")
    
    async def record_trade(self, trade_data: Dict[str, Any]):
        """Record a completed trade"""
        try:
            # Update trade history
            self.trade_history.append(trade_data)
            
            # Update metrics
            self.total_trades += 1
            pnl = trade_data.get("pnl", 0)
            
            if pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # Update portfolio value
            self.portfolio_value = trade_data.get("portfolio_value_after", self.portfolio_value)
            
            # Update daily P&L
            self.daily_pnl += pnl
            
            # Update drawdown
            self._update_drawdown()
            
            # Keep only recent trade history
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-1000:]
                
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    def _update_drawdown(self):
        """Update drawdown calculations"""
        try:
            if self.portfolio_value > 0:
                # Calculate current drawdown
                peak_value = max([trade.get("portfolio_value_after", 0) for trade in self.trade_history] + [self.portfolio_value])
                current_drawdown = (peak_value - self.portfolio_value) / peak_value
                
                self.current_drawdown = current_drawdown
                self.max_drawdown = max(self.max_drawdown, current_drawdown)
                
        except Exception as e:
            logger.error(f"Error updating drawdown: {e}")
    
    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed based on risk limits"""
        if not self.enabled:
            return False
        
        if self.circuit_breaker_active:
            return False
        
        if len(self.positions) >= self.max_positions:
            return False
        
        # Check daily loss limit
        daily_loss_ratio = abs(self.daily_pnl) / max(self.daily_start_value, 1)
        if daily_loss_ratio >= self.max_daily_loss:
            return False
        
        return True
    
    def get_position_size_limit(self, token_address: str) -> float:
        """Get maximum position size for a token"""
        try:
            # Get risk assessment
            assessment = self.risk_assessments.get(token_address)
            if assessment:
                return assessment.position_size_recommendation
            
            # Default position size
            return self.max_position_risk * 0.5
            
        except Exception as e:
            logger.error(f"Error getting position size limit: {e}")
            return self.max_position_risk * 0.5
    
    def _create_default_risk_assessment(self, token_address: str) -> RiskAssessment:
        """Create default risk assessment"""
        return RiskAssessment(
            token_address=token_address,
            risk_score=0.8,  # High risk by default
            risk_level="high",
            risk_factors=["analysis_failed"],
            position_size_recommendation=self.max_position_risk * 0.25,
            max_loss_amount=self.portfolio_value * self.max_position_risk * 0.25,
            assessment_timestamp=datetime.now()
        )
    
    def _create_default_portfolio_risk(self) -> PortfolioRisk:
        """Create default portfolio risk"""
        return PortfolioRisk(
            total_value=self.portfolio_value,
            total_risk=0.5,
            max_drawdown=self.max_drawdown,
            var_95=self.portfolio_value * 0.1,
            sharpe_ratio=0.0,
            correlation_matrix={},
            risk_timestamp=datetime.now()
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        total_trades = self.winning_trades + self.losing_trades
        win_rate = self.winning_trades / max(total_trades, 1)
        
        return {
            "total_trades": total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": win_rate,
            "max_drawdown": self.max_drawdown,
            "current_drawdown": self.current_drawdown,
            "daily_pnl": self.daily_pnl,
            "portfolio_value": self.portfolio_value,
            "active_positions": len(self.positions),
            "circuit_breaker_active": self.circuit_breaker_active
        } 