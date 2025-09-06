"""
Independent Trading Capital Planner
Plan capital requirements and risk management for independent trading

Based on insights from experienced traders about the importance of
sufficient capital and proper risk management for sustainable trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MarketType(Enum):
    """Types of markets for capital planning"""
    FUTURES = "futures"
    CRYPTO = "crypto"
    EQUITIES = "equities"
    FOREX = "forex"
    OPTIONS = "options"


class RiskTolerance(Enum):
    """Risk tolerance levels"""
    CONSERVATIVE = 0.10  # 10% max drawdown
    MODERATE = 0.15     # 15% max drawdown
    AGGRESSIVE = 0.20   # 20% max drawdown
    EXTREME = 0.25      # 25% max drawdown


@dataclass
class CapitalPlan:
    """Capital planning results"""
    target_income: float
    risk_tolerance: RiskTolerance
    markets: List[MarketType]
    required_capital: float
    monthly_risk: float
    expected_return: float
    position_sizes: Dict[str, float]
    risk_per_trade: float
    max_positions: int
    monthly_trades: int
    breakeven_months: int
    recommendations: List[str]


@dataclass
class MarketRequirements:
    """Market-specific capital requirements"""
    market: MarketType
    min_capital: float
    typical_leverage: float
    margin_requirements: Dict[str, float]
    typical_returns: float
    risk_characteristics: Dict[str, float]
    recommended_strategies: List[str]


class CapitalPlanner:
    """
    Independent trading capital planner
    
    Helps traders plan capital requirements and risk management
    based on their goals and market preferences.
    """
    
    def __init__(self):
        """Initialize the capital planner"""
        self.market_requirements = self._load_market_requirements()
        self.risk_models = self._load_risk_models()
        
        logger.info("Capital Planner initialized")
    
    def _load_market_requirements(self) -> Dict[MarketType, MarketRequirements]:
        """Load market-specific requirements"""
        requirements = {}
        
        # Futures market requirements
        requirements[MarketType.FUTURES] = MarketRequirements(
            market=MarketType.FUTURES,
            min_capital=25000,
            typical_leverage=10.0,
            margin_requirements={
                "ES": 12000,  # E-mini S&P 500
                "NQ": 15000,  # E-mini NASDAQ
                "YM": 8000,   # E-mini Dow
                "CL": 8000,   # Crude Oil
                "GC": 8000,   # Gold
                "ZB": 4000,   # 30-Year Treasury
                "ZC": 3000,   # Corn
                "ZS": 4000,   # Soybeans
                "ZW": 3000,   # Wheat
                "6E": 4000    # Euro FX
            },
            typical_returns=0.15,  # 15% annual return
            risk_characteristics={
                "volatility": 0.20,
                "correlation": 0.30,
                "liquidity": 0.90,
                "tax_efficiency": 0.85
            },
            recommended_strategies=[
                "trend_following",
                "mean_reversion",
                "momentum",
                "breakout"
            ]
        )
        
        # Crypto market requirements
        requirements[MarketType.CRYPTO] = MarketRequirements(
            market=MarketType.CRYPTO,
            min_capital=10000,
            typical_leverage=5.0,
            margin_requirements={
                "BTC": 5000,
                "ETH": 3000,
                "ADA": 1000,
                "SOL": 2000,
                "DOT": 1500,
                "LINK": 1500,
                "UNI": 2000,
                "AAVE": 2500
            },
            typical_returns=0.25,  # 25% annual return
            risk_characteristics={
                "volatility": 0.40,
                "correlation": 0.70,
                "liquidity": 0.60,
                "tax_efficiency": 0.50
            },
            recommended_strategies=[
                "momentum",
                "trend_following",
                "arbitrage",
                "grid_trading"
            ]
        )
        
        # Equities market requirements
        requirements[MarketType.EQUITIES] = MarketRequirements(
            market=MarketType.EQUITIES,
            min_capital=50000,
            typical_leverage=2.0,
            margin_requirements={
                "SPY": 5000,
                "QQQ": 5000,
                "IWM": 4000,
                "AAPL": 3000,
                "MSFT": 3000,
                "GOOGL": 4000,
                "TSLA": 5000,
                "NVDA": 4000
            },
            typical_returns=0.10,  # 10% annual return
            risk_characteristics={
                "volatility": 0.15,
                "correlation": 0.60,
                "liquidity": 0.95,
                "tax_efficiency": 0.80
            },
            recommended_strategies=[
                "value_investing",
                "momentum",
                "dividend_growth",
                "sector_rotation"
            ]
        )
        
        # Forex market requirements
        requirements[MarketType.FOREX] = MarketRequirements(
            market=MarketType.FOREX,
            min_capital=15000,
            typical_leverage=20.0,
            margin_requirements={
                "EUR/USD": 1000,
                "GBP/USD": 1200,
                "USD/JPY": 1000,
                "USD/CHF": 1000,
                "AUD/USD": 1000,
                "USD/CAD": 1000,
                "NZD/USD": 1000,
                "EUR/GBP": 1200
            },
            typical_returns=0.12,  # 12% annual return
            risk_characteristics={
                "volatility": 0.12,
                "correlation": 0.40,
                "liquidity": 0.85,
                "tax_efficiency": 0.70
            },
            recommended_strategies=[
                "trend_following",
                "mean_reversion",
                "carry_trade",
                "breakout"
            ]
        )
        
        return requirements
    
    def _load_risk_models(self) -> Dict[str, Dict[str, float]]:
        """Load risk models for different scenarios"""
        return {
            "conservative": {
                "max_drawdown": 0.10,
                "risk_per_trade": 0.01,
                "max_positions": 3,
                "leverage_multiplier": 0.5
            },
            "moderate": {
                "max_drawdown": 0.15,
                "risk_per_trade": 0.02,
                "max_positions": 5,
                "leverage_multiplier": 0.75
            },
            "aggressive": {
                "max_drawdown": 0.20,
                "risk_per_trade": 0.03,
                "max_positions": 8,
                "leverage_multiplier": 1.0
            },
            "extreme": {
                "max_drawdown": 0.25,
                "risk_per_trade": 0.05,
                "max_positions": 12,
                "leverage_multiplier": 1.5
            }
        }
    
    def create_plan(
        self,
        target_income: float,
        risk_tolerance: RiskTolerance,
        markets: List[MarketType],
        available_capital: Optional[float] = None,
        time_horizon: int = 12
    ) -> CapitalPlan:
        """
        Create a comprehensive capital plan
        
        Args:
            target_income: Target annual income
            risk_tolerance: Risk tolerance level
            markets: List of markets to trade
            available_capital: Available capital (if known)
            time_horizon: Time horizon in months
            
        Returns:
            CapitalPlan with recommendations
        """
        logger.info(f"Creating capital plan for ${target_income:,.0f} annual income")
        
        # Calculate required capital based on markets and risk
        required_capital = self._calculate_required_capital(
            target_income, risk_tolerance, markets
        )
        
        # Calculate monthly risk allocation
        monthly_risk = self._calculate_monthly_risk(
            required_capital, risk_tolerance
        )
        
        # Calculate expected returns
        expected_return = self._calculate_expected_return(markets, risk_tolerance)
        
        # Calculate position sizes
        position_sizes = self._calculate_position_sizes(
            required_capital, markets, risk_tolerance
        )
        
        # Calculate trading parameters
        risk_per_trade = self.risk_models[risk_tolerance.name.lower()]["risk_per_trade"]
        max_positions = self.risk_models[risk_tolerance.name.lower()]["max_positions"]
        
        # Calculate monthly trade volume
        monthly_trades = self._calculate_monthly_trades(
            target_income, expected_return, required_capital
        )
        
        # Calculate breakeven timeline
        breakeven_months = self._calculate_breakeven_months(
            target_income, expected_return, required_capital
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            required_capital, available_capital, markets, risk_tolerance
        )
        
        return CapitalPlan(
            target_income=target_income,
            risk_tolerance=risk_tolerance,
            markets=markets,
            required_capital=required_capital,
            monthly_risk=monthly_risk,
            expected_return=expected_return,
            position_sizes=position_sizes,
            risk_per_trade=risk_per_trade,
            max_positions=max_positions,
            monthly_trades=monthly_trades,
            breakeven_months=breakeven_months,
            recommendations=recommendations
        )
    
    def _calculate_required_capital(
        self,
        target_income: float,
        risk_tolerance: RiskTolerance,
        markets: List[MarketType]
    ) -> float:
        """Calculate required capital based on income target and markets"""
        
        # Get market requirements
        market_reqs = [self.market_requirements[market] for market in markets]
        
        # Calculate weighted average return
        total_weight = len(markets)
        avg_return = sum(req.typical_returns for req in market_reqs) / total_weight
        
        # Calculate required capital based on return and risk tolerance
        # Using the 2x drawdown rule (Kevin Davy's approach)
        max_drawdown = risk_tolerance.value
        min_return_drawdown_ratio = 2.0
        
        # Required return to meet 2x drawdown rule
        required_return = max_drawdown * min_return_drawdown_ratio
        
        # Calculate capital needed for target income
        if required_return > 0:
            base_capital = target_income / required_return
        else:
            base_capital = target_income / avg_return
        
        # Adjust for market minimums
        min_capital_requirements = [req.min_capital for req in market_reqs]
        max_min_capital = max(min_capital_requirements) if min_capital_requirements else 25000
        
        # Use the higher of calculated or minimum requirements
        required_capital = max(base_capital, max_min_capital)
        
        # Apply safety margin (20% buffer)
        required_capital *= 1.2
        
        return required_capital
    
    def _calculate_monthly_risk(
        self,
        capital: float,
        risk_tolerance: RiskTolerance
    ) -> float:
        """Calculate monthly risk allocation"""
        max_drawdown = risk_tolerance.value
        monthly_risk = capital * max_drawdown / 12  # Monthly drawdown limit
        
        return monthly_risk
    
    def _calculate_expected_return(
        self,
        markets: List[MarketType],
        risk_tolerance: RiskTolerance
    ) -> float:
        """Calculate expected return based on markets and risk tolerance"""
        
        # Get market returns
        market_reqs = [self.market_requirements[market] for market in markets]
        avg_market_return = sum(req.typical_returns for req in market_reqs) / len(markets)
        
        # Adjust for risk tolerance
        risk_multiplier = {
            RiskTolerance.CONSERVATIVE: 0.8,
            RiskTolerance.MODERATE: 1.0,
            RiskTolerance.AGGRESSIVE: 1.2,
            RiskTolerance.EXTREME: 1.4
        }
        
        expected_return = avg_market_return * risk_multiplier[risk_tolerance]
        
        return expected_return
    
    def _calculate_position_sizes(
        self,
        capital: float,
        markets: List[MarketType],
        risk_tolerance: RiskTolerance
    ) -> Dict[str, float]:
        """Calculate position sizes for each market"""
        
        position_sizes = {}
        risk_model = self.risk_models[risk_tolerance.name.lower()]
        risk_per_trade = risk_model["risk_per_trade"]
        
        for market in markets:
            market_req = self.market_requirements[market]
            
            # Calculate position size based on risk per trade
            position_size = capital * risk_per_trade
            
            # Adjust for market-specific requirements
            leverage_multiplier = risk_model["leverage_multiplier"]
            adjusted_size = position_size * leverage_multiplier
            
            # Ensure minimum position size
            min_position = market_req.min_capital * 0.1  # 10% of minimum capital
            final_size = max(adjusted_size, min_position)
            
            position_sizes[market.value] = final_size
        
        return position_sizes
    
    def _calculate_monthly_trades(
        self,
        target_income: float,
        expected_return: float,
        capital: float
    ) -> int:
        """Calculate required monthly trades"""
        
        # Monthly income target
        monthly_income = target_income / 12
        
        # Expected monthly return
        monthly_return = expected_return / 12
        
        # Required monthly return
        required_monthly_return = monthly_income / capital
        
        # Estimate trades needed (assuming average trade return)
        avg_trade_return = 0.02  # 2% average trade return
        monthly_trades = int(required_monthly_return / avg_trade_return)
        
        # Ensure reasonable bounds
        monthly_trades = max(5, min(100, monthly_trades))
        
        return monthly_trades
    
    def _calculate_breakeven_months(
        self,
        target_income: float,
        expected_return: float,
        capital: float
    ) -> int:
        """Calculate months to breakeven"""
        
        # Monthly income target
        monthly_income = target_income / 12
        
        # Expected monthly return
        monthly_return = expected_return / 12
        
        # Expected monthly profit
        monthly_profit = capital * monthly_return
        
        if monthly_profit > 0:
            breakeven_months = int(monthly_income / monthly_profit)
        else:
            breakeven_months = float('inf')
        
        return breakeven_months
    
    def _generate_recommendations(
        self,
        required_capital: float,
        available_capital: Optional[float],
        markets: List[MarketType],
        risk_tolerance: RiskTolerance
    ) -> List[str]:
        """Generate recommendations based on analysis"""
        
        recommendations = []
        
        # Capital recommendations
        if available_capital is None:
            recommendations.append(f"Save ${required_capital:,.0f} before starting live trading")
        elif available_capital < required_capital:
            shortfall = required_capital - available_capital
            recommendations.append(f"Additional ${shortfall:,.0f} needed for proper risk management")
            recommendations.append("Consider starting with smaller capital and scaling up")
        else:
            recommendations.append("Sufficient capital available for proper risk management")
        
        # Market recommendations
        if len(markets) == 1:
            recommendations.append("Consider diversifying across multiple markets")
        elif len(markets) > 3:
            recommendations.append("Consider focusing on fewer markets for better expertise")
        
        # Risk recommendations
        if risk_tolerance == RiskTolerance.EXTREME:
            recommendations.append("Consider reducing risk tolerance for sustainable trading")
        elif risk_tolerance == RiskTolerance.CONSERVATIVE:
            recommendations.append("Conservative approach is good for long-term success")
        
        # Strategy recommendations
        for market in markets:
            market_req = self.market_requirements[market]
            recommendations.append(f"For {market.value}: Focus on {', '.join(market_req.recommended_strategies[:2])}")
        
        # General recommendations
        recommendations.append("Start with paper trading to validate strategies")
        recommendations.append("Implement proper risk management from day one")
        recommendations.append("Track performance and adjust strategies based on results")
        
        return recommendations
    
    def analyze_capital_efficiency(
        self,
        capital: float,
        markets: List[MarketType]
    ) -> Dict[str, Any]:
        """
        Analyze capital efficiency across different markets
        
        Args:
            capital: Available capital
            markets: Markets to analyze
            
        Returns:
            Capital efficiency analysis
        """
        
        analysis = {}
        
        for market in markets:
            market_req = self.market_requirements[market]
            
            # Calculate leverage potential
            leverage_potential = capital * market_req.typical_leverage
            
            # Calculate margin efficiency
            avg_margin = np.mean(list(market_req.margin_requirements.values()))
            margin_efficiency = capital / avg_margin
            
            # Calculate diversification potential
            num_positions = int(capital / avg_margin)
            
            # Calculate expected return
            expected_return = market_req.typical_returns
            
            analysis[market.value] = {
                "leverage_potential": leverage_potential,
                "margin_efficiency": margin_efficiency,
                "num_positions": num_positions,
                "expected_return": expected_return,
                "risk_score": market_req.risk_characteristics["volatility"],
                "liquidity_score": market_req.risk_characteristics["liquidity"],
                "tax_efficiency": market_req.risk_characteristics["tax_efficiency"]
            }
        
        return analysis
    
    def create_savings_plan(
        self,
        target_capital: float,
        current_savings: float,
        monthly_income: float,
        savings_rate: float = 0.2
    ) -> Dict[str, Any]:
        """
        Create a savings plan to reach target capital
        
        Args:
            target_capital: Target capital amount
            current_savings: Current savings
            monthly_income: Monthly income
            savings_rate: Percentage of income to save
            
        Returns:
            Savings plan details
        """
        
        # Calculate required additional savings
        additional_needed = target_capital - current_savings
        
        if additional_needed <= 0:
            return {
                "status": "sufficient_capital",
                "message": "You already have sufficient capital",
                "months_to_target": 0,
                "monthly_savings_needed": 0
            }
        
        # Calculate monthly savings
        monthly_savings = monthly_income * savings_rate
        
        if monthly_savings <= 0:
            return {
                "status": "insufficient_income",
                "message": "Insufficient income to save",
                "months_to_target": float('inf'),
                "monthly_savings_needed": additional_needed
            }
        
        # Calculate months to target
        months_to_target = additional_needed / monthly_savings
        
        # Calculate years to target
        years_to_target = months_to_target / 12
        
        return {
            "status": "savings_plan",
            "additional_needed": additional_needed,
            "monthly_savings": monthly_savings,
            "months_to_target": months_to_target,
            "years_to_target": years_to_target,
            "recommendations": self._generate_savings_recommendations(
                months_to_target, monthly_savings, additional_needed
            )
        }
    
    def _generate_savings_recommendations(
        self,
        months_to_target: float,
        monthly_savings: float,
        additional_needed: float
    ) -> List[str]:
        """Generate savings recommendations"""
        
        recommendations = []
        
        if months_to_target > 60:  # More than 5 years
            recommendations.append("Consider increasing savings rate to 30-40%")
            recommendations.append("Look for additional income sources")
            recommendations.append("Consider starting with smaller capital")
        elif months_to_target > 24:  # More than 2 years
            recommendations.append("Consider increasing savings rate to 25%")
            recommendations.append("Look for ways to reduce expenses")
        else:
            recommendations.append("Savings plan is reasonable")
            recommendations.append("Consider starting paper trading while saving")
        
        recommendations.append("Use this time to learn and develop strategies")
        recommendations.append("Build emergency fund before trading capital")
        
        return recommendations


# Example usage
if __name__ == "__main__":
    # Initialize capital planner
    planner = CapitalPlanner()
    
    # Create capital plan
    plan = planner.create_plan(
        target_income=50000,
        risk_tolerance=RiskTolerance.MODERATE,
        markets=[MarketType.FUTURES, MarketType.CRYPTO],
        available_capital=30000
    )
    
    print("Capital Plan Analysis:")
    print(f"Required Capital: ${plan.required_capital:,.0f}")
    print(f"Monthly Risk: ${plan.monthly_risk:,.0f}")
    print(f"Expected Return: {plan.expected_return:.1%}")
    print(f"Risk Per Trade: {plan.risk_per_trade:.1%}")
    print(f"Max Positions: {plan.max_positions}")
    print(f"Monthly Trades: {plan.monthly_trades}")
    print(f"Breakeven Months: {plan.breakeven_months}")
    
    print("\nPosition Sizes:")
    for market, size in plan.position_sizes.items():
        print(f"  {market}: ${size:,.0f}")
    
    print("\nRecommendations:")
    for rec in plan.recommendations:
        print(f"  • {rec}")
    
    # Analyze capital efficiency
    efficiency = planner.analyze_capital_efficiency(30000, [MarketType.FUTURES, MarketType.CRYPTO])
    print("\nCapital Efficiency Analysis:")
    for market, data in efficiency.items():
        print(f"\n{market.upper()}:")
        print(f"  Leverage Potential: ${data['leverage_potential']:,.0f}")
        print(f"  Expected Positions: {data['num_positions']}")
        print(f"  Expected Return: {data['expected_return']:.1%}")
        print(f"  Risk Score: {data['risk_score']:.2f}")
    
    # Create savings plan
    savings_plan = planner.create_savings_plan(
        target_capital=100000,
        current_savings=20000,
        monthly_income=8000,
        savings_rate=0.25
    )
    
    print(f"\nSavings Plan:")
    print(f"Status: {savings_plan['status']}")
    print(f"Additional Needed: ${savings_plan.get('additional_needed', 0):,.0f}")
    print(f"Months to Target: {savings_plan.get('months_to_target', 0):.1f}")
    print(f"Years to Target: {savings_plan.get('years_to_target', 0):.1f}") 