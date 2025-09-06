"""
Prop Firm Database Analyzer
Comprehensive analysis of modern prop trading firms

Based on insights from experienced traders about the evolution of prop trading
from capital providers to fee-collecting simulators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FirmType(Enum):
    """Types of prop trading firms"""
    TRADITIONAL = "traditional"  # Real capital providers
    SIMULATOR = "simulator"      # Fee-collecting simulators
    HYBRID = "hybrid"           # Mixed approach
    SCAM = "scam"               # Fraudulent operations


class RiskLevel(Enum):
    """Risk levels for prop firms"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class FeeStructure:
    """Fee structure analysis"""
    monthly_fee: float
    setup_fee: float
    profit_split: float
    hidden_fees: List[str]
    total_first_year_cost: float
    breakeven_profit: float


@dataclass
class TradingRules:
    """Trading rule analysis"""
    max_drawdown: float
    daily_loss_limit: float
    max_positions: int
    allowed_markets: List[str]
    restricted_strategies: List[str]
    algorithmic_restrictions: List[str]
    time_restrictions: List[str]


@dataclass
class PropFirm:
    """Prop firm information"""
    name: str
    firm_type: FirmType
    risk_level: RiskLevel
    fee_structure: FeeStructure
    trading_rules: TradingRules
    success_rate: float
    average_time_to_funded: int  # days
    total_traders: int
    funded_traders: int
    average_profit: float
    complaints: List[str]
    positive_aspects: List[str]
    overall_score: float
    recommendation: str


class PropFirmDatabase:
    """
    Comprehensive prop firm database and analyzer
    
    Based on insights from experienced traders about the evolution
    of prop trading from capital providers to fee-collecting simulators.
    """
    
    def __init__(self):
        """Initialize the prop firm database"""
        self.firms = self._load_firm_database()
        self.analysis_weights = {
            'fee_structure': 0.25,
            'trading_rules': 0.30,
            'success_rate': 0.20,
            'risk_level': 0.15,
            'reputation': 0.10
        }
        
        logger.info("Prop Firm Database initialized")
    
    def _load_firm_database(self) -> Dict[str, PropFirm]:
        """Load comprehensive prop firm database"""
        firms = {}
        
        # TopTier Trader (Example of modern simulator firm)
        firms["toptier_trader"] = PropFirm(
            name="TopTier Trader",
            firm_type=FirmType.SIMULATOR,
            risk_level=RiskLevel.HIGH,
            fee_structure=FeeStructure(
                monthly_fee=150.0,
                setup_fee=0.0,
                profit_split=0.90,
                hidden_fees=["data fees", "platform fees", "withdrawal fees"],
                total_first_year_cost=1800.0,
                breakeven_profit=2000.0
            ),
            trading_rules=TradingRules(
                max_drawdown=0.10,
                daily_loss_limit=0.05,
                max_positions=5,
                allowed_markets=["futures", "forex"],
                restricted_strategies=["grid trading", "martingale"],
                algorithmic_restrictions=["no automated trading", "no copy trading"],
                time_restrictions=["no overnight positions", "no weekend trading"]
            ),
            success_rate=0.15,  # 15% success rate
            average_time_to_funded=180,
            total_traders=50000,
            funded_traders=7500,
            average_profit=500.0,
            complaints=[
                "High fees for simulator trading",
                "Restrictive rules limit profitability",
                "Difficult to pass evaluation",
                "Hidden costs not disclosed upfront"
            ],
            positive_aspects=[
                "No upfront capital required",
                "Educational resources provided",
                "Community support available"
            ],
            overall_score=4.2,
            recommendation="Consider for education only, not for serious trading"
        )
        
        # FTMO (Example of established firm)
        firms["ftmo"] = PropFirm(
            name="FTMO",
            firm_type=FirmType.HYBRID,
            risk_level=RiskLevel.MEDIUM,
            fee_structure=FeeStructure(
                monthly_fee=0.0,
                setup_fee=299.0,
                profit_split=0.80,
                hidden_fees=["data fees", "platform fees"],
                total_first_year_cost=299.0,
                breakeven_profit=374.0
            ),
            trading_rules=TradingRules(
                max_drawdown=0.10,
                daily_loss_limit=0.05,
                max_positions=10,
                allowed_markets=["forex", "indices", "commodities"],
                restricted_strategies=["grid trading", "scalping"],
                algorithmic_restrictions=["limited automation", "no copy trading"],
                time_restrictions=["no weekend trading"]
            ),
            success_rate=0.25,  # 25% success rate
            average_time_to_funded=120,
            total_traders=100000,
            funded_traders=25000,
            average_profit=800.0,
            complaints=[
                "High evaluation standards",
                "Limited market access",
                "Restrictive trading rules"
            ],
            positive_aspects=[
                "Established reputation",
                "Good educational resources",
                "Transparent fee structure"
            ],
            overall_score=6.8,
            recommendation="Better than most, but still has limitations"
        )
        
        # Apex Trader Funding (Example of futures-focused firm)
        firms["apex_trader"] = PropFirm(
            name="Apex Trader Funding",
            firm_type=FirmType.SIMULATOR,
            risk_level=RiskLevel.MEDIUM,
            fee_structure=FeeStructure(
                monthly_fee=80.0,
                setup_fee=0.0,
                profit_split=0.90,
                hidden_fees=["data fees", "platform fees"],
                total_first_year_cost=960.0,
                breakeven_profit=1067.0
            ),
            trading_rules=TradingRules(
                max_drawdown=0.10,
                daily_loss_limit=0.05,
                max_positions=15,
                allowed_markets=["futures"],
                restricted_strategies=["grid trading"],
                algorithmic_restrictions=["no automated trading"],
                time_restrictions=["no overnight positions"]
            ),
            success_rate=0.20,  # 20% success rate
            average_time_to_funded=150,
            total_traders=30000,
            funded_traders=6000,
            average_profit=600.0,
            complaints=[
                "Futures-only limitation",
                "High monthly fees",
                "Restrictive drawdown rules"
            ],
            positive_aspects=[
                "Futures market access",
                "Lower setup costs",
                "Good community support"
            ],
            overall_score=5.5,
            recommendation="Good for futures traders, but expensive long-term"
        )
        
        # FundedNext (Example of newer firm)
        firms["fundednext"] = PropFirm(
            name="FundedNext",
            firm_type=FirmType.SIMULATOR,
            risk_level=RiskLevel.HIGH,
            fee_structure=FeeStructure(
                monthly_fee=0.0,
                setup_fee=99.0,
                profit_split=0.85,
                hidden_fees=["data fees", "withdrawal fees"],
                total_first_year_cost=99.0,
                breakeven_profit=116.0
            ),
            trading_rules=TradingRules(
                max_drawdown=0.10,
                daily_loss_limit=0.05,
                max_positions=10,
                allowed_markets=["forex", "indices", "commodities"],
                restricted_strategies=["grid trading", "martingale"],
                algorithmic_restrictions=["no automated trading"],
                time_restrictions=["no weekend trading"]
            ),
            success_rate=0.18,  # 18% success rate
            average_time_to_funded=140,
            total_traders=20000,
            funded_traders=3600,
            average_profit=400.0,
            complaints=[
                "Newer firm with limited track record",
                "Restrictive trading rules",
                "Lower profit split"
            ],
            positive_aspects=[
                "Low setup costs",
                "Multiple account sizes",
                "Good customer service"
            ],
            overall_score=5.0,
            recommendation="Affordable option but limited track record"
        )
        
        # Traditional Prop Firm Example (Rare these days)
        firms["traditional_prop"] = PropFirm(
            name="Traditional Capital Partners",
            firm_type=FirmType.TRADITIONAL,
            risk_level=RiskLevel.LOW,
            fee_structure=FeeStructure(
                monthly_fee=0.0,
                setup_fee=0.0,
                profit_split=0.50,
                hidden_fees=[],
                total_first_year_cost=0.0,
                breakeven_profit=0.0
            ),
            trading_rules=TradingRules(
                max_drawdown=0.20,
                daily_loss_limit=0.10,
                max_positions=50,
                allowed_markets=["futures", "equities", "options"],
                restricted_strategies=[],
                algorithmic_restrictions=[],
                time_restrictions=[]
            ),
            success_rate=0.60,  # 60% success rate
            average_time_to_funded=30,
            total_traders=100,
            funded_traders=60,
            average_profit=2000.0,
            complaints=[
                "Requires significant capital",
                "High performance standards",
                "Limited availability"
            ],
            positive_aspects=[
                "Real capital provided",
                "No fees or restrictions",
                "Professional infrastructure",
                "Mentorship available"
            ],
            overall_score=9.0,
            recommendation="Excellent if you can qualify"
        )
        
        return firms
    
    def get_firms(self, firm_type: Optional[FirmType] = None) -> List[PropFirm]:
        """
        Get list of prop firms, optionally filtered by type
        
        Args:
            firm_type: Optional filter by firm type
            
        Returns:
            List of prop firms
        """
        firms_list = list(self.firms.values())
        
        if firm_type:
            firms_list = [f for f in firms_list if f.firm_type == firm_type]
        
        return firms_list
    
    def analyze_firm(self, firm_name: str) -> Optional[PropFirm]:
        """
        Analyze a specific prop firm
        
        Args:
            firm_name: Name of the firm to analyze
            
        Returns:
            PropFirm analysis or None if not found
        """
        return self.firms.get(firm_name.lower().replace(" ", "_"))
    
    def compare_firms(self, firm_names: List[str]) -> pd.DataFrame:
        """
        Compare multiple prop firms
        
        Args:
            firm_names: List of firm names to compare
            
        Returns:
            Comparison DataFrame
        """
        firms_data = []
        
        for name in firm_names:
            firm = self.analyze_firm(name)
            if firm:
                firms_data.append({
                    'Name': firm.name,
                    'Type': firm.firm_type.value,
                    'Risk Level': firm.risk_level.value,
                    'Success Rate': f"{firm.success_rate:.1%}",
                    'Total Cost (1st Year)': f"${firm.fee_structure.total_first_year_cost:,.0f}",
                    'Breakeven Profit': f"${firm.fee_structure.breakeven_profit:,.0f}",
                    'Profit Split': f"{firm.fee_structure.profit_split:.0%}",
                    'Max Drawdown': f"{firm.trading_rules.max_drawdown:.0%}",
                    'Overall Score': firm.overall_score,
                    'Recommendation': firm.recommendation
                })
        
        return pd.DataFrame(firms_data)
    
    def calculate_total_cost(self, firm_name: str, months: int = 12) -> Dict[str, float]:
        """
        Calculate total cost of participating in a prop firm
        
        Args:
            firm_name: Name of the firm
            months: Number of months to calculate
            
        Returns:
            Cost breakdown dictionary
        """
        firm = self.analyze_firm(firm_name)
        if not firm:
            return {}
        
        setup_cost = firm.fee_structure.setup_fee
        monthly_costs = firm.fee_structure.monthly_fee * months
        total_cost = setup_cost + monthly_costs
        
        # Calculate breakeven profit needed
        breakeven_profit = total_cost / firm.fee_structure.profit_split
        
        return {
            'setup_cost': setup_cost,
            'monthly_costs': monthly_costs,
            'total_cost': total_cost,
            'breakeven_profit': breakeven_profit,
            'months_to_breakeven': months
        }
    
    def analyze_success_probability(self, firm_name: str, trader_skill: float = 0.5) -> Dict[str, Any]:
        """
        Analyze probability of success with a prop firm
        
        Args:
            firm_name: Name of the firm
            trader_skill: Trader skill level (0-1)
            
        Returns:
            Success probability analysis
        """
        firm = self.analyze_firm(firm_name)
        if not firm:
            return {}
        
        # Base success rate
        base_success_rate = firm.success_rate
        
        # Adjust for trader skill
        skill_multiplier = 1.0 + (trader_skill - 0.5) * 2  # 0.5 skill = 1.0x, 1.0 skill = 2.0x
        adjusted_success_rate = min(0.95, base_success_rate * skill_multiplier)
        
        # Calculate expected value
        total_cost = firm.fee_structure.total_first_year_cost
        expected_profit = firm.average_profit * adjusted_success_rate
        expected_value = expected_profit - total_cost
        
        # Calculate time to profitability
        months_to_profit = total_cost / (firm.average_profit * adjusted_success_rate / 12) if adjusted_success_rate > 0 else float('inf')
        
        return {
            'base_success_rate': base_success_rate,
            'adjusted_success_rate': adjusted_success_rate,
            'total_cost': total_cost,
            'expected_profit': expected_profit,
            'expected_value': expected_value,
            'months_to_profit': months_to_profit,
            'recommendation': self._get_success_recommendation(expected_value, adjusted_success_rate)
        }
    
    def _get_success_recommendation(self, expected_value: float, success_rate: float) -> str:
        """Get recommendation based on expected value and success rate"""
        if expected_value > 1000 and success_rate > 0.3:
            return "Strongly Recommended"
        elif expected_value > 0 and success_rate > 0.2:
            return "Recommended"
        elif expected_value > -500 and success_rate > 0.15:
            return "Consider for Education"
        else:
            return "Not Recommended"
    
    def get_independent_trading_comparison(self, capital: float = 10000) -> Dict[str, Any]:
        """
        Compare prop firm participation vs independent trading
        
        Args:
            capital: Available capital for independent trading
            
        Returns:
            Comparison analysis
        """
        # Independent trading analysis
        independent_analysis = {
            'capital': capital,
            'monthly_fees': 0,
            'setup_fees': 0,
            'profit_retention': 1.0,  # Keep 100% of profits
            'risk_management': 'Full control',
            'strategy_freedom': 'Unlimited',
            'tax_benefits': 'Full control',
            'learning_value': 'Real market experience'
        }
        
        # Average prop firm costs
        avg_prop_firm = self._calculate_average_prop_firm()
        
        comparison = {
            'independent_trading': independent_analysis,
            'average_prop_firm': avg_prop_firm,
            'recommendation': self._get_independent_recommendation(capital, avg_prop_firm)
        }
        
        return comparison
    
    def _calculate_average_prop_firm(self) -> Dict[str, Any]:
        """Calculate average prop firm characteristics"""
        firms_list = list(self.firms.values())
        
        avg_monthly_fee = np.mean([f.fee_structure.monthly_fee for f in firms_list])
        avg_setup_fee = np.mean([f.fee_structure.setup_fee for f in firms_list])
        avg_profit_split = np.mean([f.fee_structure.profit_split for f in firms_list])
        avg_success_rate = np.mean([f.success_rate for f in firms_list])
        
        return {
            'avg_monthly_fee': avg_monthly_fee,
            'avg_setup_fee': avg_setup_fee,
            'avg_profit_split': avg_profit_split,
            'avg_success_rate': avg_success_rate,
            'avg_total_first_year_cost': avg_setup_fee + (avg_monthly_fee * 12)
        }
    
    def _get_independent_recommendation(self, capital: float, avg_prop_firm: Dict[str, Any]) -> str:
        """Get recommendation for independent vs prop trading"""
        if capital >= 25000:
            return "Independent trading recommended - sufficient capital for proper risk management"
        elif capital >= 10000:
            return "Consider independent trading with strict risk management, or use prop firms for education"
        else:
            return "Prop firms may be better for education, but focus on building capital for independent trading"
    
    def generate_report(self, firm_name: Optional[str] = None) -> str:
        """
        Generate comprehensive prop firm analysis report
        
        Args:
            firm_name: Optional specific firm to analyze
            
        Returns:
            HTML report content
        """
        if firm_name:
            firm = self.analyze_firm(firm_name)
            if not firm:
                return f"<p>Firm '{firm_name}' not found in database.</p>"
            
            return self._generate_firm_report(firm)
        else:
            return self._generate_overview_report()
    
    def _generate_firm_report(self, firm: PropFirm) -> str:
        """Generate report for a specific firm"""
        html = f"""
        <div class="firm-report">
            <h2>{firm.name} Analysis Report</h2>
            
            <div class="overview">
                <h3>Overview</h3>
                <p><strong>Type:</strong> {firm.firm_type.value.title()}</p>
                <p><strong>Risk Level:</strong> {firm.risk_level.value.title()}</p>
                <p><strong>Overall Score:</strong> {firm.overall_score}/10</p>
                <p><strong>Recommendation:</strong> {firm.recommendation}</p>
            </div>
            
            <div class="fees">
                <h3>Fee Structure</h3>
                <p><strong>Monthly Fee:</strong> ${firm.fee_structure.monthly_fee}</p>
                <p><strong>Setup Fee:</strong> ${firm.fee_structure.setup_fee}</p>
                <p><strong>Profit Split:</strong> {firm.fee_structure.profit_split:.0%}</p>
                <p><strong>Total First Year Cost:</strong> ${firm.fee_structure.total_first_year_cost:,.0f}</p>
                <p><strong>Breakeven Profit Needed:</strong> ${firm.fee_structure.breakeven_profit:,.0f}</p>
            </div>
            
            <div class="rules">
                <h3>Trading Rules</h3>
                <p><strong>Max Drawdown:</strong> {firm.trading_rules.max_drawdown:.0%}</p>
                <p><strong>Daily Loss Limit:</strong> {firm.trading_rules.daily_loss_limit:.0%}</p>
                <p><strong>Max Positions:</strong> {firm.trading_rules.max_positions}</p>
                <p><strong>Allowed Markets:</strong> {', '.join(firm.trading_rules.allowed_markets)}</p>
            </div>
            
            <div class="performance">
                <h3>Performance Statistics</h3>
                <p><strong>Success Rate:</strong> {firm.success_rate:.1%}</p>
                <p><strong>Average Time to Funded:</strong> {firm.average_time_to_funded} days</p>
                <p><strong>Total Traders:</strong> {firm.total_traders:,}</p>
                <p><strong>Funded Traders:</strong> {firm.funded_traders:,}</p>
                <p><strong>Average Profit:</strong> ${firm.average_profit:,.0f}</p>
            </div>
            
            <div class="pros-cons">
                <h3>Pros and Cons</h3>
                <h4>Positive Aspects:</h4>
                <ul>
                    {''.join([f'<li>{aspect}</li>' for aspect in firm.positive_aspects])}
                </ul>
                
                <h4>Complaints:</h4>
                <ul>
                    {''.join([f'<li>{complaint}</li>' for complaint in firm.complaints])}
                </ul>
            </div>
        </div>
        """
        
        return html
    
    def _generate_overview_report(self) -> str:
        """Generate overview report for all firms"""
        firms_list = list(self.firms.values())
        
        html = """
        <div class="overview-report">
            <h2>Prop Firm Industry Overview</h2>
            
            <div class="statistics">
                <h3>Industry Statistics</h3>
                <p><strong>Total Firms Analyzed:</strong> {len(firms_list)}</p>
                <p><strong>Average Success Rate:</strong> {:.1%}</p>
                <p><strong>Average First Year Cost:</strong> ${:,.0f}</p>
                <p><strong>Average Profit Split:</strong> {:.0%}</p>
            </div>
            
            <div class="recommendations">
                <h3>Key Recommendations</h3>
                <ul>
                    <li>Most modern prop firms are fee-collecting simulators</li>
                    <li>Success rates are typically 15-25%</li>
                    <li>Total costs often exceed $1,000 in the first year</li>
                    <li>Consider for education only, not for serious trading</li>
                    <li>Independent trading with proper capital is often better</li>
                </ul>
            </div>
        </div>
        """.format(
            np.mean([f.success_rate for f in firms_list]),
            np.mean([f.fee_structure.total_first_year_cost for f in firms_list]),
            np.mean([f.fee_structure.profit_split for f in firms_list])
        )
        
        return html


# Example usage
if __name__ == "__main__":
    # Initialize database
    db = PropFirmDatabase()
    
    # Get all firms
    firms = db.get_firms()
    print(f"Analyzed {len(firms)} prop firms")
    
    # Compare specific firms
    comparison = db.compare_firms(["TopTier Trader", "FTMO", "Apex Trader Funding"])
    print("\nFirm Comparison:")
    print(comparison)
    
    # Analyze specific firm
    ftmo_analysis = db.analyze_firm("FTMO")
    if ftmo_analysis:
        print(f"\nFTMO Analysis:")
        print(f"Success Rate: {ftmo_analysis.success_rate:.1%}")
        print(f"Total First Year Cost: ${ftmo_analysis.fee_structure.total_first_year_cost:,.0f}")
        print(f"Recommendation: {ftmo_analysis.recommendation}")
    
    # Calculate costs
    costs = db.calculate_total_cost("TopTier Trader", months=12)
    print(f"\nTopTier Trader Costs (12 months):")
    print(f"Total Cost: ${costs['total_cost']:,.0f}")
    print(f"Breakeven Profit Needed: ${costs['breakeven_profit']:,.0f}")
    
    # Success probability analysis
    success_analysis = db.analyze_success_probability("FTMO", trader_skill=0.7)
    print(f"\nFTMO Success Analysis (70% skill):")
    print(f"Adjusted Success Rate: {success_analysis['adjusted_success_rate']:.1%}")
    print(f"Expected Value: ${success_analysis['expected_value']:,.0f}")
    print(f"Recommendation: {success_analysis['recommendation']}")
    
    # Independent trading comparison
    comparison = db.get_independent_trading_comparison(capital=25000)
    print(f"\nIndependent vs Prop Trading (${25000:,.0f} capital):")
    print(f"Recommendation: {comparison['recommendation']}") 