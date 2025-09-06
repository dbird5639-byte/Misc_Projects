"""
Prop Trading Analysis & Strategy Platform
Main Entry Point

Comprehensive platform for analyzing prop trading firms and building
independent algorithmic trading systems.
"""

import asyncio
import logging
from typing import Dict, List, Optional
from pathlib import Path

# Import platform components
from prop_firm_analyzer.firm_database import PropFirmDatabase
from independent_trading.capital_planner import CapitalPlanner, RiskTolerance, MarketType
from tax_optimization.futures_tax_calculator import FuturesTaxCalculator, InstrumentType
from market_analysis.futures_analyzer import FuturesAnalyzer
from market_analysis.crypto_analyzer import CryptoAnalyzer
from education_platform.course_library import CourseLibrary
from strategy_library.futures_strategies import FuturesStrategyBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PropTradingAnalysisPlatform:
    """
    Main platform class for prop trading analysis and independent trading tools
    
    Provides comprehensive analysis of prop firms and tools for building
    independent algorithmic trading systems.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the platform"""
        self.config = self._load_config(config_path)
        
        # Initialize core components
        self.prop_firm_db = PropFirmDatabase()
        self.capital_planner = CapitalPlanner()
        self.tax_calculator = FuturesTaxCalculator()
        self.futures_analyzer = FuturesAnalyzer()
        self.crypto_analyzer = CryptoAnalyzer()
        self.course_library = CourseLibrary()
        self.strategy_builder = FuturesStrategyBuilder()
        
        # Platform state
        self.analysis_results = {}
        self.user_profiles = {}
        
        logger.info("Prop Trading Analysis Platform initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load platform configuration"""
        # Default configuration
        config = {
            "data_dir": "data",
            "reports_dir": "reports",
            "education_dir": "education",
            "default_tax_rate": 0.24,
            "default_state": "TX",
            "risk_tolerance": "moderate",
            "target_markets": ["futures", "crypto"],
            "analysis_depth": "comprehensive"
        }
        
        # Load from file if provided
        if config_path and Path(config_path).exists():
            import json
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
        
        return config
    
    def analyze_prop_firms(self, firm_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze prop trading firms
        
        Args:
            firm_names: List of specific firms to analyze (None for all)
            
        Returns:
            Analysis results
        """
        logger.info("Analyzing prop trading firms")
        
        if firm_names:
            firms = [self.prop_firm_db.analyze_firm(name) for name in firm_names]
            firms = [f for f in firms if f is not None]
        else:
            firms = self.prop_firm_db.get_firms()
        
        # Generate comparison
        firm_names_list = [f.name for f in firms]
        comparison = self.prop_firm_db.compare_firms(firm_names_list)
        
        # Calculate industry statistics
        avg_success_rate = sum(f.success_rate for f in firms) / len(firms)
        avg_total_cost = sum(f.fee_structure.total_first_year_cost for f in firms) / len(firms)
        avg_profit_split = sum(f.fee_structure.profit_split for f in firms) / len(firms)
        
        results = {
            "firms": firms,
            "comparison": comparison,
            "industry_stats": {
                "total_firms": len(firms),
                "avg_success_rate": avg_success_rate,
                "avg_total_cost": avg_total_cost,
                "avg_profit_split": avg_profit_split
            },
            "recommendations": self._generate_prop_firm_recommendations(firms)
        }
        
        self.analysis_results["prop_firms"] = results
        return results
    
    def _generate_prop_firm_recommendations(self, firms: List) -> List[str]:
        """Generate recommendations based on prop firm analysis"""
        recommendations = []
        
        # Calculate average success rate
        avg_success_rate = sum(f.success_rate for f in firms) / len(firms)
        
        if avg_success_rate < 0.20:
            recommendations.append("Industry success rates are very low (15-25%)")
            recommendations.append("Most participants lose money to fees and restrictions")
        
        # Analyze fee structures
        avg_total_cost = sum(f.fee_structure.total_first_year_cost for f in firms) / len(firms)
        if avg_total_cost > 1000:
            recommendations.append(f"Average first-year costs exceed ${avg_total_cost:,.0f}")
            recommendations.append("Consider if education value justifies the cost")
        
        # Check for simulator firms
        simulator_firms = [f for f in firms if f.firm_type.value == "simulator"]
        if len(simulator_firms) > len(firms) * 0.8:
            recommendations.append("Most modern firms are fee-collecting simulators")
            recommendations.append("Consider independent trading with proper capital")
        
        # General recommendations
        recommendations.append("Use prop firms for education, not as primary income source")
        recommendations.append("Focus on building capital for independent trading")
        recommendations.append("Implement proper risk management regardless of approach")
        
        return recommendations
    
    def create_capital_plan(
        self,
        target_income: float,
        risk_tolerance: str = "moderate",
        markets: List[str] = None,
        available_capital: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Create capital planning analysis
        
        Args:
            target_income: Target annual income
            risk_tolerance: Risk tolerance level
            markets: List of markets to trade
            available_capital: Available capital (if known)
            
        Returns:
            Capital planning results
        """
        logger.info(f"Creating capital plan for ${target_income:,.0f} annual income")
        
        # Convert risk tolerance string to enum
        risk_enum = getattr(RiskTolerance, risk_tolerance.upper())
        
        # Convert market strings to enums
        if markets is None:
            markets = self.config["target_markets"]
        
        market_enums = [getattr(MarketType, market.upper()) for market in markets]
        
        # Create capital plan
        plan = self.capital_planner.create_plan(
            target_income=target_income,
            risk_tolerance=risk_enum,
            markets=market_enums,
            available_capital=available_capital
        )
        
        # Analyze capital efficiency
        efficiency = self.capital_planner.analyze_capital_efficiency(
            plan.required_capital, market_enums
        )
        
        # Create savings plan if needed
        if available_capital is None or available_capital < plan.required_capital:
            savings_plan = self.capital_planner.create_savings_plan(
                target_capital=plan.required_capital,
                current_savings=available_capital or 0,
                monthly_income=target_income / 12 * 0.3,  # Assume 30% savings rate
                savings_rate=0.25
            )
        else:
            savings_plan = {"status": "sufficient_capital"}
        
        results = {
            "plan": plan,
            "efficiency": efficiency,
            "savings_plan": savings_plan,
            "recommendations": self._generate_capital_recommendations(plan, efficiency)
        }
        
        self.analysis_results["capital_plan"] = results
        return results
    
    def _generate_capital_recommendations(self, plan, efficiency) -> List[str]:
        """Generate capital planning recommendations"""
        recommendations = []
        
        if plan.required_capital > 100000:
            recommendations.append("High capital requirement - consider scaling up gradually")
            recommendations.append("Focus on building skills while saving capital")
        
        if plan.breakeven_months > 24:
            recommendations.append("Long breakeven timeline - ensure sufficient runway")
            recommendations.append("Consider starting with smaller targets")
        
        # Market-specific recommendations
        for market, data in efficiency.items():
            if data["risk_score"] > 0.3:
                recommendations.append(f"High volatility in {market} - implement strict risk controls")
            if data["liquidity_score"] < 0.7:
                recommendations.append(f"Lower liquidity in {market} - expect higher slippage")
        
        recommendations.append("Start with paper trading to validate strategies")
        recommendations.append("Implement proper risk management from day one")
        
        return recommendations
    
    def analyze_tax_benefits(
        self,
        annual_profit: float,
        federal_rate: Optional[float] = None,
        state: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze tax benefits across different instruments
        
        Args:
            annual_profit: Annual trading profit
            federal_rate: Federal tax rate
            state: State of residence
            
        Returns:
            Tax analysis results
        """
        logger.info(f"Analyzing tax benefits for ${annual_profit:,.0f} annual profit")
        
        if federal_rate is None:
            federal_rate = self.config["default_tax_rate"]
        if state is None:
            state = self.config["default_state"]
        
        # Compare instruments
        comparison = self.tax_calculator.compare_instruments(
            profit=annual_profit,
            holding_period=365,
            federal_rate=federal_rate,
            state=state
        )
        
        # Calculate annual impact for each instrument
        futures_impact = self.tax_calculator.calculate_annual_tax_impact(
            annual_profit, federal_rate, state, InstrumentType.FUTURES
        )
        
        stocks_impact = self.tax_calculator.calculate_annual_tax_impact(
            annual_profit, federal_rate, state, InstrumentType.STOCKS
        )
        
        crypto_impact = self.tax_calculator.calculate_annual_tax_impact(
            annual_profit, federal_rate, state, InstrumentType.CRYPTO
        )
        
        # Compare states
        states = ["TX", "CA", "NY", "FL", "IL"]
        state_comparison = self.tax_calculator.compare_states(
            annual_profit, federal_rate, states
        )
        
        results = {
            "comparison": comparison,
            "annual_impacts": {
                "futures": futures_impact,
                "stocks": stocks_impact,
                "crypto": crypto_impact
            },
            "state_comparison": state_comparison,
            "recommendations": self._generate_tax_recommendations(comparison, state_comparison)
        }
        
        self.analysis_results["tax_analysis"] = results
        return results
    
    def _generate_tax_recommendations(self, comparison, state_comparison) -> List[str]:
        """Generate tax optimization recommendations"""
        recommendations = []
        
        # Instrument recommendations
        if comparison.savings_vs_stocks > 1000:
            recommendations.append(f"Futures provide ${comparison.savings_vs_stocks:,.0f} tax savings vs stocks")
        
        if comparison.savings_vs_crypto > 1000:
            recommendations.append(f"Futures provide ${comparison.savings_vs_crypto:,.0f} tax savings vs crypto")
        
        # State recommendations
        best_state = state_comparison.loc[state_comparison['Futures_Tax_Efficiency'].str.rstrip('%').astype(float).idxmax()]
        recommendations.append(f"Best tax efficiency in {best_state['State']}: {best_state['Futures_Tax_Efficiency']}")
        
        recommendations.append("Consider 60/40 rule benefits for futures trading")
        recommendations.append("Consult tax professional for specific advice")
        
        return recommendations
    
    def analyze_market_opportunities(self, markets: List[str] = None) -> Dict[str, Any]:
        """
        Analyze market opportunities and characteristics
        
        Args:
            markets: List of markets to analyze
            
        Returns:
            Market analysis results
        """
        logger.info("Analyzing market opportunities")
        
        if markets is None:
            markets = self.config["target_markets"]
        
        results = {}
        
        if "futures" in markets:
            futures_analysis = self.futures_analyzer.analyze_markets()
            results["futures"] = futures_analysis
        
        if "crypto" in markets:
            crypto_analysis = self.crypto_analyzer.analyze_markets()
            results["crypto"] = crypto_analysis
        
        # Generate market recommendations
        recommendations = self._generate_market_recommendations(results)
        results["recommendations"] = recommendations
        
        self.analysis_results["market_analysis"] = results
        return results
    
    def _generate_market_recommendations(self, market_analysis) -> List[str]:
        """Generate market-specific recommendations"""
        recommendations = []
        
        if "futures" in market_analysis:
            futures_data = market_analysis["futures"]
            recommendations.append("Futures offer leverage, tax benefits, and diversification")
            recommendations.append("Consider agricultural and energy markets for uncorrelated returns")
        
        if "crypto" in market_analysis:
            crypto_data = market_analysis["crypto"]
            recommendations.append("Crypto offers 24/7 trading and easy automation")
            recommendations.append("High volatility requires strict risk management")
        
        recommendations.append("Diversify across multiple markets for better risk-adjusted returns")
        recommendations.append("Focus on markets with high liquidity for better execution")
        
        return recommendations
    
    def get_educational_resources(self, skill_level: str = "beginner") -> Dict[str, Any]:
        """
        Get educational resources for trading
        
        Args:
            skill_level: Current skill level
            
        Returns:
            Educational resources
        """
        logger.info(f"Getting educational resources for {skill_level} level")
        
        courses = self.course_library.get_courses_by_level(skill_level)
        learning_path = self.course_library.get_learning_path(skill_level)
        
        results = {
            "courses": courses,
            "learning_path": learning_path,
            "recommendations": self._generate_education_recommendations(skill_level)
        }
        
        self.analysis_results["education"] = results
        return results
    
    def _generate_education_recommendations(self, skill_level: str) -> List[str]:
        """Generate educational recommendations"""
        recommendations = []
        
        if skill_level == "beginner":
            recommendations.append("Start with risk management fundamentals")
            recommendations.append("Learn market basics before strategy development")
            recommendations.append("Practice with paper trading extensively")
        
        elif skill_level == "intermediate":
            recommendations.append("Focus on strategy development and backtesting")
            recommendations.append("Learn about market correlations and diversification")
            recommendations.append("Study tax implications of different instruments")
        
        elif skill_level == "advanced":
            recommendations.append("Develop algorithmic trading systems")
            recommendations.append("Optimize for tax efficiency and execution quality")
            recommendations.append("Consider mentorship or advanced courses")
        
        recommendations.append("Continuous learning is essential for long-term success")
        recommendations.append("Join trading communities for knowledge sharing")
        
        return recommendations
    
    def generate_comprehensive_report(self, output_path: str = "comprehensive_analysis.html") -> str:
        """
        Generate comprehensive analysis report
        
        Args:
            output_path: Output file path
            
        Returns:
            Report file path
        """
        logger.info("Generating comprehensive analysis report")
        
        report_content = self._generate_report_content()
        
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Report saved to {output_path}")
        return output_path
    
    def _generate_report_content(self) -> str:
        """Generate HTML report content"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prop Trading Analysis & Strategy Platform Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background: #e8f4f8; border-radius: 3px; }
                .warning { color: #ff6b35; }
                .success { color: #28a745; }
                .recommendation { background: #fff3cd; padding: 10px; border-radius: 3px; margin: 5px 0; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Prop Trading Analysis & Strategy Platform Report</h1>
                <p>Comprehensive analysis for independent trading success</p>
                <p><strong>Education, proper risk management, and sufficient capital are the keys to success</strong></p>
            </div>
        """
        
        # Add prop firm analysis section
        if "prop_firms" in self.analysis_results:
            prop_data = self.analysis_results["prop_firms"]
            html += '<div class="section"><h2>Prop Firm Analysis</h2>'
            
            if "industry_stats" in prop_data:
                stats = prop_data["industry_stats"]
                html += f'''
                <div class="metric"><strong>Firms Analyzed:</strong> {stats['total_firms']}</div>
                <div class="metric"><strong>Avg Success Rate:</strong> {stats['avg_success_rate']:.1%}</div>
                <div class="metric"><strong>Avg First Year Cost:</strong> ${stats['avg_total_cost']:,.0f}</div>
                <div class="metric"><strong>Avg Profit Split:</strong> {stats['avg_profit_split']:.0%}</div>
                '''
            
            if "recommendations" in prop_data:
                html += '<h3>Key Recommendations</h3>'
                for rec in prop_data["recommendations"]:
                    html += f'<div class="recommendation">• {rec}</div>'
            
            html += '</div>'
        
        # Add capital planning section
        if "capital_plan" in self.analysis_results:
            cap_data = self.analysis_results["capital_plan"]
            plan = cap_data["plan"]
            html += '<div class="section"><h2>Capital Planning</h2>'
            html += f'''
            <div class="metric"><strong>Target Income:</strong> ${plan.target_income:,.0f}</div>
            <div class="metric"><strong>Required Capital:</strong> ${plan.required_capital:,.0f}</div>
            <div class="metric"><strong>Expected Return:</strong> {plan.expected_return:.1%}</div>
            <div class="metric"><strong>Risk Per Trade:</strong> {plan.risk_per_trade:.1%}</div>
            <div class="metric"><strong>Breakeven Months:</strong> {plan.breakeven_months}</div>
            '''
            
            if "recommendations" in cap_data:
                html += '<h3>Capital Recommendations</h3>'
                for rec in cap_data["recommendations"]:
                    html += f'<div class="recommendation">• {rec}</div>'
            
            html += '</div>'
        
        # Add tax analysis section
        if "tax_analysis" in self.analysis_results:
            tax_data = self.analysis_results["tax_analysis"]
            comparison = tax_data["comparison"]
            html += '<div class="section"><h2>Tax Analysis</h2>'
            html += f'''
            <div class="metric"><strong>Futures Tax:</strong> ${comparison.futures_tax.total_tax:,.0f}</div>
            <div class="metric"><strong>Stocks Tax:</strong> ${comparison.stocks_tax.total_tax:,.0f}</div>
            <div class="metric"><strong>Crypto Tax:</strong> ${comparison.crypto_tax.total_tax:,.0f}</div>
            <div class="metric success"><strong>Futures Savings vs Stocks:</strong> ${comparison.savings_vs_stocks:,.0f}</div>
            <div class="metric success"><strong>Futures Savings vs Crypto:</strong> ${comparison.savings_vs_crypto:,.0f}</div>
            '''
            
            if "recommendations" in tax_data:
                html += '<h3>Tax Recommendations</h3>'
                for rec in tax_data["recommendations"]:
                    html += f'<div class="recommendation">• {rec}</div>'
            
            html += '</div>'
        
        # Add market analysis section
        if "market_analysis" in self.analysis_results:
            market_data = self.analysis_results["market_analysis"]
            html += '<div class="section"><h2>Market Analysis</h2>'
            
            if "futures" in market_data:
                html += '<h3>Futures Markets</h3>'
                html += '<div class="recommendation">• Leverage, tax benefits, and diversification</div>'
                html += '<div class="recommendation">• Agricultural and energy markets for uncorrelated returns</div>'
            
            if "crypto" in market_data:
                html += '<h3>Cryptocurrency Markets</h3>'
                html += '<div class="recommendation">• 24/7 trading and easy automation</div>'
                html += '<div class="recommendation">• High volatility requires strict risk management</div>'
            
            if "recommendations" in market_data:
                html += '<h3>Market Recommendations</h3>'
                for rec in market_data["recommendations"]:
                    html += f'<div class="recommendation">• {rec}</div>'
            
            html += '</div>'
        
        # Add education section
        if "education" in self.analysis_results:
            edu_data = self.analysis_results["education"]
            html += '<div class="section"><h2>Educational Resources</h2>'
            
            if "recommendations" in edu_data:
                html += '<h3>Learning Recommendations</h3>'
                for rec in edu_data["recommendations"]:
                    html += f'<div class="recommendation">• {rec}</div>'
            
            html += '</div>'
        
        # Add final recommendations
        html += '''
        <div class="section">
            <h2>Final Recommendations</h2>
            <div class="recommendation">• Focus on education before risking real capital</div>
            <div class="recommendation">• Implement proper risk management from day one</div>
            <div class="recommendation">• Build sufficient capital for sustainable trading</div>
            <div class="recommendation">• Consider futures for tax benefits and leverage</div>
            <div class="recommendation">• Diversify across multiple markets</div>
            <div class="recommendation">• Use prop firms for education, not as primary income</div>
            <div class="recommendation">• Join trading communities for knowledge sharing</div>
            <div class="recommendation">• Continuous learning is essential for long-term success</div>
        </div>
        '''
        
        html += """
        </body>
        </html>
        """
        
        return html


def main():
    """Main entry point with example usage"""
    print("🏢 Prop Trading Analysis & Strategy Platform")
    print("Comprehensive analysis for independent trading success")
    print("Education, proper risk management, and sufficient capital are the keys\n")
    
    # Initialize platform
    platform = PropTradingAnalysisPlatform()
    
    try:
        # Analyze prop firms
        print("📊 Analyzing prop trading firms...")
        prop_analysis = platform.analyze_prop_firms()
        print(f"✅ Analyzed {prop_analysis['industry_stats']['total_firms']} firms")
        print(f"   Average success rate: {prop_analysis['industry_stats']['avg_success_rate']:.1%}")
        print(f"   Average first-year cost: ${prop_analysis['industry_stats']['avg_total_cost']:,.0f}\n")
        
        # Create capital plan
        print("💰 Creating capital plan...")
        capital_plan = platform.create_capital_plan(
            target_income=50000,
            risk_tolerance="moderate",
            markets=["futures", "crypto"],
            available_capital=25000
        )
        plan = capital_plan["plan"]
        print(f"✅ Capital plan created")
        print(f"   Required capital: ${plan.required_capital:,.0f}")
        print(f"   Expected return: {plan.expected_return:.1%}")
        print(f"   Breakeven months: {plan.breakeven_months}\n")
        
        # Analyze tax benefits
        print("📈 Analyzing tax benefits...")
        tax_analysis = platform.analyze_tax_benefits(
            annual_profit=50000,
            federal_rate=0.24,
            state="TX"
        )
        comparison = tax_analysis["comparison"]
        print(f"✅ Tax analysis completed")
        print(f"   Futures savings vs stocks: ${comparison.savings_vs_stocks:,.0f}")
        print(f"   Futures savings vs crypto: ${comparison.savings_vs_crypto:,.0f}\n")
        
        # Analyze market opportunities
        print("🌍 Analyzing market opportunities...")
        market_analysis = platform.analyze_market_opportunities()
        print("✅ Market analysis completed\n")
        
        # Get educational resources
        print("📚 Getting educational resources...")
        education = platform.get_educational_resources("beginner")
        print("✅ Educational resources retrieved\n")
        
        # Generate comprehensive report
        print("📋 Generating comprehensive report...")
        report_path = platform.generate_comprehensive_report()
        print(f"✅ Report saved to {report_path}\n")
        
        # Show key insights
        print("🎯 Key Insights:")
        print("• Most modern prop firms are fee-collecting simulators")
        print("• Success rates are typically 15-25%")
        print("• Futures offer significant tax advantages")
        print("• Proper risk management is essential")
        print("• Education should come before live trading")
        print("• Independent trading with sufficient capital is often better")
        
        print("\n🚀 Platform ready for comprehensive analysis!")
        print("Use platform methods to analyze specific scenarios")
        print("Check the generated report for detailed recommendations")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 