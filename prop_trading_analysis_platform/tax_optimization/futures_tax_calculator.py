"""
Futures Tax Calculator
Calculate tax benefits of futures trading using the 60/40 rule

Based on insights from experienced traders about the significant
tax advantages of futures trading over other instruments.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TaxBracket(Enum):
    """Federal tax brackets for 2024"""
    BRACKET_10 = 0.10
    BRACKET_12 = 0.12
    BRACKET_22 = 0.22
    BRACKET_24 = 0.24
    BRACKET_32 = 0.32
    BRACKET_35 = 0.35
    BRACKET_37 = 0.37


class InstrumentType(Enum):
    """Types of trading instruments"""
    FUTURES = "futures"
    STOCKS = "stocks"
    CRYPTO = "crypto"
    FOREX = "forex"
    OPTIONS = "options"


@dataclass
class TaxAnalysis:
    """Tax analysis results"""
    instrument: InstrumentType
    profit: float
    holding_period: int  # days
    tax_rate: float
    total_tax: float
    effective_rate: float
    tax_savings: float
    after_tax_profit: float
    tax_efficiency: float  # percentage of profit retained


@dataclass
class ComparisonResult:
    """Comparison of tax implications across instruments"""
    futures_tax: TaxAnalysis
    stocks_tax: TaxAnalysis
    crypto_tax: TaxAnalysis
    savings_vs_stocks: float
    savings_vs_crypto: float
    recommendation: str


class FuturesTaxCalculator:
    """
    Futures tax calculator implementing the 60/40 rule
    
    Calculates tax benefits of futures trading compared to other
    instruments based on the 60/40 rule and other tax advantages.
    """
    
    def __init__(self):
        """Initialize the tax calculator"""
        self.tax_brackets = self._load_tax_brackets()
        self.state_tax_rates = self._load_state_tax_rates()
        
        logger.info("Futures Tax Calculator initialized")
    
    def _load_tax_brackets(self) -> Dict[str, Dict[str, float]]:
        """Load federal tax brackets for 2024"""
        return {
            "single": {
                "10": (0, 11600),
                "12": (11601, 47150),
                "22": (47151, 100525),
                "24": (100526, 191950),
                "32": (191951, 243725),
                "35": (243726, 609350),
                "37": (609351, float('inf'))
            },
            "married": {
                "10": (0, 23200),
                "12": (23201, 94300),
                "22": (94301, 201050),
                "24": (201051, 383900),
                "32": (383901, 487450),
                "35": (487451, 731200),
                "37": (731201, float('inf'))
            }
        }
    
    def _load_state_tax_rates(self) -> Dict[str, float]:
        """Load state tax rates (simplified)"""
        return {
            "CA": 0.133,  # California
            "NY": 0.109,  # New York
            "TX": 0.000,  # Texas (no state income tax)
            "FL": 0.000,  # Florida (no state income tax)
            "IL": 0.049,  # Illinois
            "PA": 0.030,  # Pennsylvania
            "OH": 0.039,  # Ohio
            "GA": 0.059,  # Georgia
            "NC": 0.049,  # North Carolina
            "MI": 0.042,  # Michigan
            "NJ": 0.107,  # New Jersey
            "VA": 0.057,  # Virginia
            "WA": 0.000,  # Washington (no state income tax)
            "NV": 0.000,  # Nevada (no state income tax)
            "SD": 0.000,  # South Dakota (no state income tax)
            "WY": 0.000,  # Wyoming (no state income tax)
            "TN": 0.000,  # Tennessee (no state income tax)
            "NH": 0.000   # New Hampshire (no state income tax)
        }
    
    def analyze_trade(
        self,
        profit: float,
        holding_period: int,
        tax_rate: float,
        state: str = "TX",
        filing_status: str = "single"
    ) -> TaxAnalysis:
        """
        Analyze tax implications for a single trade
        
        Args:
            profit: Trade profit/loss
            holding_period: Holding period in days
            tax_rate: Marginal tax rate
            state: State of residence
            filing_status: Filing status (single/married)
            
        Returns:
            TaxAnalysis with detailed breakdown
        """
        
        if profit <= 0:
            return TaxAnalysis(
                instrument=InstrumentType.FUTURES,
                profit=profit,
                holding_period=holding_period,
                tax_rate=tax_rate,
                total_tax=0,
                effective_rate=0,
                tax_savings=0,
                after_tax_profit=profit,
                tax_efficiency=1.0
            )
        
        # Calculate futures tax using 60/40 rule
        futures_tax = self._calculate_futures_tax(profit, tax_rate, state)
        
        # Calculate effective tax rate
        effective_rate = futures_tax / profit
        
        # Calculate tax efficiency (percentage of profit retained)
        tax_efficiency = 1 - effective_rate
        
        return TaxAnalysis(
            instrument=InstrumentType.FUTURES,
            profit=profit,
            holding_period=holding_period,
            tax_rate=tax_rate,
            total_tax=futures_tax,
            effective_rate=effective_rate,
            tax_savings=0,  # Will be calculated in comparison
            after_tax_profit=profit - futures_tax,
            tax_efficiency=tax_efficiency
        )
    
    def _calculate_futures_tax(
        self,
        profit: float,
        federal_rate: float,
        state: str
    ) -> float:
        """Calculate futures tax using 60/40 rule"""
        
        # 60/40 rule: 60% long-term, 40% short-term
        long_term_portion = profit * 0.60
        short_term_portion = profit * 0.40
        
        # Federal tax calculation
        # Long-term capital gains rates are typically lower
        long_term_rate = min(federal_rate, 0.20)  # Cap at 20% for long-term
        short_term_rate = federal_rate
        
        federal_tax = (long_term_portion * long_term_rate) + (short_term_portion * short_term_rate)
        
        # State tax (if applicable)
        state_rate = self.state_tax_rates.get(state, 0.0)
        state_tax = profit * state_rate
        
        total_tax = federal_tax + state_tax
        
        return total_tax
    
    def compare_instruments(
        self,
        profit: float,
        holding_period: int,
        federal_rate: float,
        state: str = "TX"
    ) -> ComparisonResult:
        """
        Compare tax implications across different instruments
        
        Args:
            profit: Trade profit
            holding_period: Holding period in days
            federal_rate: Federal tax rate
            state: State of residence
            
        Returns:
            ComparisonResult with detailed analysis
        """
        
        # Calculate futures tax
        futures_tax = self.analyze_trade(profit, holding_period, federal_rate, state)
        
        # Calculate stocks tax
        stocks_tax = self._calculate_stocks_tax(profit, holding_period, federal_rate, state)
        
        # Calculate crypto tax
        crypto_tax = self._calculate_crypto_tax(profit, holding_period, federal_rate, state)
        
        # Calculate savings
        savings_vs_stocks = stocks_tax.total_tax - futures_tax.total_tax
        savings_vs_crypto = crypto_tax.total_tax - futures_tax.total_tax
        
        # Generate recommendation
        recommendation = self._generate_tax_recommendation(
            futures_tax, stocks_tax, crypto_tax, savings_vs_stocks, savings_vs_crypto
        )
        
        return ComparisonResult(
            futures_tax=futures_tax,
            stocks_tax=stocks_tax,
            crypto_tax=crypto_tax,
            savings_vs_stocks=savings_vs_stocks,
            savings_vs_crypto=savings_vs_crypto,
            recommendation=recommendation
        )
    
    def _calculate_stocks_tax(
        self,
        profit: float,
        holding_period: int,
        federal_rate: float,
        state: str
    ) -> TaxAnalysis:
        """Calculate stocks tax based on holding period"""
        
        if profit <= 0:
            return TaxAnalysis(
                instrument=InstrumentType.STOCKS,
                profit=profit,
                holding_period=holding_period,
                tax_rate=federal_rate,
                total_tax=0,
                effective_rate=0,
                tax_savings=0,
                after_tax_profit=profit,
                tax_efficiency=1.0
            )
        
        # Determine if long-term or short-term
        if holding_period >= 365:
            # Long-term capital gains
            long_term_rate = min(federal_rate, 0.20)  # Cap at 20%
            federal_tax = profit * long_term_rate
        else:
            # Short-term capital gains (ordinary income)
            federal_tax = profit * federal_rate
        
        # State tax
        state_rate = self.state_tax_rates.get(state, 0.0)
        state_tax = profit * state_rate
        
        total_tax = federal_tax + state_tax
        effective_rate = total_tax / profit
        tax_efficiency = 1 - effective_rate
        
        return TaxAnalysis(
            instrument=InstrumentType.STOCKS,
            profit=profit,
            holding_period=holding_period,
            tax_rate=federal_rate,
            total_tax=total_tax,
            effective_rate=effective_rate,
            tax_savings=0,
            after_tax_profit=profit - total_tax,
            tax_efficiency=tax_efficiency
        )
    
    def _calculate_crypto_tax(
        self,
        profit: float,
        holding_period: int,
        federal_rate: float,
        state: str
    ) -> TaxAnalysis:
        """Calculate crypto tax (treated as property)"""
        
        if profit <= 0:
            return TaxAnalysis(
                instrument=InstrumentType.CRYPTO,
                profit=profit,
                holding_period=holding_period,
                tax_rate=federal_rate,
                total_tax=0,
                effective_rate=0,
                tax_savings=0,
                after_tax_profit=profit,
                tax_efficiency=1.0
            )
        
        # Crypto is treated as property, similar to stocks
        if holding_period >= 365:
            # Long-term capital gains
            long_term_rate = min(federal_rate, 0.20)
            federal_tax = profit * long_term_rate
        else:
            # Short-term capital gains
            federal_tax = profit * federal_rate
        
        # State tax
        state_rate = self.state_tax_rates.get(state, 0.0)
        state_tax = profit * state_rate
        
        total_tax = federal_tax + state_tax
        effective_rate = total_tax / profit
        tax_efficiency = 1 - effective_rate
        
        return TaxAnalysis(
            instrument=InstrumentType.CRYPTO,
            profit=profit,
            holding_period=holding_period,
            tax_rate=federal_rate,
            total_tax=total_tax,
            effective_rate=effective_rate,
            tax_savings=0,
            after_tax_profit=profit - total_tax,
            tax_efficiency=tax_efficiency
        )
    
    def _generate_tax_recommendation(
        self,
        futures_tax: TaxAnalysis,
        stocks_tax: TaxAnalysis,
        crypto_tax: TaxAnalysis,
        savings_vs_stocks: float,
        savings_vs_crypto: float
    ) -> str:
        """Generate tax recommendation"""
        
        if savings_vs_stocks > 0 and savings_vs_crypto > 0:
            return "Futures trading provides significant tax advantages over both stocks and crypto"
        elif savings_vs_stocks > 0:
            return "Futures trading provides tax advantages over stocks, similar to crypto"
        elif savings_vs_crypto > 0:
            return "Futures trading provides tax advantages over crypto, similar to stocks"
        else:
            return "Tax implications are similar across instruments"
    
    def calculate_annual_tax_impact(
        self,
        annual_profit: float,
        federal_rate: float,
        state: str = "TX",
        instrument: InstrumentType = InstrumentType.FUTURES
    ) -> Dict[str, Any]:
        """
        Calculate annual tax impact for different instruments
        
        Args:
            annual_profit: Annual trading profit
            federal_rate: Federal tax rate
            state: State of residence
            instrument: Trading instrument
            
        Returns:
            Annual tax analysis
        """
        
        if instrument == InstrumentType.FUTURES:
            tax_analysis = self.analyze_trade(annual_profit, 365, federal_rate, state)
        elif instrument == InstrumentType.STOCKS:
            tax_analysis = self._calculate_stocks_tax(annual_profit, 365, federal_rate, state)
        elif instrument == InstrumentType.CRYPTO:
            tax_analysis = self._calculate_crypto_tax(annual_profit, 365, federal_rate, state)
        else:
            raise ValueError(f"Unsupported instrument: {instrument}")
        
        return {
            "instrument": instrument.value,
            "annual_profit": annual_profit,
            "total_tax": tax_analysis.total_tax,
            "effective_rate": tax_analysis.effective_rate,
            "after_tax_profit": tax_analysis.after_tax_profit,
            "tax_efficiency": tax_analysis.tax_efficiency,
            "monthly_tax": tax_analysis.total_tax / 12,
            "monthly_after_tax": tax_analysis.after_tax_profit / 12
        }
    
    def compare_states(
        self,
        profit: float,
        federal_rate: float,
        states: List[str]
    ) -> pd.DataFrame:
        """
        Compare tax implications across different states
        
        Args:
            profit: Trading profit
            federal_rate: Federal tax rate
            states: List of states to compare
            
        Returns:
            Comparison DataFrame
        """
        
        results = []
        
        for state in states:
            # Calculate futures tax for each state
            futures_tax = self.analyze_trade(profit, 365, federal_rate, state)
            
            # Calculate stocks tax for each state
            stocks_tax = self._calculate_stocks_tax(profit, 365, federal_rate, state)
            
            # Calculate crypto tax for each state
            crypto_tax = self._calculate_crypto_tax(profit, 365, federal_rate, state)
            
            results.append({
                "State": state,
                "State_Tax_Rate": f"{self.state_tax_rates.get(state, 0):.1%}",
                "Futures_Tax": f"${futures_tax.total_tax:,.0f}",
                "Stocks_Tax": f"${stocks_tax.total_tax:,.0f}",
                "Crypto_Tax": f"${crypto_tax.total_tax:,.0f}",
                "Futures_Savings_vs_Stocks": f"${stocks_tax.total_tax - futures_tax.total_tax:,.0f}",
                "Futures_Savings_vs_Crypto": f"${crypto_tax.total_tax - futures_tax.total_tax:,.0f}",
                "Futures_Tax_Efficiency": f"{futures_tax.tax_efficiency:.1%}"
            })
        
        return pd.DataFrame(results)
    
    def generate_tax_report(
        self,
        profit: float,
        holding_period: int,
        federal_rate: float,
        state: str = "TX"
    ) -> str:
        """
        Generate comprehensive tax report
        
        Args:
            profit: Trading profit
            holding_period: Holding period in days
            federal_rate: Federal tax rate
            state: State of residence
            
        Returns:
            HTML report content
        """
        
        comparison = self.compare_instruments(profit, holding_period, federal_rate, state)
        
        html = f"""
        <div class="tax-report">
            <h2>Tax Analysis Report</h2>
            
            <div class="overview">
                <h3>Overview</h3>
                <p><strong>Profit:</strong> ${profit:,.0f}</p>
                <p><strong>Holding Period:</strong> {holding_period} days</p>
                <p><strong>Federal Tax Rate:</strong> {federal_rate:.1%}</p>
                <p><strong>State:</strong> {state}</p>
            </div>
            
            <div class="comparison">
                <h3>Tax Comparison</h3>
                <table>
                    <tr>
                        <th>Instrument</th>
                        <th>Total Tax</th>
                        <th>Effective Rate</th>
                        <th>After-Tax Profit</th>
                        <th>Tax Efficiency</th>
                    </tr>
                    <tr>
                        <td>Futures</td>
                        <td>${comparison.futures_tax.total_tax:,.0f}</td>
                        <td>{comparison.futures_tax.effective_rate:.1%}</td>
                        <td>${comparison.futures_tax.after_tax_profit:,.0f}</td>
                        <td>{comparison.futures_tax.tax_efficiency:.1%}</td>
                    </tr>
                    <tr>
                        <td>Stocks</td>
                        <td>${comparison.stocks_tax.total_tax:,.0f}</td>
                        <td>{comparison.stocks_tax.effective_rate:.1%}</td>
                        <td>${comparison.stocks_tax.after_tax_profit:,.0f}</td>
                        <td>{comparison.stocks_tax.tax_efficiency:.1%}</td>
                    </tr>
                    <tr>
                        <td>Crypto</td>
                        <td>${comparison.crypto_tax.total_tax:,.0f}</td>
                        <td>{comparison.crypto_tax.effective_rate:.1%}</td>
                        <td>${comparison.crypto_tax.after_tax_profit:,.0f}</td>
                        <td>{comparison.crypto_tax.tax_efficiency:.1%}</td>
                    </tr>
                </table>
            </div>
            
            <div class="savings">
                <h3>Tax Savings</h3>
                <p><strong>Futures vs Stocks:</strong> ${comparison.savings_vs_stocks:,.0f}</p>
                <p><strong>Futures vs Crypto:</strong> ${comparison.savings_vs_crypto:,.0f}</p>
            </div>
            
            <div class="recommendation">
                <h3>Recommendation</h3>
                <p>{comparison.recommendation}</p>
            </div>
            
            <div class="notes">
                <h3>Important Notes</h3>
                <ul>
                    <li>Futures use the 60/40 rule: 60% long-term, 40% short-term</li>
                    <li>Stocks depend on holding period for long-term vs short-term</li>
                    <li>Crypto is treated as property similar to stocks</li>
                    <li>State taxes vary significantly by location</li>
                    <li>Consult a tax professional for specific advice</li>
                </ul>
            </div>
        </div>
        """
        
        return html


# Example usage
if __name__ == "__main__":
    # Initialize calculator
    calculator = FuturesTaxCalculator()
    
    # Analyze single trade
    analysis = calculator.analyze_trade(
        profit=10000,
        holding_period=30,
        tax_rate=0.24,
        state="TX"
    )
    
    print("Single Trade Analysis:")
    print(f"Profit: ${analysis.profit:,.0f}")
    print(f"Total Tax: ${analysis.total_tax:,.0f}")
    print(f"Effective Rate: {analysis.effective_rate:.1%}")
    print(f"After-Tax Profit: ${analysis.after_tax_profit:,.0f}")
    print(f"Tax Efficiency: {analysis.tax_efficiency:.1%}")
    
    # Compare instruments
    comparison = calculator.compare_instruments(
        profit=10000,
        holding_period=30,
        federal_rate=0.24,
        state="TX"
    )
    
    print(f"\nInstrument Comparison:")
    print(f"Futures Tax: ${comparison.futures_tax.total_tax:,.0f}")
    print(f"Stocks Tax: ${comparison.stocks_tax.total_tax:,.0f}")
    print(f"Crypto Tax: ${comparison.crypto_tax.total_tax:,.0f}")
    print(f"Savings vs Stocks: ${comparison.savings_vs_stocks:,.0f}")
    print(f"Savings vs Crypto: ${comparison.savings_vs_crypto:,.0f}")
    print(f"Recommendation: {comparison.recommendation}")
    
    # Annual impact
    annual_impact = calculator.calculate_annual_tax_impact(
        annual_profit=50000,
        federal_rate=0.24,
        state="TX",
        instrument=InstrumentType.FUTURES
    )
    
    print(f"\nAnnual Impact (Futures):")
    print(f"Annual Profit: ${annual_impact['annual_profit']:,.0f}")
    print(f"Total Tax: ${annual_impact['total_tax']:,.0f}")
    print(f"After-Tax Profit: ${annual_impact['after_tax_profit']:,.0f}")
    print(f"Monthly After-Tax: ${annual_impact['monthly_after_tax']:,.0f}")
    
    # State comparison
    states = ["TX", "CA", "NY", "FL", "IL"]
    state_comparison = calculator.compare_states(50000, 0.24, states)
    print(f"\nState Comparison:")
    print(state_comparison) 