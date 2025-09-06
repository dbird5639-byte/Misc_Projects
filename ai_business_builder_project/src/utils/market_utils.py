"""
Market Utils Module

Provides utility functions for market analysis and business operations.
"""

from typing import Dict, List, Any
from datetime import datetime

class MarketUtils:
    """
    Utility class for market analysis and business operations
    """
    
    def __init__(self):
        """Initialize market utilities"""
        pass
    
    def calculate_market_score(self, market_size: str, competition: str, growth: str) -> float:
        """Calculate market opportunity score"""
        size_scores = {"Very Large": 5, "Large": 4, "Medium": 3, "Small": 2}
        comp_scores = {"Very High": 1, "High": 2, "Medium": 3, "Low": 4}
        growth_scores = {"High": 5, "Medium": 3, "Low": 2}
        
        score = (size_scores.get(market_size, 2) * 0.4 + 
                comp_scores.get(competition, 2) * 0.3 + 
                growth_scores.get(growth, 2) * 0.3)
        
        return round(score, 2)
    
    def format_currency(self, amount: float) -> str:
        """Format currency amounts"""
        return f"${amount:,.2f}"
    
    def calculate_growth_rate(self, current: float, previous: float) -> float:
        """Calculate growth rate percentage"""
        if previous == 0:
            return 0
        return round(((current - previous) / previous) * 100, 2) 