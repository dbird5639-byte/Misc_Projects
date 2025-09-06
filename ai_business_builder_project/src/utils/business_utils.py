"""
Business Utils Module

Provides utility functions for business operations and calculations.
"""

from typing import Dict, List, Any
from datetime import datetime

class BusinessUtils:
    """
    Utility class for business operations and calculations
    """
    
    def __init__(self):
        """Initialize business utilities"""
        pass
    
    def calculate_roi(self, investment: float, returns: float) -> float:
        """Calculate Return on Investment"""
        if investment == 0:
            return 0
        return round(((returns - investment) / investment) * 100, 2)
    
    def calculate_customer_lifetime_value(self, avg_purchase: float, 
                                        purchase_frequency: float, 
                                        customer_lifespan: float) -> float:
        """Calculate Customer Lifetime Value"""
        return round(avg_purchase * purchase_frequency * customer_lifespan, 2)
    
    def calculate_churn_rate(self, customers_lost: int, total_customers: int) -> float:
        """Calculate customer churn rate"""
        if total_customers == 0:
            return 0
        return round((customers_lost / total_customers) * 100, 2)
    
    def estimate_break_even(self, fixed_costs: float, unit_price: float, 
                           unit_cost: float) -> int:
        """Calculate break-even point in units"""
        if unit_price - unit_cost <= 0:
            return 0
        return round(fixed_costs / (unit_price - unit_cost))
    
    def format_percentage(self, value: float) -> str:
        """Format percentage values"""
        return f"{value:.1f}%" 