"""
App Store Analyzer

Analyzes app store categories to identify business opportunities
and AI enhancement possibilities.
"""

import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class AppAnalysis:
    """Data class for app analysis results"""
    name: str
    category: str
    rating: float
    reviews: int
    price: str
    description: str
    ai_opportunities: List[str]
    market_potential: str
    competition_level: str
    development_complexity: str

class AppStoreAnalyzer:
    """
    Analyzes app store categories to identify business opportunities
    """
    
    def __init__(self, config_path: str = "config/app_store_categories.json"):
        """
        Initialize the analyzer with configuration
        
        Args:
            config_path: Path to the app store categories configuration
        """
        self.config_path = config_path
        self.categories_data = self._load_categories()
        
    def _load_categories(self) -> Dict[str, Any]:
        """Load app store categories configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {self.config_path} not found. Using default categories.")
            return self._get_default_categories()
    
    def _get_default_categories(self) -> Dict[str, Any]:
        """Get default categories if config file is not found"""
        return {
            "categories": {
                "Business": {
                    "description": "Business and productivity applications",
                    "market_size": "Large",
                    "competition_level": "High",
                    "ai_opportunities": ["Automated reporting", "Smart scheduling"],
                    "revenue_potential": "High"
                },
                "Education": {
                    "description": "Educational and learning applications", 
                    "market_size": "Large",
                    "competition_level": "Medium",
                    "ai_opportunities": ["Personalized learning", "AI tutors"],
                    "revenue_potential": "Medium"
                }
            }
        }
    
    def get_categories(self) -> List[str]:
        """Get list of available app store categories"""
        return list(self.categories_data.get("categories", {}).keys())
    
    def analyze_category(self, category: str) -> Dict[str, Any]:
        """
        Analyze a specific app store category
        
        Args:
            category: Name of the category to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if category not in self.categories_data.get("categories", {}):
            return {"error": f"Category '{category}' not found"}
        
        category_data = self.categories_data["categories"][category]
        
        analysis = {
            "category": category,
            "description": category_data.get("description", ""),
            "market_size": category_data.get("market_size", "Unknown"),
            "competition_level": category_data.get("competition_level", "Unknown"),
            "revenue_potential": category_data.get("revenue_potential", "Unknown"),
            "development_complexity": category_data.get("development_complexity", "Unknown"),
            "ai_opportunities": category_data.get("ai_opportunities", []),
            "successful_examples": category_data.get("successful_examples", []),
            "analysis_timestamp": datetime.now().isoformat(),
            "recommendations": self._generate_recommendations(category_data)
        }
        
        return analysis
    
    def _generate_recommendations(self, category_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on category analysis"""
        recommendations = []
        
        market_size = category_data.get("market_size", "")
        competition = category_data.get("competition_level", "")
        revenue_potential = category_data.get("revenue_potential", "")
        
        if market_size in ["Large", "Very Large"]:
            recommendations.append("High market potential - worth exploring")
        
        if competition in ["Medium", "Low"]:
            recommendations.append("Lower competition - easier to enter market")
        
        if revenue_potential in ["High", "Very High"]:
            recommendations.append("Strong revenue potential")
        
        ai_opportunities = category_data.get("ai_opportunities", [])
        if ai_opportunities:
            recommendations.append(f"Multiple AI enhancement opportunities: {', '.join(ai_opportunities[:3])}")
        
        return recommendations
    
    def find_best_opportunities(self, max_categories: int = 5) -> List[Dict[str, Any]]:
        """
        Find the best business opportunities across all categories
        
        Args:
            max_categories: Maximum number of categories to return
            
        Returns:
            List of top opportunities sorted by potential
        """
        opportunities = []
        
        for category_name, category_data in self.categories_data.get("categories", {}).items():
            score = self._calculate_opportunity_score(category_data)
            
            opportunity = {
                "category": category_name,
                "score": score,
                "market_size": category_data.get("market_size", ""),
                "competition_level": category_data.get("competition_level", ""),
                "revenue_potential": category_data.get("revenue_potential", ""),
                "ai_opportunities": category_data.get("ai_opportunities", [])
            }
            opportunities.append(opportunity)
        
        # Sort by score (highest first) and return top results
        opportunities.sort(key=lambda x: x["score"], reverse=True)
        return opportunities[:max_categories]
    
    def _calculate_opportunity_score(self, category_data: Dict[str, Any]) -> float:
        """Calculate opportunity score based on various factors"""
        weights = self.categories_data.get("analysis_parameters", {}).get("market_size_weights", {})
        
        market_size = category_data.get("market_size", "")
        competition = category_data.get("competition_level", "")
        revenue_potential = category_data.get("revenue_potential", "")
        development_complexity = category_data.get("development_complexity", "")
        
        # Calculate weighted score
        market_score = weights.get(market_size, 1)
        competition_score = 6 - weights.get(competition, 3)  # Invert competition (lower is better)
        revenue_score = weights.get(revenue_potential, 1)
        complexity_score = 6 - weights.get(development_complexity, 3)  # Invert complexity
        
        # Weighted average
        total_score = (market_score * 0.3 + 
                      competition_score * 0.25 + 
                      revenue_score * 0.3 + 
                      complexity_score * 0.15)
        
        return round(total_score, 2)
    
    def get_ai_enhancement_ideas(self, category: str) -> List[str]:
        """
        Get AI enhancement ideas for a specific category
        
        Args:
            category: Category to get ideas for
            
        Returns:
            List of AI enhancement ideas
        """
        if category not in self.categories_data.get("categories", {}):
            return []
        
        category_data = self.categories_data["categories"][category]
        base_opportunities = category_data.get("ai_opportunities", [])
        
        # Add generic AI enhancement ideas
        generic_ideas = [
            "AI-powered personalization",
            "Smart automation features",
            "Predictive analytics",
            "Intelligent recommendations",
            "Automated content generation"
        ]
        
        return base_opportunities + generic_ideas
    
    def export_analysis(self, category: str, output_path: str) -> bool:
        """
        Export analysis results to a file
        
        Args:
            category: Category to analyze
            output_path: Path to save the analysis
            
        Returns:
            True if successful, False otherwise
        """
        try:
            analysis = self.analyze_category(category)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error exporting analysis: {e}")
            return False
    
    def generate_business_ideas(self, category: str, count: int = 5) -> List[Dict[str, str]]:
        """
        Generate specific business ideas for a category
        
        Args:
            category: Category to generate ideas for
            count: Number of ideas to generate
            
        Returns:
            List of business ideas with descriptions
        """
        if category not in self.categories_data.get("categories", {}):
            return []
        
        category_data = self.categories_data["categories"][category]
        ai_opportunities = category_data.get("ai_opportunities", [])
        
        ideas = []
        for i in range(min(count, len(ai_opportunities))):
            opportunity = ai_opportunities[i]
            idea = {
                "title": f"AI-Enhanced {category} App",
                "description": f"Build a {category.lower()} app with {opportunity.lower()} capabilities",
                "ai_feature": opportunity,
                "target_audience": f"{category} app users",
                "revenue_model": "Freemium with premium AI features"
            }
            ideas.append(idea)
        
        return ideas 