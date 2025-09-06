"""
Market Research Module

Conducts comprehensive market research and validation for business ideas.
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class MarketData:
    """Data class for market research results"""
    market_size: str
    growth_rate: str
    competition_level: str
    target_audience: str
    revenue_potential: str
    entry_barriers: List[str]
    opportunities: List[str]
    risks: List[str]

class MarketResearch:
    """
    Conducts market research and validation for business ideas
    """
    
    def __init__(self):
        """Initialize market research tools"""
        self.market_data = {}
        self.competitor_analysis = {}
        
    def analyze_market_size(self, category: str, target_audience: str) -> Dict[str, Any]:
        """
        Analyze market size for a business category
        
        Args:
            category: Business category to analyze
            target_audience: Target audience description
            
        Returns:
            Market size analysis results
        """
        # This would typically integrate with market research APIs
        # For now, using estimated data based on category
        
        market_sizes = {
            "Business": {"size": "Large", "value": "$50B+", "growth": "15% annually"},
            "Education": {"size": "Large", "value": "$100B+", "growth": "20% annually"},
            "Entertainment": {"size": "Very Large", "value": "$200B+", "growth": "25% annually"},
            "Finance": {"size": "Large", "value": "$150B+", "growth": "18% annually"},
            "Health & Fitness": {"size": "Large", "value": "$80B+", "growth": "22% annually"},
            "Lifestyle": {"size": "Medium", "value": "$30B+", "growth": "12% annually"},
            "Productivity": {"size": "Large", "value": "$60B+", "growth": "16% annually"},
            "Photo & Video": {"size": "Large", "value": "$70B+", "growth": "19% annually"},
            "Shopping": {"size": "Very Large", "value": "$300B+", "growth": "28% annually"},
            "Social Networking": {"size": "Very Large", "value": "$250B+", "growth": "30% annually"},
            "Travel": {"size": "Large", "value": "$90B+", "growth": "14% annually"},
            "Utilities": {"size": "Medium", "value": "$25B+", "growth": "10% annually"}
        }
        
        market_info = market_sizes.get(category, {
            "size": "Medium", 
            "value": "$20B+", 
            "growth": "10% annually"
        })
        
        return {
            "category": category,
            "target_audience": target_audience,
            "market_size": market_info["size"],
            "market_value": market_info["value"],
            "growth_rate": market_info["growth"],
            "analysis_date": datetime.now().isoformat(),
            "confidence_level": "High" if category in market_sizes else "Medium"
        }
    
    def analyze_competition(self, category: str, specific_idea: str = "") -> Dict[str, Any]:
        """
        Analyze competition in a market category
        
        Args:
            category: Business category
            specific_idea: Specific business idea (optional)
            
        Returns:
            Competition analysis results
        """
        competition_levels = {
            "Business": {"level": "High", "major_players": 15, "startups": 100},
            "Education": {"level": "Medium", "major_players": 8, "startups": 50},
            "Entertainment": {"level": "Very High", "major_players": 25, "startups": 200},
            "Finance": {"level": "High", "major_players": 20, "startups": 150},
            "Health & Fitness": {"level": "Medium", "major_players": 10, "startups": 80},
            "Lifestyle": {"level": "Medium", "major_players": 5, "startups": 40},
            "Productivity": {"level": "High", "major_players": 12, "startups": 90},
            "Photo & Video": {"level": "High", "major_players": 18, "startups": 120},
            "Shopping": {"level": "Very High", "major_players": 30, "startups": 300},
            "Social Networking": {"level": "Very High", "major_players": 35, "startups": 400},
            "Travel": {"level": "High", "major_players": 15, "startups": 100},
            "Utilities": {"level": "Medium", "major_players": 8, "startups": 60}
        }
        
        comp_info = competition_levels.get(category, {
            "level": "Medium",
            "major_players": 10,
            "startups": 50
        })
        
        return {
            "category": category,
            "competition_level": comp_info["level"],
            "major_players": comp_info["major_players"],
            "startup_count": comp_info["startups"],
            "market_saturation": self._calculate_saturation(comp_info),
            "entry_difficulty": self._assess_entry_difficulty(comp_info["level"]),
            "differentiation_opportunities": self._find_differentiation_opportunities(category)
        }
    
    def _calculate_saturation(self, comp_info: Dict[str, Any]) -> str:
        """Calculate market saturation level"""
        total_players = comp_info["major_players"] + comp_info["startups"]
        
        if total_players > 300:
            return "Very High"
        elif total_players > 150:
            return "High"
        elif total_players > 80:
            return "Medium"
        else:
            return "Low"
    
    def _assess_entry_difficulty(self, competition_level: str) -> str:
        """Assess difficulty of entering the market"""
        difficulty_map = {
            "Very High": "Very Difficult",
            "High": "Difficult", 
            "Medium": "Moderate",
            "Low": "Easy"
        }
        return difficulty_map.get(competition_level, "Moderate")
    
    def _find_differentiation_opportunities(self, category: str) -> List[str]:
        """Find opportunities for differentiation"""
        opportunities = {
            "Business": [
                "AI-powered automation",
                "Better user experience",
                "Integration capabilities",
                "Mobile-first approach"
            ],
            "Education": [
                "Personalized learning paths",
                "AI tutors",
                "Gamification",
                "Social learning features"
            ],
            "Entertainment": [
                "AI content generation",
                "Personalized recommendations",
                "Interactive features",
                "Cross-platform sync"
            ],
            "Finance": [
                "AI financial advice",
                "Better security",
                "Simplified interface",
                "Real-time insights"
            ]
        }
        
        return opportunities.get(category, [
            "AI enhancement",
            "Better UX/UI",
            "Mobile optimization",
            "Integration features"
        ])
    
    def validate_business_idea(self, idea: str, category: str, target_audience: str) -> Dict[str, Any]:
        """
        Validate a business idea through market research
        
        Args:
            idea: Business idea description
            category: Business category
            target_audience: Target audience
            
        Returns:
            Validation results with scores and recommendations
        """
        # Analyze market size
        market_analysis = self.analyze_market_size(category, target_audience)
        
        # Analyze competition
        competition_analysis = self.analyze_competition(category, idea)
        
        # Calculate validation score
        validation_score = self._calculate_validation_score(market_analysis, competition_analysis)
        
        # Generate recommendations
        recommendations = self._generate_validation_recommendations(
            market_analysis, competition_analysis, validation_score
        )
        
        return {
            "idea": idea,
            "category": category,
            "target_audience": target_audience,
            "validation_score": validation_score,
            "market_analysis": market_analysis,
            "competition_analysis": competition_analysis,
            "recommendations": recommendations,
            "validation_date": datetime.now().isoformat(),
            "overall_verdict": self._get_verdict(validation_score)
        }
    
    def _calculate_validation_score(self, market_analysis: Dict, competition_analysis: Dict) -> float:
        """Calculate overall validation score"""
        # Market size scoring
        market_scores = {"Very Large": 5, "Large": 4, "Medium": 3, "Small": 2}
        market_score = market_scores.get(market_analysis["market_size"], 2)
        
        # Competition scoring (inverted - lower competition is better)
        competition_scores = {"Very High": 1, "High": 2, "Medium": 3, "Low": 4}
        competition_score = competition_scores.get(competition_analysis["competition_level"], 2)
        
        # Growth rate scoring
        growth_rate = market_analysis.get("growth_rate", "10% annually")
        growth_score = 3  # Default
        if "20%" in growth_rate or "25%" in growth_rate or "30%" in growth_rate:
            growth_score = 5
        elif "15%" in growth_rate or "18%" in growth_rate:
            growth_score = 4
        elif "10%" in growth_rate or "12%" in growth_rate:
            growth_score = 3
        else:
            growth_score = 2
        
        # Weighted average
        total_score = (market_score * 0.4 + competition_score * 0.3 + growth_score * 0.3)
        return round(total_score, 2)
    
    def _generate_validation_recommendations(self, market_analysis: Dict, 
                                           competition_analysis: Dict, 
                                           validation_score: float) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        if validation_score >= 4.0:
            recommendations.append("Strong market opportunity - proceed with development")
        elif validation_score >= 3.0:
            recommendations.append("Moderate opportunity - consider niche focus")
        else:
            recommendations.append("Challenging market - reconsider or pivot")
        
        if market_analysis["market_size"] in ["Large", "Very Large"]:
            recommendations.append("Large market size provides good growth potential")
        
        if competition_analysis["competition_level"] in ["Medium", "Low"]:
            recommendations.append("Lower competition makes market entry easier")
        
        if competition_analysis["entry_difficulty"] == "Very Difficult":
            recommendations.append("Consider finding a unique niche or differentiation")
        
        opportunities = competition_analysis.get("differentiation_opportunities", [])
        if opportunities:
            recommendations.append(f"Focus on: {', '.join(opportunities[:2])}")
        
        return recommendations
    
    def _get_verdict(self, validation_score: float) -> str:
        """Get overall verdict based on validation score"""
        if validation_score >= 4.0:
            return "Strongly Recommended"
        elif validation_score >= 3.0:
            return "Recommended with Caution"
        elif validation_score >= 2.0:
            return "Not Recommended"
        else:
            return "Avoid"
    
    def generate_market_report(self, category: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive market report
        
        Args:
            category: Category to analyze
            output_path: Optional path to save report
            
        Returns:
            Complete market report
        """
        market_analysis = self.analyze_market_size(category, "General users")
        competition_analysis = self.analyze_competition(category)
        
        report = {
            "category": category,
            "report_date": datetime.now().isoformat(),
            "market_analysis": market_analysis,
            "competition_analysis": competition_analysis,
            "key_insights": self._generate_key_insights(market_analysis, competition_analysis),
            "recommendations": self._generate_market_recommendations(market_analysis, competition_analysis),
            "risk_assessment": self._assess_market_risks(market_analysis, competition_analysis)
        }
        
        if output_path:
            try:
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2)
            except Exception as e:
                print(f"Error saving report: {e}")
        
        return report
    
    def _generate_key_insights(self, market_analysis: Dict, competition_analysis: Dict) -> List[str]:
        """Generate key insights from market and competition analysis"""
        insights = []
        
        insights.append(f"Market size: {market_analysis['market_size']} ({market_analysis['market_value']})")
        insights.append(f"Growth rate: {market_analysis['growth_rate']}")
        insights.append(f"Competition level: {competition_analysis['competition_level']}")
        insights.append(f"Market saturation: {competition_analysis['market_saturation']}")
        
        return insights
    
    def _generate_market_recommendations(self, market_analysis: Dict, 
                                       competition_analysis: Dict) -> List[str]:
        """Generate market-specific recommendations"""
        recommendations = []
        
        if market_analysis["market_size"] in ["Large", "Very Large"]:
            recommendations.append("Focus on market penetration and user acquisition")
        
        if competition_analysis["competition_level"] == "Very High":
            recommendations.append("Find unique differentiation or target underserved niche")
        
        if competition_analysis["entry_difficulty"] == "Easy":
            recommendations.append("Quick market entry recommended to establish position")
        
        opportunities = competition_analysis.get("differentiation_opportunities", [])
        if opportunities:
            recommendations.append(f"Leverage AI capabilities: {', '.join(opportunities[:3])}")
        
        return recommendations
    
    def _assess_market_risks(self, market_analysis: Dict, competition_analysis: Dict) -> Dict[str, str]:
        """Assess market risks"""
        risks = {}
        
        if competition_analysis["competition_level"] in ["High", "Very High"]:
            risks["competition"] = "High - Difficult to differentiate and gain market share"
        
        if market_analysis["market_size"] == "Small":
            risks["market_size"] = "Medium - Limited growth potential"
        
        if competition_analysis["entry_difficulty"] == "Very Difficult":
            risks["entry_barriers"] = "High - Significant resources required for market entry"
        
        if not risks:
            risks["overall"] = "Low - Favorable market conditions"
        
        return risks 