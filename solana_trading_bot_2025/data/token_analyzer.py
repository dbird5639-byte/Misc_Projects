"""
Token Analyzer for Solana Trading Bot 2025

Analyzes tokens for safety, trading potential, and risk assessment.
"""

import asyncio
import aiohttp
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
import re

from bots.base_bot import TokenInfo
from config.settings import Config

class TokenAnalyzer:
    """Analyzes tokens for trading decisions"""
    
    def __init__(self, config: Config):
        self.config = config
        self.session = None
        self.analysis_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Analysis thresholds
        self.min_liquidity = 5000
        self.min_volume = 1000
        self.max_holder_percentage = 0.1  # Max 10% held by single wallet
        self.min_holders = 50
        
        # Risk indicators
        self.risk_keywords = [
            "honeypot", "rug", "scam", "fake", "test", "moon", "safe", "inu",
            "elon", "doge", "shib", "pepe", "wojak", "chad", "based"
        ]
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def analyze_token(self, token: TokenInfo) -> Dict[str, Any]:
        """Comprehensive token analysis"""
        try:
            # Check cache first
            cache_key = f"analysis_{token.address}"
            if cache_key in self.analysis_cache:
                cached_data, timestamp = self.analysis_cache[cache_key]
                if datetime.now().timestamp() - timestamp < self.cache_ttl:
                    return cached_data
            
            # Perform analysis
            analysis = {
                "token_address": token.address,
                "timestamp": datetime.now(),
                "should_trade": False,
                "confidence": 0.0,
                "risk_score": 0.0,
                "safety_checks": {},
                "technical_indicators": {},
                "risk_factors": [],
                "recommendations": []
            }
            
            # Safety checks
            safety_checks = await self._perform_safety_checks(token)
            analysis["safety_checks"] = safety_checks
            
            # Technical analysis
            technical_indicators = await self._analyze_technical_indicators(token)
            analysis["technical_indicators"] = technical_indicators
            
            # Risk assessment
            risk_factors = await self._assess_risk_factors(token)
            analysis["risk_factors"] = risk_factors
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(safety_checks, risk_factors)
            analysis["risk_score"] = risk_score
            
            # Determine if should trade
            should_trade, confidence = self._determine_trade_decision(
                safety_checks, technical_indicators, risk_score
            )
            analysis["should_trade"] = should_trade
            analysis["confidence"] = confidence
            
            # Generate recommendations
            recommendations = self._generate_recommendations(analysis)
            analysis["recommendations"] = recommendations
            
            # Cache the result
            self.analysis_cache[cache_key] = (analysis, datetime.now().timestamp())
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing token {token.address}: {e}")
            return {
                "token_address": token.address,
                "timestamp": datetime.now(),
                "should_trade": False,
                "confidence": 0.0,
                "risk_score": 1.0,
                "error": str(e)
            }
    
    async def is_safe_token(self, token: TokenInfo) -> bool:
        """Quick safety check for token"""
        try:
            safety_checks = await self._perform_safety_checks(token)
            
            # Must pass critical safety checks
            critical_checks = [
                "has_liquidity",
                "has_volume",
                "contract_verified",
                "not_honeypot"
            ]
            
            for check in critical_checks:
                if not safety_checks.get(check, False):
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error checking token safety: {e}")
            return False
    
    async def _perform_safety_checks(self, token: TokenInfo) -> Dict[str, bool]:
        """Perform basic safety checks"""
        checks = {
            "has_liquidity": token.liquidity >= self.min_liquidity,
            "has_volume": token.volume_24h >= self.min_volume,
            "has_market_cap": token.market_cap > 0,
            "has_price": token.price > 0,
            "contract_verified": True,  # Placeholder - would check actual verification
            "not_honeypot": True,  # Placeholder - would check for honeypot
            "has_holders": True,  # Placeholder - would check holder count
            "reasonable_holder_distribution": True  # Placeholder - would check distribution
        }
        
        # Additional checks can be added here
        # - Contract code analysis
        # - Holder distribution analysis
        # - Transaction pattern analysis
        
        return checks
    
    async def _analyze_technical_indicators(self, token: TokenInfo) -> Dict[str, Any]:
        """Analyze technical indicators"""
        indicators = {
            "price_momentum": 0.0,
            "volume_trend": 0.0,
            "liquidity_stability": 0.0,
            "volatility": 0.0,
            "support_resistance": {}
        }
        
        try:
            # Get historical data for analysis
            historical_data = await self._get_historical_data(token.address)
            
            if historical_data:
                # Calculate momentum
                indicators["price_momentum"] = self._calculate_momentum(historical_data)
                
                # Calculate volume trend
                indicators["volume_trend"] = self._calculate_volume_trend(historical_data)
                
                # Calculate volatility
                indicators["volatility"] = self._calculate_volatility(historical_data)
                
                # Calculate support/resistance levels
                indicators["support_resistance"] = self._calculate_support_resistance(historical_data)
            
        except Exception as e:
            print(f"Error analyzing technical indicators: {e}")
        
        return indicators
    
    async def _get_historical_data(self, token_address: str) -> Optional[List[Dict[str, Any]]]:
        """Get historical price and volume data"""
        try:
            # This would fetch actual historical data from APIs
            # For now, return placeholder data
            return [
                {"timestamp": datetime.now() - timedelta(hours=i), "price": 1.0, "volume": 1000}
                for i in range(24, 0, -1)
            ]
            
        except Exception as e:
            print(f"Error getting historical data: {e}")
            return None
    
    def _calculate_momentum(self, historical_data: List[Dict[str, Any]]) -> float:
        """Calculate price momentum"""
        try:
            if len(historical_data) < 2:
                return 0.0
            
            recent_prices = [d["price"] for d in historical_data[-6:]]  # Last 6 hours
            if len(recent_prices) < 2:
                return 0.0
            
            # Simple momentum calculation
            momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            return momentum
            
        except Exception as e:
            print(f"Error calculating momentum: {e}")
            return 0.0
    
    def _calculate_volume_trend(self, historical_data: List[Dict[str, Any]]) -> float:
        """Calculate volume trend"""
        try:
            if len(historical_data) < 2:
                return 0.0
            
            recent_volumes = [d["volume"] for d in historical_data[-6:]]  # Last 6 hours
            if len(recent_volumes) < 2:
                return 0.0
            
            # Simple volume trend calculation
            trend = (recent_volumes[-1] - recent_volumes[0]) / recent_volumes[0]
            return trend
            
        except Exception as e:
            print(f"Error calculating volume trend: {e}")
            return 0.0
    
    def _calculate_volatility(self, historical_data: List[Dict[str, Any]]) -> float:
        """Calculate price volatility"""
        try:
            if len(historical_data) < 2:
                return 0.0
            
            prices = [d["price"] for d in historical_data]
            
            # Calculate standard deviation
            mean_price = sum(prices) / len(prices)
            variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)
            volatility = variance ** 0.5
            
            return volatility / mean_price  # Normalized volatility
            
        except Exception as e:
            print(f"Error calculating volatility: {e}")
            return 0.0
    
    def _calculate_support_resistance(self, historical_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate support and resistance levels"""
        try:
            if len(historical_data) < 10:
                return {"support": 0.0, "resistance": 0.0}
            
            prices = [d["price"] for d in historical_data]
            
            # Simple support/resistance calculation
            support = min(prices)
            resistance = max(prices)
            
            return {
                "support": support,
                "resistance": resistance
            }
            
        except Exception as e:
            print(f"Error calculating support/resistance: {e}")
            return {"support": 0.0, "resistance": 0.0}
    
    async def _assess_risk_factors(self, token: TokenInfo) -> List[str]:
        """Assess risk factors for the token"""
        risk_factors = []
        
        try:
            # Check token name/symbol for suspicious keywords
            token_name = (token.name + " " + token.symbol).lower()
            for keyword in self.risk_keywords:
                if keyword in token_name:
                    risk_factors.append(f"Suspicious keyword in name: {keyword}")
            
            # Check liquidity
            if token.liquidity < self.min_liquidity:
                risk_factors.append(f"Low liquidity: ${token.liquidity:,.0f}")
            
            # Check volume
            if token.volume_24h < self.min_volume:
                risk_factors.append(f"Low volume: ${token.volume_24h:,.0f}")
            
            # Check market cap
            if token.market_cap < 1000:
                risk_factors.append(f"Very low market cap: ${token.market_cap:,.0f}")
            
            # Check price
            if token.price <= 0:
                risk_factors.append("Invalid price")
            
            # Additional risk checks can be added here
            # - Contract analysis
            # - Holder analysis
            # - Transaction pattern analysis
            
        except Exception as e:
            print(f"Error assessing risk factors: {e}")
            risk_factors.append(f"Error in risk assessment: {e}")
        
        return risk_factors
    
    def _calculate_risk_score(self, safety_checks: Dict[str, bool], risk_factors: List[str]) -> float:
        """Calculate overall risk score (0.0 = safe, 1.0 = very risky)"""
        try:
            risk_score = 0.0
            
            # Safety checks contribute to risk score
            critical_checks = ["has_liquidity", "has_volume", "contract_verified", "not_honeypot"]
            for check in critical_checks:
                if not safety_checks.get(check, False):
                    risk_score += 0.25  # Each failed check adds 25% risk
            
            # Risk factors contribute to risk score
            risk_score += len(risk_factors) * 0.1  # Each risk factor adds 10% risk
            
            # Cap at 1.0
            return min(risk_score, 1.0)
            
        except Exception as e:
            print(f"Error calculating risk score: {e}")
            return 1.0  # Return maximum risk on error
    
    def _determine_trade_decision(self, safety_checks: Dict[str, bool], 
                                technical_indicators: Dict[str, Any], 
                                risk_score: float) -> tuple[bool, float]:
        """Determine if we should trade and with what confidence"""
        try:
            # Must pass all critical safety checks
            critical_checks = ["has_liquidity", "has_volume", "contract_verified", "not_honeypot"]
            for check in critical_checks:
                if not safety_checks.get(check, False):
                    return False, 0.0
            
            # Risk score must be acceptable
            if risk_score > 0.7:  # Max 70% risk
                return False, 0.0
            
            # Calculate confidence based on various factors
            confidence = 0.5  # Base confidence
            
            # Technical indicators boost confidence
            momentum = technical_indicators.get("price_momentum", 0.0)
            volume_trend = technical_indicators.get("volume_trend", 0.0)
            
            if momentum > 0.1:  # Positive momentum
                confidence += 0.2
            
            if volume_trend > 0.2:  # Increasing volume
                confidence += 0.2
            
            # Risk score affects confidence
            confidence -= risk_score * 0.3
            
            # Cap confidence
            confidence = max(0.0, min(1.0, confidence))
            
            # Minimum confidence threshold
            should_trade = confidence >= 0.6
            
            return should_trade, confidence
            
        except Exception as e:
            print(f"Error determining trade decision: {e}")
            return False, 0.0
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate trading recommendations"""
        recommendations = []
        
        try:
            if analysis["should_trade"]:
                confidence = analysis["confidence"]
                if confidence > 0.8:
                    recommendations.append("Strong buy signal")
                elif confidence > 0.6:
                    recommendations.append("Moderate buy signal")
                else:
                    recommendations.append("Weak buy signal")
            else:
                recommendations.append("Do not trade")
            
            # Add specific recommendations based on analysis
            risk_factors = analysis.get("risk_factors", [])
            if risk_factors:
                recommendations.append(f"Risk factors: {', '.join(risk_factors[:3])}")
            
            technical_indicators = analysis.get("technical_indicators", {})
            momentum = technical_indicators.get("price_momentum", 0.0)
            if momentum > 0.2:
                recommendations.append("Strong positive momentum")
            elif momentum < -0.2:
                recommendations.append("Negative momentum - consider waiting")
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            recommendations.append("Error in analysis")
        
        return recommendations
    
    def clear_cache(self):
        """Clear the analysis cache"""
        self.analysis_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.analysis_cache),
            "cache_ttl": self.cache_ttl
        } 