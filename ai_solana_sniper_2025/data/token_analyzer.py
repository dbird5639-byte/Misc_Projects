"""
Token Analyzer for AI-Powered Solana Meme Coin Sniper
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import re

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SafetyAnalysis:
    """Represents token safety analysis results"""
    is_safe: bool
    risk_score: float  # 0.0 (safe) to 1.0 (high risk)
    risk_factors: List[str]
    honeypot_detected: bool
    rug_pull_risk: bool
    liquidity_locked: bool
    ownership_renounced: bool
    contract_verified: bool
    warnings: List[str]
    recommendations: List[str]
    analysis_timestamp: datetime


@dataclass
class TechnicalAnalysis:
    """Represents technical analysis results"""
    trend: str  # "bullish", "bearish", "neutral"
    momentum: float  # -1.0 to 1.0
    volatility: float  # 0.0 to 1.0
    support_levels: List[float]
    resistance_levels: List[float]
    rsi: float
    macd_signal: str
    volume_trend: str
    price_pattern: str
    confidence: float
    analysis_timestamp: datetime


class TokenAnalyzer:
    """
    Analyzes tokens for safety and technical indicators
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        
        # Analysis thresholds
        self.min_liquidity = config.get("min_liquidity", 1000)
        self.min_holders = config.get("min_holders", 10)
        self.max_ownership_percentage = config.get("max_ownership_percentage", 5.0)
        self.min_contract_age_hours = config.get("min_contract_age_hours", 1)
        
        # API endpoints
        self.solscan_api_url = "https://api.solscan.io"
        self.birdeye_api_url = "https://public-api.birdeye.so"
        self.dexscreener_api_url = "https://api.dexscreener.com/latest"
        
        # API keys
        self.birdeye_api_key = config.get("birdeye_api_key")
        
        # Session management
        self.session = None
        self.is_initialized = False
        
        # Cache for analysis results
        self.safety_cache: Dict[str, SafetyAnalysis] = {}
        self.technical_cache: Dict[str, TechnicalAnalysis] = {}
        self.cache_duration = config.get("cache_duration", 300)  # 5 minutes
        
        # Performance tracking
        self.analysis_count = 0
        self.safe_tokens = 0
        self.unsafe_tokens = 0
        
    async def initialize(self):
        """Initialize the token analyzer"""
        try:
            # Create aiohttp session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                headers={
                    "User-Agent": "AI-Solana-Sniper/1.0"
                }
            )
            
            self.is_initialized = True
            logger.info("Token analyzer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize token analyzer: {e}")
            return False
    
    async def analyze_token_safety(self, token_address: str) -> SafetyAnalysis:
        """Analyze token safety and security"""
        # Check cache first
        if token_address in self.safety_cache:
            cached_analysis = self.safety_cache[token_address]
            if (datetime.now() - cached_analysis.analysis_timestamp).seconds < self.cache_duration:
                return cached_analysis
        
        try:
            logger.info(f"Analyzing token safety: {token_address}")
            
            # Fetch token data from multiple sources
            tasks = [
                self._get_solscan_token_data(token_address),
                self._get_birdeye_token_data(token_address),
                self._get_dexscreener_token_data(token_address),
                self._get_contract_data(token_address)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            filtered_results = []
            for result in results:
                if isinstance(result, Exception):
                    filtered_results.append(None)
                else:
                    filtered_results.append(result)
            
            # Analyze safety based on collected data
            safety_analysis = self._analyze_safety_data(token_address, filtered_results)
            
            # Cache the result
            self.safety_cache[token_address] = safety_analysis
            
            # Update metrics
            self.analysis_count += 1
            if safety_analysis.is_safe:
                self.safe_tokens += 1
            else:
                self.unsafe_tokens += 1
            
            return safety_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing token safety for {token_address}: {e}")
            return self._create_default_safety_analysis(token_address, str(e))
    
    async def analyze_token_technical(self, token_address: str, price_history: List[Dict[str, Any]]) -> TechnicalAnalysis:
        """Analyze token technical indicators"""
        # Check cache first
        if token_address in self.technical_cache:
            cached_analysis = self.technical_cache[token_address]
            if (datetime.now() - cached_analysis.analysis_timestamp).seconds < self.cache_duration:
                return cached_analysis
        
        try:
            logger.info(f"Analyzing technical indicators: {token_address}")
            
            # Perform technical analysis
            technical_analysis = self._perform_technical_analysis(price_history)
            
            # Cache the result
            self.technical_cache[token_address] = technical_analysis
            
            return technical_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing technical indicators for {token_address}: {e}")
            return self._create_default_technical_analysis()
    
    async def _get_solscan_token_data(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Get token data from Solscan"""
        if not self.session:
            return None
        
        try:
            url = f"{self.solscan_api_url}/token/meta"
            params = {"tokenAddress": token_address}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", {})
                else:
                    logger.warning(f"Solscan API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching Solscan data: {e}")
            return None
    
    async def _get_birdeye_token_data(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Get token data from Birdeye"""
        if not self.session or not self.birdeye_api_key:
            return None
        
        try:
            url = f"{self.birdeye_api_url}/public/token"
            params = {"address": token_address}
            headers = {"X-API-KEY": self.birdeye_api_key}
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", {})
                else:
                    logger.warning(f"Birdeye API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching Birdeye data: {e}")
            return None
    
    async def _get_dexscreener_token_data(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Get token data from DexScreener"""
        if not self.session:
            return None
        
        try:
            url = f"{self.dexscreener_api_url}/dex/tokens/{token_address}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("pairs", [{}])[0] if data.get("pairs") else {}
                else:
                    logger.warning(f"DexScreener API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching DexScreener data: {e}")
            return None
    
    async def _get_contract_data(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Get contract data and analysis"""
        if not self.session:
            return None
        
        try:
            url = f"{self.solscan_api_url}/account/{token_address}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", {})
                else:
                    logger.warning(f"Solscan contract API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching contract data: {e}")
            return None
    
    def _analyze_safety_data(self, token_address: str, results: List[Optional[Dict[str, Any]]]) -> SafetyAnalysis:
        """Analyze safety based on collected data"""
        solscan_data = results[0] or {}
        birdeye_data = results[1] or {}
        dexscreener_data = results[2] or {}
        contract_data = results[3] or {}
        
        risk_factors = []
        warnings = []
        recommendations = []
        
        # Check liquidity
        liquidity = (
            dexscreener_data.get("liquidity", {}).get("usd") or
            birdeye_data.get("liquidity") or
            0.0
        )
        
        if liquidity < self.min_liquidity:
            risk_factors.append("low_liquidity")
            warnings.append(f"Low liquidity: ${liquidity:,.0f}")
            recommendations.append("Wait for more liquidity before trading")
        
        # Check holders
        holders = (
            solscan_data.get("holder") or
            birdeye_data.get("holder") or
            0
        )
        
        if holders < self.min_holders:
            risk_factors.append("few_holders")
            warnings.append(f"Few holders: {holders}")
            recommendations.append("Token may be too new or illiquid")
        
        # Check ownership concentration
        ownership_percentage = (
            birdeye_data.get("ownerPercentage") or
            0.0
        )
        
        if ownership_percentage > self.max_ownership_percentage:
            risk_factors.append("high_ownership_concentration")
            warnings.append(f"High ownership concentration: {ownership_percentage:.1f}%")
            recommendations.append("Be cautious of potential rug pull")
        
        # Check contract age
        launch_time = (
            solscan_data.get("launchTime") or
            birdeye_data.get("launchTime") or
            int(time.time())
        )
        
        contract_age_hours = (time.time() - launch_time) / 3600
        
        if contract_age_hours < self.min_contract_age_hours:
            risk_factors.append("new_contract")
            warnings.append(f"Very new contract: {contract_age_hours:.1f} hours old")
            recommendations.append("Wait for more trading history")
        
        # Check for honeypot indicators
        honeypot_detected = self._detect_honeypot(dexscreener_data, birdeye_data)
        if honeypot_detected:
            risk_factors.append("potential_honeypot")
            warnings.append("Potential honeypot detected")
            recommendations.append("Avoid this token - likely a scam")
        
        # Check for rug pull indicators
        rug_pull_risk = self._detect_rug_pull_risk(birdeye_data, dexscreener_data)
        if rug_pull_risk:
            risk_factors.append("rug_pull_risk")
            warnings.append("High rug pull risk detected")
            recommendations.append("Extreme caution required")
        
        # Check liquidity lock
        liquidity_locked = self._check_liquidity_lock(birdeye_data, dexscreener_data)
        
        # Check ownership renounced
        ownership_renounced = self._check_ownership_renounced(birdeye_data, contract_data)
        
        # Check contract verification
        contract_verified = self._check_contract_verification(contract_data)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(risk_factors, liquidity, holders, ownership_percentage)
        
        # Determine if token is safe
        is_safe = (
            risk_score < 0.7 and
            not honeypot_detected and
            not rug_pull_risk and
            liquidity >= self.min_liquidity
        )
        
        return SafetyAnalysis(
            is_safe=is_safe,
            risk_score=risk_score,
            risk_factors=risk_factors,
            honeypot_detected=honeypot_detected,
            rug_pull_risk=rug_pull_risk,
            liquidity_locked=liquidity_locked,
            ownership_renounced=ownership_renounced,
            contract_verified=contract_verified,
            warnings=warnings,
            recommendations=recommendations,
            analysis_timestamp=datetime.now()
        )
    
    def _detect_honeypot(self, dexscreener_data: Dict[str, Any], birdeye_data: Dict[str, Any]) -> bool:
        """Detect potential honeypot characteristics"""
        honeypot_indicators = []
        
        # Check for unusual buy/sell ratios
        txns = dexscreener_data.get("txns", {})
        h24_txns = txns.get("h24", {})
        buys = h24_txns.get("buys", 0)
        sells = h24_txns.get("sells", 0)
        
        if buys > 0 and sells == 0:
            honeypot_indicators.append("no_sells")
        
        if buys > 0 and sells > 0 and (sells / buys) < 0.1:
            honeypot_indicators.append("very_few_sells")
        
        # Check for price manipulation
        price_change = birdeye_data.get("priceChange24h", 0)
        if price_change > 1000:  # 1000%+ price increase
            honeypot_indicators.append("extreme_price_increase")
        
        # Check for low liquidity relative to market cap
        liquidity = dexscreener_data.get("liquidity", {}).get("usd", 0)
        market_cap = dexscreener_data.get("marketCap", 0)
        
        if market_cap > 0 and (liquidity / market_cap) < 0.01:
            honeypot_indicators.append("low_liquidity_ratio")
        
        return len(honeypot_indicators) >= 2
    
    def _detect_rug_pull_risk(self, birdeye_data: Dict[str, Any], dexscreener_data: Dict[str, Any]) -> bool:
        """Detect rug pull risk indicators"""
        rug_pull_indicators = []
        
        # Check ownership concentration
        ownership_percentage = birdeye_data.get("ownerPercentage", 0)
        if ownership_percentage > 20:  # More than 20% ownership
            rug_pull_indicators.append("high_ownership")
        
        # Check for recent large transfers
        # This would require additional API calls to check transfer history
        
        # Check for liquidity removal patterns
        liquidity = dexscreener_data.get("liquidity", {}).get("usd", 0)
        if liquidity < 100:  # Very low liquidity
            rug_pull_indicators.append("very_low_liquidity")
        
        # Check for price crash
        price_change = birdeye_data.get("priceChange24h", 0)
        if price_change < -90:  # 90%+ price drop
            rug_pull_indicators.append("extreme_price_drop")
        
        return len(rug_pull_indicators) >= 2
    
    def _check_liquidity_lock(self, birdeye_data: Dict[str, Any], dexscreener_data: Dict[str, Any]) -> bool:
        """Check if liquidity is locked"""
        # This would require checking specific liquidity lock contracts
        # For now, return False as placeholder
        return False
    
    def _check_ownership_renounced(self, birdeye_data: Dict[str, Any], contract_data: Dict[str, Any]) -> bool:
        """Check if ownership is renounced"""
        # This would require checking contract ownership
        # For now, return False as placeholder
        return False
    
    def _check_contract_verification(self, contract_data: Dict[str, Any]) -> bool:
        """Check if contract is verified"""
        # This would require checking contract verification status
        # For now, return True as placeholder
        return True
    
    def _calculate_risk_score(self, risk_factors: List[str], liquidity: float, 
                            holders: int, ownership_percentage: float) -> float:
        """Calculate overall risk score"""
        base_score = 0.0
        
        # Risk factor weights
        factor_weights = {
            "low_liquidity": 0.3,
            "few_holders": 0.2,
            "high_ownership_concentration": 0.4,
            "new_contract": 0.1,
            "potential_honeypot": 0.8,
            "rug_pull_risk": 0.9
        }
        
        # Add risk from factors
        for factor in risk_factors:
            base_score += factor_weights.get(factor, 0.1)
        
        # Add risk from metrics
        if liquidity < 1000:
            base_score += 0.2
        elif liquidity < 10000:
            base_score += 0.1
        
        if holders < 10:
            base_score += 0.2
        elif holders < 50:
            base_score += 0.1
        
        if ownership_percentage > 50:
            base_score += 0.3
        elif ownership_percentage > 20:
            base_score += 0.2
        elif ownership_percentage > 10:
            base_score += 0.1
        
        # Normalize to 0.0-1.0 range
        return min(base_score, 1.0)
    
    def _perform_technical_analysis(self, price_history: List[Dict[str, Any]]) -> TechnicalAnalysis:
        """Perform technical analysis on price history"""
        if not price_history or len(price_history) < 10:
            return self._create_default_technical_analysis()
        
        try:
            # Extract price data
            prices = [float(entry.get("price", 0)) for entry in price_history]
            volumes = [float(entry.get("volume", 0)) for entry in price_history]
            
            if not prices or len(prices) < 2:
                return self._create_default_technical_analysis()
            
            # Calculate trend
            trend = self._calculate_trend(prices)
            
            # Calculate momentum
            momentum = self._calculate_momentum(prices)
            
            # Calculate volatility
            volatility = self._calculate_volatility(prices)
            
            # Calculate support and resistance levels
            support_levels, resistance_levels = self._calculate_support_resistance(prices)
            
            # Calculate RSI
            rsi = self._calculate_rsi(prices)
            
            # Calculate MACD signal
            macd_signal = self._calculate_macd_signal(prices)
            
            # Analyze volume trend
            volume_trend = self._analyze_volume_trend(volumes)
            
            # Identify price patterns
            price_pattern = self._identify_price_pattern(prices)
            
            # Calculate confidence
            confidence = self._calculate_technical_confidence(prices, volumes)
            
            return TechnicalAnalysis(
                trend=trend,
                momentum=momentum,
                volatility=volatility,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                rsi=rsi,
                macd_signal=macd_signal,
                volume_trend=volume_trend,
                price_pattern=price_pattern,
                confidence=confidence,
                analysis_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error performing technical analysis: {e}")
            return self._create_default_technical_analysis()
    
    def _calculate_trend(self, prices: List[float]) -> str:
        """Calculate price trend"""
        if len(prices) < 2:
            return "neutral"
        
        # Simple trend calculation
        recent_prices = prices[-10:] if len(prices) >= 10 else prices
        first_price = recent_prices[0]
        last_price = recent_prices[-1]
        
        change_percent = ((last_price - first_price) / first_price) * 100
        
        if change_percent > 5:
            return "bullish"
        elif change_percent < -5:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_momentum(self, prices: List[float]) -> float:
        """Calculate price momentum"""
        if len(prices) < 2:
            return 0.0
        
        # Simple momentum calculation
        recent_prices = prices[-5:] if len(prices) >= 5 else prices
        first_price = recent_prices[0]
        last_price = recent_prices[-1]
        
        momentum = (last_price - first_price) / first_price
        return max(-1.0, min(1.0, momentum))  # Clamp to -1.0 to 1.0
    
    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility"""
        if len(prices) < 2:
            return 0.0
        
        # Calculate standard deviation of returns
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                returns.append((prices[i] - prices[i-1]) / prices[i-1])
        
        if not returns:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        volatility = variance ** 0.5
        
        return min(1.0, volatility)  # Clamp to 0.0 to 1.0
    
    def _calculate_support_resistance(self, prices: List[float]) -> Tuple[List[float], List[float]]:
        """Calculate support and resistance levels"""
        if len(prices) < 5:
            return [], []
        
        # Simple support/resistance calculation
        min_price = min(prices)
        max_price = max(prices)
        
        support_levels = [min_price * 0.95, min_price * 0.98]
        resistance_levels = [max_price * 1.02, max_price * 1.05]
        
        return support_levels, resistance_levels
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        # Calculate RSI
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) < period:
            return 50.0
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd_signal(self, prices: List[float]) -> str:
        """Calculate MACD signal"""
        if len(prices) < 26:
            return "neutral"
        
        # Simple MACD calculation
        ema12 = self._calculate_ema(prices, 12)
        ema26 = self._calculate_ema(prices, 26)
        
        if ema12 > ema26:
            return "bullish"
        elif ema12 < ema26:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _analyze_volume_trend(self, volumes: List[float]) -> str:
        """Analyze volume trend"""
        if len(volumes) < 2:
            return "neutral"
        
        recent_volumes = volumes[-5:] if len(volumes) >= 5 else volumes
        avg_volume = sum(recent_volumes) / len(recent_volumes)
        
        if avg_volume > volumes[0] * 1.5:
            return "increasing"
        elif avg_volume < volumes[0] * 0.5:
            return "decreasing"
        else:
            return "stable"
    
    def _identify_price_pattern(self, prices: List[float]) -> str:
        """Identify price patterns"""
        if len(prices) < 5:
            return "no_pattern"
        
        # Simple pattern recognition
        recent_prices = prices[-5:]
        
        # Check for double top
        if recent_prices[1] > recent_prices[0] and recent_prices[1] > recent_prices[2] and \
           recent_prices[3] > recent_prices[2] and recent_prices[3] > recent_prices[4]:
            return "double_top"
        
        # Check for double bottom
        if recent_prices[1] < recent_prices[0] and recent_prices[1] < recent_prices[2] and \
           recent_prices[3] < recent_prices[2] and recent_prices[3] < recent_prices[4]:
            return "double_bottom"
        
        # Check for uptrend
        if all(recent_prices[i] <= recent_prices[i+1] for i in range(len(recent_prices)-1)):
            return "uptrend"
        
        # Check for downtrend
        if all(recent_prices[i] >= recent_prices[i+1] for i in range(len(recent_prices)-1)):
            return "downtrend"
        
        return "no_pattern"
    
    def _calculate_technical_confidence(self, prices: List[float], volumes: List[float]) -> float:
        """Calculate confidence in technical analysis"""
        if len(prices) < 5:
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # Increase confidence with more data points
        if len(prices) >= 20:
            confidence += 0.2
        elif len(prices) >= 10:
            confidence += 0.1
        
        # Increase confidence with consistent volume
        if volumes:
            avg_volume = sum(volumes) / len(volumes)
            if avg_volume > 0:
                volume_consistency = 1 - (max(volumes) - min(volumes)) / avg_volume
                confidence += volume_consistency * 0.2
        
        # Increase confidence with clear trends
        trend = self._calculate_trend(prices)
        if trend != "neutral":
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _create_default_safety_analysis(self, token_address: str, error: str) -> SafetyAnalysis:
        """Create default safety analysis when analysis fails"""
        return SafetyAnalysis(
            is_safe=False,
            risk_score=1.0,
            risk_factors=["analysis_failed"],
            honeypot_detected=True,  # Assume worst case
            rug_pull_risk=True,      # Assume worst case
            liquidity_locked=False,
            ownership_renounced=False,
            contract_verified=False,
            warnings=[f"Analysis failed: {error}"],
            recommendations=["Avoid trading until analysis can be completed"],
            analysis_timestamp=datetime.now()
        )
    
    def _create_default_technical_analysis(self) -> TechnicalAnalysis:
        """Create default technical analysis when analysis fails"""
        return TechnicalAnalysis(
            trend="neutral",
            momentum=0.0,
            volatility=0.0,
            support_levels=[],
            resistance_levels=[],
            rsi=50.0,
            macd_signal="neutral",
            volume_trend="neutral",
            price_pattern="no_pattern",
            confidence=0.0,
            analysis_timestamp=datetime.now()
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        total_analysis = self.safe_tokens + self.unsafe_tokens
        safety_rate = self.safe_tokens / max(total_analysis, 1)
        
        return {
            "total_analysis": total_analysis,
            "safe_tokens": self.safe_tokens,
            "unsafe_tokens": self.unsafe_tokens,
            "safety_rate": safety_rate,
            "safety_cache_size": len(self.safety_cache),
            "technical_cache_size": len(self.technical_cache)
        }
    
    async def close(self):
        """Close the token analyzer"""
        if self.session:
            await self.session.close()
        
        logger.info("Token analyzer closed") 