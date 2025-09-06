"""
Market Data Manager for AI-Powered Solana Meme Coin Sniper
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import defaultdict

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TokenData:
    """Represents token market data"""
    address: str
    name: str
    symbol: str
    price: float
    volume_24h: float
    liquidity: float
    market_cap: float
    launch_time: int
    price_change_24h: float
    holders: int
    transactions_24h: int
    last_updated: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MarketData:
    """Represents market data"""
    total_volume: float
    total_liquidity: float
    active_tokens: int
    trending_tokens: List[str]
    market_sentiment: str
    volatility_index: float
    last_updated: datetime


class MarketDataManager:
    """
    Manages market data from multiple sources
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        
        # API endpoints
        self.jupiter_api_url = "https://price.jup.ag/v4"
        self.birdeye_api_url = "https://public-api.birdeye.so"
        self.dexscreener_api_url = "https://api.dexscreener.com/latest"
        self.solscan_api_url = "https://api.solscan.io"
        
        # API keys
        self.jupiter_api_key = config.get("jupiter_api_key")
        self.birdeye_api_key = config.get("birdeye_api_key")
        self.dexscreener_api_key = config.get("dexscreener_api_key")
        
        # Caching
        self.cache_duration = config.get("cache_duration", 30)  # seconds
        self.max_cache_size = config.get("max_cache_size", 1000)
        
        # Data storage
        self.token_cache: Dict[str, TokenData] = {}
        self.market_cache: Optional[MarketData] = None
        self.new_tokens_cache: List[Dict[str, Any]] = []
        self.social_mentions_cache: List[Dict[str, Any]] = []
        
        # Rate limiting
        self.rate_limits = {
            "jupiter": {"calls": 0, "last_reset": time.time(), "limit": 100},
            "birdeye": {"calls": 0, "last_reset": time.time(), "limit": 50},
            "dexscreener": {"calls": 0, "last_reset": time.time(), "limit": 200},
            "solscan": {"calls": 0, "last_reset": time.time(), "limit": 100}
        }
        
        # Session management
        self.session = None
        self.is_initialized = False
        
        # Performance tracking
        self.api_call_count = 0
        self.cache_hit_count = 0
        self.error_count = 0
        
    async def initialize(self):
        """Initialize the market data manager"""
        try:
            # Create aiohttp session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                headers={
                    "User-Agent": "AI-Solana-Sniper/1.0"
                }
            )
            
            # Test API connections
            await self._test_api_connections()
            
            self.is_initialized = True
            logger.info("Market data manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize market data manager: {e}")
            return False
    
    async def _test_api_connections(self):
        """Test API connections"""
        test_tasks = [
            self._test_jupiter_api(),
            self._test_birdeye_api(),
            self._test_dexscreener_api(),
            self._test_solscan_api()
        ]
        
        results = await asyncio.gather(*test_tasks, return_exceptions=True)
        
        apis = ["Jupiter", "Birdeye", "DexScreener", "Solscan"]
        for api, result in zip(apis, results):
            if isinstance(result, Exception):
                logger.warning(f"{api} API test failed: {result}")
            else:
                logger.info(f"{api} API connection successful")
    
    async def _test_jupiter_api(self):
        """Test Jupiter API connection"""
        if not self.session:
            return False
        
        url = f"{self.jupiter_api_url}/price"
        params = {"ids": "SOL"}
        
        async with self.session.get(url, params=params) as response:
            return response.status == 200
    
    async def _test_birdeye_api(self):
        """Test Birdeye API connection"""
        if not self.session or not self.birdeye_api_key:
            return False
        
        url = f"{self.birdeye_api_url}/public/price"
        params = {"address": "So11111111111111111111111111111111111111112"}
        headers = {"X-API-KEY": self.birdeye_api_key}
        
        async with self.session.get(url, params=params, headers=headers) as response:
            return response.status == 200
    
    async def _test_dexscreener_api(self):
        """Test DexScreener API connection"""
        if not self.session:
            return False
        
        url = f"{self.dexscreener_api_url}/dex/tokens/So11111111111111111111111111111111111111112"
        
        async with self.session.get(url) as response:
            return response.status == 200
    
    async def _test_solscan_api(self):
        """Test Solscan API connection"""
        if not self.session:
            return False
        
        url = f"{self.solscan_api_url}/account/So11111111111111111111111111111111111111112"
        
        async with self.session.get(url) as response:
            return response.status == 200
    
    async def get_token_data(self, token_address: str) -> Optional[TokenData]:
        """Get comprehensive token data"""
        # Check cache first
        if token_address in self.token_cache:
            cached_data = self.token_cache[token_address]
            if (datetime.now() - cached_data.last_updated).seconds < self.cache_duration:
                self.cache_hit_count += 1
                return cached_data
        
        try:
            # Fetch data from multiple sources
            tasks = [
                self._get_jupiter_token_data(token_address),
                self._get_birdeye_token_data(token_address),
                self._get_dexscreener_token_data(token_address),
                self._get_solscan_token_data(token_address)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            filtered_results = []
            for result in results:
                if isinstance(result, Exception):
                    filtered_results.append(None)
                else:
                    filtered_results.append(result)
            
            # Combine data from all sources
            token_data = self._combine_token_data(token_address, filtered_results)
            
            if token_data:
                # Cache the result
                self.token_cache[token_address] = token_data
                self._cleanup_cache()
            
            return token_data
            
        except Exception as e:
            logger.error(f"Error getting token data for {token_address}: {e}")
            self.error_count += 1
            return None
    
    async def _get_jupiter_token_data(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Get token data from Jupiter API"""
        if not self.session or not self._check_rate_limit("jupiter"):
            return None
        
        try:
            url = f"{self.jupiter_api_url}/price"
            params = {"ids": token_address}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    self.rate_limits["jupiter"]["calls"] += 1
                    return data.get("data", {}).get(token_address, {})
                else:
                    logger.warning(f"Jupiter API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching Jupiter data: {e}")
            return None
    
    async def _get_birdeye_token_data(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Get token data from Birdeye API"""
        if not self.session or not self.birdeye_api_key or not self._check_rate_limit("birdeye"):
            return None
        
        try:
            url = f"{self.birdeye_api_url}/public/token"
            params = {"address": token_address}
            headers = {"X-API-KEY": self.birdeye_api_key}
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    self.rate_limits["birdeye"]["calls"] += 1
                    return data.get("data", {})
                else:
                    logger.warning(f"Birdeye API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching Birdeye data: {e}")
            return None
    
    async def _get_dexscreener_token_data(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Get token data from DexScreener API"""
        if not self.session or not self._check_rate_limit("dexscreener"):
            return None
        
        try:
            url = f"{self.dexscreener_api_url}/dex/tokens/{token_address}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    self.rate_limits["dexscreener"]["calls"] += 1
                    return data.get("pairs", [{}])[0] if data.get("pairs") else {}
                else:
                    logger.warning(f"DexScreener API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching DexScreener data: {e}")
            return None
    
    async def _get_solscan_token_data(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Get token data from Solscan API"""
        if not self.session or not self._check_rate_limit("solscan"):
            return None
        
        try:
            url = f"{self.solscan_api_url}/token/meta"
            params = {"tokenAddress": token_address}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    self.rate_limits["solscan"]["calls"] += 1
                    return data.get("data", {})
                else:
                    logger.warning(f"Solscan API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching Solscan data: {e}")
            return None
    
    def _combine_token_data(self, token_address: str, results: List[Optional[Dict[str, Any]]]) -> Optional[TokenData]:
        """Combine data from multiple sources"""
        jupiter_data = results[0] or {}
        birdeye_data = results[1] or {}
        dexscreener_data = results[2] or {}
        solscan_data = results[3] or {}
        
        # Extract price (prioritize Jupiter, then Birdeye)
        price = (
            jupiter_data.get("price") or
            birdeye_data.get("price") or
            dexscreener_data.get("priceUsd") or
            0.0
        )
        
        # Extract volume (prioritize DexScreener, then Birdeye)
        volume_24h = (
            dexscreener_data.get("volume", {}).get("h24") or
            birdeye_data.get("volume24h") or
            jupiter_data.get("volume24h") or
            0.0
        )
        
        # Extract liquidity
        liquidity = (
            dexscreener_data.get("liquidity", {}).get("usd") or
            birdeye_data.get("liquidity") or
            0.0
        )
        
        # Extract market cap
        market_cap = (
            birdeye_data.get("marketCap") or
            dexscreener_data.get("marketCap") or
            price * (solscan_data.get("supply", 0) or 0)
        )
        
        # Extract other data
        name = (
            solscan_data.get("name") or
            birdeye_data.get("name") or
            dexscreener_data.get("baseToken", {}).get("name") or
            "Unknown"
        )
        
        symbol = (
            solscan_data.get("symbol") or
            birdeye_data.get("symbol") or
            dexscreener_data.get("baseToken", {}).get("symbol") or
            "UNK"
        )
        
        # Extract launch time
        launch_time = (
            solscan_data.get("launchTime") or
            birdeye_data.get("launchTime") or
            int(time.time())
        )
        
        # Extract price change
        price_change_24h = (
            birdeye_data.get("priceChange24h") or
            dexscreener_data.get("priceChange", {}).get("h24") or
            0.0
        )
        
        # Extract holders
        holders = (
            solscan_data.get("holder") or
            birdeye_data.get("holder") or
            0
        )
        
        # Extract transactions
        transactions_24h = (
            dexscreener_data.get("txns", {}).get("h24", {}).get("buys", 0) +
            dexscreener_data.get("txns", {}).get("h24", {}).get("sells", 0)
        )
        
        return TokenData(
            address=token_address,
            name=name,
            symbol=symbol,
            price=float(price),
            volume_24h=float(volume_24h),
            liquidity=float(liquidity),
            market_cap=float(market_cap),
            launch_time=int(launch_time),
            price_change_24h=float(price_change_24h),
            holders=int(holders),
            transactions_24h=int(transactions_24h),
            last_updated=datetime.now(),
            metadata={
                "jupiter": jupiter_data,
                "birdeye": birdeye_data,
                "dexscreener": dexscreener_data,
                "solscan": solscan_data
            }
        )
    
    async def get_new_tokens(self) -> List[Dict[str, Any]]:
        """Get list of new token launches"""
        try:
            # Check cache first
            if self.new_tokens_cache and (datetime.now() - self.new_tokens_cache[0].get("timestamp", datetime.now())).seconds < 60:
                return self.new_tokens_cache
            
            # Fetch from multiple sources
            tasks = [
                self._get_jupiter_new_tokens(),
                self._get_birdeye_new_tokens(),
                self._get_dexscreener_new_tokens()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine and deduplicate
            all_tokens = []
            for result in results:
                if isinstance(result, list):
                    all_tokens.extend(result)
            
            # Remove duplicates
            unique_tokens = self._deduplicate_tokens(all_tokens)
            
            # Cache results
            self.new_tokens_cache = unique_tokens
            
            return unique_tokens
            
        except Exception as e:
            logger.error(f"Error getting new tokens: {e}")
            return []
    
    async def _get_jupiter_new_tokens(self) -> List[Dict[str, Any]]:
        """Get new tokens from Jupiter"""
        if not self._check_rate_limit("jupiter"):
            return []
        
        try:
            # Jupiter doesn't have a direct new tokens endpoint
            # This would need to be implemented based on available endpoints
            return []
            
        except Exception as e:
            logger.error(f"Error fetching Jupiter new tokens: {e}")
            return []
    
    async def _get_birdeye_new_tokens(self) -> List[Dict[str, Any]]:
        """Get new tokens from Birdeye"""
        if not self.session or not self.birdeye_api_key or not self._check_rate_limit("birdeye"):
            return []
        
        try:
            url = f"{self.birdeye_api_url}/public/tokenlist"
            params = {"sort_by": "launch_time", "sort_type": "desc", "offset": 0, "limit": 50}
            headers = {"X-API-KEY": self.birdeye_api_key}
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    self.rate_limits["birdeye"]["calls"] += 1
                    
                    tokens = []
                    for token in data.get("data", {}).get("tokens", []):
                        tokens.append({
                            "address": token.get("address"),
                            "name": token.get("name"),
                            "symbol": token.get("symbol"),
                            "launch_time": token.get("launchTime"),
                            "timestamp": datetime.now()
                        })
                    
                    return tokens
                else:
                    logger.warning(f"Birdeye new tokens API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching Birdeye new tokens: {e}")
            return []
    
    async def _get_dexscreener_new_tokens(self) -> List[Dict[str, Any]]:
        """Get new tokens from DexScreener"""
        if not self._check_rate_limit("dexscreener"):
            return []
        
        try:
            # DexScreener doesn't have a direct new tokens endpoint
            # This would need to be implemented based on available endpoints
            return []
            
        except Exception as e:
            logger.error(f"Error fetching DexScreener new tokens: {e}")
            return []
    
    async def get_social_mentions(self) -> List[Dict[str, Any]]:
        """Get social media mentions of tokens"""
        try:
            # Check cache first
            if self.social_mentions_cache and (datetime.now() - self.social_mentions_cache[0].get("timestamp", datetime.now())).seconds < 300:
                return self.social_mentions_cache
            
            # This would typically integrate with social media APIs
            # For now, return empty list
            mentions = []
            
            # Cache results
            self.social_mentions_cache = mentions
            
            return mentions
            
        except Exception as e:
            logger.error(f"Error getting social mentions: {e}")
            return []
    
    async def get_market_data(self) -> Optional[MarketData]:
        """Get overall market data"""
        try:
            # Check cache first
            if self.market_cache and (datetime.now() - self.market_cache.last_updated).seconds < self.cache_duration:
                return self.market_cache
            
            # Fetch market data from multiple sources
            tasks = [
                self._get_jupiter_market_data(),
                self._get_birdeye_market_data(),
                self._get_dexscreener_market_data()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            filtered_results = []
            for result in results:
                if isinstance(result, Exception):
                    filtered_results.append(None)
                else:
                    filtered_results.append(result)
            
            # Combine market data
            market_data = self._combine_market_data(filtered_results)
            
            if market_data:
                self.market_cache = market_data
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None
    
    async def _get_jupiter_market_data(self) -> Optional[Dict[str, Any]]:
        """Get market data from Jupiter"""
        if not self._check_rate_limit("jupiter"):
            return None
        
        try:
            # Jupiter doesn't have comprehensive market data
            # This would need to be implemented based on available endpoints
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching Jupiter market data: {e}")
            return None
    
    async def _get_birdeye_market_data(self) -> Optional[Dict[str, Any]]:
        """Get market data from Birdeye"""
        if not self.session or not self.birdeye_api_key or not self._check_rate_limit("birdeye"):
            return None
        
        try:
            url = f"{self.birdeye_api_url}/public/overview"
            headers = {"X-API-KEY": self.birdeye_api_key}
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    self.rate_limits["birdeye"]["calls"] += 1
                    return data.get("data", {})
                else:
                    logger.warning(f"Birdeye market data API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching Birdeye market data: {e}")
            return None
    
    async def _get_dexscreener_market_data(self) -> Optional[Dict[str, Any]]:
        """Get market data from DexScreener"""
        if not self.session or not self._check_rate_limit("dexscreener"):
            return None
        
        try:
            url = f"{self.dexscreener_api_url}/dex/search"
            params = {"q": "SOL"}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    self.rate_limits["dexscreener"]["calls"] += 1
                    return data
                else:
                    logger.warning(f"DexScreener market data API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching DexScreener market data: {e}")
            return None
    
    def _combine_market_data(self, results: List[Optional[Dict[str, Any]]]) -> Optional[MarketData]:
        """Combine market data from multiple sources"""
        birdeye_data = results[1] or {}
        dexscreener_data = results[2] or {}
        
        # Extract total volume
        total_volume = (
            birdeye_data.get("totalVolume24h") or
            dexscreener_data.get("totalVolume") or
            0.0
        )
        
        # Extract total liquidity
        total_liquidity = (
            birdeye_data.get("totalLiquidity") or
            dexscreener_data.get("totalLiquidity") or
            0.0
        )
        
        # Extract active tokens
        active_tokens = (
            birdeye_data.get("activeTokens") or
            dexscreener_data.get("activeTokens") or
            0
        )
        
        # Extract trending tokens
        trending_tokens = (
            birdeye_data.get("trendingTokens", []) or
            dexscreener_data.get("trendingTokens", [])
        )
        
        # Calculate market sentiment (simplified)
        market_sentiment = "neutral"
        if total_volume > 1000000:  # $1M+ volume
            market_sentiment = "bullish"
        elif total_volume < 100000:  # <$100K volume
            market_sentiment = "bearish"
        
        # Calculate volatility index (simplified)
        volatility_index = 0.5  # Placeholder
        
        return MarketData(
            total_volume=float(total_volume),
            total_liquidity=float(total_liquidity),
            active_tokens=int(active_tokens),
            trending_tokens=trending_tokens,
            market_sentiment=market_sentiment,
            volatility_index=float(volatility_index),
            last_updated=datetime.now()
        )
    
    def _check_rate_limit(self, api: str) -> bool:
        """Check if API call is within rate limits"""
        current_time = time.time()
        rate_limit = self.rate_limits[api]
        
        # Reset counter if window has passed
        if current_time - rate_limit["last_reset"] > 60:  # 1 minute window
            rate_limit["calls"] = 0
            rate_limit["last_reset"] = current_time
        
        # Check if we're under the limit
        return rate_limit["calls"] < rate_limit["limit"]
    
    def _deduplicate_tokens(self, tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate tokens based on address"""
        seen = set()
        unique_tokens = []
        
        for token in tokens:
            address = token.get("address")
            if address and address not in seen:
                seen.add(address)
                unique_tokens.append(token)
        
        return unique_tokens
    
    def _cleanup_cache(self):
        """Clean up cache if it's too large"""
        if len(self.token_cache) > self.max_cache_size:
            # Remove oldest entries
            sorted_cache = sorted(
                self.token_cache.items(),
                key=lambda x: x[1].last_updated
            )
            
            # Keep only the newest entries
            self.token_cache = dict(sorted_cache[-self.max_cache_size:])
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        cache_hit_rate = self.cache_hit_count / max(self.api_call_count, 1)
        
        return {
            "api_call_count": self.api_call_count,
            "cache_hit_count": self.cache_hit_count,
            "cache_hit_rate": cache_hit_rate,
            "error_count": self.error_count,
            "cache_size": len(self.token_cache),
            "rate_limits": self.rate_limits
        }
    
    async def close(self):
        """Close the market data manager"""
        if self.session:
            await self.session.close()
        
        logger.info("Market data manager closed") 