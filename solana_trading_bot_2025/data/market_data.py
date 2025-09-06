"""
Market Data Manager for Solana Trading Bot 2025

Handles data fetching from various sources including BirdEye, DexScreener, and Jupiter.
"""

import asyncio
import aiohttp
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
import time

from bots.base_bot import TokenInfo
from config.settings import Config

class MarketDataManager:
    """Manages market data from multiple sources"""
    
    def __init__(self, config: Config):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache = {}
        self.cache_ttl = 30  # seconds
        
        # Rate limiting
        self.rate_limits = {
            "birdeye": {"calls": 0, "last_reset": time.time(), "limit": 100},
            "dexscreener": {"calls": 0, "last_reset": time.time(), "limit": 100},
            "jupiter": {"calls": 0, "last_reset": time.time(), "limit": 100}
        }
    
    async def _ensure_session(self):
        """Ensure session is initialized"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        assert self.session is not None  # Help linter understand session is not None
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def get_token_data(self, token_address: str) -> Optional[TokenInfo]:
        """Get comprehensive token data from multiple sources"""
        try:
            # Check cache first
            cache_key = f"token_{token_address}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    return cached_data
            
            # Fetch from multiple sources
            token_data = await self._fetch_token_data(token_address)
            
            if token_data:
                # Cache the result
                self.cache[cache_key] = (token_data, time.time())
                return token_data
            
            return None
            
        except Exception as e:
            print(f"Error getting token data for {token_address}: {e}")
            return None
    
    async def _fetch_token_data(self, token_address: str) -> Optional[TokenInfo]:
        """Fetch token data from multiple sources"""
        try:
            # Try BirdEye first
            birdeye_data = await self._get_birdeye_token_data(token_address)
            if birdeye_data:
                return birdeye_data
            
            # Fallback to DexScreener
            dexscreener_data = await self._get_dexscreener_token_data(token_address)
            if dexscreener_data:
                return dexscreener_data
            
            # Fallback to Jupiter
            jupiter_data = await self._get_jupiter_token_data(token_address)
            if jupiter_data:
                return jupiter_data
            
            return None
            
        except Exception as e:
            print(f"Error fetching token data: {e}")
            return None
    
    async def _get_birdeye_token_data(self, token_address: str) -> Optional[TokenInfo]:
        """Get token data from BirdEye API"""
        try:
            await self._ensure_session()
            
            if not self._check_rate_limit("birdeye"):
                return None
            
            headers = {"X-API-KEY": self.config.api_config.birdeye_api}
            url = f"https://public-api.birdeye.so/public/token_list"
            params = {"address": token_address}
            
            if self.session is None:
                await self._ensure_session()
            assert self.session is not None
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("success") and data.get("data"):
                        token_data = data["data"][0]
                        
                        return TokenInfo(
                            address=token_data["address"],
                            name=token_data.get("name", ""),
                            symbol=token_data.get("symbol", ""),
                            price=float(token_data.get("price", 0)),
                            volume_24h=float(token_data.get("volume24h", 0)),
                            liquidity=float(token_data.get("liquidity", 0)),
                            market_cap=float(token_data.get("marketCap", 0)),
                            launch_time=datetime.fromtimestamp(token_data.get("launchTime", 0)),
                            dex=token_data.get("dex", ""),
                            pair_address=token_data.get("pairAddress", "")
                        )
            
            return None
            
        except Exception as e:
            print(f"Error getting BirdEye token data: {e}")
            return None
    
    async def _get_dexscreener_token_data(self, token_address: str) -> Optional[TokenInfo]:
        """Get token data from DexScreener API"""
        try:
            await self._ensure_session()
            
            if not self._check_rate_limit("dexscreener"):
                return None
            
            url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
            
            if self.session is None:
                await self._ensure_session()
            assert self.session is not None
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    pairs = data.get("pairs", [])
                    
                    if pairs:
                        pair_data = pairs[0]  # Get first pair
                        
                        return TokenInfo(
                            address=pair_data["baseToken"]["address"],
                            name=pair_data["baseToken"].get("name", ""),
                            symbol=pair_data["baseToken"].get("symbol", ""),
                            price=float(pair_data.get("priceUsd", 0)),
                            volume_24h=float(pair_data.get("volume24h", 0)),
                            liquidity=float(pair_data.get("liquidity", {}).get("usd", 0)),
                            market_cap=float(pair_data.get("marketCap", 0)),
                            launch_time=datetime.fromtimestamp(pair_data.get("createdAt", 0) / 1000),
                            dex=pair_data.get("dexId", ""),
                            pair_address=pair_data["pairAddress"]
                        )
            
            return None
            
        except Exception as e:
            print(f"Error getting DexScreener token data: {e}")
            return None
    
    async def _get_jupiter_token_data(self, token_address: str) -> Optional[TokenInfo]:
        """Get token data from Jupiter API"""
        try:
            if not self._check_rate_limit("jupiter"):
                return None
            
            # Jupiter doesn't provide comprehensive token data
            # This would need to be implemented differently or use alternative sources
            return None
            
        except Exception as e:
            print(f"Error getting Jupiter token data: {e}")
            return None
    
    async def get_token_price(self, token_address: str) -> Optional[float]:
        """Get current token price"""
        try:
            token_data = await self.get_token_data(token_address)
            return token_data.price if token_data else None
            
        except Exception as e:
            print(f"Error getting token price: {e}")
            return None
    
    async def get_token_volume(self, token_address: str) -> Optional[float]:
        """Get 24h trading volume for token"""
        try:
            token_data = await self.get_token_data(token_address)
            return token_data.volume_24h if token_data else None
            
        except Exception as e:
            print(f"Error getting token volume: {e}")
            return None
    
    async def get_token_liquidity(self, token_address: str) -> Optional[float]:
        """Get token liquidity"""
        try:
            token_data = await self.get_token_data(token_address)
            return token_data.liquidity if token_data else None
            
        except Exception as e:
            print(f"Error getting token liquidity: {e}")
            return None
    
    async def get_market_data_batch(self, token_addresses: List[str]) -> Dict[str, TokenInfo]:
        """Get market data for multiple tokens efficiently"""
        try:
            results = {}
            tasks = []
            
            for address in token_addresses:
                task = asyncio.create_task(self.get_token_data(address))
                tasks.append((address, task))
            
            for address, task in tasks:
                try:
                    token_data = await task
                    if token_data:
                        results[address] = token_data
                except Exception as e:
                    print(f"Error getting data for {address}: {e}")
            
            return results
            
        except Exception as e:
            print(f"Error getting batch market data: {e}")
            return {}
    
    async def get_trending_tokens(self, limit: int = 50) -> List[TokenInfo]:
        """Get trending tokens from various sources"""
        try:
            trending_tokens = []
            
            # Get from BirdEye
            birdeye_trending = await self._get_birdeye_trending(limit // 2)
            trending_tokens.extend(birdeye_trending)
            
            # Get from DexScreener
            dexscreener_trending = await self._get_dexscreener_trending(limit // 2)
            trending_tokens.extend(dexscreener_trending)
            
            # Remove duplicates and sort by volume
            unique_tokens = {}
            for token in trending_tokens:
                if token.address not in unique_tokens:
                    unique_tokens[token.address] = token
            
            sorted_tokens = sorted(
                unique_tokens.values(),
                key=lambda x: x.volume_24h,
                reverse=True
            )
            
            return sorted_tokens[:limit]
            
        except Exception as e:
            print(f"Error getting trending tokens: {e}")
            return []
    
    async def _get_birdeye_trending(self, limit: int) -> List[TokenInfo]:
        """Get trending tokens from BirdEye"""
        try:
            await self._ensure_session()
            
            if not self._check_rate_limit("birdeye"):
                return []
            
            headers = {"X-API-KEY": self.config.api_config.birdeye_api}
            url = "https://public-api.birdeye.so/public/tokenlist"
            params = {"sort_by": "volume", "sort_type": "desc", "limit": limit}
            
            if self.session is None:
                await self._ensure_session()
            assert self.session is not None
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    tokens = []
                    
                    for token_data in data.get("data", []):
                        token = TokenInfo(
                            address=token_data["address"],
                            name=token_data.get("name", ""),
                            symbol=token_data.get("symbol", ""),
                            price=float(token_data.get("price", 0)),
                            volume_24h=float(token_data.get("volume24h", 0)),
                            liquidity=float(token_data.get("liquidity", 0)),
                            market_cap=float(token_data.get("marketCap", 0)),
                            launch_time=datetime.fromtimestamp(token_data.get("launchTime", 0)),
                            dex=token_data.get("dex", ""),
                            pair_address=token_data.get("pairAddress", "")
                        )
                        tokens.append(token)
                    
                    return tokens
            
            return []
            
        except Exception as e:
            print(f"Error getting BirdEye trending: {e}")
            return []
    
    async def _get_dexscreener_trending(self, limit: int) -> List[TokenInfo]:
        """Get trending tokens from DexScreener"""
        try:
            await self._ensure_session()
            
            if not self._check_rate_limit("dexscreener"):
                return []
            
            url = "https://api.dexscreener.com/latest/dex/tokens/trending"
            
            if self.session is None:
                await self._ensure_session()
            assert self.session is not None
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    tokens = []
                    
                    for pair_data in data.get("pairs", [])[:limit]:
                        token = TokenInfo(
                            address=pair_data["baseToken"]["address"],
                            name=pair_data["baseToken"].get("name", ""),
                            symbol=pair_data["baseToken"].get("symbol", ""),
                            price=float(pair_data.get("priceUsd", 0)),
                            volume_24h=float(pair_data.get("volume24h", 0)),
                            liquidity=float(pair_data.get("liquidity", {}).get("usd", 0)),
                            market_cap=float(pair_data.get("marketCap", 0)),
                            launch_time=datetime.fromtimestamp(pair_data.get("createdAt", 0) / 1000),
                            dex=pair_data.get("dexId", ""),
                            pair_address=pair_data["pairAddress"]
                        )
                        tokens.append(token)
                    
                    return tokens
            
            return []
            
        except Exception as e:
            print(f"Error getting DexScreener trending: {e}")
            return []
    
    def _check_rate_limit(self, source: str) -> bool:
        """Check and update rate limits"""
        try:
            current_time = time.time()
            rate_limit = self.rate_limits[source]
            
            # Reset counter if needed
            if current_time - rate_limit["last_reset"] > 60:  # Reset every minute
                rate_limit["calls"] = 0
                rate_limit["last_reset"] = current_time
            
            # Check if we're within limits
            if rate_limit["calls"] >= rate_limit["limit"]:
                return False
            
            # Increment counter
            rate_limit["calls"] += 1
            return True
            
        except Exception as e:
            print(f"Error checking rate limit: {e}")
            return False
    
    def clear_cache(self):
        """Clear the data cache"""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.cache),
            "rate_limits": self.rate_limits
        } 