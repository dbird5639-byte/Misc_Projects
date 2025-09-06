"""
Sniper Bot for Solana Trading Bot 2025

This bot monitors new token launches and executes trades based on predefined criteria.
"""

import asyncio
import aiohttp
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json

from bots.base_bot import BaseBot, TokenInfo, TradeSignal
from data.market_data import MarketDataManager
from data.token_analyzer import TokenAnalyzer
from utils.browser_automation import BrowserAutomation

class SniperBot(BaseBot):
    """Bot that snipes new token launches"""
    
    def __init__(self, config):
        super().__init__(config, "Sniper")
        self.market_data = MarketDataManager(config)
        self.token_analyzer = TokenAnalyzer(config)
        self.browser_automation = BrowserAutomation()
        
        # Sniper-specific state
        self.scanned_tokens = set()
        self.potential_tokens = []
        self.last_scan_time = None
        
        # Token launch detection
        self.launch_threshold = timedelta(minutes=5)  # Consider tokens launched in last 5 minutes
        self.volume_threshold = config.sniper_config.min_volume
        self.liquidity_threshold = config.sniper_config.min_liquidity
    
    async def run(self):
        """Main sniper bot loop"""
        self.logger.info("Starting sniper bot main loop...")
        
        while self.is_running:
            try:
                # Scan for new tokens
                await self.scan_new_tokens()
                
                # Analyze potential tokens
                await self.analyze_potential_tokens()
                
                # Wait for next scan
                await asyncio.sleep(self.config.sniper_config.scan_interval)
                
            except Exception as e:
                await self.handle_error(e, "main loop")
                await asyncio.sleep(10)  # Wait before retrying
    
    async def scan_new_tokens(self):
        """Scan for new token launches"""
        try:
            self.logger.debug("Scanning for new tokens...")
            
            # Get new tokens from multiple sources
            new_tokens = await self.get_new_tokens_from_sources()
            
            # Filter and validate tokens
            valid_tokens = []
            for token in new_tokens:
                if await self.validate_new_token(token):
                    valid_tokens.append(token)
                    self.scanned_tokens.add(token.address)
            
            # Add to potential tokens list
            self.potential_tokens.extend(valid_tokens)
            
            # Keep only recent tokens
            cutoff_time = datetime.now() - timedelta(hours=1)
            self.potential_tokens = [
                t for t in self.potential_tokens 
                if t.launch_time > cutoff_time
            ]
            
            self.stats["tokens_scanned"] += len(new_tokens)
            self.logger.info(f"Scanned {len(new_tokens)} tokens, found {len(valid_tokens)} valid")
            
        except Exception as e:
            await self.handle_error(e, "scan_new_tokens")
    
    async def get_new_tokens_from_sources(self) -> List[TokenInfo]:
        """Get new tokens from various data sources"""
        tokens = []
        
        try:
            # Source 1: BirdEye API
            birdeye_tokens = await self.get_birdeye_new_tokens()
            tokens.extend(birdeye_tokens)
            
            # Source 2: DexScreener API
            dexscreener_tokens = await self.get_dexscreener_new_tokens()
            tokens.extend(dexscreener_tokens)
            
            # Source 3: Jupiter API
            jupiter_tokens = await self.get_jupiter_new_tokens()
            tokens.extend(jupiter_tokens)
            
            # Remove duplicates
            unique_tokens = {}
            for token in tokens:
                if token.address not in unique_tokens:
                    unique_tokens[token.address] = token
            
            return list(unique_tokens.values())
            
        except Exception as e:
            self.logger.error(f"Error getting new tokens: {e}")
            return []
    
    async def get_birdeye_new_tokens(self) -> List[TokenInfo]:
        """Get new tokens from BirdEye API"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"X-API-KEY": self.config.api_config.birdeye_api}
                url = "https://public-api.birdeye.so/public/tokenlist"
                
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        tokens = []
                        
                        for token_data in data.get("data", []):
                            # Check if token is new (launched recently)
                            launch_time = datetime.fromtimestamp(token_data.get("launchTime", 0))
                            if datetime.now() - launch_time < self.launch_threshold:
                                token = TokenInfo(
                                    address=token_data["address"],
                                    name=token_data.get("name", ""),
                                    symbol=token_data.get("symbol", ""),
                                    price=float(token_data.get("price", 0)),
                                    volume_24h=float(token_data.get("volume24h", 0)),
                                    liquidity=float(token_data.get("liquidity", 0)),
                                    market_cap=float(token_data.get("marketCap", 0)),
                                    launch_time=launch_time,
                                    dex=token_data.get("dex", ""),
                                    pair_address=token_data.get("pairAddress", "")
                                )
                                tokens.append(token)
                        
                        return tokens
                    
        except Exception as e:
            self.logger.error(f"Error getting BirdEye tokens: {e}")
        
        return []
    
    async def get_dexscreener_new_tokens(self) -> List[TokenInfo]:
        """Get new tokens from DexScreener API"""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.dexscreener.com/latest/dex/tokens/recent"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        tokens = []
                        
                        for pair_data in data.get("pairs", []):
                            # Check if token is new
                            created_at = datetime.fromtimestamp(pair_data.get("createdAt", 0) / 1000)
                            if datetime.now() - created_at < self.launch_threshold:
                                token = TokenInfo(
                                    address=pair_data["baseToken"]["address"],
                                    name=pair_data["baseToken"].get("name", ""),
                                    symbol=pair_data["baseToken"].get("symbol", ""),
                                    price=float(pair_data.get("priceUsd", 0)),
                                    volume_24h=float(pair_data.get("volume24h", 0)),
                                    liquidity=float(pair_data.get("liquidity", {}).get("usd", 0)),
                                    market_cap=float(pair_data.get("marketCap", 0)),
                                    launch_time=created_at,
                                    dex=pair_data.get("dexId", ""),
                                    pair_address=pair_data["pairAddress"]
                                )
                                tokens.append(token)
                        
                        return tokens
                    
        except Exception as e:
            self.logger.error(f"Error getting DexScreener tokens: {e}")
        
        return []
    
    async def get_jupiter_new_tokens(self) -> List[TokenInfo]:
        """Get new tokens from Jupiter API"""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://token.jup.ag/all"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        tokens = []
                        
                        # Jupiter doesn't provide launch times, so we'll use a different approach
                        # This is a simplified version - in practice you'd need to track token history
                        
                        return tokens
                    
        except Exception as e:
            self.logger.error(f"Error getting Jupiter tokens: {e}")
        
        return []
    
    async def validate_new_token(self, token: TokenInfo) -> bool:
        """Validate if a new token meets sniper criteria"""
        try:
            # Basic validation from base class
            if not await super().validate_token(token):
                return False
            
            # Check if we've already scanned this token
            if token.address in self.scanned_tokens:
                return False
            
            # Check launch time
            if datetime.now() - token.launch_time > self.launch_threshold:
                return False
            
            # Additional sniper-specific validation
            if not await self.token_analyzer.is_safe_token(token):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating new token {token.address}: {e}")
            return False
    
    async def analyze_potential_tokens(self):
        """Analyze potential tokens for trading signals"""
        try:
            for token in self.potential_tokens[:]:  # Copy list to avoid modification during iteration
                # Get updated token data
                updated_token = await self.market_data.get_token_data(token.address)
                if not updated_token:
                    continue
                
                # Analyze token
                analysis = await self.token_analyzer.analyze_token(updated_token)
                
                # Generate signal if conditions are met
                if analysis["should_trade"]:
                    signal = await self.generate_signal(updated_token, analysis)
                    if signal:
                        await self.process_signal(signal)
                
                # Remove token if it's too old
                if datetime.now() - token.launch_time > timedelta(hours=1):
                    self.potential_tokens.remove(token)
                    
        except Exception as e:
            await self.handle_error(e, "analyze_potential_tokens")
    
    async def generate_signal(self, token: TokenInfo, analysis: Dict[str, Any]) -> Optional[TradeSignal]:
        """Generate trading signal based on analysis"""
        try:
            confidence = analysis.get("confidence", 0.5)
            
            # Only generate signal if confidence is high enough
            if confidence < 0.7:
                return None
            
            # Calculate position size
            position_size = await self.calculate_position_size(token, confidence)
            if position_size <= 0:
                return None
            
            signal = TradeSignal(
                token_address=token.address,
                action="buy",
                price=token.price,
                quantity=position_size,
                confidence=confidence,
                source="sniper_bot",
                timestamp=datetime.now(),
                metadata={
                    "analysis": analysis,
                    "token_info": {
                        "name": token.name,
                        "symbol": token.symbol,
                        "volume_24h": token.volume_24h,
                        "liquidity": token.liquidity
                    }
                }
            )
            
            self.stats["signals_generated"] += 1
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {token.address}: {e}")
            return None
    
    async def process_signal(self, signal: TradeSignal) -> bool:
        """Process a trading signal"""
        try:
            self.logger.info(f"Processing signal: {signal.action} {signal.quantity} of {signal.token_address}")
            
            # Open browser tab for manual review (if enabled)
            if self.config.notification_config.browser_notifications:
                await self.browser_automation.open_token_page(signal.token_address)
            
            # Send notification
            await self.notification_manager.send_notification(
                f"🎯 Sniper Signal: {signal.action.upper()} {signal.metadata['token_info']['symbol']}\n"
                f"Price: ${signal.price:.6f}\n"
                f"Confidence: {signal.confidence:.1%}\n"
                f"Volume: ${signal.metadata['token_info']['volume_24h']:,.0f}"
            )
            
            # Execute trade if auto-trading is enabled
            if self.config.sniper_config.auto_trade:
                return await self.execute_trade(signal)
            else:
                self.logger.info("Auto-trading disabled, signal logged for manual review")
                return True
                
        except Exception as e:
            await self.handle_error(e, "process_signal")
            return False
    
    async def get_sniper_stats(self) -> Dict[str, Any]:
        """Get sniper-specific statistics"""
        return {
            **self.get_status(),
            "scanned_tokens_count": len(self.scanned_tokens),
            "potential_tokens_count": len(self.potential_tokens),
            "last_scan_time": self.last_scan_time.isoformat() if self.last_scan_time else None,
            "launch_threshold_minutes": self.launch_threshold.total_seconds() / 60
        } 