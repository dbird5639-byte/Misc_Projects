"""
Browser Automation for Solana Trading Bot 2025

Handles automated browser operations for token monitoring and analysis.
"""

import asyncio
import subprocess
import webbrowser
from typing import List, Optional, Dict, Any
from pathlib import Path
import time
import json

class BrowserAutomation:
    """Handles browser automation tasks"""
    
    def __init__(self):
        self.browser_processes = []
        self.opened_tabs = []
        self.max_tabs = 10  # Maximum number of tabs to keep open
        
        # Browser configuration
        self.browser_config = {
            "default_browser": "chrome",
            "headless": False,
            "window_size": "1920x1080"
        }
    
    async def open_token_page(self, token_address: str, dex: str = "raydium") -> bool:
        """Open a token page in browser"""
        try:
            # Generate URL based on DEX
            url = self._generate_token_url(token_address, dex)
            
            # Open in browser
            success = await self._open_url(url)
            
            if success:
                self.opened_tabs.append({
                    "url": url,
                    "token_address": token_address,
                    "opened_at": time.time()
                })
                
                # Clean up old tabs if needed
                await self._cleanup_old_tabs()
            
            return success
            
        except Exception as e:
            print(f"Error opening token page: {e}")
            return False
    
    def _generate_token_url(self, token_address: str, dex: str) -> str:
        """Generate URL for token page based on DEX"""
        try:
            if dex.lower() == "raydium":
                return f"https://raydium.io/swap/?inputCurrency=sol&outputCurrency={token_address}"
            elif dex.lower() == "jupiter":
                return f"https://jup.ag/swap/SOL-{token_address}"
            elif dex.lower() == "birdeye":
                return f"https://birdeye.so/token/{token_address}"
            elif dex.lower() == "dexscreener":
                return f"https://dexscreener.com/solana/{token_address}"
            elif dex.lower() == "solscan":
                return f"https://solscan.io/token/{token_address}"
            else:
                # Default to Birdeye
                return f"https://birdeye.so/token/{token_address}"
                
        except Exception as e:
            print(f"Error generating token URL: {e}")
            return f"https://birdeye.so/token/{token_address}"
    
    async def _open_url(self, url: str) -> bool:
        """Open URL in browser"""
        try:
            # Try to open with default browser
            webbrowser.open(url)
            return True
            
        except Exception as e:
            print(f"Error opening URL: {e}")
            return False
    
    async def open_multiple_tokens(self, token_addresses: List[str], dex: str = "raydium") -> List[bool]:
        """Open multiple token pages"""
        try:
            results = []
            
            for address in token_addresses:
                result = await self.open_token_page(address, dex)
                results.append(result)
                
                # Small delay between openings
                await asyncio.sleep(1)
            
            return results
            
        except Exception as e:
            print(f"Error opening multiple tokens: {e}")
            return [False] * len(token_addresses)
    
    async def open_dex_pages(self, token_address: str) -> Dict[str, bool]:
        """Open token page on multiple DEXes"""
        try:
            dexes = ["raydium", "jupiter", "birdeye", "dexscreener"]
            results = {}
            
            for dex in dexes:
                url = self._generate_token_url(token_address, dex)
                success = await self._open_url(url)
                results[dex] = success
                
                # Small delay between openings
                await asyncio.sleep(0.5)
            
            return results
            
        except Exception as e:
            print(f"Error opening DEX pages: {e}")
            return {dex: False for dex in ["raydium", "jupiter", "birdeye", "dexscreener"]}
    
    async def open_portfolio_page(self, wallet_address: str) -> bool:
        """Open portfolio page for a wallet"""
        try:
            # Generate portfolio URL
            url = f"https://birdeye.so/portfolio/{wallet_address}"
            
            return await self._open_url(url)
            
        except Exception as e:
            print(f"Error opening portfolio page: {e}")
            return False
    
    async def open_chart_page(self, token_address: str, timeframe: str = "1h") -> bool:
        """Open chart page for a token"""
        try:
            # Generate chart URL
            url = f"https://birdeye.so/token/{token_address}?chart={timeframe}"
            
            return await self._open_url(url)
            
        except Exception as e:
            print(f"Error opening chart page: {e}")
            return False
    
    async def _cleanup_old_tabs(self):
        """Clean up old tabs to prevent browser overload"""
        try:
            current_time = time.time()
            max_age = 3600  # 1 hour
            
            # Remove old tabs from tracking
            self.opened_tabs = [
                tab for tab in self.opened_tabs
                if current_time - tab["opened_at"] < max_age
            ]
            
            # Limit number of tracked tabs
            if len(self.opened_tabs) > self.max_tabs:
                self.opened_tabs = self.opened_tabs[-self.max_tabs:]
                
        except Exception as e:
            print(f"Error cleaning up old tabs: {e}")
    
    async def close_all_tabs(self):
        """Close all opened tabs (placeholder - would need browser automation)"""
        try:
            # This would require browser automation tools like Selenium
            # For now, just clear the tracking
            self.opened_tabs.clear()
            
        except Exception as e:
            print(f"Error closing tabs: {e}")
    
    def get_opened_tabs(self) -> List[Dict[str, Any]]:
        """Get list of currently opened tabs"""
        return self.opened_tabs.copy()
    
    async def open_dashboard(self, port: int = 8000) -> bool:
        """Open the bot dashboard in browser"""
        try:
            url = f"http://localhost:{port}"
            return await self._open_url(url)
            
        except Exception as e:
            print(f"Error opening dashboard: {e}")
            return False
    
    async def open_telegram_bot(self, bot_username: str) -> bool:
        """Open Telegram bot in browser"""
        try:
            url = f"https://t.me/{bot_username}"
            return await self._open_url(url)
            
        except Exception as e:
            print(f"Error opening Telegram bot: {e}")
            return False
    
    async def open_discord_server(self, invite_code: str) -> bool:
        """Open Discord server in browser"""
        try:
            url = f"https://discord.gg/{invite_code}"
            return await self._open_url(url)
            
        except Exception as e:
            print(f"Error opening Discord server: {e}")
            return False
    
    def set_browser_config(self, config: Dict[str, Any]):
        """Update browser configuration"""
        try:
            self.browser_config.update(config)
            
        except Exception as e:
            print(f"Error setting browser config: {e}")
    
    def get_browser_config(self) -> Dict[str, Any]:
        """Get current browser configuration"""
        return self.browser_config.copy()
    
    async def open_analysis_pages(self, token_address: str) -> Dict[str, bool]:
        """Open comprehensive analysis pages for a token"""
        try:
            pages = {
                "birdeye": f"https://birdeye.so/token/{token_address}",
                "dexscreener": f"https://dexscreener.com/solana/{token_address}",
                "solscan": f"https://solscan.io/token/{token_address}",
                "raydium": f"https://raydium.io/swap/?inputCurrency=sol&outputCurrency={token_address}",
                "jupiter": f"https://jup.ag/swap/SOL-{token_address}"
            }
            
            results = {}
            
            for name, url in pages.items():
                success = await self._open_url(url)
                results[name] = success
                
                # Small delay between openings
                await asyncio.sleep(0.5)
            
            return results
            
        except Exception as e:
            print(f"Error opening analysis pages: {e}")
            return {name: False for name in pages.keys()}
    
    async def open_market_overview(self) -> bool:
        """Open market overview page"""
        try:
            url = "https://birdeye.so/trending"
            return await self._open_url(url)
            
        except Exception as e:
            print(f"Error opening market overview: {e}")
            return False
    
    async def open_new_tokens_page(self) -> bool:
        """Open new tokens page"""
        try:
            url = "https://birdeye.so/new-tokens"
            return await self._open_url(url)
            
        except Exception as e:
            print(f"Error opening new tokens page: {e}")
            return False
    
    def save_tab_history(self, file_path: str = "tab_history.json"):
        """Save tab history to file"""
        try:
            history_data = {
                "timestamp": time.time(),
                "tabs": self.opened_tabs,
                "config": self.browser_config
            }
            
            with open(file_path, 'w') as f:
                json.dump(history_data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving tab history: {e}")
    
    def load_tab_history(self, file_path: str = "tab_history.json"):
        """Load tab history from file"""
        try:
            if Path(file_path).exists():
                with open(file_path, 'r') as f:
                    history_data = json.load(f)
                
                self.opened_tabs = history_data.get("tabs", [])
                self.browser_config.update(history_data.get("config", {}))
                
        except Exception as e:
            print(f"Error loading tab history: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get browser automation statistics"""
        return {
            "opened_tabs_count": len(self.opened_tabs),
            "max_tabs": self.max_tabs,
            "browser_config": self.browser_config,
            "recent_tabs": self.opened_tabs[-5:] if self.opened_tabs else []
        } 