"""
Market Data Handler

Processes and manages real-time market data from Interactive Brokers.
"""

import time
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass

try:
    from loguru import logger  # type: ignore
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    last_price: float
    bid: float
    ask: float
    volume: int
    timestamp: float
    high: float
    low: float
    open: float

class MarketDataHandler:
    """
    Handles real-time market data processing and storage
    """
    
    def __init__(self):
        """Initialize market data handler"""
        self.data: Dict[str, MarketData] = {}
        self.subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()
        self._running = False
        self._update_thread: Optional[threading.Thread] = None
    
    def subscribe(self, symbol: str, callback: Callable):
        """
        Subscribe to market data updates for a symbol
        
        Args:
            symbol: Stock symbol
            callback: Function to call when data updates
        """
        with self._lock:
            if symbol not in self.subscribers:
                self.subscribers[symbol] = []
            self.subscribers[symbol].append(callback)
        
        logger.info(f"Subscribed to market data for {symbol}")
    
    def unsubscribe(self, symbol: str, callback: Callable):
        """
        Unsubscribe from market data updates
        
        Args:
            symbol: Stock symbol
            callback: Function to remove from subscribers
        """
        with self._lock:
            if symbol in self.subscribers:
                if callback in self.subscribers[symbol]:
                    self.subscribers[symbol].remove(callback)
        
        logger.info(f"Unsubscribed from market data for {symbol}")
    
    def update_data(self, symbol: str, data: Dict[str, Any]):
        """
        Update market data for a symbol
        
        Args:
            symbol: Stock symbol
            data: Market data dictionary
        """
        with self._lock:
            # Create or update market data
            if symbol not in self.data:
                self.data[symbol] = MarketData(
                    symbol=symbol,
                    last_price=data.get("last_price", 0.0),
                    bid=data.get("bid", 0.0),
                    ask=data.get("ask", 0.0),
                    volume=data.get("volume", 0),
                    timestamp=time.time(),
                    high=data.get("high", 0.0),
                    low=data.get("low", 0.0),
                    open=data.get("open", 0.0)
                )
            else:
                # Update existing data
                market_data = self.data[symbol]
                market_data.last_price = data.get("last_price", market_data.last_price)
                market_data.bid = data.get("bid", market_data.bid)
                market_data.ask = data.get("ask", market_data.ask)
                market_data.volume = data.get("volume", market_data.volume)
                market_data.timestamp = time.time()
                market_data.high = data.get("high", market_data.high)
                market_data.low = data.get("low", market_data.low)
                market_data.open = data.get("open", market_data.open)
            
            # Notify subscribers
            if symbol in self.subscribers:
                for callback in self.subscribers[symbol]:
                    try:
                        callback(symbol, self.data[symbol])
                    except Exception as e:
                        logger.error(f"Error in market data callback: {e}")
    
    def get_data(self, symbol: str) -> Optional[MarketData]:
        """
        Get market data for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Market data or None if not available
        """
        with self._lock:
            return self.data.get(symbol)
    
    def get_all_data(self) -> Dict[str, MarketData]:
        """Get all market data"""
        with self._lock:
            return self.data.copy()
    
    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Current price or None if not available
        """
        data = self.get_data(symbol)
        return data.last_price if data else None
    
    def get_bid_ask(self, symbol: str) -> Optional[tuple]:
        """
        Get bid/ask prices for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Tuple of (bid, ask) or None if not available
        """
        data = self.get_data(symbol)
        return (data.bid, data.ask) if data else None
    
    def get_spread(self, symbol: str) -> Optional[float]:
        """
        Get bid-ask spread for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Spread or None if not available
        """
        bid_ask = self.get_bid_ask(symbol)
        if bid_ask and bid_ask[0] > 0 and bid_ask[1] > 0:
            return bid_ask[1] - bid_ask[0]
        return None
    
    def is_data_fresh(self, symbol: str, max_age: float = 60.0) -> bool:
        """
        Check if market data is fresh (within max_age seconds)
        
        Args:
            symbol: Stock symbol
            max_age: Maximum age in seconds
            
        Returns:
            True if data is fresh
        """
        data = self.get_data(symbol)
        if not data:
            return False
        
        return (time.time() - data.timestamp) <= max_age
    
    def get_symbols(self) -> List[str]:
        """Get list of all symbols with data"""
        with self._lock:
            return list(self.data.keys())
    
    def start(self):
        """Start market data processing"""
        if self._running:
            return
        
        self._running = True
        self._update_thread = threading.Thread(target=self._update_loop)
        self._update_thread.daemon = True
        self._update_thread.start()
        
        logger.info("Market data handler started")
    
    def stop(self):
        """Stop market data processing"""
        self._running = False
        
        if self._update_thread:
            self._update_thread.join(timeout=5)
        
        logger.info("Market data handler stopped")
    
    def _update_loop(self):
        """Main update loop for market data processing"""
        while self._running:
            try:
                # Process any pending updates
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in market data update loop: {e}")
                time.sleep(1)
    
    def calculate_indicators(self, symbol: str) -> Dict[str, float]:
        """
        Calculate technical indicators for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary of calculated indicators
        """
        data = self.get_data(symbol)
        if not data:
            return {}
        
        indicators = {}
        
        # Simple indicators (in real implementation, use proper technical analysis library)
        if data.high > 0 and data.low > 0:
            indicators["range"] = data.high - data.low
            indicators["range_percent"] = (data.high - data.low) / data.low * 100
        
        if data.open > 0:
            indicators["change"] = data.last_price - data.open
            indicators["change_percent"] = (data.last_price - data.open) / data.open * 100
        
        spread = self.get_spread(symbol)
        if spread:
            indicators["spread"] = spread
            indicators["spread_percent"] = spread / data.last_price * 100
        
        return indicators
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all market data"""
        with self._lock:
            summary = {
                "total_symbols": len(self.data),
                "symbols": list(self.data.keys()),
                "fresh_data_count": 0,
                "total_spread": 0.0,
                "avg_price": 0.0
            }
            
            total_price = 0.0
            total_spread = 0.0
            fresh_count = 0
            
            for symbol, data in self.data.items():
                total_price += data.last_price
                
                spread = self.get_spread(symbol)
                if spread:
                    total_spread += spread
                
                if self.is_data_fresh(symbol):
                    fresh_count += 1
            
            if summary["total_symbols"] > 0:
                summary["avg_price"] = total_price / summary["total_symbols"]
                summary["avg_spread"] = total_spread / summary["total_symbols"]
            
            summary["fresh_data_count"] = fresh_count
            
            return summary 