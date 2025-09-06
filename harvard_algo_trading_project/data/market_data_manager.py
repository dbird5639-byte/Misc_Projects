"""
Market Data Manager

Handles real-time and historical market data from multiple sources
including Alpaca, Yahoo Finance, and other providers.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import URL
from alpaca_trade_api.rest import TimeFrame
from alpaca_trade_api.rest import TimeFrameUnit
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import time
import threading
from dataclasses import dataclass
import sqlite3
import json

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    source: str

class DataSource:
    """Abstract base class for data sources"""
    
    def __init__(self, name: str):
        self.name = name
    
    def get_historical_data(self, symbol: str, start_date: str, 
                           end_date: str, interval: str = "1d") -> pd.DataFrame:
        """Get historical data"""
        raise NotImplementedError
    
    def get_realtime_data(self, symbol: str) -> Optional[MarketData]:
        """Get real-time data"""
        raise NotImplementedError

class YahooFinanceSource(DataSource):
    """Yahoo Finance data source"""
    
    def __init__(self):
        super().__init__("yahoo_finance")
    
    def get_historical_data(self, symbol: str, start_date: str, 
                           end_date: str, interval: str = "1d") -> pd.DataFrame:
        """Get historical data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                return pd.DataFrame()
            
            # Rename columns to standard format
            data.columns = [col.lower() for col in data.columns]
            
            return data
            
        except Exception as e:
            print(f"Error getting Yahoo Finance data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_realtime_data(self, symbol: str) -> Optional[MarketData]:
        """Get real-time data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if 'regularMarketPrice' not in info or info['regularMarketPrice'] is None:
                return None
            
            return MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                open=info.get('regularMarketOpen', 0),
                high=info.get('dayHigh', 0),
                low=info.get('dayLow', 0),
                close=info['regularMarketPrice'],
                volume=info.get('volume', 0),
                source=self.name
            )
            
        except Exception as e:
            print(f"Error getting real-time Yahoo Finance data for {symbol}: {e}")
            return None

class AlpacaSource(DataSource):
    """Alpaca data source"""
    
    def __init__(self, api_key: str, secret_key: str, base_url: str):
        super().__init__("alpaca")
        self.api = tradeapi.REST(api_key, secret_key, URL(base_url), api_version='v2')
    
    def get_historical_data(self, symbol: str, start_date: str, 
                           end_date: str, interval: str = "1d") -> pd.DataFrame:
        """Get historical data from Alpaca"""
        try:
            # Convert interval format
            alpaca_interval = self._convert_interval(interval)
            
            bars = self.api.get_bars(
                symbol,
                alpaca_interval,
                start=start_date,
                end=end_date
            )
            
            if not bars:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for bar in bars:
                data.append({
                    'timestamp': bar.t,
                    'open': bar.o,
                    'high': bar.h,
                    'low': bar.l,
                    'close': bar.c,
                    'volume': bar.v
                })
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error getting Alpaca data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_realtime_data(self, symbol: str) -> Optional[MarketData]:
        """Get real-time data from Alpaca"""
        try:
            # Get latest bar
            bars = self.api.get_bars(symbol, TimeFrame(1, TimeFrameUnit.Minute), limit=1)
            
            if not bars:
                return None
            
            bar = bars[0]
            
            return MarketData(
                symbol=symbol,
                timestamp=bar.t,
                open=bar.o,
                high=bar.h,
                low=bar.l,
                close=bar.c,
                volume=bar.v,
                source=self.name
            )
            
        except Exception as e:
            print(f"Error getting real-time Alpaca data for {symbol}: {e}")
            return None
    
    def _convert_interval(self, interval: str) -> TimeFrame:
        """Convert interval format for Alpaca"""
        interval_map = {
            "1m": TimeFrame(1, TimeFrameUnit.Minute),
            "5m": TimeFrame(5, TimeFrameUnit.Minute),
            "15m": TimeFrame(15, TimeFrameUnit.Minute),
            "30m": TimeFrame(30, TimeFrameUnit.Minute),
            "1h": TimeFrame(1, TimeFrameUnit.Hour),
            "1d": TimeFrame(1, TimeFrameUnit.Day)
        }
        return interval_map.get(interval, TimeFrame(1, TimeFrameUnit.Day))

class MarketDataManager:
    """Main market data manager"""
    
    def __init__(self, db_path: str = "market_data.db"):
        self.db_path = db_path
        self.data_sources = {}
        self.subscriptions = {}
        self.realtime_data = {}
        self.data_cache = {}
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                source TEXT,
                UNIQUE(symbol, timestamp, source)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                config TEXT,
                active BOOLEAN DEFAULT 1
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_data_source(self, source: DataSource):
        """Add a data source"""
        self.data_sources[source.name] = source
        print(f"Added data source: {source.name}")
    
    def get_historical_data(self, symbol: str, start_date: str, 
                           end_date: str, interval: str = "1d",
                           source: str = "yahoo_finance") -> pd.DataFrame:
        """Get historical data from specified source"""
        if source not in self.data_sources:
            print(f"Data source {source} not found")
            return pd.DataFrame()
        
        # Check cache first
        cache_key = f"{symbol}_{start_date}_{end_date}_{interval}_{source}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        # Get data from source
        data_source = self.data_sources[source]
        data = data_source.get_historical_data(symbol, start_date, end_date, interval)
        
        # Cache the data
        if not data.empty:
            self.data_cache[cache_key] = data
            self._save_to_database(symbol, data, source)
        
        return data
    
    def _save_to_database(self, symbol: str, data: pd.DataFrame, source: str):
        """Save data to SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Prepare data for insertion
            data_to_insert = []
            for timestamp, row in data.iterrows():
                data_to_insert.append((
                    symbol,
                    timestamp,
                    row.get('open', 0),
                    row.get('high', 0),
                    row.get('low', 0),
                    row.get('close', 0),
                    row.get('volume', 0),
                    source
                ))
            
            # Insert data
            cursor = conn.cursor()
            cursor.executemany('''
                INSERT OR REPLACE INTO market_data 
                (symbol, timestamp, open, high, low, close, volume, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', data_to_insert)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error saving data to database: {e}")
    
    def get_data_from_database(self, symbol: str, start_date: str, 
                              end_date: str) -> pd.DataFrame:
        """Get data from local database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT timestamp, open, high, low, close, volume, source
                FROM market_data
                WHERE symbol = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            '''
            
            df = pd.read_sql_query(query, conn, params=[symbol, start_date, end_date])
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            conn.close()
            return df
            
        except Exception as e:
            print(f"Error getting data from database: {e}")
            return pd.DataFrame()
    
    def subscribe_realtime_data(self, symbols: List[str], 
                               callback: Callable, source: str = "yahoo_finance"):
        """Subscribe to real-time data updates"""
        if source not in self.data_sources:
            print(f"Data source {source} not found")
            return
        
        # Store subscription
        for symbol in symbols:
            if symbol not in self.subscriptions:
                self.subscriptions[symbol] = []
            self.subscriptions[symbol].append({
                'callback': callback,
                'source': source
            })
        
        # Start real-time data collection if not already running
        if not hasattr(self, '_realtime_thread') or not self._realtime_thread.is_alive():
            self._start_realtime_collection()
    
    def _start_realtime_collection(self):
        """Start real-time data collection thread"""
        self._realtime_thread = threading.Thread(target=self._realtime_loop, daemon=True)
        self._realtime_thread.start()
    
    def _realtime_loop(self):
        """Real-time data collection loop"""
        while True:
            try:
                for symbol, subscriptions in self.subscriptions.items():
                    for subscription in subscriptions:
                        source_name = subscription['source']
                        callback = subscription['callback']
                        
                        if source_name in self.data_sources:
                            data_source = self.data_sources[source_name]
                            realtime_data = data_source.get_realtime_data(symbol)
                            
                            if realtime_data:
                                # Store latest data
                                self.realtime_data[symbol] = realtime_data
                                
                                # Call callback
                                try:
                                    callback(realtime_data)
                                except Exception as e:
                                    print(f"Error in real-time callback: {e}")
                
                # Wait before next update
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                print(f"Error in real-time loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def get_latest_data(self, symbol: str) -> Optional[MarketData]:
        """Get latest real-time data for a symbol"""
        return self.realtime_data.get(symbol)
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        if data.empty:
            return data
        
        # Simple Moving Averages
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        data['ema_12'] = data['close'].ewm(span=12).mean()
        data['ema_26'] = data['close'].ewm(span=26).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        
        # MACD
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # Volume indicators
        data['volume_sma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        return data
    
    def get_market_summary(self, symbols: List[str]) -> Dict[str, Any]:
        """Get market summary for multiple symbols"""
        summary = {}
        
        for symbol in symbols:
            latest_data = self.get_latest_data(symbol)
            if latest_data:
                summary[symbol] = {
                    'price': latest_data.close,
                    'change': 0.0,  # Would need previous price for change
                    'volume': latest_data.volume,
                    'timestamp': latest_data.timestamp.isoformat(),
                    'source': latest_data.source
                }
        
        return summary
    
    def cleanup_cache(self, max_age_hours: int = 24):
        """Clean up old cached data"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        keys_to_remove = []
        for key in self.data_cache.keys():
            # Simple cleanup - remove old cache entries
            keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.data_cache[key]
        
        print(f"Cleaned up {len(keys_to_remove)} cache entries")

def main():
    """Main function for testing market data manager"""
    # Initialize data manager
    data_manager = MarketDataManager()
    
    # Add data sources
    yahoo_source = YahooFinanceSource()
    data_manager.add_data_source(yahoo_source)
    
    # Test historical data
    print("Getting historical data for AAPL...")
    data = data_manager.get_historical_data(
        "AAPL", 
        "2023-01-01", 
        "2023-12-31",
        source="yahoo_finance"
    )
    
    if not data.empty:
        print(f"Retrieved {len(data)} data points")
        print(data.head())
        
        # Calculate technical indicators
        data_with_indicators = data_manager.calculate_technical_indicators(data)
        print("\nData with indicators:")
        print(data_with_indicators.tail())
    
    # Test real-time data
    print("\nGetting real-time data...")
    realtime_data = yahoo_source.get_realtime_data("AAPL")
    if realtime_data:
        print(f"AAPL current price: ${realtime_data.close}")
        print(f"Volume: {realtime_data.volume}")
        print(f"Timestamp: {realtime_data.timestamp}")

if __name__ == "__main__":
    main() 