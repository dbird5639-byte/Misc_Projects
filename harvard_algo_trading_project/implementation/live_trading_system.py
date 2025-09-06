"""
Live Trading System - RBI System Phase 3

This module implements the Implement phase of the RBI system,
deploying validated strategies for live trading.
"""

import time
import threading
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import URL
from alpaca_trade_api.rest import TimeFrame
from alpaca_trade_api.rest import TimeFrameUnit
from abc import ABC, abstractmethod

@dataclass
class Position:
    """Data class for trading positions"""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    unrealized_pnl: float
    entry_time: datetime

@dataclass
class Order:
    """Data class for trading orders"""
    id: str
    symbol: str
    quantity: int
    side: str  # "buy" or "sell"
    type: str  # "market" or "limit"
    status: str
    price: float
    created_at: datetime

class RiskManager:
    """Risk management system"""
    
    def __init__(self, max_position_size: float = 0.02, 
                 max_portfolio_risk: float = 0.06,
                 stop_loss_pct: float = 0.05):
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.stop_loss_pct = stop_loss_pct
    
    def calculate_position_size(self, price: float, account_value: float) -> int:
        """Calculate safe position size"""
        max_dollar_risk = account_value * self.max_position_size
        return int(max_dollar_risk / price)
    
    def should_stop_loss(self, entry_price: float, current_price: float, side: str) -> bool:
        """Check if stop loss should be triggered"""
        if side == "buy":
            return current_price <= entry_price * (1 - self.stop_loss_pct)
        else:
            return current_price >= entry_price * (1 + self.stop_loss_pct)
    
    def check_portfolio_risk(self, positions: List[Position], account_value: float) -> bool:
        """Check if portfolio risk is within limits"""
        total_exposure = sum(abs(pos.quantity * pos.current_price) for pos in positions)
        risk_ratio = total_exposure / account_value
        return risk_ratio <= self.max_portfolio_risk

class LiveStrategy(ABC):
    """Abstract base class for live trading strategies"""
    
    def __init__(self, name: str, symbols: List[str]):
        self.name = name
        self.symbols = symbols
        self.positions = {}
        self.signals = {}
    
    @abstractmethod
    def analyze_market(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Analyze market data and generate signals"""
        pass
    
    def update_positions(self, positions: Dict[str, Position]):
        """Update current positions"""
        self.positions = positions

class LiveMomentumStrategy(LiveStrategy):
    """Live momentum trading strategy"""
    
    def __init__(self, symbols: List[str], lookback_period: int = 20, threshold: float = 0.02):
        super().__init__("Live Momentum Strategy", symbols)
        self.lookback_period = lookback_period
        self.threshold = threshold
    
    def analyze_market(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Generate momentum signals"""
        signals = {}
        
        for symbol in self.symbols:
            if symbol in market_data and len(market_data[symbol]) >= self.lookback_period:
                data = market_data[symbol]
                current_price = data['close'].iloc[-1]
                past_price = data['close'].iloc[-self.lookback_period]
                
                momentum = (current_price - past_price) / past_price
                
                if momentum > self.threshold:
                    signals[symbol] = 1.0  # Strong buy
                elif momentum < -self.threshold:
                    signals[symbol] = -1.0  # Strong sell
                else:
                    signals[symbol] = 0.0  # Hold
        
        return signals

class LiveMeanReversionStrategy(LiveStrategy):
    """Live mean reversion strategy"""
    
    def __init__(self, symbols: List[str], lookback_period: int = 50, std_dev_threshold: float = 2.0):
        super().__init__("Live Mean Reversion Strategy", symbols)
        self.lookback_period = lookback_period
        self.std_dev_threshold = std_dev_threshold
    
    def analyze_market(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Generate mean reversion signals"""
        signals = {}
        
        for symbol in self.symbols:
            if symbol in market_data and len(market_data[symbol]) >= self.lookback_period:
                data = market_data[symbol]
                current_price = data['close'].iloc[-1]
                
                # Calculate rolling statistics
                recent_prices = data['close'].tail(self.lookback_period)
                mean_price = recent_prices.mean()
                std_price = recent_prices.std()
                
                if std_price > 0:
                    z_score = (current_price - mean_price) / std_price
                    
                    if z_score < -self.std_dev_threshold:
                        signals[symbol] = 1.0  # Oversold - buy
                    elif z_score > self.std_dev_threshold:
                        signals[symbol] = -1.0  # Overbought - sell
                    else:
                        signals[symbol] = 0.0  # Hold
                else:
                    signals[symbol] = 0.0
        
        return signals

class LiveTradingSystem:
    """Main live trading system"""
    
    def __init__(self, api_key: str, secret_key: str, base_url: str = "https://paper-api.alpaca.markets"):
        # Initialize Alpaca API
        self.api = tradeapi.REST(api_key, secret_key, URL(base_url), api_version='v2')
        
        # Initialize components
        self.risk_manager = RiskManager()
        self.strategies = {}
        self.positions = {}
        self.orders = []
        self.running = False
        
        # Market data storage
        self.market_data = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    def add_strategy(self, strategy: LiveStrategy):
        """Add a trading strategy"""
        self.strategies[strategy.name] = strategy
        logging.info(f"Added strategy: {strategy.name}")
    
    def get_market_data(self, symbols: List[str], lookback_days: int = 100) -> Dict[str, pd.DataFrame]:
        """Get market data for symbols"""
        market_data = {}
        
        for symbol in symbols:
            try:
                # Get historical data
                bars = self.api.get_bars(symbol, TimeFrame(1, TimeFrameUnit.Day), limit=lookback_days)
                
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
                
                market_data[symbol] = df
                logging.debug(f"Retrieved data for {symbol}: {len(df)} bars")
                
            except Exception as e:
                logging.error(f"Error getting data for {symbol}: {e}")
        
        return market_data
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            account = self.api.get_account()
            return {
                'cash': float(str(account.cash)) if account.cash is not None else 0.0,
                'buying_power': float(str(account.buying_power)) if account.buying_power is not None else 0.0,
                'portfolio_value': float(str(account.portfolio_value)) if account.portfolio_value is not None else 0.0,
                'equity': float(str(account.equity)) if account.equity is not None else 0.0,
                'daytrade_count': int(str(account.daytrade_count)) if account.daytrade_count is not None else 0
            }
        except Exception as e:
            logging.error(f"Error getting account info: {e}")
            return {}
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        positions = {}
        
        try:
            alpaca_positions = self.api.list_positions()
            
            for pos in alpaca_positions:
                position = Position(
                    symbol=str(pos.symbol) if pos.symbol is not None else "",
                    quantity=int(str(pos.qty)) if pos.qty is not None else 0,
                    entry_price=float(str(pos.avg_entry_price)) if pos.avg_entry_price is not None else 0.0,
                    current_price=float(str(pos.current_price)) if pos.current_price is not None else 0.0,
                    unrealized_pnl=float(str(pos.unrealized_pl)) if pos.unrealized_pl is not None else 0.0,
                    entry_time=datetime.now()  # Approximate
                )
                positions[str(pos.symbol) if pos.symbol is not None else ""] = position
            
            self.positions = positions
            
        except Exception as e:
            logging.error(f"Error getting positions: {e}")
        
        return positions
    
    def place_order(self, symbol: str, quantity: int, side: str, 
                   order_type: str = "market") -> Optional[Order]:
        """Place a trading order"""
        try:
            # Submit order to Alpaca
            alpaca_order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                type=order_type,
                time_in_force='day'
            )
            
            # Create order object
            order = Order(
                id=str(alpaca_order.id) if alpaca_order and alpaca_order.id is not None else "",
                symbol=symbol,
                quantity=quantity,
                side=side,
                type=order_type,
                status=str(alpaca_order.status) if alpaca_order and alpaca_order.status is not None else "",
                price=float(str(alpaca_order.filled_avg_price)) if alpaca_order and alpaca_order.filled_avg_price is not None else 0.0,
                created_at=datetime.now()
            )
            
            self.orders.append(order)
            logging.info(f"Placed {side} order for {quantity} {symbol}")
            
            return order
            
        except Exception as e:
            logging.error(f"Error placing order: {e}")
            return None
    
    def execute_strategy_signals(self, signals: Dict[str, float]):
        """Execute trading signals"""
        account_info = self.get_account_info()
        positions = self.get_positions()
        
        if not account_info:
            logging.error("Could not get account info")
            return
        
        for symbol, signal_strength in signals.items():
            if abs(signal_strength) < 0.5:  # Weak signal, skip
                continue
            
            current_position = positions.get(symbol)
            current_price = self._get_current_price(symbol)
            
            if not current_price:
                continue
            
            if signal_strength > 0.5 and not current_position:  # Buy signal
                # Calculate position size
                quantity = self.risk_manager.calculate_position_size(
                    current_price, account_info['portfolio_value']
                )
                
                if quantity > 0:
                    self.place_order(symbol, quantity, "buy")
            
            elif signal_strength < -0.5 and current_position:  # Sell signal
                # Close position
                self.place_order(symbol, current_position.quantity, "sell")
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            # Get latest bar
            bars = self.api.get_bars(symbol, TimeFrame(1, TimeFrameUnit.Minute), limit=1)
            if bars:
                return float(bars[0].c)
        except Exception as e:
            logging.error(f"Error getting price for {symbol}: {e}")
        
        return None
    
    def run_trading_loop(self, interval_seconds: int = 300):  # 5 minutes
        """Main trading loop"""
        logging.info("Starting live trading system")
        self.running = True
        
        while self.running:
            try:
                # Get market data
                all_symbols = []
                for strategy in self.strategies.values():
                    all_symbols.extend(strategy.symbols)
                all_symbols = list(set(all_symbols))  # Remove duplicates
                
                market_data = self.get_market_data(all_symbols)
                
                # Update positions for all strategies
                positions = self.get_positions()
                for strategy in self.strategies.values():
                    strategy.update_positions(positions)
                
                # Generate and execute signals for each strategy
                for strategy_name, strategy in self.strategies.items():
                    logging.info(f"Running strategy: {strategy_name}")
                    
                    # Generate signals
                    signals = strategy.analyze_market(market_data)
                    
                    # Execute signals
                    self.execute_strategy_signals(signals)
                
                # Log status
                account_info = self.get_account_info()
                if account_info:
                    logging.info(f"Portfolio Value: ${account_info['portfolio_value']:,.2f}")
                    logging.info(f"Cash: ${account_info['cash']:,.2f}")
                
                # Wait for next iteration
                time.sleep(interval_seconds)
                
            except Exception as e:
                logging.error(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def stop(self):
        """Stop the trading system"""
        logging.info("Stopping live trading system")
        self.running = False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        account_info = self.get_account_info()
        positions = self.get_positions()
        
        return {
            'running': self.running,
            'account': account_info,
            'positions': {sym: {
                'quantity': pos.quantity,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'unrealized_pnl': pos.unrealized_pnl
            } for sym, pos in positions.items()},
            'strategies': list(self.strategies.keys()),
            'total_orders': len(self.orders)
        }

def main():
    """Main function for running live trading system"""
    # Configuration (use environment variables in production)
    API_KEY = "your_api_key_here"
    SECRET_KEY = "your_secret_key_here"
    
    # Initialize trading system
    trading_system = LiveTradingSystem(API_KEY, SECRET_KEY)
    
    # Add strategies
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    
    momentum_strategy = LiveMomentumStrategy(symbols, lookback_period=20, threshold=0.02)
    mean_reversion_strategy = LiveMeanReversionStrategy(symbols, lookback_period=50, std_dev_threshold=2.0)
    
    trading_system.add_strategy(momentum_strategy)
    trading_system.add_strategy(mean_reversion_strategy)
    
    # Start trading (in a separate thread for demonstration)
    trading_thread = threading.Thread(
        target=trading_system.run_trading_loop,
        kwargs={'interval_seconds': 300}  # 5 minutes
    )
    trading_thread.daemon = True
    trading_thread.start()
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(60)
            status = trading_system.get_system_status()
            print(f"System Status: {status}")
    except KeyboardInterrupt:
        trading_system.stop()
        print("Trading system stopped")

if __name__ == "__main__":
    main() 