"""
Base Strategy Class

Abstract base class for all trading strategies.
"""

import time
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass

try:
    from loguru import logger  # type: ignore
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

@dataclass
class Signal:
    """Trading signal data structure"""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    strength: float  # 0.0 to 1.0
    price: float
    quantity: int
    timestamp: float
    reason: str

@dataclass
class StrategyState:
    """Strategy state data structure"""
    running: bool
    last_analysis: float
    signals_generated: int
    trades_executed: int
    current_positions: List[str]
    performance_metrics: Dict[str, float]

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies
    """
    
    def __init__(self, order_manager, position_manager, risk_manager, 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize base strategy
        
        Args:
            order_manager: Order manager instance
            position_manager: Position manager instance
            risk_manager: Risk manager instance
            config: Strategy configuration
        """
        self.order_manager = order_manager
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.config = config or {}
        
        # Strategy state
        self.state = StrategyState(
            running=False,
            last_analysis=0.0,
            signals_generated=0,
            trades_executed=0,
            current_positions=[],
            performance_metrics={}
        )
        
        # Data storage
        self.market_data: Dict[str, List[Dict[str, Any]]] = {}
        self.signals: List[Signal] = []
        self.trades: List[Dict[str, Any]] = []
        
        # Threading
        self._lock = threading.Lock()
        self._analysis_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.signal_callbacks: List[Callable] = []
        self.trade_callbacks: List[Callable] = []
    
    def start(self):
        """Start the strategy"""
        try:
            logger.info(f"Starting strategy: {self.__class__.__name__}")
            
            self.state.running = True
            
            # Start analysis thread
            self._analysis_thread = threading.Thread(target=self._analysis_loop)
            self._analysis_thread.daemon = True
            self._analysis_thread.start()
            
            logger.info(f"Strategy {self.__class__.__name__} started")
            
        except Exception as e:
            logger.error(f"Error starting strategy: {e}")
            self.state.running = False
    
    def stop(self):
        """Stop the strategy"""
        try:
            logger.info(f"Stopping strategy: {self.__class__.__name__}")
            
            self.state.running = False
            
            # Wait for analysis thread to finish
            if self._analysis_thread and self._analysis_thread.is_alive():
                self._analysis_thread.join(timeout=5)
            
            logger.info(f"Strategy {self.__class__.__name__} stopped")
            
        except Exception as e:
            logger.error(f"Error stopping strategy: {e}")
    
    def is_running(self) -> bool:
        """Check if strategy is running"""
        return self.state.running
    
    def _analysis_loop(self):
        """Main analysis loop"""
        while self.state.running:
            try:
                # Perform market analysis
                self.analyze_and_trade()
                
                # Update state
                self.state.last_analysis = time.time()
                
                # Sleep between analyses
                time.sleep(self.config.get("analysis_interval", 60))
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                time.sleep(10)  # Wait before retrying
    
    def analyze_and_trade(self):
        """Main analysis and trading method"""
        try:
            # Get market data
            symbols = self.config.get("symbols", [])
            for symbol in symbols:
                self._update_market_data(symbol)
            
            # Analyze market
            analysis = self.analyze_market()
            
            # Generate signals
            signals = self.generate_signals(analysis)
            
            # Execute trades
            self._execute_signals(signals)
            
        except Exception as e:
            logger.error(f"Error in analyze_and_trade: {e}")
    
    @abstractmethod
    def analyze_market(self) -> Dict[str, Any]:
        """
        Analyze market data and return analysis results
        
        Returns:
            Market analysis results
        """
        pass
    
    @abstractmethod
    def generate_signals(self, analysis: Dict[str, Any]) -> List[Signal]:
        """
        Generate trading signals based on analysis
        
        Args:
            analysis: Market analysis results
            
        Returns:
            List of trading signals
        """
        pass
    
    def _update_market_data(self, symbol: str):
        """Update market data for a symbol"""
        try:
            # Get market data from connector
            connector = self.order_manager.connector
            market_data = connector.get_market_data(symbol)
            
            if market_data:
                with self._lock:
                    if symbol not in self.market_data:
                        self.market_data[symbol] = []
                    
                    # Add timestamp
                    market_data["timestamp"] = time.time()
                    
                    # Store data (keep last 1000 data points)
                    self.market_data[symbol].append(market_data)
                    if len(self.market_data[symbol]) > 1000:
                        self.market_data[symbol] = self.market_data[symbol][-1000:]
            
        except Exception as e:
            logger.error(f"Error updating market data for {symbol}: {e}")
    
    def _execute_signals(self, signals: List[Signal]):
        """Execute trading signals"""
        for signal in signals:
            try:
                if self._should_execute_signal(signal):
                    self._execute_signal(signal)
                    
            except Exception as e:
                logger.error(f"Error executing signal: {e}")
    
    def _should_execute_signal(self, signal: Signal) -> bool:
        """Check if signal should be executed"""
        # Check risk limits
        if not self.risk_manager.should_trade(
            signal.symbol, signal.action, signal.quantity, signal.price
        ):
            logger.info(f"Signal rejected by risk manager: {signal}")
            return False
        
        # Check signal strength
        if signal.strength < self.config.get("min_signal_strength", 0.5):
            logger.info(f"Signal strength too low: {signal.strength}")
            return False
        
        # Check if we already have a position (for buy signals)
        if signal.action == "BUY" and self.position_manager.has_position(signal.symbol):
            logger.info(f"Already have position in {signal.symbol}")
            return False
        
        return True
    
    def _execute_signal(self, signal: Signal):
        """Execute a single trading signal"""
        try:
            logger.info(f"Executing signal: {signal}")
            
            # Place order
            order_id = self.order_manager.place_market_order(
                signal.symbol, signal.action, signal.quantity
            )
            
            if order_id > 0:
                # Record trade
                trade = {
                    "order_id": order_id,
                    "signal": signal,
                    "timestamp": time.time(),
                    "status": "SUBMITTED"
                }
                
                with self._lock:
                    self.trades.append(trade)
                    self.state.trades_executed += 1
                
                # Notify callbacks
                for callback in self.trade_callbacks:
                    try:
                        callback(trade)
                    except Exception as e:
                        logger.error(f"Error in trade callback: {e}")
                
                logger.info(f"Signal executed successfully: order_id={order_id}")
            else:
                logger.error(f"Failed to execute signal: {signal}")
                
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
    
    def get_market_data(self, symbol: str, lookback: int = 100) -> List[Dict[str, Any]]:
        """
        Get market data for a symbol
        
        Args:
            symbol: Stock symbol
            lookback: Number of data points to return
            
        Returns:
            List of market data points
        """
        with self._lock:
            if symbol in self.market_data:
                return self.market_data[symbol][-lookback:]
            return []
    
    def get_signals(self, count: int = 100) -> List[Signal]:
        """
        Get recent signals
        
        Args:
            count: Number of signals to return
            
        Returns:
            List of recent signals
        """
        with self._lock:
            return self.signals[-count:]
    
    def get_trades(self, count: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent trades
        
        Args:
            count: Number of trades to return
            
        Returns:
            List of recent trades
        """
        with self._lock:
            return self.trades[-count:]
    
    def get_status(self) -> Dict[str, Any]:
        """Get strategy status"""
        return {
            "name": self.__class__.__name__,
            "running": self.state.running,
            "last_analysis": self.state.last_analysis,
            "signals_generated": self.state.signals_generated,
            "trades_executed": self.state.trades_executed,
            "current_positions": self.state.current_positions,
            "performance_metrics": self.state.performance_metrics,
            "config": self.config
        }
    
    def add_signal_callback(self, callback: Callable):
        """Add signal callback"""
        self.signal_callbacks.append(callback)
    
    def add_trade_callback(self, callback: Callable):
        """Add trade callback"""
        self.trade_callbacks.append(callback)
    
    def calculate_performance(self) -> Dict[str, float]:
        """Calculate strategy performance metrics"""
        try:
            # Get recent trades
            trades = self.get_trades(1000)
            
            if not trades:
                return {
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "avg_return": 0.0,
                    "sharpe_ratio": 0.0
                }
            
            # Calculate basic metrics
            total_trades = len(trades)
            winning_trades = 0
            total_return = 0.0
            
            for trade in trades:
                # Simplified P&L calculation
                if trade.get("status") == "FILLED":
                    # In real implementation, would calculate actual P&L
                    total_return += 0.01  # Assume 1% return per trade
                    winning_trades += 1
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            avg_return = total_return / total_trades if total_trades > 0 else 0.0
            
            # Simplified Sharpe ratio
            sharpe_ratio = avg_return / 0.02 if avg_return > 0 else 0.0  # Assume 2% volatility
            
            return {
                "total_trades": total_trades,
                "win_rate": win_rate,
                "avg_return": avg_return,
                "sharpe_ratio": sharpe_ratio
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance: {e}")
            return {}
    
    def update_performance_metrics(self):
        """Update performance metrics"""
        try:
            metrics = self.calculate_performance()
            
            with self._lock:
                self.state.performance_metrics = metrics
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}") 