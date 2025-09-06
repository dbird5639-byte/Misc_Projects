"""
Momentum Trading Strategy

Follows price momentum with trend confirmation using moving averages
and volume analysis.
"""

import time
from typing import Dict, Any, List, Optional

try:
    from loguru import logger  # type: ignore
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from .base_strategy import BaseStrategy, Signal

class MomentumStrategy(BaseStrategy):
    """
    Momentum trading strategy
    
    Buys stocks that are showing upward momentum and sells when
    momentum reverses or stops are hit.
    """
    
    def __init__(self, order_manager, position_manager, risk_manager, 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize momentum strategy
        
        Args:
            order_manager: Order manager instance
            position_manager: Position manager instance
            risk_manager: Risk manager instance
            config: Strategy configuration
        """
        super().__init__(order_manager, position_manager, risk_manager, config)
        
        # Ensure config is not None
        config = config or {}
        
        # Strategy parameters
        self.lookback_period = config.get("lookback_period", 20)
        self.momentum_threshold = config.get("momentum_threshold", 0.02)
        self.volume_threshold = config.get("volume_threshold", 1.5)
        self.stop_loss = config.get("stop_loss", 0.05)
        self.take_profit = config.get("take_profit", 0.10)
        
        # Technical indicators
        self.short_ma_period = config.get("short_ma_period", 10)
        self.long_ma_period = config.get("long_ma_period", 30)
        self.rsi_period = config.get("rsi_period", 14)
        self.rsi_oversold = config.get("rsi_oversold", 30)
        self.rsi_overbought = config.get("rsi_overbought", 70)
    
    def analyze_market(self) -> Dict[str, Any]:
        """
        Analyze market data for momentum opportunities
        
        Returns:
            Market analysis results
        """
        try:
            analysis = {
                "signals": [],
                "opportunities": [],
                "risks": []
            }
            
            symbols = self.config.get("symbols", [])
            
            for symbol in symbols:
                symbol_analysis = self._analyze_symbol(symbol)
                if symbol_analysis:
                    analysis["signals"].append(symbol_analysis)
                    
                    if symbol_analysis["signal_strength"] > 0.7:
                        analysis["opportunities"].append(symbol)
                    elif symbol_analysis["signal_strength"] < -0.7:
                        analysis["risks"].append(symbol)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            return {"signals": [], "opportunities": [], "risks": []}
    
    def _analyze_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze a single symbol for momentum signals"""
        try:
            # Get market data
            market_data = self.get_market_data(symbol, self.long_ma_period + 10)
            
            if len(market_data) < self.long_ma_period:
                return None
            
            # Calculate technical indicators
            prices = [d.get("last_price", 0) for d in market_data if d.get("last_price", 0) > 0]
            volumes = [d.get("volume", 0) for d in market_data if d.get("volume", 0) > 0]
            
            if len(prices) < self.long_ma_period:
                return None
            
            # Calculate moving averages
            short_ma = self._calculate_ma(prices, self.short_ma_period)
            long_ma = self._calculate_ma(prices, self.long_ma_period)
            
            # Calculate momentum
            momentum = self._calculate_momentum(prices)
            
            # Calculate RSI
            rsi = self._calculate_rsi(prices)
            
            # Calculate volume analysis
            volume_ratio = self._calculate_volume_ratio(volumes)
            
            # Determine signal strength
            signal_strength = self._calculate_signal_strength(
                short_ma, long_ma, momentum, rsi, volume_ratio
            )
            
            return {
                "symbol": symbol,
                "signal_strength": signal_strength,
                "short_ma": short_ma,
                "long_ma": long_ma,
                "momentum": momentum,
                "rsi": rsi,
                "volume_ratio": volume_ratio,
                "current_price": prices[-1] if prices else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing symbol {symbol}: {e}")
            return None
    
    def generate_signals(self, analysis: Dict[str, Any]) -> List[Signal]:
        """
        Generate trading signals based on momentum analysis
        
        Args:
            analysis: Market analysis results
            
        Returns:
            List of trading signals
        """
        signals = []
        
        try:
            for signal_data in analysis.get("signals", []):
                symbol = signal_data["symbol"]
                signal_strength = signal_data["signal_strength"]
                current_price = signal_data["current_price"]
                
                # Generate buy signal for strong positive momentum
                if signal_strength > self.momentum_threshold:
                    # Calculate position size
                    quantity = self.risk_manager.calculate_position_size(
                        symbol, current_price, "kelly"
                    )
                    
                    if quantity > 0:
                        signal = Signal(
                            symbol=symbol,
                            action="BUY",
                            strength=signal_strength,
                            price=current_price,
                            quantity=quantity,
                            timestamp=time.time(),
                            reason=f"Momentum signal: strength={signal_strength:.3f}"
                        )
                        signals.append(signal)
                
                # Generate sell signal for strong negative momentum
                elif signal_strength < -self.momentum_threshold:
                    # Check if we have a position to sell
                    position = self.position_manager.get_position(symbol)
                    if position and position.quantity > 0:
                        signal = Signal(
                            symbol=symbol,
                            action="SELL",
                            strength=abs(signal_strength),
                            price=current_price,
                            quantity=position.quantity,
                            timestamp=time.time(),
                            reason=f"Momentum reversal: strength={signal_strength:.3f}"
                        )
                        signals.append(signal)
            
            # Update signal count
            with self._lock:
                self.state.signals_generated += len(signals)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    def _calculate_ma(self, prices: List[float], period: int) -> float:
        """Calculate moving average"""
        if len(prices) < period:
            return 0.0
        
        return sum(prices[-period:]) / period
    
    def _calculate_momentum(self, prices: List[float]) -> float:
        """Calculate price momentum"""
        if len(prices) < self.lookback_period:
            return 0.0
        
        current_price = prices[-1]
        past_price = prices[-self.lookback_period]
        
        if past_price > 0:
            return (current_price - past_price) / past_price
        return 0.0
    
    def _calculate_rsi(self, prices: List[float]) -> float:
        """Calculate RSI (Relative Strength Index)"""
        if len(prices) < self.rsi_period + 1:
            return 50.0  # Neutral RSI
        
        # Calculate price changes
        changes = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            changes.append(change)
        
        if len(changes) < self.rsi_period:
            return 50.0
        
        # Calculate gains and losses
        gains = [max(change, 0) for change in changes[-self.rsi_period:]]
        losses = [max(-change, 0) for change in changes[-self.rsi_period:]]
        
        avg_gain = sum(gains) / self.rsi_period
        avg_loss = sum(losses) / self.rsi_period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_volume_ratio(self, volumes: List[int]) -> float:
        """Calculate volume ratio compared to average"""
        if len(volumes) < 20:
            return 1.0
        
        current_volume = volumes[-1]
        avg_volume = sum(volumes[-20:]) / 20
        
        if avg_volume > 0:
            return current_volume / avg_volume
        return 1.0
    
    def _calculate_signal_strength(self, short_ma: float, long_ma: float,
                                 momentum: float, rsi: float, 
                                 volume_ratio: float) -> float:
        """Calculate overall signal strength"""
        signal_strength = 0.0
        
        # Moving average crossover
        if short_ma > 0 and long_ma > 0:
            ma_signal = (short_ma - long_ma) / long_ma
            signal_strength += ma_signal * 0.3
        
        # Momentum component
        signal_strength += momentum * 0.4
        
        # RSI component
        if rsi < self.rsi_oversold:
            rsi_signal = (self.rsi_oversold - rsi) / self.rsi_oversold
            signal_strength += rsi_signal * 0.2
        elif rsi > self.rsi_overbought:
            rsi_signal = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
            signal_strength -= rsi_signal * 0.2
        
        # Volume confirmation
        if volume_ratio > self.volume_threshold:
            signal_strength *= 1.2  # Boost signal with high volume
        elif volume_ratio < 0.5:
            signal_strength *= 0.8  # Reduce signal with low volume
        
        # Normalize to [-1, 1] range
        signal_strength = max(-1.0, min(1.0, signal_strength))
        
        return signal_strength
    
    def check_stop_losses(self):
        """Check and execute stop losses"""
        try:
            positions = self.position_manager.get_positions()
            
            for position in positions:
                symbol = position.symbol
                current_price = position.last_price
                avg_cost = position.avg_cost
                
                if current_price <= 0 or avg_cost <= 0:
                    continue
                
                # Calculate loss percentage
                loss_pct = (avg_cost - current_price) / avg_cost
                
                # Check stop loss
                if loss_pct >= self.stop_loss:
                    logger.info(f"Stop loss triggered for {symbol}: {loss_pct:.2%}")
                    
                    signal = Signal(
                        symbol=symbol,
                        action="SELL",
                        strength=1.0,
                        price=current_price,
                        quantity=position.quantity,
                        timestamp=time.time(),
                        reason=f"Stop loss: {loss_pct:.2%}"
                    )
                    
                    self._execute_signal(signal)
                
                # Check take profit
                profit_pct = (current_price - avg_cost) / avg_cost
                if profit_pct >= self.take_profit:
                    logger.info(f"Take profit triggered for {symbol}: {profit_pct:.2%}")
                    
                    signal = Signal(
                        symbol=symbol,
                        action="SELL",
                        strength=1.0,
                        price=current_price,
                        quantity=position.quantity,
                        timestamp=time.time(),
                        reason=f"Take profit: {profit_pct:.2%}"
                    )
                    
                    self._execute_signal(signal)
                    
        except Exception as e:
            logger.error(f"Error checking stop losses: {e}")
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information"""
        return {
            "name": "Momentum Strategy",
            "description": "Follows price momentum with trend confirmation",
            "parameters": {
                "lookback_period": self.lookback_period,
                "momentum_threshold": self.momentum_threshold,
                "volume_threshold": self.volume_threshold,
                "stop_loss": self.stop_loss,
                "take_profit": self.take_profit,
                "short_ma_period": self.short_ma_period,
                "long_ma_period": self.long_ma_period,
                "rsi_period": self.rsi_period
            }
        } 