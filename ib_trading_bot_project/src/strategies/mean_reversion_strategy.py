"""
Mean Reversion Trading Strategy

Trades against extreme price movements, buying oversold stocks
and selling overbought stocks.
"""

import time
from typing import Dict, Any, List, Optional

try:
    from loguru import logger  # type: ignore
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from .base_strategy import BaseStrategy, Signal

class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion trading strategy
    
    Buys stocks that are oversold and sells stocks that are overbought,
    based on statistical measures of price extremes.
    """
    
    def __init__(self, order_manager, position_manager, risk_manager, 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize mean reversion strategy
        
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
        self.lookback_period = config.get("lookback_period", 50)
        self.std_dev_threshold = config.get("std_dev_threshold", 2.0)
        self.rsi_oversold = config.get("rsi_oversold", 30)
        self.rsi_overbought = config.get("rsi_overbought", 70)
        self.stop_loss = config.get("stop_loss", 0.03)
        self.take_profit = config.get("take_profit", 0.08)
        
        # Bollinger Bands parameters
        self.bb_period = config.get("bb_period", 20)
        self.bb_std_dev = config.get("bb_std_dev", 2.0)
        
        # Mean reversion confirmation
        self.confirmation_period = config.get("confirmation_period", 3)
        self.min_reversion_strength = config.get("min_reversion_strength", 0.5)
    
    def analyze_market(self) -> Dict[str, Any]:
        """
        Analyze market data for mean reversion opportunities
        
        Returns:
            Market analysis results
        """
        try:
            analysis = {
                "signals": [],
                "oversold": [],
                "overbought": [],
                "risks": []
            }
            
            symbols = self.config.get("symbols", [])
            
            for symbol in symbols:
                symbol_analysis = self._analyze_symbol(symbol)
                if symbol_analysis:
                    analysis["signals"].append(symbol_analysis)
                    
                    if symbol_analysis["signal_strength"] > 0.7:
                        analysis["oversold"].append(symbol)
                    elif symbol_analysis["signal_strength"] < -0.7:
                        analysis["overbought"].append(symbol)
                    elif symbol_analysis["risk_level"] > 0.8:
                        analysis["risks"].append(symbol)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            return {"signals": [], "oversold": [], "overbought": [], "risks": []}
    
    def _analyze_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze a single symbol for mean reversion signals"""
        try:
            # Get market data
            market_data = self.get_market_data(symbol, self.lookback_period + 10)
            
            if len(market_data) < self.lookback_period:
                return None
            
            # Calculate technical indicators
            prices = [d.get("last_price", 0) for d in market_data if d.get("last_price", 0) > 0]
            
            if len(prices) < self.lookback_period:
                return None
            
            current_price = prices[-1]
            
            # Calculate Bollinger Bands
            bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(prices)
            
            # Calculate RSI
            rsi = self._calculate_rsi(prices)
            
            # Calculate statistical measures
            mean_price = self._calculate_mean(prices)
            std_dev = self._calculate_std_dev(prices, mean_price)
            z_score = (current_price - mean_price) / std_dev if std_dev > 0 else 0
            
            # Calculate mean reversion probability
            reversion_prob = self._calculate_reversion_probability(prices, current_price)
            
            # Determine signal strength
            signal_strength = self._calculate_signal_strength(
                current_price, bb_upper, bb_lower, rsi, z_score, reversion_prob
            )
            
            # Calculate risk level
            risk_level = self._calculate_risk_level(prices, current_price)
            
            return {
                "symbol": symbol,
                "signal_strength": signal_strength,
                "risk_level": risk_level,
                "current_price": current_price,
                "bb_upper": bb_upper,
                "bb_lower": bb_lower,
                "bb_middle": bb_middle,
                "rsi": rsi,
                "z_score": z_score,
                "reversion_prob": reversion_prob,
                "mean_price": mean_price,
                "std_dev": std_dev
            }
            
        except Exception as e:
            logger.error(f"Error analyzing symbol {symbol}: {e}")
            return None
    
    def generate_signals(self, analysis: Dict[str, Any]) -> List[Signal]:
        """
        Generate trading signals based on mean reversion analysis
        
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
                risk_level = signal_data["risk_level"]
                
                # Only trade if risk is acceptable
                if risk_level > 0.9:
                    continue
                
                # Generate buy signal for oversold conditions
                if signal_strength > self.min_reversion_strength:
                    # Calculate position size (smaller for mean reversion)
                    quantity = self.risk_manager.calculate_position_size(
                        symbol, current_price, "fixed"
                    )
                    
                    # Reduce position size for mean reversion
                    quantity = int(quantity * 0.7)
                    
                    if quantity > 0:
                        signal = Signal(
                            symbol=symbol,
                            action="BUY",
                            strength=signal_strength,
                            price=current_price,
                            quantity=quantity,
                            timestamp=time.time(),
                            reason=f"Mean reversion buy: strength={signal_strength:.3f}"
                        )
                        signals.append(signal)
                
                # Generate sell signal for overbought conditions
                elif signal_strength < -self.min_reversion_strength:
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
                            reason=f"Mean reversion sell: strength={abs(signal_strength):.3f}"
                        )
                        signals.append(signal)
            
            # Update signal count
            with self._lock:
                self.state.signals_generated += len(signals)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    def _calculate_bollinger_bands(self, prices: List[float]) -> tuple:
        """Calculate Bollinger Bands"""
        if len(prices) < self.bb_period:
            return 0.0, 0.0, 0.0
        
        recent_prices = prices[-self.bb_period:]
        middle = sum(recent_prices) / len(recent_prices)
        
        # Calculate standard deviation
        variance = sum((p - middle) ** 2 for p in recent_prices) / len(recent_prices)
        std_dev = variance ** 0.5
        
        upper = middle + (self.bb_std_dev * std_dev)
        lower = middle - (self.bb_std_dev * std_dev)
        
        return upper, lower, middle
    
    def _calculate_rsi(self, prices: List[float]) -> float:
        """Calculate RSI (Relative Strength Index)"""
        if len(prices) < 14:
            return 50.0
        
        # Calculate price changes
        changes = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            changes.append(change)
        
        if len(changes) < 14:
            return 50.0
        
        # Calculate gains and losses
        gains = [max(change, 0) for change in changes[-14:]]
        losses = [max(-change, 0) for change in changes[-14:]]
        
        avg_gain = sum(gains) / 14
        avg_loss = sum(losses) / 14
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_mean(self, prices: List[float]) -> float:
        """Calculate mean price"""
        return sum(prices) / len(prices)
    
    def _calculate_std_dev(self, prices: List[float], mean: float) -> float:
        """Calculate standard deviation"""
        if len(prices) < 2:
            return 0.0
        
        variance = sum((p - mean) ** 2 for p in prices) / len(prices)
        return variance ** 0.5
    
    def _calculate_reversion_probability(self, prices: List[float], 
                                       current_price: float) -> float:
        """Calculate probability of mean reversion"""
        if len(prices) < 10:
            return 0.5
        
        # Count how often price reverts to mean
        reversion_count = 0
        total_count = 0
        
        for i in range(10, len(prices)):
            price = prices[i]
            prev_prices = prices[i-10:i]
            prev_mean = sum(prev_prices) / len(prev_prices)
            
            # Check if price moved away from mean and then reverted
            if abs(price - prev_mean) > abs(prices[i-1] - prev_mean):
                reversion_count += 1
            total_count += 1
        
        if total_count > 0:
            return reversion_count / total_count
        return 0.5
    
    def _calculate_signal_strength(self, current_price: float, bb_upper: float,
                                 bb_lower: float, rsi: float, z_score: float,
                                 reversion_prob: float) -> float:
        """Calculate mean reversion signal strength"""
        signal_strength = 0.0
        
        # Bollinger Bands signal
        if current_price <= bb_lower:
            bb_signal = (bb_lower - current_price) / bb_lower
            signal_strength += bb_signal * 0.3
        elif current_price >= bb_upper:
            bb_signal = (current_price - bb_upper) / bb_upper
            signal_strength -= bb_signal * 0.3
        
        # RSI signal
        if rsi < self.rsi_oversold:
            rsi_signal = (self.rsi_oversold - rsi) / self.rsi_oversold
            signal_strength += rsi_signal * 0.3
        elif rsi > self.rsi_overbought:
            rsi_signal = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
            signal_strength -= rsi_signal * 0.3
        
        # Z-score signal
        if abs(z_score) > self.std_dev_threshold:
            z_signal = abs(z_score) - self.std_dev_threshold
            if z_score < 0:  # Oversold
                signal_strength += z_signal * 0.2
            else:  # Overbought
                signal_strength -= z_signal * 0.2
        
        # Reversion probability boost
        signal_strength *= reversion_prob
        
        # Normalize to [-1, 1] range
        signal_strength = max(-1.0, min(1.0, signal_strength))
        
        return signal_strength
    
    def _calculate_risk_level(self, prices: List[float], current_price: float) -> float:
        """Calculate risk level for mean reversion trade"""
        if len(prices) < 20:
            return 1.0
        
        # Calculate volatility
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)
        
        if len(returns) < 10:
            return 1.0
        
        # Calculate volatility
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        volatility = variance ** 0.5
        
        # Higher volatility = higher risk
        risk_level = min(1.0, volatility * 10)  # Scale volatility to 0-1
        
        return risk_level
    
    def check_stop_losses(self):
        """Check and execute stop losses for mean reversion positions"""
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
                
                # Check stop loss (tighter for mean reversion)
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
                
                # Check take profit (smaller for mean reversion)
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
            "name": "Mean Reversion Strategy",
            "description": "Trades against extreme price movements",
            "parameters": {
                "lookback_period": self.lookback_period,
                "std_dev_threshold": self.std_dev_threshold,
                "rsi_oversold": self.rsi_oversold,
                "rsi_overbought": self.rsi_overbought,
                "stop_loss": self.stop_loss,
                "take_profit": self.take_profit,
                "bb_period": self.bb_period,
                "bb_std_dev": self.bb_std_dev,
                "confirmation_period": self.confirmation_period,
                "min_reversion_strength": self.min_reversion_strength
            }
        } 