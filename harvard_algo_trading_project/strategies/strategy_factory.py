"""
Strategy Factory

Factory pattern for creating and managing different trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime
import json

@dataclass
class Signal:
    """Trading signal data structure"""
    symbol: str
    action: str  # "BUY", "SELL", "HOLD"
    strength: float  # Signal strength between -1 and 1
    price: float
    quantity: int
    timestamp: datetime
    reason: str

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, name: str, symbols: List[str], config: Dict[str, Any]):
        self.name = name
        self.symbols = symbols
        self.config = config
        self.positions = {}
        self.signals = []
        self.performance_metrics = {}
        
    @abstractmethod
    def analyze_market(self, market_data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Analyze market data and generate signals"""
        pass
    
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the strategy"""
        pass
    
    def validate_config(self) -> bool:
        """Validate strategy configuration"""
        required_params = self.get_required_parameters()
        for param in required_params:
            if param not in self.config:
                print(f"Missing required parameter: {param}")
                return False
        return True
    
    @abstractmethod
    def get_required_parameters(self) -> List[str]:
        """Get list of required configuration parameters"""
        pass
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get strategy performance summary"""
        return {
            "name": self.name,
            "symbols": self.symbols,
            "total_signals": len(self.signals),
            "last_signal_time": self.signals[-1].timestamp if self.signals else None,
            "performance_metrics": self.performance_metrics
        }

class MomentumStrategy(BaseStrategy):
    """Momentum-based trading strategy"""
    
    def __init__(self, symbols: List[str], config: Dict[str, Any]):
        super().__init__("Momentum Strategy", symbols, config)
    
    def get_required_parameters(self) -> List[str]:
        return ["lookback_period", "momentum_threshold", "rsi_period", "rsi_overbought", "rsi_oversold"]
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators"""
        if data.empty:
            return data
        
        lookback = self.config.get("lookback_period", 20)
        
        # Momentum
        data['momentum'] = data['close'].pct_change(lookback)
        
        # RSI
        rsi_period = self.config.get("rsi_period", 14)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        data['sma_short'] = data['close'].rolling(window=10).mean()
        data['sma_long'] = data['close'].rolling(window=30).mean()
        
        # Volume ratio
        data['volume_sma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        return data
    
    def analyze_market(self, market_data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Generate momentum signals"""
        signals = []
        threshold = self.config.get("momentum_threshold", 0.02)
        rsi_overbought = self.config.get("rsi_overbought", 70)
        rsi_oversold = self.config.get("rsi_oversold", 30)
        
        for symbol in self.symbols:
            if symbol not in market_data:
                continue
            
            data = market_data[symbol]
            if len(data) < 30:  # Need enough data
                continue
            
            # Calculate indicators
            data_with_indicators = self.calculate_indicators(data)
            
            # Get latest values
            latest = data_with_indicators.iloc[-1]
            
            # Generate signal
            signal_strength = 0.0
            action = "HOLD"
            reason = ""
            
            # Momentum signal
            if latest['momentum'] > threshold:
                signal_strength += 0.4
                reason += "Positive momentum; "
            elif latest['momentum'] < -threshold:
                signal_strength -= 0.4
                reason += "Negative momentum; "
            
            # RSI signal
            if latest['rsi'] < rsi_oversold:
                signal_strength += 0.3
                reason += "Oversold; "
            elif latest['rsi'] > rsi_overbought:
                signal_strength -= 0.3
                reason += "Overbought; "
            
            # Moving average crossover
            if latest['sma_short'] > latest['sma_long']:
                signal_strength += 0.2
                reason += "MA crossover bullish; "
            else:
                signal_strength -= 0.2
                reason += "MA crossover bearish; "
            
            # Volume confirmation
            if latest['volume_ratio'] > 1.2:
                signal_strength *= 1.1
                reason += "High volume; "
            
            # Determine action
            if signal_strength > 0.5:
                action = "BUY"
            elif signal_strength < -0.5:
                action = "SELL"
            
            if action != "HOLD":
                signal = Signal(
                    symbol=symbol,
                    action=action,
                    strength=abs(signal_strength),
                    price=latest['close'],
                    quantity=0,  # Will be calculated by position sizer
                    timestamp=datetime.now(),
                    reason=reason.strip()
                )
                signals.append(signal)
        
        return signals

class MeanReversionStrategy(BaseStrategy):
    """Mean reversion trading strategy"""
    
    def __init__(self, symbols: List[str], config: Dict[str, Any]):
        super().__init__("Mean Reversion Strategy", symbols, config)
    
    def get_required_parameters(self) -> List[str]:
        return ["lookback_period", "std_dev_threshold", "rsi_period", "rsi_overbought", "rsi_oversold"]
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean reversion indicators"""
        if data.empty:
            return data
        
        lookback = self.config.get("lookback_period", 50)
        
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=lookback).mean()
        bb_std = data['close'].rolling(window=lookback).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        
        # Z-score
        data['z_score'] = (data['close'] - data['bb_middle']) / bb_std
        
        # RSI
        rsi_period = self.config.get("rsi_period", 14)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Mean reversion probability
        data['reversion_prob'] = self._calculate_reversion_probability(data)
        
        return data
    
    def _calculate_reversion_probability(self, data: pd.DataFrame) -> pd.Series:
        """Calculate probability of mean reversion"""
        # Simple implementation - could be more sophisticated
        z_score_abs = data['z_score'].abs()
        prob = 1.0 / (1.0 + np.exp(z_score_abs - 2.0))  # Sigmoid function
        return prob
    
    def analyze_market(self, market_data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Generate mean reversion signals"""
        signals = []
        std_dev_threshold = self.config.get("std_dev_threshold", 2.0)
        rsi_overbought = self.config.get("rsi_overbought", 70)
        rsi_oversold = self.config.get("rsi_oversold", 30)
        
        for symbol in self.symbols:
            if symbol not in market_data:
                continue
            
            data = market_data[symbol]
            if len(data) < 50:  # Need enough data
                continue
            
            # Calculate indicators
            data_with_indicators = self.calculate_indicators(data)
            
            # Get latest values
            latest = data_with_indicators.iloc[-1]
            
            # Generate signal
            signal_strength = 0.0
            action = "HOLD"
            reason = ""
            
            # Z-score signal
            if latest['z_score'] < -std_dev_threshold:
                signal_strength += 0.4
                reason += f"Oversold (z-score: {latest['z_score']:.2f}); "
            elif latest['z_score'] > std_dev_threshold:
                signal_strength -= 0.4
                reason += f"Overbought (z-score: {latest['z_score']:.2f}); "
            
            # Bollinger Bands signal
            if latest['close'] <= latest['bb_lower']:
                signal_strength += 0.3
                reason += "Below lower Bollinger Band; "
            elif latest['close'] >= latest['bb_upper']:
                signal_strength -= 0.3
                reason += "Above upper Bollinger Band; "
            
            # RSI signal
            if latest['rsi'] < rsi_oversold:
                signal_strength += 0.2
                reason += "RSI oversold; "
            elif latest['rsi'] > rsi_overbought:
                signal_strength -= 0.2
                reason += "RSI overbought; "
            
            # Reversion probability boost
            signal_strength *= latest['reversion_prob']
            
            # Determine action
            if signal_strength > 0.5:
                action = "BUY"
            elif signal_strength < -0.5:
                action = "SELL"
            
            if action != "HOLD":
                signal = Signal(
                    symbol=symbol,
                    action=action,
                    strength=abs(signal_strength),
                    price=latest['close'],
                    quantity=0,  # Will be calculated by position sizer
                    timestamp=datetime.now(),
                    reason=reason.strip()
                )
                signals.append(signal)
        
        return signals

class BreakoutStrategy(BaseStrategy):
    """Breakout trading strategy"""
    
    def __init__(self, symbols: List[str], config: Dict[str, Any]):
        super().__init__("Breakout Strategy", symbols, config)
    
    def get_required_parameters(self) -> List[str]:
        return ["breakout_period", "volume_threshold", "atr_period"]
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate breakout indicators"""
        if data.empty:
            return data
        
        breakout_period = self.config.get("breakout_period", 20)
        atr_period = self.config.get("atr_period", 14)
        
        # Support and resistance levels
        data['resistance'] = data['high'].rolling(window=breakout_period).max()
        data['support'] = data['low'].rolling(window=breakout_period).min()
        
        # ATR (Average True Range)
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        data['atr'] = true_range.rolling(window=atr_period).mean()
        
        # Volume indicators
        data['volume_sma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        # Breakout signals
        data['breakout_up'] = (data['close'] > data['resistance'].shift(1)) & (data['volume_ratio'] > 1.5)
        data['breakout_down'] = (data['close'] < data['support'].shift(1)) & (data['volume_ratio'] > 1.5)
        
        return data
    
    def analyze_market(self, market_data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """Generate breakout signals"""
        signals = []
        volume_threshold = self.config.get("volume_threshold", 1.5)
        
        for symbol in self.symbols:
            if symbol not in market_data:
                continue
            
            data = market_data[symbol]
            if len(data) < 20:  # Need enough data
                continue
            
            # Calculate indicators
            data_with_indicators = self.calculate_indicators(data)
            
            # Get latest values
            latest = data_with_indicators.iloc[-1]
            previous = data_with_indicators.iloc[-2]
            
            # Generate signal
            signal_strength = 0.0
            action = "HOLD"
            reason = ""
            
            # Breakout signals
            if latest['breakout_up']:
                signal_strength = 0.8
                action = "BUY"
                reason = f"Bullish breakout above {previous['resistance']:.2f} with high volume"
            elif latest['breakout_down']:
                signal_strength = 0.8
                action = "SELL"
                reason = f"Bearish breakout below {previous['support']:.2f} with high volume"
            
            # Volume confirmation
            if latest['volume_ratio'] > volume_threshold:
                signal_strength *= 1.2
                reason += f" (Volume: {latest['volume_ratio']:.1f}x average)"
            
            if action != "HOLD":
                signal = Signal(
                    symbol=symbol,
                    action=action,
                    strength=signal_strength,
                    price=latest['close'],
                    quantity=0,  # Will be calculated by position sizer
                    timestamp=datetime.now(),
                    reason=reason
                )
                signals.append(signal)
        
        return signals

class StrategyFactory:
    """Factory for creating trading strategies"""
    
    def __init__(self):
        self.strategies = {
            "momentum": MomentumStrategy,
            "mean_reversion": MeanReversionStrategy,
            "breakout": BreakoutStrategy
        }
    
    def create_strategy(self, strategy_type: str, symbols: List[str], 
                       config: Dict[str, Any]) -> Optional[BaseStrategy]:
        """Create a strategy instance"""
        if strategy_type not in self.strategies:
            print(f"Unknown strategy type: {strategy_type}")
            return None
        
        strategy_class = self.strategies[strategy_type]
        strategy = strategy_class(symbols, config)
        
        # Validate configuration
        if not strategy.validate_config():
            print(f"Invalid configuration for {strategy_type} strategy")
            return None
        
        return strategy
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy types"""
        return list(self.strategies.keys())
    
    def get_strategy_config_template(self, strategy_type: str) -> Dict[str, Any]:
        """Get configuration template for a strategy"""
        templates = {
            "momentum": {
                "lookback_period": 20,
                "momentum_threshold": 0.02,
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30
            },
            "mean_reversion": {
                "lookback_period": 50,
                "std_dev_threshold": 2.0,
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30
            },
            "breakout": {
                "breakout_period": 20,
                "volume_threshold": 1.5,
                "atr_period": 14
            }
        }
        
        return templates.get(strategy_type, {})
    
    def save_strategy_config(self, strategy_name: str, config: Dict[str, Any], 
                           filename: str):
        """Save strategy configuration to file"""
        config_data = {
            "strategy_name": strategy_name,
            "config": config,
            "created_at": datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def load_strategy_config(self, filename: str) -> Dict[str, Any]:
        """Load strategy configuration from file"""
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading strategy config: {e}")
            return {}

def main():
    """Main function for testing strategy factory"""
    # Initialize factory
    factory = StrategyFactory()
    
    # Test available strategies
    print("Available strategies:", factory.get_available_strategies())
    
    # Create momentum strategy
    symbols = ["AAPL", "GOOGL", "MSFT"]
    momentum_config = factory.get_strategy_config_template("momentum")
    
    momentum_strategy = factory.create_strategy("momentum", symbols, momentum_config)
    
    if momentum_strategy:
        print(f"Created {momentum_strategy.name}")
        print(f"Required parameters: {momentum_strategy.get_required_parameters()}")
        
        # Test with sample data
        sample_data = {
            "AAPL": pd.DataFrame({
                'close': [150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160],
                'volume': [1000000] * 11
            })
        }
        
        signals = momentum_strategy.analyze_market(sample_data)
        print(f"Generated {len(signals)} signals")
        
        for signal in signals:
            print(f"{signal.symbol}: {signal.action} - {signal.reason}")

if __name__ == "__main__":
    main() 