"""
Test file for trading strategies

Basic unit tests for strategy functionality.
"""

import unittest
from unittest.mock import Mock, MagicMock
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.momentum_strategy import MomentumStrategy
from src.strategies.mean_reversion_strategy import MeanReversionStrategy
from src.strategies.base_strategy import Signal

class TestMomentumStrategy(unittest.TestCase):
    """Test cases for MomentumStrategy"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock components
        self.order_manager = Mock()
        self.position_manager = Mock()
        self.risk_manager = Mock()
        
        # Configure mocks
        self.risk_manager.calculate_position_size.return_value = 100
        self.risk_manager.should_trade.return_value = True
        
        # Create strategy
        config = {
            "symbols": ["AAPL", "GOOGL"],
            "lookback_period": 20,
            "momentum_threshold": 0.02
        }
        
        self.strategy = MomentumStrategy(
            self.order_manager, self.position_manager, self.risk_manager, config
        )
    
    def test_initialization(self):
        """Test strategy initialization"""
        self.assertIsNotNone(self.strategy)
        self.assertEqual(self.strategy.lookback_period, 20)
        self.assertEqual(self.strategy.momentum_threshold, 0.02)
    
    def test_calculate_ma(self):
        """Test moving average calculation"""
        prices = [100.0, 101.0, 102.0, 103.0, 104.0]
        ma = self.strategy._calculate_ma(prices, 3)
        expected = (102.0 + 103.0 + 104.0) / 3
        self.assertEqual(ma, expected)
    
    def test_calculate_momentum(self):
        """Test momentum calculation"""
        prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0]
        momentum = self.strategy._calculate_momentum(prices)
        expected = (110.0 - 100.0) / 100.0
        self.assertEqual(momentum, expected)
    
    def test_calculate_rsi(self):
        """Test RSI calculation"""
        # Test with increasing prices (should give high RSI)
        prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0]
        rsi = self.strategy._calculate_rsi(prices)
        self.assertGreater(rsi, 70)  # Should be overbought
        
        # Test with decreasing prices (should give low RSI)
        prices = [100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0, 91.0, 90.0, 89.0, 88.0, 87.0, 86.0]
        rsi = self.strategy._calculate_rsi(prices)
        self.assertLess(rsi, 30)  # Should be oversold
    
    def test_signal_strength_calculation(self):
        """Test signal strength calculation"""
        # Test bullish signal
        strength = self.strategy._calculate_signal_strength(
            short_ma=110.0,
            long_ma=105.0,
            momentum=0.05,
            rsi=60.0,
            volume_ratio=1.2
        )
        self.assertGreater(strength, 0)
        
        # Test bearish signal
        strength = self.strategy._calculate_signal_strength(
            short_ma=90.0,
            long_ma=105.0,
            momentum=-0.05,
            rsi=80.0,
            volume_ratio=0.8
        )
        self.assertLess(strength, 0)

class TestMeanReversionStrategy(unittest.TestCase):
    """Test cases for MeanReversionStrategy"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock components
        self.order_manager = Mock()
        self.position_manager = Mock()
        self.risk_manager = Mock()
        
        # Configure mocks
        self.risk_manager.calculate_position_size.return_value = 50
        self.risk_manager.should_trade.return_value = True
        
        # Create strategy
        config = {
            "symbols": ["AAPL", "GOOGL"],
            "lookback_period": 50,
            "std_dev_threshold": 2.0
        }
        
        self.strategy = MeanReversionStrategy(
            self.order_manager, self.position_manager, self.risk_manager, config
        )
    
    def test_initialization(self):
        """Test strategy initialization"""
        self.assertIsNotNone(self.strategy)
        self.assertEqual(self.strategy.lookback_period, 50)
        self.assertEqual(self.strategy.std_dev_threshold, 2.0)
    
    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation"""
        prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0]
        upper, lower, middle = self.strategy._calculate_bollinger_bands(prices)
        
        self.assertGreater(upper, middle)
        self.assertLess(lower, middle)
        self.assertGreater(upper, lower)
    
    def test_mean_calculation(self):
        """Test mean calculation"""
        prices = [100.0, 101.0, 102.0, 103.0, 104.0]
        mean = self.strategy._calculate_mean(prices)
        expected = 102.0
        self.assertEqual(mean, expected)
    
    def test_std_dev_calculation(self):
        """Test standard deviation calculation"""
        prices = [100.0, 101.0, 102.0, 103.0, 104.0]
        mean = self.strategy._calculate_mean(prices)
        std_dev = self.strategy._calculate_std_dev(prices, mean)
        
        self.assertGreater(std_dev, 0)
    
    def test_signal_strength_calculation(self):
        """Test signal strength calculation"""
        # Test oversold signal
        strength = self.strategy._calculate_signal_strength(
            current_price=95,
            bb_upper=110,
            bb_lower=100,
            rsi=25,
            z_score=-2.5,
            reversion_prob=0.8
        )
        self.assertGreater(strength, 0)
        
        # Test overbought signal
        strength = self.strategy._calculate_signal_strength(
            current_price=115,
            bb_upper=110,
            bb_lower=100,
            rsi=75,
            z_score=2.5,
            reversion_prob=0.8
        )
        self.assertLess(strength, 0)

class TestSignal(unittest.TestCase):
    """Test cases for Signal dataclass"""
    
    def test_signal_creation(self):
        """Test Signal object creation"""
        signal = Signal(
            symbol="AAPL",
            action="BUY",
            strength=0.8,
            price=150.0,
            quantity=100,
            timestamp=1234567890,
            reason="Test signal"
        )
        
        self.assertEqual(signal.symbol, "AAPL")
        self.assertEqual(signal.action, "BUY")
        self.assertEqual(signal.strength, 0.8)
        self.assertEqual(signal.price, 150.0)
        self.assertEqual(signal.quantity, 100)
        self.assertEqual(signal.reason, "Test signal")

class TestStrategyIntegration(unittest.TestCase):
    """Integration tests for strategies"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock components
        self.order_manager = Mock()
        self.position_manager = Mock()
        self.risk_manager = Mock()
        
        # Configure mocks
        self.risk_manager.calculate_position_size.return_value = 100
        self.risk_manager.should_trade.return_value = True
        self.position_manager.get_position.return_value = None
    
    def test_momentum_strategy_analysis(self):
        """Test momentum strategy market analysis"""
        config = {
            "symbols": ["AAPL"],
            "lookback_period": 10,
            "momentum_threshold": 0.01
        }
        
        strategy = MomentumStrategy(
            self.order_manager, self.position_manager, self.risk_manager, config
        )
        
        # Mock market data
        strategy.market_data = {
            "AAPL": [
                {"last_price": 100 + i, "volume": 1000} 
                for i in range(20)
            ]
        }
        
        # Test analysis
        analysis = strategy.analyze_market()
        self.assertIn("signals", analysis)
        self.assertIn("opportunities", analysis)
        self.assertIn("risks", analysis)
    
    def test_mean_reversion_strategy_analysis(self):
        """Test mean reversion strategy market analysis"""
        config = {
            "symbols": ["AAPL"],
            "lookback_period": 20,
            "std_dev_threshold": 1.5
        }
        
        strategy = MeanReversionStrategy(
            self.order_manager, self.position_manager, self.risk_manager, config
        )
        
        # Mock market data with extreme values
        strategy.market_data = {
            "AAPL": [
                {"last_price": 100 + (i % 3 - 1) * 10, "volume": 1000} 
                for i in range(30)
            ]
        }
        
        # Test analysis
        analysis = strategy.analyze_market()
        self.assertIn("signals", analysis)
        self.assertIn("oversold", analysis)
        self.assertIn("overbought", analysis)

def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestMomentumStrategy))
    test_suite.addTest(unittest.makeSuite(TestMeanReversionStrategy))
    test_suite.addTest(unittest.makeSuite(TestSignal))
    test_suite.addTest(unittest.makeSuite(TestStrategyIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 