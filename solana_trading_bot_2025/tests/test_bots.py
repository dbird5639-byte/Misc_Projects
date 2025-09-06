"""
Tests for Solana Trading Bot 2025

Basic unit tests for bot functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime

from bots.sniper_bot import SniperBot
from bots.copy_bot import CopyBot
from bots.base_bot import BaseBot, TokenInfo, TradeSignal
from config.settings import Config

@pytest.fixture
def config():
    """Create a test configuration"""
    config = Config()
    config.sniper_config.enabled = True
    config.copy_config.enabled = True
    return config

@pytest.fixture
def sample_token():
    """Create a sample token for testing"""
    return TokenInfo(
        address="test_token_address",
        name="Test Token",
        symbol="TEST",
        price=1.0,
        volume_24h=10000.0,
        liquidity=50000.0,
        market_cap=100000.0,
        launch_time=datetime.now(),
        dex="raydium",
        pair_address="test_pair_address"
    )

@pytest.fixture
def sample_signal():
    """Create a sample trade signal for testing"""
    return TradeSignal(
        token_address="test_token_address",
        action="buy",
        price=1.0,
        quantity=0.1,
        confidence=0.8,
        source="test",
        timestamp=datetime.now(),
        metadata={"test": "data"}
    )

class TestSniperBot:
    """Test cases for SniperBot"""
    
    @pytest.mark.asyncio
    async def test_sniper_bot_initialization(self, config):
        """Test sniper bot initialization"""
        bot = SniperBot(config)
        assert bot.bot_name == "Sniper"
        assert bot.config == config
        assert not bot.is_running
    
    @pytest.mark.asyncio
    async def test_validate_new_token(self, config, sample_token):
        """Test token validation"""
        bot = SniperBot(config)
        
        # Test valid token
        result = await bot.validate_new_token(sample_token)
        assert result == True
        
        # Test invalid token (no liquidity)
        invalid_token = TokenInfo(
            address="invalid_token",
            name="Invalid Token",
            symbol="INVALID",
            price=1.0,
            volume_24h=100.0,  # Below threshold
            liquidity=100.0,   # Below threshold
            market_cap=1000.0,
            launch_time=datetime.now(),
            dex="raydium",
            pair_address="invalid_pair"
        )
        
        result = await bot.validate_new_token(invalid_token)
        assert result == False
    
    @pytest.mark.asyncio
    async def test_generate_signal(self, config, sample_token):
        """Test signal generation"""
        bot = SniperBot(config)
        
        analysis = {
            "should_trade": True,
            "confidence": 0.8,
            "risk_score": 0.2
        }
        
        signal = await bot.generate_signal(sample_token, analysis)
        assert signal is not None
        assert signal.token_address == sample_token.address
        assert signal.action == "buy"
        assert signal.confidence == 0.8
    
    @pytest.mark.asyncio
    async def test_process_signal(self, config, sample_signal):
        """Test signal processing"""
        bot = SniperBot(config)
        
        # Test with auto-trading disabled
        config.sniper_config.auto_trade = False
        result = await bot.process_signal(sample_signal)
        assert result == True

class TestCopyBot:
    """Test cases for CopyBot"""
    
    @pytest.mark.asyncio
    async def test_copy_bot_initialization(self, config):
        """Test copy bot initialization"""
        bot = CopyBot(config)
        assert bot.bot_name == "Copy"
        assert bot.config == config
        assert not bot.is_running
    
    @pytest.mark.asyncio
    async def test_validate_copy_trade(self, config):
        """Test copy trade validation"""
        bot = CopyBot(config)
        
        # Mock trader info
        trader = Mock()
        trader.success_rate = 0.7
        trader.address = "test_trader"
        
        # Valid trade
        valid_trade = {
            "hash": "test_hash",
            "timestamp": datetime.now(),
            "quantity": 0.1,
            "token_address": "test_token",
            "action": "buy",
            "price": 1.0
        }
        
        result = await bot.validate_copy_trade(valid_trade, trader)
        assert result == True
        
        # Invalid trade (too small)
        invalid_trade = {
            "hash": "test_hash",
            "timestamp": datetime.now(),
            "quantity": 0.001,  # Below minimum
            "token_address": "test_token",
            "action": "buy",
            "price": 1.0
        }
        
        result = await bot.validate_copy_trade(invalid_trade, trader)
        assert result == False
    
    @pytest.mark.asyncio
    async def test_create_copy_trade(self, config):
        """Test copy trade creation"""
        bot = CopyBot(config)
        
        # Mock trader
        trader = Mock()
        trader.address = "test_trader"
        trader.success_rate = 0.7
        
        # Mock trade
        trade = {
            "hash": "test_hash",
            "timestamp": datetime.now(),
            "quantity": 0.1,
            "token_address": "test_token",
            "action": "buy",
            "price": 1.0
        }
        
        copy_trade = await bot.create_copy_trade(trade, trader)
        assert copy_trade is not None
        assert copy_trade.original_trader == trader.address
        assert copy_trade.token_address == trade["token_address"]
        assert copy_trade.action == trade["action"]

class TestBaseBot(BaseBot):
    """Concrete implementation of BaseBot for testing"""
    
    async def run(self):
        """Test implementation of run method"""
        pass
    
    async def process_signal(self, signal: TradeSignal) -> bool:
        """Test implementation of process_signal method"""
        return True

class TestBaseBotMethods:
    """Test cases for BaseBot methods"""
    
    @pytest.mark.asyncio
    async def test_base_bot_initialization(self, config):
        """Test base bot initialization"""
        bot = TestBaseBot(config, "TestBot")
        assert bot.bot_name == "TestBot"
        assert bot.config == config
        assert not bot.is_running
    
    @pytest.mark.asyncio
    async def test_validate_token(self, config, sample_token):
        """Test token validation"""
        bot = TestBaseBot(config, "TestBot")
        
        result = await bot.validate_token(sample_token)
        assert result == True
        
        # Test invalid token
        invalid_token = TokenInfo(
            address="",
            name="",
            symbol="",
            price=0.0,
            volume_24h=0.0,
            liquidity=0.0,
            market_cap=0.0,
            launch_time=datetime.now(),
            dex="",
            pair_address=""
        )
        
        result = await bot.validate_token(invalid_token)
        assert result == False
    
    @pytest.mark.asyncio
    async def test_calculate_position_size(self, config, sample_token):
        """Test position size calculation"""
        bot = TestBaseBot(config, "TestBot")
        
        position_size = await bot.calculate_position_size(sample_token, 0.8)
        assert position_size > 0
        assert position_size <= config.sniper_config.max_position_size
    
    @pytest.mark.asyncio
    async def test_health_check(self, config):
        """Test health check"""
        bot = TestBaseBot(config, "TestBot")
        bot.is_running = True
        
        health = await bot.health_check()
        assert health == True
        
        bot.error_count = 15  # Too many errors
        health = await bot.health_check()
        assert health == False

@pytest.mark.asyncio
async def test_config_validation(config):
    """Test configuration validation"""
    # Valid config should pass
    assert config.validate_config() == True
    
    # Invalid config (missing API keys)
    config.api_config.solana_rpc = ""
    assert config.validate_config() == False

@pytest.mark.asyncio
async def test_token_info_creation():
    """Test TokenInfo creation"""
    token = TokenInfo(
        address="test",
        name="Test",
        symbol="TEST",
        price=1.0,
        volume_24h=1000.0,
        liquidity=5000.0,
        market_cap=10000.0,
        launch_time=datetime.now(),
        dex="raydium",
        pair_address="test_pair"
    )
    
    assert token.address == "test"
    assert token.name == "Test"
    assert token.symbol == "TEST"
    assert token.price == 1.0

@pytest.mark.asyncio
async def test_trade_signal_creation():
    """Test TradeSignal creation"""
    signal = TradeSignal(
        token_address="test",
        action="buy",
        price=1.0,
        quantity=0.1,
        confidence=0.8,
        source="test",
        timestamp=datetime.now(),
        metadata={"test": "data"}
    )
    
    assert signal.token_address == "test"
    assert signal.action == "buy"
    assert signal.price == 1.0
    assert signal.confidence == 0.8

if __name__ == "__main__":
    pytest.main([__file__]) 