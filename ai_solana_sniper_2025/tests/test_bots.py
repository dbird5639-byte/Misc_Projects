"""
Unit tests for AI-Powered Solana Meme Coin Sniper
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any

# Import components to test
from agents.model_factory import ModelFactory, ModelResponse, LocalLLMModel, OpenAIModel, AnthropicModel
from agents.sniper_agent import SniperAgent, TokenOpportunity, TradingDecision
from agents.chat_agent import ChatAgent, ChatMessage, AnalysisRequest, AnalysisResult
from agents.focus_agent import FocusAgent, Task, WorkflowState, SystemHealth
from data.market_data import MarketDataManager, TokenData, MarketData
from data.token_analyzer import TokenAnalyzer, SafetyAnalysis, TechnicalAnalysis
from trading.sniper_bot import SniperBot, TradeOrder, Position
from risk_management.risk_manager import RiskManager, RiskAssessment, PortfolioRisk
from utils.notifications import NotificationManager
from utils.logger import get_logger


class TestModelFactory:
    """Test AI Model Factory"""
    
    @pytest.fixture
    def config(self):
        return {
            "local_models": {
                "gemma": {
                    "enabled": True,
                    "model_path": "test_models/gemma",
                    "max_tokens": 2048,
                    "temperature": 0.7
                }
            },
            "cloud_models": {
                "openai": {
                    "enabled": True,
                    "api_key": "test_key",
                    "model": "gpt-4",
                    "max_tokens": 1000
                }
            },
            "agent_config": {
                "decision_threshold": 0.7,
                "max_analysis_time": 30,
                "parallel_processing": True
            }
        }
    
    @pytest.fixture
    def model_factory(self, config):
        return ModelFactory(config)
    
    @pytest.mark.asyncio
    async def test_initialize_models(self, model_factory):
        """Test model initialization"""
        with patch('agents.model_factory.TRANSFORMERS_AVAILABLE', False):
            with patch('agents.model_factory.OPENAI_AVAILABLE', False):
                await model_factory.initialize_models()
                assert len(model_factory.models) == 0
    
    @pytest.mark.asyncio
    async def test_generate_ensemble_response(self, model_factory):
        """Test ensemble response generation"""
        # Mock models
        mock_model = Mock()
        mock_model.generate_response = AsyncMock(return_value=ModelResponse(
            text="Test response",
            confidence=0.8,
            model_name="test_model",
            response_time=1.0
        ))
        model_factory.models["test_model"] = mock_model
        
        response = await model_factory.generate_ensemble_response("Test prompt")
        assert response.text == "Test response"
        assert response.confidence == 0.8
    
    def test_get_available_models(self, model_factory):
        """Test getting available models"""
        model_factory.models = {"model1": Mock(), "model2": Mock()}
        models = model_factory.get_available_models()
        assert len(models) == 2
        assert "model1" in models
        assert "model2" in models


class TestSniperAgent:
    """Test Sniper Agent"""
    
    @pytest.fixture
    def config(self):
        return {
            "enabled": True,
            "scan_interval": 3,
            "max_position_size": 0.05,
            "min_volume": 500,
            "min_liquidity": 2000,
            "auto_trade": False
        }
    
    @pytest.fixture
    def sniper_agent(self, config):
        return SniperAgent(config)
    
    @pytest.mark.asyncio
    async def test_initialize(self, sniper_agent):
        """Test sniper agent initialization"""
        ai_config = {"local_models": {}, "cloud_models": {}, "agent_config": {}}
        trading_config = {"market_data": {}, "token_analysis": {}, "sniper": {}, "risk_management": {}, "notifications": {}}
        
        with patch.object(sniper_agent, 'model_factory') as mock_factory:
            with patch.object(sniper_agent, 'market_data') as mock_market:
                with patch.object(sniper_agent, 'token_analyzer') as mock_analyzer:
                    with patch.object(sniper_agent, 'sniper_bot') as mock_bot:
                        with patch.object(sniper_agent, 'risk_manager') as mock_risk:
                            with patch.object(sniper_agent, 'notifications') as mock_notifications:
                                mock_factory.initialize_models = AsyncMock(return_value=True)
                                mock_market.initialize = AsyncMock(return_value=True)
                                mock_bot.initialize = AsyncMock(return_value=True)
                                
                                result = await sniper_agent.initialize(ai_config, trading_config)
                                assert result is True
    
    def test_meets_basic_criteria(self, sniper_agent):
        """Test basic criteria checking"""
        # Test token that meets criteria
        good_token = {
            "volume_24h": 1000,
            "liquidity": 5000
        }
        assert sniper_agent._meets_basic_criteria(good_token) is True
        
        # Test token that doesn't meet criteria
        bad_token = {
            "volume_24h": 100,
            "liquidity": 500
        }
        assert sniper_agent._meets_basic_criteria(bad_token) is False
    
    def test_deduplicate_tokens(self, sniper_agent):
        """Test token deduplication"""
        tokens = [
            {"address": "addr1", "name": "Token1"},
            {"address": "addr2", "name": "Token2"},
            {"address": "addr1", "name": "Token1Duplicate"}
        ]
        
        unique_tokens = sniper_agent._deduplicate_tokens(tokens)
        assert len(unique_tokens) == 2
        assert unique_tokens[0]["address"] == "addr1"
        assert unique_tokens[1]["address"] == "addr2"
    
    def test_create_analysis_prompt(self, sniper_agent):
        """Test analysis prompt creation"""
        opportunity = TokenOpportunity(
            token_address="test_addr",
            token_name="Test Token",
            token_symbol="TEST",
            price=0.001,
            volume_24h=1000,
            liquidity=5000,
            market_cap=10000,
            launch_time=datetime.now(),
            confidence_score=0.8,
            risk_score=0.3
        )
        
        prompt = sniper_agent._create_analysis_prompt(opportunity)
        assert "Test Token" in prompt
        assert "TEST" in prompt
        assert "test_addr" in prompt


class TestChatAgent:
    """Test Chat Agent"""
    
    @pytest.fixture
    def config(self):
        return {
            "enabled": True,
            "max_conversation_length": 50,
            "response_timeout": 30,
            "parallel_processing": True
        }
    
    @pytest.fixture
    def chat_agent(self, config):
        return ChatAgent(config)
    
    @pytest.mark.asyncio
    async def test_initialize(self, chat_agent):
        """Test chat agent initialization"""
        mock_factory = Mock()
        result = await chat_agent.initialize(mock_factory)
        assert result is True
        assert len(chat_agent.conversation_history) > 0
    
    @pytest.mark.asyncio
    async def test_add_user_message(self, chat_agent):
        """Test adding user message"""
        with patch.object(chat_agent, '_generate_response') as mock_generate:
            mock_generate.return_value = ModelResponse(
                text="AI response",
                confidence=0.8,
                model_name="test_model",
                response_time=1.0
            )
            
            response = await chat_agent.add_user_message("Hello AI")
            assert response == "AI response"
            assert len(chat_agent.conversation_history) == 3  # system + user + ai
    
    @pytest.mark.asyncio
    async def test_analyze_token(self, chat_agent):
        """Test token analysis"""
        token_data = {"address": "test", "price": 0.001}
        
        with patch.object(chat_agent, '_process_analysis_request') as mock_process:
            mock_process.return_value = AnalysisResult(
                request_id="test_id",
                analysis_type="token_analysis",
                result={"recommendation": "buy"},
                confidence=0.8,
                reasoning="Good token",
                model_used="test_model",
                processing_time=1.0,
                timestamp=datetime.now()
            )
            
            result = await chat_agent.analyze_token(token_data)
            assert result.analysis_type == "token_analysis"
            assert result.result["recommendation"] == "buy"


class TestFocusAgent:
    """Test Focus Agent"""
    
    @pytest.fixture
    def config(self):
        return {
            "enabled": True,
            "max_concurrent_tasks": 5,
            "task_timeout": 300,
            "health_check_interval": 60,
            "performance_review_interval": 3600
        }
    
    @pytest.fixture
    def focus_agent(self, config):
        return FocusAgent(config)
    
    @pytest.mark.asyncio
    async def test_initialize(self, focus_agent):
        """Test focus agent initialization"""
        result = await focus_agent.initialize()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_add_task(self, focus_agent):
        """Test adding task"""
        task_id = await focus_agent.add_task(
            "test_task",
            3,
            "Test description",
            {"data": "test"}
        )
        
        assert task_id in focus_agent.tasks
        task = focus_agent.tasks[task_id]
        assert task.task_type == "test_task"
        assert task.priority == 3
        assert task.description == "Test description"
    
    def test_get_performance_metrics(self, focus_agent):
        """Test performance metrics"""
        focus_agent.total_tasks_processed = 10
        focus_agent.successful_tasks = 8
        focus_agent.failed_tasks_count = 2
        
        metrics = focus_agent.get_performance_metrics()
        assert metrics["total_tasks_processed"] == 10
        assert metrics["successful_tasks"] == 8
        assert metrics["failed_trades"] == 2
        assert metrics["success_rate"] == 0.8


class TestMarketDataManager:
    """Test Market Data Manager"""
    
    @pytest.fixture
    def config(self):
        return {
            "enabled": True,
            "cache_duration": 30,
            "max_cache_size": 1000,
            "jupiter_api_key": "test_key",
            "birdeye_api_key": "test_key",
            "dexscreener_api_key": "test_key"
        }
    
    @pytest.fixture
    def market_data(self, config):
        return MarketDataManager(config)
    
    @pytest.mark.asyncio
    async def test_initialize(self, market_data):
        """Test market data manager initialization"""
        with patch.object(market_data, 'session') as mock_session:
            with patch.object(market_data, '_test_api_connections') as mock_test:
                result = await market_data.initialize()
                assert result is True
    
    def test_deduplicate_tokens(self, market_data):
        """Test token deduplication"""
        tokens = [
            {"address": "addr1", "name": "Token1"},
            {"address": "addr2", "name": "Token2"},
            {"address": "addr1", "name": "Token1Duplicate"}
        ]
        
        unique_tokens = market_data._deduplicate_tokens(tokens)
        assert len(unique_tokens) == 2
        assert unique_tokens[0]["address"] == "addr1"
        assert unique_tokens[1]["address"] == "addr2"
    
    def test_check_rate_limit(self, market_data):
        """Test rate limiting"""
        # Test within limit
        assert market_data._check_rate_limit("jupiter") is True
        
        # Test over limit
        market_data.rate_limits["jupiter"]["calls"] = 200
        assert market_data._check_rate_limit("jupiter") is False


class TestTokenAnalyzer:
    """Test Token Analyzer"""
    
    @pytest.fixture
    def config(self):
        return {
            "enabled": True,
            "min_liquidity": 1000,
            "min_holders": 10,
            "max_ownership_percentage": 5.0,
            "min_contract_age_hours": 1
        }
    
    @pytest.fixture
    def token_analyzer(self, config):
        return TokenAnalyzer(config)
    
    @pytest.mark.asyncio
    async def test_initialize(self, token_analyzer):
        """Test token analyzer initialization"""
        result = await token_analyzer.initialize()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_analyze_token_safety(self, token_analyzer):
        """Test token safety analysis"""
        token_data = {
            "liquidity": 5000,
            "volume_24h": 1000,
            "market_cap": 50000,
            "launch_time": time.time() - 3600,  # 1 hour ago
            "price_history": [{"price": 0.001}, {"price": 0.002}]
        }
        
        with patch.object(token_analyzer, '_get_solscan_token_data') as mock_solscan:
            with patch.object(token_analyzer, '_get_birdeye_token_data') as mock_birdeye:
                with patch.object(token_analyzer, '_get_dexscreener_token_data') as mock_dex:
                    with patch.object(token_analyzer, '_get_contract_data') as mock_contract:
                        mock_solscan.return_value = {"holder": 50}
                        mock_birdeye.return_value = {"ownerPercentage": 2.0}
                        mock_dex.return_value = {"liquidity": {"usd": 5000}}
                        mock_contract.return_value = {}
                        
                        result = await token_analyzer.analyze_token_safety("test_token")
                        assert isinstance(result, SafetyAnalysis)
                        assert result.is_safe is True
    
    def test_calculate_volatility(self, token_analyzer):
        """Test volatility calculation"""
        price_history = [
            {"price": 1.0},
            {"price": 1.1},
            {"price": 0.9},
            {"price": 1.2}
        ]
        
        volatility = token_analyzer._calculate_volatility(price_history)
        assert 0.0 <= volatility <= 1.0
    
    def test_calculate_risk_score(self, token_analyzer):
        """Test risk score calculation"""
        risk_factors = ["low_liquidity", "few_holders"]
        liquidity = 500
        holders = 5
        ownership_percentage = 10.0
        
        risk_score = token_analyzer._calculate_risk_score(
            risk_factors, liquidity, holders, ownership_percentage
        )
        assert 0.0 <= risk_score <= 1.0


class TestSniperBot:
    """Test Sniper Bot"""
    
    @pytest.fixture
    def config(self):
        return {
            "enabled": True,
            "auto_trade": False,
            "max_position_size": 0.05,
            "max_slippage": 0.05,
            "gas_limit": 300000,
            "rpc_url": "https://api.mainnet-beta.solana.com",
            "wallet_private_key": "test_key",
            "jupiter_api_key": "test_key"
        }
    
    @pytest.fixture
    def sniper_bot(self, config):
        return SniperBot(config)
    
    @pytest.mark.asyncio
    async def test_initialize(self, sniper_bot):
        """Test sniper bot initialization"""
        with patch.object(sniper_bot, 'client') as mock_client:
            with patch.object(sniper_bot, '_test_connection') as mock_test:
                result = await sniper_bot.initialize()
                assert result is True
    
    @pytest.mark.asyncio
    async def test_execute_buy_order(self, sniper_bot):
        """Test buy order execution"""
        with patch.object(sniper_bot, '_get_token_price') as mock_price:
            with patch.object(sniper_bot, '_execute_swap') as mock_swap:
                mock_price.return_value = 0.001
                mock_swap.return_value = True
                
                result = await sniper_bot.execute_buy_order("test_token", 100)
                assert result is True
    
    def test_get_performance_metrics(self, sniper_bot):
        """Test performance metrics"""
        sniper_bot.total_trades = 10
        sniper_bot.successful_trades = 8
        sniper_bot.failed_trades = 2
        sniper_bot.total_volume = 1000.0
        sniper_bot.total_fees = 50.0
        
        metrics = sniper_bot.get_performance_metrics()
        assert metrics["total_trades"] == 10
        assert metrics["successful_trades"] == 8
        assert metrics["failed_trades"] == 2
        assert metrics["success_rate"] == 0.8
        assert metrics["total_volume"] == 1000.0
        assert metrics["total_fees"] == 50.0


class TestRiskManager:
    """Test Risk Manager"""
    
    @pytest.fixture
    def config(self):
        return {
            "enabled": True,
            "max_portfolio_risk": 0.03,
            "max_position_risk": 0.01,
            "stop_loss_percentage": 0.15,
            "take_profit_percentage": 0.5,
            "max_positions": 5,
            "max_daily_loss": 0.1,
            "circuit_breaker_threshold": 0.2,
            "kelly_enabled": True,
            "kelly_fraction": 0.25
        }
    
    @pytest.fixture
    def risk_manager(self, config):
        return RiskManager(config)
    
    @pytest.mark.asyncio
    async def test_initialize(self, risk_manager):
        """Test risk manager initialization"""
        result = await risk_manager.initialize()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_assess_token_risk(self, risk_manager):
        """Test token risk assessment"""
        token_data = {
            "liquidity": 5000,
            "volume_24h": 1000,
            "market_cap": 50000,
            "launch_time": time.time() - 3600,
            "price_history": [{"price": 0.001}, {"price": 0.002}]
        }
        
        market_data = {
            "sentiment": "bullish"
        }
        
        result = await risk_manager.assess_token_risk("test_token", token_data, market_data)
        assert isinstance(result, RiskAssessment)
        assert 0.0 <= result.risk_score <= 1.0
    
    def test_is_trading_allowed(self, risk_manager):
        """Test trading permission"""
        # Test when trading is allowed
        risk_manager.enabled = True
        risk_manager.circuit_breaker_active = False
        risk_manager.positions = {}
        risk_manager.daily_pnl = 0
        risk_manager.daily_start_value = 1000
        
        assert risk_manager.is_trading_allowed() is True
        
        # Test when circuit breaker is active
        risk_manager.circuit_breaker_active = True
        assert risk_manager.is_trading_allowed() is False
    
    def test_get_position_size_limit(self, risk_manager):
        """Test position size limit calculation"""
        risk_manager.risk_assessments["test_token"] = RiskAssessment(
            token_address="test_token",
            risk_score=0.5,
            risk_level="medium",
            risk_factors=[],
            position_size_recommendation=0.02,
            max_loss_amount=100.0
        )
        
        limit = risk_manager.get_position_size_limit("test_token")
        assert limit == 0.02


class TestNotificationManager:
    """Test Notification Manager"""
    
    @pytest.fixture
    def config(self):
        return {
            "telegram_enabled": False,
            "discord_enabled": False,
            "browser_notifications": True,
            "email_enabled": False
        }
    
    @pytest.fixture
    def notification_manager(self, config):
        return NotificationManager(config)
    
    @pytest.mark.asyncio
    async def test_initialize(self, notification_manager):
        """Test notification manager initialization"""
        with patch.object(notification_manager, 'session') as mock_session:
            with patch.object(notification_manager, '_test_notifications') as mock_test:
                result = await notification_manager.initialize()
                assert result is True
    
    @pytest.mark.asyncio
    async def test_send_notification(self, notification_manager):
        """Test sending notification"""
        with patch.object(notification_manager, '_send_browser_notification') as mock_browser:
            mock_browser.return_value = True
            
            await notification_manager.send_notification("Test", "Test message")
            mock_browser.assert_called_once()
    
    def test_check_rate_limit(self, notification_manager):
        """Test rate limiting"""
        # Test within limit
        assert notification_manager._check_rate_limit() is True
        
        # Test over limit
        notification_manager.notification_count = 15
        notification_manager.rate_limit = 10
        assert notification_manager._check_rate_limit() is False


# Integration tests
class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete workflow from token detection to trade execution"""
        # This would test the complete integration of all components
        # For now, just verify that components can be instantiated together
        pass
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling across components"""
        # Test how the system handles various error conditions
        pass


# Performance tests
class TestPerformance:
    """Performance tests"""
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test system performance under concurrent load"""
        # Test multiple simultaneous operations
        pass
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test memory usage patterns"""
        # Monitor memory usage during operations
        pass


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 