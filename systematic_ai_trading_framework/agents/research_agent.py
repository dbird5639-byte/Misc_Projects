"""
Research Agent for discovering new trading strategies using AI.
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import logging

from .model_factory import ModelFactory
from config.settings import Settings
from utils.logger import setup_logger


@dataclass
class TradingIdea:
    """Represents a discovered trading idea."""
    source: str
    title: str
    description: str
    confidence: float
    category: str
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class Strategy:
    """Represents a generated trading strategy."""
    name: str
    description: str
    category: str
    confidence: float
    parameters: Dict[str, Any]
    risk_management: Dict[str, Any]
    expected_sharpe: float
    expected_drawdown: float
    source_idea: TradingIdea
    timestamp: datetime
    code: str


class ResearchAgent:
    """
    AI-powered research agent that discovers and analyzes trading strategies.
    """
    
    def __init__(self, model_factory: ModelFactory, settings: Settings):
        self.model_factory = model_factory
        self.settings = settings
        self.logger = setup_logger("research_agent", settings.log_level)
        
        # Research sources
        self.sources = {
            "youtube": self._research_youtube,
            "twitter": self._research_twitter,
            "reddit": self._research_reddit,
            "news": self._research_news,
            "academic": self._research_academic,
            "forums": self._research_forums
        }
        
        # Strategy templates
        self.strategy_templates = self._load_strategy_templates()
        
        self.logger.info("Research Agent initialized")
    
    def _load_strategy_templates(self) -> Dict[str, str]:
        """Load strategy code templates."""
        return {
            "momentum": """
class {strategy_name}(BaseStrategy):
    def __init__(self, parameters):
        super().__init__("{strategy_name}", parameters)
        self.lookback_period = parameters.get("lookback_period", 20)
        self.threshold = parameters.get("threshold", 0.02)
        self.rsi_period = parameters.get("rsi_period", 14)
        self.rsi_oversold = parameters.get("rsi_oversold", 30)
        self.rsi_overbought = parameters.get("rsi_overbought", 70)
    
    def generate_signals(self, data):
        # Calculate momentum indicators
        data['returns'] = data['close'].pct_change()
        data['momentum'] = data['returns'].rolling(self.lookback_period).mean()
        data['rsi'] = self.calculate_rsi(data['close'], self.rsi_period)
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        
        # Buy signal: positive momentum and oversold RSI
        buy_condition = (data['momentum'] > self.threshold) & (data['rsi'] < self.rsi_oversold)
        signals[buy_condition] = 1
        
        # Sell signal: negative momentum and overbought RSI
        sell_condition = (data['momentum'] < -self.threshold) & (data['rsi'] > self.rsi_overbought)
        signals[sell_condition] = -1
        
        return signals
""",
            "mean_reversion": """
class {strategy_name}(BaseStrategy):
    def __init__(self, parameters):
        super().__init__("{strategy_name}", parameters)
        self.bb_period = parameters.get("bb_period", 20)
        self.bb_std = parameters.get("bb_std", 2)
        self.rsi_period = parameters.get("rsi_period", 14)
        self.rsi_oversold = parameters.get("rsi_oversold", 30)
        self.rsi_overbought = parameters.get("rsi_overbought", 70)
    
    def generate_signals(self, data):
        # Calculate Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(
            data['close'], self.bb_period, self.bb_std
        )
        
        # Calculate RSI
        data['rsi'] = self.calculate_rsi(data['close'], self.rsi_period)
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        
        # Buy signal: price below lower band and oversold RSI
        buy_condition = (data['close'] < bb_lower) & (data['rsi'] < self.rsi_oversold)
        signals[buy_condition] = 1
        
        # Sell signal: price above upper band and overbought RSI
        sell_condition = (data['close'] > bb_upper) & (data['rsi'] > self.rsi_overbought)
        signals[sell_condition] = -1
        
        return signals
""",
            "regime_detection": """
class {strategy_name}(BaseStrategy):
    def __init__(self, parameters):
        super().__init__("{strategy_name}", parameters)
        self.volatility_period = parameters.get("volatility_period", 20)
        self.correlation_period = parameters.get("correlation_period", 60)
        self.regime_threshold = parameters.get("regime_threshold", 0.7)
    
    def generate_signals(self, data):
        # Calculate volatility regime
        data['volatility'] = data['close'].pct_change().rolling(self.volatility_period).std()
        data['volatility_regime'] = (data['volatility'] > data['volatility'].quantile(0.7)).astype(int)
        
        # Calculate correlation regime
        if len(data.columns) > 1:
            corr_matrix = data[['close']].corr()
            data['correlation_regime'] = (corr_matrix.abs() > self.regime_threshold).sum().sum()
        
        # Generate signals based on regime
        signals = pd.Series(0, index=data.index)
        
        # High volatility regime: reduce position size
        high_vol_condition = data['volatility_regime'] == 1
        signals[high_vol_condition] = 0.5  # Half position
        
        # Low volatility regime: normal position size
        low_vol_condition = data['volatility_regime'] == 0
        signals[low_vol_condition] = 1.0  # Full position
        
        return signals
"""
        }
    
    async def discover_strategies(self) -> List[Strategy]:
        """Discover new trading strategies from various sources."""
        self.logger.info("Starting strategy discovery...")
        
        discovered_strategies = []
        
        # Research from all enabled sources
        for source_name in self.settings.research_sources:
            if source_name in self.sources:
                try:
                    self.logger.info(f"Researching from {source_name}...")
                    ideas = await self.sources[source_name]()
                    
                    # Convert ideas to strategies
                    for idea in ideas:
                        strategy = await self._idea_to_strategy(idea)
                        if strategy:
                            discovered_strategies.append(strategy)
                    
                except Exception as e:
                    self.logger.error(f"Error researching from {source_name}: {e}")
        
        self.logger.info(f"Discovered {len(discovered_strategies)} new strategies")
        return discovered_strategies
    
    async def _idea_to_strategy(self, idea: TradingIdea) -> Optional[Strategy]:
        """Convert a trading idea to a concrete strategy."""
        try:
            # Use AI to analyze the idea and generate strategy
            prompt = self._create_strategy_prompt(idea)
            
            model = self.model_factory.get_model(self.settings.default_model)
            response = await model.generate(prompt)
            
            # Parse AI response
            strategy_data = self._parse_strategy_response(response, idea)
            
            if strategy_data:
                return Strategy(
                    name=strategy_data["name"],
                    description=strategy_data["description"],
                    category=strategy_data["category"],
                    confidence=strategy_data["confidence"],
                    parameters=strategy_data["parameters"],
                    risk_management=strategy_data["risk_management"],
                    expected_sharpe=strategy_data["expected_sharpe"],
                    expected_drawdown=strategy_data["expected_drawdown"],
                    source_idea=idea,
                    timestamp=datetime.now(),
                    code=strategy_data["code"]
                )
            
        except Exception as e:
            self.logger.error(f"Error converting idea to strategy: {e}")
        
        return None
    
    def _create_strategy_prompt(self, idea: TradingIdea) -> str:
        """Create a prompt for AI to generate strategy from idea."""
        return f"""
You are an expert quantitative trader. Analyze this trading idea and create a concrete trading strategy.

Trading Idea:
- Source: {idea.source}
- Title: {idea.title}
- Description: {idea.description}
- Category: {idea.category}
- Confidence: {idea.confidence}

Create a complete trading strategy with the following components:

1. Strategy Name: A descriptive name
2. Strategy Description: Detailed explanation
3. Strategy Category: momentum, mean_reversion, regime_detection, or other
4. Parameters: Dictionary of strategy parameters
5. Risk Management: Dictionary of risk management rules
6. Expected Performance: Expected Sharpe ratio and maximum drawdown
7. Python Code: Complete strategy implementation using the provided template

Use one of these templates based on the strategy category:

{json.dumps(self.strategy_templates, indent=2)}

Return your response as a JSON object with these fields:
- name: strategy name
- description: strategy description  
- category: strategy category
- confidence: confidence score (0-1)
- parameters: strategy parameters dict
- risk_management: risk management dict
- expected_sharpe: expected Sharpe ratio
- expected_drawdown: expected max drawdown
- code: complete Python code

Focus on creating practical, implementable strategies with clear entry/exit rules.
"""
    
    def _parse_strategy_response(self, response: str, idea: TradingIdea) -> Optional[Dict]:
        """Parse AI response to extract strategy data."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                strategy_data = json.loads(json_match.group())
                
                # Validate required fields
                required_fields = [
                    "name", "description", "category", "confidence",
                    "parameters", "risk_management", "expected_sharpe",
                    "expected_drawdown", "code"
                ]
                
                if all(field in strategy_data for field in required_fields):
                    return strategy_data
            
        except Exception as e:
            self.logger.error(f"Error parsing strategy response: {e}")
        
        return None
    
    async def _research_youtube(self) -> List[TradingIdea]:
        """Research trading ideas from YouTube."""
        # This would integrate with YouTube API
        # For now, return mock data
        return [
            TradingIdea(
                source="youtube",
                title="RSI Divergence Strategy",
                description="Using RSI divergence to identify trend reversals",
                confidence=0.75,
                category="momentum",
                timestamp=datetime.now(),
                metadata={"video_id": "abc123", "channel": "TradingChannel"}
            )
        ]
    
    async def _research_twitter(self) -> List[TradingIdea]:
        """Research trading ideas from Twitter."""
        # This would integrate with Twitter API
        return [
            TradingIdea(
                source="twitter",
                title="Volatility Breakout Strategy",
                description="Trading breakouts during high volatility periods",
                confidence=0.65,
                category="momentum",
                timestamp=datetime.now(),
                metadata={"tweet_id": "123456", "author": "@trader"}
            )
        ]
    
    async def _research_reddit(self) -> List[TradingIdea]:
        """Research trading ideas from Reddit."""
        # This would integrate with Reddit API
        return [
            TradingIdea(
                source="reddit",
                title="Mean Reversion with Bollinger Bands",
                description="Using Bollinger Bands for mean reversion trades",
                confidence=0.70,
                category="mean_reversion",
                timestamp=datetime.now(),
                metadata={"subreddit": "algotrading", "post_id": "abc123"}
            )
        ]
    
    async def _research_news(self) -> List[TradingIdea]:
        """Research trading ideas from news sources."""
        # This would integrate with news APIs
        return [
            TradingIdea(
                source="news",
                title="Sector Rotation Strategy",
                description="Rotating between sectors based on economic cycles",
                confidence=0.60,
                category="regime_detection",
                timestamp=datetime.now(),
                metadata={"source": "Bloomberg", "article_id": "123"}
            )
        ]
    
    async def _research_academic(self) -> List[TradingIdea]:
        """Research trading ideas from academic papers."""
        # This would integrate with academic APIs
        return [
            TradingIdea(
                source="academic",
                title="Regime-Dependent Momentum",
                description="Adapting momentum strategies to market regimes",
                confidence=0.80,
                category="regime_detection",
                timestamp=datetime.now(),
                metadata={"paper_id": "10.1234/paper", "authors": ["Smith", "Jones"]}
            )
        ]
    
    async def _research_forums(self) -> List[TradingIdea]:
        """Research trading ideas from trading forums."""
        # This would integrate with forum APIs
        return [
            TradingIdea(
                source="forums",
                title="Volume-Weighted Momentum",
                description="Combining volume and momentum for better signals",
                confidence=0.70,
                category="momentum",
                timestamp=datetime.now(),
                metadata={"forum": "EliteTrader", "thread_id": "123"}
            )
        ]
    
    async def analyze_market_conditions(self) -> Dict[str, Any]:
        """Analyze current market conditions for strategy selection."""
        try:
            prompt = """
Analyze current market conditions and provide insights for strategy selection.

Consider:
1. Market volatility levels
2. Trend strength and direction
3. Sector performance
4. Economic indicators
5. Market sentiment

Provide recommendations for:
- Which strategy types are most suitable
- Parameter adjustments needed
- Risk management considerations
- Expected performance expectations

Return as JSON with fields:
- market_regime: "trending", "ranging", "volatile", "calm"
- recommended_strategies: list of strategy categories
- parameter_adjustments: dict of suggested changes
- risk_level: "low", "medium", "high"
- confidence: confidence score (0-1)
"""
            
            model = self.model_factory.get_model(self.settings.default_model)
            response = await model.generate(prompt)
            
            # Parse response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {e}")
        
        return {
            "market_regime": "unknown",
            "recommended_strategies": ["momentum", "mean_reversion"],
            "parameter_adjustments": {},
            "risk_level": "medium",
            "confidence": 0.5
        }
    
    async def validate_strategy(self, strategy: Strategy) -> Dict[str, Any]:
        """Validate a strategy using AI analysis."""
        try:
            prompt = f"""
Validate this trading strategy and provide feedback:

Strategy: {strategy.name}
Description: {strategy.description}
Category: {strategy.category}
Parameters: {strategy.parameters}
Code: {strategy.code}

Analyze for:
1. Logical consistency
2. Implementation feasibility
3. Risk management adequacy
4. Expected performance realism
5. Potential issues or improvements

Return as JSON with fields:
- is_valid: boolean
- confidence: confidence score (0-1)
- issues: list of potential issues
- improvements: list of suggested improvements
- risk_assessment: risk level assessment
- performance_expectations: realistic performance expectations
"""
            
            model = self.model_factory.get_model(self.settings.default_model)
            response = await model.generate(prompt)
            
            # Parse response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
        except Exception as e:
            self.logger.error(f"Error validating strategy: {e}")
        
        return {
            "is_valid": False,
            "confidence": 0.0,
            "issues": ["Validation failed"],
            "improvements": [],
            "risk_assessment": "unknown",
            "performance_expectations": {}
        } 