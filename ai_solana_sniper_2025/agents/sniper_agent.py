"""
Sniper Agent - Main coordination agent for token detection and trading
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import json

from .model_factory import ModelFactory, ModelResponse
from ..data.market_data import MarketDataManager
from ..data.token_analyzer import TokenAnalyzer
from ..trading.sniper_bot import SniperBot
from ..risk_management.risk_manager import RiskManager
from ..utils.notifications import NotificationManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TokenOpportunity:
    """Represents a potential trading opportunity"""
    token_address: str
    token_name: str
    token_symbol: str
    price: float
    volume_24h: float
    liquidity: float
    market_cap: float
    launch_time: datetime
    confidence_score: float
    risk_score: float
    ai_analysis: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TradingDecision:
    """Represents a trading decision"""
    action: str  # "buy", "sell", "hold", "skip"
    token_address: str
    amount: float
    price: float
    confidence: float
    reasoning: str
    risk_assessment: Dict[str, Any]
    timestamp: datetime


class SniperAgent:
    """
    Main coordination agent for token detection and trading decisions
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.scan_interval = config.get("scan_interval", 3)
        self.max_position_size = config.get("max_position_size", 0.05)
        self.min_volume = config.get("min_volume", 500)
        self.min_liquidity = config.get("min_liquidity", 2000)
        self.auto_trade = config.get("auto_trade", False)
        
        # Initialize components
        self.model_factory = None
        self.market_data = None
        self.token_analyzer = None
        self.sniper_bot = None
        self.risk_manager = None
        self.notifications = None
        
        # State management
        self.is_running = False
        self.opportunities: List[TokenOpportunity] = []
        self.decisions: List[TradingDecision] = []
        self.performance_metrics = {
            "total_opportunities": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "total_profit": 0.0,
            "ai_accuracy": 0.0
        }
        
        # Performance tracking
        self.start_time = None
        self.last_scan_time = None
        
    async def initialize(self, ai_config: Dict[str, Any], trading_config: Dict[str, Any]):
        """Initialize the sniper agent and all components"""
        logger.info("Initializing Sniper Agent...")
        
        try:
            # Initialize AI model factory
            self.model_factory = ModelFactory(ai_config)
            await self.model_factory.initialize_models()
            
            # Initialize market data manager
            self.market_data = MarketDataManager(trading_config.get("market_data", {}))
            await self.market_data.initialize()
            
            # Initialize token analyzer
            self.token_analyzer = TokenAnalyzer(trading_config.get("token_analysis", {}))
            
            # Initialize sniper bot
            self.sniper_bot = SniperBot(trading_config.get("sniper", {}))
            await self.sniper_bot.initialize()
            
            # Initialize risk manager
            self.risk_manager = RiskManager(trading_config.get("risk_management", {}))
            
            # Initialize notifications
            self.notifications = NotificationManager(trading_config.get("notifications", {}))
            
            logger.info("Sniper Agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Sniper Agent: {e}")
            return False
    
    async def start(self):
        """Start the sniper agent"""
        if not self.enabled:
            logger.warning("Sniper Agent is disabled")
            return
        
        if self.is_running:
            logger.warning("Sniper Agent is already running")
            return
        
        logger.info("Starting Sniper Agent...")
        self.is_running = True
        self.start_time = datetime.now()
        
        # Send startup notification
        if self.notifications:
            await self.notifications.send_notification(
                "🚀 AI Sniper Agent Started",
                f"Monitoring Solana tokens with {len(self.model_factory.get_available_models()) if self.model_factory else 0} AI models"
            )
        
        try:
            while self.is_running:
                await self._scan_cycle()
                await asyncio.sleep(self.scan_interval)
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Error in sniper agent main loop: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the sniper agent"""
        logger.info("Stopping Sniper Agent...")
        self.is_running = False
        
        # Send shutdown notification
        if self.notifications and self.start_time:
            await self.notifications.send_notification(
                "🛑 AI Sniper Agent Stopped",
                f"Runtime: {datetime.now() - self.start_time}"
            )
        
        logger.info("Sniper Agent stopped")
    
    async def _scan_cycle(self):
        """Perform one complete scan cycle"""
        try:
            self.last_scan_time = datetime.now()
            
            # 1. Scan for new tokens
            new_tokens = await self._scan_new_tokens()
            
            # 2. Analyze opportunities
            opportunities = await self._analyze_opportunities(new_tokens)
            
            # 3. Make AI-powered decisions
            decisions = await self._make_trading_decisions(opportunities)
            
            # 4. Execute trades (if auto-trade enabled)
            if self.auto_trade:
                await self._execute_trades(decisions)
            
            # 5. Update performance metrics
            await self._update_metrics()
            
        except Exception as e:
            logger.error(f"Error in scan cycle: {e}")
    
    async def _scan_new_tokens(self) -> List[Dict[str, Any]]:
        """Scan for new token launches"""
        try:
            # Get new tokens from multiple sources
            new_tokens = []
            
            # Scan DEX aggregators
            if self.market_data:
                dex_tokens = await self.market_data.get_new_tokens()
                new_tokens.extend(dex_tokens)
                
                # Scan social media mentions
                social_tokens = await self.market_data.get_social_mentions()
                new_tokens.extend(social_tokens)
            
            # Remove duplicates
            unique_tokens = self._deduplicate_tokens(new_tokens)
            
            logger.info(f"Found {len(unique_tokens)} new token opportunities")
            return unique_tokens
            
        except Exception as e:
            logger.error(f"Error scanning for new tokens: {e}")
            return []
    
    async def _analyze_opportunities(self, tokens: List[Dict[str, Any]]) -> List[TokenOpportunity]:
        """Analyze token opportunities"""
        opportunities = []
        
        for token in tokens:
            try:
                # Basic filtering
                if not self._meets_basic_criteria(token):
                    continue
                
                # Get detailed token data
                token_data = None
                if self.market_data:
                    token_data = await self.market_data.get_token_data(token["address"])
                if not token_data:
                    continue
                
                # Analyze token safety
                safety_analysis = {}
                if self.token_analyzer:
                    safety_analysis = await self.token_analyzer.analyze_token_safety(token["address"])
                
                # Create opportunity object
                opportunity = TokenOpportunity(
                    token_address=token["address"],
                    token_name=getattr(token_data, 'name', 'Unknown'),
                    token_symbol=getattr(token_data, 'symbol', 'UNK'),
                    price=getattr(token_data, 'price', 0),
                    volume_24h=getattr(token_data, 'volume_24h', 0),
                    liquidity=getattr(token_data, 'liquidity', 0),
                    market_cap=getattr(token_data, 'market_cap', 0),
                    launch_time=datetime.fromtimestamp(getattr(token_data, 'launch_time', time.time())),
                    confidence_score=0.0,
                    risk_score=getattr(safety_analysis, 'risk_score', 1.0),
                    metadata={
                        "safety_analysis": safety_analysis,
                        "token_data": token_data
                    }
                )
                
                opportunities.append(opportunity)
                
            except Exception as e:
                logger.error(f"Error analyzing token {token.get('address', 'unknown')}: {e}")
                continue
        
        return opportunities
    
    async def _make_trading_decisions(self, opportunities: List[TokenOpportunity]) -> List[TradingDecision]:
        """Make AI-powered trading decisions"""
        decisions = []
        
        for opportunity in opportunities:
            try:
                # Create AI prompt for analysis
                prompt = self._create_analysis_prompt(opportunity)
                
                # Get AI analysis
                ai_response = None
                if self.model_factory:
                    ai_response = await self.model_factory.generate_ensemble_response(prompt)
                
                if not ai_response:
                    continue
                
                # Parse AI decision
                decision = await self._parse_ai_decision(opportunity, ai_response)
                
                if decision:
                    decisions.append(decision)
                    
                    # Update opportunity with AI analysis
                    opportunity.ai_analysis = {
                        "response": ai_response.text,
                        "confidence": ai_response.confidence,
                        "reasoning": decision.reasoning
                    }
                    opportunity.confidence_score = decision.confidence
                
            except Exception as e:
                logger.error(f"Error making decision for {opportunity.token_address}: {e}")
                continue
        
        return decisions
    
    async def _execute_trades(self, decisions: List[TradingDecision]):
        """Execute trading decisions"""
        for decision in decisions:
            try:
                if decision.action == "buy" and decision.confidence >= 0.7:
                    # Execute buy order
                    success = False
                    if self.sniper_bot:
                        success = await self.sniper_bot.execute_buy_order(
                            token_address=decision.token_address,
                            amount=decision.amount,
                            max_price=decision.price * 1.05  # 5% slippage tolerance
                        )
                    
                    if success and self.notifications:
                        await self.notifications.send_notification(
                            "💰 Buy Order Executed",
                            f"Token: {decision.token_address}\nAmount: {decision.amount}\nConfidence: {decision.confidence:.2f}"
                        )
                    elif not success and self.notifications:
                        await self.notifications.send_notification(
                            "❌ Buy Order Failed",
                            f"Token: {decision.token_address}\nReason: Execution failed"
                        )
                
                elif decision.action == "sell":
                    # Execute sell order
                    success = False
                    if self.sniper_bot:
                        success = await self.sniper_bot.execute_sell_order(
                            token_address=decision.token_address,
                            amount=decision.amount
                        )
                    
                    if success and self.notifications:
                        await self.notifications.send_notification(
                            "💸 Sell Order Executed",
                            f"Token: {decision.token_address}\nAmount: {decision.amount}"
                        )
                
            except Exception as e:
                logger.error(f"Error executing trade for {decision.token_address}: {e}")
    
    def _meets_basic_criteria(self, token: Dict[str, Any]) -> bool:
        """Check if token meets basic criteria"""
        volume = token.get("volume_24h", 0)
        liquidity = token.get("liquidity", 0)
        
        return volume >= self.min_volume and liquidity >= self.min_liquidity
    
    def _deduplicate_tokens(self, tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate tokens"""
        seen = set()
        unique_tokens = []
        
        for token in tokens:
            address = token.get("address")
            if address and address not in seen:
                seen.add(address)
                unique_tokens.append(token)
        
        return unique_tokens
    
    def _create_analysis_prompt(self, opportunity: TokenOpportunity) -> str:
        """Create AI prompt for token analysis"""
        safety_analysis = opportunity.metadata.get('safety_analysis', {}) if opportunity.metadata else {}
        
        return f"""
        Analyze this Solana token for trading potential:
        
        Token: {opportunity.token_name} ({opportunity.token_symbol})
        Address: {opportunity.token_address}
        Price: ${opportunity.price:.8f}
        Volume 24h: ${opportunity.volume_24h:,.0f}
        Liquidity: ${opportunity.liquidity:,.0f}
        Market Cap: ${opportunity.market_cap:,.0f}
        Launch Time: {opportunity.launch_time}
        Risk Score: {opportunity.risk_score:.2f}
        
        Safety Analysis:
        {json.dumps(safety_analysis, indent=2)}
        
        Based on this data, provide a trading recommendation:
        1. Action: [buy/sell/hold/skip]
        2. Confidence: [0.0-1.0]
        3. Reasoning: [detailed explanation]
        4. Risk Assessment: [key risks to consider]
        5. Position Size: [recommended % of portfolio]
        
        Consider:
        - Token fundamentals and metrics
        - Market timing and momentum
        - Risk vs reward potential
        - Liquidity and volume patterns
        - Safety concerns and red flags
        """
    
    async def _parse_ai_decision(self, opportunity: TokenOpportunity, ai_response: ModelResponse) -> Optional[TradingDecision]:
        """Parse AI response into trading decision"""
        try:
            # Simple parsing - in production, use more sophisticated NLP
            text = ai_response.text.lower()
            
            # Determine action
            if "buy" in text and "sell" not in text:
                action = "buy"
            elif "sell" in text:
                action = "sell"
            elif "hold" in text:
                action = "hold"
            else:
                action = "skip"
            
            # Extract confidence (look for numbers 0.0-1.0)
            import re
            confidence_match = re.search(r'confidence:\s*([0-9]*\.?[0-9]+)', text)
            confidence = float(confidence_match.group(1)) if confidence_match else ai_response.confidence
            
            # Extract position size
            size_match = re.search(r'position size:\s*([0-9]*\.?[0-9]+)%', text)
            position_size = float(size_match.group(1)) / 100 if size_match else self.max_position_size
            
            # Calculate amount
            amount = position_size * opportunity.price
            
            # Get safety analysis safely
            safety_analysis = {}
            if opportunity.metadata:
                safety_analysis = opportunity.metadata.get("safety_analysis", {})
            
            return TradingDecision(
                action=action,
                token_address=opportunity.token_address,
                amount=amount,
                price=opportunity.price,
                confidence=confidence,
                reasoning=ai_response.text,
                risk_assessment=safety_analysis,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error parsing AI decision: {e}")
            return None
    
    async def _update_metrics(self):
        """Update performance metrics"""
        # This would typically update metrics in a database
        # For now, just log basic stats
        logger.info(f"Performance: {self.performance_metrics}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the sniper agent"""
        return {
            "is_running": self.is_running,
            "enabled": self.enabled,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "last_scan_time": self.last_scan_time.isoformat() if self.last_scan_time else None,
            "opportunities_found": len(self.opportunities),
            "decisions_made": len(self.decisions),
            "performance_metrics": self.performance_metrics,
            "available_models": self.model_factory.get_available_models() if self.model_factory else []
        } 