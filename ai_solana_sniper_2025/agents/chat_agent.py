"""
Chat Agent - Handles AI interactions and decision making
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging

from .model_factory import ModelFactory, ModelResponse
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ChatMessage:
    """Represents a chat message"""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AnalysisRequest:
    """Represents an analysis request"""
    request_type: str  # "token_analysis", "market_analysis", "risk_assessment", "strategy_review"
    data: Dict[str, Any]
    priority: int = 1  # 1-5, higher is more important
    timeout: int = 30  # seconds


@dataclass
class AnalysisResult:
    """Represents an analysis result"""
    request_id: str
    analysis_type: str
    result: Dict[str, Any]
    confidence: float
    reasoning: str
    model_used: str
    processing_time: float
    timestamp: datetime


class ChatAgent:
    """
    AI interaction and decision making agent
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.max_conversation_length = config.get("max_conversation_length", 50)
        self.response_timeout = config.get("response_timeout", 30)
        self.parallel_processing = config.get("parallel_processing", True)
        
        # Initialize components
        self.model_factory = None
        
        # State management
        self.conversation_history: List[ChatMessage] = []
        self.pending_requests: List[AnalysisRequest] = []
        self.completed_analyses: List[AnalysisResult] = []
        self.is_processing = False
        
        # Performance tracking
        self.total_requests = 0
        self.successful_analyses = 0
        self.failed_analyses = 0
        self.avg_response_time = 0.0
        
    async def initialize(self, model_factory: ModelFactory):
        """Initialize the chat agent"""
        logger.info("Initializing Chat Agent...")
        
        try:
            self.model_factory = model_factory
            
            # Add system message
            system_message = self._create_system_message()
            self.conversation_history.append(system_message)
            
            logger.info("Chat Agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Chat Agent: {e}")
            return False
    
    def _create_system_message(self) -> ChatMessage:
        """Create the initial system message"""
        system_prompt = """You are an AI trading assistant specialized in Solana meme coin analysis. Your role is to:

1. Analyze token opportunities and provide trading recommendations
2. Assess market conditions and sentiment
3. Evaluate risk factors and potential red flags
4. Provide strategic insights for trading decisions
5. Learn from past trades and improve recommendations

Key capabilities:
- Token fundamental analysis
- Technical analysis and pattern recognition
- Risk assessment and safety evaluation
- Market sentiment analysis
- Portfolio optimization suggestions

Always provide:
- Clear action recommendations (buy/sell/hold/skip)
- Confidence scores (0.0-1.0)
- Detailed reasoning
- Risk assessments
- Position sizing recommendations

Be concise but thorough in your analysis."""
        
        return ChatMessage(
            role="system",
            content=system_prompt,
            timestamp=datetime.now()
        )
    
    async def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a user message and get AI response"""
        if not self.enabled:
            return "Chat Agent is disabled"
        
        try:
            # Add user message
            user_message = ChatMessage(
                role="user",
                content=content,
                timestamp=datetime.now(),
                metadata=metadata
            )
            self.conversation_history.append(user_message)
            
            # Get AI response
            response = await self._generate_response()
            
            # Add assistant response
            assistant_message = ChatMessage(
                role="assistant",
                content=response.text,
                timestamp=datetime.now(),
                metadata={"confidence": response.confidence, "model": response.model_name}
            )
            self.conversation_history.append(assistant_message)
            
            # Trim conversation if too long
            self._trim_conversation()
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error processing user message: {e}")
            return f"Error: {str(e)}"
    
    async def analyze_token(self, token_data: Dict[str, Any]) -> AnalysisResult:
        """Analyze a specific token"""
        request = AnalysisRequest(
            request_type="token_analysis",
            data=token_data,
            priority=3,
            timeout=30
        )
        
        return await self._process_analysis_request(request)
    
    async def analyze_market_conditions(self, market_data: Dict[str, Any]) -> AnalysisResult:
        """Analyze current market conditions"""
        request = AnalysisRequest(
            request_type="market_analysis",
            data=market_data,
            priority=2,
            timeout=45
        )
        
        return await self._process_analysis_request(request)
    
    async def assess_risk(self, risk_data: Dict[str, Any]) -> AnalysisResult:
        """Assess risk factors"""
        request = AnalysisRequest(
            request_type="risk_assessment",
            data=risk_data,
            priority=4,
            timeout=20
        )
        
        return await self._process_analysis_request(request)
    
    async def review_strategy(self, strategy_data: Dict[str, Any]) -> AnalysisResult:
        """Review trading strategy"""
        request = AnalysisRequest(
            request_type="strategy_review",
            data=strategy_data,
            priority=1,
            timeout=60
        )
        
        return await self._process_analysis_request(request)
    
    async def _process_analysis_request(self, request: AnalysisRequest) -> AnalysisResult:
        """Process an analysis request"""
        start_time = datetime.now()
        request_id = f"{request.request_type}_{int(start_time.timestamp())}"
        
        try:
            # Check if model factory is available
            if not self.model_factory:
                raise Exception("Model factory not initialized")
            
            # Create analysis prompt
            prompt = self._create_analysis_prompt(request)
            
            # Get AI response
            response = await self.model_factory.generate_ensemble_response(
                prompt,
                max_tokens=1000,
                temperature=0.7
            )
            
            # Parse response
            result = self._parse_analysis_response(request.request_type, response)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            analysis_result = AnalysisResult(
                request_id=request_id,
                analysis_type=request.request_type,
                result=result,
                confidence=response.confidence,
                reasoning=response.text,
                model_used=response.model_name,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
            self.completed_analyses.append(analysis_result)
            self.successful_analyses += 1
            
            # Update performance metrics
            self._update_performance_metrics(processing_time)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error processing analysis request: {e}")
            self.failed_analyses += 1
            
            return AnalysisResult(
                request_id=request_id,
                analysis_type=request.request_type,
                result={"error": str(e)},
                confidence=0.0,
                reasoning=f"Analysis failed: {str(e)}",
                model_used="none",
                processing_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now()
            )
    
    def _create_analysis_prompt(self, request: AnalysisRequest) -> str:
        """Create analysis prompt based on request type"""
        base_prompt = "Based on the following data, provide a detailed analysis:\n\n"
        
        if request.request_type == "token_analysis":
            return base_prompt + f"""
            Token Analysis Request:
            
            Token Data:
            {json.dumps(request.data, indent=2)}
            
            Please provide:
            1. Token fundamentals assessment
            2. Trading potential evaluation
            3. Risk factors identification
            4. Recommended action (buy/sell/hold/skip)
            5. Confidence score (0.0-1.0)
            6. Position sizing recommendation
            7. Key considerations and warnings
            """
        
        elif request.request_type == "market_analysis":
            return base_prompt + f"""
            Market Analysis Request:
            
            Market Data:
            {json.dumps(request.data, indent=2)}
            
            Please provide:
            1. Overall market sentiment
            2. Trend analysis
            3. Volatility assessment
            4. Opportunity identification
            5. Risk level evaluation
            6. Strategic recommendations
            7. Market timing insights
            """
        
        elif request.request_type == "risk_assessment":
            return base_prompt + f"""
            Risk Assessment Request:
            
            Risk Data:
            {json.dumps(request.data, indent=2)}
            
            Please provide:
            1. Risk factor identification
            2. Risk level classification (low/medium/high)
            3. Specific threats and concerns
            4. Mitigation strategies
            5. Risk vs reward analysis
            6. Safety recommendations
            7. Red flags to watch for
            """
        
        elif request.request_type == "strategy_review":
            return base_prompt + f"""
            Strategy Review Request:
            
            Strategy Data:
            {json.dumps(request.data, indent=2)}
            
            Please provide:
            1. Strategy effectiveness evaluation
            2. Performance analysis
            3. Strengths and weaknesses
            4. Optimization suggestions
            5. Risk management review
            6. Adaptation recommendations
            7. Long-term viability assessment
            """
        
        else:
            return base_prompt + f"General Analysis Request:\n\n{json.dumps(request.data, indent=2)}"
    
    def _parse_analysis_response(self, analysis_type: str, response: ModelResponse) -> Dict[str, Any]:
        """Parse AI response into structured result"""
        try:
            # Simple parsing - in production, use more sophisticated NLP
            text = response.text.lower()
            
            result = {
                "raw_response": response.text,
                "confidence": response.confidence,
                "model_used": response.model_name
            }
            
            # Extract key information based on analysis type
            if analysis_type == "token_analysis":
                result.update(self._parse_token_analysis(text))
            elif analysis_type == "market_analysis":
                result.update(self._parse_market_analysis(text))
            elif analysis_type == "risk_assessment":
                result.update(self._parse_risk_assessment(text))
            elif analysis_type == "strategy_review":
                result.update(self._parse_strategy_review(text))
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing analysis response: {e}")
            return {"error": str(e), "raw_response": response.text}
    
    def _parse_token_analysis(self, text: str) -> Dict[str, Any]:
        """Parse token analysis response"""
        import re
        
        result = {}
        
        # Extract action
        if "buy" in text and "sell" not in text:
            result["recommended_action"] = "buy"
        elif "sell" in text:
            result["recommended_action"] = "sell"
        elif "hold" in text:
            result["recommended_action"] = "hold"
        else:
            result["recommended_action"] = "skip"
        
        # Extract confidence
        confidence_match = re.search(r'confidence:\s*([0-9]*\.?[0-9]+)', text)
        result["confidence_score"] = float(confidence_match.group(1)) if confidence_match else 0.5
        
        # Extract position size
        size_match = re.search(r'position size:\s*([0-9]*\.?[0-9]+)%', text)
        result["position_size"] = float(size_match.group(1)) / 100 if size_match else 0.05
        
        return result
    
    def _parse_market_analysis(self, text: str) -> Dict[str, Any]:
        """Parse market analysis response"""
        result = {}
        
        # Extract sentiment
        if "bullish" in text:
            result["sentiment"] = "bullish"
        elif "bearish" in text:
            result["sentiment"] = "bearish"
        else:
            result["sentiment"] = "neutral"
        
        # Extract volatility
        if "high volatility" in text or "volatile" in text:
            result["volatility"] = "high"
        elif "low volatility" in text:
            result["volatility"] = "low"
        else:
            result["volatility"] = "medium"
        
        return result
    
    def _parse_risk_assessment(self, text: str) -> Dict[str, Any]:
        """Parse risk assessment response"""
        result = {}
        
        # Extract risk level
        if "high risk" in text or "very risky" in text:
            result["risk_level"] = "high"
        elif "low risk" in text or "safe" in text:
            result["risk_level"] = "low"
        else:
            result["risk_level"] = "medium"
        
        # Extract red flags
        red_flags = []
        if "honeypot" in text:
            red_flags.append("potential_honeypot")
        if "rug pull" in text:
            red_flags.append("rug_pull_risk")
        if "low liquidity" in text:
            red_flags.append("low_liquidity")
        
        result["red_flags"] = red_flags
        
        return result
    
    def _parse_strategy_review(self, text: str) -> Dict[str, Any]:
        """Parse strategy review response"""
        result = {}
        
        # Extract effectiveness
        if "effective" in text and "ineffective" not in text:
            result["effectiveness"] = "effective"
        elif "ineffective" in text:
            result["effectiveness"] = "ineffective"
        else:
            result["effectiveness"] = "moderate"
        
        return result
    
    async def _generate_response(self) -> ModelResponse:
        """Generate response based on conversation history"""
        # Check if model factory is available
        if not self.model_factory:
            return ModelResponse(
                text="Error: Model factory not initialized. Please ensure the chat agent is properly initialized.",
                confidence=0.0,
                model_name="none",
                response_time=0.0
            )
        
        # Build conversation context
        context = self._build_conversation_context()
        
        # Generate response
        response = await self.model_factory.generate_ensemble_response(
            context,
            max_tokens=500,
            temperature=0.7
        )
        
        return response
    
    def _build_conversation_context(self) -> str:
        """Build conversation context from history"""
        context = ""
        
        # Include last few messages for context
        recent_messages = self.conversation_history[-10:]  # Last 10 messages
        
        for message in recent_messages:
            context += f"{message.role.upper()}: {message.content}\n\n"
        
        return context.strip()
    
    def _trim_conversation(self):
        """Trim conversation history if too long"""
        if len(self.conversation_history) > self.max_conversation_length:
            # Keep system message and recent messages
            system_message = self.conversation_history[0]
            recent_messages = self.conversation_history[-self.max_conversation_length+1:]
            self.conversation_history = [system_message] + recent_messages
    
    def _update_performance_metrics(self, processing_time: float):
        """Update performance metrics"""
        self.total_requests += 1
        
        # Update average response time
        if self.avg_response_time == 0:
            self.avg_response_time = processing_time
        else:
            self.avg_response_time = (self.avg_response_time + processing_time) / 2
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata
            }
            for msg in self.conversation_history
        ]
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get analysis history"""
        return [
            {
                "request_id": analysis.request_id,
                "analysis_type": analysis.analysis_type,
                "result": analysis.result,
                "confidence": analysis.confidence,
                "model_used": analysis.model_used,
                "processing_time": analysis.processing_time,
                "timestamp": analysis.timestamp.isoformat()
            }
            for analysis in self.completed_analyses[-50:]  # Last 50 analyses
        ]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        success_rate = self.successful_analyses / max(self.total_requests, 1)
        
        return {
            "total_requests": self.total_requests,
            "successful_analyses": self.successful_analyses,
            "failed_analyses": self.failed_analyses,
            "success_rate": success_rate,
            "avg_response_time": self.avg_response_time,
            "conversation_length": len(self.conversation_history)
        } 