"""
AI Model Factory for managing different AI models and configurations
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import json

# AI model imports
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    """Standardized model response"""
    text: str
    confidence: float
    model_name: str
    response_time: float
    tokens_used: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseAIModel(ABC):
    """Abstract base class for AI models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("name", "unknown")
        self.enabled = config.get("enabled", True)
        self.max_tokens = config.get("max_tokens", 1000)
        self.temperature = config.get("temperature", 0.7)
        self.is_loaded = False
        
    @abstractmethod
    async def load_model(self) -> bool:
        """Load the AI model"""
        pass
    
    @abstractmethod
    async def generate_response(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response from the model"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if model is available"""
        pass


class LocalLLMModel(BaseAIModel):
    """Local language model implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_path = config.get("model_path")
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
    def is_available(self) -> bool:
        """Check if local model is available"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available for local models")
            return False
        return self.model_path is not None
    
    async def load_model(self) -> bool:
        """Load local model asynchronously"""
        if not self.is_available():
            return False
            
        try:
            logger.info(f"Loading local model: {self.model_name}")
            
            # Load model in thread to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model_sync)
            
            self.is_loaded = True
            logger.info(f"Successfully loaded local model: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load local model {self.model_name}: {e}")
            return False
    
    def _load_model_sync(self):
        """Load model synchronously (run in thread)"""
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
        except Exception as e:
            logger.error(f"Error loading model {self.model_path}: {e}")
            raise
    
    async def generate_response(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response from local model"""
        if not self.is_loaded:
            await self.load_model()
        
        start_time = time.time()
        
        try:
            # Generate response
            response = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_sync, prompt, kwargs
            )
            
            response_time = time.time() - start_time
            
            return ModelResponse(
                text=response["generated_text"],
                confidence=response.get("confidence", 0.8),
                model_name=self.model_name,
                response_time=response_time,
                tokens_used=response.get("tokens_used"),
                metadata={"model_type": "local"}
            )
            
        except Exception as e:
            logger.error(f"Error generating response from {self.model_name}: {e}")
            return ModelResponse(
                text="Error generating response",
                confidence=0.0,
                model_name=self.model_name,
                response_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    def _generate_sync(self, prompt: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response synchronously"""
        if not self.tokenizer or not self.pipeline:
            return {
                "generated_text": "Error: Model not properly loaded",
                "confidence": 0.0,
                "tokens_used": 0
            }
        
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        # Generate
        outputs = self.pipeline(prompt, **generation_kwargs)
        
        # Extract generated text
        generated_text = outputs[0]["generated_text"]
        
        return {
            "generated_text": generated_text,
            "confidence": 0.8,  # Default confidence for local models
            "tokens_used": len(self.tokenizer.encode(generated_text))
        }


class OpenAIModel(BaseAIModel):
    """OpenAI model implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.model = config.get("model", "gpt-4")
        self.client = None
        
    def is_available(self) -> bool:
        """Check if OpenAI is available"""
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI not available")
            return False
        return self.api_key is not None
    
    async def load_model(self) -> bool:
        """Load OpenAI model (just initialize client)"""
        if not self.is_available():
            return False
            
        try:
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
            self.is_loaded = True
            logger.info(f"OpenAI client initialized for {self.model}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            return False
    
    async def generate_response(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response from OpenAI"""
        if not self.is_loaded:
            await self.load_model()
        
        if not self.client:
            return ModelResponse(
                text="Error: OpenAI client not initialized",
                confidence=0.0,
                model_name=self.model_name,
                response_time=0.0,
                metadata={"error": "client_not_initialized"}
            )
        
        start_time = time.time()
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature)
            )
            
            response_time = time.time() - start_time
            
            return ModelResponse(
                text=response.choices[0].message.content,
                confidence=0.9,  # High confidence for OpenAI
                model_name=self.model_name,
                response_time=response_time,
                tokens_used=response.usage.total_tokens,
                metadata={"model_type": "openai", "model": self.model}
            )
            
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {e}")
            return ModelResponse(
                text="Error generating response",
                confidence=0.0,
                model_name=self.model_name,
                response_time=time.time() - start_time,
                metadata={"error": str(e)}
            )


class AnthropicModel(BaseAIModel):
    """Anthropic Claude model implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.model = config.get("model", "claude-3-sonnet")
        self.client = None
        
    def is_available(self) -> bool:
        """Check if Anthropic is available"""
        if not ANTHROPIC_AVAILABLE:
            logger.warning("Anthropic not available")
            return False
        return self.api_key is not None
    
    async def load_model(self) -> bool:
        """Load Anthropic model (just initialize client)"""
        if not self.is_available():
            return False
            
        try:
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
            self.is_loaded = True
            logger.info(f"Anthropic client initialized for {self.model}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            return False
    
    async def generate_response(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response from Anthropic"""
        if not self.is_loaded:
            await self.load_model()
        
        if not self.client:
            return ModelResponse(
                text="Error: Anthropic client not initialized",
                confidence=0.0,
                model_name=self.model_name,
                response_time=0.0,
                metadata={"error": "client_not_initialized"}
            )
        
        start_time = time.time()
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_time = time.time() - start_time
            
            return ModelResponse(
                text=response.content[0].text,
                confidence=0.9,  # High confidence for Anthropic
                model_name=self.model_name,
                response_time=response_time,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                metadata={"model_type": "anthropic", "model": self.model}
            )
            
        except Exception as e:
            logger.error(f"Error generating Anthropic response: {e}")
            return ModelResponse(
                text="Error generating response",
                confidence=0.0,
                model_name=self.model_name,
                response_time=time.time() - start_time,
                metadata={"error": str(e)}
            )


class ModelFactory:
    """Factory for managing AI models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models: Dict[str, BaseAIModel] = {}
        self.ensemble_weighting = config.get("ensemble_weighting", {})
        self.decision_threshold = config.get("decision_threshold", 0.7)
        
    async def initialize_models(self):
        """Initialize all configured models"""
        logger.info("Initializing AI models...")
        
        # Initialize local models
        local_models = self.config.get("local_models", {})
        for model_name, model_config in local_models.items():
            if model_config.get("enabled", False):
                model = LocalLLMModel({
                    "name": model_name,
                    **model_config
                })
                if model.is_available():
                    self.models[model_name] = model
                    logger.info(f"Added local model: {model_name}")
        
        # Initialize OpenAI models
        cloud_models = self.config.get("cloud_models", {})
        for model_name, model_config in cloud_models.items():
            if model_config.get("enabled", False):
                if model_name == "openai":
                    model = OpenAIModel({
                        "name": model_name,
                        **model_config
                    })
                elif model_name == "anthropic":
                    model = AnthropicModel({
                        "name": model_name,
                        **model_config
                    })
                else:
                    continue
                
                if model.is_available():
                    self.models[model_name] = model
                    logger.info(f"Added cloud model: {model_name}")
        
        # Load all models
        await self._load_all_models()
        
        logger.info(f"Initialized {len(self.models)} models")
    
    async def _load_all_models(self):
        """Load all models in parallel"""
        tasks = []
        for model in self.models.values():
            tasks.append(model.load_model())
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (model_name, result) in enumerate(zip(self.models.keys(), results)):
            if isinstance(result, Exception):
                logger.error(f"Failed to load model {model_name}: {result}")
    
    async def generate_ensemble_response(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response using ensemble of models"""
        if not self.models:
            return ModelResponse(
                text="No models available",
                confidence=0.0,
                model_name="ensemble",
                response_time=0.0
            )
        
        # Generate responses from all models in parallel
        tasks = []
        for model in self.models.values():
            tasks.append(model.generate_response(prompt, **kwargs))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors and calculate ensemble
        valid_responses = []
        for response in responses:
            if isinstance(response, ModelResponse) and response.confidence > 0:
                valid_responses.append(response)
        
        if not valid_responses:
            return ModelResponse(
                text="All models failed to generate response",
                confidence=0.0,
                model_name="ensemble",
                response_time=0.0
            )
        
        # Calculate weighted ensemble
        total_weight = 0
        weighted_text = ""
        total_confidence = 0
        
        for response in valid_responses:
            weight = self.ensemble_weighting.get(response.model_name, 1.0)
            total_weight += weight
            weighted_text += f"[{response.model_name}]: {response.text}\n"
            total_confidence += response.confidence * weight
        
        avg_confidence = total_confidence / total_weight if total_weight > 0 else 0
        
        # If confidence is high enough, use ensemble
        if avg_confidence >= self.decision_threshold:
            return ModelResponse(
                text=weighted_text.strip(),
                confidence=avg_confidence,
                model_name="ensemble",
                response_time=max(r.response_time for r in valid_responses),
                metadata={"ensemble_size": len(valid_responses)}
            )
        else:
            # Return best individual response
            best_response = max(valid_responses, key=lambda r: r.confidence)
            return best_response
    
    async def generate_response(self, prompt: str, model_name: Optional[str] = None, **kwargs) -> ModelResponse:
        """Generate response from specific model or ensemble"""
        if model_name and model_name in self.models:
            return await self.models[model_name].generate_response(prompt, **kwargs)
        else:
            return await self.generate_ensemble_response(prompt, **kwargs)
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return list(self.models.keys())
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        status = {}
        for name, model in self.models.items():
            status[name] = {
                "enabled": model.enabled,
                "loaded": model.is_loaded,
                "available": model.is_available()
            }
        return status 