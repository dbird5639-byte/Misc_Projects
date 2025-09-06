"""
Main Phone Agent class for handling automated phone calls
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Import local modules with error handling
try:
    from .voice_processor import VoiceProcessor  # type: ignore
    from .conversation_manager import ConversationManager  # type: ignore
    from ..utils.knowledge_utils import KnowledgeBase  # type: ignore
    from ..utils.audio_utils import AudioHandler  # type: ignore
    from ..utils.twilio_utils import TwilioHandler  # type: ignore
    IMPORTS_AVAILABLE = True
except ImportError:
    # Create placeholder classes for demonstration
    class VoiceProcessor:
        def __init__(self): pass
        def initialize(self): return True
        def speech_to_text(self, audio): return "Hello, I need help"
        def text_to_speech(self, text): return b"audio_data"
        def cleanup(self): pass
    
    class ConversationManager:
        def __init__(self): pass
        def generate_response(self, text, context=None): return "I can help you with that."
    
    class KnowledgeBase:
        def __init__(self): pass
        def get_response(self, query): return "Here's the information you requested."
    
    class AudioHandler:
        def __init__(self): pass
        def play_audio(self, audio): return True
        def cleanup(self): pass
    
    class TwilioHandler:
        def __init__(self): pass
    
    IMPORTS_AVAILABLE = False


class PhoneAgent:
    """
    Main AI Phone Agent that coordinates voice processing, conversation management,
    and knowledge retrieval to provide automated customer service.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Phone Agent with all necessary components.
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.session_id: Optional[str] = None
        self.call_start_time: Optional[datetime] = None
        self.conversation_history = []
        self.current_context = {}
        
        # Initialize components
        self.voice_processor = VoiceProcessor()
        self.conversation_manager = ConversationManager()
        self.knowledge_base = KnowledgeBase()
        self.audio_handler = AudioHandler()
        self.twilio_handler = TwilioHandler()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize state
        self.is_active = False
        self.call_duration = 0
        self.turn_count = 0
        
        self.logger.info("Phone Agent initialized successfully")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from file or use defaults."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                "greeting": "Hello! Thank you for calling. How can I help you today?",
                "goodbye": "Thank you for calling. Have a great day!",
                "fallback": "I'm sorry, I didn't understand that. Could you please repeat?",
                "max_turns": 20,
                "timeout": 300
            }
    
    def start_call(self, call_id: str, caller_number: Optional[str] = None) -> bool:
        """
        Start a new phone call session.
        
        Args:
            call_id: Unique identifier for the call
            caller_number: Phone number of the caller
            
        Returns:
            bool: True if call started successfully
        """
        try:
            self.session_id = call_id
            self.call_start_time = datetime.now()
            self.is_active = True
            self.conversation_history = []
            self.current_context = {
                "call_id": call_id,
                "caller_number": caller_number,
                "start_time": self.call_start_time.isoformat()
            }
            
            # Initialize voice processing
            self.voice_processor.initialize()
            
            # Play greeting
            greeting_response = self._generate_response(self.config["greeting"])
            self._speak_response(greeting_response)
            
            self.logger.info(f"Call started: {call_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start call: {e}")
            return False
    
    def process_audio_input(self, audio_data: bytes) -> Optional[str]:
        """
        Process incoming audio and generate a response.
        
        Args:
            audio_data: Raw audio data from the caller
            
        Returns:
            str: Generated response text, or None if processing failed
        """
        try:
            # Convert speech to text
            text_input = self.voice_processor.speech_to_text(audio_data)
            if not text_input:
                return self.config["fallback"]
            
            # Generate AI response
            response = self._generate_response(text_input)
            
            # Update conversation history
            self._update_conversation_history(text_input, response)
            
            # Check for call end conditions
            if self._should_end_call(text_input):
                return self._end_call()
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing audio input: {e}")
            return self.config["fallback"]
    
    def _generate_response(self, user_input: str) -> str:
        """
        Generate an appropriate response using AI and knowledge base.
        
        Args:
            user_input: Text input from the user
            
        Returns:
            str: Generated response
        """
        try:
            # Check knowledge base first
            knowledge_response = self.knowledge_base.get_response(user_input)
            if knowledge_response:
                return knowledge_response
            
            # Use conversation manager for context-aware responses
            context = self._get_conversation_context()
            response = self.conversation_manager.generate_response(
                user_input, context
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return self.config["fallback"]
    
    def _speak_response(self, response_text: str) -> bool:
        """
        Convert response text to speech and play it.
        
        Args:
            response_text: Text to convert to speech
            
        Returns:
            bool: True if speech was generated successfully
        """
        try:
            audio_data = self.voice_processor.text_to_speech(response_text)
            if audio_data:
                self.audio_handler.play_audio(audio_data)
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error generating speech: {e}")
            return False
    
    def _update_conversation_history(self, user_input: str, response: str):
        """Update the conversation history with new exchange."""
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "agent_response": response,
            "turn_number": self.turn_count
        }
        
        self.conversation_history.append(exchange)
        self.turn_count += 1
    
    def _get_conversation_context(self) -> Dict:
        """Get current conversation context for AI processing."""
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "recent_exchanges": self.conversation_history[-5:],  # Last 5 exchanges
            "call_duration": self.call_duration,
            "current_context": self.current_context
        }
    
    def _should_end_call(self, user_input: str) -> bool:
        """Check if the call should be ended based on user input or conditions."""
        # Check for goodbye keywords
        goodbye_keywords = ["goodbye", "bye", "end call", "hang up", "thank you"]
        if any(keyword in user_input.lower() for keyword in goodbye_keywords):
            return True
        
        # Check turn limit
        if self.turn_count >= self.config.get("max_turns", 20):
            return True
        
        # Check timeout
        if self.call_duration > self.config.get("timeout", 300):
            return True
        
        return False
    
    def _end_call(self) -> str:
        """End the current call and return goodbye message."""
        self.is_active = False
        if self.call_start_time:
            self.call_duration = datetime.now() - self.call_start_time
        
        # Save conversation
        self._save_conversation()
        
        # Cleanup
        self.voice_processor.cleanup()
        self.audio_handler.cleanup()
        
        self.logger.info(f"Call ended: {self.session_id}, Duration: {self.call_duration}")
        return self.config["goodbye"]
    
    def _save_conversation(self):
        """Save the conversation to file for analysis."""
        try:
            conversation_data = {
                "session_id": self.session_id,
                "start_time": self.call_start_time.isoformat() if self.call_start_time else None,
                "end_time": datetime.now().isoformat(),
                "duration": str(self.call_duration),
                "turn_count": self.turn_count,
                "conversation": self.conversation_history,
                "context": self.current_context
            }
            
            # Save to file
            output_path = Path("data/conversations") / f"{self.session_id}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(conversation_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving conversation: {e}")
    
    def get_call_status(self) -> Dict:
        """Get current call status and statistics."""
        return {
            "session_id": self.session_id,
            "is_active": self.is_active,
            "turn_count": self.turn_count,
            "call_duration": str(self.call_duration) if self.call_duration else "0:00",
            "start_time": self.call_start_time.isoformat() if self.call_start_time else None
        }
    
    def update_context(self, key: str, value: str):
        """Update the current conversation context."""
        self.current_context[key] = value
    
    def emergency_stop(self):
        """Emergency stop the call and cleanup resources."""
        self.logger.warning("Emergency stop called")
        self.is_active = False
        
        # Play emergency message
        emergency_message = "I apologize, but I need to end this call. Please call back later."
        self._speak_response(emergency_message)
        
        # Cleanup
        self.voice_processor.cleanup()
        self.audio_handler.cleanup()
        
        # Save conversation
        self._save_conversation() 