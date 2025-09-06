"""
Conversation management module for handling AI conversation flow and context
"""

import logging
from typing import Dict, List, Optional, Any
import json


class ConversationManager:
    """
    Manages conversation flow, context, and AI response generation.
    """
    
    def __init__(self):
        """Initialize the conversation manager."""
        self.logger = logging.getLogger(__name__)
        self.conversation_history = []
        self.context = {}
        self.personality = {
            "name": "AI Assistant",
            "tone": "professional",
            "style": "helpful",
            "expertise": "customer service"
        }
        
    def generate_response(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate an appropriate response to user input.
        
        Args:
            user_input: The user's input text
            context: Additional context for the conversation
            
        Returns:
            str: Generated response
        """
        try:
            # Update context
            if context:
                self.context.update(context)
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": self._get_timestamp()
            })
            
            # Generate response based on input type
            response = self._process_input(user_input)
            
            # Add response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": self._get_timestamp()
            })
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble processing your request. Could you please repeat that?"
    
    def _process_input(self, user_input: str) -> str:
        """
        Process user input and generate appropriate response.
        
        Args:
            user_input: The user's input text
            
        Returns:
            str: Generated response
        """
        input_lower = user_input.lower()
        
        # Check for common patterns
        if self._is_greeting(input_lower):
            return self._generate_greeting()
        elif self._is_goodbye(input_lower):
            return self._generate_goodbye()
        elif self._is_help_request(input_lower):
            return self._generate_help_response()
        elif self._is_complaint(input_lower):
            return self._generate_complaint_response()
        elif self._is_question(input_lower):
            return self._generate_question_response(user_input)
        else:
            return self._generate_general_response(user_input)
    
    def _is_greeting(self, text: str) -> bool:
        """Check if input is a greeting."""
        greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
        return any(greeting in text for greeting in greetings)
    
    def _is_goodbye(self, text: str) -> bool:
        """Check if input is a goodbye."""
        goodbyes = ["goodbye", "bye", "see you", "thank you", "thanks", "end call"]
        return any(goodbye in text for goodbye in goodbyes)
    
    def _is_help_request(self, text: str) -> bool:
        """Check if input is a help request."""
        help_keywords = ["help", "support", "assist", "problem", "issue", "trouble"]
        return any(keyword in text for keyword in help_keywords)
    
    def _is_complaint(self, text: str) -> bool:
        """Check if input is a complaint."""
        complaint_keywords = ["angry", "frustrated", "upset", "disappointed", "wrong", "broken"]
        return any(keyword in text for keyword in complaint_keywords)
    
    def _is_question(self, text: str) -> bool:
        """Check if input is a question."""
        question_words = ["what", "how", "when", "where", "why", "who", "which"]
        return any(word in text for word in question_words) or text.endswith("?")
    
    def _generate_greeting(self) -> str:
        """Generate a greeting response."""
        greetings = [
            "Hello! How can I help you today?",
            "Hi there! What can I assist you with?",
            "Good day! How may I help you?",
            "Hello! I'm here to help. What do you need?"
        ]
        return self._select_response(greetings)
    
    def _generate_goodbye(self) -> str:
        """Generate a goodbye response."""
        goodbyes = [
            "Thank you for calling. Have a great day!",
            "Goodbye! Feel free to call back if you need anything else.",
            "Thanks for reaching out. Take care!",
            "Have a wonderful day! Goodbye!"
        ]
        return self._select_response(goodbyes)
    
    def _generate_help_response(self) -> str:
        """Generate a help response."""
        help_responses = [
            "I'd be happy to help! What specific issue are you experiencing?",
            "I'm here to assist you. Can you tell me more about what you need help with?",
            "Let me help you with that. What seems to be the problem?",
            "I'm ready to help. What can I assist you with today?"
        ]
        return self._select_response(help_responses)
    
    def _generate_complaint_response(self) -> str:
        """Generate a response to complaints."""
        complaint_responses = [
            "I understand your frustration. Let me help you resolve this issue.",
            "I apologize for the inconvenience. Let's work together to fix this.",
            "I'm sorry to hear about your experience. How can I make this right?",
            "Thank you for bringing this to my attention. Let me help you with a solution."
        ]
        return self._select_response(complaint_responses)
    
    def _generate_question_response(self, question: str) -> str:
        """Generate a response to questions."""
        # In a real implementation, this would use AI to generate context-aware responses
        # For now, provide generic responses based on question type
        if "price" in question.lower() or "cost" in question.lower():
            return "Our pricing varies based on your needs. I can help you find the right plan for you."
        elif "time" in question.lower() or "hours" in question.lower():
            return "Our business hours are Monday through Friday, 9 AM to 6 PM Eastern Time."
        elif "contact" in question.lower():
            return "You can reach us by phone, email, or through our website. I'm happy to help you directly right now."
        else:
            return "That's a great question. Let me help you find the information you need."
    
    def _generate_general_response(self, input_text: str) -> str:
        """Generate a general response for other inputs."""
        general_responses = [
            "I understand. How can I help you with that?",
            "Thank you for sharing that. What would you like me to do?",
            "I see. Let me assist you with that.",
            "Got it. How can I be of help?"
        ]
        return self._select_response(general_responses)
    
    def _select_response(self, responses: List[str]) -> str:
        """Select a response from a list of options."""
        import random
        return random.choice(responses)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def update_context(self, key: str, value: Any):
        """
        Update the conversation context.
        
        Args:
            key: Context key
            value: Context value
        """
        self.context[key] = value
        self.logger.debug(f"Context updated: {key} = {value}")
    
    def get_context(self) -> Dict[str, Any]:
        """
        Get the current conversation context.
        
        Returns:
            Dict containing current context
        """
        return self.context.copy()
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.
        
        Returns:
            List of conversation exchanges
        """
        return self.conversation_history.copy()
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        self.logger.info("Conversation history cleared")
    
    def set_personality(self, personality: Dict[str, str]):
        """
        Set the AI personality.
        
        Args:
            personality: Personality configuration
        """
        self.personality.update(personality)
        self.logger.info(f"Personality updated: {personality}")
    
    def get_personality(self) -> Dict[str, str]:
        """
        Get the current AI personality.
        
        Returns:
            Dict containing personality settings
        """
        return self.personality.copy()
    
    def save_conversation(self, filepath: str):
        """
        Save the conversation to a file.
        
        Args:
            filepath: Path to save the conversation
        """
        try:
            conversation_data = {
                "personality": self.personality,
                "context": self.context,
                "history": self.conversation_history,
                "timestamp": self._get_timestamp()
            }
            
            with open(filepath, 'w') as f:
                json.dump(conversation_data, f, indent=2)
                
            self.logger.info(f"Conversation saved to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save conversation: {e}")
    
    def load_conversation(self, filepath: str) -> bool:
        """
        Load a conversation from a file.
        
        Args:
            filepath: Path to the conversation file
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            with open(filepath, 'r') as f:
                conversation_data = json.load(f)
            
            self.personality = conversation_data.get("personality", self.personality)
            self.context = conversation_data.get("context", {})
            self.conversation_history = conversation_data.get("history", [])
            
            self.logger.info(f"Conversation loaded from: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load conversation: {e}")
            return False 