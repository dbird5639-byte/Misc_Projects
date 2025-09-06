"""
Main API application for AI Phone Agent
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agent.phone_agent import PhoneAgent
from config.settings import API_SETTINGS, TESTING_SETTINGS


class PhoneAgentAPI:
    """
    Main API application for the AI Phone Agent system.
    """
    
    def __init__(self):
        """Initialize the API application."""
        self.logger = self._setup_logging()
        self.phone_agent = None
        self.is_running = False
        
        # Initialize phone agent
        self._initialize_phone_agent()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('data/logs/phone_agent.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def _initialize_phone_agent(self):
        """Initialize the phone agent."""
        try:
            self.phone_agent = PhoneAgent()
            self.logger.info("Phone agent initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize phone agent: {e}")
    
    def start_server(self):
        """Start the API server."""
        try:
            self.is_running = True
            self.logger.info(f"Starting AI Phone Agent API on {API_SETTINGS['host']}:{API_SETTINGS['port']}")
            
            if TESTING_SETTINGS['enabled']:
                self._run_test_mode()
            else:
                self._run_production_mode()
                
        except KeyboardInterrupt:
            self.logger.info("Server stopped by user")
        except Exception as e:
            self.logger.error(f"Server error: {e}")
        finally:
            self.stop_server()
    
    def _run_test_mode(self):
        """Run the application in test mode."""
        self.logger.info("Running in TEST MODE")
        print("\n" + "="*50)
        print("AI PHONE AGENT - TEST MODE")
        print("="*50)
        print("This is a simulation of the phone agent system.")
        print("Type 'quit' to exit, or enter test messages.\n")
        
        # Start a test call
        call_id = "test_call_001"
        if self.phone_agent and self.phone_agent.start_call(call_id):
            print("Test call started successfully!")
            print("Agent: Hello! Thank you for calling. How can I help you today?")
            
            # Interactive test loop
            while self.is_running:
                try:
                    user_input = input("You: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', 'bye']:
                        print("Agent: Thank you for calling. Have a great day!")
                        break
                    
                    if user_input and self.phone_agent:
                        # Simulate audio processing
                        response = self.phone_agent.process_audio_input(user_input.encode())
                        if response:
                            print(f"Agent: {response}")
                        else:
                            print("Agent: I'm sorry, I didn't understand that. Could you please repeat?")
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.logger.error(f"Test mode error: {e}")
                    print("Agent: I'm experiencing technical difficulties. Please try again.")
        
        print("\nTest call ended.")
    
    def _run_production_mode(self):
        """Run the application in production mode."""
        self.logger.info("Running in PRODUCTION MODE")
        print("AI Phone Agent API is running in production mode.")
        print("Waiting for incoming calls...")
        
        # In a real implementation, this would start a web server
        # and handle HTTP requests for Twilio webhooks
        while self.is_running:
            try:
                # Keep the server running
                import time
                time.sleep(1)
            except KeyboardInterrupt:
                break
    
    def stop_server(self):
        """Stop the API server."""
        try:
            self.is_running = False
            if self.phone_agent:
                self.phone_agent.emergency_stop()
            self.logger.info("Server stopped")
        except Exception as e:
            self.logger.error(f"Error stopping server: {e}")
    
    def handle_webhook(self, webhook_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming webhook requests from Twilio.
        
        Args:
            webhook_data: Webhook data from Twilio
            
        Returns:
            Dict containing response
        """
        try:
            webhook_type = webhook_data.get("type", "unknown")
            
            if webhook_type == "incoming_call":
                return self._handle_incoming_call(webhook_data)
            elif webhook_type == "call_status":
                return self._handle_call_status(webhook_data)
            elif webhook_type == "recording":
                return self._handle_recording(webhook_data)
            else:
                self.logger.warning(f"Unknown webhook type: {webhook_type}")
                return {"status": "error", "message": "Unknown webhook type"}
                
        except Exception as e:
            self.logger.error(f"Webhook handling error: {e}")
            return {"status": "error", "message": str(e)}
    
    def _handle_incoming_call(self, call_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming call webhook."""
        try:
            call_sid = call_data.get("CallSid")
            from_number = call_data.get("From")
            
            # Start call with phone agent
            if self.phone_agent and call_sid and self.phone_agent.start_call(str(call_sid), str(from_number) if from_number else None):
                return {
                    "status": "success",
                    "message": "Call started successfully",
                    "call_sid": call_sid
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to start call"
                }
                
        except Exception as e:
            self.logger.error(f"Incoming call handling error: {e}")
            return {"status": "error", "message": str(e)}
    
    def _handle_call_status(self, status_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle call status update webhook."""
        try:
            call_sid = status_data.get("CallSid")
            call_status = status_data.get("CallStatus")
            
            return {
                "status": "success",
                "message": f"Call status updated: {call_status}",
                "call_sid": call_sid
            }
            
        except Exception as e:
            self.logger.error(f"Call status handling error: {e}")
            return {"status": "error", "message": str(e)}
    
    def _handle_recording(self, recording_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle recording webhook."""
        try:
            recording_url = recording_data.get("RecordingUrl")
            call_sid = recording_data.get("CallSid")
            
            return {
                "status": "success",
                "message": "Recording received",
                "recording_url": recording_url,
                "call_sid": call_sid
            }
            
        except Exception as e:
            self.logger.error(f"Recording handling error: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the API.
        
        Returns:
            Dict containing API status
        """
        return {
            "running": self.is_running,
            "mode": "test" if TESTING_SETTINGS['enabled'] else "production",
            "phone_agent_status": self.phone_agent.get_call_status() if self.phone_agent else None,
            "timestamp": datetime.now().isoformat()
        }


def main():
    """Main entry point for the application."""
    try:
        # Create and start the API
        api = PhoneAgentAPI()
        api.start_server()
        
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 