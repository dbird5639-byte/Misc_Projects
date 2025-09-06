"""
Twilio utilities for phone system integration
"""

import logging
from typing import Dict, List, Optional, Any
import json


class TwilioHandler:
    """
    Handles Twilio phone system integration for call management.
    """
    
    def __init__(self):
        """Initialize the Twilio handler."""
        self.logger = logging.getLogger(__name__)
        self.is_initialized = False
        self.account_sid = None
        self.auth_token = None
        self.phone_number = None
        self.webhook_url = None
        self.active_calls = {}
        
    def initialize(self, account_sid: str, auth_token: str, phone_number: str) -> bool:
        """
        Initialize Twilio connection.
        
        Args:
            account_sid: Twilio account SID
            auth_token: Twilio auth token
            phone_number: Twilio phone number
            
        Returns:
            bool: True if initialization successful
        """
        try:
            self.account_sid = account_sid
            self.auth_token = auth_token
            self.phone_number = phone_number
            
            # In a real implementation, this would validate credentials
            # and establish connection with Twilio
            self.is_initialized = True
            self.logger.info("Twilio handler initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Twilio handler: {e}")
            return False
    
    def handle_incoming_call(self, call_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an incoming call from Twilio.
        
        Args:
            call_data: Call data from Twilio webhook
            
        Returns:
            Dict containing response for Twilio
        """
        try:
            call_sid = call_data.get("CallSid")
            from_number = call_data.get("From")
            to_number = call_data.get("To")
            
            if not call_sid:
                self.logger.error("No CallSid provided")
                return self._generate_error_response()
            
            # Store call information
            self.active_calls[call_sid] = {
                "call_sid": call_sid,
                "from_number": from_number,
                "to_number": to_number,
                "status": "ringing",
                "start_time": self._get_timestamp()
            }
            
            # Generate TwiML response
            response = self._generate_twiml_response(str(call_sid))
            
            self.logger.info(f"Incoming call handled: {call_sid}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error handling incoming call: {e}")
            return self._generate_error_response()
    
    def _generate_twiml_response(self, call_sid: str) -> Dict[str, Any]:
        """
        Generate TwiML response for call handling.
        
        Args:
            call_sid: Call SID
            
        Returns:
            Dict containing TwiML response
        """
        try:
            # In a real implementation, this would generate proper TwiML
            # For now, return a simple response structure
            twiml = f"""
            <Response>
                <Say>Hello! Thank you for calling. How can I help you today?</Say>
                <Record action="/webhook/recording" maxLength="30" />
            </Response>
            """
            
            return {
                "status": "success",
                "twiml": twiml,
                "call_sid": call_sid
            }
            
        except Exception as e:
            self.logger.error(f"Error generating TwiML: {e}")
            return self._generate_error_response()
    
    def _generate_error_response(self) -> Dict[str, Any]:
        """Generate error response for failed operations."""
        return {
            "status": "error",
            "twiml": "<Response><Say>I'm sorry, but I'm having technical difficulties. Please try again later.</Say></Response>"
        }
    
    def handle_call_status_update(self, status_data: Dict[str, Any]) -> bool:
        """
        Handle call status updates from Twilio.
        
        Args:
            status_data: Status update data from Twilio
            
        Returns:
            bool: True if status updated successfully
        """
        try:
            call_sid = status_data.get("CallSid")
            call_status = status_data.get("CallStatus")
            
            if call_sid and call_sid in self.active_calls:
                self.active_calls[call_sid]["status"] = call_status
                self.active_calls[call_sid]["last_update"] = self._get_timestamp()
                
                # Remove completed calls
                if call_status in ["completed", "failed", "busy", "no-answer"]:
                    self._cleanup_call(str(call_sid))
                
                self.logger.info(f"Call status updated: {call_sid} -> {call_status}")
                return True
            else:
                self.logger.warning(f"Unknown call SID: {call_sid}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error handling status update: {e}")
            return False
    
    def _cleanup_call(self, call_sid: str):
        """Clean up call data when call ends."""
        try:
            if call_sid in self.active_calls:
                call_data = self.active_calls.pop(call_sid)
                self.logger.info(f"Call cleaned up: {call_sid}")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up call: {e}")
    
    def make_outbound_call(self, to_number: str, message: Optional[str] = None) -> Optional[str]:
        """
        Make an outbound call using Twilio.
        
        Args:
            to_number: Number to call
            message: Message to play (optional)
            
        Returns:
            str: Call SID if successful, None otherwise
        """
        try:
            if not self.is_initialized:
                self.logger.warning("Twilio handler not initialized")
                return None
            
            # In a real implementation, this would use Twilio's API
            # to make an outbound call
            call_sid = f"outbound_{self._generate_call_id()}"
            
            # Store call information
            self.active_calls[call_sid] = {
                "call_sid": call_sid,
                "from_number": self.phone_number,
                "to_number": to_number,
                "status": "initiated",
                "start_time": self._get_timestamp(),
                "direction": "outbound"
            }
            
            self.logger.info(f"Outbound call initiated: {call_sid} to {to_number}")
            return call_sid
            
        except Exception as e:
            self.logger.error(f"Failed to make outbound call: {e}")
            return None
    
    def send_sms(self, to_number: str, message: str) -> bool:
        """
        Send SMS message using Twilio.
        
        Args:
            to_number: Number to send SMS to
            message: Message content
            
        Returns:
            bool: True if SMS sent successfully
        """
        try:
            if not self.is_initialized:
                self.logger.warning("Twilio handler not initialized")
                return False
            
            # In a real implementation, this would use Twilio's SMS API
            self.logger.info(f"SMS sent to {to_number}: {message[:50]}...")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send SMS: {e}")
            return False
    
    def get_active_calls(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about active calls.
        
        Returns:
            Dict containing active call information
        """
        return self.active_calls.copy()
    
    def get_call_info(self, call_sid: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific call.
        
        Args:
            call_sid: Call SID
            
        Returns:
            Dict containing call information, or None if not found
        """
        return self.active_calls.get(call_sid)
    
    def hangup_call(self, call_sid: str) -> bool:
        """
        Hang up a specific call.
        
        Args:
            call_sid: Call SID to hang up
            
        Returns:
            bool: True if call hung up successfully
        """
        try:
            if call_sid in self.active_calls:
                # In a real implementation, this would use Twilio's API
                # to hang up the call
                self._cleanup_call(call_sid)
                self.logger.info(f"Call hung up: {call_sid}")
                return True
            else:
                self.logger.warning(f"Call not found: {call_sid}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to hang up call: {e}")
            return False
    
    def set_webhook_url(self, webhook_url: str):
        """
        Set the webhook URL for Twilio callbacks.
        
        Args:
            webhook_url: Webhook URL
        """
        self.webhook_url = webhook_url
        self.logger.info(f"Webhook URL set to: {webhook_url}")
    
    def get_phone_number_info(self) -> Dict[str, Any]:
        """
        Get information about the Twilio phone number.
        
        Returns:
            Dict containing phone number information
        """
        return {
            "phone_number": self.phone_number,
            "account_sid": self.account_sid,
            "webhook_url": self.webhook_url,
            "initialized": self.is_initialized
        }
    
    def _generate_call_id(self) -> str:
        """Generate a unique call ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get Twilio usage statistics.
        
        Returns:
            Dict containing statistics
        """
        try:
            active_call_count = len(self.active_calls)
            call_statuses = {}
            
            for call_data in self.active_calls.values():
                status = call_data.get("status", "unknown")
                call_statuses[status] = call_statuses.get(status, 0) + 1
            
            return {
                "active_calls": active_call_count,
                "call_statuses": call_statuses,
                "initialized": self.is_initialized
            }
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}
    
    def cleanup(self):
        """Clean up Twilio resources."""
        try:
            # Hang up all active calls
            for call_sid in list(self.active_calls.keys()):
                self.hangup_call(call_sid)
            
            self.is_initialized = False
            self.logger.info("Twilio handler cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the Twilio handler.
        
        Returns:
            Dict containing status information
        """
        return {
            "initialized": self.is_initialized,
            "phone_number": self.phone_number,
            "webhook_url": self.webhook_url,
            "active_calls": len(self.active_calls),
            "statistics": self.get_statistics()
        } 