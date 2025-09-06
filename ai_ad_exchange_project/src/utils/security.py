"""
Security Manager - Handles HMAC encryption and fraud prevention
"""

import logging
import hashlib
import hmac
import base64
import json
from typing import Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SecurityManager:
    """
    Manages security features including HMAC encryption and fraud prevention
    """
    
    def __init__(self, secret_key: str = "your-secret-key-change-this"):
        self.secret_key = secret_key.encode('utf-8')
        logger.info("SecurityManager initialized")
    
    def generate_tracking_url(self, landing_page: str, campaign_id: str, publisher_id: str) -> str:
        """
        Generate a secure tracking URL with HMAC signature
        
        Args:
            landing_page: Original landing page URL
            campaign_id: Campaign ID
            publisher_id: Publisher ID
            
        Returns:
            Secure tracking URL with signature
        """
        try:
            # Create tracking parameters
            tracking_data = {
                'campaign_id': campaign_id,
                'publisher_id': publisher_id,
                'timestamp': datetime.now().isoformat(),
                'impression_id': self._generate_impression_id()
            }
            
            # Generate HMAC signature
            signature = self._generate_signature(tracking_data)
            tracking_data['signature'] = signature
            
            # Encode tracking data
            encoded_data = base64.urlsafe_b64encode(
                json.dumps(tracking_data).encode('utf-8')
            ).decode('utf-8')
            
            # Create tracking URL
            tracking_url = f"{landing_page}?tracking={encoded_data}"
            
            logger.info(f"Generated tracking URL for campaign {campaign_id}")
            return tracking_url
            
        except Exception as e:
            logger.error(f"Error generating tracking URL: {str(e)}")
            return landing_page
    
    def _generate_signature(self, data: Dict) -> str:
        """
        Generate HMAC signature for data
        
        Args:
            data: Data to sign
            
        Returns:
            HMAC signature
        """
        try:
            # Convert data to sorted JSON string for consistent signing
            data_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
            
            # Generate HMAC signature
            signature = hmac.new(
                self.secret_key,
                data_str.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return signature
            
        except Exception as e:
            logger.error(f"Error generating signature: {str(e)}")
            return ""
    
    def _generate_impression_id(self) -> str:
        """Generate unique impression ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"imp_{timestamp}"
    
    def verify_click_signature(self, click_data: Dict) -> bool:
        """
        Verify HMAC signature for click data
        
        Args:
            click_data: Click data with signature
            
        Returns:
            True if signature is valid
        """
        try:
            if 'signature' not in click_data:
                logger.warning("No signature found in click data")
                return False
            
            # Extract signature
            provided_signature = click_data['signature']
            
            # Create data copy without signature for verification
            data_for_verification = {k: v for k, v in click_data.items() if k != 'signature'}
            
            # Generate expected signature
            expected_signature = self._generate_signature(data_for_verification)
            
            # Compare signatures
            is_valid = hmac.compare_digest(provided_signature, expected_signature)
            
            if not is_valid:
                logger.warning("Invalid click signature detected")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Error verifying click signature: {str(e)}")
            return False
    
    def extract_tracking_data(self, tracking_param: str) -> Optional[Dict]:
        """
        Extract and verify tracking data from URL parameter
        
        Args:
            tracking_param: Base64 encoded tracking parameter
            
        Returns:
            Tracking data if valid, None otherwise
        """
        try:
            # Decode tracking parameter
            decoded_data = base64.urlsafe_b64decode(tracking_param).decode('utf-8')
            tracking_data = json.loads(decoded_data)
            
            # Verify signature
            if not self.verify_click_signature(tracking_data):
                return None
            
            # Check timestamp (prevent replay attacks)
            timestamp = datetime.fromisoformat(tracking_data['timestamp'])
            if datetime.now() - timestamp > timedelta(hours=24):
                logger.warning("Tracking data too old")
                return None
            
            return tracking_data
            
        except Exception as e:
            logger.error(f"Error extracting tracking data: {str(e)}")
            return None
    
    def generate_secure_token(self, user_id: str, expires_in: int = 3600) -> str:
        """
        Generate a secure token for user authentication
        
        Args:
            user_id: User ID
            expires_in: Token expiration time in seconds
            
        Returns:
            Secure token
        """
        try:
            token_data = {
                'user_id': user_id,
                'expires_at': (datetime.now() + timedelta(seconds=expires_in)).isoformat(),
                'nonce': self._generate_nonce()
            }
            
            # Generate signature
            signature = self._generate_signature(token_data)
            token_data['signature'] = signature
            
            # Encode token
            token = base64.urlsafe_b64encode(
                json.dumps(token_data).encode('utf-8')
            ).decode('utf-8')
            
            return token
            
        except Exception as e:
            logger.error(f"Error generating secure token: {str(e)}")
            return ""
    
    def verify_secure_token(self, token: str) -> Optional[str]:
        """
        Verify and extract user ID from secure token
        
        Args:
            token: Secure token
            
        Returns:
            User ID if token is valid, None otherwise
        """
        try:
            # Decode token
            decoded_data = base64.urlsafe_b64decode(token).decode('utf-8')
            token_data = json.loads(decoded_data)
            
            # Verify signature
            if not self.verify_click_signature(token_data):
                return None
            
            # Check expiration
            expires_at = datetime.fromisoformat(token_data['expires_at'])
            if datetime.now() > expires_at:
                logger.warning("Token expired")
                return None
            
            return token_data['user_id']
            
        except Exception as e:
            logger.error(f"Error verifying secure token: {str(e)}")
            return None
    
    def _generate_nonce(self) -> str:
        """Generate a random nonce"""
        import secrets
        return secrets.token_hex(16)
    
    def detect_fraudulent_activity(self, activity_data: Dict) -> Dict:
        """
        Detect potentially fraudulent activity
        
        Args:
            activity_data: Activity data to analyze
            
        Returns:
            Fraud detection results
        """
        try:
            fraud_indicators = []
            risk_score = 0.0
            
            # Check for suspicious patterns
            if activity_data.get('clicks_per_minute', 0) > 10:
                fraud_indicators.append("High click rate")
                risk_score += 0.3
            
            if activity_data.get('clicks_without_impressions', 0) > 0:
                fraud_indicators.append("Clicks without impressions")
                risk_score += 0.5
            
            if activity_data.get('suspicious_ip', False):
                fraud_indicators.append("Suspicious IP address")
                risk_score += 0.4
            
            if activity_data.get('bot_signatures', []):
                fraud_indicators.append("Bot signatures detected")
                risk_score += 0.6
            
            return {
                'is_fraudulent': risk_score > 0.7,
                'risk_score': min(risk_score, 1.0),
                'fraud_indicators': fraud_indicators,
                'recommendation': 'block' if risk_score > 0.7 else 'monitor'
            }
            
        except Exception as e:
            logger.error(f"Error detecting fraudulent activity: {str(e)}")
            return {
                'is_fraudulent': False,
                'risk_score': 0.0,
                'fraud_indicators': [],
                'recommendation': 'error'
            } 