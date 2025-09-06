"""
Publisher Manager - Handles publisher operations and data
"""

import logging
import json
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class PublisherManager:
    """
    Manages publisher data and operations
    """
    
    def __init__(self):
        self.publishers_file = Path("config/publishers.json")
        self.publishers = self._load_publishers()
        logger.info("PublisherManager initialized")
    
    def _load_publishers(self) -> Dict:
        """Load publishers from JSON file"""
        try:
            if self.publishers_file.exists():
                with open(self.publishers_file, 'r') as f:
                    return json.load(f)
            else:
                logger.warning("Publishers file not found, using empty data")
                return {"publishers": [], "settings": {}}
        except Exception as e:
            logger.error(f"Error loading publishers: {str(e)}")
            return {"publishers": [], "settings": {}}
    
    def _save_publishers(self):
        """Save publishers to JSON file"""
        try:
            with open(self.publishers_file, 'w') as f:
                json.dump(self.publishers, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving publishers: {str(e)}")
    
    def get_publisher(self, publisher_id: str) -> Optional[Dict]:
        """
        Get publisher by ID
        
        Args:
            publisher_id: Publisher ID
            
        Returns:
            Publisher data or None if not found
        """
        try:
            for publisher in self.publishers.get("publishers", []):
                if publisher.get("id") == publisher_id:
                    return publisher
            return None
        except Exception as e:
            logger.error(f"Error getting publisher {publisher_id}: {str(e)}")
            return None
    
    def get_all_publishers(self) -> List[Dict]:
        """
        Get all publishers
        
        Returns:
            List of all publishers
        """
        return self.publishers.get("publishers", [])
    
    def get_active_publishers(self) -> List[Dict]:
        """
        Get all active publishers
        
        Returns:
            List of active publishers
        """
        return [
            pub for pub in self.publishers.get("publishers", [])
            if pub.get("status") == "active" and pub.get("approved")
        ]
    
    def add_publisher(self, publisher_data: Dict) -> bool:
        """
        Add a new publisher
        
        Args:
            publisher_data: Publisher information
            
        Returns:
            True if added successfully
        """
        try:
            # Generate unique ID
            publisher_data["id"] = f"pub_{len(self.publishers.get('publishers', [])) + 1:03d}"
            
            # Set default values
            publisher_data.setdefault("status", "pending")
            publisher_data.setdefault("approved", False)
            publisher_data.setdefault("performance_metrics", {
                "avg_click_rate": 0.0,
                "avg_engagement_rate": 0.0,
                "total_earnings": 0.0
            })
            
            self.publishers["publishers"].append(publisher_data)
            self._save_publishers()
            
            logger.info(f"Publisher added: {publisher_data['id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding publisher: {str(e)}")
            return False
    
    def update_publisher(self, publisher_id: str, updates: Dict) -> bool:
        """
        Update publisher information
        
        Args:
            publisher_id: Publisher ID
            updates: Fields to update
            
        Returns:
            True if updated successfully
        """
        try:
            for publisher in self.publishers.get("publishers", []):
                if publisher.get("id") == publisher_id:
                    publisher.update(updates)
                    self._save_publishers()
                    logger.info(f"Publisher updated: {publisher_id}")
                    return True
            
            logger.warning(f"Publisher not found: {publisher_id}")
            return False
            
        except Exception as e:
            logger.error(f"Error updating publisher {publisher_id}: {str(e)}")
            return False
    
    def approve_publisher(self, publisher_id: str) -> bool:
        """
        Approve a publisher
        
        Args:
            publisher_id: Publisher ID
            
        Returns:
            True if approved successfully
        """
        return self.update_publisher(publisher_id, {"approved": True, "status": "active"})
    
    def suspend_publisher(self, publisher_id: str) -> bool:
        """
        Suspend a publisher
        
        Args:
            publisher_id: Publisher ID
            
        Returns:
            True if suspended successfully
        """
        return self.update_publisher(publisher_id, {"status": "suspended"})
    
    def add_earnings(self, publisher_id: str, amount: float) -> bool:
        """
        Add earnings to publisher account
        
        Args:
            publisher_id: Publisher ID
            amount: Amount to add
            
        Returns:
            True if added successfully
        """
        try:
            for publisher in self.publishers.get("publishers", []):
                if publisher.get("id") == publisher_id:
                    current_earnings = publisher["performance_metrics"]["total_earnings"]
                    publisher["performance_metrics"]["total_earnings"] = current_earnings + amount
                    self._save_publishers()
                    logger.info(f"Earnings added to {publisher_id}: ${amount:.2f}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error adding earnings to {publisher_id}: {str(e)}")
            return False
    
    def get_publishers_by_keywords(self, keywords: List[str]) -> List[Dict]:
        """
        Get publishers that match given keywords
        
        Args:
            keywords: Keywords to match
            
        Returns:
            List of matching publishers
        """
        try:
            matching_publishers = []
            keywords_lower = [k.lower() for k in keywords]
            
            for publisher in self.get_active_publishers():
                publisher_keywords = [k.lower() for k in publisher.get("keywords", [])]
                
                # Check for keyword overlap
                if any(k in publisher_keywords for k in keywords_lower):
                    matching_publishers.append(publisher)
            
            return matching_publishers
            
        except Exception as e:
            logger.error(f"Error getting publishers by keywords: {str(e)}")
            return []
    
    def get_publisher_stats(self, publisher_id: str) -> Dict:
        """
        Get publisher statistics
        
        Args:
            publisher_id: Publisher ID
            
        Returns:
            Publisher statistics
        """
        try:
            publisher = self.get_publisher(publisher_id)
            if not publisher:
                return {"error": "Publisher not found"}
            
            return {
                "id": publisher_id,
                "name": publisher.get("name"),
                "followers": publisher.get("followers", 0),
                "avg_viewers": publisher.get("avg_viewers", 0),
                "stream_hours_monthly": publisher.get("stream_hours_monthly", 0),
                "total_earnings": publisher.get("performance_metrics", {}).get("total_earnings", 0),
                "avg_click_rate": publisher.get("performance_metrics", {}).get("avg_click_rate", 0),
                "status": publisher.get("status")
            }
            
        except Exception as e:
            logger.error(f"Error getting publisher stats for {publisher_id}: {str(e)}")
            return {"error": str(e)} 