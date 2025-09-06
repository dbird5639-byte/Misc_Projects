"""
Advertiser Manager - Handles advertiser operations and campaign management
"""

import logging
import json
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime, date

logger = logging.getLogger(__name__)

class AdvertiserManager:
    """
    Manages advertiser data and campaign operations
    """
    
    def __init__(self):
        self.advertisers_file = Path("config/advertisers.json")
        self.advertisers = self._load_advertisers()
        logger.info("AdvertiserManager initialized")
    
    def _load_advertisers(self) -> Dict:
        """Load advertisers from JSON file"""
        try:
            if self.advertisers_file.exists():
                with open(self.advertisers_file, 'r') as f:
                    return json.load(f)
            else:
                logger.warning("Advertisers file not found, using empty data")
                return {"advertisers": [], "settings": {}}
        except Exception as e:
            logger.error(f"Error loading advertisers: {str(e)}")
            return {"advertisers": [], "settings": {}}
    
    def _save_advertisers(self):
        """Save advertisers to JSON file"""
        try:
            with open(self.advertisers_file, 'w') as f:
                json.dump(self.advertisers, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving advertisers: {str(e)}")
    
    def get_advertiser(self, advertiser_id: str) -> Optional[Dict]:
        """
        Get advertiser by ID
        
        Args:
            advertiser_id: Advertiser ID
            
        Returns:
            Advertiser data or None if not found
        """
        try:
            for advertiser in self.advertisers.get("advertisers", []):
                if advertiser.get("id") == advertiser_id:
                    return advertiser
            return None
        except Exception as e:
            logger.error(f"Error getting advertiser {advertiser_id}: {str(e)}")
            return None
    
    def get_all_advertisers(self) -> List[Dict]:
        """
        Get all advertisers
        
        Returns:
            List of all advertisers
        """
        return self.advertisers.get("advertisers", [])
    
    def get_active_advertisers(self) -> List[Dict]:
        """
        Get all active advertisers
        
        Returns:
            List of active advertisers
        """
        return [
            adv for adv in self.advertisers.get("advertisers", [])
            if adv.get("status") == "active"
        ]
    
    def add_advertiser(self, advertiser_data: Dict) -> bool:
        """
        Add a new advertiser
        
        Args:
            advertiser_data: Advertiser information
            
        Returns:
            True if added successfully
        """
        try:
            # Generate unique ID
            advertiser_data["id"] = f"adv_{len(self.advertisers.get('advertisers', [])) + 1:03d}"
            
            # Set default values
            advertiser_data.setdefault("status", "pending")
            advertiser_data.setdefault("verified", False)
            advertiser_data.setdefault("budget", {
                "total": 0.0,
                "daily_limit": 0.0,
                "spent": 0.0,
                "remaining": 0.0
            })
            advertiser_data.setdefault("campaigns", [])
            
            self.advertisers["advertisers"].append(advertiser_data)
            self._save_advertisers()
            
            logger.info(f"Advertiser added: {advertiser_data['id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding advertiser: {str(e)}")
            return False
    
    def update_advertiser(self, advertiser_id: str, updates: Dict) -> bool:
        """
        Update advertiser information
        
        Args:
            advertiser_id: Advertiser ID
            updates: Fields to update
            
        Returns:
            True if updated successfully
        """
        try:
            for advertiser in self.advertisers.get("advertisers", []):
                if advertiser.get("id") == advertiser_id:
                    advertiser.update(updates)
                    self._save_advertisers()
                    logger.info(f"Advertiser updated: {advertiser_id}")
                    return True
            
            logger.warning(f"Advertiser not found: {advertiser_id}")
            return False
            
        except Exception as e:
            logger.error(f"Error updating advertiser {advertiser_id}: {str(e)}")
            return False
    
    def verify_advertiser(self, advertiser_id: str) -> bool:
        """
        Verify an advertiser
        
        Args:
            advertiser_id: Advertiser ID
            
        Returns:
            True if verified successfully
        """
        return self.update_advertiser(advertiser_id, {"verified": True, "status": "active"})
    
    def get_campaign(self, campaign_id: str) -> Optional[Dict]:
        """
        Get campaign by ID
        
        Args:
            campaign_id: Campaign ID
            
        Returns:
            Campaign data or None if not found
        """
        try:
            for advertiser in self.advertisers.get("advertisers", []):
                for campaign in advertiser.get("campaigns", []):
                    if campaign.get("id") == campaign_id:
                        campaign["advertiser_id"] = advertiser.get("id")
                        return campaign
            return None
        except Exception as e:
            logger.error(f"Error getting campaign {campaign_id}: {str(e)}")
            return None
    
    def get_active_campaigns(self) -> List[Dict]:
        """
        Get all active campaigns
        
        Returns:
            List of active campaigns
        """
        try:
            active_campaigns = []
            for advertiser in self.advertisers.get("advertisers", []):
                for campaign in advertiser.get("campaigns", []):
                    if campaign.get("status") == "active":
                        campaign["advertiser_id"] = advertiser.get("id")
                        active_campaigns.append(campaign)
            return active_campaigns
        except Exception as e:
            logger.error(f"Error getting active campaigns: {str(e)}")
            return []
    
    def get_available_ads(self, keywords: List[str]) -> List[Dict]:
        """
        Get available ads that match keywords
        
        Args:
            keywords: Keywords to match
            
        Returns:
            List of matching ads
        """
        try:
            available_ads = []
            keywords_lower = [k.lower() for k in keywords]
            
            for advertiser in self.get_active_advertisers():
                for campaign in advertiser.get("campaigns", []):
                    if campaign.get("status") == "active":
                        campaign_keywords = [k.lower() for k in campaign.get("keywords", [])]
                        
                        # Check for keyword overlap
                        if any(k in campaign_keywords for k in keywords_lower):
                            campaign["advertiser_id"] = advertiser.get("id")
                            available_ads.append(campaign)
            
            return available_ads
            
        except Exception as e:
            logger.error(f"Error getting available ads: {str(e)}")
            return []
    
    def add_campaign(self, advertiser_id: str, campaign_data: Dict) -> bool:
        """
        Add a new campaign to an advertiser
        
        Args:
            advertiser_id: Advertiser ID
            campaign_data: Campaign information
            
        Returns:
            True if added successfully
        """
        try:
            for advertiser in self.advertisers.get("advertisers", []):
                if advertiser.get("id") == advertiser_id:
                    # Generate unique campaign ID
                    campaign_data["id"] = f"camp_{len(advertiser.get('campaigns', [])) + 1:03d}"
                    
                    # Set default values
                    campaign_data.setdefault("status", "pending")
                    campaign_data.setdefault("performance", {
                        "impressions": 0,
                        "clicks": 0,
                        "conversions": 0,
                        "ctr": 0.0,
                        "cvr": 0.0
                    })
                    
                    advertiser["campaigns"].append(campaign_data)
                    self._save_advertisers()
                    
                    logger.info(f"Campaign added: {campaign_data['id']} to {advertiser_id}")
                    return True
            
            logger.warning(f"Advertiser not found: {advertiser_id}")
            return False
            
        except Exception as e:
            logger.error(f"Error adding campaign: {str(e)}")
            return False
    
    def update_campaign(self, campaign_id: str, updates: Dict) -> bool:
        """
        Update campaign information
        
        Args:
            campaign_id: Campaign ID
            updates: Fields to update
            
        Returns:
            True if updated successfully
        """
        try:
            for advertiser in self.advertisers.get("advertisers", []):
                for campaign in advertiser.get("campaigns", []):
                    if campaign.get("id") == campaign_id:
                        campaign.update(updates)
                        self._save_advertisers()
                        logger.info(f"Campaign updated: {campaign_id}")
                        return True
            
            logger.warning(f"Campaign not found: {campaign_id}")
            return False
            
        except Exception as e:
            logger.error(f"Error updating campaign {campaign_id}: {str(e)}")
            return False
    
    def update_ad_performance(self, campaign_id: str, metric: str, value: int) -> bool:
        """
        Update ad performance metrics
        
        Args:
            campaign_id: Campaign ID
            metric: Metric to update (impressions, clicks, conversions)
            value: Value to add
            
        Returns:
            True if updated successfully
        """
        try:
            for advertiser in self.advertisers.get("advertisers", []):
                for campaign in advertiser.get("campaigns", []):
                    if campaign.get("id") == campaign_id:
                        current_value = campaign["performance"].get(metric, 0)
                        campaign["performance"][metric] = current_value + value
                        
                        # Update derived metrics
                        if metric == "impressions" and campaign["performance"]["impressions"] > 0:
                            campaign["performance"]["ctr"] = (
                                campaign["performance"]["clicks"] / 
                                campaign["performance"]["impressions"]
                            )
                        
                        if metric == "clicks" and campaign["performance"]["clicks"] > 0:
                            campaign["performance"]["cvr"] = (
                                campaign["performance"]["conversions"] / 
                                campaign["performance"]["clicks"]
                            )
                        
                        self._save_advertisers()
                        logger.info(f"Performance updated for {campaign_id}: {metric} += {value}")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating ad performance: {str(e)}")
            return False
    
    def deduct_budget(self, advertiser_id: str, amount: float) -> bool:
        """
        Deduct amount from advertiser budget
        
        Args:
            advertiser_id: Advertiser ID
            amount: Amount to deduct
            
        Returns:
            True if deducted successfully
        """
        try:
            for advertiser in self.advertisers.get("advertisers", []):
                if advertiser.get("id") == advertiser_id:
                    current_spent = advertiser["budget"]["spent"]
                    current_remaining = advertiser["budget"]["remaining"]
                    
                    if current_remaining >= amount:
                        advertiser["budget"]["spent"] = current_spent + amount
                        advertiser["budget"]["remaining"] = current_remaining - amount
                        self._save_advertisers()
                        logger.info(f"Budget deducted from {advertiser_id}: ${amount:.2f}")
                        return True
                    else:
                        logger.warning(f"Insufficient budget for {advertiser_id}")
                        return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error deducting budget from {advertiser_id}: {str(e)}")
            return False
    
    def get_advertiser_stats(self, advertiser_id: str) -> Dict:
        """
        Get advertiser statistics
        
        Args:
            advertiser_id: Advertiser ID
            
        Returns:
            Advertiser statistics
        """
        try:
            advertiser = self.get_advertiser(advertiser_id)
            if not advertiser:
                return {"error": "Advertiser not found"}
            
            total_campaigns = len(advertiser.get("campaigns", []))
            active_campaigns = len([c for c in advertiser.get("campaigns", []) if c.get("status") == "active"])
            
            return {
                "id": advertiser_id,
                "name": advertiser.get("name"),
                "status": advertiser.get("status"),
                "verified": advertiser.get("verified"),
                "total_budget": advertiser.get("budget", {}).get("total", 0),
                "spent_budget": advertiser.get("budget", {}).get("spent", 0),
                "remaining_budget": advertiser.get("budget", {}).get("remaining", 0),
                "total_campaigns": total_campaigns,
                "active_campaigns": active_campaigns
            }
            
        except Exception as e:
            logger.error(f"Error getting advertiser stats for {advertiser_id}: {str(e)}")
            return {"error": str(e)} 