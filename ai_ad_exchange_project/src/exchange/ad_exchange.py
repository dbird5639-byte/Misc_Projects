"""
Main Ad Exchange class - Core marketplace functionality
"""

import logging
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import random

from .publisher_manager import PublisherManager
from .advertiser_manager import AdvertiserManager
from ..utils.security import SecurityManager
from ..utils.analytics import AnalyticsManager

logger = logging.getLogger(__name__)

class AdExchange:
    """
    Main ad exchange that manages the marketplace between advertisers and publishers
    """
    
    def __init__(self):
        self.publisher_manager = PublisherManager()
        self.advertiser_manager = AdvertiserManager()
        self.security_manager = SecurityManager()
        self.analytics_manager = AnalyticsManager()
        
        # Exchange state
        self.active_auctions = {}
        self.ad_rotation_queue = {}
        self.performance_cache = {}
        
        logger.info("AdExchange initialized")
    
    def match_ad_to_publisher(self, publisher_id: str, keywords: List[str]) -> Optional[Dict]:
        """
        Match the best ad to a publisher based on keywords and bidding
        
        Args:
            publisher_id: ID of the publisher requesting an ad
            keywords: Keywords from the publisher's content
            
        Returns:
            Best matching ad or None if no match found
        """
        try:
            # Get publisher info
            publisher = self.publisher_manager.get_publisher(publisher_id)
            if not publisher or not publisher.get('approved'):
                return None
            
            # Get available ads from advertisers
            available_ads = self.advertiser_manager.get_available_ads(keywords)
            
            if not available_ads:
                return None
            
            # Score and rank ads based on multiple factors
            scored_ads = []
            for ad in available_ads:
                score = self._calculate_ad_score(ad, publisher, keywords)
                scored_ads.append((score, ad))
            
            # Sort by score (highest first) and select winner
            scored_ads.sort(key=lambda x: x[0], reverse=True)
            
            if scored_ads:
                winning_score, winning_ad = scored_ads[0]
                
                # Check if ad meets minimum requirements
                if winning_score >= 0.5:  # Minimum score threshold
                    return self._prepare_ad_for_delivery(winning_ad, publisher_id)
            
            return None
            
        except Exception as e:
            logger.error(f"Error matching ad to publisher {publisher_id}: {str(e)}")
            return None
    
    def _calculate_ad_score(self, ad: Dict, publisher: Dict, keywords: List[str]) -> float:
        """
        Calculate a score for an ad based on multiple factors
        
        Args:
            ad: Ad campaign information
            publisher: Publisher information
            keywords: Content keywords
            
        Returns:
            Score between 0 and 1
        """
        try:
            score = 0.0
            
            # Keyword matching (40% weight)
            keyword_match = self._calculate_keyword_match(ad['keywords'], keywords)
            score += keyword_match * 0.4
            
            # Bid amount (30% weight)
            bid_score = min(ad['bidding']['cpc'] / 2.0, 1.0)  # Normalize to 0-1
            score += bid_score * 0.3
            
            # Publisher performance (20% weight)
            perf_score = publisher['performance_metrics']['avg_click_rate']
            score += perf_score * 0.2
            
            # Ad performance (10% weight)
            ad_perf = ad['performance']['ctr']
            score += ad_perf * 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating ad score: {str(e)}")
            return 0.0
    
    def _calculate_keyword_match(self, ad_keywords: List[str], content_keywords: List[str]) -> float:
        """
        Calculate keyword matching score
        
        Args:
            ad_keywords: Keywords from the ad
            content_keywords: Keywords from the content
            
        Returns:
            Match score between 0 and 1
        """
        try:
            if not ad_keywords or not content_keywords:
                return 0.0
            
            # Convert to lowercase for comparison
            ad_keywords_lower = [k.lower() for k in ad_keywords]
            content_keywords_lower = [k.lower() for k in content_keywords]
            
            # Count matches
            matches = sum(1 for k in ad_keywords_lower if k in content_keywords_lower)
            
            # Calculate score based on match ratio
            match_ratio = matches / len(ad_keywords)
            
            return min(match_ratio, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating keyword match: {str(e)}")
            return 0.0
    
    def _prepare_ad_for_delivery(self, ad: Dict, publisher_id: str) -> Optional[Dict]:
        """
        Prepare ad for delivery with tracking and security
        
        Args:
            ad: Ad campaign information
            publisher_id: ID of the publisher
            
        Returns:
            Prepared ad for delivery or None if error
        """
        try:
            # Generate secure tracking URL
            tracking_url = self.security_manager.generate_tracking_url(
                ad['ad_creative']['landing_page'],
                ad['id'],
                publisher_id
            )
            
            # Create delivery payload
            delivery_ad = {
                'id': ad['id'],
                'title': ad['ad_creative']['title'],
                'description': ad['ad_creative']['description'],
                'image_url': ad['ad_creative']['image_url'],
                'landing_page': tracking_url,
                'campaign_id': ad['id'],
                'advertiser_id': ad.get('advertiser_id'),
                'bidding': ad['bidding'],
                'tracking_data': {
                    'publisher_id': publisher_id,
                    'timestamp': datetime.now().isoformat(),
                    'impression_id': self._generate_impression_id()
                }
            }
            
            return delivery_ad
            
        except Exception as e:
            logger.error(f"Error preparing ad for delivery: {str(e)}")
            return None
    
    def _generate_impression_id(self) -> str:
        """Generate unique impression ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = ''.join(random.choices('0123456789abcdef', k=8))
        return f"imp_{timestamp}_{random_suffix}"
    
    def record_impression(self, impression_data: Dict) -> bool:
        """
        Record an ad impression
        
        Args:
            impression_data: Impression tracking data
            
        Returns:
            True if recorded successfully
        """
        try:
            # Validate impression data
            if not self._validate_impression_data(impression_data):
                return False
            
            # Record in analytics
            self.analytics_manager.record_impression(impression_data)
            
            # Update ad performance
            self.advertiser_manager.update_ad_performance(
                impression_data['campaign_id'],
                'impressions',
                1
            )
            
            logger.info(f"Impression recorded for campaign {impression_data['campaign_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording impression: {str(e)}")
            return False
    
    def record_click(self, click_data: Dict) -> bool:
        """
        Record an ad click
        
        Args:
            click_data: Click tracking data
            
        Returns:
            True if recorded successfully
        """
        try:
            # Validate click data
            if not self._validate_click_data(click_data):
                return False
            
            # Verify click authenticity
            if not self.security_manager.verify_click_signature(click_data):
                logger.warning(f"Invalid click signature detected: {click_data}")
                return False
            
            # Record in analytics
            self.analytics_manager.record_click(click_data)
            
            # Update ad performance
            self.advertiser_manager.update_ad_performance(
                click_data['campaign_id'],
                'clicks',
                1
            )
            
            # Calculate and charge for click
            self._process_click_charge(click_data)
            
            logger.info(f"Click recorded for campaign {click_data['campaign_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording click: {str(e)}")
            return False
    
    def _validate_impression_data(self, data: Dict) -> bool:
        """Validate impression data"""
        required_fields = ['campaign_id', 'publisher_id', 'impression_id']
        return all(field in data for field in required_fields)
    
    def _validate_click_data(self, data: Dict) -> bool:
        """Validate click data"""
        required_fields = ['campaign_id', 'publisher_id', 'impression_id', 'signature']
        return all(field in data for field in required_fields)
    
    def _process_click_charge(self, click_data: Dict):
        """Process click charge to advertiser"""
        try:
            # Get campaign info
            campaign = self.advertiser_manager.get_campaign(click_data['campaign_id'])
            if not campaign:
                return
            
            # Calculate charge amount
            charge_amount = campaign['bidding']['cpc']
            
            # Deduct from advertiser budget
            self.advertiser_manager.deduct_budget(
                campaign['advertiser_id'],
                charge_amount
            )
            
            # Add to publisher earnings
            publisher_earnings = charge_amount * (1 - 0.15)  # 15% commission
            self.publisher_manager.add_earnings(
                click_data['publisher_id'],
                publisher_earnings
            )
            
            logger.info(f"Click charge processed: ${charge_amount:.2f}")
            
        except Exception as e:
            logger.error(f"Error processing click charge: {str(e)}")
    
    def get_exchange_stats(self) -> Dict:
        """
        Get exchange statistics
        
        Returns:
            Dictionary with exchange statistics
        """
        try:
            stats = {
                'total_publishers': len(self.publisher_manager.get_all_publishers()),
                'total_advertisers': len(self.advertiser_manager.get_all_advertisers()),
                'active_campaigns': len(self.advertiser_manager.get_active_campaigns()),
                'total_impressions_today': self.analytics_manager.get_today_impressions(),
                'total_clicks_today': self.analytics_manager.get_today_clicks(),
                'total_revenue_today': self.analytics_manager.get_today_revenue(),
                'exchange_health': 'healthy'
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting exchange stats: {str(e)}")
            return {'error': str(e)} 