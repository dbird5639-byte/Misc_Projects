"""
Analytics Manager - Handles performance tracking and reporting
"""

import logging
import json
from typing import Dict, List, Optional
from datetime import datetime, date, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

class AnalyticsManager:
    """
    Manages analytics data and performance tracking
    """
    
    def __init__(self):
        self.analytics_dir = Path("data/analytics")
        self.analytics_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for real-time analytics
        self.impressions_cache = {}
        self.clicks_cache = {}
        self.revenue_cache = {}
        
        logger.info("AnalyticsManager initialized")
    
    def record_impression(self, impression_data: Dict) -> bool:
        """
        Record an ad impression
        
        Args:
            impression_data: Impression tracking data
            
        Returns:
            True if recorded successfully
        """
        try:
            # Add timestamp if not present
            if 'timestamp' not in impression_data:
                impression_data['timestamp'] = datetime.now().isoformat()
            
            # Store in cache
            today = date.today().isoformat()
            if today not in self.impressions_cache:
                self.impressions_cache[today] = []
            
            self.impressions_cache[today].append(impression_data)
            
            # Save to file
            self._save_analytics_data('impressions', impression_data)
            
            logger.info(f"Impression recorded for campaign {impression_data.get('campaign_id')}")
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
            # Add timestamp if not present
            if 'timestamp' not in click_data:
                click_data['timestamp'] = datetime.now().isoformat()
            
            # Store in cache
            today = date.today().isoformat()
            if today not in self.clicks_cache:
                self.clicks_cache[today] = []
            
            self.clicks_cache[today].append(click_data)
            
            # Save to file
            self._save_analytics_data('clicks', click_data)
            
            logger.info(f"Click recorded for campaign {click_data.get('campaign_id')}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording click: {str(e)}")
            return False
    
    def record_revenue(self, revenue_data: Dict) -> bool:
        """
        Record revenue transaction
        
        Args:
            revenue_data: Revenue transaction data
            
        Returns:
            True if recorded successfully
        """
        try:
            # Add timestamp if not present
            if 'timestamp' not in revenue_data:
                revenue_data['timestamp'] = datetime.now().isoformat()
            
            # Store in cache
            today = date.today().isoformat()
            if today not in self.revenue_cache:
                self.revenue_cache[today] = []
            
            self.revenue_cache[today].append(revenue_data)
            
            # Save to file
            self._save_analytics_data('revenue', revenue_data)
            
            logger.info(f"Revenue recorded: ${revenue_data.get('amount', 0):.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording revenue: {str(e)}")
            return False
    
    def _save_analytics_data(self, data_type: str, data: Dict):
        """Save analytics data to file"""
        try:
            today = date.today().isoformat()
            file_path = self.analytics_dir / f"{data_type}_{today}.json"
            
            # Load existing data or create new
            if file_path.exists():
                with open(file_path, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []
            
            # Add new data
            existing_data.append(data)
            
            # Save back to file
            with open(file_path, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving analytics data: {str(e)}")
    
    def get_today_impressions(self) -> int:
        """Get total impressions for today"""
        try:
            today = date.today().isoformat()
            return len(self.impressions_cache.get(today, []))
        except Exception as e:
            logger.error(f"Error getting today's impressions: {str(e)}")
            return 0
    
    def get_today_clicks(self) -> int:
        """Get total clicks for today"""
        try:
            today = date.today().isoformat()
            return len(self.clicks_cache.get(today, []))
        except Exception as e:
            logger.error(f"Error getting today's clicks: {str(e)}")
            return 0
    
    def get_today_revenue(self) -> float:
        """Get total revenue for today"""
        try:
            today = date.today().isoformat()
            revenue_data = self.revenue_cache.get(today, [])
            return sum(item.get('amount', 0) for item in revenue_data)
        except Exception as e:
            logger.error(f"Error getting today's revenue: {str(e)}")
            return 0.0
    
    def get_campaign_performance(self, campaign_id: str, days: int = 30) -> Dict:
        """
        Get performance metrics for a campaign
        
        Args:
            campaign_id: Campaign ID
            days: Number of days to analyze
            
        Returns:
            Campaign performance data
        """
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=days)
            
            impressions = 0
            clicks = 0
            revenue = 0.0
            
            # Aggregate data from cache
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.isoformat()
                
                # Count impressions
                daily_impressions = self.impressions_cache.get(date_str, [])
                impressions += len([i for i in daily_impressions if i.get('campaign_id') == campaign_id])
                
                # Count clicks
                daily_clicks = self.clicks_cache.get(date_str, [])
                clicks += len([c for c in daily_clicks if c.get('campaign_id') == campaign_id])
                
                # Sum revenue
                daily_revenue = self.revenue_cache.get(date_str, [])
                revenue += sum(r.get('amount', 0) for r in daily_revenue if r.get('campaign_id') == campaign_id)
                
                current_date += timedelta(days=1)
            
            # Calculate metrics
            ctr = clicks / impressions if impressions > 0 else 0.0
            cpm = (revenue / impressions * 1000) if impressions > 0 else 0.0
            cpc = revenue / clicks if clicks > 0 else 0.0
            
            return {
                'campaign_id': campaign_id,
                'period_days': days,
                'impressions': impressions,
                'clicks': clicks,
                'revenue': revenue,
                'ctr': ctr,
                'cpm': cpm,
                'cpc': cpc
            }
            
        except Exception as e:
            logger.error(f"Error getting campaign performance: {str(e)}")
            return {'error': str(e)}
    
    def get_publisher_performance(self, publisher_id: str, days: int = 30) -> Dict:
        """
        Get performance metrics for a publisher
        
        Args:
            publisher_id: Publisher ID
            days: Number of days to analyze
            
        Returns:
            Publisher performance data
        """
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=days)
            
            impressions = 0
            clicks = 0
            earnings = 0.0
            
            # Aggregate data from cache
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.isoformat()
                
                # Count impressions
                daily_impressions = self.impressions_cache.get(date_str, [])
                impressions += len([i for i in daily_impressions if i.get('publisher_id') == publisher_id])
                
                # Count clicks
                daily_clicks = self.clicks_cache.get(date_str, [])
                clicks += len([c for c in daily_clicks if c.get('publisher_id') == publisher_id])
                
                # Sum earnings
                daily_revenue = self.revenue_cache.get(date_str, [])
                earnings += sum(r.get('publisher_earnings', 0) for r in daily_revenue if r.get('publisher_id') == publisher_id)
                
                current_date += timedelta(days=1)
            
            # Calculate metrics
            ctr = clicks / impressions if impressions > 0 else 0.0
            earnings_per_impression = earnings / impressions if impressions > 0 else 0.0
            
            return {
                'publisher_id': publisher_id,
                'period_days': days,
                'impressions': impressions,
                'clicks': clicks,
                'earnings': earnings,
                'ctr': ctr,
                'earnings_per_impression': earnings_per_impression
            }
            
        except Exception as e:
            logger.error(f"Error getting publisher performance: {str(e)}")
            return {'error': str(e)}
    
    def get_exchange_summary(self, days: int = 30) -> Dict:
        """
        Get exchange-wide summary statistics
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Exchange summary data
        """
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=days)
            
            total_impressions = 0
            total_clicks = 0
            total_revenue = 0.0
            total_commission = 0.0
            
            # Aggregate data from cache
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.isoformat()
                
                total_impressions += len(self.impressions_cache.get(date_str, []))
                total_clicks += len(self.clicks_cache.get(date_str, []))
                
                daily_revenue = self.revenue_cache.get(date_str, [])
                total_revenue += sum(r.get('amount', 0) for r in daily_revenue)
                total_commission += sum(r.get('commission', 0) for r in daily_revenue)
                
                current_date += timedelta(days=1)
            
            # Calculate metrics
            ctr = total_clicks / total_impressions if total_impressions > 0 else 0.0
            cpm = (total_revenue / total_impressions * 1000) if total_impressions > 0 else 0.0
            
            return {
                'period_days': days,
                'total_impressions': total_impressions,
                'total_clicks': total_clicks,
                'total_revenue': total_revenue,
                'total_commission': total_commission,
                'avg_ctr': ctr,
                'avg_cpm': cpm,
                'publisher_payouts': total_revenue - total_commission
            }
            
        except Exception as e:
            logger.error(f"Error getting exchange summary: {str(e)}")
            return {'error': str(e)}
    
    def generate_report(self, report_type: str, **kwargs) -> Dict:
        """
        Generate analytics report
        
        Args:
            report_type: Type of report to generate
            **kwargs: Additional parameters
            
        Returns:
            Generated report data
        """
        try:
            if report_type == 'campaign':
                campaign_id = kwargs.get('campaign_id')
                if not campaign_id:
                    return {'error': 'campaign_id is required for campaign reports'}
                return self.get_campaign_performance(campaign_id, kwargs.get('days', 30))
            elif report_type == 'publisher':
                publisher_id = kwargs.get('publisher_id')
                if not publisher_id:
                    return {'error': 'publisher_id is required for publisher reports'}
                return self.get_publisher_performance(publisher_id, kwargs.get('days', 30))
            elif report_type == 'exchange':
                return self.get_exchange_summary(kwargs.get('days', 30))
            else:
                return {'error': f'Unknown report type: {report_type}'}
                
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return {'error': str(e)} 