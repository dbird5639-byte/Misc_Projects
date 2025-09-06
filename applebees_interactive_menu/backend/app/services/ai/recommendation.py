"""
AI Recommendation Service for Applebee's Interactive Menu
Provides personalized menu recommendations based on customer preferences, dietary restrictions, and order patterns.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from transformers import pipeline

from app.core.database import get_db
from app.models.menu import MenuItem, Category
from app.models.user import User, UserPreferences
from app.models.order import Order, OrderItem
from app.utils.logger import setup_logger


@dataclass
class RecommendationContext:
    """Context for generating recommendations."""
    user_id: Optional[str] = None
    current_time: datetime = None
    location: str = ""
    party_size: int = 1
    occasion: str = ""
    mood: str = ""
    dietary_restrictions: List[str] = None
    allergies: List[str] = None
    budget_range: Tuple[float, float] = None
    previous_orders: List[str] = None
    current_cart: List[str] = None


@dataclass
class Recommendation:
    """Represents a menu recommendation."""
    item_id: str
    item_name: str
    category: str
    confidence_score: float
    reasoning: str
    price: float
    image_url: str
    dietary_tags: List[str]
    nutritional_info: Dict[str, Any]
    customization_options: List[str]
    pairing_suggestions: List[str]
    popularity_score: float
    seasonal_relevance: float


class RecommendationService:
    """
    Advanced AI recommendation service using multiple algorithms and real-time data.
    """
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        
        # Initialize ML models
        self.collaborative_filtering_model = None
        self.content_based_model = None
        self.contextual_model = None
        self.sentiment_analyzer = None
        
        # Initialize vectorizers
        self.item_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.user_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        
        # Load pre-trained models
        self._load_models()
        
        # Cache for recommendations
        self.recommendation_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        self.logger.info("AI Recommendation Service initialized")
    
    def _load_models(self):
        """Load pre-trained ML models."""
        try:
            # Load collaborative filtering model
            self.collaborative_filtering_model = RandomForestClassifier(n_estimators=100)
            
            # Load sentiment analyzer
            self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased")
            
            # Load contextual model (simplified for demo)
            self.contextual_model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            self.logger.info("ML models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
    
    async def generate_recommendations(
        self, 
        context: RecommendationContext,
        limit: int = 10
    ) -> List[Recommendation]:
        """
        Generate personalized menu recommendations.
        
        Args:
            context: Recommendation context with user preferences and current situation
            limit: Maximum number of recommendations to return
            
        Returns:
            List of personalized recommendations
        """
        try:
            # Check cache first
            cache_key = self._generate_cache_key(context)
            if cache_key in self.recommendation_cache:
                cached_time, cached_recommendations = self.recommendation_cache[cache_key]
                if datetime.now() - cached_time < timedelta(seconds=self.cache_ttl):
                    return cached_recommendations[:limit]
            
            # Generate fresh recommendations
            recommendations = await self._generate_fresh_recommendations(context, limit)
            
            # Cache results
            self.recommendation_cache[cache_key] = (datetime.now(), recommendations)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return await self._get_fallback_recommendations(limit)
    
    async def _generate_fresh_recommendations(
        self, 
        context: RecommendationContext, 
        limit: int
    ) -> List[Recommendation]:
        """Generate fresh recommendations using multiple algorithms."""
        
        # Get menu items
        menu_items = await self._get_menu_items()
        
        # Apply different recommendation strategies
        collaborative_scores = await self._collaborative_filtering(context, menu_items)
        content_scores = await self._content_based_filtering(context, menu_items)
        contextual_scores = await self._contextual_filtering(context, menu_items)
        popularity_scores = await self._popularity_based_filtering(menu_items)
        seasonal_scores = await self._seasonal_filtering(menu_items)
        
        # Combine scores with weights
        combined_scores = {}
        for item_id in menu_items.keys():
            combined_scores[item_id] = (
                collaborative_scores.get(item_id, 0) * 0.3 +
                content_scores.get(item_id, 0) * 0.25 +
                contextual_scores.get(item_id, 0) * 0.2 +
                popularity_scores.get(item_id, 0) * 0.15 +
                seasonal_scores.get(item_id, 0) * 0.1
            )
        
        # Apply filters
        filtered_items = await self._apply_filters(menu_items, context)
        
        # Sort by combined score
        sorted_items = sorted(
            filtered_items.items(),
            key=lambda x: combined_scores.get(x[0], 0),
            reverse=True
        )
        
        # Convert to recommendations
        recommendations = []
        for item_id, item in sorted_items[:limit]:
            recommendation = await self._create_recommendation(
                item, combined_scores.get(item_id, 0), context
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    async def _collaborative_filtering(
        self, 
        context: RecommendationContext, 
        menu_items: Dict[str, Any]
    ) -> Dict[str, float]:
        """Collaborative filtering based on similar users."""
        if not context.user_id:
            return {item_id: 0.5 for item_id in menu_items.keys()}
        
        try:
            # Get user's order history
            user_orders = await self._get_user_orders(context.user_id)
            
            # Find similar users
            similar_users = await self._find_similar_users(context.user_id, user_orders)
            
            # Calculate scores based on similar users' preferences
            scores = {}
            for item_id in menu_items.keys():
                score = 0
                total_weight = 0
                
                for user_id, similarity in similar_users:
                    user_preference = await self._get_user_item_preference(user_id, item_id)
                    score += user_preference * similarity
                    total_weight += similarity
                
                scores[item_id] = score / total_weight if total_weight > 0 else 0.5
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Collaborative filtering error: {e}")
            return {item_id: 0.5 for item_id in menu_items.keys()}
    
    async def _content_based_filtering(
        self, 
        context: RecommendationContext, 
        menu_items: Dict[str, Any]
    ) -> Dict[str, float]:
        """Content-based filtering based on item features."""
        try:
            # Create item features
            item_features = []
            item_ids = []
            
            for item_id, item in menu_items.items():
                features = f"{item['name']} {item['description']} {' '.join(item.get('tags', []))}"
                item_features.append(features)
                item_ids.append(item_id)
            
            # Vectorize features
            if item_features:
                feature_vectors = self.item_vectorizer.fit_transform(item_features)
                
                # Calculate similarity matrix
                similarity_matrix = cosine_similarity(feature_vectors)
                
                # Get user preferences
                user_preferences = await self._get_user_preferences(context.user_id)
                
                # Calculate scores
                scores = {}
                for i, item_id in enumerate(item_ids):
                    if user_preferences:
                        # Calculate similarity to user preferences
                        user_vector = self.user_vectorizer.transform([user_preferences])
                        item_vector = feature_vectors[i]
                        similarity = cosine_similarity(user_vector, item_vector)[0][0]
                        scores[item_id] = similarity
                    else:
                        scores[item_id] = 0.5
                
                return scores
            
            return {item_id: 0.5 for item_id in menu_items.keys()}
            
        except Exception as e:
            self.logger.error(f"Content-based filtering error: {e}")
            return {item_id: 0.5 for item_id in menu_items.keys()}
    
    async def _contextual_filtering(
        self, 
        context: RecommendationContext, 
        menu_items: Dict[str, Any]
    ) -> Dict[str, float]:
        """Contextual filtering based on current situation."""
        try:
            scores = {}
            
            for item_id, item in menu_items.items():
                score = 0.5  # Base score
                
                # Time-based scoring
                current_hour = context.current_time.hour if context.current_time else 12
                if 6 <= current_hour <= 11:  # Breakfast time
                    if 'breakfast' in item.get('tags', []):
                        score += 0.3
                elif 11 <= current_hour <= 16:  # Lunch time
                    if 'lunch' in item.get('tags', []):
                        score += 0.3
                elif 16 <= current_hour <= 22:  # Dinner time
                    if 'dinner' in item.get('tags', []):
                        score += 0.3
                
                # Party size scoring
                if context.party_size > 4:
                    if 'sharing' in item.get('tags', []) or 'appetizer' in item.get('tags', []):
                        score += 0.2
                
                # Occasion scoring
                if context.occasion == 'date':
                    if 'romantic' in item.get('tags', []) or 'premium' in item.get('tags', []):
                        score += 0.2
                elif context.occasion == 'business':
                    if 'professional' in item.get('tags', []) or 'quick' in item.get('tags', []):
                        score += 0.2
                
                # Mood scoring
                if context.mood == 'comfort':
                    if 'comfort' in item.get('tags', []) or 'classic' in item.get('tags', []):
                        score += 0.2
                elif context.mood == 'adventurous':
                    if 'spicy' in item.get('tags', []) or 'unique' in item.get('tags', []):
                        score += 0.2
                
                scores[item_id] = min(score, 1.0)
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Contextual filtering error: {e}")
            return {item_id: 0.5 for item_id in menu_items.keys()}
    
    async def _popularity_based_filtering(self, menu_items: Dict[str, Any]) -> Dict[str, float]:
        """Popularity-based filtering."""
        try:
            # Get popularity data (simplified for demo)
            popularity_data = await self._get_popularity_data()
            
            scores = {}
            max_popularity = max(popularity_data.values()) if popularity_data else 1
            
            for item_id in menu_items.keys():
                popularity = popularity_data.get(item_id, 0)
                scores[item_id] = popularity / max_popularity if max_popularity > 0 else 0.5
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Popularity filtering error: {e}")
            return {item_id: 0.5 for item_id in menu_items.keys()}
    
    async def _seasonal_filtering(self, menu_items: Dict[str, Any]) -> Dict[str, float]:
        """Seasonal filtering based on current season and holidays."""
        try:
            current_month = datetime.now().month
            scores = {}
            
            for item_id, item in menu_items.items():
                score = 0.5  # Base score
                
                # Seasonal scoring
                if current_month in [12, 1, 2]:  # Winter
                    if 'warm' in item.get('tags', []) or 'comfort' in item.get('tags', []):
                        score += 0.3
                elif current_month in [3, 4, 5]:  # Spring
                    if 'fresh' in item.get('tags', []) or 'light' in item.get('tags', []):
                        score += 0.3
                elif current_month in [6, 7, 8]:  # Summer
                    if 'refreshing' in item.get('tags', []) or 'cold' in item.get('tags', []):
                        score += 0.3
                elif current_month in [9, 10, 11]:  # Fall
                    if 'seasonal' in item.get('tags', []) or 'harvest' in item.get('tags', []):
                        score += 0.3
                
                scores[item_id] = min(score, 1.0)
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Seasonal filtering error: {e}")
            return {item_id: 0.5 for item_id in menu_items.keys()}
    
    async def _apply_filters(
        self, 
        menu_items: Dict[str, Any], 
        context: RecommendationContext
    ) -> Dict[str, Any]:
        """Apply dietary and budget filters."""
        filtered_items = {}
        
        for item_id, item in menu_items.items():
            # Dietary restrictions filter
            if context.dietary_restrictions:
                item_tags = item.get('dietary_tags', [])
                if not any(restriction.lower() in [tag.lower() for tag in item_tags] 
                          for restriction in context.dietary_restrictions):
                    continue
            
            # Allergy filter
            if context.allergies:
                item_allergens = item.get('allergens', [])
                if any(allergy.lower() in [allergen.lower() for allergen in item_allergens] 
                      for allergy in context.allergies):
                    continue
            
            # Budget filter
            if context.budget_range:
                min_budget, max_budget = context.budget_range
                item_price = item.get('price', 0)
                if not (min_budget <= item_price <= max_budget):
                    continue
            
            filtered_items[item_id] = item
        
        return filtered_items
    
    async def _create_recommendation(
        self, 
        item: Dict[str, Any], 
        confidence_score: float, 
        context: RecommendationContext
    ) -> Recommendation:
        """Create a recommendation object."""
        
        # Generate reasoning
        reasoning = await self._generate_reasoning(item, context, confidence_score)
        
        # Get pairing suggestions
        pairing_suggestions = await self._get_pairing_suggestions(item)
        
        return Recommendation(
            item_id=item['id'],
            item_name=item['name'],
            category=item['category'],
            confidence_score=confidence_score,
            reasoning=reasoning,
            price=item.get('price', 0),
            image_url=item.get('image_url', ''),
            dietary_tags=item.get('dietary_tags', []),
            nutritional_info=item.get('nutritional_info', {}),
            customization_options=item.get('customization_options', []),
            pairing_suggestions=pairing_suggestions,
            popularity_score=item.get('popularity_score', 0.5),
            seasonal_relevance=item.get('seasonal_relevance', 0.5)
        )
    
    async def _generate_reasoning(
        self, 
        item: Dict[str, Any], 
        context: RecommendationContext, 
        confidence_score: float
    ) -> str:
        """Generate human-readable reasoning for recommendation."""
        
        reasons = []
        
        # Time-based reasoning
        if context.current_time:
            current_hour = context.current_time.hour
            if 6 <= current_hour <= 11 and 'breakfast' in item.get('tags', []):
                reasons.append("Perfect for breakfast")
            elif 11 <= current_hour <= 16 and 'lunch' in item.get('tags', []):
                reasons.append("Great lunch option")
            elif 16 <= current_hour <= 22 and 'dinner' in item.get('tags', []):
                reasons.append("Excellent dinner choice")
        
        # Dietary reasoning
        if context.dietary_restrictions:
            for restriction in context.dietary_restrictions:
                if restriction.lower() in [tag.lower() for tag in item.get('dietary_tags', [])]:
                    reasons.append(f"Meets your {restriction} dietary needs")
        
        # Popularity reasoning
        if item.get('popularity_score', 0) > 0.8:
            reasons.append("Customer favorite")
        
        # Seasonal reasoning
        if item.get('seasonal_relevance', 0) > 0.8:
            reasons.append("Seasonal special")
        
        # Confidence-based reasoning
        if confidence_score > 0.8:
            reasons.append("Highly recommended for you")
        elif confidence_score > 0.6:
            reasons.append("Good match for your preferences")
        
        return " • ".join(reasons) if reasons else "Based on your preferences"
    
    async def _get_pairing_suggestions(self, item: Dict[str, Any]) -> List[str]:
        """Get pairing suggestions for an item."""
        # Simplified pairing logic
        pairings = {
            'boneless_wings': ['Ranch Dressing', 'Blue Cheese', 'Celery Sticks'],
            'riblets': ['BBQ Sauce', 'Coleslaw', 'French Fries'],
            'salad': ['House Dressing', 'Croutons', 'Fresh Bread'],
            'steak': ['Mashed Potatoes', 'Grilled Vegetables', 'Red Wine'],
            'pasta': ['Garlic Bread', 'Caesar Salad', 'White Wine']
        }
        
        item_name = item['name'].lower()
        for key, suggestions in pairings.items():
            if key in item_name:
                return suggestions
        
        return []
    
    async def _get_fallback_recommendations(self, limit: int) -> List[Recommendation]:
        """Get fallback recommendations when AI fails."""
        try:
            # Get popular items as fallback
            menu_items = await self._get_menu_items()
            popular_items = sorted(
                menu_items.items(),
                key=lambda x: x[1].get('popularity_score', 0),
                reverse=True
            )[:limit]
            
            recommendations = []
            for item_id, item in popular_items:
                recommendation = await self._create_recommendation(
                    item, 0.5, RecommendationContext()
                )
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Fallback recommendations error: {e}")
            return []
    
    # Helper methods (simplified for demo)
    async def _get_menu_items(self) -> Dict[str, Any]:
        """Get menu items from database."""
        # Mock data for demo
        return {
            '1': {
                'id': '1',
                'name': 'Boneless Wings',
                'description': 'Crispy breaded chicken wings tossed in your choice of sauce',
                'category': 'Appetizers',
                'price': 10.99,
                'image_url': '/static/images/boneless-wings.jpg',
                'dietary_tags': ['gluten-free', 'high-protein'],
                'allergens': ['soy'],
                'nutritional_info': {'calories': 450, 'protein': 25, 'carbs': 15},
                'customization_options': ['Classic Buffalo', 'Honey BBQ', 'Sweet Asian Chile'],
                'tags': ['appetizer', 'sharing', 'spicy'],
                'popularity_score': 0.9,
                'seasonal_relevance': 0.7
            },
            '2': {
                'id': '2',
                'name': 'Riblets',
                'description': 'Tender pork riblets glazed with our signature BBQ sauce',
                'category': 'Entrees',
                'price': 15.99,
                'image_url': '/static/images/riblets.jpg',
                'dietary_tags': ['high-protein'],
                'allergens': ['gluten'],
                'nutritional_info': {'calories': 650, 'protein': 35, 'carbs': 25},
                'customization_options': ['BBQ Sauce', 'Honey Mustard'],
                'tags': ['entree', 'bbq', 'comfort'],
                'popularity_score': 0.8,
                'seasonal_relevance': 0.6
            }
        }
    
    async def _get_user_orders(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's order history."""
        # Mock data for demo
        return [
            {'item_id': '1', 'quantity': 2, 'rating': 5},
            {'item_id': '2', 'quantity': 1, 'rating': 4}
        ]
    
    async def _find_similar_users(self, user_id: str, user_orders: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        """Find users with similar preferences."""
        # Mock data for demo
        return [('user2', 0.8), ('user3', 0.6)]
    
    async def _get_user_item_preference(self, user_id: str, item_id: str) -> float:
        """Get user's preference for a specific item."""
        # Mock data for demo
        return 0.7
    
    async def _get_user_preferences(self, user_id: str) -> str:
        """Get user's text preferences."""
        # Mock data for demo
        return "spicy food comfort dining casual"
    
    async def _get_popularity_data(self) -> Dict[str, float]:
        """Get popularity data for menu items."""
        # Mock data for demo
        return {'1': 0.9, '2': 0.8}
    
    def _generate_cache_key(self, context: RecommendationContext) -> str:
        """Generate cache key for recommendations."""
        return f"rec_{context.user_id}_{hash(str(context))}"
    
    async def generate_recommendations_async(
        self, 
        user_id: str, 
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Async wrapper for generating recommendations."""
        rec_context = RecommendationContext(
            user_id=user_id,
            current_time=datetime.now(),
            **context
        )
        
        recommendations = await self.generate_recommendations(rec_context)
        return [recommendation.__dict__ for recommendation in recommendations] 