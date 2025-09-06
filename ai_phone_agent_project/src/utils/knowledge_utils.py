"""
Knowledge base utilities for RAG (Retrieval-Augmented Generation)
"""

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path


class KnowledgeBase:
    """
    Manages knowledge base for retrieving relevant information and responses.
    """
    
    def __init__(self, knowledge_file: Optional[str] = None):
        """
        Initialize the knowledge base.
        
        Args:
            knowledge_file: Path to knowledge base JSON file
        """
        self.logger = logging.getLogger(__name__)
        self.knowledge_data = []
        self.categories = []
        self.cache = {}
        
        # Load knowledge base
        if knowledge_file:
            self.load_knowledge_base(knowledge_file)
        else:
            # Load default knowledge base
            default_path = Path(__file__).parent.parent.parent / "config" / "knowledge_base.json"
            if default_path.exists():
                self.load_knowledge_base(str(default_path))
    
    def load_knowledge_base(self, filepath: str) -> bool:
        """
        Load knowledge base from JSON file.
        
        Args:
            filepath: Path to knowledge base file
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.knowledge_data = data.get("knowledge_base", [])
            self.categories = data.get("settings", {}).get("categories", [])
            
            self.logger.info(f"Knowledge base loaded: {len(self.knowledge_data)} entries")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load knowledge base: {e}")
            return False
    
    def get_response(self, query: str, max_results: int = 3) -> Optional[str]:
        """
        Get the best response for a given query.
        
        Args:
            query: User query
            max_results: Maximum number of results to consider
            
        Returns:
            str: Best matching response, or None if no match found
        """
        try:
            # Check cache first
            cache_key = query.lower().strip()
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Find best matches
            matches = self._find_matches(query, max_results)
            
            if not matches:
                return None
            
            # Get the best match
            best_match = matches[0]
            response = best_match["answer"]
            
            # Cache the result
            self.cache[cache_key] = response
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error getting response: {e}")
            return None
    
    def _find_matches(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Find matching knowledge base entries for a query.
        
        Args:
            query: User query
            max_results: Maximum number of results
            
        Returns:
            List of matching entries sorted by relevance
        """
        matches = []
        query_lower = query.lower()
        
        for entry in self.knowledge_data:
            score = self._calculate_similarity(query_lower, entry)
            if score > 0.3:  # Minimum similarity threshold
                matches.append({
                    **entry,
                    "similarity_score": score
                })
        
        # Sort by similarity score (highest first)
        matches.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return matches[:max_results]
    
    def _calculate_similarity(self, query: str, entry: Dict[str, Any]) -> float:
        """
        Calculate similarity between query and knowledge base entry.
        
        Args:
            query: User query (lowercase)
            entry: Knowledge base entry
            
        Returns:
            float: Similarity score (0.0 to 1.0)
        """
        try:
            # Check question similarity
            question = entry.get("question", "").lower()
            question_score = self._word_overlap_similarity(query, question)
            
            # Check keyword similarity
            keywords = entry.get("keywords", [])
            keyword_score = 0.0
            if keywords:
                keyword_matches = sum(1 for keyword in keywords if keyword.lower() in query)
                keyword_score = keyword_matches / len(keywords)
            
            # Combine scores (question more important than keywords)
            total_score = (question_score * 0.7) + (keyword_score * 0.3)
            
            return min(total_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def _word_overlap_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate word overlap similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Similarity score (0.0 to 1.0)
        """
        try:
            words1 = set(text1.split())
            words2 = set(text2.split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union)
            
        except Exception as e:
            self.logger.error(f"Error calculating word overlap: {e}")
            return 0.0
    
    def add_entry(self, question: str, answer: str, category: str = "general", 
                  keywords: Optional[List[str]] = None, priority: str = "medium") -> bool:
        """
        Add a new entry to the knowledge base.
        
        Args:
            question: The question or query
            answer: The answer or response
            category: Category for the entry
            keywords: List of keywords for matching
            priority: Priority level (high, medium, low)
            
        Returns:
            bool: True if added successfully
        """
        try:
            entry = {
                "id": f"kb_{len(self.knowledge_data) + 1:03d}",
                "category": category,
                "question": question,
                "answer": answer,
                "keywords": keywords or [],
                "priority": priority
            }
            
            self.knowledge_data.append(entry)
            
            # Clear cache since we added new data
            self.cache.clear()
            
            self.logger.info(f"Added knowledge base entry: {entry['id']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add knowledge base entry: {e}")
            return False
    
    def search_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Search knowledge base entries by category.
        
        Args:
            category: Category to search for
            
        Returns:
            List of entries in the specified category
        """
        try:
            return [entry for entry in self.knowledge_data if entry.get("category") == category]
            
        except Exception as e:
            self.logger.error(f"Error searching by category: {e}")
            return []
    
    def search_by_priority(self, priority: str) -> List[Dict[str, Any]]:
        """
        Search knowledge base entries by priority.
        
        Args:
            priority: Priority level to search for
            
        Returns:
            List of entries with the specified priority
        """
        try:
            return [entry for entry in self.knowledge_data if entry.get("priority") == priority]
            
        except Exception as e:
            self.logger.error(f"Error searching by priority: {e}")
            return []
    
    def get_categories(self) -> List[str]:
        """
        Get all available categories.
        
        Returns:
            List of category names
        """
        return self.categories.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.
        
        Returns:
            Dict containing statistics
        """
        try:
            total_entries = len(self.knowledge_data)
            category_counts = {}
            priority_counts = {}
            
            for entry in self.knowledge_data:
                category = entry.get("category", "unknown")
                priority = entry.get("priority", "unknown")
                
                category_counts[category] = category_counts.get(category, 0) + 1
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
            
            return {
                "total_entries": total_entries,
                "categories": category_counts,
                "priorities": priority_counts,
                "cache_size": len(self.cache)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}
    
    def save_knowledge_base(self, filepath: str) -> bool:
        """
        Save knowledge base to JSON file.
        
        Args:
            filepath: Path to save the knowledge base
            
        Returns:
            bool: True if saved successfully
        """
        try:
            data = {
                "knowledge_base": self.knowledge_data,
                "settings": {
                    "version": "1.0",
                    "categories": self.categories,
                    "total_entries": len(self.knowledge_data)
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Knowledge base saved to: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save knowledge base: {e}")
            return False
    
    def clear_cache(self):
        """Clear the response cache."""
        self.cache.clear()
        self.logger.info("Knowledge base cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache information.
        
        Returns:
            Dict containing cache statistics
        """
        return {
            "cache_size": len(self.cache),
            "cached_queries": list(self.cache.keys())
        } 