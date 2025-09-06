"""
Video Analyzer Module

Analyzes video content to identify valuable segments for clipping and monetization.
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class VideoSegment:
    """Data class for video segments"""
    start_time: float
    end_time: float
    duration: float
    engagement_score: float
    content_type: str
    key_topics: List[str]
    sentiment: str
    quality_score: float

class VideoAnalyzer:
    """
    Analyzes video content to identify valuable segments for clipping
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize video analyzer
        
        Args:
            config: Configuration settings for analysis
        """
        self.config = config or {}
        self.min_segment_length = self.config.get("min_segment_length", 60)
        self.max_segment_length = self.config.get("max_segment_length", 7200)
        self.quality_threshold = self.config.get("quality_threshold", 0.8)
        
    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze a video file to identify valuable segments
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Analysis results with identified segments
        """
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return {"error": f"Could not open video file: {video_path}"}
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Analyze video content
            segments = self._identify_segments(cap, fps, duration)
            
            # Release video capture
            cap.release()
            
            return {
                "video_path": video_path,
                "duration": duration,
                "fps": fps,
                "resolution": f"{width}x{height}",
                "segments": segments,
                "total_segments": len(segments),
                "analysis_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Error analyzing video: {str(e)}"}
    
    def _identify_segments(self, cap: cv2.VideoCapture, fps: float, duration: float) -> List[VideoSegment]:
        """Identify valuable segments in the video"""
        segments = []
        
        # Sample frames for analysis
        sample_interval = int(fps * 5)  # Sample every 5 seconds
        frame_count = 0
        
        current_segment_start = 0
        current_segment_score = 0
        frame_scores = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_interval == 0:
                # Analyze frame
                frame_score = self._analyze_frame(frame)
                frame_scores.append(frame_score)
                
                # Check if we should start/end a segment
                if frame_score > self.quality_threshold:
                    if current_segment_start == 0:
                        current_segment_start = frame_count / fps
                    current_segment_score = max(current_segment_score, frame_score)
                else:
                    # End current segment if it meets criteria
                    if current_segment_start > 0:
                        segment_duration = (frame_count / fps) - current_segment_start
                        if self.min_segment_length <= segment_duration <= self.max_segment_length:
                            segment = self._create_segment(
                                current_segment_start,
                                frame_count / fps,
                                current_segment_score,
                                frame_scores
                            )
                            segments.append(segment)
                        
                        # Reset for next segment
                        current_segment_start = 0
                        current_segment_score = 0
                        frame_scores = []
            
            frame_count += 1
        
        # Handle final segment
        if current_segment_start > 0:
            segment_duration = duration - current_segment_start
            if self.min_segment_length <= segment_duration <= self.max_segment_length:
                segment = self._create_segment(
                    current_segment_start,
                    duration,
                    current_segment_score,
                    frame_scores
                )
                segments.append(segment)
        
        return segments
    
    def _analyze_frame(self, frame: np.ndarray) -> float:
        """Analyze a single frame for quality and engagement potential"""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate various quality metrics
        brightness = np.mean(gray)
        contrast = np.std(gray)
        sharpness = self._calculate_sharpness(gray)
        
        # Normalize scores (0-1 range)
        brightness_score = min(float(brightness / 128), 1.0)
        contrast_score = min(float(contrast / 50), 1.0)
        sharpness_score = min(float(sharpness / 100), 1.0)
        
        # Combined quality score
        quality_score = (brightness_score * 0.3 + 
                        contrast_score * 0.3 + 
                        sharpness_score * 0.4)
        
        return quality_score
    
    def _calculate_sharpness(self, gray_frame: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance"""
        laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
        return float(laplacian.var())
    
    def _create_segment(self, start_time: float, end_time: float, 
                       max_score: float, frame_scores: List[float]) -> VideoSegment:
        """Create a video segment with analysis results"""
        duration = end_time - start_time
        avg_score = float(np.mean(frame_scores)) if frame_scores else 0.0
        
        # Determine content type based on analysis
        content_type = self._classify_content_type(avg_score, duration)
        
        # Extract key topics (placeholder - would use AI in real implementation)
        key_topics = self._extract_key_topics(start_time, end_time)
        
        # Determine sentiment
        sentiment = self._analyze_sentiment(avg_score, frame_scores)
        
        return VideoSegment(
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            engagement_score=avg_score,
            content_type=content_type,
            key_topics=key_topics,
            sentiment=sentiment,
            quality_score=max_score
        )
    
    def _classify_content_type(self, score: float, duration: float) -> str:
        """Classify the type of content in the segment"""
        if score > 0.9:
            return "high_engagement"
        elif score > 0.7:
            return "educational"
        elif score > 0.5:
            return "entertainment"
        else:
            return "background"
    
    def _extract_key_topics(self, start_time: float, end_time: float) -> List[str]:
        """Extract key topics from the segment (placeholder)"""
        # This would integrate with AI services for topic extraction
        topics = ["programming", "coding", "development"]
        return topics[:3]  # Return top 3 topics
    
    def _analyze_sentiment(self, score: float, frame_scores: List[float]) -> str:
        """Analyze sentiment of the segment"""
        if score > 0.8:
            return "positive"
        elif score > 0.6:
            return "neutral"
        else:
            return "negative"
    
    def get_best_segments(self, analysis_result: Dict[str, Any], 
                         count: int = 5) -> List[VideoSegment]:
        """
        Get the best segments from analysis results
        
        Args:
            analysis_result: Results from analyze_video
            count: Number of best segments to return
            
        Returns:
            List of best video segments
        """
        if "error" in analysis_result:
            return []
        
        segments = analysis_result.get("segments", [])
        
        # Sort by engagement score (highest first)
        sorted_segments = sorted(segments, 
                               key=lambda x: x.engagement_score, 
                               reverse=True)
        
        return sorted_segments[:count]
    
    def export_analysis(self, analysis_result: Dict[str, Any], 
                       output_path: str) -> bool:
        """
        Export analysis results to JSON file
        
        Args:
            analysis_result: Analysis results
            output_path: Path to save results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert segments to dictionaries for JSON serialization
            if "segments" in analysis_result:
                segments_data = []
                for segment in analysis_result["segments"]:
                    segments_data.append({
                        "start_time": segment.start_time,
                        "end_time": segment.end_time,
                        "duration": segment.duration,
                        "engagement_score": segment.engagement_score,
                        "content_type": segment.content_type,
                        "key_topics": segment.key_topics,
                        "sentiment": segment.sentiment,
                        "quality_score": segment.quality_score
                    })
                analysis_result["segments"] = segments_data
            
            with open(output_path, 'w') as f:
                json.dump(analysis_result, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error exporting analysis: {e}")
            return False 