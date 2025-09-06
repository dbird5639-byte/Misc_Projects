"""
AI Clips Agent - Main agent for processing videos and generating clips
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import json

from ..utils.video_utils import VideoProcessor
from ..utils.ai_utils import AIProcessor
from config.settings import VIDEO_SETTINGS, CONTENT_SETTINGS, AI_MODELS, DEFAULT_AI_MODEL

logger = logging.getLogger(__name__)

class ClipsAgent:
    """
    Main AI agent for processing videos and generating engaging clips
    """
    
    def __init__(self, ai_model: str = DEFAULT_AI_MODEL):
        self.ai_model = ai_model
        self.video_processor = VideoProcessor()
        self.ai_processor = AIProcessor(ai_model)
        
        logger.info(f"ClipsAgent initialized with AI model: {ai_model}")
    
    def process_video(self, video_path: str, output_dir: str) -> Dict:
        """
        Process a single video and generate clips
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save generated clips
            
        Returns:
            Dictionary containing processing results
        """
        try:
            logger.info(f"Processing video: {video_path}")
            
            # Step 1: Extract transcript
            transcript = self.video_processor.extract_transcript(video_path)
            
            # Step 2: Generate summary and identify key moments
            summary = self.ai_processor.generate_summary(transcript)
            key_moments = self.ai_processor.identify_key_moments(transcript)
            
            # Step 3: Create clips
            clips = []
            for i, moment in enumerate(key_moments):
                clip_path = self.video_processor.create_clip(
                    video_path, 
                    moment['start_time'], 
                    moment['end_time'],
                    output_dir,
                    f"clip_{i+1}"
                )
                
                # Generate content for the clip
                title = self.ai_processor.generate_title(moment['content'])
                description = self.ai_processor.generate_description(moment['content'])
                tags = self.ai_processor.generate_tags(moment['content'])
                
                clips.append({
                    'path': clip_path,
                    'title': title,
                    'description': description,
                    'tags': tags,
                    'start_time': moment['start_time'],
                    'end_time': moment['end_time'],
                    'content': moment['content']
                })
            
            return {
                'success': True,
                'original_video': video_path,
                'transcript': transcript,
                'summary': summary,
                'clips': clips,
                'total_clips': len(clips)
            }
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'original_video': video_path
            }
    
    def batch_process(self, video_directory: str, output_directory: str) -> List[Dict]:
        """
        Process multiple videos in a directory
        
        Args:
            video_directory: Directory containing input videos
            output_directory: Directory to save generated clips
            
        Returns:
            List of processing results for each video
        """
        video_dir = Path(video_directory)
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mov"))
        
        logger.info(f"Found {len(video_files)} videos to process")
        
        for video_file in video_files:
            result = self.process_video(str(video_file), str(output_dir))
            results.append(result)
            
            if result['success']:
                logger.info(f"Successfully processed {video_file.name}")
            else:
                logger.error(f"Failed to process {video_file.name}")
        
        return results
    
    def generate_content_plan(self, video_path: str) -> Dict:
        """
        Generate a content plan for a video without creating clips
        
        Args:
            video_path: Path to input video file
            
        Returns:
            Dictionary containing content plan
        """
        try:
            transcript = self.video_processor.extract_transcript(video_path)
            summary = self.ai_processor.generate_summary(transcript)
            key_moments = self.ai_processor.identify_key_moments(transcript)
            
            content_plan = {
                'video_path': video_path,
                'summary': summary,
                'key_moments': key_moments,
                'estimated_clips': len(key_moments),
                'total_duration': self.video_processor.get_video_duration(video_path)
            }
            
            return content_plan
            
        except Exception as e:
            logger.error(f"Error generating content plan for {video_path}: {str(e)}")
            return {'error': str(e)}
    
    def save_results(self, results: Dict, output_path: str):
        """
        Save processing results to a JSON file
        
        Args:
            results: Processing results dictionary
            output_path: Path to save the JSON file
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}") 