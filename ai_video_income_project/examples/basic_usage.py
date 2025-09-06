"""
Basic Usage Example for AI Video Income Generator Project

This example demonstrates how to use the main components of the project
to analyze videos, generate clips, and prepare for monetization.
"""

import sys
import os
import json
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from video_processor.video_analyzer import VideoAnalyzer
from ai_agents.content_analyzer import ContentAnalyzer
from uploader.youtube_uploader import YouTubeUploader

def main():
    """Main example function"""
    print("🎬 AI Video Income Generator - Basic Usage Example")
    print("=" * 60)
    
    # Initialize components
    print("\n1. Initializing components...")
    video_analyzer = VideoAnalyzer()
    content_analyzer = ContentAnalyzer()
    youtube_uploader = YouTubeUploader()
    
    # Example 1: Analyze a video file
    print("\n2. Analyzing video content...")
    video_path = "data/source_videos/example_video.mp4"
    
    # Check if video exists (create mock data if not)
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        print("Creating mock analysis for demonstration...")
        
        # Mock analysis results
        analysis_result = {
            "video_path": video_path,
            "duration": 3600,  # 1 hour
            "fps": 30,
            "resolution": "1920x1080",
            "segments": [
                {
                    "start_time": 300,  # 5 minutes
                    "end_time": 1800,   # 30 minutes
                    "duration": 1500,
                    "engagement_score": 0.85,
                    "content_type": "educational",
                    "key_topics": ["programming", "coding", "tutorial"],
                    "sentiment": "positive",
                    "quality_score": 0.9
                },
                {
                    "start_time": 2400,  # 40 minutes
                    "end_time": 3600,    # 1 hour
                    "duration": 1200,
                    "engagement_score": 0.78,
                    "content_type": "tips",
                    "key_topics": ["development", "tips", "best_practices"],
                    "sentiment": "positive",
                    "quality_score": 0.85
                }
            ],
            "total_segments": 2,
            "analysis_date": datetime.now().isoformat()
        }
    else:
        # Real video analysis
        analysis_result = video_analyzer.analyze_video(video_path)
    
    print(f"Video Analysis Results:")
    print(f"  Duration: {analysis_result.get('duration', 0):.0f} seconds")
    print(f"  Resolution: {analysis_result.get('resolution', 'Unknown')}")
    print(f"  Segments Found: {analysis_result.get('total_segments', 0)}")
    
    # Example 2: Get best segments
    print("\n3. Identifying best segments...")
    best_segments = video_analyzer.get_best_segments(analysis_result, count=3)
    
    print(f"Top {len(best_segments)} segments:")
    for i, segment in enumerate(best_segments, 1):
        print(f"  {i}. {segment.content_type} ({segment.duration:.0f}s)")
        print(f"     Engagement Score: {segment.engagement_score:.2f}")
        print(f"     Topics: {', '.join(segment.key_topics)}")
    
    # Example 3: AI Content Analysis
    print("\n4. Performing AI content analysis...")
    
    # Prepare segments for AI analysis
    segments_for_analysis = []
    for segment in best_segments:
        segments_for_analysis.append({
            "video_path": video_path,
            "start_time": segment.start_time,
            "end_time": segment.end_time
        })
    
    # Analyze segments with AI
    content_analyses = content_analyzer.batch_analyze(segments_for_analysis)
    
    print(f"AI Content Analysis Results:")
    for i, analysis in enumerate(content_analyses, 1):
        print(f"  {i}. {analysis.content_type} segment")
        print(f"     Monetization Score: {analysis.monetization_score:.2f}")
        print(f"     Target Audience: {', '.join(analysis.target_audience)}")
        print(f"     Summary: {analysis.content_summary[:100]}...")
    
    # Example 4: Get top monetizable segments
    print("\n5. Identifying top monetizable segments...")
    top_segments = content_analyzer.get_top_segments(content_analyses, count=2)
    
    print(f"Top monetizable segments:")
    for i, segment in enumerate(top_segments, 1):
        print(f"  {i}. {segment.content_type} (Score: {segment.monetization_score:.2f})")
        print(f"     Duration: {segment.end_time - segment.start_time:.0f}s")
        print(f"     Topics: {', '.join(segment.key_topics)}")
    
    # Example 5: Export results
    print("\n6. Exporting analysis results...")
    
    # Create results directory
    results_dir = "example_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Export video analysis
    video_analyzer.export_analysis(
        analysis_result, 
        os.path.join(results_dir, "video_analysis.json")
    )
    
    # Export content analyses
    content_analyzer.export_analyses(
        content_analyses, 
        os.path.join(results_dir, "content_analyses.json")
    )
    
    print(f"Results exported to '{results_dir}' directory")
    
    # Example 6: Revenue estimation
    print("\n7. Estimating potential revenue...")
    
    total_potential_revenue = 0
    for segment in top_segments:
        # Estimate views based on monetization score
        estimated_views = int(segment.monetization_score * 15000)  # 15k max views
        
        # Calculate revenue (example: $69 per 10k views)
        revenue = (estimated_views / 10000) * 69
        total_potential_revenue += revenue
        
        print(f"  {segment.content_type}: {estimated_views:,} estimated views = ${revenue:.2f}")
    
    print(f"Total Potential Revenue: ${total_potential_revenue:.2f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Example completed successfully!")
    print("\nKey Takeaways:")
    print("  • Use VideoAnalyzer to identify valuable segments")
    print("  • Use ContentAnalyzer to assess monetization potential")
    print("  • Focus on high-engagement, educational content")
    print("  • Target specific audiences for better performance")
    print("  • Track and optimize based on view performance")
    print("\nNext Steps:")
    print("  1. Find source videos from approved creators")
    print("  2. Use AI tools to identify best segments")
    print("  3. Create engaging thumbnails and titles")
    print("  4. Upload to YouTube and track performance")
    print("  5. Submit for payment when reaching thresholds")
    
    print(f"\nExample completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 