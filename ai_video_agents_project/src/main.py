"""
Main entry point for AI Video Agents Project
"""

import logging
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import LOGGING, INPUT_VIDEOS_DIR, OUTPUT_CLIPS_DIR
from agents.clips_agent import ClipsAgent

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, LOGGING['level']),
        format=LOGGING['format'],
        handlers=[
            logging.FileHandler(LOGGING['file']),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main function to demonstrate the AI Video Agents system"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting AI Video Agents Project")
    
    try:
        # Initialize the clips agent
        agent = ClipsAgent()
        
        # Check if there are any videos to process
        video_files = list(INPUT_VIDEOS_DIR.glob("*.mp4")) + list(INPUT_VIDEOS_DIR.glob("*.avi")) + list(INPUT_VIDEOS_DIR.glob("*.mov"))
        
        if not video_files:
            logger.info("No video files found in input directory. Creating example workflow...")
            demonstrate_workflow(agent)
        else:
            logger.info(f"Found {len(video_files)} video files to process")
            process_videos(agent, video_files)
            
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        return 1
    
    logger.info("AI Video Agents Project completed successfully")
    return 0

def demonstrate_workflow(agent):
    """Demonstrate the workflow with example data"""
    logger = logging.getLogger(__name__)
    
    logger.info("Demonstrating AI Video Agents workflow...")
    
    # Create an example video file for demonstration
    example_video = INPUT_VIDEOS_DIR / "example_video.mp4"
    example_video.parent.mkdir(parents=True, exist_ok=True)
    
    with open(example_video, 'w') as f:
        f.write("This is an example video file for demonstration purposes.")
    
    # Process the example video
    result = agent.process_video(str(example_video), str(OUTPUT_CLIPS_DIR))
    
    if result['success']:
        logger.info(f"Successfully processed example video")
        logger.info(f"Generated {result['total_clips']} clips")
        
        # Display results
        print("\n=== Processing Results ===")
        print(f"Original video: {result['original_video']}")
        print(f"Summary: {result['summary']}")
        print(f"Total clips generated: {result['total_clips']}")
        
        for i, clip in enumerate(result['clips'], 1):
            print(f"\nClip {i}:")
            print(f"  Title: {clip['title']}")
            print(f"  Description: {clip['description']}")
            print(f"  Tags: {', '.join(clip['tags'])}")
            print(f"  Duration: {clip['end_time'] - clip['start_time']:.1f}s")
    else:
        logger.error(f"Failed to process example video: {result['error']}")

def process_videos(agent, video_files):
    """Process multiple video files"""
    logger = logging.getLogger(__name__)
    
    results = []
    
    for video_file in video_files:
        logger.info(f"Processing {video_file.name}...")
        
        result = agent.process_video(str(video_file), str(OUTPUT_CLIPS_DIR))
        results.append(result)
        
        if result['success']:
            logger.info(f"Successfully processed {video_file.name} - Generated {result['total_clips']} clips")
        else:
            logger.error(f"Failed to process {video_file.name}: {result['error']}")
    
    # Save results to file
    results_file = OUTPUT_CLIPS_DIR / "processing_results.json"
    agent.save_results(results, str(results_file))
    
    # Display summary
    successful = sum(1 for r in results if r['success'])
    total_clips = sum(r['total_clips'] for r in results if r['success'])
    
    print(f"\n=== Processing Summary ===")
    print(f"Videos processed: {len(video_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(video_files) - successful}")
    print(f"Total clips generated: {total_clips}")
    print(f"Results saved to: {results_file}")

def show_help():
    """Show help information"""
    print("""
AI Video Agents Project - Help

Usage:
    python src/main.py                    # Run the main application
    python src/main.py --help             # Show this help message

Features:
    - Automated video processing with AI
    - Transcript extraction and analysis
    - Key moment identification
    - Automatic clip generation
    - Content optimization for social media

Setup:
    1. Place your video files in the data/input_videos/ directory
    2. Run the main script
    3. Find generated clips in data/output_clips/

Supported video formats: MP4, AVI, MOV, MKV, WEBM

For more information, see the README.md file.
    """)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        show_help()
    else:
        sys.exit(main()) 