"""
Basic usage example for AI Video Agents Project
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.clips_agent import ClipsAgent
from config.settings import INPUT_VIDEOS_DIR, OUTPUT_CLIPS_DIR

def basic_example():
    """Basic example of using the AI Video Agents system"""
    
    print("=== AI Video Agents - Basic Usage Example ===\n")
    
    # Initialize the clips agent
    print("1. Initializing AI Clips Agent...")
    agent = ClipsAgent()
    print("   ✓ Agent initialized successfully\n")
    
    # Create a sample video file for demonstration
    print("2. Creating sample video file...")
    sample_video = INPUT_VIDEOS_DIR / "sample_video.mp4"
    sample_video.parent.mkdir(parents=True, exist_ok=True)
    
    with open(sample_video, 'w') as f:
        f.write("This is a sample video file for demonstration purposes.")
    print(f"   ✓ Sample video created: {sample_video}\n")
    
    # Generate content plan
    print("3. Generating content plan...")
    content_plan = agent.generate_content_plan(str(sample_video))
    
    if 'error' not in content_plan:
        print(f"   ✓ Content plan generated")
        print(f"   - Estimated clips: {content_plan['estimated_clips']}")
        print(f"   - Total duration: {content_plan['total_duration']:.1f}s")
        print(f"   - Summary: {content_plan['summary'][:100]}...\n")
    else:
        print(f"   ✗ Error generating content plan: {content_plan['error']}\n")
    
    # Process the video
    print("4. Processing video and generating clips...")
    result = agent.process_video(str(sample_video), str(OUTPUT_CLIPS_DIR))
    
    if result['success']:
        print(f"   ✓ Video processed successfully")
        print(f"   - Generated {result['total_clips']} clips")
        print(f"   - Summary: {result['summary'][:100]}...\n")
        
        # Display clip details
        print("5. Generated clips:")
        for i, clip in enumerate(result['clips'], 1):
            print(f"   Clip {i}:")
            print(f"     - Title: {clip['title']}")
            print(f"     - Duration: {clip['end_time'] - clip['start_time']:.1f}s")
            print(f"     - Tags: {', '.join(clip['tags'][:3])}...")
            print(f"     - Path: {clip['path']}")
            print()
    else:
        print(f"   ✗ Error processing video: {result['error']}\n")
    
    # Save results
    print("6. Saving results...")
    results_file = OUTPUT_CLIPS_DIR / "example_results.json"
    agent.save_results(result, str(results_file))
    print(f"   ✓ Results saved to: {results_file}\n")
    
    print("=== Example completed successfully! ===")
    print("\nNext steps:")
    print("1. Add your own video files to data/input_videos/")
    print("2. Configure your API keys in .env file")
    print("3. Run the main script: python src/main.py")
    print("4. Check generated clips in data/output_clips/")

if __name__ == "__main__":
    basic_example() 