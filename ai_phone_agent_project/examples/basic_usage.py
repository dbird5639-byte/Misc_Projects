"""
Basic usage example for AI Phone Agent
"""

import sys
import os
from pathlib import Path

# Add to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import from packages
# from agent.phone_agent import PhoneAgent
# from agent.voice_processor import VoiceProcessor
# from agent.conversation_manager import ConversationManager
# from utils.knowledge_utils import KnowledgeBase
# from utils.audio_utils import AudioHandler
# from utils.twilio_utils import TwilioHandler
# from config.settings import TESTING_SETTINGS

# Handle import errors gracefully for demonstration
try:
    # type: ignore
    from agent.phone_agent import PhoneAgent  # noqa
    from config.settings import TESTING_SETTINGS
    IMPORTS_AVAILABLE = True
except ImportError:
    print("Note: Some modules not available - this is expected for demonstration")
    IMPORTS_AVAILABLE = False


def main():
    """Demonstrate basic usage of the AI Phone Agent."""
    print("🤖 AI Phone Agent - Basic Usage Example")
    print("=" * 50)
    
    # Initialize the phone agent
    print("Initializing phone agent...")
    agent = PhoneAgent()
    
    # Start a test call
    print("\nStarting a test call...")
    call_id = "example_call_001"
    if agent.start_call(call_id):
        print("✅ Test call started successfully!")
        
        # Simulate some conversation
        test_messages = [
            "Hello, I need help with my account",
            "What are your business hours?",
            "How much does your product cost?",
            "I need to reset my password",
            "Thank you for your help, goodbye"
        ]
        
        print("\nSimulating conversation:")
        print("-" * 30)
        
        for message in test_messages:
            print(f"\nUser: {message}")
            
            # Process the message
            response = agent.process_audio_input(message.encode())
            print(f"Agent: {response}")
            
            # Check if call should end
            if "goodbye" in message.lower():
                break
        
        # Get call status
        status = agent.get_call_status()
        print(f"\nCall Status: {status}")
        
    else:
        print("❌ Failed to start test call")
    
    print("\nExample completed!")


def demonstrate_knowledge_base():
    """Demonstrate knowledge base functionality."""
    print("\n📚 Knowledge Base Example")
    print("=" * 30)
    
    from utils.knowledge_utils import KnowledgeBase
    
    # Initialize knowledge base
    kb = KnowledgeBase()
    
    # Test some queries
    test_queries = [
        "What are your business hours?",
        "How much does your product cost?",
        "I need help with my account",
        "What is your return policy?",
        "Where is your company located?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = kb.get_response(query)
        if response:
            print(f"Response: {response}")
        else:
            print("Response: No matching information found")
    
    # Show statistics
    stats = kb.get_statistics()
    print(f"\nKnowledge Base Statistics: {stats}")


def demonstrate_voice_processing():
    """Demonstrate voice processing functionality."""
    print("\n🎤 Voice Processing Example")
    print("=" * 30)
    
    from agent.voice_processor import VoiceProcessor
    
    # Initialize voice processor
    vp = VoiceProcessor()
    
    if vp.initialize():
        print("✅ Voice processor initialized")
        
        # Test text-to-speech
        test_text = "Hello, this is a test of the voice processing system."
        print(f"\nConverting text to speech: '{test_text}'")
        
        audio_data = vp.text_to_speech(test_text)
        if audio_data:
            print("✅ Text-to-speech conversion successful")
        else:
            print("❌ Text-to-speech conversion failed")
        
        # Test speech-to-text
        print("\nSimulating speech-to-text conversion...")
        test_audio = b"simulated_audio_data"
        text_result = vp.speech_to_text(test_audio)
        if text_result:
            print(f"✅ Speech-to-text result: '{text_result}'")
        else:
            print("❌ Speech-to-text conversion failed")
        
        # Get status
        status = vp.get_status()
        print(f"\nVoice Processor Status: {status}")
        
        vp.cleanup()
    else:
        print("❌ Failed to initialize voice processor")


def demonstrate_conversation_management():
    """Demonstrate conversation management functionality."""
    print("\n💬 Conversation Management Example")
    print("=" * 35)
    
    from agent.conversation_manager import ConversationManager
    
    # Initialize conversation manager
    cm = ConversationManager()
    
    # Set personality
    cm.set_personality({
        "name": "Customer Service AI",
        "tone": "friendly",
        "style": "helpful",
        "expertise": "customer support"
    })
    
    # Simulate conversation
    conversation = [
        "Hello, I need help",
        "I can't log into my account",
        "I forgot my password",
        "Thank you for helping me"
    ]
    
    print("Simulating conversation:")
    print("-" * 25)
    
    for message in conversation:
        print(f"\nUser: {message}")
        response = cm.generate_response(message)
        print(f"AI: {response}")
    
    # Get conversation history
    history = cm.get_conversation_history()
    print(f"\nConversation History Length: {len(history)}")
    
    # Get context
    context = cm.get_context()
    print(f"Conversation Context: {context}")


def demonstrate_audio_handling():
    """Demonstrate audio handling functionality."""
    print("\n🔊 Audio Handling Example")
    print("=" * 25)
    
    from utils.audio_utils import AudioHandler
    
    # Initialize audio handler
    ah = AudioHandler()
    
    if ah.initialize():
        print("✅ Audio handler initialized")
        
        # Get audio devices
        devices = ah.get_audio_devices()
        print(f"\nAvailable Audio Devices: {devices}")
        
        # Test audio devices
        test_results = ah.test_audio_devices()
        print(f"\nAudio Device Test Results: {test_results}")
        
        # Get active devices
        active_devices = ah.get_active_devices()
        print(f"\nActive Devices: {active_devices}")
        
        # Get status
        status = ah.get_status()
        print(f"\nAudio Handler Status: {status}")
        
        ah.cleanup()
    else:
        print("❌ Failed to initialize audio handler")


def demonstrate_twilio_integration():
    """Demonstrate Twilio integration functionality."""
    print("\n📞 Twilio Integration Example")
    print("=" * 30)
    
    from utils.twilio_utils import TwilioHandler
    
    # Initialize Twilio handler (with dummy credentials for demo)
    th = TwilioHandler()
    
    # Simulate initialization
    if th.initialize("dummy_sid", "dummy_token", "+1234567890"):
        print("✅ Twilio handler initialized")
        
        # Simulate incoming call
        call_data = {
            "CallSid": "test_call_123",
            "From": "+1987654321",
            "To": "+1234567890",
            "type": "incoming_call"
        }
        
        print("\nSimulating incoming call...")
        response = th.handle_incoming_call(call_data)
        print(f"Call Response: {response}")
        
        # Get active calls
        active_calls = th.get_active_calls()
        print(f"\nActive Calls: {active_calls}")
        
        # Get statistics
        stats = th.get_statistics()
        print(f"\nTwilio Statistics: {stats}")
        
        # Get status
        status = th.get_status()
        print(f"\nTwilio Status: {status}")
        
        th.cleanup()
    else:
        print("❌ Failed to initialize Twilio handler")


if __name__ == "__main__":
    print("AI Phone Agent - Complete Example Suite")
    print("=" * 50)
    
    try:
        # Run all demonstrations
        main()
        demonstrate_knowledge_base()
        demonstrate_voice_processing()
        demonstrate_conversation_management()
        demonstrate_audio_handling()
        demonstrate_twilio_integration()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed successfully!")
        print("\nTo run the full system:")
        print("1. Configure your API keys in config/settings.py")
        print("2. Run: python src/api/main.py")
        print("3. Open the web dashboard at: http://localhost:5000")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("This is expected if dependencies are not installed.")
        print("The examples demonstrate the structure and functionality.") 