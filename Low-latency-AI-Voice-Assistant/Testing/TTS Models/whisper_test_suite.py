import asyncio
import os
import time
from pathlib import Path
import sys

# Add parent directory to path to import from Models_interaction
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from Models_interaction.faster_whisper_stt_tiny import transcribe_audio, capture_audio

async def test_transcription_from_file():
    """Test transcription using a known audio file"""
    print("\n=== Testing Whisper Transcription from File ===")
    
    # Use the bella_edit.mp3 file we know exists in the project
    test_audio = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "clone_files",
        "bella_edit.mp3"
    )
    
    if not os.path.exists(test_audio):
        print(f"Error: Test audio file not found at {test_audio}")
        return False
    
    try:
        print(f"Transcribing file: {test_audio}")
        start_time = time.time()
        
        result = await transcribe_audio(test_audio)
        
        transcription_time = time.time() - start_time
        
        if result:
            print(f"Transcription successful!")
            print(f"Text: {result}")
            print(f"Time taken: {transcription_time:.2f} seconds")
            return True
        else:
            print("Transcription failed - no text returned")
            return False
            
    except Exception as e:
        print(f"Error during transcription: {e}")
        return False

async def test_audio_capture():
    """Test audio capture functionality"""
    print("\n=== Testing Audio Capture ===")
    print("Recording... Please speak something...")
    
    try:
        audio_file = await capture_audio()
        
        if audio_file and os.path.exists(audio_file):
            print(f"Audio capture successful!")
            print(f"File saved at: {audio_file}")
            
            # Try to transcribe the captured audio
            print("\nTranscribing captured audio...")
            result = await transcribe_audio(audio_file)
            
            if result:
                print(f"Transcription: {result}")
                return True
            else:
                print("Transcription failed for captured audio")
                return False
        else:
            print("Audio capture failed - no file created")
            return False
            
    except Exception as e:
        print(f"Error during audio capture: {e}")
        return False

async def test_noise_handling():
    """Test transcription with background noise"""
    print("\n=== Testing Noise Handling ===")
    print("Please make some background noise while speaking...")
    
    try:
        audio_file = await capture_audio()
        
        if audio_file and os.path.exists(audio_file):
            print(f"\nAudio capture successful!")
            print(f"File saved at: {audio_file}")
            
            print("\nTranscribing noisy audio...")
            result = await transcribe_audio(audio_file)
            
            if result:
                print(f"Transcription: {result}")
                return True
            else:
                print("Transcription returned empty result")
                return False
        else:
            print("Audio capture failed - no file created")
            return False
            
    except Exception as e:
        print(f"Error during noise test: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return False

async def run_test_suite():
    """Run all Whisper integration tests"""
    print("Starting Whisper Integration Test Suite...")
    
    tests = [
        ("File Transcription", test_transcription_from_file),
        ("Audio Capture", test_audio_capture),
        ("Noise Handling", test_noise_handling)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\nRunning test: {test_name}")
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"Test failed with error: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n=== Test Summary ===")
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")

if __name__ == "__main__":
    asyncio.run(run_test_suite())