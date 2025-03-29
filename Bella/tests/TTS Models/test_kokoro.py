"""Test script for Kokoro TTS integration.

This test script verifies the Kokoro TTS functionality using PipeWire/PulseAudio for audio output.
Does NOT use PortAudio - all audio playback is handled through PipeWire's PulseAudio compatibility
layer using paplay and pactl commands.

Requirements:
    - Python packages:
        - kokoro>=0.9.2
        - misaki[en]
        - numpy
    - System dependencies:
        - PipeWire/PulseAudio (for audio output)
            - paplay command
            - pactl command
        - espeak-ng (for fallback and non-English languages)
"""
import os
import sys
import asyncio
import argparse
import subprocess
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Define common results directory structure
RESULTS_DIR = os.path.join(project_root, "results")
TEST_RESULTS_DIR = os.path.join(RESULTS_DIR, "test_results")
AUDIO_TEST_DIR = os.path.join(TEST_RESULTS_DIR, "audio_tests")
STT_TEST_DIR = os.path.join(TEST_RESULTS_DIR, "stt_tests") 
TTS_TEST_DIR = os.path.join(TEST_RESULTS_DIR, "tts_tests")

# Create directories if they don't exist
for dir_path in [RESULTS_DIR, TEST_RESULTS_DIR, AUDIO_TEST_DIR, STT_TEST_DIR, TTS_TEST_DIR]:
    os.makedirs(dir_path, exist_ok=True)

from src.audio.kokoro_tts.kokoro_tts import KokoroTTSWrapper

async def test_tts(sink_name: str = None):
    """Test Kokoro TTS functionality.
    
    Args:
        sink_name (str, optional): PulseAudio sink name to use
    """
    print("\n=== Testing Kokoro TTS ===")
    
    try:
        # Initialize TTS engine
        print("\nInitializing TTS engine...")
        tts = KokoroTTSWrapper(sink_name=sink_name)
        
        # Test with sample text
        test_text = "Hello! This is a test of the Kokoro text-to-speech system."
        print(f"\nGenerating speech for: {test_text}")
        
        # Save test details
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_log = os.path.join(TTS_TEST_DIR, f"tts_test_{timestamp}.log")
        with open(test_log, "w") as f:
            f.write(f"Test timestamp: {timestamp}\n")
            f.write(f"Test text: {test_text}\n")
            f.write(f"Audio sink: {sink_name or 'default'}\n")
        
        await tts.generate_speech(test_text)
        print("\nSpeech generation completed!")
        print(f"Test log saved to: {test_log}")
        
    except Exception as e:
        print(f"\nError during TTS test: {e}")
        # Log error
        if 'test_log' in locals():
            with open(test_log, "a") as f:
                f.write(f"\nError: {str(e)}\n")
        raise
    finally:
        if 'tts' in locals():
            tts.stop()

def list_audio_devices():
    """List available PulseAudio output sinks."""
    print("\nAvailable audio output devices (PulseAudio sinks):")
    try:
        result = subprocess.run(['pactl', 'list', 'sinks'], 
                              capture_output=True, text=True, check=True)
        print("\nFull audio device list:")
        for line in result.stdout.split('\n'):
            if any(key in line for key in ['Name:', 'Description:', 'State:']):
                print(line.strip())
            elif line.startswith('Sink #'):
                print(f"\n{line.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"Error listing audio devices: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Kokoro TTS integration")
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit"
    )
    parser.add_argument(
        "--sink",
        type=str,
        help="Name of PulseAudio sink to use"
    )
    
    args = parser.parse_args()
    
    if args.list_devices:
        list_audio_devices()
        sys.exit(0)
        
    asyncio.run(test_tts(args.sink))