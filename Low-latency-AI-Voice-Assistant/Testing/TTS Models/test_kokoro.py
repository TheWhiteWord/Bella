"""Test script for Kokoro TTS integration."""
import os
import sys
import asyncio
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.kokoro_tts.kokoro_tts import KokoroTTSWrapper

async def test_tts(device_index: int = None):
    """Test Kokoro TTS functionality.
    
    Args:
        device_index (int, optional): Audio device index to use
    """
    print("\n=== Testing Kokoro TTS ===")
    
    try:
        # Initialize TTS engine
        print("\nInitializing TTS engine...")
        tts = KokoroTTSWrapper(device_index=device_index)
        
        # Test with sample text
        test_text = "Hello! This is a test of the Kokoro text-to-speech system."
        print(f"\nGenerating speech for: {test_text}")
        
        await tts.generate_speech(test_text)
        print("\nSpeech generation completed!")
        
    except Exception as e:
        print(f"\nError during TTS test: {e}")
        raise
    finally:
        if 'tts' in locals():
            tts.stop()

def list_audio_devices():
    """List available audio output devices."""
    import sounddevice as sd
    
    print("\nAvailable audio devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        out_channels = dev['max_output_channels']
        if out_channels > 0:  # Only show output devices
            print(f"{i}: {dev['name']}")
            print(f"   Outputs: {out_channels}")
            print(f"   Sample rates: {dev['default_samplerate']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Kokoro TTS integration")
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit"
    )
    parser.add_argument(
        "--device",
        type=int,
        help="Index of audio output device to use"
    )
    
    args = parser.parse_args()
    
    if args.list_devices:
        list_audio_devices()
        sys.exit(0)
        
    asyncio.run(test_tts(args.device))