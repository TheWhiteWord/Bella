import asyncio
import logging
import numpy as np
import sounddevice as sd
from RealtimeTTS import TextToAudioStream, KokoroEngine

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_tts():
    """Test TTS with explicit audio stream control."""
    stream = None
    engine = None
    
    try:
        
        print("\nInitializing Kokoro TTS...")
        engine = KokoroEngine(default_voice="af_heart")
        
        print("\nCreating audio stream...")
        stream = TextToAudioStream(engine=engine)
        
        # Test text
        text = "Testing, 1, 2, 3."
        print(f"\nTesting with text: {text}")
        
        print("\nGenerating and playing audio...")
        stream.feed(text)
        stream.play_async()
        
        # Wait for playback
        print("Waiting for playback...")
        await asyncio.sleep(4)
        
        print("\nTest completed.")
        input("Did you hear the TTS output? Press Enter to continue...")
        
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        logger.error("Test error details:", exc_info=True)
    finally:
        if stream:
            stream.stop()
        if engine and hasattr(engine, 'shutdown'):
            engine.shutdown()

async def simple_test():
    # Create engine
    engine = KokoroEngine(default_voice="af_heart")
    
    # Create stream
    stream = TextToAudioStream(engine=engine)
    
    # Feed text and play
    stream.feed("This is a simple test.")
    stream.play_async()
    
    # Wait a few seconds
    print("Playing...")
    await asyncio.sleep(4)
    
    # Cleanup
    stream.stop()
    engine.shutdown()

async def main():
    # Basic TTS setup
    engine = KokoroEngine(default_voice="af_heart")
    stream = TextToAudioStream(engine=engine)
    
    print("Starting TTS test...")
    stream.feed("This is a test of text to speech.")
    stream.play_async()
    
    print("Playing audio...")
    await asyncio.sleep(4)
    
    stream.stop()
    engine.shutdown()

if __name__ == "__main__":
    print("\nAvailable audio devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        print(f"{i}: {dev['name']}")
        print(f"   Outputs: {dev['max_output_channels']}")
        print(f"   Sample rate: {dev['default_samplerate']}")
    
    try:
        device = int(input("\nEnter audio device number to test (-1 for default): "))
        if device >= 0:
            sd.default.device = (None, device)  # Set output device
        asyncio.run(test_tts())
        asyncio.run(simple_test())
        asyncio.run(main())
    except ValueError:
        print("Please enter a valid number")