"""Kokoro TTS implementation for voice synthesis."""
import os
import asyncio
import logging
import sounddevice as sd
from pathlib import Path
from RealtimeTTS import TextToAudioStream, KokoroEngine

# Set up logging
logger = logging.getLogger(__name__)

class KokoroTTSWrapper:
    """Wrapper for Kokoro TTS engine with specific model and voice settings."""
    
    def __init__(self, 
                 default_voice: str = "af_heart",
                 speed: float = 0.9,
                 device_index: int = None):
        """Initialize Kokoro TTS with specific voice and settings."""
        try:
            # List available audio devices first
            devices = sd.query_devices()
            logger.info("Available audio devices:")
            for i, dev in enumerate(devices):
                logger.info(f"{i}: {dev['name']} (outputs: {dev['max_output_channels']})")
            
            # Set default device if specified
            if device_index is not None:
                # Get device info to match sample rate
                device_info = devices[device_index]
                sample_rate = int(device_info['default_samplerate'])
                logger.info(f"Using audio device {device_index}: {device_info['name']}")
                logger.info(f"Sample rate: {sample_rate}")
                
                # Configure sounddevice settings
                sd.default.device = (None, device_index)  # (input, output)
                sd.default.samplerate = sample_rate
                
            # Initialize Kokoro engine
            logger.info("Initializing Kokoro engine...")
            self.engine = KokoroEngine(
                default_voice=default_voice
            )
            
            # Set speech speed
            self.engine.set_speed(speed)
            
            # Create stream
            logger.info("Creating audio stream...")
            self.stream = TextToAudioStream(
                engine=self.engine
            )
            logger.info("Kokoro TTS initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro TTS: {e}")
            raise
    
    async def generate_speech(self, text: str):
        """Generate and play speech for the given text."""
        try:
            logger.info("Feeding text to TTS engine...")
            self.stream.feed(text)
            
            logger.info("Starting audio playback...")
            self.stream.play_async()
            
            # Wait for playback to start and complete
            await asyncio.sleep(1.0)
            
            # Wait for playback to complete
            max_wait = 30
            waited = 0
            while self.stream.is_playing and waited < max_wait:
                await asyncio.sleep(0.1)
                waited += 0.1
                
            if waited >= max_wait:
                logger.warning("TTS playback timed out")
                self.stop()
            else:
                logger.info("Audio playback completed")
                
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            self.stop()
            raise
    
    @property
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self.stream.is_playing if hasattr(self, 'stream') else False
    
    def stop(self):
        """Stop audio playback and clean up resources."""
        try:
            if hasattr(self, 'stream'):
                logger.info("Stopping audio stream...")
                self.stream.stop()
            if hasattr(self, 'engine') and hasattr(self.engine, 'shutdown'):
                logger.info("Shutting down TTS engine...")
                self.engine.shutdown()
        except Exception as e:
            logger.error(f"Error stopping TTS: {e}")