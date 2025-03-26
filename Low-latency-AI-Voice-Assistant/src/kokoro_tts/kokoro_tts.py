"""Kokoro TTS implementation for voice synthesis."""
import os
import asyncio
from pathlib import Path
from RealtimeTTS import TextToAudioStream, KokoroEngine

class KokoroTTSWrapper:
    """Wrapper for Kokoro TTS engine with specific model and voice settings."""
    
    def __init__(self, 
                 default_voice: str = "af_heart",
                 speed: float = 0.9):
        """Initialize Kokoro TTS with specific voice and settings.
        
        Args:
            default_voice (str): Voice ID to use (default: af_heart) 
            speed (float): Speech speed multiplier (default: 0.9)
        """
        # Initialize Kokoro engine with specified settings
        self.engine = KokoroEngine(
            default_voice=default_voice,
        )
        
        # Set speech speed
        self.engine.set_speed(speed)
        
        # Create audio stream
        self.stream = TextToAudioStream(engine=self.engine)
    
    async def generate_speech(self, text: str):
        """Generate and play speech for the given text.
        
        Args:
            text (str): Text to convert to speech
        """
        try:
            self.stream.feed(text)
            await self.stream.play_async()
            while self.stream.is_playing():
                await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Error generating speech: {e}")
    
    def is_playing(self) -> bool:
        """Check if audio is currently playing.
        
        Returns:
            bool: True if audio is playing, False otherwise
        """
        return self.stream.is_playing()
    
    def stop(self):
        """Stop audio playback."""
        self.stream.stop()
        if hasattr(self.engine, 'shutdown'):
            self.engine.shutdown()  # Call shutdown if the method exists