"""Kokoro TTS wrapper for voice synthesis.

This module provides a wrapper around the Kokoro TTS library for real-time voice synthesis
with configurable voices. Uses PipeWire's PulseAudio compatibility layer for audio output.
"""
import os
import asyncio
import tempfile
import wave
import subprocess
from loguru import logger
from kokoro import KPipeline
import numpy as np

class KokoroTTSWrapper:
    """Wrapper for Kokoro TTS with PipeWire audio output."""
    
    def __init__(self, 
                 model_path: str = None,
                 default_voice: str = "af_heart",
                 speed: float = 0.9,
                 device_index: int = None):
        """Initialize Kokoro TTS with specified settings.
        
        Args:
            model_path (str, optional): Path to Kokoro model. If None, uses default path
            default_voice (str): Voice ID to use (default: "af_heart")
            speed (float): Speech speed multiplier (default: 0.9)
            device_index (int, optional): Index of audio output device to use
        """
        logger.info("Initializing Kokoro TTS...")
        
        # Initialize Kokoro pipeline with English
        self.pipeline = KPipeline(lang_code='a')  # 'a' for American English
        
        # Store settings
        self.default_voice = default_voice
        self.speed = speed
        self.sample_rate = 24000  # Kokoro's default sample rate
        self.is_stopped = False
        
        # Get default sink for audio output
        try:
            result = subprocess.run(['pactl', 'get-default-sink'], 
                                  capture_output=True, text=True, check=True)
            self.default_sink = result.stdout.strip()
            if device_index is not None:
                # If device index specified, get device name
                devices = subprocess.run(['pactl', 'list', 'sinks'], 
                                      capture_output=True, text=True).stdout
                self.default_sink = f"alsa_output.{device_index}"
            logger.info(f"Using audio output: {self.default_sink}")
        except Exception as e:
            logger.warning(f"Could not get default audio sink: {e}")
            self.default_sink = None
        
        logger.info("Kokoro TTS initialized successfully")
        
    def stop(self):
        """Stop any ongoing audio playback."""
        self.is_stopped = True
        try:
            subprocess.run(['pactl', 'send-message', 'stop-playback'], 
                         capture_output=True, check=False)
        except Exception as e:
            logger.warning(f"Error stopping audio: {e}")
        
    async def generate_speech(self, text: str, voice: str = None) -> None:
        """Generate and play speech from text.
        
        Args:
            text (str): Text to convert to speech
            voice (str, optional): Voice ID to use. If None, uses default_voice
        """
        if not text:
            return
            
        self.is_stopped = False
        voice_id = voice or self.default_voice
        
        try:
            # Generate audio with Kokoro pipeline
            generator = self.pipeline(
                text,
                voice=voice_id,
                speed=self.speed,
                split_pattern=r'[.!?]+\s+'  # Split on sentence boundaries
            )
            
            # Process each generated segment
            for i, (gs, ps, audio) in enumerate(generator):
                if self.is_stopped:
                    break
                    
                # Normalize audio
                if audio.size > 0:
                    # Remove DC offset
                    audio = audio - np.mean(audio)
                    # Normalize to prevent clipping
                    max_val = np.max(np.abs(audio))
                    if max_val > 0:
                        audio = audio / max_val * 0.9
                
                # Save as temporary WAV file for PipeWire playback
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                    with wave.open(temp_wav.name, 'wb') as wf:
                        wf.setnchannels(1)  # Mono
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(self.sample_rate)
                        # Convert to 16-bit PCM
                        wav_int16 = (audio * 32767).astype(np.int16)
                        wf.writeframes(wav_int16.tobytes())
                    
                    # Play using PipeWire's PulseAudio compatibility
                    cmd = ['paplay', '--playback-time=0', temp_wav.name]
                    if self.default_sink:
                        cmd.extend(['--device', self.default_sink])
                        
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    await process.wait()
                    os.unlink(temp_wav.name)
            
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            raise