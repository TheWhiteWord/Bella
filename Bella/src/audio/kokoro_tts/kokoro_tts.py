"""Kokoro TTS wrapper for voice synthesis.

This module provides a wrapper around the Kokoro TTS library for real-time voice synthesis
with configurable voices. Uses PipeWire/PulseAudio for audio output (NOT PortAudio).

Dependencies:
    - kokoro>=0.9.2: Main TTS engine
    - misaki[en]: For English language support
    - numpy: For audio array processing
    - PipeWire/PulseAudio: For audio output (system dependency)
        - Uses paplay for audio playback
        - Uses pactl for device management
    - espeak-ng: For fallback and non-English languages (system dependency)

Note:
    This implementation specifically avoids using PortAudio/sounddevice for audio output,
    instead relying on PipeWire's PulseAudio compatibility layer for better system integration
    and stability.
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
                 sink_name: str = None):
        """Initialize Kokoro TTS with specified settings.
        
        Args:
            model_path (str, optional): Path to Kokoro model. If None, uses default path
            default_voice (str): Voice ID to use (default: "af_heart")
            speed (float): Speech speed multiplier (default: 0.9)
            sink_name (str, optional): Name of PulseAudio sink to use for output
        """
        logger.info("Initializing Kokoro TTS...")
        
        # Initialize Kokoro pipeline with English
        try:
            self.pipeline = KPipeline(lang_code='a')  # 'a' for American English
            logger.info("Kokoro pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro pipeline: {e}")
            raise
        
        # Store settings
        self.default_voice = default_voice
        self.speed = speed
        self.sample_rate = 24000  # Kokoro's default sample rate
        self.is_stopped = False
        self.sink_name = sink_name
        
        if not self.sink_name:
            try:
                # Get default sink if none specified
                result = subprocess.run(['pactl', 'get-default-sink'], 
                                    capture_output=True, text=True, check=True)
                self.sink_name = result.stdout.strip()
                logger.info(f"Using default audio sink: {self.sink_name}")
            except Exception as e:
                logger.warning(f"Could not get default audio sink: {e}")

        # Test audio output by playing a brief silence
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                silence = np.zeros(int(0.1 * self.sample_rate), dtype=np.int16)  # 0.1s silence
                with wave.open(temp_wav.name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(silence.tobytes())
                
                cmd = ['paplay']
                if self.sink_name:
                    cmd.extend(['--device', self.sink_name])
                cmd.append(temp_wav.name)
                
                subprocess.run(cmd, check=True, capture_output=True)
                logger.info("Audio output system test successful")
                os.unlink(temp_wav.name)
        except Exception as e:
            logger.warning(f"Audio output test failed: {e}")
        
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
        logger.info(f"Generating speech for text: '{text}' with voice: {voice_id}")
        
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
                
                logger.info(f"Processing segment {i}: '{gs}'")
                    
                # Convert PyTorch tensor to numpy array and normalize
                audio_array = audio.detach().cpu().numpy()
                if audio_array.size > 0:
                    # Remove DC offset
                    audio_array = audio_array - np.mean(audio_array)
                    # Normalize to prevent clipping
                    max_val = np.max(np.abs(audio_array))
                    if max_val > 0:
                        audio_array = audio_array / max_val * 0.9
                    
                    logger.info(f"Audio array shape: {audio_array.shape}, min: {audio_array.min():.3f}, max: {audio_array.max():.3f}")
                    
                    # Save and play using PipeWire/PulseAudio
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                        with wave.open(temp_wav.name, 'wb') as wf:
                            wf.setnchannels(1)  # Mono
                            wf.setsampwidth(2)  # 16-bit
                            wf.setframerate(self.sample_rate)
                            # Convert to 16-bit PCM
                            wav_int16 = (audio_array * 32767).astype(np.int16)
                            wf.writeframes(wav_int16.tobytes())
                        
                        logger.info(f"Saved audio segment to temporary file: {temp_wav.name}")
                        
                        # Play using PipeWire's PulseAudio compatibility
                        cmd = ['paplay']
                        if self.sink_name:
                            cmd.extend(['--device', self.sink_name])
                        cmd.append(temp_wav.name)
                        
                        logger.info(f"Playing audio with command: {' '.join(cmd)}")
                        
                        process = await asyncio.create_subprocess_exec(
                            *cmd,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        
                        stdout, stderr = await process.communicate()
                        if process.returncode != 0:
                            logger.error(f"paplay failed: {stderr.decode()}")
                        else:
                            logger.info("paplay completed successfully")
                        
                        os.unlink(temp_wav.name)
                    
                # Small pause between segments for natural pacing
                if i < len(text.split()) - 1:
                    await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error generating speech: {e}", exc_info=True)
            raise