"""Chatterbox TTS wrapper for voice synthesis.

This module provides a wrapper around the Chatterbox-Turbo TTS model for real-time 
voice synthesis with zero-shot voice cloning. Uses PipeWire/PulseAudio for audio output.
"""
import os
import asyncio
import tempfile
import wave
import subprocess
import logging
import numpy as np
import torch
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Default paths
PROJECT_ROOT = "/media/theww/AI/Code/AI/Bella/Bella"
DEFAULT_REFERENCE_VOICE = os.path.join(PROJECT_ROOT, "src/audio/clone_files/her_clone_short_15s_24k.wav")

# Set up basic logger
logger = logging.getLogger("chatterbox_tts")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

def check_cuda_availability():
    """Check if CUDA is actually available."""
    if not torch.cuda.is_available():
        return False, "cpu"
    return True, "cuda"

class ChatterboxTTSWrapper:
    """Wrapper for Chatterbox-Turbo TTS with PipeWire audio output."""

    def __init__(
        self,
        audio_prompt_path: str = DEFAULT_REFERENCE_VOICE,
        sink_name: str = None,
        device: str = None,
    ):
        """Initialize Chatterbox TTS engine.
        
        Args:
            audio_prompt_path (str): Path to reference audio for voice cloning.
            sink_name (str, optional): Name of PulseAudio sink to use.
            device (str, optional): Device to use for inference.
        """
        logger.info("Initializing Chatterbox-Turbo TTS...")

        if device is not None:
            self.device = device
        else:
            _, self.device = check_cuda_availability()
            
        try:
            self.model = ChatterboxTurboTTS.from_pretrained(device=self.device)
            logger.info(f"Chatterbox-Turbo loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load Chatterbox-Turbo: {e}")
            raise

        self.audio_prompt_path = audio_prompt_path
        self.sample_rate = self.model.sr
        self.sink_name = sink_name
        self.is_stopped = False

        if not self.sink_name:
            try:
                result = subprocess.run(['pactl', 'get-default-sink'], capture_output=True, text=True, check=True)
                self.sink_name = result.stdout.strip()
                logger.info(f"Using default audio sink: {self.sink_name}")
            except Exception as e:
                logger.debug(f"Could not get default audio sink: {e}")

    def stop(self):
        """Stop any ongoing audio playback."""
        self.is_stopped = True
        try:
            subprocess.run(['pactl', 'send-message', 'stop-playback'], 
                         capture_output=True, check=False)
        except Exception as e:
            logger.debug(f"Error stopping audio: {e}")

    async def generate_speech(self, text: str, audio_prompt_path: str = None, 
                             temperature: float = 0.7) -> None:
        """Generate and play speech from text.
        
        Args:
            text (str): Text to convert to speech.
            audio_prompt_path (str, optional): Path to reference audio. If None, uses default.
            temperature (float): Controls vocal variety (0.1-1.0).
        """
        if not text:
            return

        self.is_stopped = False
        prompt_path = audio_prompt_path or self.audio_prompt_path
        
        try:
            # Generation (Note: Turbo variant expects fewer parameters to avoid warnings)
            wav = self.model.generate(
                text, 
                audio_prompt_path=prompt_path,
                temperature=temperature
            )
            
            if self.is_stopped:
                return

            # Save to temporary file and play via paplay
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                ta.save(temp_wav.name, wav.cpu(), self.sample_rate)
                
                cmd = ['paplay']
                if self.sink_name:
                    cmd.extend(['--device', self.sink_name])
                cmd.append(temp_wav.name)
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()
                os.unlink(temp_wav.name)
                
            # Clear CUDA cache after synthesis to free up memory for Whisper
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            raise
