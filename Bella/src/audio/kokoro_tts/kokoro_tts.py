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
import logging
from kokoro import KPipeline
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

LOCAL_MODEL_DIR = "/media/theww/AI/Code/AI/Bella/Bella/models/Kokoro-82M"
CONFIG_PATH = f"{LOCAL_MODEL_DIR}/config.json"
MODEL_PATH = f"{LOCAL_MODEL_DIR}/kokoro-v1_0.pth"
VOICE_TENSOR_PATH = f"{LOCAL_MODEL_DIR}/voices/af_bella.pt"
SPEED = 0.9

# Set up basic logger
logger = logging.getLogger("kokoro_tts")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

def check_cuda_availability():
    """Check if CUDA is actually available and properly configured.
    
    Returns:
        tuple: (bool, str) - Whether CUDA is available and the device to use
    """
    # Basic CUDA availability check
    if not torch.cuda.is_available():
        logger.info("CUDA not available according to PyTorch")
        return False, "cpu"
    
    # Try to get device count
    try:
        device_count = torch.cuda.device_count()
        if device_count == 0:
            logger.info("No CUDA devices found despite torch.cuda.is_available() returning True")
            return False, "cpu"
    except Exception as e:
        logger.info(f"Error checking CUDA device count: {e}")
        return False, "cpu"
    
    # Try to initialize a small tensor on CUDA to ensure it's actually working
    try:
        test_tensor = torch.zeros(1).cuda()
        del test_tensor
        logger.info(f"CUDA is available and working with {device_count} device(s)")
        return True, "cuda"
    except Exception as e:
        logger.info(f"Failed to initialize tensor on CUDA: {e}")
        return False, "cpu"


class KokoroTTSWrapper:
    """Wrapper for Kokoro TTS with PipeWire audio output."""

    def __init__(
        self,
        model_config_path: str = CONFIG_PATH,
        model_weights_path: str = MODEL_PATH,
        default_voice: str = VOICE_TENSOR_PATH,
        speed: float = SPEED,
        sink_name: str = None,
        device: str = None,
        lang_code: str = 'a',
    ):
        """Initialize Kokoro TTS with specified settings.
        
        Args:
            model_config_path (str, optional): Path to Kokoro config.json
            model_weights_path (str, optional): Path to Kokoro .pth weights
            default_voice (str): Voice ID or path/tensor to use (default: "af_bella")
            speed (float): Speech speed multiplier (default: 0.9)
            sink_name (str, optional): Name of PulseAudio sink to use for output
            device (str, optional): Device to use for inference (e.g. 'cuda', 'cpu')
            lang_code (str): Language code for pipeline (default: 'a')
        """
        logger.info("Initializing Kokoro TTS...")

        # Determine the optimal device to use
        if device is not None:
            self.device = device
            logger.debug(f"Using user-specified device: {self.device}")
        else:
            cuda_available, auto_device = check_cuda_availability()
            self.device = auto_device
            logger.debug(f"Auto-detected device: {self.device} (CUDA available: {cuda_available})")

        # Initialize Kokoro pipeline with local model if provided
        try:
            if model_config_path and model_weights_path:
                from kokoro.model import KModel
                model = KModel(config=model_config_path, model=model_weights_path, repo_id=None)
                self.pipeline = KPipeline(lang_code=lang_code, model=model)
                logger.info("Kokoro pipeline initialized with local model.")
            else:
                self.pipeline = KPipeline(lang_code=lang_code)
                logger.info("Kokoro pipeline initialized with default model.")
            pass  # Remove verbose model/voice info logging
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro pipeline: {e}")
            raise

        self.default_voice = default_voice
        self.speed = speed
        self.sample_rate = 24000  # Kokoro's default sample rate
        self.is_stopped = False
        self.sink_name = sink_name

        if not self.sink_name:
            try:
                result = subprocess.run(['pactl', 'get-default-sink'], capture_output=True, text=True, check=True)
                self.sink_name = result.stdout.strip()
                logger.info(f"Using default audio sink: {self.sink_name}")
            except Exception as e:
                logger.info(f"Could not get default audio sink: {e}")

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
                logger.debug("Audio output system test successful")
                os.unlink(temp_wav.name)
        except Exception as e:
                logger.info(f"Audio output test failed: {e}")
        
    def stop(self):
        """Stop any ongoing audio playback."""
        self.is_stopped = True
        try:
            subprocess.run(['pactl', 'send-message', 'stop-playback'], 
                         capture_output=True, check=False)
        except Exception as e:
            logger.info(f"Error stopping audio: {e}")
        
    async def generate_speech(self, text: str, voice=None, speed: float = None) -> None:
        """Generate and play speech from text.
        
        Args:
            text (str): Text to convert to speech
            voice (str|Tensor|None): Voice ID, path to .pt, or tensor. If None, uses default_voice
            speed (float, optional): Speech speed. If None, uses self.speed
        """
        if not text:
            return

        self.is_stopped = False
        # Allow passing a path or tensor for voice
        voice_arg = voice if voice is not None else self.default_voice
        if isinstance(voice_arg, str) and voice_arg.endswith('.pt'):
            import torch
            logger.info(f"Loading voice tensor from file: {voice_arg}")
            voice_arg = torch.load(voice_arg, weights_only=True)
        logger.info(f"Generating speech...")

        use_speed = speed if speed is not None else self.speed

        try:
            generator = self.pipeline(
                text,
                voice=voice_arg,
                speed=use_speed,
                split_pattern=r'[.!?]+\s+'
            )
            for i, (gs, ps, audio) in enumerate(generator):
                if self.is_stopped:
                    break
                audio_array = audio.detach().cpu().numpy()
                if audio_array.size > 0:
                    audio_array = audio_array - np.mean(audio_array)
                    max_val = np.max(np.abs(audio_array))
                    if max_val > 0:
                        audio_array = audio_array / max_val * 0.9
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                        with wave.open(temp_wav.name, 'wb') as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(self.sample_rate)
                            wav_int16 = (audio_array * 32767).astype(np.int16)
                            wf.writeframes(wav_int16.tobytes())
                        cmd = ['paplay']
                        if self.sink_name:
                            cmd.extend(['--device', self.sink_name])
                        cmd.append(temp_wav.name)
                        process = await asyncio.create_subprocess_exec(
                            *cmd,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        stdout, stderr = await process.communicate()
                        if process.returncode != 0:
                            logger.error(f"paplay failed: {stderr.decode()}")
                        os.unlink(temp_wav.name)
                if i < len(text.split()) - 1:
                    await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            raise