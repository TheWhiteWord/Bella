"""Simple audio recorder implementation."""
import os
import wave
import numpy as np
import sounddevice as sd
import logging
from datetime import datetime
from typing import Optional
import asyncio
import time

class AudioRecorder:
    """Basic audio recorder using sounddevice."""
    
    def __init__(self, config_path: str):
        # Setup basic logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Fixed settings for reliable recording
        self.sample_rate = 48000  # Native rate for most Linux audio
        self.channels = 1         # Mono recording
        self.dtype = np.float32   # Standard format
        
        # Simple buffer configuration
        self.chunk_size = 1024    # Standard chunk size
        
        # Create output directory if needed
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize recording control
        self.stop_recording = asyncio.Event()
        self.recording_started = asyncio.Event()
        self.recorded_data = []
        
        # Set up audio device
        self._setup_audio_device()

    def _setup_audio_device(self) -> None:
        """Configure the default audio input device."""
        try:
            # Get default input device
            device = sd.query_devices(kind='input')
            self.logger.info(f"Using input device: {device['name']}")
            
            # Configure sounddevice
            sd.default.device = device['name']
            sd.default.samplerate = self.sample_rate
            sd.default.channels = self.channels
            sd.default.dtype = self.dtype
            
        except Exception as e:
            self.logger.error(f"Error setting up audio device: {e}")
            raise

    async def record_async(self, duration: Optional[float] = None) -> Optional[np.ndarray]:
        """Record audio.
        
        Args:
            duration: Optional recording duration in seconds.
            
        Returns:
            np.ndarray: Recorded audio data, or None if recording failed.
        """
        self.recorded_data = []
        self.stop_recording.clear()
        self.recording_started.clear()
        
        def callback(indata, frames, time, status):
            if status:
                self.logger.warning(f"Audio callback status: {status}")
            if self.recording_started.is_set():
                self.recorded_data.append(indata.copy())
            if self.stop_recording.is_set():
                raise sd.CallbackStop()

        try:
            # Create and start stream
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=self.chunk_size,
                callback=callback
            )
            
            with stream:
                # Wait for start signal
                await self._wait_for_enter()
                self.recording_started.set()
                self.logger.info("Recording started")
                
                if duration:
                    await asyncio.sleep(duration)
                    self.stop_recording.set()
                else:
                    # Wait for stop signal
                    await self._wait_for_enter()
                    self.stop_recording.set()
            
            if not self.recorded_data:
                self.logger.warning("No audio data recorded")
                return None
            
            # Process recorded data
            audio_data = np.concatenate(self.recorded_data)
            
            # Ensure mono and normalize
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Basic cleanup
            audio_data = np.nan_to_num(audio_data)
            max_val = np.abs(audio_data).max()
            if max_val > 0:
                audio_data = audio_data / max_val
            
            # Save debug recording
            await self._save_debug_recording(audio_data)
            
            return audio_data
            
        except Exception as e:
            self.logger.error(f"Error recording audio: {e}")
            return None

    async def _wait_for_enter(self):
        """Wait for Enter key press."""
        while True:
            if await asyncio.get_event_loop().run_in_executor(None, input) == '':
                break

    async def _save_debug_recording(self, audio_data: np.ndarray):
        """Save a debug copy of the recording."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_path = os.path.join(self.output_dir, f"debug_recording_{timestamp}.wav")
            
            with wave.open(debug_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(4)  # 32-bit float
                wf.setframerate(self.sample_rate)
                audio_int = (audio_data * 2147483647).astype(np.int32)
                wf.writeframes(audio_int.tobytes())
            
            self.logger.info(f"Saved recording to {debug_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving debug recording: {e}")

    def record(self, duration: Optional[float] = None) -> Optional[np.ndarray]:
        """Synchronous wrapper for record_async."""
        return asyncio.run(self.record_async(duration))

    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'stream') and self.stream is not None:
            self.stream.stop()
            self.stream.close()