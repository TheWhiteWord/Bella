import os
import yaml
import wave
import pyaudio
import numpy as np
import torch
from threading import Event
import queue
import sys
import select
import time
import logging
from datetime import datetime

class AudioRecorder:
    def __init__(self, config_path):
        self.config_path = os.path.abspath(config_path)
        
        # Load config
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)['audio']
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Expanded error handling for device initialization
        try:
            self.audio = pyaudio.PyAudio()
        except OSError as e:
            self.logger.error(f"Failed to initialize PyAudio: {e}")
            self.logger.info("Trying to recover by resetting ALSA...")
            os.system("pulseaudio -k && sudo alsa force-reload")
            self.audio = pyaudio.PyAudio()
            
        # Find input device with supported sample rate
        self.device_index = self._find_input_device()
        if self.device_index is None:
            available_devices = self._list_available_devices()
            self.logger.error("No suitable audio input device found. Available devices:")
            for device in available_devices:
                self.logger.error(f"  {device}")
            raise RuntimeError("No suitable audio input device found")
            
        # Initialize other attributes
        self.stream = None
        self.stop_recording = Event()
        self.frames = []
        self.error_count = 0
        self.max_retries = 3
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _list_available_devices(self):
        """List all available audio devices with details"""
        devices = []
        for i in range(self.audio.get_device_count()):
            try:
                device_info = self.audio.get_device_info_by_index(i)
                devices.append({
                    'index': i,
                    'name': device_info['name'],
                    'inputs': device_info['maxInputChannels'],
                    'rate': device_info['defaultSampleRate']
                })
            except Exception as e:
                self.logger.error(f"Error getting device {i} info: {e}")
        return devices

    def _find_input_device(self):
        """Find an input device that supports our desired sample rate with enhanced error checking"""
        target_rate = self.config['sample_rate']
        
        # Print available devices for debugging
        self.logger.info("\nAvailable Audio Devices:")
        for i in range(self.audio.get_device_count()):
            try:
                device_info = self.audio.get_device_info_by_index(i)
                self.logger.info(f"Device {i}: {device_info['name']}")
                self.logger.info(f"  Max Input Channels: {device_info['maxInputChannels']}")
                self.logger.info(f"  Default Sample Rate: {device_info['defaultSampleRate']}")
                self.logger.info(f"  Supported Sample Rates: {self._get_supported_rates(i)}")
            except Exception as e:
                self.logger.error(f"Error getting device {i} info: {e}")
        
        # First try default input device
        try:
            default_device = self.audio.get_default_input_device_info()
            if default_device['maxInputChannels'] > 0:
                # Verify sample rate support
                supported_rates = self._get_supported_rates(default_device['index'])
                if target_rate in supported_rates:
                    self.logger.info(f"Using default input device: {default_device['name']}")
                    return default_device['index']
        except Exception:
            self.logger.warning("Could not use default input device")
            
        # Scan all devices
        for i in range(self.audio.get_device_count()):
            try:
                device_info = self.audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    # Verify sample rate support
                    supported_rates = self._get_supported_rates(i)
                    if target_rate in supported_rates:
                        self.logger.info(f"Using input device: {device_info['name']}")
                        return i
            except Exception:
                continue
                
        return None

    def _get_supported_rates(self, device_index):
        """Test which sample rates are supported by the device"""
        test_rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 96000]
        supported_rates = []
        
        for rate in test_rates:
            try:
                # Try to open a test stream
                stream = self.audio.open(
                    format=eval(self.config['format']),
                    channels=self.config['channels'],
                    rate=rate,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=self.config['chunk_size'],
                    start=False
                )
                stream.close()
                supported_rates.append(rate)
            except:
                pass
                
        return supported_rates

    def _save_debug_recording(self, audio_data, is_float=True):
        """Save a debug copy of the recording with timestamp"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_filename = os.path.join(self.output_dir, f"debug_recording_{timestamp}.wav")
            
            if is_float:
                # Convert float32 back to int16 for WAV file
                audio_data = (audio_data.numpy() * 32768.0).astype(np.int16)
            
            with wave.open(debug_filename, 'wb') as wf:
                wf.setnchannels(self.config['channels'])
                wf.setsampwidth(self.audio.get_sample_size(eval(self.config['format'])))
                wf.setframerate(self.config['sample_rate'])
                wf.writeframes(audio_data.tobytes())
                
            self.logger.info(f"Saved debug recording to: {debug_filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving debug recording: {e}")
        
    def start_stream(self):
        """Initialize and start the audio stream with enhanced error handling"""
        try:
            if self.stream:
                self.stream.close()
                
            self.stream = self.audio.open(
                format=eval(self.config['format']),
                channels=self.config['channels'],
                rate=self.config['sample_rate'],
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.config['chunk_size'],
                stream_callback=self._audio_callback
            )
            
            self.stop_recording.clear()
            self.frames = []
            self.error_count = 0
            self.logger.info("Audio stream started successfully")
            
        except Exception as e:
            self.logger.error(f"Error starting audio stream: {e}")
            if self.stream:
                self.stream.close()
            self.stream = None
            raise

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Handle audio input with error detection"""
        if status:
            self.logger.warning(f"Audio input status: {status}")
            self.error_count += 1
            if self.error_count > self.max_retries:
                self.logger.error("Too many audio errors, stopping recording")
                self.stop_recording.set()
                return (None, pyaudio.paComplete)
        
        try:
            audio_chunk = np.frombuffer(in_data, dtype=np.int16)
            
            # Check for audio issues
            if np.max(np.abs(audio_chunk)) < 100:  # Near silence
                self.logger.warning("Very low audio level detected")
            elif np.max(np.abs(audio_chunk)) > 32000:  # Near clipping
                self.logger.warning("Audio level near clipping")
                
            float_chunk = (audio_chunk / 32768.0).astype(np.float32)
            self.frames.append(float_chunk)
            
            return (in_data, pyaudio.paContinue)
            
        except Exception as e:
            self.logger.error(f"Error in audio callback: {e}")
            return (None, pyaudio.paComplete)

    def _check_for_enter(self):
        """Check if Enter key was pressed"""
        if select.select([sys.stdin], [], [], 0.1)[0]:
            sys.stdin.readline()
            return True
        return False
        
    def record(self, duration=None):
        """Record audio until Enter is pressed or duration is reached"""
        try:
            if not self.stream:
                self.start_stream()
            
            if not self.stream:
                self.logger.error("Failed to initialize audio stream")
                return None, None
                
            self.logger.info("\nRecording... Press Enter to stop.")
            
            try:
                if duration:
                    end_time = time.time() + duration
                    while time.time() < end_time and not self.stop_recording.is_set():
                        data = self.stream.read(self.config['chunk_size'], exception_on_overflow=False)
                        audio_chunk = np.frombuffer(data, dtype=np.int16)
                        float_chunk = (audio_chunk / 32768.0).astype(np.float32)
                        self.frames.append(float_chunk)
                        if self._check_for_enter():
                            break
                else:
                    while not self.stop_recording.is_set():
                        data = self.stream.read(self.config['chunk_size'], exception_on_overflow=False)
                        audio_chunk = np.frombuffer(data, dtype=np.int16)
                        float_chunk = (audio_chunk / 32768.0).astype(np.float32)
                        self.frames.append(float_chunk)
                        if self._check_for_enter():
                            break
                            
                # Convert to tensor and save debug recording
                if self.frames:
                    audio_data = torch.from_numpy(np.concatenate(self.frames))
                    
                    # Save debug recording
                    self._save_debug_recording(audio_data)
                    
                    # Log audio stats
                    self.logger.info(f"Recording stats:")
                    self.logger.info(f"  Duration: {len(audio_data)/self.config['sample_rate']:.2f}s")
                    self.logger.info(f"  Max amplitude: {audio_data.abs().max():.4f}")
                    self.logger.info(f"  Min amplitude: {audio_data.abs().min():.4f}")
                    self.logger.info(f"  Mean amplitude: {audio_data.abs().mean():.4f}")
                    
                    return audio_data, self.config['sample_rate']
                    
                return None, None
                
            except Exception as e:
                self.logger.error(f"Error recording audio: {e}")
                return None, None
                
        finally:
            self.stop()
            
    def stop(self):
        """Stop recording"""
        self.stop_recording.set()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
    def save_to_file(self, audio_data, filename):
        """Save recorded audio to WAV file"""
        if audio_data is None:
            return False
            
        try:
            # Convert float32 back to int16 for WAV file
            int16_data = (audio_data.numpy() * 32768.0).astype(np.int16)
            
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.config['channels'])
                wf.setsampwidth(self.audio.get_sample_size(eval(self.config['format'])))
                wf.setframerate(self.config['sample_rate'])
                wf.writeframes(int16_data.tobytes())
                
            self.logger.info(f"Successfully saved audio to: {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving audio: {e}")
            return False
            
    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'stream') and self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if hasattr(self, 'audio') and self.audio:
                self.audio.terminate()
        except:
            pass