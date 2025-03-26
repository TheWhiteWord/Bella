"""
Buffered Audio Recorder module with voice activity detection and pre-buffering.
Provides high-quality audio capture with smooth transitions and proper silence handling.
Optimized for PipeWire and general audio input.

This module's sole responsibility is to record good audio and ensure voice detection and silence detection.
The AudioSessionManager handles the session management and gap detection between clips.
"""
import numpy as np
import soundfile as sf
from collections import deque
from datetime import datetime
import sounddevice as sd
import os
import time
import traceback

class BufferedRecorder:
    def __init__(self):
        # Audio settings optimized for PipeWire
        self.hardware_rate = 44100
        self.channels = 1
        self.dtype = np.float32
        self.block_size = 256  # Small block size for low latency
        
        # Pre-buffer and timing settings
        self.pre_buffer_seconds = 2.0
        self.fade_duration = 0.1
        
        # Voice detection settings - Adjusted for better detection
        self.voice_threshold = 0.02  # Increased slightly
        self.initial_voice_threshold = 0.03  # Higher threshold for initial detection
        self.silence_energy_threshold = 0.015  # Increased slightly
        self.min_valid_frames = 8  # Increased for more stable detection
        self.valid_frame_count = 0
        self.last_energies = deque(maxlen=15)  # Increased history
        
        # State management
        self.should_stop = False
        self.is_recording = False
        self.voice_detected = False
        self.last_debug_time = time.time()
        self.debug_interval = 1.0
        
        # Silence detection settings - Adjusted for better gap detection
        self.silence_timeout = 1.5  # Increased to 1.5 seconds
        self.consecutive_silence_frames = 0
        self.silence_frame_threshold = None  # Will be set in initialize_audio_settings
        
        # Buffer settings
        self.max_buffer_size = int(self.hardware_rate * 2)  # 2 seconds of audio
        self.audio_buffer = np.zeros(self.max_buffer_size, dtype=self.dtype)
        self.buffer_index = 0
        self.recording_buffer = []
        
        # Last recording tracking
        self.last_recording = None
        self.debug_mode = False

    def start_recording(self):
        """Start recording with pre-buffer content and smooth transition"""
        if not self.is_recording:
            self.is_recording = True
            self.should_stop = False
            
            # Calculate fade window size (100ms)
            fade_samples = int(0.1 * self.hardware_rate)
            
            # Get pre-buffer content efficiently
            if self.buffer_index == 0:
                pre_buffer = self.audio_buffer.copy()
            else:
                pre_buffer = np.concatenate([
                    self.audio_buffer[self.buffer_index:],
                    self.audio_buffer[:self.buffer_index]
                ])
            
            # Apply fade-in to the last 100ms of pre-buffer for smooth transition
            if len(pre_buffer) > fade_samples:
                fade_in = np.linspace(0.5, 1.0, fade_samples)
                pre_buffer[-fade_samples:] *= fade_in
            
            # Ensure pre-buffer is properly shaped and normalized
            pre_buffer = pre_buffer.reshape(-1, 1)
            if len(pre_buffer) > 0:
                # Remove DC offset from pre-buffer
                pre_buffer = pre_buffer - np.mean(pre_buffer)
                # Normalize pre-buffer
                max_val = np.max(np.abs(pre_buffer))
                if max_val > 0:
                    pre_buffer = pre_buffer / max_val * 0.9
            
            self.recording_buffer = [pre_buffer]
            print("\nRecording started (with pre-buffer)...")

    def stop_recording(self):
        """Stop recording and save the audio with proper format for PipeWire"""
        if not self.is_recording:
            return None
            
        try:
            # Only process if we have data
            if self.recording_buffer:
                # Combine all audio chunks
                audio = np.concatenate(self.recording_buffer)
                
                # Save the audio file
                output_dir = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "Testing",
                    "audio files"
                )
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(output_dir, f"recording_{timestamp}.wav")
                
                duration = len(audio) / self.hardware_rate
                if (duration < 0.1):
                    print("Recording too short.")
                    self.reset_state()
                    return None
                
                # Save as 16-bit PCM WAV file
                audio_int16 = (audio * 32767).astype(np.int16)
                sf.write(filename, audio_int16, self.hardware_rate, subtype='PCM_16')
                
                print(f"\nRecorded {duration:.1f}s of audio.")
                print(f"Audio saved to: {filename}")
                
                # Store the last recording path before reset
                self.last_recording = filename
                
                # Reset recording state but preserve last_recording
                self.reset_state()
                
                return filename
            else:
                print("No audio recorded.")
                self.reset_state()
                return None
                
        except Exception as e:
            print(f"\nError saving audio: {e}")
            traceback.print_exc()
            self.reset_state()
            return None

    def reset_state(self):
        """Reset all state variables to their initial values"""
        self.is_recording = False
        self.voice_detected = False
        self.valid_frame_count = 0
        self.consecutive_silence_frames = 0
        self.should_stop = False
        self.recording_buffer = []
        # Don't reset last_recording as it needs to be processed first

    def _update_circular_buffer(self, new_data):
        """Update circular buffer with new audio data."""
        new_size = len(new_data)
        space_left = self.max_buffer_size - self.buffer_index
        
        if new_size <= space_left:
            # Simple case: enough space at the end
            self.audio_buffer[self.buffer_index:self.buffer_index + new_size] = new_data
            self.buffer_index += new_size
        else:
            # Split the new data between end and beginning of buffer
            first_part = space_left
            second_part = new_size - space_left
            
            # Fill to the end of buffer
            self.audio_buffer[self.buffer_index:] = new_data[:first_part]
            # Wrap around to start of buffer
            self.audio_buffer[:second_part] = new_data[first_part:]
            self.buffer_index = second_part
        
        if self.buffer_index >= self.max_buffer_size:
            self.buffer_index = 0

    def audio_callback(self, indata, frames, time_info, status):
        """Process incoming audio data with voice detection"""
        try:
            if self.should_stop:
                return

            # Get audio data and calculate energy
            audio_data = indata.flatten().astype(np.float32)
            frame_energy = float(np.sqrt(np.mean(audio_data**2)))
            
            # Update energy history
            self.last_energies.append(frame_energy)
            
            # Print debug info at regular intervals
            current_time = time.time()
            if current_time - self.last_debug_time >= self.debug_interval:
                energy_level = "█" * int(min(frame_energy * 100, 20))
                threshold = self.initial_voice_threshold if not self.is_recording else self.voice_threshold
                voice_status = "VOICE" if frame_energy > threshold else "silence"
                print(f"\rEnergy: {energy_level} ({frame_energy:.4f}) | Status: {voice_status} | {'🎤' if self.is_recording else '·'}", end='')
                self.last_debug_time = current_time
            
            # Voice activity detection with two-stage thresholds
            if not self.is_recording:
                # Use higher threshold for initial detection
                if frame_energy > self.initial_voice_threshold:
                    self.valid_frame_count += 1
                    if self.valid_frame_count >= self.min_valid_frames:
                        print("\nVoice detected - Starting recording...")
                        self.voice_detected = True
                        self.start_recording()
                else:
                    self.valid_frame_count = max(0, self.valid_frame_count - 1)
            elif self.is_recording:
                # Use lower threshold to maintain recording
                if frame_energy < self.silence_energy_threshold:
                    self.consecutive_silence_frames += 1
                    if self.consecutive_silence_frames >= self.silence_frame_threshold:
                        print(f"\nSilence detected for {self.silence_timeout}s")
                        self.stop_recording()
                        return
                else:
                    # Only decrease silence counter if we have strong energy
                    if frame_energy > self.voice_threshold:
                        self.consecutive_silence_frames = max(0, self.consecutive_silence_frames - 2)
                    else:
                        # If energy is between silence and voice threshold, decrease slower
                        self.consecutive_silence_frames = max(0, self.consecutive_silence_frames - 1)
            
            # Update circular buffer
            self._update_circular_buffer(audio_data)
            
            # If recording, store audio
            if self.is_recording:
                self.recording_buffer.append(audio_data.reshape(-1, 1))
                
        except Exception as e:
            print(f"\nError in callback: {e}")
            traceback.print_exc()

    def initialize_audio_settings(self, device_info):
        """Initialize audio settings based on device info"""
        self.hardware_rate = int(device_info['default_samplerate'])
        self.max_buffer_size = int(self.pre_buffer_seconds * self.hardware_rate)
        self.audio_buffer = np.zeros(self.max_buffer_size, dtype=self.dtype)
        self.frames_per_second = self.hardware_rate / self.block_size
        self.silence_frame_threshold = int(self.silence_timeout * self.frames_per_second)
        
        if self.debug_mode:
            print(f"\nAudio Settings:")
            print(f"Sample Rate: {self.hardware_rate}Hz")
            print(f"Block Size: {self.block_size} samples")
            print(f"Pre-buffer: {self.pre_buffer_seconds}s")
            print(f"Voice Threshold: {self.voice_threshold}")
            print(f"Frames per second: {self.frames_per_second}")
            print(f"Silence frame threshold: {self.silence_frame_threshold}")
            print(f"Fade duration: {self.fade_duration}s\n")

async def create_audio_stream(recorder, device_id=None):
    """Create and configure the audio input stream"""
    # List available devices
    devices = sd.query_devices()
    
    # If no device specified, find a suitable one
    if device_id is None:
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                device_id = i
                break
    
    if device_id is None:
        raise RuntimeError("No input devices found!")
    
    device_info = devices[device_id]
    recorder.initialize_audio_settings(device_info)
    
    # Create the input stream with PipeWire optimized settings
    stream = sd.InputStream(
        device=device_id,
        samplerate=recorder.hardware_rate,
        channels=1,  # Force mono input
        dtype=recorder.dtype,
        callback=recorder.audio_callback,
        blocksize=recorder.block_size,
        latency='low'  # Use low latency mode
    )
    
    return stream

async def record_audio(output_dir=None, device_id=None, debug=False):
    """Record audio using buffered recorder with VAD processing"""
    recorder = BufferedRecorder()
    recorder.debug_mode = debug
    
    try:
        stream = await create_audio_stream(recorder, device_id)
        
        if debug:
            print("\n🎤 Ready! Speak loudly and clearly...")
            print("Detection indicators: ", end='', flush=True)
        
        with stream:
            while not recorder.should_stop:
                await asyncio.sleep(0.1)
                
        return recorder.stop_recording()
        
    except Exception as e:
        if debug:
            print(f"\n❌ Error during recording: {e}")
            import traceback
            traceback.print_exc()
        return None

if __name__ == "__main__":
    try:
        asyncio.run(record_audio(debug=True))
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    except Exception as e:
        print(f"\nError: {e}")