"""
Buffered Audio Recorder module with voice activity detection and pre-buffering.
Provides high-quality audio capture with smooth transitions and proper silence handling.
Optimized for PipeWire and general audio input.
This module sole responsability is that to record good audio, and ensure thet voice recognition is achieved, and silence detection.
This module does not handles the audio session management, which is done by the AudioSessionManager class.
This module should have a pause and resume function, but it is not used in the current implementation.
This module is used by the main.py file, which handles the audio session management and the interaction with the LLM.
"""
import queue
import webrtcvad
import numpy as np
import soundfile as sf
import threading
import sys
from collections import deque
from datetime import datetime
from scipy import signal
import sounddevice as sd
import os
import asyncio
import time

class BufferedRecorder:
    def __init__(self):
        # Audio settings optimized for PipeWire
        self.hardware_rate = 44100
        self.channels = 1
        self.dtype = np.float32
        self.block_size = 256  # Small block size for low latency
        
        # Pre-buffer and timing settings
        self.pre_buffer_seconds = 2.0
        self.warm_up_frames = 50
        self.is_warmed_up = False
        self.fade_duration = 0.1
        self.debounce_time = 0.2
        
        # Voice detection settings - Adjusted thresholds
        self.voice_threshold = 0.02  # Increased to avoid false triggers
        self.silence_energy_threshold = 0.012  # Adjusted for better silence detection
        self.spike_threshold = 0.8
        self.voice_sustain_frames = 15
        self.min_valid_frames = 3  # Increased to ensure valid voice activity
        self.valid_frame_count = 0
        self.last_energies = deque(maxlen=10)
        
        # State management
        self.should_stop = False
        self.is_recording = False
        self.voice_detected = False
        
        # Silence detection settings
        self.silence_timeout = 1.0  # Time in seconds to wait before stopping
        self.consecutive_silence_frames = 0
        self.silence_frame_threshold = int(self.silence_timeout * (self.hardware_rate / self.block_size))
        self.min_silence_frames = 20  # Increased for more stable silence detection
        
        # VAD settings
        self.vad = webrtcvad.Vad(2)
        self.vad_rate = 16000
        self.vad_frame_length = int(self.vad_rate * 0.03)  # 30ms frames
        
        # Buffer settings
        self.max_buffer_size = int(self.hardware_rate * 2)  # 2 seconds of audio
        self.audio_buffer = np.zeros(self.max_buffer_size, dtype=self.dtype)
        self.buffer_index = 0
        
        # Processing and recording state
        self.process_queue = queue.Queue()
        self.processing_active = True
        self.recording_buffer = []
        self.silence_frames = 0
        self.current_frame = 0
        self.frame_count = 0
        
        # Debug settings
        self.debug_mode = True
        self.debug_info = {
            'max_energy': 0.0,
            'min_energy': float('inf'),
            'frame_count': 0,
            'voice_detections': 0,
            'recording_frames': 0,
            'last_energy': 0.0,
            'overflow_count': 0,
            'spikes_rejected': 0,
            'silence_frames': 0
        }
        
        # Silence detection
        self.silence_threshold = 0.02
        self.silence_timeout = 1.5
        self.consecutive_silence_frames = 0
        self.min_speech_frames = 3
        
        # Time between triggers
        self.last_trigger_time = 0
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_audio_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Resampling for VAD
        self.resampler = signal.resample_poly
        
    def initialize_audio_settings(self, device_info):
        """Initialize audio settings based on device info"""
        self.hardware_rate = int(device_info['default_samplerate'])
        self.max_buffer_size = int(self.pre_buffer_seconds * self.hardware_rate)
        self.audio_buffer = np.zeros(self.max_buffer_size, dtype=self.dtype)
        self.frames_per_second = self.hardware_rate / self.block_size
        self.silence_frame_threshold = int(self.silence_timeout * self.frames_per_second)
        
        # Initialize fade windows
        fade_samples = int(0.1 * self.hardware_rate)  # 100ms fade
        self.fade_in = np.linspace(0.5, 1.0, fade_samples)
        self.fade_out = np.linspace(1.0, 0.5, fade_samples)
        
        if self.debug_mode:
            print(f"\nAudio Settings:")
            print(f"Sample Rate: {self.hardware_rate}Hz")
            print(f"Block Size: {self.block_size} samples")
            print(f"Pre-buffer: {self.pre_buffer_seconds}s")
            print(f"Voice Threshold: {self.voice_threshold}")
            print(f"Frames per second: {self.frames_per_second}")
            print(f"Silence frame threshold: {self.silence_frame_threshold}")
            print(f"Fade duration: {self.fade_duration}s")

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

    def _resample_for_vad(self, audio_data):
        """Resample audio data to 16kHz for WebRTC VAD."""
        target_rate = 16000  # WebRTC VAD requires 16kHz
        
        if self.hardware_rate != target_rate:
            # Calculate resampling ratio
            ratio = target_rate / self.hardware_rate
            output_length = int(len(audio_data) * ratio)
            
            # Resample using scipy's resample function
            resampled = signal.resample(audio_data, output_length)
            
            # Convert to int16 format required by WebRTC VAD
            return (resampled * 32767).astype(np.int16)
        else:
            return (audio_data * 32767).astype(np.int16)

    def start_processing(self):
        """Start the audio processing thread"""
        self.processing_active = True
        self.processor_thread = threading.Thread(target=self._process_audio_frames)
        self.processor_thread.daemon = True  # Thread will exit when main program exits
        self.processor_thread.start()

    def stop_processing(self):
        """Stop the audio processing thread"""
        self.processing_active = False
        if self.processor_thread:
            self.processor_thread.join()

    def _process_audio_frames(self):
        """Process audio frames with enhanced VAD"""
        while self.processing_active:
            try:
                frames = self.process_queue.get(timeout=0.1)
                if frames is None:
                    continue
                    
                # Calculate buffer duration
                buffer_duration = len(frames) / self.hardware_rate
                
                if (buffer_duration >= 0.03):  # Only process chunks of sufficient duration
                    # Resample to VAD rate
                    vad_data = self.resampler(frames, self.vad_rate, self.hardware_rate)
                    vad_data = (vad_data * 32767).astype(np.int16)
                    
                    # Process VAD frames
                    for i in range(0, len(vad_data), self.vad_frame_length):
                        frame = vad_data[i:i + self.vad_frame_length]
                        if len(frame) == self.vad_frame_length:
                            try:
                                is_speech = self.vad.is_speech(frame.tobytes(), self.vad_rate)
                                if is_speech:
                                    self.voice_detected = True
                                    self.voice_frames_counter = self.voice_sustain_frames
                                    if not self.is_recording:
                                        self._start_recording()
                                elif self.voice_frames_counter > 0:
                                    self.voice_frames_counter -= 1
                                else:
                                    self.voice_detected = False
                                    if self.is_recording:
                                        self._stop_recording()
                            except Exception as e:
                                print(f"VAD error: {e}", file=sys.stderr)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}", file=sys.stderr)
                continue
                
    def audio_callback(self, indata, frames, time_info, status):
        """Lightweight audio callback with improved voice detection"""
        try:
            # Increment frame counter
            self.current_frame += 1
            self.frame_count += 1
            self.debug_info['frame_count'] += 1
            
            # Handle warm-up period
            if not self.is_warmed_up:
                if self.current_frame >= self.warm_up_frames:
                    self.is_warmed_up = True
                    print("\nAudio system warmed up, ready for recording...")
                return
            
            # Get audio data and calculate energy
            audio_data = indata.flatten().astype(np.float32)
            frame_energy = float(np.sqrt(np.mean(audio_data**2)))
            
            # Update energy history and debug stats
            self.last_energies.append(frame_energy)
            self.debug_info['max_energy'] = max(self.debug_info['max_energy'], frame_energy)
            self.debug_info['min_energy'] = min(self.debug_info['min_energy'], frame_energy)
            self.debug_info['last_energy'] = frame_energy
            
            # Voice activity detection
            if frame_energy > self.voice_threshold:
                self.consecutive_silence_frames = 0
                self.valid_frame_count += 1
                
                if self.valid_frame_count >= self.min_valid_frames:
                    if not self.is_recording:
                        self.voice_detected = True
                        self.start_recording()
                        print("\nVoice detected - Starting recording...")
            else:
                self.valid_frame_count = max(0, self.valid_frame_count - 1)
                
                # Silence detection
                if frame_energy < self.silence_energy_threshold:
                    self.consecutive_silence_frames += 1
                    if self.consecutive_silence_frames >= self.silence_frame_threshold and self.is_recording:
                        print(f"\nSilence detected for {self.silence_timeout}s - Stopping recording...")
                        self.should_stop = True
                        return
                else:
                    self.consecutive_silence_frames = max(0, self.consecutive_silence_frames - 2)
            
            # Update circular buffer
            self._update_circular_buffer(audio_data)
            
            # If recording, store audio
            if self.is_recording:
                self.recording_buffer.append(audio_data.reshape(-1, 1))
                self.debug_info['recording_frames'] += 1
            
            # Visual feedback - but not too often
            if self.debug_info['frame_count'] % 2 == 0:
                if self.is_recording:
                    sys.stdout.write('🎤')
                else:
                    sys.stdout.write('.')
                sys.stdout.flush()
                
        except Exception as e:
            print(f"\nError in callback: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    def _start_recording(self):
        """Start recording with pre-buffer content"""
        self.is_recording = True
        # Include pre-buffer content
        self.recording_buffer = list(self.audio_buffer)
        print("\nRecording started...")
        
    def _stop_recording(self):
        """Stop recording and save the audio"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        print("\nRecording stopped...")
        
        if not self.recording_buffer:
            print("No audio recorded")
            return
            
        # Process and save the recording
        audio_data = np.concatenate(self.recording_buffer)
        
        # Normalize audio
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val * 0.9
            
        # Save the audio file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        output_path = f"Testing/audio files/{filename}"
        
        sf.write(output_path, audio_data, self.hardware_rate)
        print(f"Audio saved to: {output_path}")
        return output_path

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
            
        self.is_recording = False
        
        if not self.recording_buffer:
            print("No audio recorded.")
            return None
            
        # Print final debug statistics
        print("\nFinal Recording Statistics:")
        print(f"Total frames processed: {self.debug_info['frame_count']}")
        print(f"Voice detections: {self.debug_info['voice_detections']}")
        print(f"Recording frames: {self.debug_info['recording_frames']}")
        print(f"Max energy level: {self.debug_info['max_energy']:.6f}")
        print(f"Min energy level: {self.debug_info['min_energy']:.6f}")
        print(f"Last energy level: {self.debug_info['last_energy']:.6f}")
        print(f"Voice threshold: {self.voice_threshold}")
        print(f"Overflow count: {self.debug_info.get('overflow_count', 0)}")
        
        try:
            # Combine all audio chunks
            audio = np.concatenate(self.recording_buffer)
            
            # Print audio statistics
            print("\nAudio Statistics:")
            print(f"Audio shape: {audio.shape}")
            print(f"Raw data range: [{np.min(audio):.6f}, {np.max(audio):.6f}]")
            print(f"Mean value: {np.mean(audio):.6f}")
            print(f"RMS level: {np.sqrt(np.mean(audio**2)):.6f}")
            
            # Normalize audio
            if len(audio) > 0:
                # Remove DC offset
                audio = audio - np.mean(audio)
                
                # Normalize to [-0.9, 0.9] range
                max_val = np.max(np.abs(audio))
                if max_val > 0:
                    audio = audio / max_val * 0.9
                
                print(f"Normalized data range: [{np.min(audio):.6f}, {np.max(audio):.6f}]")
            
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
                return None
            
            # Save as 16-bit PCM WAV file (PipeWire compatible)
            audio_int16 = (audio * 32767).astype(np.int16)
            sf.write(filename, audio_int16, self.hardware_rate, subtype='PCM_16')
            
            print(f"\nRecorded {duration:.1f}s of audio.")
            print(f"Audio saved to: {filename}")
            return filename
            
        except Exception as e:
            print(f"\nError saving audio: {e}")
            import traceback
            traceback.print_exc()
            return None

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
            # Start VAD processing
            recorder.start_processing()
            
            while not recorder.should_stop:
                await asyncio.sleep(0.1)
                
            # Stop VAD processing
            recorder.stop_processing()
            
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