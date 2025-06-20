"""
Buffered Audio Recorder module with voice activity detection and pre-buffering.
Provides high-quality audio capture with smooth transitions and proper silence handling.
Uses PipeWire/PulseAudio for audio input (NOT PortAudio).

This module's sole responsibility is to record good audio and ensure voice detection and silence detection.
The AudioSessionManager handles the session management and gap detection between clips.
"""
import numpy as np
import wave
from collections import deque
from datetime import datetime
import os
import time
import traceback
import asyncio
import subprocess
import tempfile

class BufferedRecorder:
    def __init__(self):
        # Audio settings optimized for PipeWire
        self.sample_rate = 44100
        self.channels = 1
        self.dtype = np.float32
        self.block_size = 4096  # Increased for better chunk processing
        
        # Pre-buffer and timing settings
        self.pre_buffer_seconds = 2.0
        self.fade_duration = 0.1
        
        # Voice detection settings
        self.voice_threshold = 0.12
        self.initial_voice_threshold = 0.12
        self.silence_energy_threshold = 0.03
        self.min_valid_frames = 5
        self.valid_frame_count = 0
        self.last_energies = deque(maxlen=20)
        
        # State management
        self.should_stop = False
        self.is_recording = False
        self.voice_detected = False
        self.last_debug_time = time.time()
        self.debug_interval = 0.5
        
        # Silence detection settings
        self.silence_timeout = 1.0
        self.consecutive_silence_frames = 0
        self.silence_frame_threshold = None  # Will be set in initialize_audio_settings
        
        # Buffer settings
        self.max_buffer_size = int(self.sample_rate * 2)
        self.audio_buffer = np.zeros(self.max_buffer_size, dtype=self.dtype)
        self.buffer_index = 0
        self.recording_buffer = []
        
        # Recording process
        self.recording_process = None
        self.temp_file = None
        
        # Last recording tracking
        self.last_recording = None
        self.debug_mode = False  # Default to False
        self.debug_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "debug",
            "audio files"
        )  # Default debug directory

    def start_recording(self):
        """Start recording with pre-buffer content and smooth transition"""
        if self.is_recording:
            # If already recording, force stop first to ensure clean state
            self.stop_recording()
            
        self.is_recording = True
        self.should_stop = False
        self.voice_detected = False  # Reset voice detection state
        self.recording_buffer = []  # Clear any old data
        self.valid_frame_count = 0
        self.consecutive_silence_frames = 0
        
        # Start parec process for recording
        cmd = [
            'parec',  # PulseAudio recorder
            '--format=float32le',
            '--rate', str(self.sample_rate),
            '--channels', '1',
            '--latency-msec=20',
            '--process-time-msec=20'
        ]
        
        try:
            # Kill any existing recording process first
            if self.recording_process:
                try:
                    self.recording_process.terminate()
                    self.recording_process.wait(timeout=1)
                except:
                    pass
                self.recording_process = None
                
            # Start a fresh recording process
            self.recording_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self.block_size * 4  # Buffer size in bytes
            )
            print("\nRecording started...")
            
            # Start processing thread
            asyncio.create_task(self._process_audio_stream())
            
        except Exception as e:
            print(f"\nError starting recording: {e}")
            self.stop_recording()

    async def _process_audio_stream(self):
        """Process the audio stream from parec"""
        buffer_size = self.block_size * 4  # 4 bytes per float32
        while self.is_recording and not self.should_stop:
            try:
                if self.recording_process:
                    data = self.recording_process.stdout.read(buffer_size)
                    if not data:
                        break
                        
                    # Convert bytes to numpy array
                    audio_chunk = np.frombuffer(data, dtype=np.float32)
                    
                    # Process the audio chunk
                    self._process_audio_chunk(audio_chunk)
                    
                    # If should_stop was set by silence detection, break the loop
                    if self.should_stop:
                        break
                        
                await asyncio.sleep(0.001)
                    
            except Exception as e:
                print(f"\nError processing audio: {e}")
                break
        
        # When we break out of the loop, stop recording and save the file
        if self.voice_detected:
            await asyncio.get_event_loop().run_in_executor(None, self.stop_recording)

    def _process_audio_chunk(self, audio_chunk):
        """Process a chunk of audio data with voice activity detection and silence detection."""
        if len(audio_chunk) == 0:
            return
            
        # Calculate energy
        frame_energy = float(np.sqrt(np.mean(audio_chunk**2)))
        self.last_energies.append(frame_energy)
        
        # Print debug info at regular intervals
        current_time = time.time()
        if current_time - self.last_debug_time >= self.debug_interval:
            energy_level = "█" * int(min(frame_energy * 100, 20))
            threshold = self.initial_voice_threshold if not self.voice_detected else self.voice_threshold
            voice_status = "VOICE" if frame_energy > threshold else "silence"
            print(f"\rEnergy: {energy_level} ({frame_energy:.4f}) | Status: {voice_status} | {'🎤' if self.is_recording else '·'}", end='')
            self.last_debug_time = current_time
        
        # Voice activity detection
        is_voice = frame_energy > (self.initial_voice_threshold if not self.voice_detected else self.voice_threshold)
        
        if is_voice:
            self.valid_frame_count += 1
            self.consecutive_silence_frames = 0
            if self.valid_frame_count >= self.min_valid_frames and not self.voice_detected:
                self.voice_detected = True
                if self.debug_mode:
                    print("\nVoice detected!")
        else:
            self.valid_frame_count = 0
            if self.voice_detected:
                self.consecutive_silence_frames += 1
                
                # Check if silence has lasted long enough to stop
                if self.consecutive_silence_frames >= self.silence_frame_threshold:
                    if self.debug_mode:
                        print("\nSilence detected, stopping recording...")
                    self.should_stop = True
                    # Don't return here, let the audio stream processor handle the stop
        
        # Always store audio if we're recording and either voice was detected or we're in pre-buffer
        if self.is_recording:
            self.recording_buffer.append(audio_chunk)

    def stop_recording(self):
        """Stop recording and save the audio with proper format"""
        if not self.is_recording:
            return None
            
        try:
            # Stop the recording process first
            if self.recording_process:
                try:
                    self.recording_process.terminate()
                    self.recording_process.wait(timeout=1)
                except:
                    pass
                self.recording_process = None
            
            # Only process if we have data and voice was detected
            if self.recording_buffer and self.voice_detected:
                # Combine all audio chunks
                audio = np.concatenate(self.recording_buffer)
                
                duration = len(audio) / self.sample_rate
                if duration < 0.1:  # Skip very short recordings
                    print("Recording too short.")
                    self.reset_state()
                    return None
                
                # Normalize audio
                audio = audio - np.mean(audio)  # Remove DC offset
                max_val = np.max(np.abs(audio))
                if max_val > 0:
                    audio = audio / max_val * 0.9
                
                # Convert to 16-bit PCM
                audio_int16 = (audio * 32767).astype(np.int16)
                
                # Create temporary file for processing
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                filename = temp_file.name
                self.temp_file = temp_file  # Store reference for cleanup
                
                # If debug mode is enabled, also save a debug copy
                if self.debug_mode:
                    os.makedirs(self.debug_dir, exist_ok=True)
                    debug_filename = os.path.join(
                        self.debug_dir,
                        f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                    )
                    # Save debug copy
                    with wave.open(debug_filename, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(self.sample_rate)
                        wf.writeframes(audio_int16.tobytes())
                    if self.debug_mode:
                        print(f"\nSaved debug file: {debug_filename}")
                
                # Save the temporary file
                with wave.open(filename, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(audio_int16.tobytes())
                
                if self.debug_mode:
                    print(f"\nRecorded {duration:.1f}s of audio.")
                    print(f"Temp file created at: {filename}")
                
                # Store the last recording path before reset
                self.last_recording = filename
                
                # Reset recording state but preserve last_recording and temp_file
                self.reset_state(preserve_temp=True)
                
                return filename
            else:
                print("\nNo valid audio recorded.")
                self.reset_state()
                return None
                
        except Exception as e:
            print(f"\nError saving audio: {e}")
            traceback.print_exc()
            self.reset_state()
            return None

    def reset_state(self, preserve_temp=False):
        """Reset all state variables to their initial values"""
        self.is_recording = False
        self.voice_detected = False
        self.valid_frame_count = 0
        self.consecutive_silence_frames = 0
        self.should_stop = False
        self.recording_buffer = []
        
        # Reset energy detection state to make sure we're ready for new input
        self.last_energies = deque(maxlen=20)
        
        # Terminate any existing recording process
        if self.recording_process:
            try:
                self.recording_process.terminate()
                self.recording_process.wait(timeout=1)
            except:
                pass
            self.recording_process = None
        
        # Clean up temp files if needed
        if not preserve_temp and self.temp_file:
            try:
                os.unlink(self.temp_file.name)
            except:
                pass
            self.temp_file = None
            self.last_recording = None

    def initialize_audio_settings(self, device_info=None):
        """Initialize audio settings"""
        self.frames_per_second = self.sample_rate / self.block_size
        self.silence_frame_threshold = int(self.silence_timeout * self.frames_per_second)
        
        if self.debug_mode:
            print(f"\nAudio Settings:")
            print(f"Sample Rate: {self.sample_rate}Hz")
            print(f"Block Size: {self.block_size} samples")
            print(f"Pre-buffer: {self.pre_buffer_seconds}s")
            print(f"Voice Threshold: {self.voice_threshold}")
            print(f"Frames per second: {self.frames_per_second}")
            print(f"Silence frame threshold: {self.silence_frame_threshold}")
            print(f"Fade duration: {self.fade_duration}s\n")

async def create_audio_stream(recorder, device_id=None):
    """Create and configure the audio input stream using PipeWire/PulseAudio"""
    try:
        # Get default source if no device specified
        if device_id is None:
            result = subprocess.run(
                ['pactl', 'get-default-source'],
                capture_output=True,
                text=True,
                check=True
            )
            device_id = result.stdout.strip()
        
        # Initialize recorder settings
        recorder.initialize_audio_settings()
        
        # Start recording immediately
        recorder.start_recording()
        
        return None  # We don't need to return a stream object anymore
        
    except Exception as e:
        print(f"Error creating audio stream: {e}")
        raise

async def record_audio(output_dir=None, device_id=None, debug=False):
    """Record audio using buffered recorder with VAD processing"""
    recorder = BufferedRecorder()
    recorder.debug_mode = debug
    
    try:
        await create_audio_stream(recorder, device_id)
        
        if debug:
            print("\n🎤 Ready! Speak loudly and clearly...")
            print("Detection indicators: ", end='', flush=True)
        
        while not recorder.should_stop:
            await asyncio.sleep(0.1)
            
        return recorder.stop_recording()
        
    except Exception as e:
        if debug:
            print(f"\n❌ Error during recording: {e}")
            traceback.print_exc()
        return None

if __name__ == "__main__":
    try:
        asyncio.run(record_audio(debug=True))
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    except Exception as e:
        print(f"\nError: {e}")