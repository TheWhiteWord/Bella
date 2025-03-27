import sounddevice as sd
import numpy as np
import soundfile as sf
import os
import asyncio
import sys
from faster_whisper import WhisperModel
import time
from datetime import datetime
from collections import deque
import threading
import queue
import webrtcvad
from scipy import signal

class BufferedRecorder:
    def __init__(self):
        # Audio settings optimized for PipeWire
        self.hardware_rate = None  # Will be set based on device
        self.channels = 1
        self.dtype = np.float32
        self.block_size = 256  # Smaller blocks for PipeWire's low-latency design
        
        # VAD settings
        self.vad_rate = 16000  # Required sample rate for WebRTC VAD
        self.vad_frame_duration = 0.03  # 30ms frames for VAD
        self.vad_frame_length = int(self.vad_rate * self.vad_frame_duration)
        self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3
        
        # Warm-up settings
        self.warm_up_frames = 40  # Increased warm-up (~0.25s at 172 fps)
        self.current_frame = 0
        self.is_warmed_up = False
        
        # Spike rejection
        self.spike_threshold = 0.8  # Threshold for audio spikes
        self.min_valid_frames = 10  # Require consecutive frames above threshold
        self.valid_frame_count = 0
        self.last_energies = deque(maxlen=10)  # History for energy levels
        
        # Processing queue
        self.process_queue = queue.Queue(maxsize=100)
        self.processor_thread = None
        self.processing_active = False
        self.drop_count = 0
        self.frame_count = 0
        
        # Buffer settings
        self.pre_buffer_seconds = 3  # Increased back to 3s for better quality
        self.max_buffer_size = None  # Will be set once we know the sample rate
        self.audio_buffer = None  # Will be initialized once we know the sample rate
        self.buffer_index = 0
        self.recording_buffer = []
        
        # Transition settings
        self.fade_duration = 0.1  # 100ms fade-in
        self.fade_samples = None  # Will be set once we know the sample rate
        
        # Voice detection settings
        self.voice_threshold = 0.04  # Voice detection threshold
        self.voice_sustain_frames = 30  # Sustain voice detection longer
        self.voice_frames_counter = 0
        
        # Debounce settings
        self.debounce_time = 0.3  # Time between triggers
        self.last_trigger_time = 0
        
        # Silence detection
        self.silence_timeout = 1.5  # Silence timeout
        self.silence_frames = 0
        self.frames_per_second = None  # Will be set once we know the sample rate
        self.silence_frame_threshold = None  # Will be set once we know the sample rate
        self.consecutive_silence_frames = 0
        self.silence_energy_threshold = 0.02  # Energy threshold for silence
        
        # State variables
        self.is_recording = False
        self.voice_detected = False
        self.should_stop = False
        self.debug_mode = True
        
        # Debug information
        self.debug_info = {
            'max_energy': 0,
            'min_energy': float('inf'),
            'frame_count': 0,
            'voice_detections': 0,
            'recording_frames': 0,
            'last_energy': 0,
            'overflow_count': 0,
            'spikes_rejected': 0,
            'silence_frames': 0
        }

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
        
        # Adjust buffer settings for PipeWire 1.0.3
        self.block_size = 128  # Reduced block size for lower latency
        self.pre_buffer_frames = int(0.5 * self.hardware_rate)  # 500ms pre-buffer
        self.stream_latency = 'low'  # Request low latency mode
        
        if self.debug_mode:
            print(f"\nAudio Settings:")
            print(f"Sample Rate: {self.hardware_rate}Hz")
            print(f"Block Size: {self.block_size} samples")
            print(f"Pre-buffer: {self.pre_buffer_seconds}s")
            print(f"Voice Threshold: {self.voice_threshold}")
            print(f"Frames per second: {self.frames_per_second}")
            print(f"Silence frame threshold: {self.silence_frame_threshold}")
            print(f"Fade duration: {self.fade_duration}s")
            print(f"Stream latency mode: {self.stream_latency}")

    def _update_circular_buffer(self, new_data):
        """Efficiently update circular buffer using numpy operations"""
        new_size = len(new_data)
        space_left = self.max_buffer_size - self.buffer_index
        
        if new_size <= space_left:
            self.audio_buffer[self.buffer_index:self.buffer_index + new_size] = new_data
            self.buffer_index += new_size
        else:
            # Split the new data across the end and start of buffer
            first_part = space_left
            second_part = new_size - space_left
            
            self.audio_buffer[self.buffer_index:] = new_data[:first_part]
            self.audio_buffer[:second_part] = new_data[first_part:]
            self.buffer_index = second_part
            
        if self.buffer_index >= self.max_buffer_size:
            self.buffer_index = 0

    def _resample_for_vad(self, audio_data):
        """Resample audio to 16kHz for VAD processing"""
        resampled = signal.resample_poly(audio_data, 
                                       self.vad_rate, 
                                       self.hardware_rate)
        # Convert to int16 for VAD
        return (resampled * 32767).astype(np.int16)

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
        """Process audio frames in a separate thread"""
        audio_buffer = []  # Buffer for resampling
        resample_ratio = self.vad_rate / self.hardware_rate
        
        while self.processing_active:
            try:
                # Get frame from queue with timeout
                frame_data = self.process_queue.get(timeout=0.1)
                audio_buffer.append(frame_data)
                
                # Process only when we have enough data (120ms worth)
                buffer_duration = len(audio_buffer) * self.block_size / self.hardware_rate
                if buffer_duration >= 0.12:  # 120ms
                    # Combine audio and resample efficiently
                    audio_chunk = np.concatenate(audio_buffer)
                    # More efficient resampling using resample_poly
                    num_samples = int(len(audio_chunk) * resample_ratio)
                    vad_data = signal.resample_poly(audio_chunk, self.vad_rate, self.hardware_rate)
                    vad_data = (vad_data * 32767).astype(np.int16)
                    
                    # Process each VAD frame (30ms chunks)
                    for i in range(0, len(vad_data), self.vad_frame_length):
                        frame = vad_data[i:i + self.vad_frame_length]
                        if len(frame) == self.vad_frame_length:
                            try:
                                is_speech = self.vad.is_speech(frame.tobytes(), self.vad_rate)
                                if is_speech:
                                    self.voice_frames_counter = 15
                                    if not self.voice_detected:
                                        self.voice_detected = True
                                        self.start_recording()
                                    self.silence_frames = 0
                                elif self.voice_frames_counter > 0:
                                    self.voice_frames_counter -= 1
                                else:
                                    self.voice_detected = False
                                    if self.is_recording:
                                        self.silence_frames += 1
                                        if self.silence_frames >= self.silence_frame_threshold:
                                            print("\nSilence detected for {:.2f} seconds...".format(
                                                self.silence_timeout))
                                            self.should_stop = True
                                            return
                            except Exception as e:
                                print(f"\nVAD error: {e}", file=sys.stderr)
                    
                    # Keep only the most recent frame in case it's incomplete
                    audio_buffer = [audio_buffer[-1]] if audio_buffer else []
                
                # Visual feedback - but not too often
                if len(audio_buffer) % 2 == 0:  # Only show every other frame
                    sys.stdout.write('🎤' if self.voice_detected else '.')
                    sys.stdout.flush()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"\nError in processing thread: {e}", file=sys.stderr)

    def audio_callback(self, indata, frames, time_info, status):
        """Lightweight audio callback with warm-up and spike rejection"""
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
            
            # Spike detection
            if frame_energy > self.spike_threshold:
                is_spike = True
                # Check if this is part of consistent voice activity
                if len(self.last_energies) >= 3:
                    prev_energies = list(self.last_energies)[-3:]
                    if all(e > self.voice_threshold for e in prev_energies):
                        is_spike = False
                
                if is_spike:
                    self.debug_info['spikes_rejected'] += 1
                    if self.debug_mode and self.debug_info['spikes_rejected'] % 10 == 0:
                        print(f"\nRejected spike: {frame_energy:.3f}", file=sys.stderr)
                    return
            
            # Voice activity detection with improved silence handling
            current_time = time.time()
            if frame_energy > self.voice_threshold:
                # Reset silence counter when voice is detected
                self.consecutive_silence_frames = 0
                self.valid_frame_count += 1
                
                if self.valid_frame_count >= self.min_valid_frames:
                    if current_time - self.last_trigger_time > self.debounce_time:
                        self.debug_info['voice_detections'] += 1
                        self.voice_frames_counter = self.voice_sustain_frames
                        if not self.voice_detected:
                            self.voice_detected = True
                            self.start_recording()
                        self.last_trigger_time = current_time
            else:
                self.valid_frame_count = 0
                # Track silence when energy is below threshold
                if frame_energy < self.silence_energy_threshold:
                    self.consecutive_silence_frames += 1
                    if self.consecutive_silence_frames >= self.silence_frame_threshold:
                        if self.is_recording:
                            print(f"\nSilence detected for {self.silence_timeout:.1f}s")
                            self.should_stop = True
                            return
                else:
                    # Reset silence counter if energy is between voice and silence thresholds
                    self.consecutive_silence_frames = 0
            
            # Update circular buffer
            self._update_circular_buffer(audio_data)
            
            # If recording, store reference and update debug info
            if self.is_recording:
                self.recording_buffer.append(audio_data.reshape(-1, 1))
                self.debug_info['recording_frames'] += 1
            
            # Visual feedback - but not too often
            if self.debug_info['frame_count'] % 2 == 0:
                sys.stdout.write('🎤' if self.voice_detected else '.')
                sys.stdout.flush()
            
            # Print debug info periodically
            if self.debug_mode and self.debug_info['frame_count'] % 50 == 0:
                print(f"\nDebug Info:", file=sys.stderr)
                print(f"Frame Energy: {frame_energy:.6f}", file=sys.stderr)
                print(f"Spikes Rejected: {self.debug_info['spikes_rejected']}", file=sys.stderr)
                print(f"Voice Detections: {self.debug_info['voice_detections']}", file=sys.stderr)
                print(f"Recording Frames: {self.debug_info['recording_frames']}", file=sys.stderr)
                print(f"Silence Frames: {self.consecutive_silence_frames}", file=sys.stderr)
                
        except Exception as e:
            print(f"\nError in callback: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

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
            output_dir = os.path.join(os.path.dirname(__file__), "recorded_audio")
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_dir, f"buffered_recording_{timestamp}.wav")
            
            duration = len(audio) / self.hardware_rate
            if duration < 0.1:
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

async def record_audio():
    """Record audio using buffered recorder with PipeWire optimization"""
    recorder = BufferedRecorder()
    
    # List available devices and their capabilities
    print("\nAvailable audio devices:")
    devices = sd.query_devices()
    print(devices)
    
    # Find input devices with channels > 0
    input_devices = []
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"\nInput Device {i}:")
            print(f"  Name: {dev['name']}")
            print(f"  Channels: {dev['max_input_channels']}")
            print(f"  Sample Rate: {dev['default_samplerate']}")
            input_devices.append(i)
    
    if not input_devices:
        print("No input devices found!")
        return None
    
    # Try to find PipeWire devices
    device_id = None
    for i in input_devices:
        name = devices[i]['name'].lower()
        if 'pipewire' in name or 'pulse' in name:
            device_id = i
            break
    
    # If no PipeWire device found, use first input device
    if device_id is None:
        device_id = input_devices[0]
    
    device_info = devices[device_id]
    print(f"\nUsing input device {device_id}: {device_info['name']}")
    
    # Initialize audio settings
    recorder.initialize_audio_settings(device_info)
    
    # Create and start the input stream with PipeWire optimized settings
    stream = sd.InputStream(
        device=device_id,
        samplerate=recorder.hardware_rate,
        channels=1,  # Force mono input
        dtype=recorder.dtype,
        callback=recorder.audio_callback,
        blocksize=256,  # Small block size for PipeWire
        latency='low'  # Use low latency mode
    )
    
    print("\n🎤 Ready! Speak loudly and clearly. You may need to speak up...")
    print("Detection indicators: ", end='', flush=True)
    
    try:
        with stream:
            while True:
                if recorder.should_stop:
                    print("\n\nStopping due to silence timeout...")
                    break
                try:
                    await asyncio.sleep(0.1)
                except KeyboardInterrupt:
                    print("\n\nStopping recording...")
                    break
        
        return recorder.stop_recording()
        
    except Exception as e:
        print(f"\n❌ Error during recording: {e}")
        return None

async def transcribe_audio(audio_file):
    """Transcribe audio using Whisper"""
    MODEL_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "models",
        "whisper",
        "tiny"
    )
    
    if not os.path.exists(MODEL_PATH):  # Fixed typo here
        print(f"Error: Whisper model not found at {MODEL_PATH}")
        return None
    
    try:
        print("\nInitializing Whisper model...")
        model = WhisperModel(MODEL_PATH, device="cpu", compute_type="int8")
        
        print("Transcribing audio...")
        start_time = time.time()
        
        segments, info = model.transcribe(
            audio_file,
            language="en",
            vad_filter=True
        )
        
        transcribed_text = " ".join(segment.text for segment in segments)
        
        print(f"\nTranscription completed in {time.time() - start_time:.2f} seconds")
        return transcribed_text.strip()
        
    except Exception as e:
        print(f"\n❌ Error during transcription: {e}")
        return None

async def main():
    """Run the buffered recording and transcription test"""
    print("\n=== Testing Buffered Audio Recording with Whisper Transcription ===")
    
    # Record audio
    print("\nSpeak when ready. Press Ctrl+C when done speaking.")
    audio_file = await record_audio()
    
    if audio_file:
        # Transcribe the recorded audio
        transcription = await transcribe_audio(audio_file)
        
        if transcription:
            print("\nTranscription Results:")
            print("-" * 40)
            print(transcription)
            print("-" * 40)
        else:
            print("\nTranscription failed.")
    else:
        print("\nNo audio was recorded.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest stopped by user.")
    except Exception as e:
        print(f"\nError: {e}")