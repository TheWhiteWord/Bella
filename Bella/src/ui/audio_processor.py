#!/usr/bin/env python3
"""
Bella Voice Visualizer - Audio Processor

This module provides audio processing for the Bella Voice Visualizer,
capturing the voice assistant's audio output and calculating intensity
metrics for visualization.

The AudioProcessor class connects to the voice system and provides
real-time intensity measurements that the visualizer can use to adjust
its amplitude.
"""

import numpy as np
import time
import asyncio
import threading
from typing import Optional, Callable, Dict, List, Union, Tuple
from queue import Queue, Empty
from PyQt5.QtCore import QObject, pyqtSignal, QTimer

class AudioProcessor(QObject):
    """
    Processes audio from Bella's voice system to extract intensity information.
    Emits intensity updates that can be used by the visualizer to adjust amplitude.
    """
    
    # Signal emitted when audio intensity changes
    intensity_updated = pyqtSignal(float)
    
    def __init__(self, 
                 smoothing_factor: float = 0.3, 
                 update_interval_ms: int = 30,
                 parent: Optional[QObject] = None):
        """
        Initialize the audio processor
        
        Args:
            smoothing_factor: Value between 0-1 determining how much to smooth intensity changes
                              (higher = smoother transitions but less responsive)
            update_interval_ms: How frequently to emit intensity updates (milliseconds)
            parent: Parent QObject
        """
        super().__init__(parent)
        
        # Configuration
        self.smoothing_factor = smoothing_factor
        self.update_interval_ms = update_interval_ms
        
        # State tracking
        self.current_intensity = 0.0
        self.is_speaking = False
        self.audio_buffer = Queue(maxsize=100)
        
        # Set up timer for regular intensity updates
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._process_audio)
        self.update_timer.setInterval(update_interval_ms)
        
        # Track when speaking ends to fade out gracefully
        self.last_audio_time = 0
        self.fade_out_threshold_sec = 0.5  # Time after last audio chunk to consider speech ended
    
    def start(self):
        """Start audio processing"""
        self.update_timer.start()
    
    def stop(self):
        """Stop audio processing"""
        self.update_timer.stop()
    
    def add_audio_chunk(self, audio_data: np.ndarray):
        """
        Add a new chunk of audio data to process
        
        Args:
            audio_data: Audio data as numpy array (typically float32 samples in [-1, 1] range)
        """
        self.last_audio_time = time.time()
        self.is_speaking = True
        
        # Add to buffer without blocking if buffer is full
        try:
            self.audio_buffer.put_nowait(audio_data)
        except:
            # If buffer is full, just discard oldest data
            try:
                self.audio_buffer.get_nowait()
                self.audio_buffer.put_nowait(audio_data)
            except:
                pass  # Ignore if can't add
    
    def _process_audio(self):
        """Process audio in the buffer and update intensity"""
        current_time = time.time()
        
        # Handle silence detection
        if self.is_speaking and current_time - self.last_audio_time > self.fade_out_threshold_sec:
            self.is_speaking = False
        
        # If not speaking, gradually reduce intensity to zero
        if not self.is_speaking:
            target_intensity = 0.0
            self.current_intensity = (self.smoothing_factor * self.current_intensity + 
                                     (1 - self.smoothing_factor) * target_intensity)
            
            # Emit updated intensity
            self.intensity_updated.emit(self.current_intensity)
            return
        
        # Process any audio in buffer
        audio_chunks = []
        while not self.audio_buffer.empty():
            try:
                audio_chunks.append(self.audio_buffer.get_nowait())
            except Empty:
                break
        
        # If no audio chunks, keep current intensity
        if not audio_chunks:
            return
        
        # Combine audio chunks and calculate intensity
        audio_data = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
        
        # Calculate RMS (root mean square) as intensity metric
        rms = np.sqrt(np.mean(np.square(audio_data)))
        
        # Apply loudness curve (nonlinear mapping to make visualization more responsive)
        # This emphasizes medium levels which are more common in speech
        target_intensity = self._loudness_mapping(rms)
        
        # Apply smoothing
        self.current_intensity = (self.smoothing_factor * self.current_intensity + 
                                 (1 - self.smoothing_factor) * target_intensity)
        
        # Emit updated intensity
        self.intensity_updated.emit(self.current_intensity)
    
    def _loudness_mapping(self, rms_value: float) -> float:
        """
        Map RMS value to a perceptual intensity level (0.0-1.0)
        Using a curve that emphasizes the mid-range of speech
        
        Args:
            rms_value: RMS value of audio
            
        Returns:
            Intensity value between 0.0 and 1.0
        """
        # These threshold values should be tuned based on your specific audio setup
        silence_threshold = 0.01
        normal_speech = 0.1
        loud_speech = 0.3
        
        if rms_value < silence_threshold:
            return 0.0
        elif rms_value >= loud_speech:
            # Compress very loud sounds to avoid maxing out
            return min(0.8 + (rms_value - loud_speech) * 0.2, 1.0)
        elif rms_value < normal_speech:
            # Linear mapping for quiet to normal speech (0.0 - 0.6)
            normalized = (rms_value - silence_threshold) / (normal_speech - silence_threshold)
            return normalized * 0.6
        else:
            # Mapping for normal to loud speech (0.6 - 0.8)
            normalized = (rms_value - normal_speech) / (loud_speech - normal_speech)
            return 0.6 + normalized * 0.2
    
    def set_speaking(self, is_speaking: bool):
        """
        Manually set speaking state (for integration with external speech systems)
        
        Args:
            is_speaking: Whether the system is currently speaking
        """
        self.is_speaking = is_speaking
        if is_speaking:
            self.last_audio_time = time.time()
        else:
            # Add a small delay before considering speech ended
            self.last_audio_time = time.time() - (self.fade_out_threshold_sec / 2)


class BellaAudioIntegration:
    """
    Integration with Bella's audio output system.
    Connects to the TTS system to capture audio output for visualization.
    """
    
    def __init__(self, audio_processor: AudioProcessor):
        """
        Initialize Bella audio integration
        
        Args:
            audio_processor: The AudioProcessor instance to feed audio data to
        """
        self.audio_processor = audio_processor
        self.is_connected = False
    
    async def connect_to_bella_audio(self):
        """
        Connect to Bella's audio output system (TTS)
        This is an implementation placeholder - modify to match your actual TTS system
        """
        # This would be replaced with actual code to connect to your TTS system
        # For example, hooking into your TTS pipeline or stream
        self.is_connected = True
        print("Connected to Bella audio system")
        return True
    
    def handle_audio_callback(self, audio_data: np.ndarray):
        """
        Callback function to be called from Bella's TTS system when audio is played
        
        Args:
            audio_data: Audio data chunk as numpy array
        """
        self.audio_processor.add_audio_chunk(audio_data)
    
    def on_speech_start(self):
        """Called when Bella starts speaking"""
        self.audio_processor.set_speaking(True)
    
    def on_speech_end(self):
        """Called when Bella stops speaking"""
        self.audio_processor.set_speaking(False)
    
    def disconnect(self):
        """Disconnect from Bella's audio system"""
        self.is_connected = False


# Example usage (would be integrated with your voice assistant system)
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    # Create app for Qt signals
    app = QApplication(sys.argv)
    
    # Create audio processor
    processor = AudioProcessor()
    
    # Connect intensity signal to a print function for debugging
    processor.intensity_updated.connect(
        lambda intensity: print(f"Current intensity: {intensity:.2f}")
    )
    
    # Start the processor
    processor.start()
    
    # Simulate some audio with increasing then decreasing intensity
    for i in range(100):
        # Create a sine wave
        frequency = 440  # Hz
        sample_rate = 44100  # samples per second
        duration = 0.05  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Apply an envelope to simulate changing loudness
        if i < 50:
            amplitude = i / 50.0  # Increasing
        else:
            amplitude = (100 - i) / 50.0  # Decreasing
        audio = audio * amplitude
        
        # Add to processor
        processor.add_audio_chunk(audio)
        
        # Sleep briefly
        time.sleep(0.05)
    
    # Manual clean-up to stop ongoing timer
    processor.stop()
    
    # Exit after 1 second to see final intensity values
    QTimer.singleShot(1000, app.quit)
    sys.exit(app.exec_())
