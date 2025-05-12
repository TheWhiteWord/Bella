#!/usr/bin/env python3
"""
Voice Visualizer Integration for Bella Assistant

This module connects the voice visualizer UI with Bella's audio system,
providing real-time visualization of the assistant's voice output.
"""

import os
import time
import asyncio
import numpy as np
from typing import Optional, Callable, Dict, Any

# Import the visualizer
from Bella.src.ui.voice_visualizer import VoiceVisualizerUI

# This will be connected to Bella's audio system
class AudioIntensityAnalyzer:
    """
    Analyzes audio data from Bella's speech output to calculate intensity.
    This class bridges between the TTS system and the voice visualizer.
    """
    
    def __init__(self):
        """Initialize the audio intensity analyzer."""
        self.is_active = False
        self.intensity_callbacks = []
        self.audio_buffer = None
        self.sample_rate = 0
        self.window_size = 1024  # Audio analysis window size
        self.intensity_scale = 1.0  # Scaling factor for intensity values
        
    def register_intensity_callback(self, callback: Callable[[float], None]):
        """
        Register a callback function that will receive intensity updates.
        
        Args:
            callback: Function that accepts an intensity value
        """
        self.intensity_callbacks.append(callback)
    
    def set_audio_params(self, sample_rate: int):
        """
        Set audio parameters for analysis.
        
        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        # Adjust window size to about 20ms of audio
        self.window_size = int(0.02 * sample_rate)
    
    def process_audio_chunk(self, audio_data: np.ndarray):
        """
        Process an audio chunk and calculate intensity.
        
        Args:
            audio_data: Audio data as numpy array
        """
        if not self.is_active or audio_data is None:
            return
        
        # Buffer the audio data for analysis
        if self.audio_buffer is None:
            self.audio_buffer = audio_data
        else:
            self.audio_buffer = np.concatenate((self.audio_buffer, audio_data))
        
        # Process complete windows
        while len(self.audio_buffer) >= self.window_size:
            # Extract a window
            window = self.audio_buffer[:self.window_size]
            self.audio_buffer = self.audio_buffer[self.window_size:]
            
            # Calculate intensity (RMS amplitude)
            intensity = self._calculate_intensity(window)
            
            # Notify callbacks
            for callback in self.intensity_callbacks:
                callback(intensity)
    
    def _calculate_intensity(self, audio_window: np.ndarray) -> float:
        """
        Calculate intensity from audio window.
        
        Args:
            audio_window: Audio data window
            
        Returns:
            float: Intensity value between 0.0 and 1.0
        """
        if len(audio_window) == 0:
            return 0.0
        
        # Calculate RMS
        rms = np.sqrt(np.mean(np.square(audio_window)))
        
        # Apply scaling and normalization
        # Typical voice RMS values are between 0.01 and 0.1
        intensity = min(1.0, rms * self.intensity_scale)
        
        return intensity
    
    def start(self):
        """Start the intensity analyzer."""
        self.is_active = True
        self.audio_buffer = None
    
    def stop(self):
        """Stop the intensity analyzer."""
        self.is_active = False
        self.audio_buffer = None


class BellaVoiceVisualizer:
    """
    Main integration class for Bella's voice visualization.
    This connects the UI visualizer with the audio analysis system.
    """
    
    def __init__(self, parent=None, size=(400, 400)):
        """
        Initialize the Bella voice visualizer integration.
        
        Args:
            parent: Parent UI container
            size: Tuple of (width, height) for the visualizer
        """
        self.visualizer_ui = VoiceVisualizerUI(parent=parent, size=size)
        self.intensity_analyzer = AudioIntensityAnalyzer()
        
        # Connect intensity analyzer to visualizer
        self.intensity_analyzer.register_intensity_callback(
            self.visualizer_ui.update_intensity
        )
    
    async def initialize(self):
        """Initialize the visualizer system."""
        return await self.visualizer_ui.initialize()
    
    async def start_visualization(self):
        """Start the visualization system."""
        self.intensity_analyzer.start()
        await self.visualizer_ui.start_animation()
    
    async def stop_visualization(self):
        """Stop the visualization system."""
        self.intensity_analyzer.stop()
        await self.visualizer_ui.stop_animation()
    
    def process_audio(self, audio_data: np.ndarray, sample_rate: int = 22050):
        """
        Process audio data from the TTS system.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
        """
        # Set sample rate if it has changed
        if self.intensity_analyzer.sample_rate != sample_rate:
            self.intensity_analyzer.set_audio_params(sample_rate)
        
        # Process the audio data
        self.intensity_analyzer.process_audio_chunk(audio_data)


# Simplified example of connecting with a TTS system
async def connect_to_tts_system(visualizer: BellaVoiceVisualizer):
    """
    Connect a voice visualizer to the TTS system.
    This is a simplified example - actual implementation will
    depend on Bella's TTS architecture.
    
    Args:
        visualizer: Initialized BellaVoiceVisualizer instance
    """
    # This is where you would hook into Bella's TTS system
    # For example:
    
    # 1. Initialize the visualizer
    await visualizer.initialize()
    
    # 2. Set up hooks to start/stop visualization when TTS starts/stops
    # tts_system.register_on_start_speaking(visualizer.start_visualization)
    # tts_system.register_on_stop_speaking(visualizer.stop_visualization)
    
    # 3. Set up hook to process audio chunks as they're generated
    # def on_audio_chunk(audio_data, sample_rate):
    #     visualizer.process_audio(audio_data, sample_rate)
    # tts_system.register_audio_chunk_callback(on_audio_chunk)
    
    # The specific implementation will depend on the structure of Bella's
    # audio/TTS system, but this outlines the key integration points.
