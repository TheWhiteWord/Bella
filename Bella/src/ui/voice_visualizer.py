#!/usr/bin/env python3
"""
Voice Visualizer for Bella Assistant

This module provides a voice visualization component that displays
animated wave patterns synchronized with voice output. The visualization
adjusts both:
- Wave amplitude based on voice intensity
- Wave phase for continuous horizontal movement

It uses pre-generated frames from the wave_frame_generator to create smooth,
efficient animations during speech.
"""

import os
import re
import time
import math
import numpy as np
from pathlib import Path
import asyncio
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union, Any

# Import PIL for image handling
from PIL import Image, ImageTk

# Import Tkinter for UI if available, else use placeholder
try:
    import tkinter as tk
    from tkinter import ttk
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

# Constants
FRAMES_DIR = os.path.join(os.path.dirname(__file__), "wave_frames")
DEFAULT_FPS = 24
DEFAULT_PHASE_SPEED = 1.0  # Complete cycles per second


class VoiceWaveFrames:
    """Manages the preloaded animation frames for voice visualization."""
    
    def __init__(self, frames_dir: str = FRAMES_DIR):
        """
        Initialize the frame manager.
        
        Args:
            frames_dir: Directory containing the pre-generated wave frames
        """
        self.frames_dir = frames_dir
        self.frames_by_amplitude = {}  # Dict[amplitude, Dict[phase_idx, Image]]
        self.amplitudes = []
        self.phase_count = 0
        self.loaded = False
    
    def load_frames(self) -> bool:
        """
        Load all animation frames from the directory.
        
        Returns:
            bool: True if frames were loaded successfully, False otherwise
        """
        if not os.path.exists(self.frames_dir):
            print(f"Error: Frames directory not found: {self.frames_dir}")
            return False
            
        try:
            frame_files = sorted([f for f in os.listdir(self.frames_dir) 
                                 if f.endswith('.png')])
            
            if not frame_files:
                print(f"Error: No PNG frames found in {self.frames_dir}")
                return False
                
            frames_by_amplitude = defaultdict(dict)
            
            # Parse amplitude and phase from filenames (wave_frame_a30.0_p05.png)
            for filename in frame_files:
                match = re.search(r'a(\d+\.\d+)_p(\d+)', filename)
                if match:
                    amplitude = float(match.group(1))
                    phase_idx = int(match.group(2))
                    
                    # Load the image
                    img_path = os.path.join(self.frames_dir, filename)
                    img = Image.open(img_path)
                    
                    # Store by amplitude and phase
                    frames_by_amplitude[amplitude][phase_idx] = img
            
            # Save the loaded frames
            self.frames_by_amplitude = dict(frames_by_amplitude)
            self.amplitudes = sorted(self.frames_by_amplitude.keys())
            
            # Find the number of phases (should be the same for each amplitude)
            if self.amplitudes:
                self.phase_count = len(self.frames_by_amplitude[self.amplitudes[0]])
            
            self.loaded = True
            print(f"Loaded {len(frame_files)} frames across {len(self.amplitudes)} amplitude levels")
            print(f"Each amplitude has {self.phase_count} phase positions")
            
            return True
            
        except Exception as e:
            print(f"Error loading frames: {e}")
            return False
    
    def get_frame(self, intensity: float, time_position: float, 
                  max_intensity: float = 1.0) -> Optional[Image.Image]:
        """
        Get the appropriate frame based on audio intensity and time position.
        
        Args:
            intensity: Audio intensity value between 0 and max_intensity
            time_position: Current time position in seconds (for phase selection)
            max_intensity: Maximum possible intensity value
            
        Returns:
            PIL Image of the selected frame, or None if frames aren't loaded
        """
        if not self.loaded or not self.amplitudes:
            return None
            
        # Apply a non-linear mapping to make the visualizer more responsive 
        # to subtle voice changes in the lower intensity range
        if intensity < 0.25:
            # For very low intensities (0-0.25), use stronger exponential mapping 
            # to make even tiny whispers visible
            mapped_intensity = (intensity / 0.25) ** 0.6 * 0.25
        elif intensity < 0.5:
            # For low-medium intensities (0.25-0.5), use gentler curve
            # This provides good responsiveness for normal speech
            mapped_intensity = 0.25 + (intensity - 0.25) * (0.3 / 0.25)
        elif intensity < 0.8:
            # For medium-high intensities (0.5-0.8), slightly steeper curve
            # This adds more visual impact for emphasis in speech
            mapped_intensity = 0.55 + (intensity - 0.5) * (0.25 / 0.3)
        else:
            # For high intensities (0.8-1.0), use exponential mapping
            # This makes loud peaks really stand out visually
            mapped_intensity = 0.8 + (1.0 - 0.8) * ((intensity - 0.8) / 0.2) ** 1.2
        
        # Map intensity to amplitude (find closest available amplitude)
        target_amplitude = mapped_intensity * max(self.amplitudes) / max_intensity
        amplitude = min(self.amplitudes, key=lambda a: abs(a - target_amplitude))
        
        # Select phase based on time position (creates continuous movement)
        # time_position * DEFAULT_PHASE_SPEED determines how many cycles per second
        phase_position = (time_position * DEFAULT_PHASE_SPEED) % 1.0
        phase_idx = int(phase_position * self.phase_count) % self.phase_count
        
        # Return the frame
        if amplitude in self.frames_by_amplitude and phase_idx in self.frames_by_amplitude[amplitude]:
            return self.frames_by_amplitude[amplitude][phase_idx]
        
        return None


class AudioIntensityMonitor:
    """
    Monitors audio intensity from the voice assistant output.
    This is a simplified implementation that can be connected to the actual audio system.
    """
    
    def __init__(self):
        """Initialize the audio intensity monitor."""
        self.is_speaking = False
        self.current_intensity = 0.0
        self.last_intensity = 0.0
        self.base_smoothing = 0.3     # Increased base smoothing factor for faster response
        self.attack_smoothing = 0.7   # Increased attack smoothing for quicker rises
        self.release_smoothing = 0.2  # Increased release smoothing for smoother falls
        self.min_delta_threshold = 0.008  # Lower threshold to respond to smaller changes
        
    async def start_monitoring(self):
        """Start monitoring audio intensity (placeholder for actual implementation)."""
        self.is_speaking = True
    
    async def stop_monitoring(self):
        """Stop monitoring audio intensity."""
        self.is_speaking = False
        self.current_intensity = 0.0
        self.last_intensity = 0.0
    
    def update_intensity(self, new_intensity: float):
        """
        Update the current intensity value with adaptive smoothing.
        Uses faster response for rising intensity and slower for falling intensity.
        
        Args:
            new_intensity: New intensity value from audio system
        """
        if not self.is_speaking:
            self.current_intensity = 0.0
            self.last_intensity = 0.0
            return
            
        # Calculate change from previous intensity
        delta = new_intensity - self.last_intensity
        
        # Choose smoothing factor based on whether intensity is rising or falling
        if abs(delta) < self.min_delta_threshold:
            # Small change, use base smoothing
            smoothing = self.base_smoothing
        elif delta > 0:
            # Rising intensity (attack) - respond quickly
            smoothing = self.attack_smoothing
        else:
            # Falling intensity (release) - respond more slowly
            smoothing = self.release_smoothing
        
        # Apply smoothing with the selected factor
        self.current_intensity = (self.current_intensity * (1 - smoothing) + 
                                 new_intensity * smoothing)
        
        # Store last intensity for next comparison
        self.last_intensity = self.current_intensity
    
    def get_current_intensity(self) -> float:
        """
        Get the current smoothed intensity value.
        
        Returns:
            float: Current intensity value between 0.0 and 1.0
        """
        return self.current_intensity


class VoiceVisualizerUI:
    """
    Voice visualizer UI component that can be integrated into the main application.
    """
    
    def __init__(self, parent=None, size: Tuple[int, int] = (400, 400)):
        """
        Initialize the voice visualizer UI.
        
        Args:
            parent: Parent UI container (for Tkinter integration)
            size: Tuple of (width, height) for the visualizer
        """
        self.parent = parent
        self.size = size
        self.frames = VoiceWaveFrames()
        self.intensity_monitor = AudioIntensityMonitor()
        
        # Animation state
        self.animation_running = False
        self.start_time = 0
        self.current_frame = None
        self.current_frame_image = None  # For Tkinter PhotoImage
        
        # UI elements
        self.canvas = None
        self.frame_image_id = None
        
        # Initialize UI if parent is provided
        if parent is not None and TKINTER_AVAILABLE:
            self._init_ui()
    
    def _init_ui(self):
        """Initialize the UI components (Tkinter specific)."""
        if not TKINTER_AVAILABLE:
            print("Tkinter not available, UI initialization skipped")
            return
            
        # Create a canvas for displaying the frames
        self.canvas = tk.Canvas(
            self.parent, 
            width=self.size[0], 
            height=self.size[1],
            bg="black",
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
    
    async def initialize(self):
        """Initialize the visualizer, loading frames and preparing for animation."""
        # Load the frame images
        success = self.frames.load_frames()
        if not success:
            print("Failed to load visualization frames")
            return False
        
        return True
    
    async def start_animation(self):
        """Start the voice visualization animation."""
        if self.animation_running:
            return
            
        self.animation_running = True
        self.start_time = time.time()
        await self.intensity_monitor.start_monitoring()
        
        # Start the animation loop
        asyncio.create_task(self._animation_loop())
    
    async def stop_animation(self):
        """Stop the voice visualization animation."""
        self.animation_running = False
        await self.intensity_monitor.stop_monitoring()
    
    async def _animation_loop(self):
        """Main animation loop that updates the visualization."""
        frame_delay = 1.0 / DEFAULT_FPS
        
        while self.animation_running:
            # Get current time position for phase selection
            current_time = time.time() - self.start_time
            
            # Get current intensity from audio monitor
            intensity = self.intensity_monitor.get_current_intensity()
            
            # Get the appropriate frame
            self.current_frame = self.frames.get_frame(
                intensity=intensity,
                time_position=current_time
            )
            
            # Update the UI with the new frame
            if self.current_frame:
                self._update_ui_frame(self.current_frame)
            
            # Wait for next frame
            await asyncio.sleep(frame_delay)
    
    def _update_ui_frame(self, frame: Image.Image):
        """
        Update the UI with a new frame image.
        
        Args:
            frame: PIL Image to display
        """
        if not TKINTER_AVAILABLE or not self.canvas:
            return
        
        try:
            # Resize the frame if needed
            if frame.size != self.size:
                frame = frame.resize(self.size, Image.Resampling.LANCZOS)
            
            # Convert PIL image to Tkinter PhotoImage
            self.current_frame_image = ImageTk.PhotoImage(frame)
            
            # Update or create the image on canvas
            if self.frame_image_id is None:
                self.frame_image_id = self.canvas.create_image(
                    self.size[0] // 2, self.size[1] // 2,  # Center position
                    image=self.current_frame_image
                )
            else:
                self.canvas.itemconfig(self.frame_image_id, image=self.current_frame_image)
        except (RuntimeError, tk.TclError) as e:
            # Handle Tkinter errors gracefully
            print(f"Tkinter error in updating frame: {e}")
            # Don't propagate the error - just skip this frame
    
    def update_intensity(self, intensity: float):
        """
        Update the current voice intensity.
        
        Args:
            intensity: New intensity value between 0.0 and 1.0
        """
        self.intensity_monitor.update_intensity(intensity)


# Simple standalone demo if run directly
async def run_demo():
    """Run a simple standalone demo of the voice visualizer."""
    if not TKINTER_AVAILABLE:
        print("Tkinter not available, cannot run demo")
        return
        
    # Create the Tkinter root window
    root = tk.Tk()
    root.title("Bella Voice Visualizer Demo")
    root.geometry("500x600")
    root.configure(bg="black")
    
    # Create a frame for the visualizer
    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
    # Create the visualizer
    visualizer = VoiceVisualizerUI(parent=frame, size=(400, 400))
    await visualizer.initialize()
    
    # Create intensity slider for demo
    intensity_var = tk.DoubleVar(value=0.0)
    
    def on_intensity_change(event=None):
        """Handle intensity slider changes."""
        visualizer.update_intensity(intensity_var.get())
    
    # Add control widgets
    controls_frame = ttk.Frame(root)
    controls_frame.pack(fill=tk.X, padx=20, pady=10)
    
    # Intensity slider
    ttk.Label(controls_frame, text="Voice Intensity:").pack(anchor=tk.W)
    intensity_slider = ttk.Scale(
        controls_frame, 
        from_=0.0, 
        to=1.0, 
        orient=tk.HORIZONTAL,
        variable=intensity_var,
        command=on_intensity_change
    )
    intensity_slider.pack(fill=tk.X, pady=5)
    
    # Start/stop buttons
    buttons_frame = ttk.Frame(controls_frame)
    buttons_frame.pack(fill=tk.X, pady=10)
    
    async def start_animation():
        await visualizer.start_animation()
        
    async def stop_animation():
        await visualizer.stop_animation()
    
    start_button = ttk.Button(
        buttons_frame, 
        text="Start Animation",
        command=lambda: asyncio.create_task(start_animation())
    )
    start_button.pack(side=tk.LEFT, padx=5)
    
    stop_button = ttk.Button(
        buttons_frame, 
        text="Stop Animation",
        command=lambda: asyncio.create_task(stop_animation())
    )
    stop_button.pack(side=tk.LEFT, padx=5)
    
    # Start the animation
    await visualizer.start_animation()
    
    # Set up periodic UI updates for tkinter
    async def update_tk():
        while True:
            root.update()
            await asyncio.sleep(1/60)  # ~60 fps for UI updates
    
    # Run the UI update loop
    asyncio.create_task(update_tk())
    
    # Demo animation pattern - simulate speech with varying intensity
    async def demo_intensity_pattern():
        while True:
            # Simulate voice starting
            for i in range(30):
                # Ramp up intensity
                intensity = min(1.0, i / 30 * 0.8)
                intensity_var.set(intensity)
                on_intensity_change()
                await asyncio.sleep(0.05)
            
            # Hold at medium intensity with variations
            base_intensity = 0.6
            for i in range(100):
                # Add some natural variation
                variation = 0.2 * math.sin(i / 10 * math.pi) + 0.1 * math.sin(i / 3 * math.pi)
                intensity = max(0.1, min(1.0, base_intensity + variation))
                intensity_var.set(intensity)
                on_intensity_change()
                await asyncio.sleep(0.05)
            
            # Ramp down intensity (ending speech)
            for i in range(20):
                intensity = max(0.0, 0.6 - (i / 20 * 0.6))
                intensity_var.set(intensity)
                on_intensity_change()
                await asyncio.sleep(0.05)
            
            # Pause before next cycle
            intensity_var.set(0.0)
            on_intensity_change()
            await asyncio.sleep(2.0)
    
    # Start the demo pattern
    asyncio.create_task(demo_intensity_pattern())
    
    # Keep the asyncio event loop running
    while True:
        await asyncio.sleep(0.1)


if __name__ == "__main__":
    # Run the demo application
    asyncio.run(run_demo())
