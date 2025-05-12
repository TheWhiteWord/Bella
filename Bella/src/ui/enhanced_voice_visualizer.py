#!/usr/bin/env python3
"""
Voice Visualizer with Hue-Shifting Screen Overlay

This module enhances the voice visualizer by adding a hue-shifting effect
to the screen overlay, creating a dynamic and engaging visual experience.
"""

import os
import math
import time
import asyncio
import numpy as np
import colorsys
from PIL import Image, ImageTk
from typing import Optional, Dict, Tuple

# Import the original voice visualizer as base
from Bella.src.ui.voice_visualizer import VoiceVisualizerUI, VoiceWaveFrames, AudioIntensityMonitor

# Default constants for the hue shift effect
DEFAULT_HUE_SHIFT_SPEED = 0.1  # Hue cycles per second
DEFAULT_SATURATION_FACTOR = 1.0  # Color saturation multiplier
DEFAULT_HUE_SHIFT_ENABLED = True  # Enable hue shifting by default


class HueShiftingScreen:
    """
    Manages the hue-shifting effect for the screen overlay image.
    This adds a dynamic color effect to the visualization.
    """
    
    def __init__(self, 
                 screen_path: str,
                 hue_shift_speed: float = DEFAULT_HUE_SHIFT_SPEED,
                 saturation_factor: float = DEFAULT_SATURATION_FACTOR,
                 enabled: bool = DEFAULT_HUE_SHIFT_ENABLED,
                 opacity: float = 0.7):  # Added opacity parameter
        """
        Initialize the hue shifting screen manager.
        
        Args:
            screen_path: Path to the screen overlay image
            hue_shift_speed: Speed of hue cycling in cycles per second
            saturation_factor: Multiplier for color saturation
            enabled: Whether hue shifting is enabled
            opacity: Opacity of the screen overlay (0.0-1.0)
        """
        self.screen_path = screen_path
        self.hue_shift_speed = hue_shift_speed
        self.saturation_factor = saturation_factor
        self.enabled = enabled
        self.opacity = opacity  # Store opacity value
        
        # State variables
        self.screen_image = None
        self.current_hue = 0.0
        self.start_time = time.time()
        
        # Pre-processed screen images at different hues (for caching)
        self.cached_screens = {}
        self.cache_size = 36  # Cache 36 different hue values (every 10 degrees)
    
    def load_screen(self) -> bool:
        """
        Load the screen overlay image.
        
        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            if os.path.exists(self.screen_path):
                # Load screen image
                self.screen_image = Image.open(self.screen_path).convert("RGBA")
                
                # Adjust opacity if needed for better visibility with the wave
                if self.opacity < 1.0:
                    # Create a copy of the image
                    img_arr = np.array(self.screen_image, dtype=np.float32) / 255.0
                    
                    # Scale the alpha channel
                    img_arr[:, :, 3] *= self.opacity
                    
                    # Convert back to PIL image
                    self.screen_image = Image.fromarray((img_arr * 255).astype(np.uint8), 'RGBA')
                
                print(f"Loaded screen overlay from {self.screen_path} with opacity {self.opacity}")
                return True
            else:
                print(f"Error: Screen image not found at {self.screen_path}")
                return False
        except Exception as e:
            print(f"Error loading screen image: {e}")
            return False
    
    def reset(self):
        """Reset the hue shifting animation timer."""
        self.start_time = time.time()
        self.current_hue = 0.0
    
    def apply_hue_shift(self, image: Image.Image, hue_value: float) -> Image.Image:
        """
        Apply a hue shift to the given image.
        
        Args:
            image: PIL Image in RGBA format
            hue_value: New hue value (0-1)
            
        Returns:
            PIL Image with shifted hue
        """
        # First check if we have this hue cached
        cache_key = round(hue_value * self.cache_size) % self.cache_size
        if cache_key in self.cached_screens:
            return self.cached_screens[cache_key]
        
        # Convert to numpy array for faster processing
        img_arr = np.array(image, dtype=np.float32) / 255.0
        
        # Extract RGB and alpha channels
        r, g, b, a = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2], img_arr[:, :, 3]
        
        # Find non-transparent pixels to process
        mask = a > 0.1
        
        # Create output array
        output = np.zeros_like(img_arr)
        output[:, :, 3] = a  # Preserve alpha
        
        # Only process non-transparent pixels
        if np.any(mask):
            # Convert RGB to HSV
            h, s, v = np.zeros_like(r[mask]), np.zeros_like(r[mask]), np.zeros_like(r[mask])
            
            # Process in batches to avoid memory issues
            for i, (ri, gi, bi) in enumerate(zip(r[mask].flat, g[mask].flat, b[mask].flat)):
                hi, si, vi = colorsys.rgb_to_hsv(ri, gi, bi)
                h.flat[i], s.flat[i], v.flat[i] = hi, si, vi
            
            # Shift hue and adjust saturation
            h = (h + hue_value) % 1.0
            s = np.clip(s * self.saturation_factor, 0, 1)
            
            # Convert back to RGB
            new_r, new_g, new_b = np.zeros_like(h), np.zeros_like(h), np.zeros_like(h)
            for i, (hi, si, vi) in enumerate(zip(h.flat, s.flat, v.flat)):
                ri, gi, bi = colorsys.hsv_to_rgb(hi, si, vi)
                new_r.flat[i], new_g.flat[i], new_b.flat[i] = ri, gi, bi
            
            # Copy back to output
            output_flat = output[mask]
            output_flat[:, 0], output_flat[:, 1], output_flat[:, 2] = new_r, new_g, new_b
            output[mask] = output_flat
        
        # Convert back to PIL Image
        output = (output * 255).astype(np.uint8)
        result = Image.fromarray(output, 'RGBA')
        
        # Store in cache
        self.cached_screens[cache_key] = result
        
        return result
    
    def get_current_screen(self) -> Optional[Image.Image]:
        """
        Get the screen overlay with current hue shift applied.
        
        Returns:
            PIL Image with current hue shift, or None if no screen is loaded
        """
        if self.screen_image is None:
            return None
        
        if not self.enabled:
            return self.screen_image
        
        # Calculate current hue based on elapsed time
        elapsed_time = time.time() - self.start_time
        self.current_hue = (elapsed_time * self.hue_shift_speed) % 1.0
        
        # Apply hue shift
        return self.apply_hue_shift(self.screen_image, self.current_hue)
    
    def set_opacity(self, opacity: float):
        """
        Set the opacity of the screen overlay and reload the image.
        
        Args:
            opacity: Opacity value between 0.0 and 1.0
        """
        self.opacity = max(0.0, min(1.0, opacity))
        
        # Reload the screen with the new opacity if image is already loaded
        if self.screen_image is not None:
            self.load_screen()


class EnhancedVoiceVisualizerUI(VoiceVisualizerUI):
    """
    Enhanced voice visualizer with hue-shifting screen overlay effect.
    """
    
    def __init__(self, parent=None, size: Tuple[int, int] = (400, 400)):
        """
        Initialize the enhanced voice visualizer UI.
        
        Args:
            parent: Parent UI container (for Tkinter integration)
            size: Tuple of (width, height) for the visualizer
        """
        # Initialize the base class
        super().__init__(parent, size)
        
        # Set up the screen overlay with hue shifting
        elements_dir = os.path.join(os.path.dirname(__file__), "elements")
        screen_path = os.path.join(elements_dir, "screen.png")
        # Use a lower opacity (0.6) to ensure the wave is clearly visible through the screen overlay
        self.hue_shifting_screen = HueShiftingScreen(
            screen_path,
            opacity=0.6  # Reduced opacity for better wave visibility
        )
        
        # Add hue shift configuration
        self.hue_shift_enabled = DEFAULT_HUE_SHIFT_ENABLED
        self.hue_shift_speed = DEFAULT_HUE_SHIFT_SPEED
        self.saturation_factor = DEFAULT_SATURATION_FACTOR
        
        # Wave hue shifting configuration
        self.wave_hue_shift_enabled = True  # Enable hue shifting for the wave
        self.wave_hue_offset = 0.5  # Complementary color to the screen (half cycle offset)
        self.wave_saturation_factor = 1.5  # Slightly higher saturation for wave
    
    async def initialize(self):
        """Initialize the visualizer, loading frames and preparing for animation."""
        # Initialize the base visualizer
        success = await super().initialize()
        if not success:
            return False
        
        # Load the screen overlay
        screen_success = self.hue_shifting_screen.load_screen()
        if not screen_success:
            print("Warning: Failed to load screen overlay. Visualizer will work without it.")
        
        return True
    
    async def start_animation(self):
        """Start the voice visualization animation with hue-shifting screen."""
        # Reset the hue shifting timer when starting animation
        self.hue_shifting_screen.reset()
        
        # Call the base class method to start animation
        await super().start_animation()
    
    def set_hue_shift_speed(self, speed: float):
        """
        Set the speed of the hue shifting effect.
        
        Args:
            speed: Hue cycles per second (0 = no shift, higher = faster)
        """
        self.hue_shift_speed = speed
        self.hue_shifting_screen.hue_shift_speed = speed
    
    def set_saturation_factor(self, factor: float):
        """
        Set the saturation factor for the hue shifting effect.
        
        Args:
            factor: Saturation multiplier (0 = grayscale, 1 = normal, >1 = increased)
        """
        self.saturation_factor = factor
        self.hue_shifting_screen.saturation_factor = factor
    
    def set_hue_shift_enabled(self, enabled: bool):
        """
        Enable or disable the hue shifting effect.
        
        Args:
            enabled: Whether the hue shifting effect is enabled
        """
        self.hue_shift_enabled = enabled
        self.hue_shifting_screen.enabled = enabled
        
    def set_wave_hue_shift_enabled(self, enabled: bool):
        """
        Enable or disable hue shifting for the wave itself.
        
        Args:
            enabled: Whether the wave should have hue shifting applied
        """
        self.wave_hue_shift_enabled = enabled
        
    def set_wave_hue_offset(self, offset: float):
        """
        Set the hue offset for the wave relative to the screen.
        
        Args:
            offset: Hue offset between 0.0 and 1.0 (0.5 = complementary color)
        """
        self.wave_hue_offset = max(0.0, min(1.0, offset))
    
    def set_screen_opacity(self, opacity: float):
        """
        Set the opacity of the screen overlay.
        
        Args:
            opacity: Opacity value between 0.0 and 1.0
        """
        self.hue_shifting_screen.set_opacity(opacity)
    
    def apply_hue_shift_to_wave(self, wave_frame: Image.Image, hue_value: float) -> Image.Image:
        """
        Apply a hue shift to the wave frame.
        
        Args:
            wave_frame: Original wave frame image
            hue_value: Hue value to apply
            
        Returns:
            Wave frame with hue shift applied
        """
        # Convert to numpy array for faster processing
        img_arr = np.array(wave_frame, dtype=np.float32) / 255.0
        
        # Extract RGB and alpha channels
        r, g, b, a = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2], img_arr[:, :, 3]
        
        # Find non-transparent pixels to process (only process the wave part)
        mask = a > 0.1
        
        # Create output array
        output = np.zeros_like(img_arr)
        output[:, :, 3] = a  # Preserve alpha
        
        # Only process non-transparent pixels
        if np.any(mask):
            # Convert RGB to HSV
            h, s, v = np.zeros_like(r[mask]), np.zeros_like(r[mask]), np.zeros_like(r[mask])
            
            # Process in batches to avoid memory issues
            for i, (ri, gi, bi) in enumerate(zip(r[mask].flat, g[mask].flat, b[mask].flat)):
                hi, si, vi = colorsys.rgb_to_hsv(ri, gi, bi)
                h.flat[i], s.flat[i], v.flat[i] = hi, si, vi
            
            # Shift hue and adjust saturation
            h = (h + hue_value) % 1.0
            s = np.clip(s * self.saturation_factor, 0, 1)
            
            # Convert back to RGB
            new_r, new_g, new_b = np.zeros_like(h), np.zeros_like(h), np.zeros_like(h)
            for i, (hi, si, vi) in enumerate(zip(h.flat, s.flat, v.flat)):
                ri, gi, bi = colorsys.hsv_to_rgb(hi, si, vi)
                new_r.flat[i], new_g.flat[i], new_b.flat[i] = ri, gi, bi
            
            # Copy back to output
            output_flat = output[mask]
            output_flat[:, 0], output_flat[:, 1], output_flat[:, 2] = new_r, new_g, new_b
            output[mask] = output_flat
        
        # Convert back to PIL Image
        output = (output * 255).astype(np.uint8)
        return Image.fromarray(output, 'RGBA')
            
    async def _animation_loop(self):
        """Enhanced animation loop that adds the hue-shifting screen overlay."""
        frame_delay = 1.0 / 30  # Target 30 FPS
        
        while self.animation_running:
            # Get current time for phase selection
            current_time = time.time() - self.start_time
            
            # Get current intensity from audio monitor
            intensity = self.intensity_monitor.get_current_intensity()
            
            # Get the appropriate wave frame
            wave_frame = self.frames.get_frame(
                intensity=intensity,
                time_position=current_time
            )
            
            # Get the current hue-shifted screen
            screen_overlay = self.hue_shifting_screen.get_current_screen()
            
            if wave_frame:
                # Apply hue shift to the wave if enabled
                if self.wave_hue_shift_enabled and self.hue_shift_enabled:
                    # Calculate wave hue based on screen hue plus offset
                    wave_hue = (self.hue_shifting_screen.current_hue + self.wave_hue_offset) % 1.0
                    wave_frame = self.apply_hue_shift_to_wave(wave_frame, wave_hue)
            
            # Combine the wave frame with the screen overlay
            if wave_frame and screen_overlay:
                # IMPORTANT: Changed order - put screen BEHIND the wave for proper visibility
                # Create a base black image first
                base_image = Image.new("RGBA", wave_frame.size, (0, 0, 0, 255))
                
                # Apply screen overlay to base image
                base_with_screen = Image.alpha_composite(base_image, screen_overlay)
                
                # Apply wave on TOP of the screen overlay
                self.current_frame = Image.alpha_composite(base_with_screen, wave_frame)
            elif wave_frame:
                # Just use the wave frame if no screen overlay
                self.current_frame = wave_frame
            
            # Update the UI with the new frame
            if self.current_frame:
                self._update_ui_frame(self.current_frame)
            
            # Wait for next frame
            await asyncio.sleep(frame_delay)


# Example usage
async def run_enhanced_demo():
    """Run a simple standalone demo of the enhanced voice visualizer."""
    import tkinter as tk
    from tkinter import ttk
    
    # Create the Tkinter root window
    root = tk.Tk()
    root.title("Enhanced Bella Voice Visualizer")
    root.geometry("600x700")
    root.configure(bg="#1E1E2E")  # Dark theme background
    
    # Create a frame for the visualizer
    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
    # Create the enhanced visualizer
    visualizer = EnhancedVoiceVisualizerUI(parent=frame, size=(400, 400))
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
    
    # Hue shift speed slider
    hue_speed_var = tk.DoubleVar(value=DEFAULT_HUE_SHIFT_SPEED)
    
    def on_hue_speed_change(event=None):
        """Handle hue speed slider changes."""
        visualizer.set_hue_shift_speed(hue_speed_var.get())
    
    ttk.Label(controls_frame, text="Hue Shift Speed:").pack(anchor=tk.W)
    hue_speed_slider = ttk.Scale(
        controls_frame, 
        from_=0.0, 
        to=0.5, 
        orient=tk.HORIZONTAL,
        variable=hue_speed_var,
        command=on_hue_speed_change
    )
    hue_speed_slider.pack(fill=tk.X, pady=5)
    
    # Saturation slider
    saturation_var = tk.DoubleVar(value=DEFAULT_SATURATION_FACTOR)
    
    def on_saturation_change(event=None):
        """Handle saturation slider changes."""
        visualizer.set_saturation_factor(saturation_var.get())
    
    ttk.Label(controls_frame, text="Color Saturation:").pack(anchor=tk.W)
    saturation_slider = ttk.Scale(
        controls_frame, 
        from_=0.0, 
        to=2.0, 
        orient=tk.HORIZONTAL,
        variable=saturation_var,
        command=on_saturation_change
    )
    saturation_slider.pack(fill=tk.X, pady=5)
    
    # Wave hue offset slider
    ttk.Label(controls_frame, text="Wave Hue Offset:").pack(anchor=tk.W)
    wave_hue_offset_var = tk.DoubleVar(value=0.5)  # Default to complementary color
    
    def on_wave_hue_offset_change(event=None):
        """Handle wave hue offset slider changes."""
        visualizer.set_wave_hue_offset(wave_hue_offset_var.get())
    
    wave_hue_offset_slider = ttk.Scale(
        controls_frame,
        from_=0.0,
        to=1.0,
        orient=tk.HORIZONTAL,
        variable=wave_hue_offset_var,
        command=on_wave_hue_offset_change
    )
    wave_hue_offset_slider.pack(fill=tk.X, pady=5)
    
    # Toggle buttons frame
    toggle_frame = ttk.Frame(controls_frame)
    toggle_frame.pack(fill=tk.X, pady=5)
    
    # Toggle button for screen hue shifting
    hue_shift_enabled = True
    
    def toggle_hue_shift():
        """Toggle hue shifting on/off."""
        nonlocal hue_shift_enabled
        hue_shift_enabled = not hue_shift_enabled
        visualizer.set_hue_shift_enabled(hue_shift_enabled)
        toggle_button.config(text="Enable Screen Hue" if not hue_shift_enabled else "Disable Screen Hue")
    
    toggle_button = ttk.Button(
        toggle_frame, 
        text="Disable Screen Hue",
        command=toggle_hue_shift
    )
    toggle_button.pack(side=tk.LEFT, padx=5)
    
    # Toggle button for wave hue shifting
    wave_hue_shift_enabled = True
    
    def toggle_wave_hue_shift():
        """Toggle wave hue shifting on/off."""
        nonlocal wave_hue_shift_enabled
        wave_hue_shift_enabled = not wave_hue_shift_enabled
        visualizer.set_wave_hue_shift_enabled(wave_hue_shift_enabled)
        wave_toggle_button.config(text="Enable Wave Hue" if not wave_hue_shift_enabled else "Disable Wave Hue")
    
    wave_toggle_button = ttk.Button(
        toggle_frame, 
        text="Disable Wave Hue",
        command=toggle_wave_hue_shift
    )
    wave_toggle_button.pack(side=tk.LEFT, padx=5)
    
    # Start/stop buttons
    buttons_frame = ttk.Frame(controls_frame)
    buttons_frame.pack(fill=tk.X, pady=10)
    
    async def start_animation():
        """Start the visualization animation."""
        await visualizer.start_animation()
    
    async def stop_animation():
        """Stop the visualization animation."""
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
    
    # Generate test buttons
    test_frame = ttk.LabelFrame(root, text="Test Patterns")
    test_frame.pack(fill=tk.X, padx=20, pady=10)
    
    # Start the animation
    await visualizer.start_animation()
    
    # Create test pattern functions
    async def run_amplitude_test():
        """Run an amplitude test pattern."""
        # Slow amplitude ramp up and down
        for i in range(100):
            intensity = i / 100.0
            intensity_var.set(intensity)
            on_intensity_change()
            await asyncio.sleep(0.02)
        
        for i in range(100, -1, -1):
            intensity = i / 100.0
            intensity_var.set(intensity)
            on_intensity_change()
            await asyncio.sleep(0.02)
    
    async def run_speech_pattern():
        """Run a realistic speech pattern simulation."""
        # Speech pattern parameters
        duration = 10  # seconds
        
        start_time = time.time()
        end_time = start_time + duration
        
        # Define speech envelope
        def speech_intensity(t):
            # Simulate speech intensity pattern
            base = 0.3 + 0.3 * math.sin(t * 1.5)
            syllables = 0.1 * math.sin(t * 8) * math.sin(t * 0.5 + 0.3)
            detail = 0.05 * math.sin(t * 25)
            return max(0.0, min(1.0, base + syllables + detail))
        
        # Run the pattern
        while time.time() < end_time:
            t = time.time() - start_time
            intensity = speech_intensity(t)
            intensity_var.set(intensity)
            on_intensity_change()
            await asyncio.sleep(1/30)  # 30 FPS
        
        # End at zero
        intensity_var.set(0.0)
        on_intensity_change()
    
    # Add test buttons
    ttk.Button(
        test_frame, 
        text="Amplitude Test",
        command=lambda: asyncio.create_task(run_amplitude_test())
    ).pack(side=tk.LEFT, padx=5, pady=5)
    
    ttk.Button(
        test_frame, 
        text="Speech Pattern",
        command=lambda: asyncio.create_task(run_speech_pattern())
    ).pack(side=tk.LEFT, padx=5, pady=5)
    
    # Set up periodic UI updates for tkinter
    async def update_tk():
        while True:
            try:
                root.update()
                await asyncio.sleep(1/60)  # ~60 fps for UI updates
            except:
                # Window was closed
                break
    
    # Run the UI update loop
    await update_tk()


if __name__ == "__main__":
    asyncio.run(run_enhanced_demo())
