"""Waveform audio visualization for GUI applications.

This module provides a reusable component for visualizing audio waveforms
in Tkinter applications. It supports both PipeWire and PulseAudio for capturing
audio data without using PortAudio.

Typical usage:
    visualizer = WaveformVisualizer(canvas_widget, sink_name="your_sink_name")
    visualizer.start(visualize_tts=True)  # Start visualization
    # ... later ...
    visualizer.stop()  # Stop visualization
"""
import os
import asyncio
import subprocess
import threading
import queue
import time
import traceback
from typing import Optional, Tuple, List, Dict, Any, Union
import tkinter as tk
import numpy as np


class WaveformVisualizer:
    """Audio waveform visualization component for Tkinter applications.
    
    This class handles capturing audio data from PipeWire/PulseAudio
    and visualizing it on a Tkinter canvas widget.
    """
    
    def __init__(
        self, 
        canvas: tk.Canvas,
        sink_name: Optional[str] = None,
        num_bars: int = 30,
        update_interval_ms: int = 30,
        bar_color: str = "#8B0000",
        wave_color: str = "#00FF00",
        bar_spacing: float = 0.2,
    ):
        """Initialize the waveform visualizer.
        
        Args:
            canvas: Tkinter canvas widget to draw on
            sink_name: Optional PulseAudio sink name for audio capture
            num_bars: Number of bars to display in the visualization (default: 30)
            update_interval_ms: Update interval in milliseconds (default: 30)
            bar_color: Color for visualization bars (default: dark red)
            wave_color: Color for the oscilloscope line (default: green)
            bar_spacing: Fraction of bar width to use as spacing (0-1)
        """
        self.canvas = canvas
        self.sink_name = sink_name
        self.num_bars = num_bars
        self.update_interval_ms = update_interval_ms
        self.bar_color = bar_color
        self.wave_color = wave_color
        self.bar_width_factor = 1.0 - bar_spacing
        
        # Internal state
        self.audio_queue = queue.Queue(maxsize=100)
        self.audio_subprocess = None
        self.capture_thread = None
        self.should_stop = True
        self.is_running = False
        self.after_id = None
        self.last_audio_timestamp = 0
        
        # Try to detect audio system early
        self.use_pipewire = self._check_pipewire_available()
        
    def _check_pipewire_available(self) -> bool:
        """Check if PipeWire is available by looking for pw-record."""
        try:
            result = subprocess.run(
                ['which', 'pw-record'], 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
            
    def get_monitor_source(self) -> str:
        """Get the appropriate monitor source for audio capture.
        
        Returns:
            str: The monitor source name to capture from
        """
        if self.sink_name:
            # When sink is specified, use its monitor
            monitor_source = f"{self.sink_name}.monitor"
        else:
            # Get default sink's monitor
            try:
                result = subprocess.run(
                    ['pactl', 'get-default-sink'], 
                    capture_output=True, 
                    text=True, 
                    check=True
                )
                default_sink = result.stdout.strip()
                monitor_source = f"{default_sink}.monitor"
            except Exception:
                monitor_source = "default.monitor"  # Fallback
                
        return monitor_source
        
    def start(self, visualize_tts: bool = False) -> bool:
        """Start waveform visualization.
        
        Args:
            visualize_tts: Whether visualizing TTS output (affects display style)
            
        Returns:
            bool: True if successfully started, False otherwise
        """
        # Don't restart if already running
        if self.is_running and not self.should_stop:
            return True
            
        # Reset state
        self.should_stop = False
        self.audio_queue = queue.Queue(maxsize=100)
        
        # Clear the canvas of any previous visualization
        self.canvas.delete("bar")
        self.canvas.delete("wave")
        
        # Start the capture thread
        self.capture_thread = threading.Thread(
            target=self._capture_audio_thread,
            daemon=True
        )
        self.capture_thread.start()
        
        # Start the periodic update function
        self._schedule_update()
        
        self.is_running = True
        self.last_audio_timestamp = time.time()
        return True
        
    def stop(self) -> None:
        """Stop waveform visualization and clean up resources."""
        # Set stop flag for thread
        self.should_stop = True
        
        # Cancel any pending updates
        if self.after_id:
            try:
                self.canvas.after_cancel(self.after_id)
                self.after_id = None
            except Exception:
                pass
                
        # Wait for the thread to finish if it's running
        if self.capture_thread and self.capture_thread.is_alive():
            try:
                self.capture_thread.join(timeout=1.0)  # Wait up to 1 second
            except Exception:
                pass
                
        # Clean up subprocess if it exists
        if self.audio_subprocess:
            try:
                self.audio_subprocess.terminate()
                self.audio_subprocess.wait(timeout=0.5)
            except Exception:
                try:
                    self.audio_subprocess.kill()  # Force kill if terminate fails
                except Exception:
                    pass
            finally:
                self.audio_subprocess = None
                
        # Reset the thread
        self.capture_thread = None
        
        # Clear the display
        self.canvas.delete("bar")
        self.canvas.delete("wave")
        
        self.is_running = False
        
    def _capture_audio_thread(self) -> None:
        """Thread function for capturing audio data for visualization."""
        try:
            # Get monitor source
            monitor_source = self.get_monitor_source()
            
            # Build the command with higher sample rate for better visualization
            if self.use_pipewire:
                cmd = [
                    'pw-record', 
                    '--raw', 
                    '--format=s16le', 
                    '--rate=32000', 
                    '--channels=1', 
                    '--target', 
                    monitor_source, 
                    '-'
                ]
            else:
                cmd = [
                    'parec', 
                    '--raw', 
                    '--format=s16le', 
                    '--rate=32000', 
                    '--channels=1', 
                    '-d', 
                    monitor_source
                ]
            
            # Start the subprocess with larger buffer
            self.audio_subprocess = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=8192  # Larger buffer for smoother visualization
            )
            
            # Process the audio data
            chunk_size = 4096
            sample_count = 0
            
            while not self.should_stop:
                # Read a chunk of raw audio data
                raw_data = self.audio_subprocess.stdout.read(chunk_size)
                if not raw_data:
                    break
                    
                # Convert to numpy array
                audio_data = np.frombuffer(raw_data, dtype=np.int16)
                
                # Normalize to float between -1 and 1
                normalized_data = audio_data.astype(np.float32) / 32768.0
                
                # Check if we're getting actual audio data
                rms = np.sqrt(np.mean(normalized_data**2))
                if rms > 0.001:  # Non-silent audio
                    self.last_audio_timestamp = time.time()
                
                # Log occasionally for debugging
                sample_count += 1
                if sample_count % 100 == 0:
                    # Share debug stats
                    print(f"Audio capture sample #{sample_count}: RMS={rms:.6f}, shape={normalized_data.shape}")
                
                # Put on the queue for the GUI thread
                try:
                    self.audio_queue.put(normalized_data, block=False)
                except queue.Full:
                    # Skip if the queue is full
                    pass
                    
        except Exception as e:
            print(f"Error in waveform capture: {e}")
            traceback.print_exc()
        finally:
            # Make sure to clean up
            if self.audio_subprocess:
                try:
                    self.audio_subprocess.terminate()
                    self.audio_subprocess.wait(timeout=1)
                except Exception:
                    pass
                self.audio_subprocess = None
    
    def _schedule_update(self) -> None:
        """Schedule the next update of the visualization."""
        self.after_id = self.canvas.after(
            self.update_interval_ms, 
            self._update_visualization
        )
    
    def _update_visualization(self) -> None:
        """Update the waveform visualization on the canvas."""
        # Check if we should be updating
        if self.should_stop:
            return
            
        # Try to get data from the queue
        try:
            # Get all available data (up to a limit)
            data = []
            for _ in range(5):  # Limit to 5 chunks to avoid lagging
                try:
                    chunk = self.audio_queue.get(block=False)
                    data.append(chunk)
                except queue.Empty:
                    break
                    
            if data:
                # Combine the chunks
                combined = np.concatenate(data)
                
                # Apply processing to make visualization more appealing
                # 1. Take absolute values (for display purposes)
                # 2. Apply light smoothing
                abs_data = np.abs(combined)
                if len(abs_data) > 10:  # Ensure enough samples for smoothing
                    window_size = 5
                    smoothed = np.convolve(abs_data, np.ones(window_size)/window_size, mode='valid')
                    display_data = smoothed
                else:
                    display_data = abs_data
                
                # Update visualization with the processed data
                self._draw_visualization(display_data, combined)
            else:
                # If no data, check if we've been silent for too long
                if time.time() - self.last_audio_timestamp > 0.5:
                    # Fade existing bars for a smooth visual effect
                    self._fade_visualization()
                
        except Exception as e:
            print(f"Error updating waveform: {e}")
            traceback.print_exc()
        
        # Schedule next update
        self._schedule_update()
        
    def _draw_visualization(self, amplitude_data: np.ndarray, raw_data: np.ndarray) -> None:
        """Draw the visualization bars and wave line.
        
        Args:
            amplitude_data: Processed amplitude data for bars
            raw_data: Raw audio data for waveform line
        """
        # Clear previous visualization
        self.canvas.delete("bar")
        self.canvas.delete("wave")
        
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 10 or canvas_height <= 10:
            # Canvas too small to draw meaningfully
            return
            
        # Calculate bar width
        bar_width = canvas_width / self.num_bars
        
        # Choose data points evenly spaced through the available data
        if len(amplitude_data) >= self.num_bars:
            indices = np.linspace(0, len(amplitude_data) - 1, self.num_bars).astype(int)
            amplitudes = amplitude_data[indices]
        else:
            # If we don't have enough data points, use what we have
            amplitudes = amplitude_data
            if len(amplitudes) < self.num_bars:
                # Pad with zeros if needed
                amplitudes = np.pad(amplitudes, (0, self.num_bars - len(amplitudes)))
        
        # Calculate display properties
        max_amplitude = max(np.max(amplitudes), 0.01)  # Avoid division by zero
        normalized_amplitudes = amplitudes / max_amplitude
        
        # Draw bars based on the processed amplitudes
        for i, amplitude in enumerate(normalized_amplitudes):
            # Scale to canvas height with minimum height
            bar_height = max(amplitude * canvas_height * 0.8, 2.0)
            
            # Center vertically
            y_top = (canvas_height - bar_height) / 2
            y_bottom = y_top + bar_height
            
            # Calculate horizontal position with spacing
            x_left = i * bar_width
            x_right = x_left + bar_width * self.bar_width_factor
            
            # Draw the bar
            self.canvas.create_rectangle(
                x_left, y_top, x_right, y_bottom,
                fill=self.bar_color,
                width=0,
                tags="bar"
            )
        
        # Draw oscilloscope-like line if we have enough data
        if len(raw_data) > 2:
            self._draw_waveform_line(raw_data, canvas_width, canvas_height)
    
    def _draw_waveform_line(self, audio_data: np.ndarray, width: float, height: float) -> None:
        """Draw an oscilloscope-like line visualization.
        
        Args:
            audio_data: Raw audio data
            width: Canvas width
            height: Canvas height
        """
        points = []
        num_points = min(int(width), len(audio_data))
        step = len(audio_data) // num_points
        
        # Create line points evenly spaced across the canvas
        for i in range(num_points):
            idx = i * step
            if idx < len(audio_data):
                # Normalize to canvas coordinates
                x = i  # Evenly spaced across width
                # Map values from [-1,1] to [0,canvas_height/3] and position in top third
                y = ((-audio_data[idx] + 1) / 2) * height / 3 + height / 3
                points.extend([x, y])
        
        if len(points) >= 4:  # Need at least 2 points (4 coordinates)
            self.canvas.create_line(
                points, 
                fill=self.wave_color,
                width=1.5,
                smooth=True,
                tags="wave"
            )
    
    def _fade_visualization(self) -> None:
        """Fade existing visualization elements for a smooth visual effect."""
        existing_bars = self.canvas.find_withtag("bar")
        for bar_id in existing_bars:
            # Get current color and make it slightly more transparent
            current_color = self.canvas.itemcget(bar_id, "fill")
            if current_color.startswith("#"):
                # Make slightly dimmer
                r = int(current_color[1:3], 16)
                g = int(current_color[3:5], 16)
                b = int(current_color[5:7], 16)
                
                # Reduce intensity by 10%
                r = int(r * 0.9)
                g = int(g * 0.9)
                b = int(b * 0.9)
                
                new_color = f"#{r:02x}{g:02x}{b:02x}"
                self.canvas.itemconfig(bar_id, fill=new_color)


async def async_visualize_audio(
    canvas: tk.Canvas, 
    duration: float = 5.0,
    sink_name: Optional[str] = None
) -> None:
    """Asynchronously visualize audio for a specified duration.
    
    This is a convenience function for applications using async/await patterns.
    
    Args:
        canvas: Tkinter canvas to draw visualization on
        duration: Duration in seconds to visualize (default: 5.0)
        sink_name: Optional PulseAudio sink name
    """
    visualizer = WaveformVisualizer(canvas, sink_name=sink_name)
    visualizer.start()
    
    try:
        await asyncio.sleep(duration)
    finally:
        visualizer.stop()