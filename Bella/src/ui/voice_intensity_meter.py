#!/usr/bin/env python3
"""
Bella Voice Visualizer - Voice Intensity Meter

A simple utility that helps calibrate the voice visualizer by analyzing
the intensity levels of Bella's voice output and providing statistics to
help tune the visualizer parameters.

This tool can be used to:
1. Monitor real-time intensity levels during speech
2. Collect statistics on typical voice levels
3. Help determine appropriate threshold values for the visualizer
"""

import sys
import os
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from typing import List, Deque, Dict, Tuple

# Add parent directory to path to allow importing audio_processor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from audio_processor import AudioProcessor

# Maximum samples to keep in history
MAX_HISTORY = 1000

class IntensityMonitor:
    """
    Monitors and records audio intensity levels over time
    """
    
    def __init__(self, history_size: int = MAX_HISTORY):
        """
        Initialize the intensity monitor
        
        Args:
            history_size: Number of intensity samples to keep in history
        """
        self.intensities: Deque[float] = deque(maxlen=history_size)
        self.timestamps: Deque[float] = deque(maxlen=history_size)
        self.start_time = time.time()
        
        # Statistics
        self.min_intensity = 1.0
        self.max_intensity = 0.0
        self.total_intensity = 0.0
        self.sample_count = 0
        
        # For silence detection
        self.silence_threshold = 0.05
        self.speech_active = False
        self.speech_duration = 0.0
        self.speech_start_time = None
        
    def add_intensity(self, intensity: float):
        """
        Add a new intensity sample
        
        Args:
            intensity: The intensity value (0.0-1.0)
        """
        current_time = time.time() - self.start_time
        
        # Update speech detection
        if intensity > self.silence_threshold:
            if not self.speech_active:
                self.speech_active = True
                self.speech_start_time = current_time
        elif self.speech_active:
            self.speech_active = False
            if self.speech_start_time is not None:
                self.speech_duration += current_time - self.speech_start_time
        
        # Add to history
        self.intensities.append(intensity)
        self.timestamps.append(current_time)
        
        # Update statistics
        self.min_intensity = min(self.min_intensity, intensity)
        self.max_intensity = max(self.max_intensity, intensity)
        self.total_intensity += intensity
        self.sample_count += 1
    
    def get_average_intensity(self) -> float:
        """Get the average intensity across all samples"""
        return self.total_intensity / max(1, self.sample_count)
    
    def get_percentile(self, percentile: float) -> float:
        """Get a specific percentile of intensity values"""
        if not self.intensities:
            return 0.0
        return np.percentile(list(self.intensities), percentile)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistics about the recorded intensities"""
        return {
            "min": self.min_intensity,
            "max": self.max_intensity,
            "average": self.get_average_intensity(),
            "p10": self.get_percentile(10),
            "p25": self.get_percentile(25),
            "p50": self.get_percentile(50),
            "p75": self.get_percentile(75),
            "p90": self.get_percentile(90),
            "speech_duration": self.speech_duration,
            "total_duration": self.timestamps[-1] if self.timestamps else 0
        }
    
    def get_history(self) -> Tuple[List[float], List[float]]:
        """Get the full history of timestamps and intensities"""
        return list(self.timestamps), list(self.intensities)


class IntensityVisualizer:
    """
    Creates a real-time visualization of audio intensity levels
    """
    
    def __init__(self, monitor: IntensityMonitor):
        """
        Initialize the visualizer
        
        Args:
            monitor: IntensityMonitor instance to visualize
        """
        self.monitor = monitor
        
        # Create figure and axes
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.canvas.manager.set_window_title("Bella Voice Intensity Monitor")
        
        # Set up plot
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.set_ylim(0, 1.1)
        self.ax.set_xlim(0, 30)  # Start with 30 seconds window
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("Intensity (0-1)")
        self.ax.set_title("Bella Voice Intensity Monitor")
        self.ax.grid(True)
        
        # Add thresholds
        self.silence_line = self.ax.axhline(
            y=self.monitor.silence_threshold, 
            color='r', linestyle='--', alpha=0.5, 
            label=f"Silence Threshold ({self.monitor.silence_threshold:.2f})"
        )
        
        # Statistics text
        self.stats_text = self.ax.text(
            0.02, 0.95, "", transform=self.ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5)
        )
        
        # Animation
        self.ani = FuncAnimation(
            self.fig, self._update, interval=100,
            init_func=self._init_plot, blit=True
        )
    
    def _init_plot(self):
        """Initialize the plot"""
        self.line.set_data([], [])
        return self.line, self.stats_text, self.silence_line
    
    def _update(self, frame):
        """Update the plot with latest data"""
        timestamps, intensities = self.monitor.get_history()
        if not timestamps:
            return self.line, self.stats_text, self.silence_line
        
        # Update line data
        self.line.set_data(timestamps, intensities)
        
        # Auto-adjust x-axis
        x_max = max(30, max(timestamps) + 5)
        if x_max > self.ax.get_xlim()[1]:
            self.ax.set_xlim(0, x_max)
        
        # Update statistics text
        stats = self.monitor.get_statistics()
        stats_str = (
            f"Min: {stats['min']:.3f}\n"
            f"Max: {stats['max']:.3f}\n"
            f"Avg: {stats['average']:.3f}\n"
            f"P50: {stats['p50']:.3f}\n"
            f"P90: {stats['p90']:.3f}\n"
            f"Speech: {stats['speech_duration']:.1f}s"
        )
        self.stats_text.set_text(stats_str)
        
        return self.line, self.stats_text, self.silence_line
    
    def show(self):
        """Show the visualization window"""
        plt.show()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Bella Voice Intensity Monitor")
    parser.add_argument("--simulate", action="store_true", default=False,
                        help="Use simulated audio instead of real audio")
    args = parser.parse_args()
    
    # Create intensity monitor
    monitor = IntensityMonitor()
    
    # Set up audio processor
    processor = AudioProcessor(smoothing_factor=0.2)
    
    # Connect processor to monitor
    processor.intensity_updated.connect(monitor.add_intensity)
    
    # Create visualizer
    visualizer = IntensityVisualizer(monitor)
    
    # Start processor
    processor.start()
    
    # If using simulation, create simulated audio
    if args.simulate:
        import threading
        from PyQt5.QtCore import QCoreApplication, QTimer
        
        # Create Qt application for signals
        app = QCoreApplication([])
        
        # Create audio simulation in a thread
        def audio_simulation():
            # Import here to avoid circular import
            from voice_visualizer import AudioSimulator
            
            simulator = AudioSimulator()
            simulator.intensity_updated.connect(processor._on_intensity_update)
            simulator.start_simulation()
            simulator.set_speaking(True)
            
            # Run for a limited time
            QTimer.singleShot(120000, app.quit)  # 2 minutes
            app.exec_()
        
        # Start simulation thread
        threading.Thread(target=audio_simulation, daemon=True).start()
    
    # Show visualizer (blocks until window is closed)
    visualizer.show()
    
    # Clean up
    processor.stop()
    
    # Print final statistics
    stats = monitor.get_statistics()
    print("\nVoice Intensity Statistics:")
    print(f"Total duration: {stats['total_duration']:.1f} seconds")
    print(f"Speech duration: {stats['speech_duration']:.1f} seconds")
    print(f"Minimum intensity: {stats['min']:.4f}")
    print(f"Maximum intensity: {stats['max']:.4f}")
    print(f"Average intensity: {stats['average']:.4f}")
    print(f"Median (P50): {stats['p50']:.4f}")
    print(f"P10: {stats['p10']:.4f}")
    print(f"P25: {stats['p25']:.4f}")
    print(f"P75: {stats['p75']:.4f}")
    print(f"P90: {stats['p90']:.4f}")
    
    print("\nRecommended settings for audio_processor.py:")
    print(f"silence_threshold = {stats['p10']:.4f}")
    print(f"normal_speech = {stats['p50']:.4f}")
    print(f"loud_speech = {stats['p90']:.4f}")


if __name__ == "__main__":
    main()
