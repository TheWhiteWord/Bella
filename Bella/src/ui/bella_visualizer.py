#!/usr/bin/env python3
"""
Bella Voice Visualizer - Main Integration Script

This script integrates the Bella Voice Visualizer with the voice assistant,
providing a visual representation of Bella's voice through a floating window.

Usage:
    python bella_visualizer.py [options]

Options:
    --size SIZE           Set the size (diameter) of the visualizer in pixels
    --position X,Y        Position the visualizer at specific screen coordinates
    --always-on-top       Keep the visualizer on top of other windows (default)
    --frames-dir DIR      Directory containing wave frame images
    --no-simulation       Disable audio simulation and use real audio input
"""

import sys
import os
import argparse
import asyncio
import threading

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import pyqtSignal, QObject

# Import visualizer components
from voice_visualizer import VoiceVisualizerWindow, DEFAULT_SIZE, DEFAULT_POSITION
from audio_processor import AudioProcessor, BellaAudioIntegration

# Set default paths
DEFAULT_FRAMES_DIR = os.path.join(os.path.dirname(__file__), "wave_frames")

class AsyncHelper(QObject):
    """Helper class to run asyncio methods from a Qt application"""
    
    finished = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
    
    def _run_loop(self):
        """Run the asyncio event loop in a separate thread"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    
    def run_coroutine(self, coro, callback=None):
        """
        Run a coroutine in the asyncio event loop
        
        Args:
            coro: Coroutine to run
            callback: Optional callback to call with the result
        """
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        
        if callback:
            future.add_done_callback(
                lambda fut: self.finished.emit((callback, fut.result()))
            )
        
        return future
    
    def stop(self):
        """Stop the asyncio event loop"""
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=1.0)


class BellaVisualizer:
    """
    Main class that integrates the voice visualizer with Bella's audio system
    """
    
    def __init__(self, frames_dir=DEFAULT_FRAMES_DIR,
                 size=DEFAULT_SIZE, position=DEFAULT_POSITION):
        """
        Initialize the Bella Visualizer integration
        
        Args:
            frames_dir: Directory containing wave frame images
            size: Size (diameter) of the visualizer window
            position: Position (x, y) of the visualizer window
        """
        self.frames_dir = frames_dir
        self.size = size
        self.position = position
        
        # Create application
        self.app = QApplication(sys.argv)
        
        # Create async helper
        self.async_helper = AsyncHelper()
        self.async_helper.finished.connect(self._handle_async_result)
        
        # Create audio processor (if using real audio)
        self.audio_processor = None
        self.audio_integration = None
        
        self.system_audio_listener = None

        from system_audio_listener import SystemAudioListener
        self.audio_processor = AudioProcessor()
        self.audio_integration = None  # Not using BellaAudioIntegration for system audio

        # Determine monitor source
        monitor_source = os.environ.get("BELLA_MONITOR_SOURCE", None)
        if not monitor_source or monitor_source == "auto":
            # Try to auto-detect or instruct user
            print("\n[Visualizer] To capture system audio, set the monitor source.\n" \
                  "You can find your monitor source with:\n  pactl list short sources | grep monitor\n" \
                  "Set BELLA_MONITOR_SOURCE env var or pass it in code.\n" \
                  "Defaulting to first available monitor source.\n")
            # Try to auto-detect
            try:
                import subprocess
                out = subprocess.check_output(["pactl", "list", "short", "sources"]).decode()
                lines = [l for l in out.splitlines() if ".monitor" in l]
                if lines:
                    monitor_source = lines[0].split()[1]
                    print(f"[Visualizer] Using monitor source: {monitor_source}")
                else:
                    print("[Visualizer] No monitor source found! System audio will not be captured.")
                    monitor_source = None
            except Exception as e:
                print(f"[Visualizer] Could not auto-detect monitor source: {e}")
                monitor_source = None

        # Create system audio listener if monitor source is available
        if monitor_source:
            # Use lower latency chunk size and pass intensity directly to visualizer
            def intensity_callback(audio_chunk):
                import numpy as np
                if len(audio_chunk) == 0:
                    intensity = 0.0
                else:
                    rms = np.sqrt(np.mean(np.square(audio_chunk)))
                    intensity = min(max(rms, 0.0), 1.0)
                sensitivity = getattr(self.visualizer, 'intensity_sensitivity', 2.0)
                scaled_intensity = intensity ** (1 / sensitivity)
                scaled_intensity = min(max(scaled_intensity, 0.0), 1.0)
                self.visualizer.set_voice_intensity(scaled_intensity)

            self.system_audio_listener = SystemAudioListener(
                monitor_source=monitor_source,
                sample_rate=48000,
                chunk_size=1024,
                callback=intensity_callback
            )
            print("[Visualizer] SystemAudioListener initialized.")
        else:
            print("[Visualizer] System audio input is not available. Visualizer will not react to real audio.")

        # Create visualizer window
        self.visualizer = VoiceVisualizerWindow(
            frames_dir=frames_dir,
            size=size,
            position=position
        )

        # Start system audio listener if using real audio
        if self.system_audio_listener:
            self.system_audio_listener.start()
            print("[Visualizer] System audio listener started.")
    
    def _handle_async_result(self, result):
        """Handle results from async operations"""
        callback, data = result
        callback(data)
    
    def _on_audio_system_connected(self, success):
        """Handle connection to Bella's audio system"""
        if success:
            print("Successfully connected to Bella's audio system")
        else:
            print("Failed to connect to Bella's audio system, falling back to simulation")
            # Enable simulation mode
            self.visualizer.audio_simulator.set_speaking(True)
    
    def show(self):
        """Show the visualizer window"""
        self.visualizer.show()
    
    def run(self):
        """Run the application event loop"""
        exit_code = self.app.exec_()

        # Clean up
        if self.audio_processor:
            self.audio_processor.stop()

        if self.audio_integration:
            self.audio_integration.disconnect()

        if self.system_audio_listener:
            self.system_audio_listener.stop()
            print("[Visualizer] System audio listener stopped.")

        self.async_helper.stop()

        return exit_code


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Bella Voice Visualizer")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE,
                        help=f"Size of the visualizer in pixels (default: {DEFAULT_SIZE})")
    parser.add_argument("--position", type=str, default=f"{DEFAULT_POSITION[0]},{DEFAULT_POSITION[1]}",
                        help=f"Position of the visualizer as X,Y (default: {DEFAULT_POSITION[0]},{DEFAULT_POSITION[1]})")
    parser.add_argument("--always-on-top", action="store_true", default=True,
                        help="Keep the visualizer on top of other windows")
    parser.add_argument("--frames-dir", type=str, default=DEFAULT_FRAMES_DIR,
                        help="Directory containing wave frame images")
    return parser.parse_args()


def main():
    """Main entry point for the Bella Visualizer"""
    # Parse command line arguments
    args = parse_args()
    
    # Parse position
    position = tuple(map(int, args.position.split(',')))
    
    # Create and run visualizer
    bella_visualizer = BellaVisualizer(
        frames_dir=args.frames_dir,
        size=args.size,
        position=position
    )
    
    # Show the visualizer window
    bella_visualizer.show()
    
    # Run the application
    return bella_visualizer.run()


if __name__ == "__main__":
    sys.exit(main())
