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
from typing import Optional, Tuple

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer, pyqtSignal, QObject

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
    
    def __init__(self, use_real_audio=False, frames_dir=DEFAULT_FRAMES_DIR,
                 size=DEFAULT_SIZE, position=DEFAULT_POSITION):
        """
        Initialize the Bella Visualizer integration
        
        Args:
            use_real_audio: Whether to use real audio input (vs. simulation)
            frames_dir: Directory containing wave frame images
            size: Size (diameter) of the visualizer window
            position: Position (x, y) of the visualizer window
        """
        self.use_real_audio = use_real_audio
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
        
        if use_real_audio:
            self.audio_processor = AudioProcessor()
            self.audio_integration = BellaAudioIntegration(self.audio_processor)
        
        # Create visualizer window
        self.visualizer = VoiceVisualizerWindow(
            frames_dir=frames_dir,
            size=size,
            position=position
        )
        
        # Connect audio processor to visualizer (if using real audio)
        if self.audio_processor:
            self.audio_processor.intensity_updated.connect(self.visualizer.set_voice_intensity)
            self.audio_processor.start()
            
            # Connect to Bella's audio system asynchronously
            self.async_helper.run_coroutine(
                self.audio_integration.connect_to_bella_audio(),
                self._on_audio_system_connected
            )
    
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
    parser.add_argument("--no-simulation", action="store_true", default=False,
                        help="Disable audio simulation and use real audio input")
    return parser.parse_args()


def main():
    """Main entry point for the Bella Visualizer"""
    # Parse command line arguments
    args = parse_args()
    
    # Parse position
    position = tuple(map(int, args.position.split(',')))
    
    # Create and run visualizer
    bella_visualizer = BellaVisualizer(
        use_real_audio=args.no_simulation,
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
