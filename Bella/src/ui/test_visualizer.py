#!/usr/bin/env python3
"""
Test script for Bella Voice Visualizer

This script runs the visualizer with simulated audio to test its appearance
and functionality without needing to integrate with Bella's voice system.

Usage:
    python test_visualizer.py
"""

import sys
import os
from PyQt5.QtWidgets import QApplication

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Import the visualizer components
    from voice_visualizer import VoiceVisualizerWindow
except ImportError as e:
    print(f"Error importing visualizer components: {e}")
    print("Make sure you're running this script from the src/ui directory.")
    sys.exit(1)

def main():
    """Main entry point for the test script"""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to wave frames directory
    frames_dir = os.path.join(script_dir, "wave_frames")
    
    # Check if frames directory exists
    if not os.path.exists(frames_dir):
        print(f"ERROR: Wave frames directory not found at {frames_dir}")
        print("You need to generate the wave frames first by running:")
        print("    python wave_frame_generator.py")
        sys.exit(1)
    
    # Check if frames exist in the directory
    frames = [f for f in os.listdir(frames_dir) if f.endswith('.png')]
    if not frames:
        print(f"ERROR: No frame files found in {frames_dir}")
        print("You need to generate the wave frames first by running:")
        print("    python wave_frame_generator.py")
        sys.exit(1)
    
    print(f"Found {len(frames)} wave frames in {frames_dir}")
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Create and configure the visualizer
    size = 300  # Size in pixels
    position = (100, 100)  # Position on screen (x, y)
    
    print(f"Creating visualizer with size {size}x{size} at position {position}")
    visualizer = VoiceVisualizerWindow(
        frames_dir=frames_dir,
        size=size,
        position=position
    )
    
    # Enable speech simulation
    visualizer.audio_simulator.set_speaking(True)
    
    # Show the visualizer
    visualizer.show()
    
    print("Visualizer started with simulated speech.")
    print("- Left-click and drag to move the visualizer")
    print("- Right-click for settings menu")
    print("- Press Esc to exit")
    
    # Run the application event loop
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())
