#!/usr/bin/env python3
"""
Voice Visualizer Test

Simple test script to demonstrate the voice visualizer without
needing to integrate with the full Bella assistant.

This allows for testing the animation, responsiveness, and
appearance of the visualizer independently.
"""

import os
import sys
import asyncio
import numpy as np
import math
import tkinter as tk
from tkinter import ttk

# Add project root to path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Import the visualizer
from Bella.src.ui.voice_visualizer import VoiceVisualizerUI

async def run_test(test_type="demo"):
    """
    Run a test of the voice visualizer.
    
    Args:
        test_type: Type of test to run 
                   ("demo", "amplitude", "phase", "pattern")
    """
    # Create the tkinter window
    # Initialize tkinter before any other operations
    root = tk.Tk()
    # Make it visible immediately to avoid "Too early to create image" error
    root.update()
    
    root.title("Bella Voice Visualizer Test")
    root.geometry("500x600")
    root.configure(bg="#1E1E2E")  # Dark theme background
    
    # Create a frame for the visualizer
    frame = ttk.Frame(root, width=400, height=400)
    frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
    # Create the visualizer
    visualizer = VoiceVisualizerUI(parent=frame, size=(400, 400))
    success = await visualizer.initialize()
    
    if not success:
        print("Failed to initialize visualizer")
        return
    
    # Create controls
    controls_frame = ttk.Frame(root)
    controls_frame.pack(fill=tk.X, padx=20, pady=10)
    
    # Intensity slider
    intensity_var = tk.DoubleVar(value=0.0)
    ttk.Label(controls_frame, text="Voice Intensity:").pack(anchor=tk.W)
    
    intensity_slider = ttk.Scale(
        controls_frame, 
        from_=0.0, 
        to=1.0, 
        orient=tk.HORIZONTAL,
        variable=intensity_var
    )
    intensity_slider.pack(fill=tk.X, pady=5)
    
    # Test selection
    test_frame = ttk.Frame(controls_frame)
    test_frame.pack(fill=tk.X, pady=10)
    
    # Function references for the tests
    test_functions = {
        "demo": run_demo_pattern,
        "amplitude": run_amplitude_test,
        "phase": run_phase_test,
        "pattern": run_speech_pattern
    }
    
    # Set up and run the tests
    await visualizer.start_animation()
    
    # Periodic UI updates for tkinter
    async def update_tk():
        while True:
            try:
                root.update()
                await asyncio.sleep(1/60)  # ~60 fps
            except tk.TclError:
                # Window was closed
                break
            except Exception as e:
                print(f"Error updating UI: {e}")
                break
    
    # Create buttons for each test
    for test_name, test_func in test_functions.items():
        button = ttk.Button(
            test_frame,
            text=f"Run {test_name.title()}",
            command=lambda f=test_func: asyncio.create_task(f(visualizer, intensity_var))
        )
        button.pack(side=tk.LEFT, padx=5)
    
    # Also add a stop button
    stop_button = ttk.Button(
        test_frame,
        text="Stop Test",
        command=lambda: asyncio.create_task(stop_current_test(visualizer, intensity_var))
    )
    stop_button.pack(side=tk.LEFT, padx=5)
    
    # Start with the requested test
    if test_type in test_functions:
        asyncio.create_task(test_functions[test_type](visualizer, intensity_var))
    
    # Run the UI update loop
    await update_tk()


# Test function references
async def stop_current_test(visualizer, intensity_var):
    """Stop any running test patterns."""
    intensity_var.set(0.0)
    visualizer.update_intensity(0.0)


async def run_demo_pattern(visualizer, intensity_var):
    """Run a complete demonstration pattern."""
    # First run the amplitude test
    await run_amplitude_test(visualizer, intensity_var)
    
    # Then run the phase test
    await run_phase_test(visualizer, intensity_var)
    
    # Finally run a speech pattern
    await run_speech_pattern(visualizer, intensity_var)


async def run_amplitude_test(visualizer, intensity_var):
    """Test showing increasing then decreasing amplitude with fine-grained control."""
    # Start at zero
    intensity_var.set(0.0)
    visualizer.update_intensity(0.0)
    await asyncio.sleep(1.0)
    
    # Very slow ramp up focusing on low intensities (0-0.3)
    # This shows the improved responsiveness in subtle voice changes
    print("Testing subtle low-range intensity changes (0.0-0.3)...")
    for i in range(60):
        intensity = i / 200  # Very gradual increase to 0.3
        intensity_var.set(intensity)
        visualizer.update_intensity(intensity)
        await asyncio.sleep(0.05)
    
    # Hold at low-medium intensity
    await asyncio.sleep(1.0)
    
    # Medium range (0.3-0.6)
    print("Testing medium-range intensity changes (0.3-0.6)...")
    for i in range(60):
        intensity = 0.3 + (i / 200)  # Gradual increase from 0.3 to 0.6
        intensity_var.set(intensity)
        visualizer.update_intensity(intensity)
        await asyncio.sleep(0.05)
    
    # Hold at medium intensity
    await asyncio.sleep(1.0)
    
    # High range (0.6-1.0)
    print("Testing high-range intensity changes (0.6-1.0)...")
    for i in range(80):
        intensity = 0.6 + (i / 200)  # Gradual increase from 0.6 to 1.0
        intensity_var.set(intensity)
        visualizer.update_intensity(intensity)
        await asyncio.sleep(0.05)
    
    # Hold at maximum
    await asyncio.sleep(1.0)
    
    # Quickly step down through specific intensity levels to show amplitude levels
    print("Demonstrating discrete amplitude levels...")
    amplitudes = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0]
    for amplitude in amplitudes:
        intensity_var.set(amplitude)
        visualizer.update_intensity(amplitude)
        await asyncio.sleep(0.5)  # Hold each level briefly
    
    # End at zero
    await asyncio.sleep(1.0)


async def run_phase_test(visualizer, intensity_var):
    """Test showing constant amplitude with phase movement."""
    # Set a constant medium intensity
    intensity = 0.5
    intensity_var.set(intensity)
    visualizer.update_intensity(intensity)
    
    # Let it run for a few seconds to show phase movement
    await asyncio.sleep(5.0)


async def run_speech_pattern(visualizer, intensity_var):
    """Simulate a realistic speech pattern with varying intensity."""
    # Pattern parameters
    duration = 20  # seconds
    updates_per_second = 60  # Increased update rate for smoother transitions
    total_updates = duration * updates_per_second
    
    # Speech pattern: multiple sentences with pauses and subtle variations
    # Each sentence has its own envelope with micro-variations
    sentences = [
        {"start": 0.0, "end": 0.2, "peak": 0.3, "type": "soft"},      # Quiet intro
        {"start": 0.25, "end": 0.45, "peak": 0.5, "type": "normal"},  # Medium sentence
        {"start": 0.5, "end": 0.65, "peak": 0.8, "type": "emphasis"}, # Emphasis (louder)
        {"start": 0.68, "end": 0.75, "peak": 0.2, "type": "whisper"}, # Near whisper
        {"start": 0.78, "end": 0.95, "peak": 0.6, "type": "normal"}   # Normal conclusion
    ]
    
    # Function to calculate speech intensity at a given time
    def get_speech_intensity(progress):
        # Default to silence
        base_intensity = 0.0
        
        # Check if we're in a sentence
        for sentence in sentences:
            if sentence["start"] <= progress <= sentence["end"]:
                # Relative position in the sentence
                sentence_progress = (progress - sentence["start"]) / (sentence["end"] - sentence["start"])
                
                # Create a natural envelope shape based on sentence type
                if sentence["type"] == "normal":
                    # Standard rise and fall
                    envelope = math.sin(sentence_progress * math.pi)
                elif sentence["type"] == "emphasis":
                    # Quick rise, sustained, then fall
                    if sentence_progress < 0.2:
                        envelope = sentence_progress * 5  # Quick rise
                    elif sentence_progress > 0.8:
                        envelope = (1 - (sentence_progress - 0.8) * 5)  # Quick fall
                    else:
                        envelope = 1.0  # Sustained
                elif sentence["type"] == "whisper":
                    # Low with subtle variations
                    envelope = 0.7 + 0.3 * math.sin(sentence_progress * math.pi)
                elif sentence["type"] == "soft":
                    # Gentle rise and fall
                    envelope = 0.5 * (1 - math.cos(sentence_progress * math.pi))
                else:
                    envelope = math.sin(sentence_progress * math.pi)
                
                # Apply the envelope to the peak intensity
                intensity = sentence["peak"] * envelope
                
                # Add detailed micro-variations for natural feel:
                
                # 1. Fast syllable variations (5-8 Hz)
                syllable_rate = 6 + sentence_progress * 4  # Accelerate slightly through sentence
                syllable_var = 0.08 * sentence["peak"] * math.sin(sentence_progress * syllable_rate * 2 * math.pi)
                
                # 2. Word-level variations (1-3 Hz)
                word_rate = 2
                word_var = 0.15 * sentence["peak"] * math.sin(sentence_progress * word_rate * 2 * math.pi)
                
                # 3. Ultra-fine detail variations (30-50 Hz) - subtle
                fine_detail = 0.02 * sentence["peak"] * math.sin(sentence_progress * 40 * 2 * math.pi)
                
                # Combine all variations
                base_intensity = intensity + syllable_var + word_var + fine_detail
                break
        
        # Ensure intensity stays in valid range
        return max(0.0, min(1.0, base_intensity))
    
    # Run the pattern
    for i in range(total_updates):
        progress = i / total_updates
        intensity = get_speech_intensity(progress)
        
        intensity_var.set(intensity)
        visualizer.update_intensity(intensity)
        
        # Shorter sleep time for more responsive updates
        await asyncio.sleep(1/updates_per_second)
    
    # End at zero
    intensity_var.set(0.0)
    visualizer.update_intensity(0.0)


if __name__ == "__main__":
    # Determine which test to run based on command line args
    import argparse
    parser = argparse.ArgumentParser(description="Test the Bella voice visualizer")
    parser.add_argument("--test", choices=["demo", "amplitude", "phase", "pattern"],
                      default="demo", help="Test pattern to run")
    args = parser.parse_args()
    
    # Run the specified test
    asyncio.run(run_test(args.test))
