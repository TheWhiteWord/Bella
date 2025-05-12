#!/usr/bin/env python3
"""
Wave Frame Generator for Audio Visualizer

This script generates a series of frames with:
1. Varying wave amplitudes (0-60) for intensity visualization
2. Phase shifts for each amplitude to show horizontal movement
3. Constant wave period across all frames

This combination allows for a complete audio visualization where:
- Amplitude changes represent sound intensity
- Horizontal movement creates a dynamic flowing effect

The output is a collection of PNG frames that can be imported and animated
in other Python applications.
"""
import math
import os
import numpy as np
from PIL import Image, ImageEnhance, ImageChops
import asyncio
from pathlib import Path

# --- Configuration parameters ---
# File paths - updated to use correct element paths
ELEMENTS_DIR = os.path.join(os.path.dirname(__file__), "elements")
WAVE_SOURCE_PATH = os.path.join(ELEMENTS_DIR, "wave.png")  # Source wave image
FRAME_PATH = os.path.join(ELEMENTS_DIR, "frame.png")       # Optional frame image
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "wave_frames")  # Directory to save frames

# Wave parameters
RIPPLE_PERIOD = 41.4           # Constant period for all frames
RIPPLE_INITIAL_PHASE = 0.593   # Starting phase value
MAX_AMPLITUDE = 90.0           # Increased maximum amplitude for stronger visual effect
AMPLITUDE_STEPS = 24           # Increased number of amplitude steps for more responsive visualization

# Phase shift parameters
PHASE_STEPS = 12               # Number of phase shifts per amplitude (horizontal movement frames)
MAX_PHASE_SHIFT = 2 * math.pi  # Full 360° phase cycle

# Wave appearance
WAVE_OPACITY = 0.55            # Opacity of the wave effect (0.0 to 1.0)

# --- Helper Functions ---
def apply_ripple_effect(image, amplitude, period, phase_shift):
    """
    Apply a horizontal ripple/wave effect to an image
    
    Args:
        image: PIL Image in RGBA format
        amplitude: Wave height (0 = no wave, higher = more intense wave)
        period: Wave length/frequency
        phase_shift: Wave position offset
        
    Returns:
        PIL Image with the ripple effect applied
    """
    img_rgba = image.convert("RGBA")  # Ensure RGBA format
    img_arr = np.array(img_rgba)
    height, width = img_arr.shape[:2]
    
    # Create coordinate grid and apply sine wave distortion
    y_coords, x_coords = np.ogrid[:height, :width]
    y_displacement = amplitude * np.sin(2 * np.pi * x_coords / period + phase_shift)
    y_source_float = y_coords - y_displacement
    
    # Clip coordinates to valid image range and convert to integers
    y_source = np.clip(y_source_float, 0, height - 1).astype(int)
    x_source = x_coords
    
    # Sample the image at the new coordinates
    output_arr = img_arr[y_source, x_source]
    return Image.fromarray(output_arr, 'RGBA')

def set_opacity(image, opacity_factor):
    """
    Adjust the opacity/transparency of an RGBA image
    
    Args:
        image: PIL Image
        opacity_factor: Opacity multiplier (0.0 to 1.0)
        
    Returns:
        PIL Image with adjusted opacity
    """
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
        
    alpha = image.split()[-1]
    alpha = ImageEnhance.Brightness(alpha).enhance(opacity_factor)
    
    image_copy = image.copy()
    image_copy.putalpha(alpha)
    return image_copy

def screen_blend(base_img, top_img):
    """
    Apply screen blending mode between two images
    
    Args:
        base_img: PIL Image (background)
        top_img: PIL Image (overlay)
        
    Returns:
        PIL Image with screen blend applied
    """
    base_rgb = base_img.convert("RGB")
    top_rgb = top_img.convert("RGB")
    
    # Apply screen blend
    blended_rgb = ImageChops.screen(base_rgb, top_rgb)
    result = blended_rgb.convert("RGBA")
    
    # Preserve alpha from top image
    result.putalpha(top_img.split()[-1])
    return result

async def generate_frames(
    wave_img_path, 
    frame_img_path=None, 
    amplitudes=None, 
    phases=None,
    period=RIPPLE_PERIOD, 
    output_dir=OUTPUT_DIR
):
    """
    Generate a series of wave animation frames with varying amplitudes and phase shifts
    
    Args:
        wave_img_path: Path to the wave source image
        frame_img_path: Optional path to a frame image (can be None)
        amplitudes: List of amplitude values to generate frames for
        phases: List of phase shift values to generate frames for each amplitude
        period: Wave period/frequency (constant across all frames)
        output_dir: Directory to save output frames
        
    Returns:
        Dictionary mapping amplitude values to lists of frame paths at different phases
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the wave source image
    print(f"Loading wave source image from {wave_img_path}")
    wave_source = Image.open(wave_img_path).convert("RGBA")
    
    # Load the frame image if provided
    frame = None
    if frame_img_path and os.path.exists(frame_img_path):
        print(f"Loading frame image from {frame_img_path}")
        frame = Image.open(frame_img_path).convert("RGBA")
        
        # Resize wave to match frame if needed
        if frame.size != wave_source.size:
            print(f"Resizing wave image from {wave_source.size} to {frame.size}")
            wave_source = wave_source.resize(frame.size, Image.Resampling.LANCZOS)
    
    # Use default amplitudes if none provided
    if amplitudes is None:
        amplitudes = np.linspace(0, MAX_AMPLITUDE, AMPLITUDE_STEPS)
    
    # Use default phases if none provided
    if phases is None:
        phases = np.linspace(0, MAX_PHASE_SHIFT, PHASE_STEPS)
    
    # Dictionary to store frame paths organized by amplitude
    frame_paths = {amplitude: [] for amplitude in amplitudes}
    total_frames = len(amplitudes) * len(phases)
    frame_counter = 0
    
    # Generate frames for each amplitude and phase combination
    for i, amplitude in enumerate(amplitudes):
        for j, phase in enumerate(phases):
            frame_counter += 1
            
            # Create wave effect with current amplitude and phase
            rippled_wave = apply_ripple_effect(
                wave_source, 
                amplitude=amplitude,
                period=period,
                phase_shift=RIPPLE_INITIAL_PHASE + phase
            )
            
            # Apply opacity to the wave layer
            rippled_wave_with_opacity = set_opacity(rippled_wave, WAVE_OPACITY)
            
            # Composite with frame if available, otherwise just use the wave
            if frame:
                # Blend wave with frame using screen blend
                wave_effect = screen_blend(frame, rippled_wave_with_opacity)
                final_image = Image.alpha_composite(frame.copy(), wave_effect)
            else:
                final_image = rippled_wave_with_opacity
            
            # Save the frame - include both amplitude and phase in filename
            frame_filename = f"wave_frame_a{amplitude:.1f}_p{j:02d}.png"
            frame_path = os.path.join(output_dir, frame_filename)
            final_image.save(frame_path)
            
            # Store frame path in the dictionary
            frame_paths[amplitude].append(frame_path)
            
            # Print progress
            if frame_counter % 10 == 0 or frame_counter == total_frames:
                print(f"Generated frame {frame_counter}/{total_frames}: amplitude={amplitude:.1f}, phase={j+1}/{len(phases)}")
            
            # Yield control to allow other async operations
            await asyncio.sleep(0)
    
    print(f"\nGenerated {total_frames} frames in {output_dir}")
    return frame_paths

async def generate_amplitude_phase_matrix():
    """
    Generate a complete matrix of frames for all amplitude and phase combinations.
    This creates a comprehensive set of wave states that can be selected based on:
    1. Amplitude (vertical intensity of the wave)
    2. Phase (horizontal position of the wave)
    
    Returns:
        Dictionary mapping amplitude values to lists of frames at different phases
    """
    # Create a non-linear distribution of amplitudes with more detail in lower ranges
    # This makes the visualizer more responsive to subtle voice changes
    
    # More detailed low-range amplitudes (0-30)
    low_range = np.linspace(0, 30, 12)
    
    # Medium range with less density (30-60)
    mid_range = np.linspace(35, 55, 6)[1:]  # Skip first to avoid duplication
    
    # High range (60-90)
    high_range = np.linspace(60, 90, 6)
    
    # Combine all ranges
    amplitudes = np.concatenate([low_range, mid_range, high_range])
    
    # Generate phases for a complete wave cycle (0 to 2π)
    phases = np.linspace(0, MAX_PHASE_SHIFT, PHASE_STEPS)
    
    # Generate frames for all amplitude/phase combinations
    return await generate_frames(
        wave_img_path=WAVE_SOURCE_PATH,
        frame_img_path=FRAME_PATH if os.path.exists(FRAME_PATH) else None,
        amplitudes=amplitudes,
        phases=phases
    )

async def main():
    """
    Main function to generate the complete matrix of amplitude/phase frames
    """
    print("Generating amplitude/phase matrix for wave visualization...")
    print(f"- Amplitudes: {AMPLITUDE_STEPS} levels (0 to {MAX_AMPLITUDE})")
    print(f"- Phases: {PHASE_STEPS} positions per amplitude")
    print(f"- Total frames: {AMPLITUDE_STEPS * PHASE_STEPS}")
    
    # Generate frames with all amplitude/phase combinations
    frame_matrix = await generate_amplitude_phase_matrix()
    
    # Count total frames generated
    total_frames = sum(len(phases) for phases in frame_matrix.values())
    
    # Print information on how to use the frames
    print("\n✅ Frame generation complete!")
    print(f"Generated {total_frames} frames across {len(frame_matrix)} amplitude levels")
    print(f"Each amplitude has {PHASE_STEPS} phase positions for horizontal animation")
    print(f"Frames saved to: {OUTPUT_DIR}/")
    
    # Example code for using the frames
    print("\nExample code to use these frames in your visualization:")
    print("""
import os
import re
import numpy as np
from PIL import Image

# Load all frames from directory and organize by amplitude and phase
def load_wave_frames(frames_dir="wave_frames"):
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    frames_by_amplitude = {}
    
    # Parse amplitude and phase from filenames (wave_frame_a30.0_p05.png)
    for filename in frame_files:
        match = re.search(r'a(\d+\.\d+)_p(\d+)', filename)
        if match:
            amplitude = float(match.group(1))
            phase_idx = int(match.group(2))
            
            if amplitude not in frames_by_amplitude:
                frames_by_amplitude[amplitude] = {}
                
            frames_by_amplitude[amplitude][phase_idx] = os.path.join(frames_dir, filename)
    
    return frames_by_amplitude

# Load frame matrix
frame_matrix = load_wave_frames()
amplitudes = sorted(frame_matrix.keys())

# Function to get the appropriate frame based on audio properties
def get_frame(intensity, time_position, max_intensity=1.0):
    # Map intensity to amplitude
    amplitude = min(amplitudes, key=lambda a: abs(a - intensity * 90 / max_intensity))
    
    # Select phase based on time position (creates continuous movement)
    phase_positions = len(frame_matrix[amplitude])
    phase_idx = int(time_position * phase_positions) % phase_positions
    
    # Load and return the appropriate frame
    frame_path = frame_matrix[amplitude][phase_idx]
    return Image.open(frame_path)

# Example usage in an audio visualization:
# current_time = 0
# for audio_chunk in audio_stream:
#     # Get amplitude from audio (simplified example)
#     intensity = np.abs(audio_chunk).mean()
#     
#     # Update time position (for phase selection)
#     current_time += chunk_duration_seconds
#     
#     # Get the appropriate frame
#     frame = get_frame(intensity, current_time)
#     
#     # Display the frame in your UI
#     display_frame(frame)
    """)

if __name__ == "__main__":
    asyncio.run(main())
