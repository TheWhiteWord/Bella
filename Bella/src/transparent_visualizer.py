#!/usr/bin/env python3
"""
Fully transparent floating visualizer for Bella's voice.

This enhanced version provides a completely transparent background with
only the visual elements visible, removing any borders or background colors.
"""

import os
import sys
import asyncio
import argparse
import platform
import subprocess
import time

def check_display_server():
    """Check if we're running under X11 or Wayland."""
    wayland_display = os.environ.get('WAYLAND_DISPLAY')
    if wayland_display:
        return "wayland"
    else:
        return "x11"

def detect_window_manager():
    """Attempt to detect the window manager being used."""
    desktop_env = os.environ.get('XDG_CURRENT_DESKTOP', 'unknown')
    window_manager = "unknown"
    
    try:
        # Try using wmctrl to get window manager info
        result = subprocess.run(
            ["wmctrl", "-m"], 
            capture_output=True, 
            text=True, 
            check=False
        )
        
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.startswith("Name:"):
                    window_manager = line.split(":", 1)[1].strip()
                    break
    except:
        pass
    
    return desktop_env, window_manager

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the transparent Bella voice visualizer"
    )
    
    parser.add_argument(
        "--size", 
        type=int,
        default=400,
        help="Size of the visualizer in pixels (default: 400)"
    )
    
    parser.add_argument(
        "--position", 
        type=str,
        default=None,
        help="Window position as 'x,y' (default: center of screen)"
    )
    
    parser.add_argument(
        "--always-on-top", 
        action="store_true",
        help="Keep the visualizer always on top of other windows"
    )
    
    parser.add_argument(
        "--no-simulation", 
        action="store_true",
        help="Disable audio simulation (use real audio input)"
    )
    
    parser.add_argument(
        "--fullscreen", 
        action="store_true",
        help="Use the fullscreen transparent overlay version"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug output"
    )
    
    return parser.parse_args()


async def main():
    """Run the transparent visualizer based on platform."""
    args = parse_arguments()
    
    # Check the current display server and window manager
    display_server = check_display_server()
    desktop_env, window_manager = detect_window_manager()
    
    print(f"Display server: {display_server}")
    print(f"Desktop environment: {desktop_env}")
    print(f"Window manager: {window_manager}")
    
    # Process position argument
    position = None
    if args.position:
        try:
            x, y = map(int, args.position.split(','))
            position = (x, y)
        except:
            print(f"Invalid position format: {args.position}, using default")
    
    # Check platform
    if platform.system() == "Linux":
        print("Starting Linux transparent visualizer...")
        
        # Import the appropriate module
        if args.fullscreen:
            print("Using fullscreen transparent mode")
            from ui.fullscreen_transparent_visualizer import FullscreenTransparentVisualizer
            
            # Create visualizer instance
            visualizer = FullscreenTransparentVisualizer(size=args.size, position=position)
            visualizer.use_simulated_audio = not args.no_simulation
            
            # Fine-tune based on window manager if needed
            if "KWin" in window_manager:
                print("Applying KDE Plasma (KWin) optimizations")
                # KWin specific settings would go here
            elif "Mutter" in window_manager:
                print("Applying GNOME (Mutter) optimizations")
                # Mutter specific settings would go here
        else:
            print("Using floating transparent mode")
            from ui.floating_visualizer_linux import FloatingVisualizer
            
            # Create visualizer instance
            visualizer = FloatingVisualizer(size=args.size, position=position)
            visualizer.settings['always_on_top'].set(args.always_on_top)
            visualizer.use_simulated_audio = not args.no_simulation
            
            # Set the transparency color and optimize based on window manager
            if display_server == "wayland":
                print("Optimizing for Wayland")
                visualizer.transparency_color = "#FF01FE"  # Magenta for Wayland
            else:
                print("Optimizing for X11")
                # Different window managers may handle transparency differently
                if "KWin" in window_manager:
                    print("Applying KDE Plasma optimizations")
                    visualizer.transparency_color = "#FF00FF"  # Pure magenta for KWin
                elif "Mutter" in window_manager:
                    print("Applying GNOME optimizations")
                    visualizer.transparency_color = "#FF01FE"  # Slightly off magenta for Mutter
        
        # Run the visualizer
        await visualizer.run()
    else:
        # For other platforms, we'll use the standard visualizer (not implemented yet)
        print(f"Sorry, the visualizer is not yet implemented for {platform.system()}.")
        print("Currently, only Linux is supported.")
        return 1
    
    return 0


if __name__ == "__main__":
    # Add the project root to the path if needed
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    # Run the main function
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
