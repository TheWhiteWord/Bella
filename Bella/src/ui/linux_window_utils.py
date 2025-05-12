"""
Linux-specific window utilities for the Bella voice visualizer.

This module provides helper functions to deal with Linux-specific window
management issues, particularly for transparent, clickable windows.
"""

import os
import subprocess
import tkinter as tk
import tempfile
import traceback

def get_window_id(window):
    """
    Get the X11 window ID for a Tkinter window.
    
    Args:
        window: A Tkinter window instance
    
    Returns:
        str: Window ID or None if not found
    """
    # Ensure window is visible first
    window.update_idletasks()
    
    # Get window ID using xwininfo
    try:
        # Create a temporary file to store the window name
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp:
            temp_path = temp.name
            temp.write(window.wm_title())
        
        # Use xwininfo to find window by name
        cmd = ["xwininfo", "-name", window.wm_title()]
        result = subprocess.check_output(cmd, text=True)
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        # Extract window ID
        for line in result.splitlines():
            if "Window id:" in line:
                return line.split()[3]
        
        return None
    except Exception as e:
        print(f"Warning: Could not get window ID: {e}")
        return None

def set_input_mask(window):
    """
    Set an input shape mask for a window to make it clickable.
    This is important for transparent windows on X11.
    
    Args:
        window: A Tkinter window instance
    """
    try:
        # Get window ID
        window_id = get_window_id(window)
        if not window_id:
            print("Warning: Could not get window ID for input mask")
            return False
        
        # Use xprop to set the input shape mask
        cmd = [
            "xprop", "-id", window_id, 
            "-f", "_NET_WM_WINDOW_TYPE", "32a", 
            "-set", "_NET_WM_WINDOW_TYPE", "_NET_WM_WINDOW_TYPE_NORMAL"
        ]
        subprocess.run(cmd)
        
        # Use xprop to ensure window receives input
        cmd = [
            "xprop", "-id", window_id,
            "-f", "_NET_WM_BYPASS_COMPOSITOR", "32c",
            "-set", "_NET_WM_BYPASS_COMPOSITOR", "1"
        ]
        subprocess.run(cmd)
        
        return True
    except Exception as e:
        print(f"Warning: Could not set input mask: {e}")
        return False

def check_compositor():
    """
    Check if a compositor is running (for transparency effects).
    Prioritizes GNOME/Mutter detection.
    Returns:
        tuple: (bool, str or None) - (True/False if compositor detected, Name of detected compositor)
    """
    # List known compositors/shells embedding them
    # Prioritize gnome-shell/mutter as Pop!_OS uses GNOME
    compositors = [
        "gnome-shell", # Primary target for Pop!_OS/GNOME
        "mutter",      # Standalone or embedded
        "compiz", "compiz.real", # For Unity/Compiz environments
        "kwin_x11", "kwin_wayland", # KDE
        "picom", "compton", # Standalone compositors
        "wayfire",     # Wayland compositor
        "xfwm4",       # XFCE (compositing optional)
        "marco",       # MATE (compositing optional)
        "dwm",         # Often used with picom
        "xcompmgr"     # Older standalone
    ]
    print(f"Checking for compositors/shells: {compositors}")
    detected_compositor = None

    # --- Method 1: pgrep (Often more reliable) ---
    try:
        for compositor in compositors:
            try:
                # Use -f to match full command line, case-insensitive (-i)
                pgrep_cmd = ["pgrep", "-i", "-f", compositor]
                print(f"Running: {' '.join(pgrep_cmd)}")
                result = subprocess.check_output(pgrep_cmd, text=True, stderr=subprocess.DEVNULL).strip()
                # Filter out self-matching grep/pgrep processes
                lines = result.splitlines()
                for line in lines:
                     # Ensure it's not the pgrep process itself or the script
                     if "pgrep" not in line and "grep" not in line and "check_compositor" not in line and compositor in line.lower():
                         detected_compositor = compositor # Found one
                         print(f"Detected running compositor via pgrep: '{detected_compositor}' in line: {line.strip()}")
                         return True, detected_compositor
            except subprocess.CalledProcessError:
                 pass # pgrep returned error (likely no process found)
            except FileNotFoundError:
                print("pgrep command not found, falling back to ps aux.")
                break # Stop trying pgrep if not found
        if not detected_compositor:
            print("pgrep did not find any known running compositors.")
    except Exception as e:
        print(f"Warning: Error during pgrep compositor check: {e}")
        traceback.print_exc()

    # --- Method 2: ps aux (Fallback) ---
    print("Trying compositor check via ps aux...")
    try:
        ps_cmd = ["ps", "aux"]
        print(f"Running: {' '.join(ps_cmd)}")
        result = subprocess.check_output(ps_cmd, text=True, stderr=subprocess.DEVNULL)
        lines = result.splitlines()

        for line in lines:
            # Avoid matching the grep/check_compositor process itself
            if "grep" in line or "check_compositor" in line or "defunct" in line:
                 continue

            parts = line.split(maxsplit=10)
            if len(parts) > 10:
                command_full = parts[10]
                command_base = os.path.basename(command_full.split()[0]) if command_full else ""

                for compositor in compositors:
                    # Check base name (case-insensitive) OR if name is in full command line (case-insensitive)
                    if command_base.lower() == compositor.lower() or compositor.lower() in command_full.lower():
                        detected_compositor = compositor
                        print(f"Detected running compositor (via ps aux): '{detected_compositor}' in line: {line.strip()}")
                        return True, detected_compositor

        print("Could not detect any known running compositor via ps aux.")
        return False, None
    except FileNotFoundError:
         print("ps command not found. Cannot check for compositor.")
         return False, None
    except Exception as e:
        print(f"Warning: Error during ps aux compositor check: {e}")
        traceback.print_exc()
        print("Assuming no compositor due to error during check.")
        return False, None
