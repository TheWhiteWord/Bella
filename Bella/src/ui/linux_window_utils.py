"""
Linux-specific window utilities for the Bella voice visualizer.

This module provides helper functions to deal with Linux-specific window
management issues, particularly for transparent, clickable windows.
"""

import os
import subprocess
import tkinter as tk
import tempfile

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
    
    Returns:
        bool: True if a compositor is likely running
    """
    try:
        # Check for common compositors
        compositors = [
            "picom", "compton", "xcompmgr", "kwin_x11", 
            "kwin_wayland", "mutter", "compiz"
        ]
        
        # Use ps to check for running compositors
        result = subprocess.check_output(["ps", "aux"], text=True)
        
        for line in result.splitlines():
            for compositor in compositors:
                if compositor in line and "grep" not in line:
                    return True
        
        return False
    except Exception:
        # Default to assuming compositor is present
        return True
