#!/usr/bin/env python3
"""
Linux-optimized floating visualizer for Bella's voice.

This is a variant of the floating_visualizer.py script that includes
specific optimizations for Linux window managers to improve transparency
and click-through behavior.
"""

import os
import sys
import time
import math
import asyncio
import subprocess
import tkinter as tk
from tkinter import ttk
import colorsys
import platform
import signal
import numpy as np
from PIL import Image, ImageTk
from typing import Optional, Dict, Tuple, Any
import numpy as np

# Add the project root to the path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Import required modules from the enhanced visualizer
from Bella.src.ui.enhanced_voice_visualizer import (
    EnhancedVoiceVisualizerUI,
    DEFAULT_HUE_SHIFT_SPEED,
    DEFAULT_SATURATION_FACTOR
)


def get_x11_window_id(window_title):
    """
    Try to get the X11 window ID for a window with the given title.
    
    Args:
        window_title: The title of the window to look for
        
    Returns:
        The window ID as a string, or None if not found
    """
    try:
        # Use xwininfo to find window by name
        cmd = ["xwininfo", "-name", window_title]
        result = subprocess.check_output(cmd, text=True)
        
        # Extract window ID
        for line in result.splitlines():
            if "Window id:" in line:
                return line.split()[3]
        return None
    except Exception as e:
        print(f"Error getting window ID: {e}")
        return None


def apply_x11_window_properties(window_id):
    """
    Apply X11-specific window properties to make the window more usable.
    
    Args:
        window_id: The X11 window ID
    """
    if not window_id:
        return False
        
    try:
        # Set window type to normal to ensure proper input handling
        subprocess.run([
            "xprop", "-id", window_id, 
            "-f", "_NET_WM_WINDOW_TYPE", "32a", 
            "-set", "_NET_WM_WINDOW_TYPE", "_NET_WM_WINDOW_TYPE_NORMAL"
        ], check=False)
        
        # Bypass compositor to improve performance
        subprocess.run([
            "xprop", "-id", window_id,
            "-f", "_NET_WM_BYPASS_COMPOSITOR", "32c",
            "-set", "_NET_WM_BYPASS_COMPOSITOR", "1"
        ], check=False)
        
        # Allow direct input to window
        subprocess.run([
            "xprop", "-id", window_id,
            "-f", "WM_HINTS", "32c",
            "-set", "WM_HINTS", "0x2, 0x0, 0x1"
        ], check=False)
        
        print(f"Applied X11 properties to window {window_id}")
        return True
    except Exception as e:
        print(f"Error applying X11 properties: {e}")
        return False


class AudioMonitorMock:
    """
    Mock audio monitor for testing and development.
    In production, this would be replaced with a real audio monitor.
    """
    
    def __init__(self):
        """Initialize the mock audio monitor."""
        self.intensity = 0.0
        self.running = False
        
    def start(self):
        """Start the mock audio monitor."""
        self.running = True
        
    def stop(self):
        """Stop the mock audio monitor."""
        self.running = False
        
    def get_current_intensity(self):
        """Get the current audio intensity."""
        return self.intensity
    
    def set_intensity(self, value):
        """Set the current audio intensity (for manual testing)."""
        self.intensity = max(0.0, min(1.0, value))


class FloatingVisualizer:
    """
    Linux-optimized floating visualizer with improved click-through handling.
    """
    
    def __init__(self, size=400, position=None):
        """
        Initialize the floating visualizer.
        
        Args:
            size: Size of the visualizer (diameter in pixels)
            position: Optional starting position (x, y) on screen
        """
        self.size = size
        self.position = position or (100, 100)
        
        # Set a unique window title for identification
        self.window_title = f"BellaVisualizer-{int(time.time())}"
        
        # Special color for transparency - use a very specific shade of magenta
        self.transparency_color = "#FF01FE"
        
        # Create main window
        self.root = tk.Tk()
        self.root.title(self.window_title)
        
        # Make window borderless
        self.root.overrideredirect(True)
        
        # Position the window
        self.root.geometry(f"{size}x{size}+{self.position[0]}+{self.position[1]}")
        
        # Configure window for Linux window managers
        self._configure_for_linux_wm()
        
        # Configure window transparency
        self._configure_transparency()
        
        # Create a frame container with transparent background
        self.frame = tk.Frame(self.root, background=self.transparency_color, bd=0, highlightthickness=0)
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        # Add a thin border frame to make dragging easier
        self._create_drag_border()
        
        # Create the audio monitor (mock for now)
        self.audio_monitor = AudioMonitorMock()
        
        # Create the visualizer
        self.visualizer = None  # Will be initialized later
        
        # Track mouse position for dragging
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.window_x = 0
        self.window_y = 0
        self.is_dragging = False
        
        # Set up right-click context menu
        self.context_menu = tk.Menu(self.root, tearoff=0)
        
        # Control variables
        self.settings = {
            'hue_shift_speed': tk.DoubleVar(value=DEFAULT_HUE_SHIFT_SPEED),
            'saturation': tk.DoubleVar(value=DEFAULT_SATURATION_FACTOR),
            'wave_hue_offset': tk.DoubleVar(value=0.5),
            'screen_opacity': tk.DoubleVar(value=0.6),
            'screen_enabled': tk.BooleanVar(value=True),
            'wave_color_enabled': tk.BooleanVar(value=True),
            'always_on_top': tk.BooleanVar(value=True),  # Default to always on top
        }
        
        # Audio simulation
        self.use_simulated_audio = True
        self.simulation_task = None
        
        # Window ID for X11-specific operations
        self.window_id = None
        
        # Set up signal handlers for graceful exit
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _configure_for_linux_wm(self):
        """Apply Linux-specific window manager configurations."""
        # Set window type - try different types that might work
        try:
            # First try 'dock' window type - more likely to be click-through on Linux
            self.root.attributes("-type", "dock")
            print("Using dock window type")
        except Exception as e:
            print(f"Could not set dock window type: {e}")
            try:
                # If 'dock' fails, try 'utility' which is usually borderless too
                self.root.attributes("-type", "utility")
                print("Using utility window type")
            except Exception as e2:
                print(f"Could not set utility window type: {e2}")
                try:
                    # Last resort: dialog type
                    self.root.attributes("-type", "dialog")
                    print("Using dialog window type")
                except Exception as e3:
                    print(f"Could not set dialog window type: {e3}")
                
        # Set window manager class - older method, may work on some WMs
        try:
            # This doesn't work with newer Tk versions, but we'll try anyway
            self.root.attributes("-class", "BellaFloatingVisualizer")
            print("Set window class via attributes")
        except Exception as e:
            print(f"Could not set window class via attributes: {e}")
            
        # Try to enforce a borderless appearance
        try:
            self.root.attributes("-toolwindow", True)
            print("Set toolwindow attribute")
        except Exception as e:
            print(f"Could not set toolwindow attribute: {e}")
    
    def _create_drag_border(self):
        """Create an invisible border around the window to make dragging easier."""
        # Use invisible border frames on all sides
        border_color = self.transparency_color  # Use our transparent color
        border_width = 2
        
        # Top border - main drag handle, completely transparent
        self.top_border = tk.Frame(self.root, bg=border_color, height=border_width)
        self.top_border.place(x=0, y=0, relwidth=1.0)
        
        # Bottom border
        self.bottom_border = tk.Frame(self.root, bg=border_color, height=border_width)
        self.bottom_border.place(x=0, rely=1.0, relwidth=1.0, y=-border_width)
        
        # Left border
        self.left_border = tk.Frame(self.root, bg=border_color, width=border_width)
        self.left_border.place(x=0, y=0, relheight=1.0)
        
        # Right border
        self.right_border = tk.Frame(self.root, bg=border_color, width=border_width)
        self.right_border.place(relx=1.0, y=0, relheight=1.0, x=-border_width)
        
        # Bind drag events to all borders
        for border in [self.top_border, self.bottom_border, self.left_border, self.right_border]:
            border.bind("<Button-1>", self._start_drag)
            border.bind("<ButtonRelease-1>", self._stop_drag)
            border.bind("<B1-Motion>", self._on_drag)
            border.bind("<Button-3>", self._show_context_menu)
    
    def _configure_transparency(self):
        """Configure window transparency for Linux."""
        # Configure for Linux - use a special color that won't appear in our UI
        # We'll use a very specific shade of magenta that we can make transparent
        transparency_color = "#FF01FE"  # Special magenta for transparency
        
        # Set background color for root and frame
        self.root.config(bg=transparency_color)
        
        try:
            # Wait for window to be visible
            self.root.wait_visibility(self.root)
            
            # Remove window decorations entirely
            self.root.overrideredirect(True)
            
            # Make the window always on top initially
            self.root.attributes("-topmost", True)
            
            # Try different transparency approaches
            transparency_success = False
            
            # Try method 1: transparentcolor with our special color
            try:
                self.root.attributes("-transparentcolor", transparency_color)
                transparency_success = True
                print("Using -transparentcolor method for transparency")
            except Exception as e:
                print(f"Could not use transparentcolor: {e}")
            
            # Try method 2: shape extension via wm_attributes if available
            if not transparency_success:
                try:
                    # Some window managers support shape mask
                    self.root.wm_attributes("-type", "splash")  # Override type to splash for borderless
                    self.root.wm_attributes("-transparentcolor", transparency_color)
                    transparency_success = True
                    print("Using wm_attributes method for transparency")
                except Exception as e:
                    print(f"Could not use wm_attributes for transparency: {e}")
            
            # Try method 3: use X11 window properties if available
            if self.window_id:
                try:
                    # X11 transparency hint
                    subprocess.run([
                        "xprop", "-id", self.window_id,
                        "-f", "_NET_WM_WINDOW_OPACITY", "32c",
                        "-set", "_NET_WM_WINDOW_OPACITY", "0xf0000000"
                    ], check=False)
                    print(f"Applied X11 opacity property to window {self.window_id}")
                except Exception as e:
                    print(f"Could not apply X11 properties: {e}")
                
        except Exception as e:
            print(f"Warning: Could not set all transparency attributes: {e}")
    
    def _signal_handler(self, sig, frame):
        """Handle termination signals."""
        print("Shutting down visualizer...")
        if self.simulation_task:
            self.simulation_task.cancel()
        self.root.quit()
        sys.exit(0)
    
    def _start_drag(self, event):
        """Start window dragging operation."""
        # Use screen coordinates for more reliable dragging
        self.drag_start_x = event.x_root
        self.drag_start_y = event.y_root
        self.window_x = self.root.winfo_x()
        self.window_y = self.root.winfo_y()
        self.is_dragging = True
        
        # Change cursor to indicate dragging
        self.root.config(cursor="fleur")
        
        # Stop event propagation to prevent canvas from receiving the event
        return "break"
    
    def _stop_drag(self, event):
        """Stop window dragging operation."""
        self.is_dragging = False
        self.root.config(cursor="")
        
        # Stop event propagation
        return "break"
    
    def _on_drag(self, event):
        """Handle window dragging to reposition the window."""
        if not self.is_dragging:
            return "break"
            
        # Calculate the distance moved
        delta_x = event.x_root - self.drag_start_x
        delta_y = event.y_root - self.drag_start_y
        
        # Move window to new position
        new_x = self.window_x + delta_x
        new_y = self.window_y + delta_y
        
        # Set new position
        self.root.geometry(f"+{new_x}+{new_y}")
        
        # Update UI immediately for better responsiveness
        self.root.update_idletasks()
        
        # Stop event propagation
        return "break"
    
    def _setup_bindings(self):
        """Set up mouse bindings for dragging and context menu."""
        # Bind events directly to all necessary components
        self.root.bind("<Button-1>", self._start_drag)
        self.root.bind("<ButtonRelease-1>", self._stop_drag)
        self.root.bind("<B1-Motion>", self._on_drag)
        self.root.bind("<Button-3>", self._show_context_menu)
        self.root.bind("<Double-Button-1>", self._toggle_settings_panel)
        
        # Keyboard shortcuts
        self.root.bind("<Escape>", lambda e: self.root.quit())
        self.root.bind("<Control-q>", lambda e: self.root.quit())
        
        # Print helpful information
        print("Keyboard shortcuts:")
        print("- ESC or Ctrl+Q: Exit the visualizer")
        print("- Right-click: Open settings menu")
        print("- Left-click and drag: Move the visualizer")
        print("- Double-click: Open detailed settings panel")
    
    def _setup_context_menu(self):
        """Set up the right-click context menu with settings."""
        # Clear any existing menu items
        self.context_menu.delete(0, tk.END)
        
        # Add settings submenu
        settings_menu = tk.Menu(self.context_menu, tearoff=0)
        
        # Hue shift speed options
        hue_speed_menu = tk.Menu(settings_menu, tearoff=0)
        for speed in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
            hue_speed_menu.add_radiobutton(
                label=f"{speed:.2f}",
                variable=self.settings['hue_shift_speed'],
                value=speed,
                command=self._apply_settings
            )
        settings_menu.add_cascade(label="Hue Shift Speed", menu=hue_speed_menu)
        
        # Color saturation options
        saturation_menu = tk.Menu(settings_menu, tearoff=0)
        for sat in [0.0, 0.5, 1.0, 1.5, 2.0]:
            saturation_menu.add_radiobutton(
                label=f"{sat:.1f}",
                variable=self.settings['saturation'],
                value=sat,
                command=self._apply_settings
            )
        settings_menu.add_cascade(label="Color Saturation", menu=saturation_menu)
        
        # Wave/screen color relationship options
        offset_menu = tk.Menu(settings_menu, tearoff=0)
        offsets = {
            "Same as Screen": 0.0,
            "Complementary": 0.5,
            "Analogous+": 0.25,
            "Analogous-": 0.75
        }
        for name, value in offsets.items():
            offset_menu.add_radiobutton(
                label=name,
                variable=self.settings['wave_hue_offset'],
                value=value,
                command=self._apply_settings
            )
        settings_menu.add_cascade(label="Wave Color Relationship", menu=offset_menu)
        
        # Enable/disable toggles
        settings_menu.add_separator()
        settings_menu.add_checkbutton(
            label="Screen Effect Enabled",
            variable=self.settings['screen_enabled'],
            command=self._apply_settings
        )
        settings_menu.add_checkbutton(
            label="Wave Color Effect Enabled",
            variable=self.settings['wave_color_enabled'],
            command=self._apply_settings
        )
        
        # Add settings to main menu
        self.context_menu.add_cascade(label="Visual Effects", menu=settings_menu)
        
        # Size submenu
        size_menu = tk.Menu(self.context_menu, tearoff=0)
        for s in [200, 250, 300, 350, 400, 450, 500]:
            size_menu.add_radiobutton(
                label=f"{s}x{s}",
                command=lambda size=s: self._resize(size)
            )
        self.context_menu.add_cascade(label="Size", menu=size_menu)
        
        # Always on top toggle
        self.context_menu.add_checkbutton(
            label="Always On Top",
            variable=self.settings['always_on_top'],
            command=self._toggle_always_on_top
        )
        
        # Test animations
        test_menu = tk.Menu(self.context_menu, tearoff=0)
        test_menu.add_command(
            label="Amplitude Test",
            command=lambda: asyncio.create_task(self._run_amplitude_test())
        )
        test_menu.add_command(
            label="Speech Pattern",
            command=lambda: asyncio.create_task(self._run_speech_pattern())
        )
        self.context_menu.add_cascade(label="Test Animation", menu=test_menu)
        
        # Exit option
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Exit", command=self.root.quit)
    
    def _show_context_menu(self, event):
        """Show the context menu at the current mouse position."""
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()
    
    def _toggle_settings_panel(self, event=None):
        """Show a detailed settings panel."""
        if hasattr(self, 'settings_window') and self.settings_window.winfo_exists():
            self.settings_window.destroy()
            return
        
        # Create settings window
        self.settings_window = tk.Toplevel(self.root)
        self.settings_window.title("Visualizer Settings")
        self.settings_window.geometry("350x450")
        
        # Keep on top if main window is on top
        if self.settings['always_on_top'].get():
            self.settings_window.attributes('-topmost', True)
        
        # Create settings frame
        frame = ttk.Frame(self.settings_window, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Add settings controls
        row = 0
        
        # Hue shift speed slider
        ttk.Label(frame, text="Hue Shift Speed:").grid(
            row=row, column=0, sticky=tk.W, padx=5, pady=5
        )
        ttk.Scale(
            frame, 
            from_=0.0, 
            to=1.0, 
            orient=tk.HORIZONTAL,
            variable=self.settings['hue_shift_speed'],
            command=lambda _: self._apply_settings()
        ).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        row += 1
        
        # Color saturation slider
        ttk.Label(frame, text="Color Saturation:").grid(
            row=row, column=0, sticky=tk.W, padx=5, pady=5
        )
        ttk.Scale(
            frame, 
            from_=0.0, 
            to=2.0, 
            orient=tk.HORIZONTAL,
            variable=self.settings['saturation'],
            command=lambda _: self._apply_settings()
        ).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        row += 1
        
        # Wave hue offset slider
        ttk.Label(frame, text="Wave Hue Offset:").grid(
            row=row, column=0, sticky=tk.W, padx=5, pady=5
        )
        ttk.Scale(
            frame, 
            from_=0.0, 
            to=1.0, 
            orient=tk.HORIZONTAL,
            variable=self.settings['wave_hue_offset'],
            command=lambda _: self._apply_settings()
        ).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        row += 1
        
        # Screen opacity slider
        ttk.Label(frame, text="Screen Opacity:").grid(
            row=row, column=0, sticky=tk.W, padx=5, pady=5
        )
        ttk.Scale(
            frame, 
            from_=0.1, 
            to=1.0, 
            orient=tk.HORIZONTAL,
            variable=self.settings['screen_opacity'],
            command=lambda _: self._apply_settings()
        ).grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        row += 1
        
        # Add separator
        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10
        )
        row += 1
        
        # Checkboxes for toggles
        ttk.Checkbutton(
            frame,
            text="Screen Effect Enabled",
            variable=self.settings['screen_enabled'],
            command=self._apply_settings
        ).grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        row += 1
        
        ttk.Checkbutton(
            frame,
            text="Wave Color Effect Enabled",
            variable=self.settings['wave_color_enabled'],
            command=self._apply_settings
        ).grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        row += 1
        
        ttk.Checkbutton(
            frame,
            text="Always on Top",
            variable=self.settings['always_on_top'],
            command=self._toggle_always_on_top
        ).grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        row += 1
        
        # Size selection
        ttk.Label(frame, text="Visualizer Size:").grid(
            row=row, column=0, sticky=tk.W, padx=5, pady=5
        )
        size_var = tk.StringVar(value=f"{self.size}")
        size_combo = ttk.Combobox(
            frame,
            textvariable=size_var,
            values=["200", "250", "300", "350", "400", "450", "500"]
        )
        size_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        size_combo.bind("<<ComboboxSelected>>", 
                         lambda _: self._resize(int(size_var.get())))
        row += 1
        
        # Close button
        ttk.Button(
            frame,
            text="Close",
            command=self.settings_window.destroy
        ).grid(row=row, column=0, columnspan=2, pady=10)
        
        # Configure column weights
        frame.columnconfigure(1, weight=1)
    
    def _toggle_always_on_top(self):
        """Toggle whether the window stays on top of other windows."""
        on_top = self.settings['always_on_top'].get()
        self.root.attributes('-topmost', on_top)
        
        # Also apply to settings window if it exists
        if hasattr(self, 'settings_window') and self.settings_window.winfo_exists():
            self.settings_window.attributes('-topmost', on_top)
    
    def _resize(self, new_size):
        """Resize the visualizer window."""
        # Store current position
        x, y = self.root.winfo_x(), self.root.winfo_y()
        
        # Properly clean up the old visualizer
        if hasattr(self, 'visualizer') and self.visualizer:
            # Save current settings before destroying
            current_settings = None
            if hasattr(self.visualizer, 'get_current_settings'):
                try:
                    current_settings = self.visualizer.get_current_settings()
                except:
                    pass
            
            # Stop any ongoing animations
            if hasattr(self.visualizer, 'stop_animation'):
                try:
                    self.visualizer.stop_animation()
                except:
                    pass
            
            # Destroy canvas carefully
            try:
                self.visualizer.canvas.destroy()
            except tk.TclError as e:
                print(f"Warning when destroying canvas: {e}")
        
        # Update size property before recreating
        self.size = new_size
        
        # Update window geometry
        self.root.geometry(f"{new_size}x{new_size}+{x}+{y}")
        
        # Force update to make sure old widgets are gone
        self.root.update_idletasks()
        
        # Create a new visualizer with the new size
        try:
            self._create_visualizer()
            
            # Initialize the new visualizer
            asyncio.create_task(self._initialize_after_resize())
        except Exception as e:
            import traceback
            print(f"Error during resize: {e}")
            traceback.print_exc()
    
    async def _initialize_after_resize(self):
        """Initialize the visualizer after a resize operation."""
        try:
            # Initialize the visualizer
            if hasattr(self.visualizer, 'initialize'):
                await self.visualizer.initialize()
            
            # Rebind events
            self._setup_bindings()
            
            # Apply settings
            self._apply_settings()
            
            # Update UI
            self.root.update()
            
            # Restart animation if needed
            if hasattr(self.visualizer, 'start_animation'):
                await self.visualizer.start_animation()
        except Exception as e:
            print(f"Error initializing after resize: {e}")
            import traceback
            traceback.print_exc()
    
    def _apply_settings(self):
        """Apply current settings to the visualizer."""
        if self.visualizer:
            # Apply visualizer settings
            self.visualizer.set_hue_shift_speed(self.settings['hue_shift_speed'].get())
            self.visualizer.set_saturation_factor(self.settings['saturation'].get())
            self.visualizer.set_screen_opacity(self.settings['screen_opacity'].get())
            self.visualizer.set_hue_shift_enabled(self.settings['screen_enabled'].get())
            self.visualizer.set_wave_hue_shift_enabled(self.settings['wave_color_enabled'].get())
            self.visualizer.set_wave_hue_offset(self.settings['wave_hue_offset'].get())
    
    async def _run_amplitude_test(self):
        """Run a test animation that ramps amplitude up and down."""
        # Temporarily pause simulation
        was_simulating = self.use_simulated_audio
        self._stop_simulation()
        
        # Run ramp up
        for i in range(101):
            intensity = i / 100.0
            self.audio_monitor.set_intensity(intensity)
            await asyncio.sleep(0.02)
        
        # Run ramp down
        for i in range(100, -1, -1):
            intensity = i / 100.0
            self.audio_monitor.set_intensity(intensity)
            await asyncio.sleep(0.02)
        
        # Resume simulation if it was active
        if was_simulating:
            self._start_simulation()
    
    async def _run_speech_pattern(self):
        """Run a test animation that simulates speech patterns."""
        # Temporarily pause simulation
        was_simulating = self.use_simulated_audio
        self._stop_simulation()
        
        # Run speech pattern for 5 seconds
        start_time = time.time()
        duration = 5.0
        
        while time.time() - start_time < duration:
            t = time.time() - start_time
            
            # Create a speech-like pattern with multiple frequencies
            base = 0.3 + 0.2 * math.sin(t * 1.2)
            syllables = 0.25 * math.sin(t * 8.5) * math.sin(t * 0.7)
            detail = 0.1 * math.sin(t * 20)
            pauses = max(0, 0.2 * math.sin(t * 0.3) - 0.1)
            
            intensity = max(0.0, min(0.95, base + syllables + detail - pauses))
            self.audio_monitor.set_intensity(intensity)
            await asyncio.sleep(1/30)  # 30 FPS
        
        # Resume simulation if it was active
        if was_simulating:
            self._start_simulation()
    
    def _start_simulation(self):
        """Start the audio simulation task."""
        self.use_simulated_audio = True
        
        if self.simulation_task:
            self.simulation_task.cancel()
            
        self.simulation_task = asyncio.create_task(self._audio_simulation_loop())
    
    def _stop_simulation(self):
        """Stop the audio simulation task."""
        self.use_simulated_audio = False
        
        if self.simulation_task:
            self.simulation_task.cancel()
            self.simulation_task = None
    
    async def _audio_simulation_loop(self):
        """Run a continuous audio simulation loop."""
        start_time = time.time()
        
        try:
            while True:
                t = time.time() - start_time
                
                # Create a gentle ambient pattern with occasional pulses
                base = 0.1 + 0.05 * math.sin(t * 0.3)
                detail = 0.03 * math.sin(t * 1.7)
                
                # Every ~5 seconds, add a pulse
                if (t % 5) < 0.5:
                    pulse = 0.1 + 0.15 * math.sin(t * 8)
                else:
                    pulse = 0
                
                intensity = max(0.0, min(0.8, base + detail + pulse))
                self.audio_monitor.set_intensity(intensity)
                await asyncio.sleep(1/30)  # 30 FPS
                
        except asyncio.CancelledError:
            pass
    
    def _create_visualizer(self):
        """Create the visualizer component."""
        # Make sure the frame exists
        if not hasattr(self, 'frame') or not self.frame.winfo_exists():
            self.frame = tk.Frame(self.root, background=self.transparency_color, bd=0, highlightthickness=0)
            self.frame.pack(fill=tk.BOTH, expand=True)
            
            # Add drag event handling to frame
            def frame_start_drag(event):
                return self._start_drag(event)
                
            def frame_stop_drag(event):
                return self._stop_drag(event)
                
            def frame_on_drag(event):
                return self._on_drag(event)
                
            self.frame.bind("<Button-1>", frame_start_drag)
            self.frame.bind("<ButtonRelease-1>", frame_stop_drag)
            self.frame.bind("<B1-Motion>", frame_on_drag)
        
        # Clean up any existing visualizer
        try:
            if hasattr(self, 'visualizer') and self.visualizer:
                if hasattr(self.visualizer, 'canvas') and self.visualizer.canvas.winfo_exists():
                    self.visualizer.canvas.destroy()
                self.visualizer = None
        except tk.TclError:
            # Canvas might already be destroyed
            pass
        
        # Update the frame before adding new components
        self.frame.update_idletasks()
        
        # Create a new visualizer component
        try:
            self.visualizer = LinuxFloatingVisualizerUI(
                parent=self.frame,
                size=(self.size, self.size),
                audio_monitor=self.audio_monitor,
                transparency_color=self.transparency_color
            )
            
            # Make sure the canvas is properly configured
            if hasattr(self.visualizer, 'canvas'):
                self.visualizer.canvas.pack(fill=tk.BOTH, expand=True)
                
                # Configure canvas for transparency
                self.visualizer.canvas.configure(bg=self.transparency_color, highlightthickness=0, borderwidth=0)
                
                # Define explicit handlers to avoid lambda issues
                def canvas_start_drag(event):
                    return self._start_drag(event)
                    
                def canvas_stop_drag(event):
                    return self._stop_drag(event)
                    
                def canvas_on_drag(event):
                    return self._on_drag(event)
                
                # Apply these bindings with priority (add="+")
                self.visualizer.canvas.bind("<Button-1>", canvas_start_drag, add="+")
                self.visualizer.canvas.bind("<ButtonRelease-1>", canvas_stop_drag, add="+")
                self.visualizer.canvas.bind("<B1-Motion>", canvas_on_drag, add="+")
                
                # Explicitly disable the canvas's built-in dragging behavior 
                # by overriding the default bindings
                self.visualizer.canvas.bind("<Button-1>", canvas_start_drag)
                self.visualizer.canvas.bind("<B1-Motion>", canvas_on_drag)
                self.visualizer.canvas.bind("<ButtonRelease-1>", canvas_stop_drag)
            
            # Update immediately
            self.frame.update_idletasks()
            return True
        except Exception as e:
            print(f"Error creating visualizer: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _apply_x11_fixes(self):
        """Apply X11-specific fixes to improve window behavior."""
        # Get the window ID if we don't have it yet
        if not self.window_id:
            self.window_id = get_x11_window_id(self.window_title)
            
        # Apply X11 properties
        if self.window_id:
            apply_x11_window_properties(self.window_id)
    
    async def initialize(self):
        """Initialize the visualizer."""
        print("Creating visualizer component...")
        # Create the visualizer component
        self._create_visualizer()
        
        print("Initializing visualizer...")
        # Initialize visualizer
        await self.visualizer.initialize()
        
        print("Setting up context menu...")
        # Set up context menu
        self._setup_context_menu()
        
        # Make sure the UI is ready before proceeding
        self.root.update_idletasks()
        
        print("Configuring window...")
        # Apply any additional X11 fixes for Linux
        self._apply_x11_fixes()
        
        # Add special window class to help window managers
        try:
            self.root.wm_class('BellaFloatingVisualizer', 'Bella')
        except:
            pass
            
        # Make sure click-through is properly working
        self.root.attributes("-type", "dock")  # Try dock type for better dragging
        
        # Force global window draggability
        try:
            # Add this hint to allow window dragging by content
            self.root.wm_attributes('-draggable', True)
        except:
            pass
        
        print("Setting up input bindings...")
        # Set up input bindings - specifically just on the window frame
        self._setup_bindings()
        
        # Disable internal canvas bindings to prevent canvas from handling mouse events
        if hasattr(self.visualizer, 'canvas'):
            self._disable_canvas_drag()
        
        print("Applying settings...")
        # Apply settings
        self._apply_settings()
        self._toggle_always_on_top()
        
        # Update UI
        self.root.update_idletasks()
        
        print("Starting audio...")
        # Start audio monitoring
        self.audio_monitor.start()
        
        # Start simulation if needed
        if self.use_simulated_audio:
            self._start_simulation()
            
    def _disable_canvas_drag(self):
        """Disable all mouse drag events on the canvas."""
        if not hasattr(self.visualizer, 'canvas'):
            return
            
        # Create a custom empty event handler that does nothing but block events
        def block_event(event):
            # Redirect to window drag
            return self._start_drag(event)
            
        def block_motion(event):
            return self._on_drag(event)
            
        def block_release(event):
            return self._stop_drag(event)
        
        # Apply to the canvas
        canvas = self.visualizer.canvas
        canvas.bind("<Button-1>", block_event, add="+")
        canvas.bind("<B1-Motion>", block_motion, add="+")
        canvas.bind("<ButtonRelease-1>", block_release, add="+")
    
    async def run(self):
        """Run the visualizer application."""
        try:
            # Initialize first
            print("Initializing visualizer...")
            await self.initialize()
            
            # Make sure window is visible and ready
            self.root.deiconify()
            self.root.update_idletasks()
            self.root.update()
            
            print("Starting animation...")
            # Start animation if not already started
            if hasattr(self.visualizer, 'start_animation'):
                await self.visualizer.start_animation()
                
            # Print ready message
            print("Visualizer running - use mouse to interact")
            print("Click and drag the window border to move the visualizer")
            
            # Main event loop with improved responsiveness
            is_running = True
            frame_time = 1/120  # 120 fps for smooth UI
            
            while is_running:
                try:
                    # Process multiple events per frame for better responsiveness
                    for _ in range(5):
                        self.root.update()
                    
                    # Short sleep to prevent CPU hogging
                    await asyncio.sleep(frame_time)
                    
                except tk.TclError as e:
                    if "application has been destroyed" in str(e):
                        is_running = False
                    else:
                        print(f"Warning in UI update: {e}")
                except Exception as e:
                    print(f"Error in main loop: {e}")
                
        except tk.TclError as e:
            if "application has been destroyed" not in str(e):
                print(f"Window error: {e}")
        except Exception as e:
            import traceback
            print(f"Error in visualizer: {e}")
            traceback.print_exc()
        finally:
            # Clean up
            self._stop_simulation()
            print("Visualizer closed")
            
            # Make sure to exit cleanly
            try:
                self.root.quit()
            except:
                pass


class LinuxFloatingVisualizerUI(EnhancedVoiceVisualizerUI):
    """
    Linux-optimized floating visualizer UI component.
    """
    
    def __init__(self, parent=None, size=(400, 400), audio_monitor=None, transparency_color="#FF01FE"):
        """
        Initialize the visualizer UI.
        
        Args:
            parent: Parent widget/frame
            size: (width, height) of the visualizer
            audio_monitor: Optional custom audio monitor
            transparency_color: Color to use for transparency (will be made fully transparent)
        """
        # Store reference to custom audio monitor if provided
        self.custom_audio_monitor = audio_monitor
        
        # Store transparency color
        self.transparency_color = transparency_color
        
        # Track if animation is active
        self.animation_running = False
        self._animation_task = None
        
        # Initialize parent class
        super().__init__(parent, size)
    
    def _create_canvas(self):
        """Create the tkinter canvas for drawing."""
        # Create a transparent canvas with no border
        self.canvas = tk.Canvas(
            self.parent,
            width=self.size[0],
            height=self.size[1],
            bg=self.transparency_color,  # Use transparency color
            highlightthickness=0,
            borderwidth=0
        )
        
        # Configure canvas
        self.canvas.config(takefocus=0)
        
        # Pack the canvas to fill the parent
        self.canvas.pack(fill=tk.BOTH, expand=True)
    
    def _setup_audio_monitor(self):
        """Set up the audio monitor."""
        # Use provided monitor if available, otherwise create default
        if self.custom_audio_monitor:
            self.intensity_monitor = self.custom_audio_monitor
        else:
            # Fall back to parent implementation
            super()._setup_audio_monitor()
    
    def set_screen_opacity(self, opacity):
        """Set the opacity of the screen overlay."""
        self.screen_opacity = opacity
        # Apply immediately if initialized
        if hasattr(self, 'hue_shifting_screen') and self.hue_shifting_screen:
            self.hue_shifting_screen.set_opacity(opacity)
    
    def get_current_settings(self):
        """Get current visualizer settings."""
        settings = {}
        
        # Save basic settings
        if hasattr(self, 'hue_shift_speed'):
            settings['hue_shift_speed'] = self.hue_shift_speed
            
        if hasattr(self, 'saturation_factor'):
            settings['saturation_factor'] = self.saturation_factor
            
        if hasattr(self, 'screen_opacity'):
            settings['screen_opacity'] = self.screen_opacity
            
        return settings
    
    def _update_visualization(self):
        """
        Update the visualization display with the current audio intensity.
        This method is called by the animation loop to update the display.
        """
        try:
            # Get current time for phase selection
            current_time = time.time() - self.start_time
            
            # Get current intensity from audio monitor
            intensity = self.intensity_monitor.get_current_intensity()
            
            # Get the appropriate wave frame
            wave_frame = self.frames.get_frame(
                intensity=intensity,
                time_position=current_time
            )
            
            # Get the current hue-shifted screen
            screen_overlay = self.hue_shifting_screen.get_current_screen()
            
            if wave_frame:
                # Apply hue shift to the wave if enabled
                if self.wave_hue_shift_enabled and self.hue_shift_enabled:
                    # Calculate wave hue based on screen hue plus offset
                    wave_hue = (self.hue_shifting_screen.current_hue + self.wave_hue_offset) % 1.0
                    wave_frame = self.apply_hue_shift_to_wave(wave_frame, wave_hue)
            
            # Combine the wave frame with the screen overlay
            if wave_frame and screen_overlay:
                # Create a transparent base image (completely transparent)
                base_image = Image.new("RGBA", wave_frame.size, (0, 0, 0, 0))
                
                # Apply screen overlay to base image
                base_with_screen = Image.alpha_composite(base_image, screen_overlay)
                
                # Apply wave on TOP of the screen overlay
                self.current_frame = Image.alpha_composite(base_with_screen, wave_frame)
            elif wave_frame:
                # Just use the wave frame if no screen overlay
                self.current_frame = wave_frame
            
            # Update the UI with the new frame
            if self.current_frame:
                self._update_ui_frame(self.current_frame)
                
        except tk.TclError as e:
            # Handle Tkinter errors (like destroyed window)
            if "application has been destroyed" in str(e) or "invalid command name" in str(e):
                self.animation_running = False
                raise
            print(f"Warning in visualization update: {e}")
        except Exception as e:
            # Handle other errors
            print(f"Error updating visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_ui_frame(self, frame: Image.Image):
        """
        Update the UI with a new frame image, preserving transparency.
        
        Args:
            frame: PIL Image to display
        """
        if not self.canvas:
            return
        
        try:
            # Resize the frame if needed
            if frame.size != self.size:
                frame = frame.resize(self.size, Image.Resampling.LANCZOS)
            
            # For complete transparency, modify the image to use our transparency color for transparent areas
            if hasattr(self, 'transparency_color'):
                # Convert transparency_color from hex to RGB
                tr_color = self.transparency_color.lstrip('#')
                tr_r, tr_g, tr_b = tuple(int(tr_color[i:i+2], 16) for i in (0, 2, 4))
                
                # Create a new RGBA array
                frame_array = np.array(frame)
                
                # Find fully transparent pixels (alpha = 0)
                transparent_mask = frame_array[:, :, 3] == 0
                
                # Apply transparency color to those pixels
                if np.any(transparent_mask):
                    frame_array[transparent_mask, 0] = tr_r
                    frame_array[transparent_mask, 1] = tr_g
                    frame_array[transparent_mask, 2] = tr_b
                    frame_array[transparent_mask, 3] = 0  # Keep alpha as 0
                    
                    # Create new image from array
                    frame = Image.fromarray(frame_array, 'RGBA')
            
            # Convert PIL image to Tkinter PhotoImage
            self.current_frame_image = ImageTk.PhotoImage(frame)
            
            # Update or create the image on canvas
            if not hasattr(self, 'frame_image_id') or self.frame_image_id is None:
                self.frame_image_id = self.canvas.create_image(
                    self.size[0] // 2, self.size[1] // 2,  # Center position
                    image=self.current_frame_image
                )
            else:
                self.canvas.itemconfig(self.frame_image_id, image=self.current_frame_image)
                
        except (RuntimeError, tk.TclError) as e:
            # Handle Tkinter errors gracefully
            print(f"Tkinter error in updating frame: {e}")
        except Exception as e:
            print(f"Error updating UI frame: {e}")
            import traceback
            traceback.print_exc()
    
    async def start_animation(self):
        """Start the visualization animation loop."""
        # Prevent multiple animation loops
        if self.animation_running:
            return
            
        self.animation_running = True
        
        # Create a new animation task
        if self._animation_task:
            try:
                self._animation_task.cancel()
            except:
                pass
                
        self._animation_task = asyncio.create_task(self._animation_loop())
    
    def stop_animation(self):
        """Stop the animation loop."""
        self.animation_running = False
        
        if self._animation_task:
            try:
                self._animation_task.cancel()
                self._animation_task = None
            except:
                pass
    
    async def _animation_loop(self):
        """Run the animation loop."""
        try:
            while self.animation_running:
                # Update the visualization
                try:
                    self._update_visualization()
                except tk.TclError as e:
                    # If the canvas was destroyed, stop the animation
                    if "application has been destroyed" in str(e) or "invalid command name" in str(e):
                        self.animation_running = False
                        break
                    else:
                        print(f"Warning in animation: {e}")
                
                # Wait for next frame
                await asyncio.sleep(1/30)  # 30 FPS animation
        except asyncio.CancelledError:
            # Task was cancelled
            pass
        except Exception as e:
            import traceback
            print(f"Error in animation loop: {e}")
            traceback.print_exc()
        finally:
            self.animation_running = False


async def main():
    """Main entry point for the Linux-optimized floating visualizer."""
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run the Linux-optimized Bella floating visualizer"
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
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process position argument
    position = None
    if args.position:
        try:
            x, y = map(int, args.position.split(','))
            position = (x, y)
        except:
            print(f"Invalid position format: {args.position}, using default")
    
    # Create visualizer
    visualizer = FloatingVisualizer(size=args.size, position=position)
    
    # Apply initial settings from command line
    visualizer.settings['always_on_top'].set(args.always_on_top)
    visualizer.use_simulated_audio = not args.no_simulation
    
    # Run the visualizer
    await visualizer.run()


if __name__ == "__main__":
    asyncio.run(main())
