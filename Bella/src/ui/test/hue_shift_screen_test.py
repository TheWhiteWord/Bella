#!/usr/bin/env python3
"""
Hue-Shifting Screen Overlay Test

This script demonstrates how to apply a continuously shifting hue effect
to the screen.png overlay, creating a dynamic color effect for the voice visualizer.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk
import asyncio
import time
import numpy as np
from PIL import Image, ImageTk
import colorsys
from pathlib import Path

# Add project root to path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))


class HueShiftingScreenTest:
    """Test application that shows the hue-shifting screen effect"""
    
    def __init__(self):
        """Initialize the test application"""
        # Set up main Tkinter window
        self.root = tk.Tk()
        self.root.title("Hue-Shifting Screen Overlay Test")
        self.root.geometry("600x600")
        self.root.configure(bg="#1E1E2E")
        
        # Element paths
        self.elements_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "elements")
        self.screen_img_path = os.path.join(self.elements_dir, "screen.png")
        self.frame_img_path = os.path.join(self.elements_dir, "frame.png")
        
        # Hue shifting parameters
        self.hue_shift_speed = 0.2  # Full hue cycle per second
        self.hue_shift_enabled = True
        self.hue_min = 0.0
        self.hue_max = 1.0
        self.current_hue = 0.0
        self.saturation_factor = 1.0  # Adjust to control color intensity
        
        # UI elements
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Canvas for displaying the image
        self.canvas = tk.Canvas(
            self.main_frame,
            width=400,
            height=400,
            bg="black",
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Image containers
        self.base_image = None
        self.screen_image = None
        self.current_display_image = None
        self.tk_image = None
        self.image_id = None
        
        # Controls frame
        self.controls_frame = ttk.Frame(self.root)
        self.controls_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Speed slider
        ttk.Label(self.controls_frame, text="Hue Shift Speed:").pack(anchor=tk.W)
        self.speed_var = tk.DoubleVar(value=self.hue_shift_speed)
        self.speed_slider = ttk.Scale(
            self.controls_frame,
            from_=0.0,
            to=2.0,
            orient=tk.HORIZONTAL,
            variable=self.speed_var,
            command=self.on_speed_change
        )
        self.speed_slider.pack(fill=tk.X, pady=5)
        
        # Saturation slider
        ttk.Label(self.controls_frame, text="Color Saturation:").pack(anchor=tk.W)
        self.saturation_var = tk.DoubleVar(value=self.saturation_factor)
        self.saturation_slider = ttk.Scale(
            self.controls_frame,
            from_=0.0,
            to=2.0,
            orient=tk.HORIZONTAL,
            variable=self.saturation_var,
            command=self.on_saturation_change
        )
        self.saturation_slider.pack(fill=tk.X, pady=5)
        
        # Toggle button
        self.toggle_button = ttk.Button(
            self.controls_frame,
            text="Disable Hue Shift",
            command=self.toggle_hue_shift
        )
        self.toggle_button.pack(anchor=tk.W, pady=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Initializing...")
        self.status_label = ttk.Label(
            self.controls_frame,
            textvariable=self.status_var
        )
        self.status_label.pack(anchor=tk.W, pady=5)
    
    def on_speed_change(self, event=None):
        """Handle speed slider changes"""
        self.hue_shift_speed = self.speed_var.get()
        self.status_var.set(f"Hue shift speed: {self.hue_shift_speed:.2f} cycles per second")
    
    def on_saturation_change(self, event=None):
        """Handle saturation slider changes"""
        self.saturation_factor = self.saturation_var.get()
        self.status_var.set(f"Color saturation: {self.saturation_factor:.2f}")
    
    def toggle_hue_shift(self):
        """Toggle hue shifting on/off"""
        self.hue_shift_enabled = not self.hue_shift_enabled
        if self.hue_shift_enabled:
            self.toggle_button.configure(text="Disable Hue Shift")
            self.status_var.set("Hue shifting enabled")
        else:
            self.toggle_button.configure(text="Enable Hue Shift")
            self.status_var.set("Hue shifting disabled")
    
    async def initialize(self):
        """Load and prepare the images"""
        try:
            # Load the base frame image
            if os.path.exists(self.frame_img_path):
                self.base_image = Image.open(self.frame_img_path).convert("RGBA")
                self.status_var.set("Loaded frame image")
            else:
                # Create a default black background with the same size as the screen
                self.screen_image = Image.open(self.screen_img_path).convert("RGBA")
                self.base_image = Image.new("RGBA", self.screen_image.size, (0, 0, 0, 255))
                self.status_var.set("Created default black background")
            
            # Load the screen overlay image
            if os.path.exists(self.screen_img_path):
                self.screen_image = Image.open(self.screen_img_path).convert("RGBA")
                self.status_var.set("Loaded screen overlay image")
            else:
                self.status_var.set("Error: Screen image not found")
                return False
            
            # Resize canvas to match image size
            self.canvas.config(width=self.base_image.width, height=self.base_image.height)
            
            # Initial display
            await self.update_hue_shifted_image()
            return True
            
        except Exception as e:
            self.status_var.set(f"Error initializing: {e}")
            return False
    
    def apply_hue_shift(self, image, hue_value):
        """
        Apply a hue shift to the given image
        
        Args:
            image: PIL Image in RGBA format
            hue_value: New hue value (0-1)
            
        Returns:
            PIL Image with shifted hue
        """
        # Convert to numpy array
        img_arr = np.array(image, dtype=np.float32) / 255.0
        
        # Extract RGB and alpha channels
        r, g, b, a = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2], img_arr[:, :, 3]
        
        # Find non-transparent pixels
        mask = a > 0.1
        
        # Create output array
        output = np.zeros_like(img_arr)
        output[:, :, 3] = a  # Preserve alpha
        
        # Only process non-transparent pixels
        if np.any(mask):
            # Convert RGB to HSV
            h, s, v = np.zeros_like(r[mask]), np.zeros_like(r[mask]), np.zeros_like(r[mask])
            
            # Process in small batches to avoid memory issues
            for i, (ri, gi, bi) in enumerate(zip(r[mask].flat, g[mask].flat, b[mask].flat)):
                hi, si, vi = colorsys.rgb_to_hsv(ri, gi, bi)
                h.flat[i], s.flat[i], v.flat[i] = hi, si, vi
            
            # Shift hue and adjust saturation
            h = (h + hue_value) % 1.0
            s = np.clip(s * self.saturation_factor, 0, 1)
            
            # Convert back to RGB
            new_r, new_g, new_b = np.zeros_like(h), np.zeros_like(h), np.zeros_like(h)
            for i, (hi, si, vi) in enumerate(zip(h.flat, s.flat, v.flat)):
                ri, gi, bi = colorsys.hsv_to_rgb(hi, si, vi)
                new_r.flat[i], new_g.flat[i], new_b.flat[i] = ri, gi, bi
            
            # Copy back to output
            output_flat = output[mask]
            output_flat[:, 0], output_flat[:, 1], output_flat[:, 2] = new_r, new_g, new_b
            output[mask] = output_flat
        
        # Convert back to uint8
        output = (output * 255).astype(np.uint8)
        return Image.fromarray(output, 'RGBA')
    
    def composite_images(self, base, overlay):
        """
        Composite the base image with the overlay using overlay blend mode
        
        Args:
            base: Base PIL Image
            overlay: Overlay PIL Image with hue shift applied
            
        Returns:
            Composited PIL Image
        """
        # For now, just use alpha compositing
        # A more complex blend mode could be implemented here
        return Image.alpha_composite(base, overlay)
    
    async def update_hue_shifted_image(self):
        """Update the displayed image with the current hue shift value"""
        start_time = time.time()
        
        # Apply hue shift to screen overlay if enabled
        if self.hue_shift_enabled:
            hue_shifted_screen = self.apply_hue_shift(self.screen_image, self.current_hue)
        else:
            hue_shifted_screen = self.screen_image
        
        # Composite with base image
        self.current_display_image = self.composite_images(
            self.base_image.copy(), hue_shifted_screen
        )
        
        # Convert to Tkinter PhotoImage and display
        self.tk_image = ImageTk.PhotoImage(self.current_display_image)
        if self.image_id is None:
            self.image_id = self.canvas.create_image(
                self.current_display_image.width // 2, 
                self.current_display_image.height // 2,
                image=self.tk_image
            )
        else:
            self.canvas.itemconfig(self.image_id, image=self.tk_image)
        
        # Update frame rate in status if it took significant time
        process_time = time.time() - start_time
        if process_time > 0.05:  # Only show if processing takes >50ms
            fps = 1.0 / process_time
            self.status_var.set(f"Frame rate: {fps:.1f} FPS")
    
    async def animation_loop(self):
        """Main animation loop for hue shifting"""
        start_time = time.time()
        
        while True:
            # Calculate current hue based on time
            if self.hue_shift_enabled:
                elapsed_time = time.time() - start_time
                self.current_hue = (elapsed_time * self.hue_shift_speed) % 1.0
            
            # Update the image
            await self.update_hue_shifted_image()
            
            # Sleep to maintain reasonable frame rate
            await asyncio.sleep(1/30)  # Target 30 FPS
    
    async def run(self):
        """Run the test application"""
        # Initialize images
        success = await self.initialize()
        if not success:
            self.status_var.set("Failed to initialize images")
            return
        
        # Start animation loop
        animation_task = asyncio.create_task(self.animation_loop())
        
        # Set up tkinter event loop integration with asyncio
        async def update_tkinter():
            try:
                while True:
                    self.root.update()
                    await asyncio.sleep(0.01)  # ~100 updates per second
            except tk.TclError:
                # Window was closed
                animation_task.cancel()
        
        # Run the tkinter update loop
        await update_tkinter()


async def main():
    """Main entry point"""
    app = HueShiftingScreenTest()
    await app.run()


if __name__ == "__main__":
    asyncio.run(main())
