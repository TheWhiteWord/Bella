#!/usr/bin/env python3
"""
Bella Voice Visualizer - PyQt Implementation

A transparent floating window that visualizes Bella's voice audio
using pre-generated wave animations. The visualization intensity 
changes with voice volume, while maintaining a continuous animated
wave effect through phase transitions.

Features:
- Linux-optimized transparent floating window
- Automatic audio intensity visualization
- Color-shifting visual effects
- Adjustable size and positioning
- Responsive drag and resize functionality
"""

import sys
import os
import re
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import asyncio
import threading
import argparse

# PyQt imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QLabel, QMenu, QAction, QSlider, QDialog, QColorDialog
)
from PyQt5.QtGui import (
    QPixmap, QPainter, QColor, QBrush, QPen, QFont, 
    QCursor, QMouseEvent, QResizeEvent, QPaintEvent, QImage
)
from PyQt5.QtCore import (
    Qt, QRectF, QTimer, QPoint, QSize, pyqtSignal, 
    pyqtSlot, QObject, QPropertyAnimation, QEasingCurve
)

# Import PIL for image processing
from PIL import Image, ImageEnhance, ImageOps, ImageChops

# Define constants
DEFAULT_SIZE = 200
DEFAULT_OPACITY = 0.85
DEFAULT_POSITION = (100, 100)
DEFAULT_FPS = 30
FRAME_INTERVAL_MS = int(1000 / DEFAULT_FPS)

# Color and hue shifting constants
DEFAULT_HUE_SHIFT_SPEED = 0.1  # Hue shift per second (0.0 to 1.0)
DEFAULT_SATURATION = 1.0       # Saturation level (0.0 to 1.0) - increased for better visibility
DEFAULT_WAVE_HUE_OFFSET = 0.25  # Hue offset for wave relative to screen (0.0 to 1.0)
DEFAULT_SCREEN_OPACITY = 0.4   # Screen overlay opacity - reduced to make wave more visible
DEFAULT_SCREEN_ENABLED = True  # Enable hue shift for screen
DEFAULT_WAVE_COLOR_ENABLED = False  # Enable hue shift for wave

class FrameManager:
    """
    Manages loading and caching of animation frames, organizing them by amplitude and phase.
    Provides methods to select appropriate frames based on audio intensity and time position.
    """
    
    def __init__(self, frames_dir: str):
        """
        Initialize the frame manager with the directory containing wave frames
        
        Args:
            frames_dir: Path to the directory containing wave frame PNG files
        """
        self.frames_dir = Path(frames_dir)
        self.frames_by_amplitude = {}
        self.amplitudes = []
        self.phases_per_amplitude = 0
        self.cached_pixmaps = {}
        
        # Load frame index
        self._load_frame_index()
    
    def _load_frame_index(self):
        """Load and index all frame files from the frames directory"""
        if not self.frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {self.frames_dir}")
        
        frame_files = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.png')])
        
        # Parse amplitude and phase from filenames (e.g., wave_frame_a30.0_p05.png)
        for filename in frame_files:
            match = re.search(r'a(\d+\.\d+)_p(\d+)', filename)
            if match:
                amplitude = float(match.group(1))
                phase_idx = int(match.group(2))
                
                if amplitude not in self.frames_by_amplitude:
                    self.frames_by_amplitude[amplitude] = {}
                
                frame_path = self.frames_dir / filename
                self.frames_by_amplitude[amplitude][phase_idx] = str(frame_path)
        
        # Store sorted list of available amplitudes
        self.amplitudes = sorted(self.frames_by_amplitude.keys())
        
        # Store the number of phases per amplitude (assuming all have the same number)
        if self.amplitudes and self.frames_by_amplitude[self.amplitudes[0]]:
            self.phases_per_amplitude = len(self.frames_by_amplitude[self.amplitudes[0]])
        
        print(f"Loaded {len(frame_files)} frames across {len(self.amplitudes)} amplitude levels")
        print(f"Each amplitude has {self.phases_per_amplitude} phase positions")
    
    def get_frame_pixmap(self, intensity: float, time_position: float, max_intensity: float = 1.0) -> QPixmap:
        """
        Get the appropriate frame based on audio intensity and time position
        
        Args:
            intensity: Audio intensity value (0.0 to max_intensity)
            time_position: Current time position in seconds (for phase selection)
            max_intensity: Maximum expected intensity value
            
        Returns:
            QPixmap of the selected frame
        """
        # Map intensity to amplitude (with constraint to available range)
        normalized_intensity = min(max(0.0, intensity), max_intensity)
        target_amplitude = normalized_intensity * max(self.amplitudes) / max_intensity
        
        # Find the closest available amplitude
        amplitude = min(self.amplitudes, key=lambda a: abs(a - target_amplitude))
        
        # Select phase based on time position (creates continuous movement)
        phase_idx = int(time_position * self.phases_per_amplitude) % self.phases_per_amplitude
        
        # Get frame path
        frame_path = self.frames_by_amplitude[amplitude][phase_idx]
        
        # Use cached pixmap or create a new one
        if frame_path not in self.cached_pixmaps:
            self.cached_pixmaps[frame_path] = QPixmap(frame_path)
        
        return self.cached_pixmaps[frame_path]

    def preload_frames(self):
        """Preload all frames to cache for smoother animation"""
        print("Preloading frames to cache...")
        for amplitude in self.amplitudes:
            for phase_idx in self.frames_by_amplitude[amplitude]:
                frame_path = self.frames_by_amplitude[amplitude][phase_idx]
                if frame_path not in self.cached_pixmaps:
                    self.cached_pixmaps[frame_path] = QPixmap(frame_path)
        print(f"Preloaded {len(self.cached_pixmaps)} frames")


class AudioSimulator(QObject):
    """
    Simulates audio intensity for testing the visualizer without real audio input.
    Provides smooth transitions between intensity levels for natural visualization.
    """
    intensity_updated = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_intensity)
        self.current_intensity = 0.0
        self.target_intensity = 0.0
        self.transition_speed = 0.15  # How quickly to change intensity (0-1)
        self.is_speaking = False
        self.time_since_last_change = 0
        self.intensity_change_interval = 0.5  # Time between intensity changes
        
    def start_simulation(self, interval_ms=30):
        """Start the simulation timer"""
        self.timer.start(interval_ms)
        
    def stop_simulation(self):
        """Stop the simulation timer"""
        self.timer.stop()
        
    def set_speaking(self, is_speaking: bool):
        """Set whether the assistant is currently speaking"""
        self.is_speaking = is_speaking
        if not is_speaking:
            self.target_intensity = 0.0
    
    def _update_intensity(self):
        """Update current audio intensity, moving toward target intensity"""
        # Only change target periodically
        self.time_since_last_change += self.timer.interval() / 1000
        if self.time_since_last_change >= self.intensity_change_interval:
            if self.is_speaking:
                # When speaking, create a random but natural pattern of intensities
                self.target_intensity = 0.1 + 0.8 * np.random.beta(2, 2)  # Beta distribution for natural pattern
            else:
                self.target_intensity = 0.0
            self.time_since_last_change = 0
        
        # Smoothly transition toward target
        transition_amount = self.transition_speed * (self.timer.interval() / 1000) * 10
        diff = self.target_intensity - self.current_intensity
        self.current_intensity += diff * transition_amount
        
        # Emit the updated intensity
        self.intensity_updated.emit(self.current_intensity)


class VoiceVisualizerWindow(QMainWindow):
    """
    Main visualizer window that displays animated wave frames
    and responds to user interactions.
    """
    
    def __init__(self, frames_dir=None, size=DEFAULT_SIZE, position=DEFAULT_POSITION):
        super().__init__()
        
        # If no frames directory specified, use default location
        if frames_dir is None:
            frames_dir = os.path.join(os.path.dirname(__file__), "wave_frames")
        
        # Initialize frame manager
        self.frame_manager = FrameManager(frames_dir)
        
        # Set up window properties
        self.setWindowFlags(
            Qt.FramelessWindowHint |  # No window frame
            Qt.WindowStaysOnTopHint |  # Stay on top of other windows
            Qt.Tool  # No taskbar entry
        )
        self.setAttribute(Qt.WA_TranslucentBackground)  # Allow transparency
        self.setFixedSize(size, size)
        self.move(*position)
        
        # Initialize central widget and layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create label to display frames
        self.frame_label = QLabel(self)
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.frame_label)
        
        # Load screen overlay
        self.screen_overlay = None
        self.screen_opacity = DEFAULT_SCREEN_OPACITY
        self._load_screen_overlay()
        
        # Color and hue shifting settings
        self.hue_shift_speed = DEFAULT_HUE_SHIFT_SPEED
        self.saturation = DEFAULT_SATURATION
        self.wave_hue_offset = DEFAULT_WAVE_HUE_OFFSET
        self.screen_enabled = DEFAULT_SCREEN_ENABLED
        self.wave_color_enabled = DEFAULT_WAVE_COLOR_ENABLED
        self.current_hue = 0.0  # Current hue value (0.0 to 1.0)
        
        # Track mouse position for dragging
        self.dragging = False
        self.drag_start_position = QPoint()
        
        # Animation state
        self.start_time = time.time()
        self.last_frame_time = time.time()
        self.current_intensity = 0.0
        
        # Set up animation timer
        self.animation_timer = QTimer(self)
        self.animation_timer.setInterval(FRAME_INTERVAL_MS)
        self.animation_timer.timeout.connect(self._update_animation)
        self.animation_timer.start()
        
        # Set up audio simulator for testing
        self.audio_simulator = AudioSimulator(self)
        self.audio_simulator.intensity_updated.connect(self._on_intensity_update)
        self.audio_simulator.start_simulation()
        self.audio_simulator.set_speaking(True)  # Start in speaking mode for demo
        
        # Preload frames in a separate thread to avoid freezing the UI
        threading.Thread(target=self.frame_manager.preload_frames, daemon=True).start()
        
        # Set window title
        self.setWindowTitle("Bella Voice Visualizer")
        
    def _load_screen_overlay(self):
        """Load the screen overlay image with hue shift and opacity"""
        screen_path = os.path.join(os.path.dirname(__file__), "elements", "screen.png")
        if os.path.exists(screen_path):
            try:
                # Use PIL to load with transparency
                screen_img = Image.open(screen_path).convert("RGBA")
                
                # Store the original image for dynamic hue shifting
                self.original_screen_img = screen_img.copy()
                
                # Apply hue shift if enabled
                if hasattr(self, 'screen_enabled') and self.screen_enabled:
                    screen_img = self._apply_hue_shift(
                        screen_img, 
                        hue_shift=self.current_hue, 
                        saturation=self.saturation
                    )
                
                # Apply opacity
                if self.screen_opacity < 1.0:
                    # Create a new image with same size but with applied opacity
                    alpha = screen_img.split()[3]
                    alpha = alpha.point(lambda p: int(p * self.screen_opacity))
                    screen_img.putalpha(alpha)
                
                # Convert to QPixmap
                data = screen_img.tobytes("raw", "RGBA")
                qimage = QImage(data, screen_img.width, screen_img.height, QImage.Format_RGBA8888)
                self.screen_overlay = QPixmap.fromImage(qimage)
            except Exception as e:
                print(f"Error loading screen overlay: {e}")
                self.screen_overlay = None
        else:
            print(f"Screen overlay not found at {screen_path}")
            self.screen_overlay = None
    
    def _update_animation(self):
        """Update the animation frame based on current time and intensity"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Calculate time delta for hue shifting
        dt = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        # Update the current hue value based on shift speed
        self.current_hue = (self.current_hue + self.hue_shift_speed * dt) % 1.0
        
        # Get the wave frame for current intensity and time
        pixmap = self.frame_manager.get_frame_pixmap(
            intensity=self.current_intensity,
            time_position=elapsed,
            max_intensity=1.0
        )
        
        # Convert pixmap to PIL Image for color processing if wave color is enabled
        if self.wave_color_enabled:
            # Convert QPixmap to PIL Image using a safer method
            buffer = pixmap.toImage()
            buffer_width = buffer.width()
            buffer_height = buffer.height()
            
            # Create a PIL Image from QImage
            ptr = buffer.constBits()
            ptr.setsize(buffer_height * buffer_width * 4)
            array = np.array(ptr).reshape(buffer_height, buffer_width, 4)
            pil_image = Image.fromarray(array)
            
            # Apply hue shift with offset
            wave_hue = (self.current_hue + self.wave_hue_offset) % 1.0
            pil_image = self._apply_hue_shift(pil_image, hue_shift=wave_hue, saturation=self.saturation)
            
            # Convert back to QPixmap
            data = pil_image.tobytes("raw", "RGBA")
            qimage = QImage(data, pil_image.width, pil_image.height, QImage.Format_RGBA8888)
            pixmap = QPixmap.fromImage(qimage)
        
        # Scale pixmap to window size if needed
        if pixmap.width() != self.width() or pixmap.height() != self.height():
            pixmap = pixmap.scaled(
                self.width(), 
                self.height(),
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
        
        # Update screen overlay with new hue if enabled
        if hasattr(self, 'screen_enabled') and self.screen_enabled and hasattr(self, 'original_screen_img'):
            screen_img = self._apply_hue_shift(
                self.original_screen_img, 
                hue_shift=self.current_hue,
                saturation=self.saturation
            )
            
            # Apply opacity
            if self.screen_opacity < 1.0:
                alpha = screen_img.split()[3]
                alpha = alpha.point(lambda p: int(p * self.screen_opacity))
                screen_img.putalpha(alpha)
                
            # Convert to QPixmap
            data = screen_img.tobytes("raw", "RGBA")
            qimage = QImage(data, screen_img.width, screen_img.height, QImage.Format_RGBA8888)
            self.screen_overlay = QPixmap.fromImage(qimage)
        
        # Apply screen overlay if available
        if self.screen_overlay:
            # Create a new pixmap to compose the layers
            final_pixmap = QPixmap(pixmap.size())
            final_pixmap.fill(Qt.transparent)  # Start with transparent background
            
            # Create painter for composition
            painter = QPainter(final_pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            
            # Draw the wave frame as base layer
            painter.drawPixmap(0, 0, pixmap)
            
            # Scale and draw the screen overlay on top
            scaled_overlay = self.screen_overlay.scaled(
                pixmap.width(), 
                pixmap.height(),
                Qt.IgnoreAspectRatio,  # Changed to IgnoreAspectRatio to fit exactly
                Qt.SmoothTransformation
            )
            painter.drawPixmap(0, 0, scaled_overlay)
            painter.end()
            
            # Use the composited result
            pixmap = final_pixmap
        
        # Update display
        self.frame_label.setPixmap(pixmap)
    
    @pyqtSlot(float)
    def _on_intensity_update(self, intensity):
        """Handle intensity updates from audio processor or simulator"""
        self.current_intensity = intensity
    
    def mousePressEvent(self, event):
        """Handle mouse press events for dragging and context menu"""
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_start_position = event.pos()
        elif event.button() == Qt.RightButton:
            self._show_context_menu(event.pos())
            
    def mouseMoveEvent(self, event):
        """Handle mouse move events for dragging the window"""
        if self.dragging and event.buttons() & Qt.LeftButton:
            diff = event.pos() - self.drag_start_position
            new_pos = self.pos() + diff
            self.move(new_pos)
            
    def mouseReleaseEvent(self, event):
        """Handle mouse release events to end dragging"""
        if event.button() == Qt.LeftButton:
            self.dragging = False
            
    def mouseDoubleClickEvent(self, event):
        """Handle double-click events to open settings dialog"""
        self._show_settings_dialog()
    
    def _show_context_menu(self, position):
        """Show the context menu with visualizer options"""
        context_menu = QMenu(self)
        
        # Size submenu
        size_menu = context_menu.addMenu("Size")
        sizes = [150, 200, 250, 300, 350, 400]
        for size in sizes:
            action = size_menu.addAction(f"{size}x{size}")
            action.triggered.connect(lambda checked, s=size: self._set_visualizer_size(s))
        
        # Screen overlay opacity submenu
        opacity_menu = context_menu.addMenu("Screen Overlay")
        opacities = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for opacity in opacities:
            action = opacity_menu.addAction(f"{int(opacity * 100)}%")
            action.setCheckable(True)
            action.setChecked(abs(self.screen_opacity - opacity) < 0.01)
            action.triggered.connect(lambda checked, o=opacity: self._set_screen_opacity(o))
            
        # Always on top toggle
        always_on_top_action = context_menu.addAction("Always on Top")
        always_on_top_action.setCheckable(True)
        always_on_top_action.setChecked(self.windowFlags() & Qt.WindowStaysOnTopHint)
        always_on_top_action.triggered.connect(self._toggle_always_on_top)
        
        # Test speaking toggle
        speak_action = context_menu.addAction("Test Speaking")
        speak_action.setCheckable(True)
        speak_action.setChecked(self.audio_simulator.is_speaking)
        speak_action.triggered.connect(lambda checked: self.audio_simulator.set_speaking(checked))
        
        # Settings action
        settings_action = context_menu.addAction("Settings...")
        settings_action.triggered.connect(self._show_settings_dialog)
        
        # Exit action
        exit_action = context_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)
        
        # Show the menu
        context_menu.exec_(self.mapToGlobal(position))
    
    def _toggle_always_on_top(self, checked):
        """Toggle the always-on-top window flag"""
        flags = self.windowFlags()
        if checked:
            flags |= Qt.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowStaysOnTopHint
        
        # Need to hide and show to apply flag changes
        self.setWindowFlags(flags)
        self.show()
    
    def _show_settings_dialog(self):
        """Show a dialog with detailed visualizer settings"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Visualizer Settings")
        dialog.resize(400, 450)
        
        # Create layout for the dialog
        main_layout = QVBoxLayout(dialog)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Hue Shift Speed slider
        hue_speed_label = QLabel("Hue Shift Speed:", dialog)
        main_layout.addWidget(hue_speed_label)
        
        hue_speed_slider = QSlider(Qt.Horizontal, dialog)
        hue_speed_slider.setMinimum(0)
        hue_speed_slider.setMaximum(100)
        hue_speed_slider.setValue(int(self.hue_shift_speed * 100))
        hue_speed_slider.setTickPosition(QSlider.TicksBelow)
        hue_speed_slider.setTickInterval(10)
        main_layout.addWidget(hue_speed_slider)
        
        hue_speed_value = QLabel(f"{self.hue_shift_speed:.2f}", dialog)
        main_layout.addWidget(hue_speed_value)
        
        # Saturation slider
        saturation_label = QLabel("Saturation:", dialog)
        main_layout.addWidget(saturation_label)
        
        saturation_slider = QSlider(Qt.Horizontal, dialog)
        saturation_slider.setMinimum(0)
        saturation_slider.setMaximum(100)
        saturation_slider.setValue(int(self.saturation * 100))
        saturation_slider.setTickPosition(QSlider.TicksBelow)
        saturation_slider.setTickInterval(10)
        main_layout.addWidget(saturation_slider)
        
        saturation_value = QLabel(f"{self.saturation:.2f}", dialog)
        main_layout.addWidget(saturation_value)
        
        # Wave Hue Offset slider
        wave_offset_label = QLabel("Wave Hue Offset:", dialog)
        main_layout.addWidget(wave_offset_label)
        
        wave_offset_slider = QSlider(Qt.Horizontal, dialog)
        wave_offset_slider.setMinimum(0)
        wave_offset_slider.setMaximum(100)
        wave_offset_slider.setValue(int(self.wave_hue_offset * 100))
        wave_offset_slider.setTickPosition(QSlider.TicksBelow)
        wave_offset_slider.setTickInterval(10)
        main_layout.addWidget(wave_offset_slider)
        
        wave_offset_value = QLabel(f"{self.wave_hue_offset:.2f}", dialog)
        main_layout.addWidget(wave_offset_value)
        
        # Screen Opacity slider
        screen_opacity_label = QLabel("Screen Opacity:", dialog)
        main_layout.addWidget(screen_opacity_label)
        
        screen_opacity_slider = QSlider(Qt.Horizontal, dialog)
        screen_opacity_slider.setMinimum(0)
        screen_opacity_slider.setMaximum(100)
        screen_opacity_slider.setValue(int(self.screen_opacity * 100))
        screen_opacity_slider.setTickPosition(QSlider.TicksBelow)
        screen_opacity_slider.setTickInterval(10)
        main_layout.addWidget(screen_opacity_slider)
        
        screen_opacity_value = QLabel(f"{self.screen_opacity:.2f}", dialog)
        main_layout.addWidget(screen_opacity_value)
        
        # Checkboxes for toggles
        from PyQt5.QtWidgets import QCheckBox
        
        screen_enabled_cb = QCheckBox("Enable Screen Color Shifting", dialog)
        screen_enabled_cb.setChecked(self.screen_enabled)
        main_layout.addWidget(screen_enabled_cb)
        
        wave_color_cb = QCheckBox("Enable Wave Color Shifting", dialog)
        wave_color_cb.setChecked(self.wave_color_enabled)
        main_layout.addWidget(wave_color_cb)
        
        always_on_top_cb = QCheckBox("Always on Top", dialog)
        always_on_top_cb.setChecked(bool(self.windowFlags() & Qt.WindowStaysOnTopHint))
        main_layout.addWidget(always_on_top_cb)
        
        # Connect value change events
        def update_hue_speed():
            self.hue_shift_speed = hue_speed_slider.value() / 100
            hue_speed_value.setText(f"{self.hue_shift_speed:.2f}")
            
        def update_saturation():
            self.saturation = saturation_slider.value() / 100
            saturation_value.setText(f"{self.saturation:.2f}")
            
        def update_wave_offset():
            self.wave_hue_offset = wave_offset_slider.value() / 100
            wave_offset_value.setText(f"{self.wave_hue_offset:.2f}")
            
        def update_screen_opacity():
            self.screen_opacity = screen_opacity_slider.value() / 100
            screen_opacity_value.setText(f"{self.screen_opacity:.2f}")
            self._load_screen_overlay()
        
        # Connect signals
        hue_speed_slider.valueChanged.connect(update_hue_speed)
        saturation_slider.valueChanged.connect(update_saturation)
        wave_offset_slider.valueChanged.connect(update_wave_offset)
        screen_opacity_slider.valueChanged.connect(update_screen_opacity)
        
        screen_enabled_cb.toggled.connect(lambda checked: setattr(self, 'screen_enabled', checked))
        wave_color_cb.toggled.connect(lambda checked: setattr(self, 'wave_color_enabled', checked))
        always_on_top_cb.toggled.connect(self._toggle_always_on_top)
        
        # Add spacer
        main_layout.addStretch()
        
        # Execute the dialog
        dialog.exec_()
    
    def set_voice_intensity(self, intensity: float):
        """
        Set the voice intensity directly (to be called from external voice processing)
        
        Args:
            intensity: Audio intensity value (0.0 to 1.0)
        """
        self._on_intensity_update(intensity)
    
    def start_speaking(self):
        """Indicate that the assistant has started speaking"""
        self.audio_simulator.set_speaking(False)  # Disable simulator
        self._on_intensity_update(0.2)  # Set initial intensity
    
    def stop_speaking(self):
        """Indicate that the assistant has stopped speaking"""
        self._on_intensity_update(0.0)  # Reset intensity to 0
    
    def keyPressEvent(self, event):
        """Handle key press events for keyboard shortcuts"""
        # Escape or Ctrl+Q to exit
        if event.key() == Qt.Key_Escape or (event.key() == Qt.Key_Q and event.modifiers() & Qt.ControlModifier):
            self.close()
        else:
            super().keyPressEvent(event)
    
    def _set_screen_opacity(self, opacity):
        """
        Set the opacity of the screen overlay and reload it
        
        Args:
            opacity: Opacity value between 0.0 and 1.0
        """
        self.screen_opacity = opacity
        self._load_screen_overlay()  # Reload the screen overlay with new opacity
    
    def _apply_hue_shift(self, image, hue_shift=0.0, saturation=1.0):
        """
        Apply a hue shift to a PIL Image
        
        Args:
            image: PIL Image in RGBA format
            hue_shift: Amount of hue shift (0.0 to 1.0)
            saturation: Saturation level (0.0 to 1.0)
            
        Returns:
            PIL Image with hue shift applied
        """
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Split the image into RGBA channels
        r, g, b, a = image.split()
        
        # Convert RGB channels to HSV
        rgb_image = Image.merge('RGB', (r, g, b))
        hsv_image = rgb_image.convert('HSV')
        
        # Split into H, S, V channels
        h, s, v = hsv_image.split()
        
        # Apply hue shift - each pixel value is shifted by the same amount (mod 256)
        h_data = np.array(h)
        # Cast to int32 to avoid overflow during addition
        h_data = np.int32(h_data)
        h_data = (h_data + int(hue_shift * 255)) % 256
        h = Image.fromarray(h_data.astype('uint8'))
        
        # Apply saturation adjustment (increase saturation to make colors pop)
        s_data = np.array(s)
        s_data = np.clip(s_data * saturation * 1.2, 0, 255).astype('uint8')
        s = Image.fromarray(s_data)
        
        # Enhance brightness for better visibility
        v_data = np.array(v)
        v_data = np.clip(v_data * 1.15, 0, 255).astype('uint8')  # Increase brightness by 15%
        v = Image.fromarray(v_data)
        
        # Recombine HSV channels
        hsv_image = Image.merge('HSV', (h, s, v))
        
        # Convert back to RGB
        rgb_image = hsv_image.convert('RGB')
        
        # Recombine with alpha channel
        r, g, b = rgb_image.split()
        result = Image.merge('RGBA', (r, g, b, a))
        
        return result
    
    def _set_visualizer_size(self, size):
        """
        Set the size of the visualizer window
        
        Args:
            size: Size in pixels (both width and height)
        """
        # Update the fixed size constraint
        self.setFixedSize(size, size)
        
        # Update frame label size
        self.frame_label.setFixedSize(size, size)
        
        # Request an immediate animation update to reflect new size
        self._update_animation()
        
        # Center the window if position is off-screen
        screen_geometry = QApplication.desktop().screenGeometry()
        x = max(0, min(self.pos().x(), screen_geometry.width() - size))
        y = max(0, min(self.pos().y(), screen_geometry.height() - size))
        self.move(x, y)
        

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Bella Voice Visualizer")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, 
                        help=f"Size of the visualizer in pixels (default: {DEFAULT_SIZE})")
    parser.add_argument("--position", type=str, default=f"{DEFAULT_POSITION[0]},{DEFAULT_POSITION[1]}",
                        help=f"Position of the visualizer as X,Y (default: {DEFAULT_POSITION[0]},{DEFAULT_POSITION[1]})")
    parser.add_argument("--always-on-top", action="store_true", default=True,
                        help="Keep the visualizer on top of other windows")
    parser.add_argument("--frames-dir", type=str, default=None,
                        help="Directory containing wave frame images (default: ./wave_frames)")
    return parser.parse_args()


def main():
    """Main entry point"""
    # Parse command line arguments
    args = parse_args()
    
    # Parse position
    position = tuple(map(int, args.position.split(',')))
    
    # Create application
    app = QApplication(sys.argv)
    
    # Create and show visualizer window
    visualizer = VoiceVisualizerWindow(
        frames_dir=args.frames_dir,
        size=args.size,
        position=position
    )
    visualizer.show()
    
    # Start the application event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
