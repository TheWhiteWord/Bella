# Voice Visualizer Integration Guide

This document explains how to integrate the voice visualizer into Bella Assistant.

## Overview

The voice visualizer provides a visual representation of Bella's voice output, with waves that:
1. Constantly move horizontally (phase animation)
2. Change intensity based on voice volume (amplitude changes)
3. Overlay constantly and gradually shifting hue

## Hue-Shifting Features

The visualizer now includes dynamic color-shifting capabilities:

- **Wave Hue Shifting**: Continuously shifts the hue of the wave animation
- **Screen Overlay Hue Shifting**: Applies dynamic color to the screen overlay
- **Customizable Settings**: Control hue shift speed, saturation, wave/screen color offsets and opacity

## Integration Steps

1. Import and initialize the visualizer:

   ```python
   from src.ui.bella_visualizer import BellaVisualizer
   
   # Create and start the visualizer
   visualizer = BellaVisualizer(use_real_audio=True)
   visualizer.show()
   ```

2. Connect to Bella's speech system:

   ```python
   # When Bella starts speaking
   visualizer.audio_integration.on_speech_start()
   
   # For each audio chunk produced during TTS
   visualizer.audio_integration.handle_audio_callback(audio_data)
   
   # When Bella stops speaking
   visualizer.audio_integration.on_speech_end()
   ```

## User Controls

Users can access visualizer settings through:

1. **Right-click menu**: Quick access to size, opacity, and always-on-top settings
2. **Double-click**: Opens detailed settings dialog for color and animation controls
3. **Settings Dialog**: Provides sliders for:
   - Hue shift speed
   - Color saturation
   - Wave hue offset (relative to screen)
   - Screen overlay opacity
   - Toggle options for wave and screen color shifting

## Testing

Use the included test script to verify the visualizer works correctly:

```bash
cd src/ui
python test_visualizer.py
```

## Hue-Shifting Features

The visualizer now includes dynamic color-shifting capabilities:

- **Wave Hue Shifting**: Continuously shifts the hue of the wave animation 
- **Screen Overlay Hue Shifting**: Applies dynamic color to the screen overlay
- **Customizable Settings**: Control hue shift speed, saturation, wave/screen color offsets and opacity

## Integration Steps

1. Import and initialize the visualizer:
   ```python
   from src.ui.bella_visualizer import BellaVisualizer
   
   # Create and start the visualizer
   visualizer = BellaVisualizer(use_real_audio=True)
   visualizer.show()
   ```

2. Connect to Bella's speech system:
   ```python
   # When Bella starts speaking
   visualizer.audio_integration.on_speech_start()
   
   # For each audio chunk produced during TTS
   visualizer.audio_integration.handle_audio_callback(audio_data)
   
   # When Bella stops speaking
   visualizer.audio_integration.on_speech_end()
   ```

## User Controls

Users can access visualizer settings through:

1. **Right-click menu**: Quick access to size, opacity, and always-on-top settings
2. **Double-click**: Opens detailed settings dialog for color and animation controls
3. **Settings Dialog**: Provides sliders for:
   - Hue shift speed
   - Color saturation
   - Wave hue offset (relative to screen)
   - Screen overlay opacity
   - Toggle options for wave and screen color shifting

## Testing

Use the included test script to verify the visualizer works correctly:
```bash
cd src/ui
python test_visualizer.py
```

