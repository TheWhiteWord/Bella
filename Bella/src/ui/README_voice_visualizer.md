# Voice Visualizer Integration Guide

This document explains how to integrate the voice visualizer into Bella Assistant.

## Overview

The voice visualizer provides a visual representation of Bella's voice output, with waves that:
1. Constantly move horizontally (phase animation)
2. Change intensity based on voice volume (amplitude changes)

## Files Structure

```
Bella/src/ui/
├── voice_visualizer.py            # Core visualizer implementation
├── voice_visualizer_integration.py # Integration with audio system
├── floating_visualizer_linux.py    # Complete UI with visualization
├── wave_frames/                   # Generated animation frames
└── test/
    └── voice_visualizer_test.py   # Test script for visualization
```

## Integration Steps

### 1. Run the frame generator first

Before using the visualizer, you need to generate the wave frames:

```bash
cd /media/theww/AI/Code/AI/Bella
conda activate bella
python wave_frame_generator.py
```

This will create the necessary animation frames in the `wave_frames` directory.

### 2. Import the visualizer in your UI

```python
from Bella.src.ui.voice_visualizer_integration import BellaVoiceVisualizer
```

### 3. Create a visualizer instance

```python
# In your UI initialization:
self.visualizer = BellaVoiceVisualizer(
    parent=your_tkinter_container,
    size=(400, 400)  # Adjust size as needed
)

# Initialize it asynchronously
await self.visualizer.initialize()
```

### 4. Connect to Bella's audio system

You need to create a callback adapter that connects the audio system to the visualizer:

```python
class AudioVisualizerAdapter:
    def __init__(self, visualizer):
        self.visualizer = visualizer
        
    async def on_speech_start(self):
        await self.visualizer.start_visualization()
        
    async def on_speech_end(self):
        await self.visualizer.stop_visualization()
        
    def process_audio(self, audio_data, sample_rate):
        self.visualizer.process_audio(audio_data, sample_rate)

# Create and register the adapter
adapter = AudioVisualizerAdapter(self.visualizer)
bella_audio_system.register_audio_callback(adapter)
```

## Testing

You can test the visualizer without integrating with Bella using:

```bash
python -m Bella.src.ui.test.voice_visualizer_test
```

This will show various test patterns that demonstrate the visualizer's capabilities.

## Example Implementation

For a complete implementation example, see `bella_ui_with_visualization.py`,
which demonstrates integrating the visualizer into a full Bella UI.
