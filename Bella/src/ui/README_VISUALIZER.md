# Bella Voice Visualizer

A PyQt-based voice visualization tool for the Bella AI assistant that provides real-time visual feedback during speech. The visualizer uses pre-generated wave frames to create smooth animations that respond to voice intensity.

## Features

- **Linux-optimized transparent floating window**
- **Automatic audio intensity visualization**
- **Color-shifting visual effects**
- **Adjustable size and positioning**
- **Responsive drag and resize functionality**

## Overview

The voice visualizer provides a visual representation of Bella's voice output, with waves that:

1. Constantly move horizontally (phase animation)
2. Change intensity based on voice volume (amplitude changes)
3. React instantly to voice intensity changes

### Command Line Options

The following options are available when running the visualizer:

| Option | Description |
|--------|-------------|
| `--size=SIZE` | Set the size (diameter) of the visualizer in pixels |
| `--position=X,Y` | Position the visualizer at specific screen coordinates |
| `--always-on-top` | Keep the visualizer on top of other windows (default) |
| `--frames-dir=DIR` | Directory containing wave frame images |
| `--no-simulation` | Disable audio simulation and use real audio input |

## Visualizer Controls

### Mouse Controls

- **Left-click and drag**: Move the visualizer
- **Right-click**: Open settings menu
- **Double-click**: Open detailed settings panel

### Keyboard Shortcuts

- **Esc or Ctrl+Q**: Exit the visualizer

### Settings Menu

Right-click on the visualizer to access settings:

- **Size**: Choose different sizes for the visualizer
- **Always On Top**: Toggle whether the visualizer stays on top of other windows
- **Test Speaking**: Enable/disable simulated speech for testing
- **Exit**: Close the visualizer

## Visualization Modes

### Floating Window

The visualization displays as a borderless, transparent floating window that can be positioned anywhere on the screen. This window can be dragged, resized, and configured through the right-click settings menu.

## Installation and Setup

The visualizer is built into Bella and uses the existing conda environment. Make sure your environment is activated:

```bash
conda activate bella
```

### Running the Visualizer

To launch the visualizer, run:

```bash
python src/ui/bella_visualizer.py
```

Example with options:

```bash
python src/ui/bella_visualizer.py --size=300 --position=500,300 --no-simulation
```

## Technical Notes

### Window Manager Compatibility

The visualizer uses PyQt window manager hints to achieve proper transparency and click-through behavior. Different Linux window managers may handle these hints differently, so minor adjustments might be needed for your specific desktop environment.

### Audio Integration

The visualizer can work with:

1. **Simulated audio** (default for testing): Enabled by default, creates realistic voice patterns without real audio
2. **Real-time audio input**: Connects to Bella's speech processing for live visualization

To use with real audio, use the `--no-simulation` flag.

### Integration With Bella

To integrate the visualizer with your Bella voice assistant code:

```python
from src.ui.bella_visualizer import BellaVisualizer

# Create and start the visualizer
visualizer = BellaVisualizer(use_real_audio=True)
visualizer.show()

# When Bella starts speaking
visualizer.audio_integration.on_speech_start()

# For each audio chunk produced during TTS
visualizer.audio_integration.handle_audio_callback(audio_data)

# When Bella stops speaking
visualizer.audio_integration.on_speech_end()
```

### Customization

The visualizer uses pre-generated frames created by `wave_frame_generator.py`. If you want to customize the appearance, you can:

1. Modify the source images in the `elements` directory
2. Adjust parameters in `wave_frame_generator.py`
3. Regenerate frames with:
   ```bash
   python src/ui/wave_frame_generator.py
   ```

