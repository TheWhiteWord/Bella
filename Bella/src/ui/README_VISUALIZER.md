# Bella Voice Visualizer

A configurable voice visualization tool for the Bella AI assistant that provides real-time visual feedback during speech.

## Features

- **Linux-optimized transparent floating window**
- **Fullscreen transparent overlay option**
- **Automatic audio intensity visualization**
- **Color-shifting visual effects**
- **Adjustable size and positioning**
- **Responsive drag and resize functionality**

## Running the Visualizer

### Basic Usage

The simplest way to run the visualizer is with the shell script:

```bash
./run_visualizer.sh
```

This will launch the default floating window visualizer with standard settings.

### Command Line Options

The following options are available when running the visualizer:

| Option | Description |
|--------|-------------|
| `--size=SIZE` | Set the size (diameter) of the visualizer in pixels |
| `--position=X,Y` | Position the visualizer at specific screen coordinates |
| `--always-on-top` | Keep the visualizer on top of other windows |
| `--no-simulation` | Disable audio simulation for real audio input |

### Examples

Launch a 300px visualizer at position (500,300):
```bash
./run_visualizer.sh --size=300 --position=500,300
```

Launch the fullscreen version:
```bash
./run_visualizer.sh --fullscreen
```

### Directly via Python

You can also run the visualizer directly with Python:

```bash
python3 run_visualizer.py [options]
```

## Visualizer Controls

### Mouse Controls

- **Left-click and drag**: Move the visualizer
- **Right-click**: Open settings menu
- **Double-click**: Open detailed settings panel

### Keyboard Shortcuts

- **Esc or Ctrl+Q**: Exit the visualizer

### Settings Menu

Right-click on the visualizer to access settings:

- **Visual Effects**: Configure hue shifting, saturation, and color relationships
- **Size**: Choose different sizes for the visualizer
- **Always On Top**: Toggle whether the visualizer stays on top of other windows
- **Test Animation**: Run test animations to verify functionality

## Visualization Modes

### Standard Floating Window

The default visualization mode creates a borderless, transparent floating window that can be positioned anywhere on the screen. This window can be dragged, resized, and configured through the settings menu.

## Technical Notes

### Window Manager Compatibility

The visualizer uses several different window manager hints to achieve proper transparency and click-through behavior. Different Linux window managers may handle these hints differently, so the best mode to use may depend on your desktop environment:

- **GNOME/Ubuntu**: Both modes should work well
- **KDE Plasma**: Fullscreen mode often works better
- **i3/Sway**: Standard floating mode typically works well
- **XFCE**: Standard floating mode is recommended
- **Wayland compositors**: Fullscreen mode is generally recommended

The visualizer attempts to automatically detect and apply the best settings for your environment, but you may need to experiment with different modes if you encounter issues.

### Transparency Methods

The visualizer tries several different approaches to achieve transparency, in this order:

1. Using the `-transparentcolor` attribute (supported by most X11 window managers)
2. Using the `-alpha` attribute for partial transparency
3. Applying X11-specific window properties via `xprop` (requires X11)

### Audio Integration

The visualizer can work with:

1. **Simulated audio** (default for testing)
2. **Real-time audio input** from Bella's speech processing

To use with real audio, use the `--no-simulation` flag and ensure the audio input source is properly configured.

## Troubleshooting

### Window Not Draggable

If you can't drag the visualizer window:

1. Try dragging from the thin border around the edge
2. Check if your window manager supports the transparency attributes
3. For KDE Plasma or Wayland, fullscreen mode typically works better

### Transparency Issues

If the window isn't transparent or has visual artifacts:

1. Try toggling the "Screen Effect Enabled" setting
2. Adjust the opacity in the detailed settings panel
3. For Wayland environments, transparency handling varies widely between compositors

### Visual Artifacts

If you see visual artifacts or transparency issues:

1. Try toggling the "Screen Effect Enabled" setting
2. Adjust the opacity in the detailed settings panel
3. Try a different size setting

### Performance Issues

If you experience lag or performance issues:

1. Try a smaller visualizer size
2. Disable the hue shifting effect
3. Check if compositing is enabled in your window manager

### Error Reports

If you encounter specific errors:

1. **"No attribute set_opacity"**: This has been fixed in the latest version
