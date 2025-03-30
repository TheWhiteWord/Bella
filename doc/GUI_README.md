# Bella Voice Assistant - GUI Guide

## Overview

Bella is a voice assistant with a graphical user interface that allows for natural speech interactions with local large language models. The application combines advanced speech recognition, local AI models, and high-quality text-to-speech in an easy-to-use interface.

![GUI Screenshot](https://placeholder-for-screenshot.png)

## Features

- **Voice-Based Chat**: Speak naturally with the assistant, which uses whisper-based speech recognition and local LLMs for responses
- **Context Enhancement**: Drag and drop a text file to provide additional context for more informed responses
- **Read Aloud**: Have the assistant read your context documents aloud with high-quality voice synthesis
- **Audio Download**: Save synthesized speech to WAV files for later use
- **Visual Feedback**: Real-time audio waveform visualization during speech playback

## Getting Started

### Prerequisites

- Python 3.10+
- Conda environment (recommended)
- PulseAudio or PipeWire for audio I/O
- Ollama for local LLM support

### Installation

1. Clone the repository
2. Set up the conda environment:
```bash
conda env create -f environment.yml
conda activate bella
```

3. Make sure Ollama is installed and running with at least one model (e.g., Lexi)
```bash
ollama pull lexi
ollama run lexi  # Test that it works
```

4. Launch the application with the GUI interface:
```bash
python launcher.py --gui
```

## Using the GUI

### Basic Interaction

1. When the application launches, it will immediately begin listening for your voice
2. Speak clearly and wait for the assistant to process your speech
3. The assistant will respond using synthesized speech
4. The conversation will continue with the assistant listening for your next input

### Working with Context

1. **Adding Context**:
   - Drag and drop any .txt file onto the application window
   - The Context button will change color to green and display "Context: ON"
   - The assistant will now use this information when responding to your questions

2. **Reading Context Aloud**:
   - After loading a context file, click the "Read Context" button (red) 
   - The assistant will read the entire text file aloud
   - The waveform visualization will show the audio being played

3. **Downloading Audio**:
   - After the context has been read aloud, the Download button becomes enabled
   - Click it to save the synthesized speech as a .wav file
   - Choose your desired location in the file save dialog

4. **Clearing Context**:
   - Click the green Context button to clear the loaded context
   - The button will return to its default state showing "Context: OFF"

### Command Line Options

The launcher supports several command line options:

- `--gui`: Start in GUI mode (default is CLI mode)
- `--sink`: Specify a PulseAudio sink name for audio output
- `--list-devices`: List available audio output devices
- `--model`: Specify which language model to use (e.g., "Lexi")

Example:
```bash
python launcher.py --gui --model Lexi --sink alsa_output.pci-0000_00_1f.3.analog-stereo
```

## Troubleshooting

### No Audio Input

- Make sure your microphone is properly connected and not muted
- Check that the system default input device is set correctly
- Try increasing microphone sensitivity in system settings

### No Audio Output

- Verify the speaker/headphones are connected and volume is up
- Check system audio settings to ensure default output is configured
- Run with `--list-devices` to see available sinks and specify one with `--sink`

### Application Not Responding to Speech

- Make sure you're speaking loudly enough (check the energy level indicator)
- Try reducing background noise
- Ensure the system microphone has proper permissions

### Waveform Visualization Not Showing

- Check that the audio monitor source exists for your output device
- Try using a different audio sink with the `--sink` parameter
- Make sure PulseAudio or PipeWire is properly installed and running

## Advanced Configuration

The application can be customized through configuration files:

- `models.yaml`: Configure available language models and parameters
- `prompts.yaml`: Modify system prompts for the LLM behavior

## Extending the Application

The modular design allows for easy extension:

- Add new TTS voices by modifying the Kokoro TTS wrapper
- Integrate different STT engines by implementing the appropriate interface
- Connect to alternative LLM providers by updating the LLM client

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Kokoro TTS for high-quality speech synthesis
- OpenAI's Whisper for speech recognition
- Ollama for local LLM support