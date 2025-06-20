# Bella - Voice Assistant with GUI

A modern voice assistant that features both a graphical user interface and command-line interface. Bella uses local LLMs via Ollama, high-quality Kokoro TTS for speech synthesis, and Whisper for speech recognition.

## Features

- **Dual Interface:** Choose between an intuitive GUI or efficient CLI mode
- **Local Processing:** Runs entirely on your local machine with no cloud dependencies
- **High-Quality TTS:** Kokoro TTS provides natural-sounding speech output
- **Accurate Speech Recognition:** Whisper-based STT with voice activity detection
- **Context-Aware Interactions:** Add context to enhance assistant responses
- **Waveform Visualization:** Real-time audio visualization during playback
- **PulseAudio/PipeWire Integration:** Uses native Linux audio subsystem for I/O

## Getting Started

### Prerequisites
- Python 3.10+
- Conda environment (recommended)
- PulseAudio or PipeWire for audio I/O
- Ollama for local LLM support

### Installation

1. **Set up the conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate bella
   ```

2. **Ensure Ollama is installed and running:**
   ```bash
   # Install Ollama if needed: https://ollama.com/download
   ollama pull lexi  # Or any other model you prefer
   ```

3. **Launch the application:**
   ```bash
   # GUI Mode
   python launcher.py --gui
   
   # CLI Mode
   python launcher.py
   ```

## Usage

### GUI Mode

The graphical interface provides an intuitive way to interact with the voice assistant:

1. **Voice Chat:** The assistant begins listening automatically - just speak!
2. **Context Loading:** Drag and drop any .txt file to provide context for more informed responses
3. **Read Context:** Have the assistant read your documents aloud
4. **Audio Download:** Save synthesized speech as WAV files

For detailed information about GUI usage, see [GUI_README.md](/doc/GUI_README.md).

### CLI Mode

The command-line interface provides a more streamlined experience:

1. **Start the assistant:** `python launcher.py`
2. **Speak when prompted:** The assistant indicates when it's listening
3. **Get responses:** The assistant processes your speech and responds automatically

### Command-Line Options

```bash
python launcher.py [options]

Options:
  --gui               Start in GUI mode (default is CLI mode)
  --model MODEL       Specify which LLM to use
  --sink SINK         Specify PulseAudio sink name for audio output
  --list-devices      List available audio output devices and exit
```

## Components

- **BackendAdapter:** Bridges GUI and async backend components
- **BufferedRecorder:** Handles audio recording with VAD
- **AudioSessionManager:** Manages speech processing sessions
- **KokoroTTSWrapper:** Provides high-quality text-to-speech
- **VoiceAssistantGUI:** Implements the graphical interface

## Technical Details

- **Speech Recognition:** Whisper model for accurate transcription
- **LLM Integration:** Local models via Ollama (Lexi, Mistral, etc.)
- **Text-to-Speech:** Kokoro TTS with custom voice profiles
- **Audio I/O:** PulseAudio/PipeWire for native Linux audio support
- **Threading Model:** Mix of threading and asyncio for responsiveness

## Documentation

- [GUI Guide](/doc/GUI_README.md) - User guide for the graphical interface
- [Dev Documentation](/doc/Dev/Dev.md) - Technical details for developers
- [Kokoro TTS](/doc/Kokoro.md) - Information about the TTS system
- [Ollama Integration](/doc/OLLAMA_LIBRARY.md) - Details on LLM integration

## Contributing

Feel free to submit issues, fork the repository and send pull requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Kokoro TTS](https://github.com/hexgrad/kokoro) for high-quality speech synthesis
- [OpenAI's Whisper](https://github.com/openai/whisper) for speech recognition
- [Ollama](https://ollama.com/) for local LLM support
