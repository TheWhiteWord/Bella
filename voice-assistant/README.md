# Voice Assistant Integration

A modular implementation combining Phi-4 Multimodal LLM for speech recognition and CSM TTS for voice generation.

## Structure
```
voice-assistant/
├── config/                  # Configuration files
│   └── settings.yaml       # Main configuration
├── models/                  # Model loading and management
│   ├── llm.py             # Phi-4 multimodal handling
│   └── tts.py             # CSM TTS handling
├── audio/                  # Audio processing
│   ├── recorder.py        # Microphone recording
│   └── processor.py       # Audio preprocessing
├── pipeline/              # Core pipeline components
│   ├── manager.py        # Pipeline orchestration
│   └── conversation.py    # Conversation handling
└── utils/                 # Utility functions
    ├── memory.py         # Memory management
    └── logger.py         # Logging utilities

## Getting Started

1. Ensure your environment has all dependencies installed
2. Configure settings.yaml with your model paths and preferences
3. Run the example script to test the integration