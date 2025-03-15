# CSM-Multi Documentation

## Overview
CSM-Multi is a focused implementation of the Sesame CSM-1B voice cloning system, primarily designed for creating a single-voice chat companion. The core functionality prioritizes consistent, high-quality single-voice interaction, with additional features available as extensions.

## Primary Use Case
The main focus is creating a chat companion with a specific, consistent voice:
- Single voice cloning from reference audio
- Persistent voice characteristics across sessions
- Natural conversational flow
- Memory-efficient long-form conversations

## Core Features

### Voice Cloning
- Based on CSM-1B model with enhanced context handling
- Support for longer context windows (up to 4096 tokens)
- Improved audio preprocessing and normalization
- Voice characteristics preservation across sessions
- Optimized for single-voice continuous conversation

### Context Management
- `$CLEAR$` command for efficient memory management
- Automatic context pruning
- Reference audio persistence

### Audio Processing
- Sample Rate: 24kHz
- Supported Input Formats: WAV, MP3
- Automatic preprocessing:
  - Volume normalization
  - Audio resampling
  - Format conversion

## Technical Details

### Model Configuration
- Base Model: Sesame CSM-1B
- Required GPU: CUDA-compatible
- Python Version: 3.10+
- Key Dependencies:
  - torch==2.4.0
  - torchaudio==2.4.0
  - transformers==4.49.0
  - huggingface_hub==0.28.1

### Performance Considerations
- Context window size affects memory usage
- Recommended reference audio length: 2-3 minutes
- Maximum generated audio length: 25 seconds per segment

## Usage Guide

### Basic Setup
```python
# Example voice cloning setup
from enhanced_voice_clone import clone_voice

clone_voice(
    context_audio_path="reference.wav",
    context_text="Reference audio transcript",
    max_seq_len=4096
)
```

### Best Practices
1. Voice Cloning:
   - Use clear, high-quality reference audio
   - Provide accurate transcriptions
   - Keep reference audio between 2-3 minutes
   - Clear context regularly for optimal performance

2. Performance Optimization:
   - Monitor memory usage during long conversations
   - Keep individual responses under the maximum length
   - Use appropriate sequence length for your GPU

## Extended Features

For additional capabilities beyond single-voice cloning, including:
- Multi-speaker support
- Advanced voice cloning features
- Audio watermarking
- Batch processing

See [EXTENDED_FEATURES.md](EXTENDED_FEATURES.md) in the documentation.

## Troubleshooting

### Common Issues
1. CUDA Out of Memory:
   - Reduce context window size
   - Clear context more frequently
   - Reduce reference audio length

2. Generation Quality Issues:
   - Check audio input quality
   - Verify transcription accuracy
   - Adjust sequence length if needed

### Error Messages
- "Inputs too long": Reduce context or clear existing context
- "CUDA out of memory": Reduce batch size or context length

### Project structure

csm-multi/
├── enhanced_voice_clone.py  (main implementation)
├── generator.py            (core generation logic)
├── models.py              (model architecture)
├── watermarking.py        (audio watermarking)
├── download_models.py     (model setup)
├── requirements.txt       (dependencies)
├── LICENSE
├── README.md
├── clone_files/     (source files for cloning main voice) (not visible due to .gitignore)
|   ├── bella_edit.mp3         (voice audio reference)
|   └── Bella_transcript.txt   (transcript reference)
├── models/     (source files for cloning main voice) (not visible due to .gitignore)
|   ├── models--kyutai--moshiko-pytorch-bf16/
|   ├── models--meta-llama--Llama-3.2-1B/
|   └── models--sesame--csm-1b/
├── doc/                   (documenatation)
|   ├── EXTENDED_FEATURES.md         (voice cloning documentation)
|   ├── CLONING.md         (voice cloning documentation)
|   └── DOCUMENATATION.md  (main doc)
└── extended_features/     (additional features)
    ├── example.py         (multi speakers example)
    ├── multi_speaker.py   (multi speakes code)
    ├── conversation.py    (multi speaker conversation)
    ├── watermark.py       (watermark feature)
    ├── diffe.py           (multispeaker related file)
    ├── test_voice_clone.py  (old voice cloning test)
    └── test_bella_clone.py  (old test for cloning the main voice)

## References
- Original CSM Repository: [Sesame CSM](https://github.com/SesameAILabs/csm)
- Voice Cloning Base: [CSM Voice Cloning](https://github.com/isaiahbjork/csm-voice-cloning)
- HuggingFace Model: [CSM-1B](https://huggingface.co/sesame/csm-1b)