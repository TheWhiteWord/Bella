# CSM-Multi Extended Features

This document describes the extended features of CSM-Multi that are not part of the core single-voice functionality.
These features are maintained for future development but are not essential for the primary use case.

## Multi-Speaker Support

The multi-speaker functionality allows generating conversations between multiple voices:

### Features
- Support for up to 4 different speakers in a single generation
- Text separation using `||` syntax for different speakers
- Speaker offset notation for flexible voice assignment

### Syntax Examples
```
"Hello there!||Hi, how are you?||I'm doing great!"
```
Each `||` separates different speakers, with default offset increments.

### Speaker Offsets
You can explicitly set speaker offsets by appending numbers:
```
"First speaker||Second speaker||Back to first speaker.0"
```
The `.0` returns to the first speaker's voice.

### Commands
* `$SWAP$` - Switch to next speaker index
* `$BACK$` - Switch to previous speaker index
* `$CLEAR$` - Clear conversation context (also available in core features)

## Voice Cloning Enhancements

### Advanced Features
- Support for longer audio samples (>3 minutes)
- Batch processing capabilities
- Multiple reference audio support
- Custom voice characteristic adjustments

### Performance Optimizations
- Memory-efficient context management
- Batch processing for multiple generations
- Caching for frequently used voice profiles

## Watermarking Support

The system includes optional audio watermarking capabilities:
- Secure watermarking of generated audio
- Watermark verification
- Configurable watermark strength
- requires dependency (silentcipher @ git+https://github.com/SesameAILabs/silentcipher@master)

## Project Structure
```
extended_features/
├── __init__.py          (module initialization)
├── conversation.py      (conversation management)
├── multi_speaker.py     (multi-speaker generation)
├── watermarking.py      (audio watermarking)
├── example.py          (usage examples)
├── voice_clone.py      (extended cloning features)
└── test_bella_clone.py (testing utilities)
```

## Usage Examples

### Multi-Speaker Conversation
```python
from extended_features import MultiSpeakerGenerator, ConversationManager

# Initialize multi-speaker generator
generator = MultiSpeakerGenerator()

# Create conversation with multiple speakers
text = "Hello!||Hi there!||How are you?0"
output = generator.generate_multi_speaker(text)
```

### Advanced Voice Cloning
```python
from extended_features.voice_clone import VoiceCloner

cloner = VoiceCloner()
cloner.add_reference_audio("reference.wav", "Reference text")
cloner.clone_voice("Text to synthesize", speaker_offset=1)
```

## Future Development

Planned enhancements:
- [ ] Extended speaker limit beyond 4
- [ ] Advanced voice mixing capabilities
- [ ] Real-time voice switching
- [ ] Voice style transfer between speakers

## Integration with Core Features

These extended features are built on top of the core functionality but kept separate to maintain code clarity and performance. When needed, they can be integrated with the basic voice cloning system:

```python
# Example of combining core and extended features
from enhanced_voice_clone import clone_voice
from extended_features import MultiSpeakerGenerator

# Initialize both systems
cloner = clone_voice(...)
multi_speaker = MultiSpeakerGenerator()

# Use them together
multi_speaker.add_voice(cloner.context_segments[0])
```

For basic voice cloning needs, refer to the main [DOCUMENTATION.md](DOCUMENTATION.md).