# CSM Voice Integration

## Overview
This document outlines the voice processing architecture focusing on CSM's TTS system and audio sentiment analysis capabilities. For implementation details of the TTS component, see [Technical Plan](TECHNICAL_PLAN.md).

## Architecture Components

### 1. Input Processing Pipeline
```
Speech Input --> Audio Feature Extraction
                       |
                       v
                 Sentiment Analysis (Mimi Tokenizer)
                       |
                       v
                    CSM TTS
```

### 2. Processing Layers

#### Audio Processing Layer
- **Sampling Rate**: 24kHz (CSM native)
- **Feature Extraction**: Using Mimi tokenizer's capabilities
  - Semantic tokens (zeroth codebook)
  - Acoustic tokens (remaining codebooks)
- **Memory Usage**: ~100MB for feature extraction
- **Shared Processing**: Utilizes CSM's existing tokenization pipeline

#### Sentiment Analysis Layer
- **Approach**: Leveraging CSM's compute amortization (1/16th sampling)
- **Features Analyzed**:
  - Energy levels (from acoustic tokens)
  - Pitch variation (from semantic tokens)
  - Speaking rate (computed from frame timing)
  - Emotional markers (derived from semantic analysis)
- **Memory Footprint**: Minimal due to amortized computation
- **Integration Point**: Direct access to Mimi tokenizer output

### 3. CSM TTS Integration

#### Emotion Mapping
```
Sentiment Analysis --> CSM Parameters
[happy] -> temperature: 0.8, top_k: 30
[calm]  -> temperature: 0.5, top_k: 20
[excited] -> temperature: 0.9, top_k: 35
```

#### Context Management
- **Short-term**: Current conversation state
- **Long-term**: Voice profile characteristics
- **Memory Usage**: ~2-4GB

## Implementation Considerations

### 1. Performance Impact

#### GPU Memory Usage
- CSM Base: 2.5-3GB
- Feature Extraction: Shared with CSM (~0MB additional)
- Sentiment Analysis: ~50MB
- Context Storage: ~500MB
- **Total**: ~3-4GB

#### Processing Latency
- Feature Extraction: Shared with CSM (no additional latency)
- Sentiment Analysis: 20-30ms (amortized)
- CSM Generation: 1-2s per second of audio

### 2. Optimization Techniques

1. **Compute Amortization**
   - Use 1/16th sampling for sentiment analysis (CSM's technique)
   - Share Mimi tokenizer for all audio processing
   - Batch process audio features during generation

2. **Memory Management**
   - Dynamic tensor clearing
   - Shared cache between components
   - Efficient context pruning
   - Single copy of audio features

### 3. Integration Points

#### Audio Feature Sharing
```python
# Conceptual Integration
class SharedFeatureExtractor:
    def extract(audio: torch.Tensor):
        # Use CSM's Mimi tokenizer (already loaded)
        features = mimi_tokenizer.encode(audio)
        return {
            'csm_tokens': features[:, 0],     # For TTS
            'semantic_features': features[:, 0],  # For sentiment
            'acoustic_features': features[:, 1:]  # For detailed analysis
        }
```

## Technical Requirements

### Hardware Requirements
- GPU: 8GB+ VRAM recommended
- RAM: 16GB recommended
- CUDA: 11.x or higher

### Software Dependencies
- PyTorch with CUDA support
- torchaudio
- Additional dependencies in requirements.txt

## Implementation Phases

### Phase 1: Basic Integration
- [x] Shared feature extraction
- [x] Basic sentiment analysis
- [x] Simple emotion mapping

### Phase 2: Enhanced Processing
- [ ] Full sentiment analysis pipeline
- [ ] Context-aware processing
- [ ] Optimized memory management

### Phase 3: Advanced Features
- [ ] Real-time adaptation
- [ ] Dynamic voice profile updates
- [ ] Conversation flow optimization

## Usage Example

```python
# Conceptual flow
async def process_conversation_turn(audio_input):
    # 1. Extract shared features
    features = feature_extractor.extract(audio_input)
    
    # 2. Analyze sentiment
    sentiment = sentiment_analyzer.process(
        features['sentiment_features']
    )
    
    # 3. Generate emotional speech
    speech = csm_generator.generate(
        text=text_input,
        emotion_params=EMOTION_PARAMS[sentiment]
    )
    
    return speech
```

## Limitations and Considerations

1. **Technical Limitations**
   - Initial latency overhead
   - Memory requirements
   - GPU dependencies

2. **Integration Challenges**
   - Resource management
   - Error handling
   - Performance optimization

3. **Future Research Areas**
   - Improved emotion detection
   - Real-time adaptation
   - Resource optimization

## References

1. CSM Research Paper
2. PyTorch CUDA Documentation
3. Sesame Voice Presence Research