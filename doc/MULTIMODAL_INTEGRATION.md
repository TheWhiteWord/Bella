# Multimodal LLM and CSM TTS Integration

## Overview
This document outlines the integration architecture between a multimodal ONNX-based LLM and the CSM TTS system, focusing on audio sentiment analysis and emotional speech generation. For implementation details of the TTS component, see [Technical Plan](TECHNICAL_PLAN.md).

## Architecture Components

### 1. Input Processing Pipeline
```
Speech Input --> Audio Feature Extraction
                       |
                       v
                 Sentiment Analysis (Mimi Tokenizer)
                       |
                       v
Multimodal LLM <-- Context Integration --> CSM TTS
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

#### Multimodal LLM Layer (ONNX)
- **Input**: 
  - Semantic tokens from Mimi tokenizer
  - Derived sentiment features
  - Text transcription
- **Output**: Structured response with emotional markers
- **Integration Point**: Pre-processed audio features

### 3. CSM TTS Integration

#### Emotion Mapping
```
LLM Emotional Output --> CSM Parameters
[happy] -> temperature: 0.8, top_k: 30
[calm]  -> temperature: 0.5, top_k: 20
[excited] -> temperature: 0.9, top_k: 35
```

#### Context Management
- **Short-term**: Current conversation state
- **Long-term**: Voice profile characteristics
- **Memory Usage**: ~2-4GB (shared with CSM)

## Implementation Considerations

### 1. Performance Impact

#### GPU Memory Usage
- CSM Base: 2.5-3GB
- Feature Extraction: Shared with CSM (~0MB additional)
- Sentiment Analysis: ~50MB
- Context Storage: ~500MB
- **Total Additional**: ~550MB (reduced from previous estimate due to shared processing)

#### Processing Latency
- Feature Extraction: Shared with CSM (no additional latency)
- Sentiment Analysis: 20-30ms (amortized)
- LLM Processing: Model dependent
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

3. **Pipeline Optimization**
   - Parallel feature processing
   - Cached emotional patterns
   - Pre-computed voice profiles
   - Unified tokenization pipeline

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

#### Emotion Mapping
```python
# Emotion to TTS parameters
EMOTION_PARAMS = {
    'happy': {'temp': 0.8, 'top_k': 30},
    'calm': {'temp': 0.5, 'top_k': 20},
    'excited': {'temp': 0.9, 'top_k': 35}
}
```

## Technical Requirements

### Hardware Requirements
- GPU: 12GB+ VRAM recommended
- RAM: 32GB recommended
- CUDA: 11.x or higher

### Software Dependencies
- PyTorch with CUDA support
- ONNX Runtime
- torchaudio
- transformers
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
    
    # 3. Get LLM response with emotion
    llm_response = await llm.generate(
        audio_features=features['csm_tokens'],
        sentiment=sentiment
    )
    
    # 4. Generate emotional speech
    speech = csm_generator.generate(
        text=llm_response.text,
        emotion_params=EMOTION_PARAMS[llm_response.emotion]
    )
    
    return speech
```

## Future Enhancements

1. **Real-time Adaptation**
   - Dynamic emotion adjustment
   - Continuous learning from interaction
   - Adaptive voice profiles

2. **Advanced Context Management**
   - Long-term memory integration
   - Personality consistency
   - Cross-session continuity

3. **Performance Optimizations**
   - Enhanced compute amortization
   - Dynamic resource allocation
   - Cached response patterns

## Limitations and Considerations

1. **Technical Limitations**
   - Initial latency overhead
   - Memory requirements
   - GPU dependencies

2. **Integration Challenges**
   - Synchronization between components
   - Error propagation
   - Resource contention

3. **Future Research Areas**
   - Improved emotion detection
   - Real-time adaptation
   - Resource optimization

## References

1. CSM Research Paper
2. ONNX Runtime Documentation
3. PyTorch CUDA Documentation
4. Sesame Voice Presence Research