"""Audio context processor for enhanced emotional understanding."""
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

@dataclass
class AudioContext:
    """Container for audio context information"""
    sentiment_score: float  # -1 to 1
    energy_level: float    # 0 to 1
    pitch_variance: float  # 0 to 1
    speaking_rate: float   # words per second
    emotion_label: str     # predicted emotion
    confidence: float      # prediction confidence

class AudioContextProcessor:
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.feature_extractor = AudioFeatureExtractor(sample_rate)
        self.context_window = []  # Store recent audio contexts
        self.max_context_length = 5  # Keep last 5 utterances for context

    def process_audio(self, audio: torch.Tensor) -> AudioContext:
        """Extract context information from audio input"""
        # Extract basic audio features
        features = self.feature_extractor.extract_features(audio)
        
        # Basic sentiment analysis based on audio features
        sentiment_score = self._analyze_sentiment(features)
        
        # Determine emotion from features
        emotion, confidence = self._classify_emotion(features)
        
        context = AudioContext(
            sentiment_score=sentiment_score,
            energy_level=features['energy_level'],
            pitch_variance=features['pitch_variance'],
            speaking_rate=features['speaking_rate'],
            emotion_label=emotion,
            confidence=confidence
        )
        
        # Update context window
        self.context_window.append(context)
        if len(self.context_window) > self.max_context_length:
            self.context_window.pop(0)
            
        return context

    def get_tts_parameters(self, audio_context: AudioContext) -> Dict[str, float]:
        """Convert audio context to TTS control parameters"""
        # Map sentiment and emotion to TTS parameters
        params = {
            'temperature': self._map_emotion_to_temperature(audio_context),
            'top_k': self._map_emotion_to_topk(audio_context),
            'speaking_rate': audio_context.speaking_rate,
            'energy_scale': audio_context.energy_level
        }
        
        return params

    def get_emotion_marker(self, audio_context: AudioContext) -> str:
        """Generate emotion marker for TTS text based on audio context"""
        # Map sentiment and features to emotion markers
        if audio_context.sentiment_score > 0.5:
            return "[happy]" if audio_context.energy_level > 0.5 else "[content]"
        elif audio_context.sentiment_score < -0.5:
            return "[sad]" if audio_context.energy_level < 0.5 else "[angry]"
        elif audio_context.energy_level > 0.7:
            return "[excited]"
        elif audio_context.energy_level < 0.3:
            return "[calm]"
        else:
            return "[neutral]"

    def _analyze_sentiment(self, features: Dict[str, float]) -> float:
        """Basic sentiment analysis from audio features"""
        # Simple sentiment heuristic based on energy and pitch variance
        sentiment = (
            features['energy_level'] * 0.6 +
            features['pitch_variance'] * 0.4
        )
        return (sentiment * 2) - 1  # Scale to -1 to 1

    def _classify_emotion(self, features: Dict[str, float]) -> Tuple[str, float]:
        """Classify emotion based on audio features"""
        # Simple rule-based emotion classification
        energy = features['energy_level']
        pitch_var = features['pitch_variance']
        
        if energy > 0.7 and pitch_var > 0.6:
            return "excited", 0.8
        elif energy < 0.3 and pitch_var < 0.3:
            return "sad", 0.7
        elif energy > 0.6 and pitch_var < 0.4:
            return "angry", 0.6
        elif energy > 0.5 and pitch_var > 0.4:
            return "happy", 0.7
        elif energy < 0.4 and pitch_var < 0.4:
            return "calm", 0.6
        else:
            return "neutral", 0.5

    def _map_emotion_to_temperature(self, context: AudioContext) -> float:
        """Map emotion to temperature parameter"""
        emotion_temp_map = {
            "excited": 0.9,
            "happy": 0.8,
            "angry": 0.85,
            "neutral": 0.7,
            "calm": 0.5,
            "sad": 0.6
        }
        return emotion_temp_map.get(context.emotion_label, 0.7)

    def _map_emotion_to_topk(self, context: AudioContext) -> int:
        """Map emotion to top-k parameter"""
        emotion_topk_map = {
            "excited": 35,
            "happy": 30,
            "angry": 25,
            "neutral": 25,
            "calm": 20,
            "sad": 20
        }
        return emotion_topk_map.get(context.emotion_label, 25)


class AudioFeatureExtractor:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=80
        )

    def extract_features(self, audio: torch.Tensor) -> Dict[str, float]:
        """Extract relevant features from audio"""
        # Convert to mono if needed
        if audio.dim() > 1:
            audio = audio.mean(dim=0)
            
        # Calculate energy level
        energy = torch.mean(torch.abs(audio)).item()
        
        # Calculate pitch variance using mel spectrogram
        mel_spec = self.mel_transform(audio)
        pitch_var = torch.std(torch.mean(mel_spec, dim=1)).item()
        
        # Estimate speaking rate (simplified)
        speaking_rate = self._estimate_speaking_rate(audio)
        
        return {
            'energy_level': min(energy * 10, 1.0),  # Normalize to 0-1
            'pitch_variance': min(pitch_var * 5, 1.0),  # Normalize to 0-1
            'speaking_rate': speaking_rate
        }

    def _estimate_speaking_rate(self, audio: torch.Tensor) -> float:
        """Estimate speaking rate from audio"""
        # Simple energy-based syllable detection
        energy = torch.abs(audio)
        threshold = torch.mean(energy) * 1.2
        peaks = torch.where(energy > threshold)[0]
        
        # Count peaks with minimum distance
        min_distance = int(0.1 * self.sample_rate)  # 100ms minimum between peaks
        valid_peaks = 1
        last_peak = peaks[0] if len(peaks) > 0 else 0
        
        for peak in peaks[1:]:
            if peak - last_peak >= min_distance:
                valid_peaks += 1
                last_peak = peak
                
        duration = len(audio) / self.sample_rate
        rate = valid_peaks / duration
        
        return min(rate, 5.0)  # Cap at 5 syllables/sec