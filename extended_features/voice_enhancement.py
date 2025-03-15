"""Voice enhancement features for emotional context and conversation flow."""
from typing import Tuple
import torch
import re

class VoiceEnhancement:
    # Emotion markers and their associated parameters
    EMOTION_SETTINGS = {
        'calm': {'temp': 0.5, 'top_k': 20},
        'neutral': {'temp': 0.7, 'top_k': 25},
        'happy': {'temp': 0.8, 'top_k': 30},
        'excited': {'temp': 0.9, 'top_k': 35},
        'sad': {'temp': 0.6, 'top_k': 20},
    }

    # Punctuation-based pause durations (in ms)
    PAUSE_DURATIONS = {
        '.': 500,
        '!': 400,
        '?': 450,
        ',': 200,
        ';': 300,
        '...': 600,
    }

    @staticmethod
    def process_text(text: str) -> Tuple[str, dict]:
        """Process text to extract emotion markers and get generation parameters."""
        # Default settings
        params = {'temperature': 0.7, 'top_k': 25}
        
        # Check for emotion markers like [happy], [excited], etc.
        emotion_match = re.match(r'\[(.*?)\](.*)', text)
        if emotion_match:
            emotion, clean_text = emotion_match.groups()
            if emotion in VoiceEnhancement.EMOTION_SETTINGS:
                params.update(VoiceEnhancement.EMOTION_SETTINGS[emotion])
                return clean_text.strip(), params

        # Dynamic adjustment based on punctuation
        if text.endswith('!'):
            params['temperature'] = min(params['temperature'] + 0.1, 0.9)
        elif text.endswith('?'):
            params['temperature'] = min(params['temperature'] + 0.05, 0.85)
            
        return text, params

    @staticmethod
    def add_pauses(audio: torch.Tensor, text: str, sample_rate: int) -> torch.Tensor:
        """Add natural pauses based on punctuation."""
        # Convert pause durations from ms to samples
        pause_samples = {k: int(v * sample_rate / 1000) 
                        for k, v in VoiceEnhancement.PAUSE_DURATIONS.items()}
        
        # Add a small pause buffer
        pause_buffer = torch.zeros(int(0.1 * sample_rate))
        
        # Add longer pauses at punctuation marks
        for punct, duration in pause_samples.items():
            if punct in text:
                pause = torch.zeros(duration)
                audio = torch.cat([audio, pause])
        
        return audio

    @staticmethod
    def adjust_prosody(audio: torch.Tensor, text: str) -> torch.Tensor:
        """Basic prosody adjustments based on text content."""
        # Simple amplitude adjustment for emphasis
        emphasis_scale = 1.0
        
        if '!' in text:
            emphasis_scale = 1.2
        elif '?' in text:
            emphasis_scale = 1.1
            
        return audio * emphasis_scale