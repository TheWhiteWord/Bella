import os
import pygame
import asyncio
import torch
import torchaudio
from typing import Optional, Tuple, List, Dict
from pathlib import Path
from .generator import Generator, load_csm_1b, Segment
import re

def play_audio(file_path):
    """Play the audio file using pygame."""
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"Error playing audio: {e}")
    finally:
        pygame.mixer.quit()

class VoiceContextManager:
    """Manages the persistent voice context for voice cloning characteristics only."""
    
    def __init__(self, reference_audio: str, reference_text: str, sample_rate: int = 24000):
        """Initialize voice context manager.
        
        Args:
            reference_audio (str): Path to reference audio file (bella_edit.mp3)
            reference_text (str): Transcript of the reference audio
            sample_rate (int): Target sample rate for audio processing
        """
        self.reference_audio = reference_audio
        self.reference_text = reference_text
        self.target_sample_rate = sample_rate
        self._voice_context = None
        self._load_reference()
    
    def _load_reference(self):
        """Load and process reference audio for voice cloning."""
        # Load and convert reference audio
        context_audio, sr = torchaudio.load(self.reference_audio)
        context_audio = context_audio.mean(dim=0)  # Convert to mono
        
        if sr != self.target_sample_rate:
            context_audio = torchaudio.functional.resample(
                context_audio, 
                orig_freq=sr,
                new_freq=self.target_sample_rate
            )
        
        # Normalize audio
        context_audio = context_audio / (torch.max(torch.abs(context_audio)) + 1e-8)
        
        # Create voice context segment (speaker 0 is always the reference voice)
        self._voice_context = [
            Segment(
                text=self.reference_text,
                audio=context_audio,
                speaker=0
            )
        ]
    
    def get_voice_context(self) -> List[Segment]:
        """Get the voice cloning context (reference only)."""
        if self._voice_context is None:
            self._load_reference()
        return self._voice_context
    
    def reset_context(self):
        """Reset to initial reference if needed."""
        if self._voice_context is not None:
            self._load_reference()

class SemanticContextManager:
    """Manages semantic and prosodic context between related sentence segments."""
    
    def __init__(self, max_context_length: int = 3):
        """Initialize semantic context manager.
        
        Args:
            max_context_length (int): Maximum number of previous segments to keep for context
        """
        self.max_context_length = max_context_length
        self.current_segments: List[Tuple[str, torch.Tensor]] = []
        self.punctuation_pattern = re.compile(r'[.!?]+')
    
    def get_context_for_segment(self, text: str) -> List[Segment]:
        """Get relevant context segments for the current text segment.
        
        Args:
            text (str): Current text segment to generate
            
        Returns:
            List[Segment]: List of context segments relevant for generation
        """
        # Always include last segment for continuity
        if not self.current_segments:
            return []
            
        relevant_segments = []
        
        # Check if current text is continuation of a sentence
        is_sentence_start = bool(re.match(r'^[A-Z]', text.lstrip()))
        ends_with_punct = bool(self.punctuation_pattern.search(text))
        
        # If it's mid-sentence, include previous segment
        if not is_sentence_start and self.current_segments:
            relevant_segments.append(
                Segment(
                    text=self.current_segments[-1][0],
                    audio=self.current_segments[-1][1],
                    speaker=0
                )
            )
        
        # If it's end of sentence, include more context
        if ends_with_punct and len(self.current_segments) > 1:
            # Add up to 2 previous segments for better prosodic flow
            for prev_text, prev_audio in reversed(self.current_segments[-3:-1]):
                relevant_segments.append(
                    Segment(text=prev_text, audio=prev_audio, speaker=0)
                )
        
        return relevant_segments

    def update_context(self, text: str, audio: torch.Tensor):
        """Update context with new generated segment.
        
        Args:
            text (str): Generated text segment
            audio (torch.Tensor): Generated audio for the segment
        """
        self.current_segments.append((text, audio))
        
        # Keep only recent segments
        if len(self.current_segments) > self.max_context_length:
            self.current_segments = self.current_segments[-self.max_context_length:]
    
    def reset_context(self):
        """Reset semantic context."""
        self.current_segments.clear()

class CSMVoiceGenerator:
    def __init__(self, reference_audio: str, reference_text: str):
        """Initialize CSM voice generator with reference audio for voice cloning.
        
        Args:
            reference_audio (str): Path to reference audio file
            reference_text (str): Transcript of the reference audio
        """
        self.setup_environment()
        self.load_model()
        self.voice_context = VoiceContextManager(reference_audio, reference_text)
        self.semantic_context = SemanticContextManager()

    def setup_environment(self):
        """Configure environment for optimal performance"""
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    def load_model(self):
        """Load CSM model"""
        model_path = str(Path(__file__).parent.parent / "models" / "ckpt.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"CSM model not found at {model_path}")
            
        self.generator = load_csm_1b(
            ckpt_path=model_path,
            device="cuda",
            max_seq_len=4096
        )

    async def generate_speech(self, text: str, output_path: Optional[str] = None) -> Tuple[torch.Tensor, int]:
        """Generate speech from text using both voice and semantic context.
        
        Args:
            text (str): Text to convert to speech
            output_path (Optional[str]): Path to save the generated audio file
            
        Returns:
            Tuple[torch.Tensor, int]: Generated audio tensor and sample rate
        """
        try:
            # Get voice cloning context
            voice_context = self.voice_context.get_voice_context()
            
            # Get semantic context for current segment
            semantic_context = self.semantic_context.get_context_for_segment(text)
            
            # Combine contexts, prioritizing voice context
            combined_context = voice_context + semantic_context
            
            # Run generation in ThreadPoolExecutor to avoid blocking
            audio = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.generator.generate(
                    text=text,
                    speaker=0,
                    context=combined_context,
                    max_audio_length_ms=10_000,
                    temperature=0.7,
                    topk=25
                )
            )

            # Update semantic context with new generation
            self.semantic_context.update_context(text, audio)

            # Save audio if path provided
            if output_path:
                torchaudio.save(output_path, audio.unsqueeze(0).cpu(), self.generator.sample_rate)

            return audio, self.generator.sample_rate

        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            return None, None
        finally:
            torch.cuda.empty_cache()

    async def convert_text_to_speech(self, text: str, is_new_conversation: bool = False) -> str:
        """Convert text to speech and save to file.
        
        Args:
            text (str): Text to convert to speech
            is_new_conversation (bool): Whether this is start of new conversation
            
        Returns:
            str: Path to the generated audio file
        """
        if is_new_conversation:
            self.semantic_context.reset_context()
            
        output_dir = "Testing/audio files"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "tts_response.wav")
        
        audio, sr = await self.generate_speech(text, output_file)
        return output_file if audio is not None else None

    def reset_all_contexts(self):
        """Reset both voice and semantic contexts."""
        self.voice_context.reset_context()
        self.semantic_context.reset_context()

    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            del self.generator
            torch.cuda.empty_cache()
        except:
            pass

if __name__ == "__main__":
    # Test the functionality of the convert_text_to_speech function
    text = "Hello , how are you doing today?"
    generator = CSMVoiceGenerator(reference_audio="path/to/reference_audio.wav", reference_text="reference text")
    output_file = asyncio.run(generator.convert_text_to_speech(text))
    print(f"Audio file saved at: {output_file}")
    play_audio(output_file)

conversation_history = "assistant: Hello, how are you doing today?\n"
def speech_to_text(audio_file):
    """Convert audio file to text using STT."""
    transcribed_text = "Hello, how are you doing today?"    
    return transcribed_text

def generate_response(text):
    """Generate a response based on the input text."""
    response = "I'm doing well, thank you for asking."
    return response 

import streamlit as st

st.title("AI-Powered Voice Assistant")

# Upload audio
audio_input = st.file_uploader("Upload your audio file")

# Display conversation history
st.text_area("Conversation History", value=conversation_history, height=300)

# Tuning parameters
pitch = st.slider("Pitch", -10, 10, 0)
rate = st.slider("Rate", -50, 50, 0)

# Process the audio
if st.button("Process"):
    text = speech_to_text(audio_input)
    response = generate_response(text)
    generator = CSMVoiceGenerator(reference_audio="path/to/reference_audio.wav", reference_text="reference text")
    audio_output = asyncio.run(generator.convert_text_to_speech(response))
    st.audio(audio_output)
    conversation_history += f"""You: {text}\nAssistant: {response}\n"""