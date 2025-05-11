"""
Faster Whisper STT module for efficient transcription.
Handles transcription of audio files using the Whisper tiny model.
Whisper transcribes all audio clips that are sent to it and place them in message to be sent to an LLM.
The audio sessio manager is responsible for managing the audio session: once a GAP of 2 sec BETWEEN RECEIVED AUDIO CLIPS (nothing to do with silence detection) is reached, the recorder is paused and the transcription is sent to the LLM.
Since the recorder get paused when the GAP is reached, Whisper will not receive any more audio clips, therefore the message session is defined by the audio clips that are received within the gap timeout.
Once the LLM respons is generated, and the TTS engine has converted the response to speech, and the speech played, the recorder is resumed. 
This way Whisper can continue to receive and transcribe new audio clips without interruption. 
The process repeats.
"""
import os
from faster_whisper import WhisperModel
import asyncio
from functools import lru_cache

# Define model path - use absolute path for local model
MODEL_NAME = "/media/theww/AI/Code/AI/Bella/Bella/models/whisper/small"

# Initialize model once and cache it
@lru_cache(maxsize=1)
def get_whisper_model():
    """Get or initialize the Whisper model with optimized settings. Only loads from local path."""
    return WhisperModel(
        MODEL_NAME,
        device="cuda",  # Use GPU if available
        compute_type="int8",  # Use int8 quantization for efficiency
        cpu_threads=4,  # Adjust based on system
    )

async def transcribe_audio(audio_file: str) -> str:
    """Transcribe audio using Whisper small model.

    Args:
        audio_file (str): Path to the audio file to transcribe
        
    Returns:
        str: Transcribed text, or None if transcription failed
    """
    if not audio_file or not os.path.exists(audio_file):
        return None
    
    try:
        # Get cached model instance
        model = get_whisper_model()
        
        # Run transcription in executor to avoid blocking
        loop = asyncio.get_event_loop()
        segments, _ = await loop.run_in_executor(
            None,
            lambda: model.transcribe(
                audio_file,
                beam_size=5,
                language="en",
                condition_on_previous_text=False,
                no_speech_threshold=0.3,  # More sensitive to speech
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,  # Minimum silence duration between phrases
                    speech_pad_ms=400,  # Add padding around voice segments
                    threshold=0.3  # Voice activity detection threshold
                )
            )
        )
        
        # Combine all segments
        transcribed_text = " ".join(segment.text for segment in segments)
        return transcribed_text.strip()
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None