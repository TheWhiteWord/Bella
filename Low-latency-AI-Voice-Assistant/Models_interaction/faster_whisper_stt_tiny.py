"""
Speech-to-Text module using Faster Whisper with optimized audio capture and sentence buffering.
"""
import sys
import os
import time
from typing import Optional, Tuple, List
from pathlib import Path
import asyncio

# Add parent directory to path to import local modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from faster_whisper import WhisperModel
from Models_interaction.buffered_recorder import record_audio
from Models_interaction.llm_response import generate

# Get the absolute path to the models directory
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "whisper", "tiny")

class TranscriptionBuffer:
    def __init__(self, completion_timeout: float = 2.0):
        """Initialize the transcription buffer
        
        Args:
            completion_timeout: Time in seconds to wait after last clip before completing
        """
        self.sentences = []
        self.last_clip_time = None
        self.completion_timeout = completion_timeout
        self.buffer_complete = False
    
    def add_sentence(self, text: str) -> None:
        """Add a new transcribed sentence to the buffer"""
        if text and text.strip():
            if not self.sentences or text.strip() != self.sentences[-1]:
                self.sentences.append(text.strip())
    
    def update_clip_time(self) -> None:
        """Update timestamp of last received audio clip"""
        self.last_clip_time = time.time()
    
    def get_buffered_text(self) -> Optional[str]:
        """Get accumulated transcription if buffer is complete"""
        if self.buffer_complete and self.sentences:
            text = " ".join(self.sentences)
            self.sentences = []
            self.buffer_complete = False
            self.last_clip_time = None
            return text
        return None
    
    def check_completion(self) -> bool:
        """Check if we should complete based on time since last audio clip"""
        if not self.last_clip_time:
            return False
            
        current_time = time.time()
        time_since_last_clip = current_time - self.last_clip_time
        
        # Complete if we haven't received a new clip for completion_timeout duration
        if time_since_last_clip > self.completion_timeout:
            self.buffer_complete = True
            print(f"\nNo new clips received for {time_since_last_clip:.1f}s - completing transcription")
            return True
            
        return False

# Global transcription buffer
_transcription_buffer = TranscriptionBuffer()

# Global state for pausing recording
_should_pause_recording = False

async def transcribe_audio(filename, language="en") -> Tuple[Optional[str], bool]:
    """Transcribe audio using local Whisper tiny model. Returns (text, is_complete)"""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please ensure you have downloaded the model using download_whisper_models.py")
        return None, False

    # Always update clip time when we receive an audio file
    _transcription_buffer.update_clip_time()

    # Initialize Faster-Whisper model with low-memory usage
    model = WhisperModel(MODEL_PATH, device="cpu", compute_type="int8")
    
    # Transcribe the audio clip
    segments, info = model.transcribe(
        filename, 
        language=language,
        beam_size=5,
        condition_on_previous_text=False,
        no_speech_threshold=0.2,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300)
    )
    
    # Add transcribed text to buffer if any
    text = " ".join(segment.text for segment in segments).strip()
    if text:
        _transcription_buffer.add_sentence(text)
    
    # Check if we should complete based on time between clips
    is_complete = _transcription_buffer.check_completion()
    buffered_text = _transcription_buffer.get_buffered_text() if is_complete else None
    
    if is_complete:
        print("\nCompleting transcription - no new clips received")
        if buffered_text:
            print(f"Final transcript: '{buffered_text}'")
        else:
            print("No speech was transcribed in this session")
    
    return buffered_text, is_complete

async def capture_and_transcribe(output_dir=None, device_id=None, debug=False) -> Tuple[Optional[str], bool]:
    """Record audio and transcribe it using Whisper"""
    # Don't record if we're paused (waiting for LLM)
    if _should_pause_recording:
        return None, False
        
    print("\n=== Recording and Transcribing Speech ===")
    
    # Record audio using our optimized recorder
    audio_file = await record_audio(output_dir, device_id, debug)
    if audio_file:
        # Transcribe audio asynchronously using Faster-Whisper
        start = time.time()
        transcribed_text, is_complete = await transcribe_audio(audio_file)
        
        if transcribed_text:
            if debug:
                print(f"\nTranscribed text: '{transcribed_text}'")
                print(f"Processing time: {time.time() - start:.2f}s")
            return transcribed_text, is_complete
        else:
            if debug and not is_complete:
                print("\nSentence buffered, waiting for more...")
    
    return None, False

async def process_voice_input(output_dir=None, device_id=None, debug=False):
    """Main processing loop that handles voice input and LLM responses"""
    conversation_history = []
    
    try:
        while True:
            # Get next audio clip and transcribe
            text, is_complete = await capture_and_transcribe(output_dir, device_id, debug)
            
            if text and is_complete:
                print(f"\nProcessing complete transcript: {text}")
                conversation_history.append(f"User: {text}")
                
                # Pause recording while waiting for LLM
                global _should_pause_recording
                _should_pause_recording = True
                
                try:
                    # Get LLM response
                    system_prompt = """You are a helpful voice assistant. Be concise and natural in your responses.
                    Keep responses under 40 words. Focus on being helpful while maintaining a conversational tone.
                    Use complete sentences but be brief."""
                    
                    llm_response = await generate(
                        prompt=text,
                        system_prompt=system_prompt,
                        verbose=debug
                    )
                    
                    if llm_response:
                        print(f"\nAssistant: {llm_response}")
                        conversation_history.append(f"Assistant: {llm_response}")
                    else:
                        print("\nNo response generated from LLM")
                        
                except Exception as e:
                    print(f"\nError getting LLM response: {e}")
                    
                finally:
                    # Resume recording after LLM processing is complete
                    _should_pause_recording = False
                    print("\nResumed listening for new audio clips...")
                    
            elif text:
                if debug:
                    print("\nAdded clip to buffer, waiting for more clips or timeout...")
                
            await asyncio.sleep(0.1)  # Prevent busy waiting
            
    except KeyboardInterrupt:
        print("\nStopping voice processing...")
    except Exception as e:
        print(f"\nError in processing loop: {e}")
        raise

# Example usage
async def main():
    try:
        await process_voice_input(debug=True)
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    asyncio.run(main())