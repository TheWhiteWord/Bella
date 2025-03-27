"""
Test script to validate BufferedRecorder with Whisper integration.
Shows real-time recording status and transcription results.
"""
import os
import sys
import asyncio
from faster_whisper import WhisperModel
import time
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from Models_interaction.buffered_recorder import record_audio

# Get the absolute path to the Whisper model
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "models",
    "whisper",
    "tiny"
)

async def transcribe_with_whisper(audio_file: str) -> str:
    """Transcribe audio using local Whisper model"""
    if not os.path.exists(MODEL_PATH):
        print(f"\nError: Whisper model not found at {MODEL_PATH}")
        return None
    
    try:
        print("\nInitializing Whisper model...")
        model = WhisperModel(MODEL_PATH, device="cpu", compute_type="int8")
        
        print("Transcribing audio...")
        start_time = time.time()
        
        # Transcribe with optimized settings
        segments, info = model.transcribe(
            audio_file,
            beam_size=5,
            language="en",
            condition_on_previous_text=False,
            no_speech_threshold=0.3,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Combine all segments
        transcribed_text = " ".join(segment.text for segment in segments)
        
        print(f"\nTranscription completed in {time.time() - start_time:.2f} seconds")
        return transcribed_text.strip()
        
    except Exception as e:
        print(f"\nError during transcription: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Test the BufferedRecorder with Whisper integration"""
    print("\n=== Testing BufferedRecorder with Whisper Integration ===")
    print("\nSpeak when ready. Recording will stop automatically after silence.")
    
    try:
        # Start recording with debug output
        audio_file = await record_audio(debug=True)
        
        if audio_file:
            # Transcribe the recorded audio
            transcription = await transcribe_with_whisper(audio_file)
            
            if transcription:
                print("\nTranscription Results:")
                print("-" * 60)
                print(transcription)
                print("-" * 60)
            else:
                print("\nTranscription failed.")
        else:
            print("\nNo audio was recorded.")
            
    except KeyboardInterrupt:
        print("\nTest stopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())