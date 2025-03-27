import sounddevice as sd
import numpy as np
import soundfile as sf
import asyncio
import time
import os
from faster_whisper import WhisperModel

# Get the absolute path to the models directory
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "whisper", "tiny")

# VAD Parameters
VAD_THRESHOLD = 0.5  # Adjust this threshold based on your environment

# Asynchronously capture and save audio with VAD filtering
async def capture_audio(duration=5, filename="audio.flac"):
    fs = 16000  # Sampling rate
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")

    # Apply VAD filtering
    if vad_filter(audio, threshold=VAD_THRESHOLD):
        # Save the audio to a FLAC file
        sf.write(filename, audio, fs, format='FLAC')
        return filename
    else:
        print("No significant audio detected.")
        return None

# VAD filtering function
def vad_filter(audio, threshold=VAD_THRESHOLD):
    # Simple VAD logic: checks if the max amplitude exceeds the threshold
    return np.max(np.abs(audio)) > threshold

# Function to transcribe audio using local Whisper model
async def transcribe_audio(filename):
    """Transcribe audio using local Whisper tiny model"""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please ensure you have downloaded the model using download_whisper_models.py")
        return None

    print(f"Using local model from: {MODEL_PATH}")
    
    try:
        model = WhisperModel(MODEL_PATH, device="cpu", compute_type="int8")
        segments, info = model.transcribe(
            filename,
            beam_size=5,
            language="en",
            vad_filter=True
        )
        
        transcribed_text = " ".join(segment.text for segment in segments)
        return transcribed_text.strip()
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

# Main function to handle audio capture and processing
async def main():
    # Capture audio
    audio_file = await capture_audio()
    if audio_file:
        start = time.time()
        # Transcribe using local model
        result = await transcribe_audio(audio_file)
        if result:
            print("Transcribed Text:", result)
        end = time.time()
        print("Time taken: ", end-start)
    else:
        print("No valid audio to process.")

# Run the main function in an asyncio event loop
if __name__ == "__main__":
    asyncio.run(main())
