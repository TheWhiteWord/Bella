import sounddevice as sd
import numpy as np
import soundfile as sf
import os
from faster_whisper import WhisperModel
import asyncio
import time
import sys

# Get the absolute path to the models directory
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "whisper", "tiny")

async def capture_audio(filename="stt_transcribe.flac", max_silence=0.8, min_speech=0.3):
    """Capture audio using stream-based approach with visual feedback"""
    fs = 16000  # Sampling rate
    print("\nListening... (Speak now)")
    print("Speech detection: ", end='', flush=True)
    
    # Parameters for voice detection
    audio_frames = []
    silent_frames = 0
    speech_frames = 0
    is_recording = False
    samples_per_frame = int(fs * 0.05)  # 50ms frames for faster response
    
    def audio_callback(indata, frames, time, status):
        nonlocal silent_frames, speech_frames, is_recording
        
        if status:
            print(f"\nStatus: {status}")
            
        # Convert to float32 and get frame energy
        frame = indata.astype(np.float32)
        frame_energy = np.sqrt(np.mean(frame**2))
        
        # Voice activity detection with lower threshold
        is_speech = frame_energy > 0.005  # More sensitive threshold
        
        if is_speech:
            if not is_recording:
                sys.stdout.write('🎤')  # Show microphone when speech starts
                sys.stdout.flush()
            silent_frames = 0
            speech_frames += 1
            is_recording = True
        elif is_recording:
            silent_frames += 1
            sys.stdout.write('.')  # Show dots during silence
            sys.stdout.flush()
        
        # Store frame if we're recording
        if is_recording:
            audio_frames.append(frame.copy())

    # Setup and start recording
    stream = sd.InputStream(
        samplerate=fs,
        channels=1,
        dtype=np.float32,
        callback=audio_callback,
        blocksize=samples_per_frame
    )
    
    with stream:
        while True:
            await asyncio.sleep(0.05)  # Check every 50ms for faster response
            
            # Convert frame counts to seconds
            silence_duration = silent_frames * 0.05
            speech_duration = speech_frames * 0.05
            
            # Stop conditions
            if is_recording:
                if silence_duration >= max_silence and speech_duration >= min_speech:
                    print("\n\nSpeech detected! Processing...")
                    break
                elif silence_duration >= max_silence:
                    print("\n\nNo clear speech detected, please try again.")
                    return None
    
    if not audio_frames or speech_duration < min_speech:
        print("No valid speech detected.")
        return None
        
    # Process recorded audio
    audio = np.concatenate(audio_frames, axis=0)
    
    # Normalize audio to prevent clipping
    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = audio / max_amp * 0.9
    
    # Save the audio
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Testing", "audio files")
    os.makedirs(output_dir, exist_ok=True)
    full_filename = os.path.join(output_dir, filename)
    
    sf.write(full_filename, audio, fs, format='FLAC')
    print(f"Recorded {speech_duration:.1f}s of speech.")
    return full_filename

# VAD filtering function
def vad_filter(audio, threshold=0.01):
    # Simple VAD logic: checks if the max amplitude exceeds the threshold
    return np.max(np.abs(audio)) > threshold

# Function to transcribe audio using Faster-Whisper
async def transcribe_audio(filename, language="en"):
    """Transcribe audio using local Whisper tiny model"""
    # Verify model path exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please ensure you have downloaded the model using download_whisper_models.py")
        return ""

    print(f"Loading model from: {MODEL_PATH}")
    
    # Initialize Faster-Whisper model with low-memory usage
    model = WhisperModel(MODEL_PATH, device="cpu", compute_type="int8")

    segments, info = model.transcribe(filename, language=language)
    transcribed_text = ""
    for segment in segments:
        transcribed_text += segment.text + " "

    return transcribed_text.strip()

# Main function to handle audio capture and processing
async def main():
    print("\n=== Testing Whisper STT ===")
    print("Waiting for speech...")
    
    # Capture audio with VAD
    audio_file = await capture_audio()

    if audio_file:
        # Transcribe audio asynchronously using Faster-Whisper
        start = time.time()
        transcribed_text = await transcribe_audio(audio_file)
        
        if transcribed_text:
            print(f"\nTranscribed text: '{transcribed_text}'")
        else:
            print("\nNo transcription produced")
            
        print(f"Time taken: {time.time() - start:.2f}s")
    else:
        print("\nNo audio file was created")

# Run the main function in an asyncio event loop
if __name__ == "__main__":
    asyncio.run(main())