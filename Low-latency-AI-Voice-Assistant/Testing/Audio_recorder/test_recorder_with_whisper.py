import sounddevice as sd
import numpy as np
import soundfile as sf
import os
import asyncio
import sys
from faster_whisper import WhisperModel
import time
from datetime import datetime

# Get the absolute path to the Whisper model
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "models",
    "whisper",
    "tiny"
)

async def record_audio(filename="recording.wav"):
    """Record audio using our working recorder implementation"""
    global recording_active
    recording_active = True
    
    # Audio settings
    sample_rate = 44100
    channels = 1
    dtype = np.float32
    
    # Initialize audio buffer and state variables
    audio_buffer = []
    callback_state = {'is_speaking': False}
    
    def audio_callback(indata, frames, time, status):
        nonlocal audio_buffer
        
        if status:
            print(f'\nStatus: {status}', file=sys.stderr)
        
        # Get the audio data
        frame = indata.copy()
        
        # Store audio
        audio_buffer.append(frame)
        
        # Check voice activity
        frame_energy = np.sqrt(np.mean(frame**2))
        
        # Update speech detection state
        if frame_energy > 0.01:
            if not callback_state['is_speaking']:
                sys.stdout.write('🎤')
                sys.stdout.flush()
            callback_state['is_speaking'] = True
        else:
            if callback_state['is_speaking']:
                sys.stdout.write('.')
                sys.stdout.flush()
            callback_state['is_speaking'] = False

    # Create and start the input stream
    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype=dtype,
        callback=audio_callback,
        blocksize=1024
    )
    
    print("\n🎤 Recording... (Press Ctrl+C to stop when done speaking)")
    print("Speech detection: ", end='', flush=True)
    
    try:
        with stream:
            while recording_active:
                await asyncio.sleep(0.001)
                
        print("\n\nProcessing recorded audio...")
        
        if not audio_buffer:
            print("No audio recorded.")
            return None
        
        # Combine all audio chunks
        audio = np.concatenate(audio_buffer)
        
        # Normalize audio
        if len(audio) > 0:
            audio = audio - np.mean(audio)
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.9
        
        # Save the audio file
        output_dir = os.path.join(os.path.dirname(__file__), "recorded_audio")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"test_recording_{timestamp}.wav")
        
        duration = len(audio) / sample_rate
        if duration < 0.1:
            print("Recording too short.")
            return None
            
        sf.write(filename, audio, sample_rate, format='WAV')
        print(f"Recorded {duration:.1f}s of audio.")
        print(f"Audio saved to: {filename}")
        return filename
        
    except Exception as e:
        print(f"\n❌ Error during recording: {e}")
        return None

async def transcribe_audio(audio_file):
    """Transcribe audio using Whisper tiny model"""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Whisper model not found at {MODEL_PATH}")
        return None
    
    try:
        print("\nInitializing Whisper model...")
        model = WhisperModel(MODEL_PATH, device="cpu", compute_type="int8")
        
        print("Transcribing audio...")
        start_time = time.time()
        
        # Transcribe with Whisper
        segments, info = model.transcribe(
            audio_file,
            language="en",
            vad_filter=True  # Enable voice activity detection
        )
        
        # Combine all segments
        transcribed_text = " ".join(segment.text for segment in segments)
        
        print(f"\nTranscription completed in {time.time() - start_time:.2f} seconds")
        return transcribed_text.strip()
        
    except Exception as e:
        print(f"\n❌ Error during transcription: {e}")
        return None

def signal_handler(signum, frame):
    """Handle Ctrl+C to stop recording gracefully"""
    global recording_active
    print("\n\nStopping recording...")
    recording_active = False

async def main():
    """Run the recording and transcription test"""
    print("\n=== Testing Audio Recording with Whisper Transcription ===")
    
    # Set up signal handler for Ctrl+C
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    
    # Record audio
    print("\nSpeak into the microphone. Press Ctrl+C when done speaking.")
    audio_file = await record_audio()
    
    if audio_file:
        # Transcribe the recorded audio
        transcription = await transcribe_audio(audio_file)
        
        if transcription:
            print("\nTranscription Results:")
            print("-" * 40)
            print(transcription)
            print("-" * 40)
        else:
            print("\nTranscription failed.")
    else:
        print("\nNo audio was recorded.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest stopped by user.")
    except Exception as e:
        print(f"\nError: {e}")