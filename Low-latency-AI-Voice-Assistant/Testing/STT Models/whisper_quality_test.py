import sounddevice as sd
import numpy as np
import soundfile as sf
import asyncio
from faster_whisper import WhisperModel
import matplotlib.pyplot as plt
import os
import time
import sys

# Constants for audio
SAMPLE_RATE = 16000
CHANNELS = 1

def plot_waveform(audio_data, sample_rate, title="Audio Waveform", save_path=None):
    """Plot and save audio waveform for analysis"""
    duration = len(audio_data) / sample_rate
    time_axis = np.linspace(0, duration, len(audio_data))
    
    plt.figure(figsize=(15, 5))
    plt.plot(time_axis, audio_data)
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"Waveform saved to: {save_path}")

async def record_audio(max_duration=60, filename="test_audio.wav", debug_dir="audio_debug"):
    """Record audio with natural speech pattern detection"""
    debug_dir = os.path.join(os.path.dirname(filename), debug_dir)
    os.makedirs(debug_dir, exist_ok=True)
    
    print("\nWaiting for speech...")
    
    # Parameters
    fs = SAMPLE_RATE
    audio_frames = []
    is_recording = False
    silence_frames = 0
    frame_duration = 0.05  # 50ms frames
    max_silence_frames = int(3.14 / frame_duration)  # π seconds of silence to stop (natural pause)
    silence_threshold = 0.015
    speech_started = False
    speech_frames = 0
    min_speech_frames = int(0.2 / frame_duration)  # Minimum 0.2s of speech
    
    def audio_callback(indata, frames, time, status):
        nonlocal is_recording, silence_frames, speech_started, speech_frames
        
        if status:
            print(f"Status: {status}")
            
        # Calculate frame energy
        frame_energy = np.sqrt(np.mean(indata**2))
        
        # Voice activity detection with hysteresis
        if frame_energy > silence_threshold:
            if not speech_started:
                print("\nSpeech detected! Recording...")
                speech_started = True
            silence_frames = 0
            speech_frames += 1
            is_recording = True
            sys.stdout.write('🎤')
            sys.stdout.flush()
        elif is_recording and frame_energy > silence_threshold * 0.5:
            silence_frames = 0
            speech_frames += 1
            sys.stdout.write('🎤')
            sys.stdout.flush()
        elif is_recording:
            silence_frames += 1
            # Show silence counter every second
            if silence_frames % int(1.0 / frame_duration) == 0:
                sys.stdout.write(f'({silence_frames * frame_duration:.1f}s)')
            else:
                sys.stdout.write('.')
            sys.stdout.flush()
        
        # Store frame if recording
        if is_recording:
            audio_frames.append(indata.copy())
    
    # Setup audio stream
    stream = sd.InputStream(
        samplerate=fs,
        channels=CHANNELS,
        dtype=np.float32,
        callback=audio_callback,
        blocksize=int(fs * frame_duration)
    )
    
    print("Speak now...")
    
    start_time = time.time()
    with stream:
        while True:
            await asyncio.sleep(frame_duration)
            
            # Stop conditions
            if is_recording:
                current_duration = time.time() - start_time
                
                # Only stop on silence if we have minimum speech duration
                if speech_frames >= min_speech_frames:
                    if silence_frames >= max_silence_frames:
                        print("\nLong silence detected (π seconds), stopping recording.")
                        break
                
                # Stop if max duration reached
                if current_duration >= max_duration:
                    print("\nMax duration reached.")
                    break
            
            # Timeout if no speech detected initially
            if not speech_started and time.time() - start_time > 10:  # Increased initial wait time
                print("\nNo speech detected, stopping.")
                return None, None
    
    if not audio_frames:
        print("No audio captured!")
        return None, None
    
    # Process recorded audio
    audio_data = np.concatenate(audio_frames, axis=0).flatten()
    
    # Print audio statistics
    print("\nAudio Statistics:")
    print(f"Duration: {len(audio_data)/fs:.2f} seconds")
    print(f"Speech duration: {speech_frames * frame_duration:.2f} seconds")
    print(f"Final silence duration: {silence_frames * frame_duration:.2f} seconds")
    print(f"Max amplitude: {np.max(np.abs(audio_data)):.3f}")
    print(f"RMS level: {np.sqrt(np.mean(audio_data**2)):.3f}")
    
    # Save raw audio
    raw_audio_path = os.path.join(debug_dir, "raw_input.wav")
    sf.write(raw_audio_path, audio_data, fs)
    print(f"\nRaw audio saved to: {raw_audio_path}")
    
    # Generate waveform plot
    plot_waveform(
        audio_data, 
        fs,
        "Raw Audio Input",
        os.path.join(debug_dir, "waveform.png")
    )
    
    return raw_audio_path, audio_data

async def transcribe_with_whisper(audio_file, model_size="tiny"):
    """Transcribe audio using local Whisper model"""
    if not audio_file:
        return None
    
    # Define local model path
    models_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "models",
        "whisper",
        model_size
    )
    
    print(f"\nTranscribing with Whisper...")
    print(f"Model path: {models_dir}")
    start_time = time.time()
    
    try:
        # Initialize model
        model = WhisperModel(models_dir, device="cpu", compute_type="int8")
        
        # Transcribe
        segments, info = model.transcribe(
            audio_file,
            beam_size=5,
            language="en",
            vad_filter=True
        )
        
        # Get results
        transcribed_text = " ".join(segment.text for segment in segments)
        processing_time = time.time() - start_time
        
        print(f"\nTranscription Results:")
        print(f"Text: '{transcribed_text}'")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Language detected: {info.language} (confidence: {info.language_probability:.2f})")
        
        return transcribed_text
        
    except Exception as e:
        print(f"\nError during transcription: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return None

async def run_whisper_test():
    """Run a simple Whisper test"""
    print("=== Simple Whisper Test ===")
    
    # Record audio with longer max duration to allow for natural pauses
    audio_file, audio_data = await record_audio(
        max_duration=300,  # 5 minutes max to prevent infinite recording
        filename="Testing/audio files/whisper_test.wav"
    )
    
    if audio_file:
        # Transcribe
        await transcribe_with_whisper(audio_file)
    
    print("\nTest complete!")

if __name__ == "__main__":
    asyncio.run(run_whisper_test())