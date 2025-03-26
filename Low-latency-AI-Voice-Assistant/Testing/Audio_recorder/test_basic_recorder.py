import sounddevice as sd
import numpy as np
import soundfile as sf
import os
import asyncio
import sys
import time
from datetime import datetime

# Global flag for recording state
recording_active = True

def signal_handler(signum, frame):
    """Handle Ctrl+C to stop recording gracefully"""
    global recording_active
    recording_active = False

async def record_audio(filename="test_recording.wav"):
    """Record audio until stopped with voice activity detection"""
    global recording_active
    recording_active = True
    
    # Audio settings
    sample_rate = 44100
    channels = 1
    dtype = np.float32
    
    # Initialize audio buffer and state variables
    audio_buffer = []
    callback_state = {'is_speaking': False}  # Use dict to modify state in callback
    
    def audio_callback(indata, frames, time, status):
        nonlocal audio_buffer  # Ensure we can modify the buffer
        
        if status:
            print(f'\nStatus: {status}', file=sys.stderr)
        
        # Get the audio data
        frame = indata.copy()
        
        # Always store audio to prevent missing the start of speech
        audio_buffer.append(frame)
        
        # Check voice activity
        frame_energy = np.sqrt(np.mean(frame**2))
        
        # Update speech detection state
        if frame_energy > 0.01:  # Lowered threshold for better sensitivity
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
        blocksize=1024  # Reduced block size for better responsiveness
    )
    
    print("\n🎤 Recording... (Press Ctrl+C to stop)")
    print("Speech detection: ", end='', flush=True)
    
    try:
        with stream:
            while recording_active:
                await asyncio.sleep(0.001)  # Tiny sleep to prevent CPU overload
                
        print("\n\nProcessing recorded audio...")
        
        if not audio_buffer:
            print("No audio recorded.")
            return None
        
        # Combine all audio chunks
        audio = np.concatenate(audio_buffer)
        
        # Normalize audio
        if len(audio) > 0:
            audio = audio - np.mean(audio)  # Remove DC offset
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.9
        
        # Save the audio file
        output_dir = os.path.join(os.path.dirname(__file__), "recorded_audio")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"test_recording_{timestamp}.wav")
        
        duration = len(audio) / sample_rate
        if duration < 0.1:  # Don't save very short recordings
            print("Recording too short.")
            return None
            
        sf.write(filename, audio, sample_rate, format='WAV')
        print(f"Recorded {duration:.1f}s of audio.")
        print(f"Audio saved to: {filename}")
        return filename
        
    except Exception as e:
        print(f"\n❌ Error during recording: {e}")
        return None

def list_audio_devices():
    """List all available audio input devices."""
    print("\nAvailable audio devices:")
    print("-" * 60)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"Device {i}: {device['name']}")
        print(f"  Channels (in/out): {device['max_input_channels']}/{device['max_output_channels']}")
        print(f"  Sample rates: {device['default_samplerate']}")
        print("-" * 60)

async def main():
    print("\n=== Testing Audio Recording ===")
    print("Recording will continue until you press Ctrl+C")
    
    # List available devices
    list_audio_devices()
    
    # Set up signal handler for Ctrl+C
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    
    # Record audio
    audio_file = await record_audio()
    
    if audio_file:
        print("\nRecording completed successfully!")
    else:
        print("\nRecording failed or no audio captured.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    except Exception as e:
        print(f"\nError: {e}")