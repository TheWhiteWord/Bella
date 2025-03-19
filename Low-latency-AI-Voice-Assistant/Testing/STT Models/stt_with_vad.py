import sounddevice as sd
import numpy as np
import soundfile as sf
import aiohttp
import asyncio
import webrtcvad
import time
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
from faster_whisper import WhisperModel

load_dotenv()

def plot_audio_data(audio_data, sample_rate, title="Audio Waveform", show_plot=True):
    """Plot audio waveform for debugging"""
    duration = len(audio_data) / sample_rate
    time_axis = np.linspace(0, duration, len(audio_data))
    
    plt.figure(figsize=(15, 5))
    plt.plot(time_axis, audio_data)
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    if show_plot:
        plt.show()
    plt.close()

async def capture_audio_vad(duration=5, filename="audio.flac"):
    """Capture audio with VAD processing"""
    fs = 16000  # Sampling rate
    vad = webrtcvad.Vad(2)  # VAD with aggressive mode (0-3)
    buffer_duration = 0.03  # 30 ms for better VAD performance
    num_frames = int(fs * buffer_duration)
    
    # Create arrays to store diagnostics
    raw_audio_data = []
    vad_decisions = []
    audio_data = []
    frame_times = []
    processing_times = []
    
    print(f"\nInitializing recording...")
    print(f"Sample rate: {fs} Hz")
    print(f"Frame size: {num_frames} samples ({buffer_duration * 1000}ms)")
    print(f"Recording duration: {duration}s")
    
    def audio_callback(indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        
        start_process = time.currentTime
        
        # Normalize int16 input to float32 (-1 to 1 range)
        float_data = indata.astype(np.float32) / 32768.0
        
        # Apply soft clipping to prevent harsh peaks
        float_data = np.tanh(float_data)
        
        raw_audio_data.append(float_data.copy())
        frame_times.append(start_process)
        
        # Convert normalized audio back to int16 for VAD
        int16_data = (float_data * 32767).astype(np.int16)
        try:
            if len(int16_data) == num_frames:
                is_speech = vad.is_speech(int16_data.tobytes(), fs)
                vad_decisions.append(is_speech)
                if is_speech:
                    audio_data.append(float_data.copy())
            else:
                print(f"Wrong chunk size: {len(int16_data)} (expected {num_frames})")
        except Exception as e:
            print(f"VAD error: {str(e)}")
        
        end_process = time.currentTime
        processing_times.append(end_process - start_process)

    # Setup and start recording
    print("\nStarting recording...")
    stream = sd.InputStream(
        samplerate=fs,
        channels=1,
        dtype=np.int16,
        blocksize=num_frames,
        callback=audio_callback
    )
    
    with stream:
        sd.sleep(int(duration * 1000))
    
    print("Recording complete.")
    
    # Print diagnostic information
    print("\nRecording Statistics:")
    print(f"Total frames captured: {len(raw_audio_data)}")
    print(f"Frames with speech: {sum(vad_decisions)}")
    if processing_times:
        avg_process_time = np.mean([t for t in processing_times if t is not None])
        print(f"Average processing time per frame: {avg_process_time*1000:.2f}ms")
    
    # Convert and analyze raw audio
    if raw_audio_data:
        raw_audio = np.concatenate(raw_audio_data, axis=0)
        print(f"\nRaw Audio Stats:")
        print(f"Shape: {raw_audio.shape}")
        print(f"Max amplitude: {np.max(np.abs(raw_audio)):.3f}")
        print(f"RMS level: {np.sqrt(np.mean(raw_audio**2)):.3f}")
        
        # Plot raw audio
        plot_audio_data(raw_audio.flatten(), fs, "Raw Audio Input")
    
    # Process and save speech segments
    if audio_data:
        print("\nProcessing speech segments...")
        audio = np.concatenate(audio_data, axis=0)
        
        # Normalize the final audio to prevent clipping
        max_amp = np.max(np.abs(audio))
        if max_amp > 0:
            audio = audio / max_amp * 0.9  # Leave some headroom
        
        print(f"Speech segments stats:")
        print(f"Total duration: {len(audio)/fs:.2f}s")
        print(f"Shape: {audio.shape}")
        print(f"Final max amplitude: {np.max(np.abs(audio)):.3f}")
        
        # Plot processed audio
        plot_audio_data(audio.flatten(), fs, "Processed Audio (Speech Only)")
        
        # Save the audio
        output_dir = os.path.join("Testing", "audio files")
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, filename)
        
        sf.write(full_path, audio, fs, format='FLAC')
        print(f"\nAudio saved to: {full_path}")
        return full_path
    else:
        print("\nNo speech segments detected!")
        return None

async def transcribe_audio(audio_file: str, model_size: str = "tiny") -> str:
    """Transcribe audio using local Whisper model"""
    if not audio_file:
        return None
    
    # Define local model path
    models_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "models",
        "whisper"
    )
    model_path = os.path.join(models_dir, model_size)
        
    print(f"\nTranscribing with local Whisper model: {model_path}")
    start_time = time.time()
    
    try:
        # Check if we have the model locally
        if os.path.exists(model_path):
            print(f"Using local model from: {model_path}")
            model = WhisperModel(model_path, device="cpu", compute_type="int8")
        else:
            print(f"Model {model_size} not found at {model_path}. Using default path.")
            model = WhisperModel(model_size, device="cpu", compute_type="int8")

        segments, info = model.transcribe(
            audio_file,
            beam_size=5,
            language="en",
            condition_on_previous_text=False,
            no_speech_threshold=0.3,  # More sensitive to speech
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Collect all segments
        transcribed_text = " ".join(segment.text for segment in segments)
        
        print(f"\nTranscription complete in {time.time() - start_time:.2f}s")
        print(f"Text: {transcribed_text}")
        
        return transcribed_text.strip()
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None

# Main function to handle audio capture and processing
async def main():
    # Capture audio with VAD
    audio_file = await capture_audio_vad()
    
    # Transcribe using local Whisper model
    if audio_file:
        start = time.time()
        text = await transcribe_audio(audio_file)
        end = time.time()
        print(f"\nTotal processing time: {end - start:.2f}s")
        return text

# Run the main function in an asyncio event loop
if __name__ == "__main__":
    asyncio.run(main())
