import os
import sys
import ctypes

# Surgical fix for CUDA/cuDNN library conflicts (fixes Whisper session crashes)
def fix_cuda_paths():
    import site
    packages = site.getsitepackages()
    if packages:
        site_packages = packages[0]
        # Priority paths for pip-installed NVIDIA libraries
        added_paths = []
        for lib in ['cudnn', 'cublas', 'cuda_runtime']:
            lib_path = os.path.join(site_packages, "nvidia", lib, "lib")
            if os.path.isdir(lib_path):
                added_paths.append(lib_path)
        
        # Preload critical libraries to ensure process uses correct versions
        # this bypasses conflicting libraries often found in conda's env/lib
        for path in added_paths:
            if not os.path.isdir(path):
                continue
            for f in os.listdir(path):
                # Preload cuDNN components specifically needed by ctranslate2/Whisper
                if f.startswith("libcudnn_cnn.so.9") or f.startswith("libcudnn.so.9") or \
                   f.startswith("libcublas.so.12") or f.startswith("libcublasLt.so.12"):
                    lib_full_path = os.path.join(path, f)
                    try:
                        ctypes.CDLL(lib_full_path, mode=ctypes.RTLD_GLOBAL)
                    except Exception:
                        pass # Silently continue if some can't be loaded

fix_cuda_paths()

"""Main application module for voice assistant with Chatterbox-Turbo TTS integration.

This module coordinates audio recording, speech recognition, LLM interaction,
and text-to-speech using Chatterbox. Uses PipeWire/PulseAudio for audio I/O.
"""
import subprocess
import asyncio
import argparse
import json
import tempfile
import logging
from typing import Dict, Any, Optional, Tuple, List
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.utility.audio_session_manager import AudioSessionManager
from src.utility.buffered_recorder import BufferedRecorder, create_audio_stream
from src.llm.chat_manager import generate_chat_response
from src.audio.chatterbox_tts.chatterbox_tts import ChatterboxTTSWrapper
from src.llm.config_manager import ModelConfig
from src.audio.whisper.faster_whisper_stt_tiny import get_whisper_model

# Path to the search signal file used by web_search_mcp
SEARCH_SIGNAL_PATH = os.path.join(tempfile.gettempdir(), "bella_search_status.json")

def list_audio_devices() -> None:
    """List all available PulseAudio output sinks."""
    print("\nAvailable audio devices (PulseAudio sinks):")
    try:
        result = subprocess.run(['pactl', 'list', 'sinks'], 
                              capture_output=True, text=True, check=True)
        print("\nFull audio device list:")
        for line in result.stdout.split('\n'):
            if any(key in line for key in ['Name:', 'Description:', 'State:']):
                print(line.strip())
            elif line.startswith('Sink #'):
                print(f"\n{line.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"Error listing audio devices: {e}")
        sys.exit(1)

async def check_search_signal() -> Optional[Dict[str, Any]]:
    """Check if a search operation is in progress by reading the signal file.
    
    Returns:
        Optional[Dict[str, Any]]: Search signal data if a search is in progress
    """
    if os.path.exists(SEARCH_SIGNAL_PATH):
        try:
            with open(SEARCH_SIGNAL_PATH, "r") as f:
                signal_data = json.load(f)
                
            # Only return if status is "searching" (active search)
            if signal_data.get("status") == "searching":
                return signal_data
        except:
            pass
    return None

async def wait_for_search_completion(timeout: float = 30.0) -> Optional[Dict[str, Any]]:
    """Wait for a search operation to complete.
    
    Args:
        timeout: Maximum time to wait for completion in seconds
        
    Returns:
        Optional[Dict[str, Any]]: Final search status or None if timed out
    """
    start_time = asyncio.get_event_loop().time()
    
    while (asyncio.get_event_loop().time() - start_time) < timeout:
        if os.path.exists(SEARCH_SIGNAL_PATH):
            try:
                with open(SEARCH_SIGNAL_PATH, "r") as f:
                    signal_data = json.load(f)
                    
                # If status is "completed" or "failed", search is done
                if signal_data.get("status") in ["completed", "failed"]:
                    return signal_data
            except:
                pass
                
        await asyncio.sleep(0.5)  # Check twice per second
        
    return None  # Timed out

async def init_tts_engine(sink_name: Optional[str] = None) -> ChatterboxTTSWrapper:
    """Initialize the Chatterbox-Turbo TTS engine.
    
    Args:
        sink_name (str, optional): Name of PulseAudio sink to use
        
    Returns:
        ChatterboxTTSWrapper: Initialized TTS engine
        
    Raises:
        Exception: If TTS engine initialization fails
    """
    print("\nInitializing Chatterbox-Turbo TTS engine...")
    try:
        engine = ChatterboxTTSWrapper(
            sink_name=sink_name
        )
        # Test TTS engine with a short message
        await engine.generate_speech("Chatterbox TTS system initialized.")
        return engine
    except Exception as e:
        print(f"Error initializing TTS engine: {e}")
        raise

async def main_interaction_loop(model: str = None, sink_name: Optional[str] = None) -> None:
    """Main loop for capturing speech, generating responses, and playing audio.
    
    Args:
        model (str, optional): Model nickname for Ollama. If None, uses default from config
        sink_name (str, optional): Name of PulseAudio sink to use for output
    """
    print("\nInitializing voice assistant components...")
    tts_engine = None
    recorder = None
    self_awareness_summary = None
    
    try:
        # Get model from config if not specified
        if model is None:
            model_config = ModelConfig()
            model = model_config.get_default_model()
            
        # Initialize TTS
        try:
            # Print GPU diagnostic
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"\nCUDA detected: {gpu_name}")
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"Available VRAM: {vram_gb:.2f} GB")
            else:
                print("\nWARNING: CUDA not detected by PyTorch!")

            tts_engine = await init_tts_engine(sink_name)
        except Exception as e:
            print(f"\nError initializing TTS engine: {e}")
            print("\nFalling back to CPU mode as a last resort...")
            try:
                # Try again with explicit CPU device
                tts_engine = ChatterboxTTSWrapper(
                    sink_name=sink_name,
                    device="cpu"
                )
                await tts_engine.generate_speech("TTS system initialized in CPU mode.")
            except Exception as second_e:
                print(f"\nFallback TTS initialization also failed: {second_e}")
                raise
        
        # Warm up Whisper STT
        try:
            print("\nWarming up Whisper STT model...")
            # Run initialization in a thread to avoid blocking if it takes time
            # though get_whisper_model is fast if cached, the first load takes time
            # We just need to trigger the lru_cache
            await asyncio.to_thread(get_whisper_model)
            print("Whisper STT model warmed up.")
        except Exception as e:
            print(f"\nWarning: Whisper warm-up failed (will try again during first transcription): {e}")

        print(f"\nUsing model: {model}")
        


        welcome_message = "Voice Assistant ready! Start speaking when ready."
        
        await tts_engine.generate_speech(welcome_message)
        print("\nSay 'stop' or 'exit' to end the conversation.\n")
        
        # Create audio session manager with debug mode
        audio_manager = AudioSessionManager(gap_timeout=2.0, debug=True)
        recorder = BufferedRecorder()
        
        # Connect recorder to audio manager
        audio_manager.set_recorder(recorder)
        
        # Initialize audio settings
        recorder.initialize_audio_settings()
        
        # Conversation history
        conversation_history = []
        
        while True:
            try:
                # Ensure recording is fully stopped before starting new interaction
                if recorder.is_recording:
                    recorder.should_stop = True
                    recorder.is_recording = False
                    await audio_manager.pause_session()
                    await asyncio.sleep(0.2)  # Give time for everything to stop
                
                # Reset states for new interaction
                recorder.reset_state()
                await audio_manager.resume_session()
                
                # Start recording
                print("\nWaiting for voice...")
                recorder.should_stop = False
                recorder.start_recording()
                
                # Wait for complete utterance
                transcribed_text, segments = await audio_manager.start_session()
                
                if not transcribed_text:
                    print("\nNo speech detected, continuing...")
                    continue

                print(f"\nYou said: {transcribed_text}")

                if any(word in transcribed_text.lower() for word in ['exit']):
                    await tts_engine.generate_speech("Goodbye!")
                    break

                # Fully stop recording while processing response
                print("\nRecording paused...")
                recorder.should_stop = True
                recorder.is_recording = False
                await audio_manager.pause_session()
                await asyncio.sleep(0.3)  # Give more time to ensure stop

                # Format conversation history for context
                formatted_history = []
                for i, entry in enumerate(conversation_history):
                    role = "user" if i % 2 == 0 else "assistant"
                    formatted_history.append({"role": role, "content": entry})

                # Generate response using local Ollama model
                print(f"\nThinking... (using {model})")
                
                # Simple generation without tools or memory
                response = await generate_chat_response(
                    user_input=transcribed_text,
                    history_context="\n".join(conversation_history[-10:]), # Pass recent history string
                    model=model,
                    timeout=120.0
                )
                
                print(f"Assistant: {response}")
                
                # Update conversation history
                conversation_history.append(transcribed_text)  # User input
                conversation_history.append(response)  # Assistant response

                # Convert response to speech and play it
                print("\nGenerating speech...")
                try:
                    await tts_engine.generate_speech(response)
                    # Add a delay after speech to ensure TTS fully completes before resuming recording
                    await asyncio.sleep(0.5)
                except Exception as e:
                    print(f"\nError during speech generation: {e}")
                    continue

                # Only resume recording after response has fully played
                await asyncio.sleep(0.5)  # Reduced delay since Chatterbox handles timing
                
                print("\nRecording resumed...")
                recorder.should_stop = False
                recorder.start_recording()
                await audio_manager.resume_session()
                print("\n" + "="*50)
                
            except Exception as e:
                print(f"\nError in interaction loop: {e}")
                await asyncio.sleep(1)
                continue
            
    except Exception as e:
        print(f"\nFatal error in main loop: {e}")
        
    finally:
        # Ensure everything is properly cleaned up
        print("\nCleaning up...")
        if recorder:
            recorder.should_stop = True
            recorder.is_recording = False
            recorder.stop_recording()
        if tts_engine:
            tts_engine.stop()
        print("\nGoodbye!")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Assistant with Local LLM and Chatterbox TTS")
    parser.add_argument(
        "--model",
        default=None,
        help="Model to use for responses. If not specified, uses default from config"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit"
    )
    parser.add_argument(
        "--sink",
        type=str,
        help="Name of PulseAudio sink to use"
    )
    parser.add_argument(
        "--no-visualizer",
        action="store_true",
        help="Do not launch the Bella Voice Visualizer"
    )

    args = parser.parse_args()

    # Launch the visualizer as a separate process unless disabled
    visualizer_proc = None
    if not args.no_visualizer:
        print("Launching visualizer...")
        visualizer_path = os.path.join(os.path.dirname(__file__), "src", "ui", "bella_visualizer.py")
        # Redirect output to a log file for debugging
        vis_log = open("visualizer.log", "w")
        visualizer_proc = subprocess.Popen(
            [sys.executable, visualizer_path],
            stdout=vis_log,
            stderr=subprocess.STDOUT
        )
        print(f"Visualizer launched with PID {visualizer_proc.pid}")

    try:
        if args.list_devices:
            list_audio_devices()
            sys.exit(0)
        
        asyncio.run(main_interaction_loop(args.model, args.sink))
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)
    finally:
        # Ensure the visualizer process is terminated when the app exits
        if visualizer_proc is not None:
            try:
                visualizer_proc.terminate()
                # Give it a moment to cleanup gracefully
                for _ in range(10):
                    if visualizer_proc.poll() is not None:
                        break
                    import time
                    time.sleep(0.1)
                if visualizer_proc.poll() is None:
                    visualizer_proc.kill()
            except Exception:
                pass
