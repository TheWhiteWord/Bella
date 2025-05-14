"""Main application module for voice assistant with Kokoro TTS integration.

This module coordinates audio recording, speech recognition, LLM interaction,
and text-to-speech using Kokoro. Uses PipeWire/PulseAudio for audio I/O.
"""
import os
import sys
import subprocess
import asyncio
import argparse
import subprocess
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
from src.llm.chat_manager import generate_chat_response, generate_chat_response_with_tools
from src.audio.kokoro_tts.kokoro_tts import KokoroTTSWrapper
from src.llm.config_manager import ModelConfig


# Use new BellaMemory conversation adapter
from src.bella_memory.memory_conversation_adapter import MemoryConversationAdapter

# Import self/user awareness tools to ensure registration at startup
from src.bella_memory.tools_self_user_awareness import set_bella_memory_instance, summarize_self_awareness

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

async def init_tts_engine(sink_name: Optional[str] = None) -> KokoroTTSWrapper:
    """Initialize the Kokoro TTS engine.
    
    Args:
        sink_name (str, optional): Name of PulseAudio sink to use
        
    Returns:
        KokoroTTSWrapper: Initialized TTS engine
        
    Raises:
        Exception: If TTS engine initialization fails
    """
    print("\nInitializing Kokoro TTS engine...")
    try:
        engine = KokoroTTSWrapper(
            default_voice="af_bella",
            speed=0.9,  # Slightly slower for better clarity
            sink_name=sink_name
        )
        # Test TTS engine with a short message
        await engine.generate_speech("TTS system initialized.")
        return engine
    except Exception as e:
        print(f"Error initializing TTS engine: {e}")
        raise




async def main_interaction_loop(model: str = None, sink_name: Optional[str] = None, use_memory: bool = True) -> None:
    """Main loop for capturing speech, generating responses, and playing audio.
    
    Args:
        model (str, optional): Model nickname for Ollama. If None, uses default from config
        sink_name (str, optional): Name of PulseAudio sink to use for output
        use_memory (bool): Whether to use the memory system
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
            
        # Initialize TTS with fallback to CPU if CUDA fails
        try:
            tts_engine = await init_tts_engine(sink_name)
        except Exception as e:
            print(f"\nError initializing TTS engine: {e}")
            print("\nTrying to fall back to CPU mode...")
            try:
                # Try again with explicit CPU device
                tts_engine = KokoroTTSWrapper(
                    default_voice="af_bella",
                    speed=0.9,  # Slightly slower for better clarity
                    sink_name=sink_name,
                    device="cpu"  # Force CPU mode
                )
                await tts_engine.generate_speech("TTS system initialized in CPU mode.")
            except Exception as second_e:
                print(f"\nFallback TTS initialization also failed: {second_e}")
                raise  # Re-raise the exception if both attempts fail
        
        print(f"\nUsing model: {model}")
        


        # Initialize memory system if enabled
        memory_adapter = None
        if use_memory:
            memory_adapter = MemoryConversationAdapter()
            set_bella_memory_instance(memory_adapter.bella_memory)
            print("BellaMemory conversation adapter initialized and tools registered.")
            # Generate self-awareness summary for system prompt
            try:
                self_awareness_summary = await summarize_self_awareness()
                print("[System] Self-awareness summary generated for system prompt.")
            except Exception as e:
                print(f"[Warning] Failed to generate self-awareness summary: {e}")

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
                
                # Add memory context to enhance response (if memory system enabled)
                enhanced_context = ""
                memory_context = {}
                if memory_adapter and use_memory:
                    try:
                        # Pre-process input through memory system
                        memory_context = await memory_adapter.pre_process_input(transcribed_text, formatted_history)
                        
                        if memory_context and memory_context.get("memory_context"):
                            confidence = memory_context.get("confidence", "low")
                            print(f"\nFound relevant memory (confidence: {confidence}).")
                            enhanced_context = f"Based on my memory: {memory_context.get('memory_context')}"
                            
                            # Log memory source for debugging
                            if "memory_source" in memory_context:
                                print(f"Memory source: {memory_context['memory_source']}")
                    except Exception as mem_err:
                        print(f"\nWarning: Memory processing error: {mem_err}")

                # Generate response with tool capabilities
                if enhanced_context:
                    # Add memory context as a system message
                    sys_message = {"role": "system", "content": enhanced_context}
                    tool_history = [sys_message] + formatted_history
                else:
                    tool_history = formatted_history
                
                # Try generating response with tools first
                response, updated_history = await generate_chat_response_with_tools(
                    transcribed_text,
                    tool_history,
                    model,
                    timeout=15.0,
                    verbose=False,
                    self_awareness_summary=self_awareness_summary
                )
                
                print(f"Assistant: {response}")
                
                # Process for memory post-response (without changing spoken response)
                if memory_adapter and use_memory:
                    try:
                        # Post-process response through memory system
                        modified_response = await memory_adapter.post_process_response(
                            transcribed_text, response
                        )
                        
                        # Use the modified response if it's different (rare)
                        if modified_response and modified_response != response:
                            print("\nMemory system modified response.")
                            response = modified_response
                    except Exception as mem_err:
                        print(f"\nWarning: Memory post-processing error: {mem_err}")

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
                await asyncio.sleep(0.5)  # Reduced delay since Kokoro handles timing
                
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
    parser = argparse.ArgumentParser(description="Voice Assistant with Local LLM and Kokoro TTS")
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
        "--disable-memory",
        action="store_true",
        help="Disable the autonomous memory system"
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
        visualizer_path = os.path.join(os.path.dirname(__file__), "src", "ui", "bella_visualizer.py")
        visualizer_proc = subprocess.Popen([sys.executable, visualizer_path])

    try:
        if args.list_devices:
            list_audio_devices()
            sys.exit(0)
        asyncio.run(main_interaction_loop(args.model, args.sink, not args.disable_memory))
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
            except Exception:
                pass
