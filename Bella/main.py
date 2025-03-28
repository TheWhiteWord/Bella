"""Main application module for voice assistant with Kokoro TTS integration.

This module coordinates audio recording, speech recognition, LLM interaction,
and text-to-speech using Kokoro. Uses PipeWire/PulseAudio for audio I/O.
"""
import os
import sys
import asyncio
import argparse
import subprocess
from typing import Dict, Any, Optional, Tuple, List
import re

# Add project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.utility.audio_session_manager import AudioSessionManager
from src.utility.buffered_recorder import BufferedRecorder, create_audio_stream
from src.llm.chat_manager import generate_chat_response, get_available_models, format_search_response
from src.audio.kokoro_tts.kokoro_tts import KokoroTTSWrapper
from src.llm.config_manager import ModelConfig
from src.agents.search_agent import SearchAgent

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

async def main_interaction_loop(model: str = None, sink_name: Optional[str] = None) -> None:
    """Main loop for capturing speech, generating responses, and playing audio.
    
    Args:
        model (str, optional): Model nickname for Ollama. If None, uses default from config
        sink_name (str, optional): Name of PulseAudio sink to use for output
    """
    print("\nInitializing voice assistant components...")
    tts_engine = None
    recorder = None
    
    try:
        # Get model from config if not specified
        if model is None:
            model_config = ModelConfig()
            model = model_config.get_default_model()
            
        tts_engine = await init_tts_engine(sink_name)
        
        # Print available models
        models = await get_available_models()
        print("\nAvailable models:")
        for model_id, info in models.items():
            status = "✅" if info['available'] else "❌"
            print(f"{status} {model_id}: {info['description']}")
        
        print(f"\nUsing model: {model}")
        await tts_engine.generate_speech("Voice Assistant ready! Start speaking when ready.")
        print("\nSay 'stop' or 'exit' to end the conversation.\n")

        conversation_history = []  # Store history of user inputs and assistant responses
        
        # Create audio session manager with debug mode
        audio_manager = AudioSessionManager(gap_timeout=2.0, debug=True)
        recorder = BufferedRecorder()
        
        # Connect recorder to audio manager
        audio_manager.set_recorder(recorder)
        
        # Initialize audio settings
        recorder.initialize_audio_settings()
        
        while True:
            try:
                # Ensure recording is fully stopped before starting new interaction
                if recorder.is_recording:
                    recorder.should_stop = True
                    recorder.is_recording = False
                    audio_manager.pause_session()
                    await asyncio.sleep(0.2)  # Give time for everything to stop
                
                # Reset states for new interaction
                recorder.reset_state()
                audio_manager.resume_session()
                
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

                if any(word in transcribed_text.lower() for word in ['exit', 'quit']):
                    await tts_engine.generate_speech("Goodbye!")
                    break

                # Fully stop recording while processing response
                print("\nRecording paused...")
                recorder.should_stop = True
                recorder.is_recording = False
                audio_manager.pause_session()
                await asyncio.sleep(0.3)  # Give more time to ensure stop

                # Format conversation history for context
                history_text = ""
                for i, entry in enumerate(conversation_history[-3:]):
                    role = "User" if i % 2 == 0 else "Assistant"
                    history_text += f"{role}: {entry}\n"

                # Generate response using local Ollama model
                print(f"\nThinking... (using {model})")
                response = await generate_chat_response(transcribed_text, history_text, model, timeout=15.0)
                print(f"Assistant: {response}")
                
                # Check if this is a search acknowledgment
                if "[SEARCH_INITIATED]" in response:
                    # Extract and speak the acknowledgment
                    acknowledgment = response.split("[SEARCH_INITIATED]")[0].strip()
                    print(f"Assistant: {acknowledgment}")
                    await tts_engine.generate_speech(acknowledgment)
                    
                    # Extract the search query and perform the search
                    search_query = re.sub(r'^(?:search|look up|find|tell me about)\s+(?:for|about)?\s*', '', transcribed_text, flags=re.IGNORECASE)
                    search_agent = SearchAgent(model=model)
                    research_results = await search_agent.research_topic(search_query)
                    
                    # Format results conversationally
                    response = await format_search_response(research_results)
                else:
                    print(f"Assistant: {response}")

                # Update conversation history
                conversation_history.append(transcribed_text)  # User input
                conversation_history.append(response)  # Assistant response

                # Convert response to speech and play it
                print("\nGenerating speech...")
                try:
                    await tts_engine.generate_speech(response)
                except Exception as e:
                    print(f"\nError during speech generation: {e}")
                    continue

                # Only resume recording after response has fully played
                await asyncio.sleep(0.5)  # Reduced delay since Kokoro handles timing
                
                print("\nRecording resumed...")
                recorder.should_stop = False
                recorder.start_recording()
                audio_manager.resume_session()
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
    
    args = parser.parse_args()
    
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
