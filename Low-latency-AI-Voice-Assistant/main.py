import os
import sys
import asyncio
import argparse
import sounddevice as sd

# Add project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.utility.audio_session_manager import AudioSessionManager
from src.utility.buffered_recorder import BufferedRecorder, create_audio_stream
from src.llm.llm_interaction import generate_llm_response, get_available_models
from src.kokoro_tts.kokoro_tts import KokoroTTSWrapper

def list_audio_devices():
    """List all available audio output devices."""
    devices = sd.query_devices()
    print("\nAvailable audio devices:")
    for i, dev in enumerate(devices):
        out_channels = dev['max_output_channels']
        print(f"{i}: {dev['name']}")
        print(f"   Outputs: {out_channels}")

async def init_tts_engine(device_index: int = None):
    """Initialize the TTS engine with Kokoro's voice.
    
    Args:
        device_index (int, optional): Index of audio device to use
    """
    print("Initializing TTS engine...")
    engine = KokoroTTSWrapper(
        default_voice="af_heart",
        speed=0.9,  # Slightly slower for better clarity
        device_index=device_index
    )
    # Wait for engine to fully initialize
    await asyncio.sleep(1)
    return engine

async def main_interaction_loop(model: str = "Gemma3", device_index: int = None):
    """Main loop for capturing speech, generating responses, and playing audio.
    
    Args:
        model (str): Model nickname for Ollama (default: Gemma3)
        device_index (int, optional): Index of audio device to use
    """
    print("\nInitializing voice synthesis...")
    try:
        tts_engine = await init_tts_engine(device_index)
    except Exception as e:
        print(f"\nError initializing TTS engine: {e}")
        return
    
    # Print available models
    models = await get_available_models()
    print("\nAvailable models:")
    for model_id, info in models.items():
        status = "✅" if info['available'] else "❌"
        print(f"{status} {model_id}: {info['description']}")
    
    print(f"\nUsing model: {model}")
    print("\nVoice Assistant ready! Start speaking when ready.")
    print("Say 'stop' or 'exit' to end the conversation.\n")

    conversation_history = []  # Store history of user inputs and assistant responses
    
    try:
        # Create audio session manager with debug mode
        audio_manager = AudioSessionManager(gap_timeout=2.0, debug=True)
        recorder = BufferedRecorder()
        
        # Connect recorder to audio manager
        audio_manager.set_recorder(recorder)
        
        # Create and start the audio stream
        stream = await create_audio_stream(recorder)
        
        with stream:
            while True:
                # Ensure recording is fully stopped before starting new interaction
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
                recorder.is_recording = True
                
                # Wait for complete utterance
                transcribed_text, segments = await audio_manager.start_session()
                
                if not transcribed_text:
                    continue

                print(f"\nYou said: {transcribed_text}")

                if any(word in transcribed_text.lower() for word in ['stop', 'exit', 'quit']):
                    print("\nGoodbye!")
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
                response = await generate_llm_response(transcribed_text, history_text, model, timeout=15.0)
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
                    continue  # Skip to next iteration if TTS fails

                # Only resume recording after response has fully played
                await asyncio.sleep(1.0)  # Increased delay to ensure audio completion
                
                print("\nRecording resumed...")
                recorder.should_stop = False
                recorder.is_recording = True
                audio_manager.resume_session()
                print("\n" + "="*50)
            
    finally:
        # Ensure everything is properly cleaned up
        recorder.should_stop = True
        recorder.is_recording = False
        if 'stream' in locals():
            stream.stop()
            stream.close()
        tts_engine.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Assistant with Local LLM")
    parser.add_argument(
        "--model",
        default="Gemma3",
        help="Model to use for responses (e.g., Gemma3, hermes8b, dolphin8b)"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit"
    )
    parser.add_argument(
        "--device",
        type=int,
        help="Index of audio output device to use"
    )
    
    args = parser.parse_args()
    
    if args.list_devices:
        list_audio_devices()
        sys.exit(0)
        
    asyncio.run(main_interaction_loop(args.model, args.device))