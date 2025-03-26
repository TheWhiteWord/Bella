import asyncio
import argparse
from Models_interaction.audio_session_manager import AudioSessionManager
from Models_interaction.buffered_recorder import BufferedRecorder, create_audio_stream
from Models_interaction.llm_interaction import generate_llm_response, get_available_models
import os
from RealtimeTTS import TextToAudioStream, KokoroEngine

async def init_tts_engine():
    """Initialize the TTS engine with Kokoro's voice."""
    engine = KokoroEngine(voice="Bella")
    stream = TextToAudioStream(engine=engine)
    return stream

async def main_interaction_loop(model: str = "Gemma3"):
    """Main loop for capturing speech, generating responses, and playing audio.
    
    Args:
        model (str): Model nickname for Ollama (default: Gemma3)
    """
    print("\nInitializing voice synthesis...")
    tts_engine = await init_tts_engine()
    
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
                tts_engine.feed(response)
                await tts_engine.play_async()
                while tts_engine.is_playing():
                    await asyncio.sleep(0.1)

                # Only resume recording after response has fully played
                await asyncio.sleep(0.5)  # Add small delay to prevent cutting off end of response
                
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Assistant with Local LLM")
    parser.add_argument(
        "--model",
        default="Gemma3",
        help="Model to use for responses (e.g., Gemma3, hermes8b, dolphin8b)"
    )
    
    args = parser.parse_args()
    asyncio.run(main_interaction_loop(args.model))