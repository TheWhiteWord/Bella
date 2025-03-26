import asyncio
import argparse
from Models_interaction.audio_session_manager import AudioSessionManager
from utils.llm_interaction import generate_llm_response, get_available_models
from utils.csm_tts import CSMSpeechProcessor, play_audio
import os

async def init_tts_engine():
    """Initialize the TTS engine with Bella's voice."""
    # Get paths relative to main.py
    root_dir = os.path.dirname(__file__)
    reference_audio = os.path.join(root_dir, "clone_files", "bella_edit.mp3")
    reference_text_path = os.path.join(root_dir, "clone_files", "Bella_transcript.txt")
    
    with open(reference_text_path, 'r') as f:
        reference_text = f.read().strip()
    
    return CSMSpeechProcessor(
        reference_audio=reference_audio,
        reference_text=reference_text
    )

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
        # Create audio session manager
        audio_manager = AudioSessionManager(gap_timeout=2.0, debug=True)
        
        while True:
            # Step 1: Start a new audio session and wait for complete utterance
            print("\nWaiting for voice...")
            transcribed_text, segments = await audio_manager.start_session()
            
            if not transcribed_text:
                continue

            print(f"\nYou said: {transcribed_text}")

            if any(word in transcribed_text.lower() for word in ['stop', 'exit', 'quit']):
                print("\nGoodbye!")
                break

            # Append user input to history
            conversation_history.append({"User": transcribed_text})
            
            # Step 2: Generate a response using local Ollama model
            print(f"\nThinking... (using {model})")
            history_context = ' '.join(
                [f"{key}: {value}" for entry in conversation_history[-3:] for key, value in entry.items()]
            )
            response = await generate_llm_response(transcribed_text, history_context, model)
            print(f"Assistant: {response}")

            # Append assistant response to history
            conversation_history.append({"Assistant": response})

            # Step 3: Convert response to speech and play it
            print("\nGenerating speech...")
            audio_file = await tts_engine.convert_text_to_speech(response)
            
            if audio_file:
                print("\nPlaying response...")
                play_audio(audio_file)
                # Clean up audio file
                try:
                    os.remove(audio_file)
                except:
                    pass
                    
            print("\n" + "="*50 + "\n")
            
    finally:
        # Clean up TTS engine
        if tts_engine:
            tts_engine.reset_all_contexts()
            del tts_engine

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Assistant with Local LLM")
    parser.add_argument(
        "--model",
        default="Gemma3",
        help="Model to use for responses (e.g., Gemma3, hermes8b, dolphin8b)"
    )
    
    args = parser.parse_args()
    asyncio.run(main_interaction_loop(args.model))