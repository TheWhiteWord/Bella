import asyncio
import argparse
from utils.audio_processing import capture_and_transcribe_audio
from utils.llm_interaction import generate_llm_response, get_available_models
from utils.tts_conversion import convert_text_to_speech, play_audio
import os

async def main_interaction_loop(model: str = "hermes8b"):
    """Main loop for capturing speech, generating responses, and playing audio.
    
    Args:
        model (str): Model nickname for Ollama (default: hermes8b)
    """
    # Print available models
    models = get_available_models()
    print("\nAvailable models:")
    for model_id, description in models.items():
        print(f"- {model_id}: {description}")
    print(f"\nUsing model: {model}")
    
    conversation_history = []  # Store history of user inputs and assistant responses
    
    print("\nVoice Assistant ready! Start speaking when ready.")
    print("Say 'stop' or 'exit' to end the conversation.\n")

    while True:
        # Step 1: Capture and transcribe speech
        print("Listening...")
        transcribed_text = await capture_and_transcribe_audio()
        if not transcribed_text:
            print("Please try speaking again.")
            continue

        print(f"\nYou said: {transcribed_text}")

        if any(word in transcribed_text.lower() for word in ['stop', 'exit', 'quit']):
            print("Goodbye!")
            break

        # Append user input to history
        conversation_history.append({"User": transcribed_text})
        
        # Step 2: Generate a response using local Ollama model
        print(f"\nThinking... (using {model})")
        history_context = ' '.join(
            [f"{key}: {value}" for entry in conversation_history[-3:] for key, value in entry.items()]
        )
        response = generate_llm_response(transcribed_text, history_context, model)
        print(f"Assistant: {response}")

        # Append assistant response to history
        conversation_history.append({"Assistant": response})

        # Step 3: Convert response to speech and play it
        print("\nGenerating speech...")
        audio_file = await convert_text_to_speech(response, rate="+0%", pitch="+0Hz")
        play_audio(audio_file)
        
        # Clean up audio file
        try:
            os.remove(audio_file)
        except:
            pass
            
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Assistant with Local LLM")
    parser.add_argument(
        "--model",
        default="hermes8b",
        help="Model to use for responses (e.g., hermes8b, dolphin8b, hermes24b)"
    )
    
    args = parser.parse_args()
    asyncio.run(main_interaction_loop(args.model))
