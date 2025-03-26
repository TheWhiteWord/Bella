"""
Main entry point for the Voice Assistant application.
Handles the voice recording, transcription, LLM response generation, and text-to-speech conversion.
"""

import asyncio
import argparse
import os
from Models_interaction.faster_whisper_stt_tiny import capture_and_transcribe
from utils.llm_interaction import generate_llm_response
from utils.csm_tts import CSMSpeechProcessor
from utils.csm_tts import play_audio

async def main_interaction_loop():
    """Main loop for voice assistant interaction.
    Handles continuous voice input, transcription, LLM response, and speech synthesis.
    """
    # Initialize conversation history
    conversation_history = []
    
    # Initialize CSM TTS with Bella's voice
    reference_audio = os.path.join("clone_files", "bella_edit.mp3")
    reference_text_path = os.path.join("clone_files", "Bella_transcript.txt")
    
    with open(reference_text_path, 'r') as f:
        reference_text = f.read().strip()
    
    tts_engine = CSMSpeechProcessor(
        reference_audio=reference_audio,
        reference_text=reference_text
    )
    
    print("\nVoice Assistant ready! Start speaking when ready.")
    print("Say 'stop' or 'exit' to end the conversation.\n")
    
    current_buffer = []  # Buffer to accumulate transcribed text

    while True:
        # Step 1: Capture and transcribe speech with sentence buffering
        print("Listening...")
        transcribed_text, is_complete = await capture_and_transcribe(debug=True)
        
        if transcribed_text:
            current_buffer.append(transcribed_text)
            
            if is_complete:  # We've detected sufficient silence
                full_transcription = " ".join(current_buffer)
                print(f"\nYou said: {full_transcription}")
                
                # Check for exit commands
                if any(word in full_transcription.lower() for word in ['stop', 'exit', 'quit']):
                    print("\nGoodbye!")
                    break

                print("\nThinking...")
                
                # Get conversation history as context (last 3 exchanges)
                history_context = ' '.join(
                    [f"{key}: {value}" for entry in conversation_history[-3:] for key, value in entry.items()]
                )
                
                # Generate response using Gemma3
                response = await generate_llm_response(
                    user_input=full_transcription,
                    history_context=history_context,
                    model="gemma3"  # Using Gemma3 model specifically
                )
                
                if response:
                    print(f"Assistant: {response}")
                    
                    # Update conversation history
                    conversation_history.append({
                        "user": full_transcription,
                        "assistant": response
                    })
                    
                    # Convert response to speech
                    try:
                        audio_file = await tts_engine.convert_text_to_speech(response)
                        if audio_file:
                            await play_audio(audio_file)
                    except Exception as e:
                        print(f"Error in speech synthesis: {e}")
                
                # Clear the buffer after processing
                current_buffer = []
            
        await asyncio.sleep(0.1)  # Small sleep to prevent busy waiting

# Run the main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Assistant with CSM TTS")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    try:
        asyncio.run(main_interaction_loop())
    except KeyboardInterrupt:
        print("\nStopping voice assistant...")
    except Exception as e:
        print(f"\nError: {e}")
