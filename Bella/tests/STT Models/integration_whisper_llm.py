import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import asyncio
from Models_interaction.faster_whisper_stt_tiny import capture_audio, transcribe_audio
from Models_interaction.llm_response import generate
import time

# Main interaction loop
async def main_interaction_loop():
    while True:
        # Step 1: Capture and transcribe speech
        audio_file = await capture_audio()
        if not audio_file:
            print("Please try speaking again.")
            continue

        transcribed_text = await transcribe_audio(audio_file)
        if not transcribed_text or transcribed_text.isspace():
            print("No speech detected, please try again.")
            continue
            
        print(f"You said: {transcribed_text}\n")

        # Step 2: Check for stop command
        if 'stop' in transcribed_text.lower():
            print("Goodbye!")
            break

        # Step 3: Generate a response from LLM
        response = await generate(
            prompt=transcribed_text,
            system_prompt="Be concise, helpful, and friendly. Keep responses under 20 words.",
            verbose=True
        )
        
        if response:
            print(f"Assistant: {response}\n")
        else:
            print("Sorry, I couldn't generate a response. Please try again.\n")

# Run the interaction loop in an asyncio event loop
if __name__ == "__main__":
    asyncio.run(main_interaction_loop())
