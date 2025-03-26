import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import asyncio
from utils.csm_tts import CSMSpeechProcessor
import pygame
import torch
import torchaudio

# Test responses with varying prosody and semantic characteristics
TEST_RESPONSES = [
    "Hello! I'm here to help you.",  # Simple greeting with enthusiasm
    "That's an interesting question. Let me think about it...",  # Thoughtful pause
    "Based on my analysis, there are three key points to consider.",  # Structured response
]

def play_audio_file(file_path):
    """Play an audio file using pygame and wait for it to finish."""
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        print(f"\nPlaying audio from: {file_path}")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"Error playing audio: {e}")
    finally:
        pygame.mixer.quit()

async def test_csm_tts_basic():
    """Basic test of CSM's text-to-speech capabilities with voice cloning."""
    print("\n=== Testing CSM Text-to-Speech ===")
    
    # Get absolute paths
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    audio_dir = os.path.join(root_dir, "Testing", "audio files")
    os.makedirs(audio_dir, exist_ok=True)
    
    # Initialize CSM with Bella's voice reference
    reference_audio = os.path.join(root_dir, "clone_files", "bella_edit.mp3")
    reference_text_path = os.path.join(root_dir, "clone_files", "Bella_transcript.txt")
    
    with open(reference_text_path, 'r') as f:
        reference_text = f.read().strip()
    
    print(f"Loading CSM with voice reference: {reference_audio}")
    csm = CSMSpeechProcessor(
        reference_audio=reference_audio,
        reference_text=reference_text
    )
    
    # Test each response type
    for i, response in enumerate(TEST_RESPONSES, 1):
        print(f"\nTest {i}: Converting to speech: '{response}'")
        
        try:
            # Generate speech from text
            audio_file = await csm.convert_text_to_speech(response)
            
            if audio_file and os.path.exists(audio_file):
                print(f"✓ Successfully generated speech file: {audio_file}")
                
                # Print file stats
                file_size = os.path.getsize(audio_file) / 1024  # KB
                print(f"File size: {file_size:.2f}KB")
                
                # Load and play the generated audio
                play_audio_file(audio_file)
                
                # Wait for user confirmation before next test
                input("\nPress Enter for next test...")
            else:
                print(f"✗ Failed to generate speech for test {i}")
                
        except Exception as e:
            print(f"✗ Error during speech generation: {str(e)}")
            continue
            
    print("\nTest complete!")

if __name__ == "__main__":
    asyncio.run(test_csm_tts_basic())