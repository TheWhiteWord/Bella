import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import asyncio
from Models_interaction.faster_whisper_stt_tiny import capture_audio, transcribe_audio
from utils.csm_stt import CSMSpeechProcessor

# Test sentences that worked well with Whisper tiny
TEST_SENTENCES = [
    "Hey, I'm checking the text of speech.",
    "Wow, this is way faster.",
    "So this is decent.",
]

async def test_basic_stt():
    """Basic test of STT functionality with predefined test sentences"""
    print("\n=== Testing Basic STT with CSM ===")
    
    # Initialize CSM for speech processing
    csm = CSMSpeechProcessor(
        reference_audio=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                   "clone_files", "bella_edit.mp3"),
        reference_text="Hello, I am Bella, your voice assistant."
    )
    
    results = []
    
    for i, test_sentence in enumerate(TEST_SENTENCES, 1):
        print(f"\nTest {i}: Converting '{test_sentence}' to speech...")
        
        # Process speech
        audio_file = await csm.speech_to_text(test_sentence)
        if not audio_file:
            print(f"Failed to process speech for test {i}")
            continue
            
        print(f"Successfully processed speech, now transcribing...")
        
        # Transcribe the processed audio
        transcribed_text = await transcribe_audio(audio_file)
        
        print(f"Original: '{test_sentence}'")
        print(f"Transcribed: '{transcribed_text}'")
        
        # Store results
        results.append({
            'original': test_sentence,
            'transcribed': transcribed_text,
            'success': test_sentence.lower() in transcribed_text.lower()
        })
        
        # Cleanup
        try:
            os.remove(audio_file)
        except:
            pass
            
    # Print summary
    print("\n=== Test Summary ===")
    successful = sum(1 for r in results if r['success'])
    print(f"Successful transcriptions: {successful}/{len(TEST_SENTENCES)}")
    
    for i, result in enumerate(results, 1):
        status = "✓" if result['success'] else "✗"
        print(f"\nTest {i}: {status}")
        print(f"Original: {result['original']}")
        print(f"Transcribed: {result['transcribed']}")

if __name__ == "__main__":
    asyncio.run(test_basic_stt())