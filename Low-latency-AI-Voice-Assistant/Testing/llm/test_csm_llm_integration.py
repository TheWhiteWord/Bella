import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import asyncio
import pytest
import pytest_asyncio
from utils.csm_tts import CSMSpeechProcessor
from utils.llm_interaction import generate_llm_response
import torch
import torchaudio

# Test cases with different types of queries and expected response characteristics
TEST_CASES = [
    {
        "input": "Hello, how are you?",
        "context": "",
        "expected_type": "greeting",
        "desc": "Basic greeting to test natural conversation flow"
    },
    {
        "input": "What is artificial intelligence?",
        "context": "User: Hello! Assistant: Hi there! How can I help you today?",
        "expected_type": "informative",
        "desc": "Technical explanation to test clarity and articulation"
    },
    {
        "input": "Tell me a short joke.",
        "context": "",
        "expected_type": "entertainment",
        "desc": "Tests timing and expression in casual speech"
    },
    {
        "input": "Can you explain quantum computing?",
        "context": "User: I'm interested in advanced physics.",
        "expected_type": "educational",
        "desc": "Complex topic to test clear pronunciation of technical terms"
    },
    {
        "input": "How do you feel about rainy days?",
        "context": "",
        "expected_type": "opinion",
        "desc": "Tests emotional expression and natural prosody"
    },
    {
        "input": "What's the recipe for chocolate cake?",
        "context": "User: I love baking.",
        "expected_type": "instructional",
        "desc": "Tests list delivery and step-by-step explanation"
    }
]

@pytest_asyncio.fixture(scope="module")
async def csm():
    """Fixture to set up CSM processor for tests"""
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    reference_audio = os.path.join(root_dir, "clone_files", "bella_edit.mp3")
    reference_text_path = os.path.join(root_dir, "clone_files", "Bella_transcript.txt")
    
    with open(reference_text_path, 'r') as f:
        reference_text = f.read().strip()
    
    processor = CSMSpeechProcessor(
        reference_audio=reference_audio,
        reference_text=reference_text
    )
    return processor  # Changed from yield to return

@pytest.mark.asyncio
@pytest.mark.parametrize("test_case", TEST_CASES)
async def test_llm_tts_pipeline(csm: CSMSpeechProcessor, test_case):
    """Test the LLM -> TTS pipeline."""
    input_text = test_case["input"]
    context = test_case["context"]
    expected_type = test_case["expected_type"]
    
    print(f"\nTesting with input: '{input_text}'")
    print(f"Context: '{context}'")
    print(f"Expected response type: {expected_type}")
    
    try:
        # Step 1: Generate LLM Response
        print("\nGenerating LLM response...")
        llm_response = await generate_llm_response(
            user_input=input_text,
            history_context=context,
            model="Gemma3"
        )
        assert llm_response is not None, "LLM failed to generate response"
        print(f"LLM Response: {llm_response}")
        
        # Step 2: Convert response to speech
        print("\nConverting LLM response to speech...")
        audio_file = await csm.convert_text_to_speech(llm_response)
        
        # Validate output
        assert audio_file is not None, "Failed to generate audio file"
        assert os.path.exists(audio_file), f"Audio file {audio_file} does not exist"
        
        # Get audio stats
        audio_info = torchaudio.info(audio_file)
        print(f"Audio duration: {audio_info.num_frames / audio_info.sample_rate:.2f}s")
        print(f"Sample rate: {audio_info.sample_rate}Hz")
        
        # Play the audio for manual verification
        from utils.csm_tts import play_audio
        play_audio(audio_file)
        
        # Get user feedback
        feedback = input("\nDid the audio sound natural and appropriate? (y/n): ")
        assert feedback.lower() == 'y', "User reported audio quality issues"
        
        # Clean up
        os.remove(audio_file)
            
    except Exception as e:
        pytest.fail(f"Test failed: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])