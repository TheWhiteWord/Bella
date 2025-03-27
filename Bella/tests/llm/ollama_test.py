import asyncio
import ollama
import time
from typing import Optional, Dict, Any

async def test_ollama_connection() -> bool:
    """Test if Ollama service is running and accessible"""
    try:
        response = ollama.list()
        print("Ollama connection successful!")
        
        # Extract model names directly from the ListResponse
        if hasattr(response, 'models') and isinstance(response.models, list):
            models = [model.model for model in response.models]
            print(f"Available models: {models}")
        else:
            print("Unexpected response format:")
            print(f"Response type: {type(response)}")
            print(f"Response content: {response}")
        return True
    except Exception as e:
        print(f"Failed to connect to Ollama: {e}")
        print("Make sure Ollama is running (ollama serve)")
        return False

async def test_model_response(
    model: str = "Gemma3:latest", 
    test_prompt: str = "What is 2+2? Answer in one word.",
    system_prompt: str = "You are a helpful assistant. Keep responses very brief."
) -> Optional[Dict[Any, Any]]:
    """Test model response generation with a simple prompt"""
    try:
        print(f"\nTesting {model} model...")
        print(f"Prompt: {test_prompt}")
        
        start_time = time.time()
        
        response = await asyncio.to_thread(
            ollama.chat,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": test_prompt}
            ]
        )
        
        generation_time = time.time() - start_time
        
        print(f"Response: {response['message']['content']}")
        print(f"Generation time: {generation_time:.2f} seconds")
        return response
        
    except Exception as e:
        print(f"Error testing {model}: {e}")
        print(f"Make sure the model '{model}' is available in Ollama")
        return None

async def run_test_suite():
    """Run a series of tests for Ollama integration"""
    print("Starting Ollama integration tests...")
    
    # Test 1: Connection
    if not await test_ollama_connection():
        return
    
    # Test 2: Basic response generation with models from our config
    test_cases = [
        {
            "model": "Gemma3:latest",
            "test_prompt": "What is 2+2? Answer in one word.",
            "system_prompt": "You are a helpful assistant. Keep responses very brief."
        },
        {
            "model": "Gemma3:latest",
            "test_prompt": "Write a haiku about programming.",
            "system_prompt": "You are a poet. Respond with exactly one haiku."
        }
    ]
    
    for test in test_cases:
        print("\n" + "="*50)
        await test_model_response(**test)
        print("="*50)

if __name__ == "__main__":
    asyncio.run(run_test_suite())