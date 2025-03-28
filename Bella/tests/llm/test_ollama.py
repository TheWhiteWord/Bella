import asyncio
import ollama
import time
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import os

def save_test_results(results: list[Dict[str, Any]]):
    """Save test results to a markdown file"""
    results_dir = Path("Bella/results/model_evaluations")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"ollama_test_{timestamp}.md"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Ollama Integration Test Results\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for result in results:
            f.write(f"## Test Case: {result['test_name']}\n")
            f.write(f"Model: {result['model']}\n")
            f.write(f"Prompt: {result['prompt']}\n")
            if 'response' in result:
                f.write(f"Response: {result['response']}\n")
            if 'time' in result:
                f.write(f"Generation time: {result['time']:.2f}s\n")
            if 'error' in result:
                f.write(f"Error: {result['error']}\n")
            f.write("\n")
    
    print(f"\nTest results saved to: {output_file}")
    return output_file

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
        
        return {
            "test_name": "Basic Response Generation",
            "model": model,
            "prompt": test_prompt,
            "response": response['message']['content'],
            "time": generation_time
        }
        
    except Exception as e:
        print(f"Error testing {model}: {e}")
        print(f"Make sure the model '{model}' is available in Ollama")
        return {
            "test_name": "Basic Response Generation",
            "model": model,
            "prompt": test_prompt,
            "error": str(e)
        }

async def run_test_suite():
    """Run a series of tests for Ollama integration"""
    print("Starting Ollama integration tests...")
    
    results = []
    
    # Test 1: Connection
    connection_result = await test_ollama_connection()
    results.append({
        "test_name": "Connection Test",
        "model": "N/A",
        "prompt": "N/A",
        "response": "Success" if connection_result else "Failed"
    })
    
    if not connection_result:
        save_test_results(results)
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
        result = await test_model_response(**test)
        if result:
            results.append(result)
        print("="*50)
    
    # Save all results
    save_test_results(results)

if __name__ == "__main__":
    asyncio.run(run_test_suite())