"""Module for handling core Ollama API interactions.

This module provides low-level functions for interacting with Ollama,
including response generation and model listing.
"""

import re
import ollama
import asyncio
from typing import Optional, List
from .config_manager import ModelConfig

def clean_response(text: str) -> str:
    """Remove emojis and emoticons from text.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text without emojis/emoticons
    """
    # Remove unicode emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    
    # Remove ASCII-style emoticons
    ascii_pattern = re.compile(r'[:;=]-?[)(/\\|pPoO]')
    
    # Clean the text
    text = emoji_pattern.sub('', text)
    text = ascii_pattern.sub('', text)
    
    # Remove multiple spaces and trim
    text = ' '.join(text.split())
    
    return text.strip()

async def generate(
    prompt: str,
    model: str = None,
    system_prompt: str = None,
    verbose: bool = False,
    config_path: str = None,
    timeout: float = 15.0
) -> Optional[str]:
    """Generate a response using local Ollama model asynchronously.
    
    Args:
        prompt (str): The user prompt to generate a response for
        model (str, optional): Model nickname from config. If None, uses default model
        system_prompt (str, optional): System prompt that defines AI behavior. Required
        verbose (bool, optional): Whether to print debug information
        config_path (str, optional): Path to models.yaml config file
        timeout (float, optional): Timeout in seconds for generation
        
    Returns:
        Optional[str]: Generated response or None if generation failed
    """
    try:
        if not system_prompt:
            raise ValueError("system_prompt is required")
            
        # Load model config
        model_config = ModelConfig(config_path)
        if not model:
            model = model_config.get_default_model()
            print(f"Using default model: {model}")
            
        model_info = model_config.get_model_config(model)
        if not model_info:
            print(f"No configuration found for model: {model}")
            return None
            
        actual_model_name = model_info['name']
        
        # Check if model is actually available in Ollama
        available_models = await list_available_models()
        if not available_models:
            print("Failed to get list of available models from Ollama")
            return None
            
        print(f"\n=== Model Configuration ===")
        print(f"Requested model: {model}")
        print(f"Model config found: {bool(model_info)}")
        print(f"Actual model name: {actual_model_name}")
        print(f"Available models: {available_models}")
        print(f"Model available: {actual_model_name in available_models}")
        print(f"Parameters: {model_info.get('parameters', {})}")
        print("========================\n")
        
        if actual_model_name not in available_models:
            print(f"Model {actual_model_name} is not available in Ollama")
            return None
            
        if verbose:
            print(f"\nGenerating response with Ollama")
            print(f"Model nickname: {model}")
            print(f"Actual model name: {actual_model_name}")
            print(f"Parameters: {model_info['parameters']}")
            
        # Generate response using ollama library with timeout
        try:
            response_future = asyncio.create_task(
                asyncio.to_thread(
                    ollama.chat,
                    model=actual_model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    options=model_info.get('parameters', {})
                )
            )
            
            response = await asyncio.wait_for(response_future, timeout=timeout)
            
            if not response or 'message' not in response:
                if verbose:
                    print("Received empty or invalid response from Ollama")
                return None
                
            response_text = clean_response(response['message']['content'].strip())
            return response_text
            
        except asyncio.TimeoutError:
            if verbose:
                print(f"LLM response timed out after {timeout}s")
            return None
            
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        print(f"Attempted to use model nickname: {model}")
        if 'model_info' in locals():
            print(f"Model info: {model_info}")
        if "connection" in str(e).lower():
            print("Make sure Ollama is running (ollama serve)")
        return None

async def list_available_models() -> List[str]:
    """Get list of models directly from Ollama service.
    
    Returns:
        List[str]: List of available model names
    """
    try:
        response = ollama.list()
        if hasattr(response, 'models') and isinstance(response.models, list):
            return [model.model for model in response.models]
        return []
    except Exception as e:
        print(f"Error listing Ollama models: {str(e)}")
        return []