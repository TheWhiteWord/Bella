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
    system_prompt: str = "Keep your response short and concise.",
    verbose: bool = False,
    config_path: str = None,
    timeout: float = 15.0
) -> Optional[str]:
    """Generate a response using local Ollama model asynchronously.
    
    Args:
        prompt (str): The input text to generate a response for
        model (str): Model nickname from config
        system_prompt (str): System prompt for setting model behavior
        verbose (bool): Whether to print debug info
        config_path (str): Path to models.yaml config file
        timeout (float): Maximum time to wait for response in seconds
    
    Returns:
        str: Generated response, or None if there was an error
    """
    try:
        # Load model config
        model_config = ModelConfig(config_path)
        if not model:
            model = model_config.get_default_model()
            
        model_info = model_config.get_model_config(model)
        
        if verbose:
            print(f"\nGenerating response with Ollama ({model})")
            print(f"Model: {model_info['name']}")
            print(f"Parameters: {model_info['parameters']}")
            
        # Generate response using ollama library with timeout
        response_future = asyncio.create_task(
            asyncio.to_thread(
                ollama.chat,
                model=model_info['name'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                options=model_info['parameters']
            )
        )
        
        try:
            response = await asyncio.wait_for(response_future, timeout=timeout)
        except asyncio.TimeoutError:
            if verbose:
                print(f"LLM response timed out after {timeout}s")
            return None
        
        response_text = clean_response(response['message']['content'].strip())
            
        return response_text
        
    except Exception as e:
        print(f"Error generating response: {str(e)}")
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