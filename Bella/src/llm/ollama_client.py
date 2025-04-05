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
    system_prompt: str = "", 
    verbose: bool = False,
    timeout: float = 30.0  # Increase default timeout to 30s
) -> Optional[str]:
    """Generate a response from Ollama.
    
    Args:
        prompt (str): The prompt to send to Ollama
        model (str, optional): The model to use. If None, uses default from config
        system_prompt (str): Optional system prompt
        verbose (bool): Whether to print debug info
        timeout (float): Timeout in seconds
        
    Returns:
        Optional[str]: Generated text or None if failed
    """
    # Get model from config if not specified
    if model is None:
        model_config = ModelConfig()
        model = model_config.get_default_model()
    
    if verbose:
        print("\nGenerating response with Ollama")
        print(f"Model nickname: {model}")
    
    try:
        # Get the actual model name from the config if this is a model nickname
        model_config = ModelConfig()
        model_info = model_config.get_model_config(model)
        
        # If we found a config for this model, use the actual name from the config
        if model_info and 'name' in model_info:
            actual_model = model_info['name']
            if verbose:
                print(f"Using actual model name from config: {actual_model}")
        else:
            # If no config found, use the provided model name directly
            actual_model = model
            if verbose:
                print(f"No config found for {model}, using as-is")
        
        # Add retry logic for better reliability
        max_retries = 2
        retry_delay = 2.0
        
        for attempt in range(max_retries + 1):
            try:
                # Use asyncio.wait_for to enforce timeout
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: ollama.chat(
                            model=actual_model,
                            messages=[
                                {
                                    'role': 'system',
                                    'content': system_prompt
                                },
                                {
                                    'role': 'user',
                                    'content': prompt
                                }
                            ],
                            stream=False,
                            options={
                                "num_predict": 512  # Limit response length
                            }
                        )
                    ),
                    timeout=timeout
                )
                
                if verbose:
                    print(f"Actual model name: {result.get('model', 'unknown')}")
                
                # Extract and return the response content
                if "message" in result and "content" in result["message"]:
                    return result["message"]["content"].strip()
                else:
                    print("Unexpected response format from Ollama")
                    return None
                    
            except asyncio.TimeoutError:
                if attempt < max_retries:
                    print(f"LLM response timed out after {timeout}s, retrying ({attempt+1}/{max_retries})...")
                    await asyncio.sleep(retry_delay)
                else:
                    print(f"LLM response timed out after {timeout}s")
                    return None
            except Exception as e:
                if attempt < max_retries:
                    print(f"Error generating response: {e}, retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    print(f"Error generating response after {max_retries} retries: {e}")
                    return None
                
    except Exception as e:
        print(f"Unexpected error in generate: {e}")
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