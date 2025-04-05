"""Client for interacting with Ollama models.

This module provides functions for generating text using Ollama models,
with proper formatting for Ollama's API requirements.
"""
import os
import sys
import json
import re
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='[%(asctime)s] %(levelname)-8s %(message)s',
                   datefmt='%m/%d/%y %H:%M:%S')

try:
    import ollama
except ImportError:
    logging.warning("Ollama Python package not found. Install with: pip install ollama")

async def generate(
    prompt: str,
    model: str = "mistral",
    system_prompt: str = "",
    verbose: bool = False,
    timeout: float = 30.0
) -> str:
    """Generate text using Ollama model.
    
    Args:
        prompt: The prompt to send to the model
        model: The model name to use
        system_prompt: System instructions for the model
        verbose: Whether to print debug information
        timeout: Maximum time to wait for response in seconds
        
    Returns:
        str: Generated text response
    """
    try:
        if verbose:
            print(f"\nGenerating response with Ollama")
            print(f"Model nickname: {model}")
        
        # Check if model is a nickname and get the actual model name
        
        actual_model = model
        try:
            from .config_manager import ModelConfig
            model_config = ModelConfig()
            model_info = model_config.get_model_config(model)
            if model_info and "model" in model_info:
                actual_model = model_info["model"]            
                if verbose:
                    print(f"Using actual model name from config: {actual_model}")
        except Exception as e:
            if verbose:
                print(f"Error resolving model name: {e}")
            # Continue with the original model name
        
        # Make async call to Ollama
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": 0.7,
                    "top_k": 50,
                    "top_p": 0.95
                }
            )
        )
        
        if verbose:
            print(f"Actual model name: {response.get('model', model)}")
        
        if "message" not in response:
            return "Error: Missing response message"
            
        content = response["message"].get("content", "")
        
        return content
        
    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        return f"Error generating response: {str(e)}"

async def list_available_models() -> List[str]:
    """List models available locally in Ollama.
    
    Returns:
        List[str]: List of available model names
    """
    try:
        loop = asyncio.get_event_loop()
        models = await loop.run_in_executor(None, ollama.list)
        return [model["name"] for model in models["models"]]
    except Exception as e:
        logging.error(f"Error listing models: {str(e)}")
        return []