"""Module for managing high-level chat interactions.

This module provides the main interface for chat interactions,
handling conversation context and model management.
"""

import asyncio
from typing import Dict, Any
from .config_manager import ModelConfig, PromptConfig
from .ollama_client import generate, list_available_models

async def generate_chat_response(
    user_input: str, 
    history_context: str, 
    model: str = "Lexi", 
    timeout: float = 15.0
) -> str:
    """Generate a chat response using local Ollama model."""
    try:
        # Load system prompt from config
        prompt_config = PromptConfig()
        system_prompt = prompt_config.get_system_prompt()
        
        print(f"Attempting to use model: {model}")  # Debug line
        
        response = await generate(
            prompt=f"Given this conversation history:\n{history_context}\n\nRespond to: {user_input}",
            model=model,
            system_prompt=system_prompt,
            verbose=True,  # Enable verbose output
            timeout=timeout
        )
        
        if not response:
            # Check if model exists in config
            model_config = ModelConfig()
            model_info = model_config.get_model_config(model)
            if not model_info:
                return f"Model '{model}' not found in configuration. Please check models.yaml."
            return "I apologize, but I'm having trouble generating a response. Please make sure Ollama is running."
        
        return response
        
    except Exception as e:
        print(f"Error in generate_chat_response: {str(e)}")
        return "An error occurred while generating the response."

async def get_available_models() -> Dict[str, Dict[str, Any]]:
    """Get list of available local models and their descriptions, including runtime check.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of model info including availability status
    """
    config_models = ModelConfig().list_models()
    ollama_models = await list_available_models()
    
    # Add runtime availability status
    models_status = {}
    for nickname, desc in config_models.items():
        # Check if either the nickname or the actual model name exists in ollama_models
        model_exists = any(m.lower().startswith(nickname.lower()) for m in ollama_models)
        models_status[nickname] = {
            'description': desc,
            'available': model_exists
        }
    
    return models_status