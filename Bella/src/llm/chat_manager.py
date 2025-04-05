"""Module for managing high-level chat interactions.

This module provides the main interface for chat interactions,
handling conversation context and model management.
"""

import asyncio
import re
from typing import Dict, Any, Tuple
from .config_manager import ModelConfig, PromptConfig
from .ollama_client import generate, list_available_models

async def format_search_response(research_results: str) -> str:
    """Format search results in a conversational manner.
    
    Args:
        research_results (str): Raw research results
        
    Returns:
        str: Conversationally formatted response
    """
    try:
        # Extract the most relevant information
        if "Research Results" not in research_results:
            return "I searched but couldn't find any relevant information. Could you try rephrasing your question?"
            
        # Split into sections and process
        sections = research_results.split("## Depth Level")
        if len(sections) < 2:
            return "I found some information but it wasn't very detailed. Would you like me to search again?"
            
        # Focus on the first (most relevant) depth level
        primary_results = sections[1]  # Skip the header section
        
        # Extract bullet points
        bullet_points = [p.strip() for p in primary_results.split('\n') if p.strip().startswith('-')]
        
        if not bullet_points:
            return "I found some information but it wasn't very well structured. Would you like me to try a different search?"
            
        # Compose conversational response
        response = "Based on what I found, "
        for i, point in enumerate(bullet_points[:3]):  # Limit to top 3 points
            # Clean up the bullet point
            clean_point = point.replace('- ', '').split('Source:')[0].strip()
            
            if i == 0:
                response += clean_point
            elif i == len(bullet_points[:3]) - 1:
                response += f", and {clean_point}"
            else:
                response += f", {clean_point}"
                
        response += ". Would you like me to elaborate on any of these points?"
        return response
        
    except Exception as e:
        print(f"Error formatting search response: {e}")
        return "I found some information but had trouble organizing it. Would you like me to try again?"

async def generate_chat_response(
    user_input: str, 
    history_context: str, 
    model: str = None, 
    timeout: float = 20.0,
    use_mcp: bool = False
) -> str:
    """Generate a chat response using local Ollama model.
    
    Args:
        user_input (str): User's input text
        history_context (str): Previous conversation history
        model (str, optional): Model to use for generation. If None, uses default from config
        timeout (float): Maximum time to wait for response
        use_mcp (bool): Whether MCP servers are active
        
    Returns:
        str: Generated response or error message
    """
    try:
        # Get model from config if not specified
        if model is None:
            model_config = ModelConfig()
            model = model_config.get_default_model()
            
        # If not a search request, proceed with normal chat response
        prompt_config = PromptConfig()
        
        # Use MCP-aware system prompt if MCP is active
        if use_mcp:
            system_prompt = prompt_config.get_system_prompt("system_with_mcp")
        else:
            system_prompt = prompt_config.get_system_prompt()
        
        print(f"Attempting to use model: {model}")  # Debug line
        
        response = await generate(
            prompt=f"Given this conversation history:\n{history_context}\n\nRespond to: {user_input}",
            model=model,
            system_prompt=system_prompt,
            verbose=True,
            timeout=timeout
        )
        
        if not response:
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