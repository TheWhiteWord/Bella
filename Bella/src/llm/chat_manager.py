"""Module for managing high-level chat interactions.

This module provides the main interface for chat interactions,
handling conversation context and model management.
"""

import asyncio
import re
import logging
from typing import Dict, Any, Tuple, List, Optional

from .config_manager import ModelConfig, PromptConfig
from .ollama_client import generate, list_available_models
from ..memory import MemoryManager

# Initialize memory manager (will be lazy-loaded when needed)
_memory_manager = None

def get_memory_manager() -> MemoryManager:
    """Get or initialize the memory manager.
    
    Returns:
        MemoryManager: The memory manager instance
    """
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(enable_memory=True)
    return _memory_manager

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
    timeout: float = 20.0
) -> str:
    """Generate a chat response using local Ollama model.
    
    Args:
        user_input (str): User's input text
        history_context (str): Previous conversation history
        model (str, optional): Model to use for generation. If None, uses default from config
        timeout (float): Maximum time to wait for response
        
    Returns:
        str: Generated response or error message
    """
    try:
        # Get model from config if not specified
        if model is None:
            model_config = ModelConfig()
            model = model_config.get_default_model()
            
        # Get memory manager
        memory_manager = get_memory_manager()
        
        # Record user input in memory
        memory_manager.record_interaction("User", user_input)
            
        # Get relevant context from memory
        memory_context = await memory_manager.get_context(user_input)
        
        # Combine conversation history with memory context
        enhanced_context = f"{history_context}\n\n{memory_context}" if memory_context else history_context
        
        prompt_config = PromptConfig()
        system_prompt = prompt_config.get_system_prompt()
        
        # Enhance system prompt with memory capabilities
        if memory_manager.enable_memory:
            system_prompt += "\nYou have access to memory and can remember previous interactions. Use this to provide more personalized and contextually relevant responses."
        
        print(f"Attempting to use model: {model}")  # Debug line
        
        response = await generate(
            prompt=f"Given this conversation history and memory context:\n{enhanced_context}\n\nRespond to: {user_input}",
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
        
        # Record assistant response in memory
        memory_manager.record_interaction("Assistant", response)
        
        # Check if we should store important information in long-term memory
        if _is_important_information(user_input, response):
            await memory_manager.store_important(
                f"User asked: {user_input}\nAssistant responded: {response}",
                metadata={"importance": "high", "topic": _extract_topic(user_input)}
            )
        
        return response
        
    except Exception as e:
        print(f"Error in generate_chat_response: {str(e)}")
        return "An error occurred while generating the response."

def _is_important_information(user_input: str, response: str) -> bool:
    """Determine if the current exchange contains important information to remember.
    
    This is a simple heuristic that can be enhanced with more sophisticated logic.
    
    Args:
        user_input: The user's input
        response: The assistant's response
        
    Returns:
        bool: Whether the exchange contains important information
    """
    # Keywords that might indicate important information
    important_keywords = [
        'remember', 'don\'t forget', 'important', 
        'my name is', 'I am', 'I like', 'I prefer',
        'favorite', 'address', 'phone', 'email'
    ]
    
    # Check if any important keywords are in the user input or response
    for keyword in important_keywords:
        if keyword.lower() in user_input.lower() or keyword.lower() in response.lower():
            return True
            
    # Is it a long, detailed response? Might be worth remembering
    if len(response) > 500:
        return True
        
    return False

def _extract_topic(text: str) -> str:
    """Extract the main topic from text.
    
    Args:
        text: The text to extract topic from
        
    Returns:
        str: The extracted topic or 'general'
    """
    # Very simple topic extraction - this could be enhanced with NLP
    words = text.lower().split()
    stopwords = {'a', 'an', 'the', 'to', 'and', 'or', 'but', 'in', 'on', 'at', 'is', 'are', 'was', 'were'}
    
    # Filter out stopwords and get the most frequent words
    filtered = [word for word in words if word not in stopwords and len(word) > 3]
    
    if filtered:
        # Just return the first non-stopword as the topic
        return filtered[0]
    return 'general'

async def get_available_models() -> Dict[str, Dict[str, Any]]:
    """Get list of available local models and their descriptions, including runtime check.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of model info including availability status
    """
    config_models = ModelConfig().list_models()
    try:
        ollama_models = await list_available_models()
        logging.info(f"Ollama models found: {ollama_models}")
    except Exception as e:
        logging.error(f"Failed to get Ollama models: {e}")
        ollama_models = []
    
    # Add runtime availability status
    models_status = {}
    for nickname, desc in config_models.items():
        # Get the actual model name from config if available
        actual_model_name = desc.get("name", nickname) if isinstance(desc, dict) else nickname
        
        # Check for model availability with more flexible matching
        model_exists = False
        for ollama_model in ollama_models:
            # Compare with nickname
            if ollama_model.lower() == nickname.lower():
                model_exists = True
                break
                
            # Compare with actual model name
            if isinstance(actual_model_name, str):
                # Clean up model name for comparison (remove tags like :latest)
                clean_actual = actual_model_name.split(':')[0].lower()
                clean_ollama = ollama_model.split(':')[0].lower()
                
                if clean_ollama == clean_actual:
                    model_exists = True
                    break
        
        models_status[nickname] = {
            'description': desc.get("description", str(desc)) if isinstance(desc, dict) else str(desc),
            'available': model_exists
        }
    
    return models_status