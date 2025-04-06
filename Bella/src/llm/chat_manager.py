"""Module for managing high-level chat interactions.

This module provides the main interface for chat interactions,
handling conversation context and model management.
"""

import asyncio
import re
import logging
import json
from typing import Dict, Any, Tuple, List, Optional

from .config_manager import ModelConfig, PromptConfig
from .ollama_client import generate, generate_with_tools, execute_tool_calls, list_available_models
from .tools_registry import registry as tools_registry


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
        
        prompt_config = PromptConfig()
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

async def generate_chat_response_with_tools(
    user_input: str,
    conversation_history: List[Dict[str, Any]],
    model: str = None,
    timeout: float = 20.0,
    max_tool_iterations: int = 3,
    verbose: bool = False
) -> Tuple[str, List[Dict[str, Any]]]:
    """Generate a chat response with potential tool usage.
    
    Args:
        user_input: User's input text
        conversation_history: Previous conversation turns as a list of message objects
        model: Model to use for generation
        timeout: Maximum time to wait for response
        max_tool_iterations: Maximum number of tool call iterations
        verbose: Whether to print debug info
        
    Returns:
        Tuple of (final response text, updated conversation history)
    """
    try:
        # Get model from config if not specified
        if model is None:
            model_config = ModelConfig()
            model = model_config.get_default_model()
        
        prompt_config = PromptConfig()
        system_prompt = prompt_config.get_system_prompt()
        
        # Add detailed instructions for tool usage with emphasis on memory tools
        tool_instructions = """
        You have access to tools to help you manage your memory and perform tasks.
        Only use tools when necessary and relevant to the user's question.
        Do not mention the tools to the user unless explicitly asked.
        When you use a tool, carefully examine the results before responding.
        
        Memory Tools:
        - Use 'semantic_memory_search' when the user asks you to recall information from your memory.
        - Use 'save_to_memory' when the user shares important information they want you to remember.
        - Use 'read_specific_memory' when you need to access a specific memory by ID.
        - Use 'save_conversation' when the conversation contains valuable information worth saving.
        - Use 'evaluate_memory_importance' to determine if information is worth remembering.
        - Use 'list_memories_by_type' when you want to see what memories are available.
        
        IMPORTANT: After using any tools, you MUST provide a conversational response to the user - never return an empty response.
        """
        system_prompt = f"{system_prompt}\n\n{tool_instructions}"
        
        # Get available tools
        available_tools = tools_registry.get_available_tools()
        available_functions = tools_registry.get_all_functions()
        
        if verbose:
            print(f"Available tools: {len(available_tools)}")
            for tool in available_tools:
                print(f" - {tool['function']['name']}: {tool['function']['description']}")
        
        # Deep copy the history to avoid modifying the original
        history = conversation_history.copy()
        
        # First attempt at response
        response = await generate_with_tools(
            prompt=user_input,
            history=history,
            tools=available_tools,
            model=model,
            system_prompt=system_prompt,
            verbose=verbose,
            timeout=timeout
        )
        
        # Check if response has tool calls
        iterations = 0
        while (response.get("message", {}).get("tool_calls") and 
               iterations < max_tool_iterations):
            
            iterations += 1
            if verbose:
                print(f"Tool iteration {iterations}/{max_tool_iterations}")
            
            # Extract tool calls and execute them
            tool_calls = response["message"]["tool_calls"]
            
            # Add the assistant's response with tool calls to history
            history.append(response["message"])
            
            # Execute the tools
            tool_results = await execute_tool_calls(tool_calls, available_functions)
            
            # Add tool results to history
            history.extend(tool_results)
            
            # Get follow-up response from model
            response = await generate_with_tools(
                prompt="",  # No new prompt needed
                history=history,
                tools=available_tools,
                model=model,
                system_prompt=system_prompt,
                verbose=verbose,
                timeout=timeout
            )
        
        # Return final response and updated history
        final_content = response.get("message", {}).get("content", "")
        
        # Debug info to help identify issues
        if verbose or not final_content:
            print(f"Response structure: {json.dumps(response, indent=2)}")
            
        # Fallback if response is empty
        if not final_content:
            # Extract tool results to provide a meaningful response
            tool_info = []
            for item in history:
                if item.get("role") == "tool" and item.get("content"):
                    try:
                        # Try to parse tool content as JSON
                        content = json.loads(item.get("content"))
                        if isinstance(content, dict) and "message" in content:
                            tool_info.append(content["message"])
                    except:
                        # Use as is if not JSON
                        tool_info.append(item.get("content"))
            
            # Create fallback response
            if tool_info:
                tool_responses = "; ".join(tool_info[-2:])  # Use last two tool responses
                final_content = f"I processed your request. {tool_responses}"
            else:
                final_content = "I processed your request but couldn't generate a proper response. Could you try again with a different question?"
        
        return final_content, history
        
    except Exception as e:
        error_message = f"Error in generate_chat_response_with_tools: {str(e)}"
        logging.error(error_message)
        return f"I encountered an error while processing your request: {str(e)}", conversation_history

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