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

async def generate_with_tools(
    prompt: str,
    history: List[Dict[str, Any]] = None,
    tools: List[Dict[str, Any]] = None,
    model: str = "mistral",
    system_prompt: str = "",
    verbose: bool = False,
    timeout: float = 30.0
) -> Dict[str, Any]:
    """Generate text using Ollama model with tool calling support.
    
    Args:
        prompt: The prompt to send to the model
        history: Previous conversation history
        tools: List of tool definitions
        model: The model name to use
        system_prompt: System instructions for the model
        verbose: Whether to print debug information
        timeout: Maximum time to wait for response in seconds
        
    Returns:
        Dict: Response with message and any tool calls
    """
    try:
        if verbose:
            print(f"\nGenerating response with Ollama (with tools)")
            print(f"Model nickname: {model}")
            print(f"Number of tools available: {len(tools) if tools else 0}")
        
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
            
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        # Add history if provided
        if history:
            messages.extend(history)
            
        # Add current user message
        if prompt:  # Only add if not empty
            messages.append({"role": "user", "content": prompt})
        
        # Create an async Ollama client
        client = ollama.AsyncClient()
        
        # Make call with tools
        try:
            response = await asyncio.wait_for(
                client.chat(
                    model=actual_model,
                    messages=messages,
                    tools=tools,
                    options={
                        "temperature": 0.7,
                        "top_k": 50,
                        "top_p": 0.95
                    }
                ),
                timeout=timeout
            )
            
            # Convert the response to a serializable dict
            serializable_response = {}
            
            # Extract message content
            if hasattr(response, 'message'):
                message = response.message
                serializable_message = {"role": "assistant"}
                
                # Add content if present
                if hasattr(message, 'content'):
                    serializable_message["content"] = message.content
                else:
                    serializable_message["content"] = ""
                
                # Add tool calls if present
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    tool_calls = []
                    for tool_call in message.tool_calls:
                        serialized_tool = {
                            "id": tool_call.id if hasattr(tool_call, 'id') else f"tool_{len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name if hasattr(tool_call.function, 'name') else "",
                                "arguments": tool_call.function.arguments if hasattr(tool_call.function, 'arguments') else "{}"
                            }
                        }
                        tool_calls.append(serialized_tool)
                    
                    serializable_message["tool_calls"] = tool_calls
                
                serializable_response["message"] = serializable_message
            
            # Extract model info
            if hasattr(response, 'model'):
                serializable_response["model"] = response.model
            else:
                serializable_response["model"] = actual_model
            
            if verbose:
                print(f"Actual model name: {serializable_response.get('model', actual_model)}")
                if serializable_response.get("message", {}).get("tool_calls"):
                    print(f"Tool calls detected: {len(serializable_response['message']['tool_calls'])}")
            
            return serializable_response
        except TypeError as e:
            # For backward compatibility with older ollama versions that return dicts directly
            if isinstance(response, dict):
                if verbose:
                    print("Response is already a dict, using directly")
                return response
            else:
                raise e
        
    except asyncio.TimeoutError:
        logging.error(f"Request timed out after {timeout} seconds")
        return {"message": {"content": f"I'm sorry, but I took too long to respond. Could you try again?"}}
        
    except Exception as e:
        logging.error(f"Error generating response with tools: {str(e)}")
        return {"message": {"content": f"Error generating response: {str(e)}"}}

async def execute_tool_calls(
    tool_calls: List[Dict[str, Any]], 
    available_functions: Dict[str, callable]
) -> List[Dict[str, Any]]:
    """Execute tool calls and return results.
    
    Args:
        tool_calls: List of tool calls from model response
        available_functions: Dict of available function implementations
        
    Returns:
        List: Tool results to add to conversation
    """
    if not tool_calls:
        return []
        
    tool_results = []
    
    for tool in tool_calls:
        function_name = tool["function"]["name"]
        function_args = tool["function"]["arguments"]
        
        # Try to parse arguments as JSON if it's a string,
        # or use directly if it's already a dictionary
        args_dict = {}
        try:
            if isinstance(function_args, str):
                args_dict = json.loads(function_args)
            elif isinstance(function_args, dict):
                args_dict = function_args
            else:
                logging.error(f"Unexpected arguments type: {type(function_args)}")
                tool_results.append({
                    "role": "tool",
                    "name": function_name,
                    "content": f"Error: Invalid function arguments type: {type(function_args)}"
                })
                continue
        except json.JSONDecodeError:
            logging.error(f"Failed to parse function arguments: {function_args}")
            tool_results.append({
                "role": "tool",
                "name": function_name,
                "content": "Error: Invalid function arguments format"
            })
            continue
            
        # Check if function exists
        if function := available_functions.get(function_name):
            try:
                # Call the function with arguments
                logging.info(f"Calling function: {function_name} with args: {args_dict}")
                result = function(**args_dict)
                
                # Handle async functions
                if asyncio.iscoroutine(result):
                    result = await result
                
                # Convert result to string if it's a dict or other complex type
                if isinstance(result, (dict, list)):
                    # Log the raw result for debugging
                    logging.debug(f"Raw tool result: {result}")
                    result_str = json.dumps(result, ensure_ascii=False)
                else:
                    result_str = str(result)
                
                # Log the formatted result
                logging.debug(f"Formatted tool result: {result_str}")
                
                tool_results.append({
                    "role": "tool",
                    "name": function_name,
                    "content": result_str
                })
                
                # Log tool result added to conversation for debugging
                logging.info(f"Tool {function_name} completed successfully with result type: {type(result_str)}")
                
            except Exception as e:
                error_message = f"Error executing function {function_name}: {str(e)}"
                logging.error(error_message)
                tool_results.append({
                    "role": "tool",
                    "name": function_name,
                    "content": f"Error: {str(e)}"
                })
        else:
            logging.warning(f"Function not found: {function_name}")
            tool_results.append({
                "role": "tool",
                "name": function_name,
                "content": "Error: Function not found"
            })
            
    return tool_results

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