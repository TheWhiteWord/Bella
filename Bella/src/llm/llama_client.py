"""Client for interacting with Llama.cpp server module (OpenAI compatible).

This module provides functions for generating text using the local Llama.cpp server
running in router mode.
"""
import os
import sys
import json
import re
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
import pprint

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='[%(asctime)s] %(levelname)-8s %(message)s',
                   datefmt='%m/%d/%y %H:%M:%S')

from dotenv import load_dotenv
import pathlib
from openai import AsyncOpenAI, APIConnectionError

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
_ENV_PATH = _PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)

# Llama.cpp server default URL (Router Mode)
BASE_URL = "http://localhost:8080/v1"
API_KEY = "none" # Not used by local server

async def generate(
    prompt: str,
    model: str = None,
    system_prompt: str = "",
    verbose: bool = False,
    timeout: float = 60.0,
    qwen_size: str = "XS", # Kept for signature compatibility
    thinking_mode: bool = False # Kept for signature compatibility
) -> str:
    """Generate text using Llama.cpp server.
    
    Args:
        prompt: The prompt to send to the model
        model: The model name to use (must match a loaded model in router)
        system_prompt: System instructions for the model
        verbose: Whether to print debug information
        timeout: Maximum time to wait for response in seconds
        
    Returns:
        str: Generated text response
    """
    try:
        if not model:
            from .config_manager import ModelConfig
            model = ModelConfig().get_default_model()
            
        if verbose:
            print(f"\nGenerating response with Llama.cpp")
            print(f"Model: {model}")

        # Note: thinking_mode logic (adding /think tags) is REMOVED 
        # because we now use distinct models for thinking vs normal.
        
        client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7, # Default, can be tuned or passed
                    top_p=0.9,
                    max_tokens=None, # Let model decide
                ),
                timeout=timeout
            )
        except APIConnectionError:
            return "Error: Llama.cpp server is not running. Please run ./start_bella_llama.sh"
        
        content = response.choices[0].message.content

        if verbose:
            print(f"Response received from {model}")

        # Strip <think> tags if present in output (common in thinking models)
        # We might want to KEEP them for internal logs but strip for user?
        # ollama_client stripped them, so we will too.
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
        return content

    except asyncio.TimeoutError:
        logging.error(f"Request timed out after {timeout} seconds")
        return "Error: Request timed out."
    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        return f"Error generating response: {str(e)}"

async def generate_with_tools(
    prompt: str,
    history: List[Dict[str, Any]] = None,
    tools: List[Dict[str, Any]] = None,
    model: str = None,
    system_prompt: str = "",
    verbose: bool = False,
    timeout: float = 60.0,
    qwen_size: str = "XS",
    thinking_mode: bool = False
) -> Dict[str, Any]:
    """Generate text using Llama.cpp model with tool calling support.
    
    Note: Llama.cpp's tool calling support via OpenAI API is partial/experimental.
    We rely on the standard OpenAI 'tools' param.
    """
    try:
        if not model:
            from .config_manager import ModelConfig
            model = ModelConfig().get_default_model()
            
        if verbose:
            print(f"\nGenerating response with Llama.cpp (Tools)")
            print(f"Model: {model}")
            print(f"Tools count: {len(tools) if tools else 0}")

        client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        if prompt:
            messages.append({"role": "user", "content": prompt})

        # Format tools for OpenAI API
        # The input tools list is likely already in OpenAI format or similar
        openai_tools = None
        if tools:
            openai_tools = tools 

        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=openai_tools if openai_tools else None,
                    tool_choice="auto" if openai_tools else None,
                    temperature=0.7,
                ),
                timeout=timeout
            )
        except APIConnectionError:
            return {"message": {"content": "Error: Llama.cpp server not reachable."}}

        message = response.choices[0].message
        
        serializable_response = {}
        serializable_message = {"role": "assistant"}
        
        if message.content:
            content = message.content
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            serializable_message["content"] = content
        else:
            serializable_message["content"] = ""
            
        if message.tool_calls:
            tool_calls = []
            for tool_call in message.tool_calls:
                serialized_tool = {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
                tool_calls.append(serialized_tool)
            serializable_message["tool_calls"] = tool_calls
            
        serializable_response["message"] = serializable_message
        serializable_response["model"] = model
        
        return serializable_response

    except asyncio.TimeoutError:
         return {"message": {"content": "I'm sorry, I timed out."}}
    except Exception as e:
        logging.error(f"Error in generate_with_tools: {str(e)}")
        return {"message": {"content": f"Error: {str(e)}"}}

# Re-export execute_tool_calls from ollama_client or generic location?
# Actually it was defined IN ollama_client. We should copy it here or move it to a shared utility.
# For now, I will include it here to make this a drop-in replacement.

async def execute_tool_calls(
    tool_calls: List[Dict[str, Any]], 
    available_functions: Dict[str, callable]
) -> List[Dict[str, Any]]:
    """Execute tool calls and return results."""
    if not tool_calls:
        return []
        
    tool_results = []
    
    for tool in tool_calls:
        function_name = tool["function"]["name"]
        function_args = tool["function"]["arguments"]

        args_dict = {}
        try:
            if isinstance(function_args, str):
                args_dict = json.loads(function_args)
            elif isinstance(function_args, dict):
                args_dict = function_args
        except json.JSONDecodeError:
            tool_results.append({
                "role": "tool",
                "name": function_name,
                "content": "Error: Invalid JSON args"
            })
            continue

        function = available_functions.get(function_name)
        if function:
            try:
                logging.info(f"Calling function: {function_name}")
                result = function(**args_dict)
                if asyncio.iscoroutine(result):
                    result = await result
                
                if isinstance(result, (dict, list)):
                    result_str = json.dumps(result, ensure_ascii=False)
                else:
                    result_str = str(result)
                    
                tool_results.append({
                    "role": "tool", 
                    "tool_call_id": tool.get("id"), # Important for OpenAI API correlation
                    "name": function_name,
                    "content": result_str
                })
            except Exception as e:
                tool_results.append({
                    "role": "tool",
                     "tool_call_id": tool.get("id"),
                    "name": function_name,
                    "content": f"Error: {str(e)}"
                })
        else:
             tool_results.append({
                "role": "tool",
                 "tool_call_id": tool.get("id"),
                "name": function_name,
                "content": "Error: Function not found"
            })
    return tool_results

async def list_available_models() -> List[str]:
    """List loaded models from Llama.cpp server."""
    try:
        client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
        models = await client.models.list()
        return [m.id for m in models.data]
    except Exception as e:
        logging.error(f"Error listing models: {str(e)}")
        return []
