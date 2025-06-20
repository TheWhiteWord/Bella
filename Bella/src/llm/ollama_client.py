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
import pprint

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='[%(asctime)s] %(levelname)-8s %(message)s',
                   datefmt='%m/%d/%y %H:%M:%S')


# Explicitly load .env from the project root (Bella/.env)
from dotenv import load_dotenv
import pathlib
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
_ENV_PATH = _PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)

try:
    import ollama
except ImportError:
    logging.warning("Ollama Python package not found. Install with: pip install ollama")

def _get_qwen_model(size: str = "XS") -> str:
    """Get Qwen model name from .env by size (TINY, XS, S, M, L). Defaults to XS (2B)."""
    env_map = {
        "XXS": "QWEN_XXS",
        "XS": "QWEN_XS",
        "S": "QWEN_S",
        "M": "QWEN_M",
        "L": "QWEN_L",
        "LEXI": "LEXI"
    }
    env_var = env_map.get(size.upper(), "QWEN_XS")
    # Always prefer QWEN_L for 'L', fallback to qwen3:14B
    if size.upper() == "L":
        return os.getenv("QWEN_L", "qwen3:14B")
    return os.getenv(env_var, "qwen3:2B")

async def generate(
    prompt: str,
    model: str = None,
    system_prompt: str = "",
    verbose: bool = False,
    timeout: float = 30.0,
    qwen_size: str = "XS",
    thinking_mode: bool = False
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

        # Always use unified model getter unless explicitly overridden
        if not model:
            from .config_manager import ModelConfig
            model = ModelConfig().get_default_model()
        if verbose:
            print(f"\nGenerating response with Ollama (Qwen)")
            print(f"Model: {model}")

        # Compose system prompt with /think or /no_think prefix
        sys_prompt = system_prompt.strip() if system_prompt else ""
        if thinking_mode:
            sys_prompt = f"/think\n{sys_prompt}"
        else:
            sys_prompt = f"/no_think\n{sys_prompt}"

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": 0.6 if thinking_mode else 0.7,
                    "top_p": 0.95 if thinking_mode else 0.8,
                    "top_k": 20,
                    "min_p": 0
                }
            )
        )

        if verbose:
            print(f"Actual model name: {response.get('model', model)}")

        if "message" not in response:
            return "Error: Missing response message"

        content = response["message"].get("content", "")

        # Remove <think>...</think> tags if present (Qwen models output them by default)
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

        return content

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
    timeout: float = 30.0,
    qwen_size: str = "XS",
    thinking_mode: bool = False
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
        # Always use unified model getter unless explicitly overridden
        if not model:
            from .config_manager import ModelConfig
            model = ModelConfig().get_default_model()
        if verbose:
            print(f"\nGenerating response with Ollama (Qwen, with tools)")
            print(f"Model: {model}")
            print(f"Number of tools available: {len(tools) if tools else 0}")

        # Compose system prompt with /think or /no_think prefix
        sys_prompt = system_prompt.strip() if system_prompt else ""
        if thinking_mode:
            sys_prompt = f"/think\n{sys_prompt}"
        else:
            sys_prompt = f"/no_think\n{sys_prompt}"

        # Prepare messages
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        if history:
            messages.extend(history)
        if prompt:
            messages.append({"role": "user", "content": prompt})

        client = ollama.AsyncClient()
        try:
            response = await asyncio.wait_for(
                client.chat(
                    model=model,
                    messages=messages,
                    tools=tools,
                    options={
                        "temperature": 0.6 if thinking_mode else 0.7,
                        "top_p": 0.95 if thinking_mode else 0.8,
                        "top_k": 20,
                        "min_p": 0
                    }
                ),
                timeout=timeout
            )
            serializable_response = {}
            if hasattr(response, 'message'):
                message = response.message
                serializable_message = {"role": "assistant"}
                if hasattr(message, 'content'):
                    # Remove <think>...</think> tags if present
                    content = re.sub(r'<think>.*?</think>', '', message.content, flags=re.DOTALL).strip()
                    serializable_message["content"] = content
                else:
                    serializable_message["content"] = ""
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
            if hasattr(response, 'model'):
                serializable_response["model"] = response.model
            else:
                serializable_response["model"] = model
            if verbose:
                print(f"Actual model name: {serializable_response.get('model', model)}")
                if serializable_response.get("message", {}).get("tool_calls"):
                    print(f"Tool calls detected: {len(serializable_response['message']['tool_calls'])}")
            return serializable_response
        except TypeError as e:
            if isinstance(response, dict):
                if verbose:
                    print("Response is already a dict, using directly")
                # Remove <think>...</think> tags if present
                if "message" in response and "content" in response["message"]:
                    response["message"]["content"] = re.sub(r'<think>.*?</think>', '', response["message"]["content"], flags=re.DOTALL).strip()
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
        function = available_functions.get(function_name)
        if function:
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

                # Remove <think>...</think> tags if present in tool result
                if isinstance(result_str, str):
                    result_str = re.sub(r'<think>.*?</think>', '', result_str, flags=re.DOTALL).strip()

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