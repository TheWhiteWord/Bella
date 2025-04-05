"""Client for interacting with Ollama models.

This module provides functions for generating text using Ollama models,
with support for the Model Context Protocol (MCP) and proper formatting
for Mistral's tool usage requirements.
"""

import os
import json
import asyncio
import re
from typing import Dict, Any, List, Optional, Union
import logging
import mcp

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='[%(asctime)s] %(levelname)-8s %(message)s',
                   datefmt='%m/%d/%y %H:%M:%S')

try:
    import ollama
except ImportError:
    logging.warning("Ollama Python package not found. Install with: pip install ollama")

async def execute_tool_call(tool_name: str, tool_args: Dict[str, Any], mcp_tools: List[Dict[str, Any]]) -> str:
    """Execute a tool call and return the result.
    
    Args:
        tool_name: The name of the tool to call
        tool_args: Arguments to pass to the tool
        mcp_tools: List of available MCP tools
    
    Returns:
        str: Result of the tool execution
    """
    # Find the matching MCP server(s) that might have this tool
    from src.utility.mcp_server_manager import MCPServerManager
    mcp_manager = MCPServerManager()
    servers = mcp_manager.get_active_servers()
    
    logging.info(f"Executing tool call: {tool_name} with args: {tool_args}")
    
    result = f"Error: Tool '{tool_name}' not found or execution failed"
    
    for server in servers:
        try:
            # Skip servers without MCP instance
            if not hasattr(server, 'mcp'):
                continue
            
            logging.info(f"Checking server {server.server_name} for tool {tool_name}")
            
            # Direct method call for BellaMemoryMCP write_note tool
            if tool_name == "write_note" and hasattr(server, '_write_markdown_file') and hasattr(server, '_index_note'):
                title = tool_args.get("title", "")
                content = tool_args.get("content", "")
                folder = tool_args.get("folder", "")
                tags = tool_args.get("tags", [])
                
                if title and content:
                    # Generate permalink as done in the original method
                    import re
                    permalink = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
                    
                    # Call write_markdown_file to create the file
                    file_path = server._write_markdown_file(
                        title=title,
                        permalink=permalink,
                        content=content,
                        folder=folder,
                        tags=tags
                    )
                    
                    # Index the note in the database
                    server._index_note(title, permalink, content, tags)
                    
                    logging.info(f"Successfully executed write_note: {title} -> {file_path}")
                    return f"Created note: {title} at {file_path}"
                else:
                    return "Error: write_note requires title and content"
                
            # Try finding the tool via the MCP registry
            for tool in server.mcp.tools:
                if tool.name == tool_name:
                    try:
                        # Execute the tool
                        logging.info(f"Found tool {tool_name} in server {server.server_name}")
                        
                        # Get the tool function
                        func = tool.function
                        
                        # Execute directly if it's a callable
                        if callable(func):
                            result = func(**tool_args)
                            
                            # Handle async functions
                            if asyncio.iscoroutine(result):
                                result = await result
                                
                            logging.info(f"Tool {tool_name} executed successfully")
                            return f"Tool {tool_name} executed successfully: {result}"
                    except Exception as e:
                        logging.error(f"Error executing tool {tool_name}: {str(e)}")
                        result = f"Error executing tool {tool_name}: {str(e)}"
                        
        except Exception as e:
            logging.error(f"Error checking server for tool {tool_name}: {str(e)}")
            
    return result

async def parse_mistral_tool_calls(content: str, mcp_tools: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
    """Parse and execute tool calls in Mistral's format.
    
    Args:
        content: Response content from Mistral
        mcp_tools: Available MCP tools
        
    Returns:
        Optional[str]: Tool execution result or None if no tool calls
    """
    if not content or not mcp_tools:
        return None
        
    # Look for Mistral's [TOOL_CALLS][...] format
    tool_call_match = re.search(r'\[TOOL_CALLS\]\[(.*?)\]', content, re.DOTALL)
    if not tool_call_match:
        return None
        
    try:
        # Extract the tool call JSON
        tool_call_json = tool_call_match.group(1)
        
        # Handle the case where the JSON might be incomplete due to truncation
        if not tool_call_json.strip().endswith("}"):
            tool_call_json += "}"
            
        # Parse the JSON
        tool_call = json.loads(tool_call_json)
        
        # Extract tool name and arguments
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("arguments", {})
        
        if not tool_name:
            return "Error: Invalid tool call format - missing tool name"
            
        # Execute the tool
        return await execute_tool_call(tool_name, tool_args, mcp_tools)
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing tool call JSON: {str(e)}")
        return f"Error parsing tool call: {str(e)}"
    except Exception as e:
        logging.error(f"Error processing tool call: {str(e)}")
        return f"Error processing tool call: {str(e)}"

async def generate(
    prompt: str,
    model: str = "mistral",
    system_prompt: str = "",
    verbose: bool = False,
    timeout: float = 30.0,
    mcp_tools: Optional[List[Dict[str, Any]]] = None  # Add support for MCP tools
) -> str:
    """Generate text using Ollama model with support for MCP tools.
    
    Args:
        prompt: The prompt to send to the model
        model: The model name to use
        system_prompt: System instructions for the model
        verbose: Whether to print debug information
        timeout: Maximum time to wait for response in seconds
        mcp_tools: Optional list of MCP tools to provide to the model
        
    Returns:
        str: Generated text response
    """
    try:
        if verbose:
            print(f"\nGenerating response with Ollama")
            print(f"Model nickname: {model}")
        
        # Check if model is a nickname and get the actual model name
        model_lower = model.lower()
        actual_model = model
        try:
            from .config_manager import ModelConfig
            model_config = ModelConfig()
            model_info = model_config.get_model_config(model)
            if model_info and "model" in model_info:
                actual_model = model_info["model"]
                model_lower = actual_model.lower()
                if verbose:
                    print(f"Using actual model name from config: {actual_model}")
        except Exception as e:
            if verbose:
                print(f"Error resolving model name: {e}")
            # Continue with the original model name
        
        # SPECIAL HANDLING FOR MISTRAL MODEL WITH MEMORY TOOLS
        # Use a more general approach for note creation with any subject
        if "mistral" in model_lower and mcp_tools and any(t.get("name") == "write_note" for t in mcp_tools if isinstance(t, dict)):
            # Check if this is a note creation request (keywords that indicate note creation intent)
            note_creation_indicators = [
                "create a note", 
                "take a note", 
                "write a note", 
                "save this as a note", 
                "remember this", 
                "store this information"
            ]
            
            if any(indicator in prompt.lower() for indicator in note_creation_indicators):
                logging.info("Detected note creation request in prompt - handling with note extraction")
                
                # Extract potential topic from the prompt
                import re
                topic_patterns = [
                    r"(?:note|notes) (?:about|on|regarding|for|titled) ['\"]?([^\"'.,;!?]+)['\"]?",  # "create a note about X"
                    r"(?:create|write|take) a ['\"]?([^\"'.,;!?]+)['\"]? note",  # "create an X note"
                    r"(?:titled|title|called|named) ['\"]?([^\"'.,;!?]+)['\"]?",  # "titled X"
                ]
                
                topic = "Untitled Note"  # Default title
                for pattern in topic_patterns:
                    match = re.search(pattern, prompt, re.IGNORECASE)
                    if match:
                        # Found a topic in the prompt
                        topic = match.group(1).strip()
                        if len(topic) > 3:  # Only use if reasonably descriptive
                            break
                
                if verbose:
                    print(f"Extracted topic for note: '{topic}'")
                    
                # Make a smaller prompt to specifically generate note content
                note_generation_prompt = f"""
                Create a well-structured note about "{topic}" based on this context:

                {prompt}

                Format the note with:
                1. A clear title
                2. An "Observations" section with relevant facts/insights prefixed with category tags like [fact], [concept], [method], etc.
                3. A "Relations" section with links to related topics using [[Topic]] format
                
                Just provide the raw note content without explanation.
                """
                
                # Generate the note content using a direct call to Ollama
                try:
                    note_response = ollama.chat(
                        model=model,
                        messages=[
                            {"role": "user", "content": note_generation_prompt}
                        ],
                        options={
                            "temperature": 0.7,
                            "top_k": 50,
                            "top_p": 0.95
                        }
                    )
                    
                    # Extract the generated note content
                    if "message" in note_response and "content" in note_response["message"]:
                        generated_content = note_response["message"]["content"]
                        
                        # Extract title and content from the generated note
                        title_match = re.search(r'^#\s+(.+?)$', generated_content, re.MULTILINE)
                        if title_match:
                            title = title_match.group(1).strip()
                        else:
                            title = topic
                            
                        # Use the generated content but make sure it has proper sections
                        content = generated_content
                        
                        # Direct execution of write_note tool using MCP manager
                        from src.utility.mcp_server_manager import MCPServerManager
                        mcp_manager = MCPServerManager()
                        servers = mcp_manager.get_active_servers()
                        
                        for server in servers:
                            if hasattr(server, '_write_markdown_file') and hasattr(server, '_index_note'):
                                try:
                                    # Extract potential tags from the content
                                    tags = []
                                    tag_matches = re.findall(r'#(\w+)', content)
                                    if tag_matches:
                                        tags = list(set(tag_matches))
                                    
                                    # Generate permalink
                                    permalink = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
                                    
                                    # Write the file
                                    file_path = server._write_markdown_file(
                                        title=title,
                                        permalink=permalink,
                                        content=content,
                                        folder="",
                                        tags=tags
                                    )
                                    
                                    # Index the note
                                    server._index_note(title, permalink, content, tags)
                                    
                                    # Create a response about the note creation
                                    return f"""I've created a note titled "{title}" based on our conversation. 

The note has been saved to your memory system, and you can reference it anytime.

Here's a quick preview of what I included:

{content[:300]}{'...' if len(content) > 300 else ''}

You can find this note in your memory system under the title "{title}"."""
                                    
                                except Exception as e:
                                    logging.error(f"Error in note creation: {e}")
                                    # Continue with normal flow if direct execution fails
                    
                except Exception as e:
                    logging.error(f"Error generating note content: {e}")
        
        # Format tools in the way Mistral expects them
        formatted_tools = None
        if mcp_tools:
            # Convert MCP tools to OpenAI-compatible format for Mistral
            formatted_tools = []
            for tool in mcp_tools:
                if "name" in tool and "description" in tool:
                    # Format parameters for OpenAI-compatible structure
                    parameters = {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                    
                    # Add parameters if present
                    if "parameters" in tool:
                        if isinstance(tool["parameters"], dict) and "properties" in tool["parameters"]:
                            parameters = tool["parameters"]  # Already in correct format
                        else:
                            for param_name, param_info in tool["parameters"].items():
                                parameters["properties"][param_name] = {
                                    "type": param_info.get("type", "string"),
                                    "description": param_info.get("description", "")
                                }
                                if param_info.get("required", False):
                                    parameters["required"].append(param_name)
                    
                    formatted_tool = {
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool["description"],
                            "parameters": parameters
                        }
                    }
                    formatted_tools.append(formatted_tool)
            
            if verbose and formatted_tools:
                print(f"Providing {len(formatted_tools)} tools to the model")
                # Print first tool for debugging
                if len(formatted_tools) > 0:
                    print(f"Example tool: {formatted_tools[0]['function']['name']}")
        
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
                },
                tools=formatted_tools  # Pass the formatted tools to Ollama
            )
        )
        
        if verbose:
            print(f"Actual model name: {response.get('model', model)}")
        
        if "message" not in response:
            return "Error: Missing response message"
            
        content = response["message"].get("content", "")
        
        # Handle standard tool calls if present in the response
        tool_calls = response["message"].get("tool_calls", None)
        if tool_calls and verbose:
            print(f"Model made {len(tool_calls)} standard tool calls")
            
        # Check for and handle Mistral-specific tool call format 
        if "[TOOL_CALLS]" in content:
            if verbose:
                print(f"Detected Mistral-format tool call in response")
                # Print first 200 chars of response for debugging
                print(f"Response preview: {content[:200]}")
                
            # Parse and execute the tool call
            if mcp_tools:
                print(f"Attempting to execute Mistral tool call with {len(mcp_tools)} available tools")
                tool_result = await parse_mistral_tool_calls(content, mcp_tools)
                
                if tool_result:
                    print(f"✅ Tool execution result: {tool_result}")
                    # For now, just append the tool result to the response
                    content += f"\n\nTool execution result: {tool_result}"
                else:
                    print("❌ Tool execution failed or returned None")
            else:
                print("❌ No MCP tools available to execute the tool call")
            
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