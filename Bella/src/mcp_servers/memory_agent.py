"""Basic Memory MCP integration for Bella using PraisonAI agents.

This module provides integration with the basic-memory MCP server
through PraisonAI agents, allowing Bella to create and access persistent
notes stored as Markdown files.
"""

import asyncio
import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, cast
import traceback

# Set environment variables before importing any Praison modules
os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"

# Load environment variables from .env file
from dotenv import load_dotenv
dotenv_path = Path(__file__).parents[3] / '.env'
load_dotenv(dotenv_path)

from praisonaiagents import Agent, MCP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# Log OPENAI_BASE_URL to verify it's loaded correctly
openai_url = os.environ.get("OPENAI_BASE_URL")
if openai_url:
    logger.info(f"Using OpenAI compatibility with URL: {openai_url}")
else:
    logger.warning("OPENAI_BASE_URL not set. Ensure .env file is loaded correctly.")


class MemoryAgent:
    """Agent for interacting with Basic Memory MCP functionality.
    
    This class provides a simple interface to the basic-memory MCP server
    through PraisonAI agents, allowing for creation and access of notes.
    
    Attributes:
        model (str): The model name to use with OpenAI compatibility
        memory_dir (str): Directory to store memory files
        verbose (bool): Whether to output debug information
        agent (Agent): PraisonAI agent with MCP tools
    """
    
    def __init__(
        self, 
        model: str = "Llama3B-uncensored:latest",
        memory_dir: str = os.path.expanduser("~/basic-memory"),
        verbose: bool = False,
        mcp_url: str = "https://smithery.ai/server/@basicmachines-co/basic-memory"
    ):
        """Initialize the memory agent.
        
        Args:
            model: The model name to use (without ollama/ prefix when using OpenAI compatibility)
            memory_dir: Directory to store memory files
            verbose: Whether to print debug information
            mcp_url: URL for the basic-memory MCP server (default: Smithery hosted)
        """
        self.model = model
        self.memory_dir = Path(memory_dir).expanduser()
        self.verbose = verbose
        self.mcp_url = mcp_url
        
        # Ensure memory directory exists
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # Log configuration if verbose
        if verbose:
            logger.info(f"Initializing memory agent with model: {model}")
            logger.info(f"Using memory directory: {self.memory_dir}")
            logger.info(f"Using MCP URL: {self.mcp_url}")
        
        # Create agent with OpenAI compatibility for Ollama
        try:
            self.agent = Agent(
                instructions="""You are an expert memory assistant that helps manage knowledge through structured notes.

                Your task is to manage notes and information using the memory tools available to you.
                
                ALWAYS use these memory tools for specific tasks:
                
                1. write_note(title, content, tags): Create or update a note with these parameters
                   - title: The title of the note (string)
                   - content: The content in markdown format (string)
                   - tags: A list of relevant tags for categorization (list of strings)
                
                2. search_notes(query): Search for notes with a specific query
                   - query: The search query to find relevant notes (string)
                
                3. read_note(title): Read a specific note by title or permalink
                   - title: The title or permalink of the note to read (string)
                
                4. build_context(topic, depth): Navigate the knowledge graph around a topic
                   - topic: The topic to explore (string)
                   - depth: How deeply to explore related notes (integer)
                
                Always respond with clear function calls rather than just describing what you would do.
                """,
                llm=model,  # When using OpenAI compatibility, don't use the ollama/ prefix
                tools=MCP(self.mcp_url)
            )
            if verbose:
                logger.info(f"Memory agent initialized successfully with model: {model}")
                
                # Log available tools if possible
                if hasattr(self.agent, "tools"):
                    tool_names = [getattr(tool, "name", str(tool)) for tool in self.agent.tools]
                    logger.info(f"Available tools: {', '.join(tool_names)}")
                
        except Exception as e:
            error_msg = f"Error initializing memory agent: {str(e)}"
            logger.error(error_msg)
            if verbose:
                logger.error(traceback.format_exc())
            raise RuntimeError(error_msg) from e
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query with the memory agent.
        
        Args:
            query: The query to process
            
        Returns:
            Dict[str, Any]: The agent's response with extracted tool calls
        """
        if self.verbose:
            logger.info(f"Processing query: {query}")
        
        try:
            # Get the event loop
            loop = asyncio.get_running_loop()
            
            # Use start() without any extra parameters
            response = await loop.run_in_executor(None, lambda: self.agent.start(query))
            
            # Normalize response to dictionary format
            if isinstance(response, str):
                normalized_response = {"response": response, "success": True}
            else:
                normalized_response = response if isinstance(response, dict) else {"response": str(response), "success": True}
            
            # Extract any tool calls from the response text
            tool_calls = self._extract_tool_calls(normalized_response.get("response", ""))
            normalized_response["tool_calls"] = tool_calls
            
            # Add parsed tool parameters if available
            if tool_calls:
                normalized_response["parsed_tools"] = [
                    self._parse_tool_arguments(tool["name"], tool["arguments"])
                    for tool in tool_calls
                ]
            
            if self.verbose and tool_calls:
                logger.info(f"Extracted {len(tool_calls)} tool calls from response")
                
            return normalized_response
        
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            if self.verbose:
                logger.error(traceback.format_exc())
            return {
                "error": error_msg, 
                "success": False, 
                "response": str(e),
                "tool_calls": []
            }
    
    @lru_cache(maxsize=10)
    def _get_tool_patterns(self) -> List[Tuple[str, re.Pattern]]:
        """Get cached regex patterns for extracting tool calls.
        
        Returns:
            List[Tuple[str, re.Pattern]]: List of (pattern_name, compiled_pattern) tuples
        """
        return [
            # Standard function call pattern
            ("function_call", re.compile(r"(\w+)\s*\(([^)]+)\)")),
            
            # write_note pattern with named parameters
            ("write_note", re.compile(
                r"write_note\s*\(\s*"
                r"(?:title\s*=\s*['\"]([^'\"]+)['\"]|['\"]([^'\"]+)['\"])\s*,\s*"
                r"(?:content\s*=\s*['\"]([^'\"]+)['\"]|['\"]([^'\"]+)['\"])\s*"
                r"(?:,\s*(?:tags\s*=\s*(\[[^\]]+\])|(\[[^\]]+\])))?",
                re.DOTALL
            )),
            
            # YAML-like format that LLMs sometimes generate
            ("yaml_format", re.compile(
                r"title:\s*([^\n]+)\n\s*content:\s*\|?\n([\s\S]*?)(?:\n\s*tags:\n([\s\S]*?))?(?:\n\s*(?:Note created|$))",
                re.DOTALL
            ))
        ]
    
    def _extract_tool_calls(self, text: str) -> List[Dict[str, str]]:
        """Extract tool calls from the response text.
        
        Args:
            text: The response text to parse
            
        Returns:
            List[Dict[str, str]]: List of extracted tool calls
        """
        if not text:
            return []
            
        tool_calls = []
        patterns = self._get_tool_patterns()
        
        # Try to extract standard function calls first
        for pattern_name, pattern in patterns:
            matches = pattern.findall(text)
            
            if not matches:
                continue
                
            for match in matches:
                if pattern_name == "function_call" and len(match) >= 2:
                    function_name, args = match
                    
                    # Only process known memory tool functions
                    if function_name in ["write_note", "read_note", "search_notes", "build_context"]:
                        tool_calls.append({
                            "name": function_name,
                            "arguments": args.strip(),
                            "raw": f"{function_name}({args})",
                            "pattern": pattern_name
                        })
                        
                elif pattern_name == "write_note" and len(match) >= 3:
                    # Extract parameters from the write_note specific pattern
                    title = match[0] or match[1]
                    content = match[2] or match[3]
                    tags = match[4] or match[5] or "[]"
                    
                    if title and content:
                        args = f"title='{title}', content='{content}', tags={tags}"
                        tool_calls.append({
                            "name": "write_note",
                            "arguments": args,
                            "raw": f"write_note({args})",
                            "pattern": pattern_name
                        })
                        
                elif pattern_name == "yaml_format" and len(match) >= 2:
                    # Handle YAML-like format
                    title = match[0].strip()
                    content = match[1].strip()
                    tags_raw = match[2] if len(match) > 2 and match[2] else ""
                    
                    # Extract tags from YAML format (- tag1\n- tag2)
                    tags = []
                    if tags_raw:
                        for line in tags_raw.split('\n'):
                            tag_match = re.match(r'\s*-\s*(\w+)', line)
                            if tag_match:
                                tags.append(tag_match.group(1))
                    
                    tags_str = str(tags) if tags else "[]"
                    args = f"title='{title}', content='{content}', tags={tags_str}"
                    
                    tool_calls.append({
                        "name": "write_note",
                        "arguments": args,
                        "raw": f"write_note({args})",
                        "pattern": pattern_name
                    })
        
        return tool_calls
    
    def _parse_tool_arguments(self, tool_name: str, arguments_str: str) -> Dict[str, Any]:
        """Parse tool arguments into a structured dictionary.
        
        Args:
            tool_name: Name of the tool
            arguments_str: String containing tool arguments
            
        Returns:
            Dict[str, Any]: Parsed arguments
        """
        result = {"tool": tool_name}
        
        # Handle different argument formats
        if "=" in arguments_str:
            # Named parameter format: param1='value1', param2='value2'
            param_pattern = re.compile(r"(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\"|(\[[^\]]*\])|(\d+))")
            matches = param_pattern.findall(arguments_str)
            
            for match in matches:
                param_name = match[0]
                # Find the first non-empty value among the capture groups
                param_value = next((val for val in match[1:] if val), "")
                
                # Convert list strings to actual lists
                if param_value.startswith('[') and param_value.endswith(']'):
                    try:
                        # Try to parse as JSON
                        param_value = json.loads(param_value.replace("'", '"'))
                    except json.JSONDecodeError:
                        # Fall back to regex-based parsing for lists with single quotes
                        items = re.findall(r"'([^']*)'", param_value)
                        param_value = items
                
                result[param_name] = param_value
        else:
            # Positional parameter format
            if tool_name == "write_note":
                # Try to parse positional arguments for write_note
                parts = re.findall(r"'([^']*)'|\"([^\"]*)\"|\[([^\]]*)\]", arguments_str)
                
                if len(parts) >= 2:
                    result["title"] = parts[0][0] or parts[0][1]
                    result["content"] = parts[1][0] or parts[1][1]
                    
                    if len(parts) >= 3:
                        tags_str = parts[2][0] or parts[2][1] or parts[2][2]
                        try:
                            result["tags"] = json.loads(tags_str.replace("'", '"'))
                        except json.JSONDecodeError:
                            result["tags"] = [tag.strip() for tag in tags_str.split(',')]
            
            elif tool_name == "read_note" or tool_name == "search_notes":
                # Simple case for tools with a single parameter
                parts = re.findall(r"'([^']*)'|\"([^\"]*)\"", arguments_str)
                if parts:
                    param_name = "title" if tool_name == "read_note" else "query"
                    result[param_name] = parts[0][0] or parts[0][1]
        
        return result


async def main():
    """Test the memory agent functionality."""
    try:
        memory = MemoryAgent(verbose=True)
        
        # Test creating a note
        query = (
            "Create a note titled 'Test Note' with the following content:\n\n"
            "This is a test of the basic memory integration with Bella assistant.\n\n"
            "Add these tags: test, bella, memory"
        )
        
        print(f"\nSending query: {query}")
        result = await memory.process_query(query)
        
        print("\nResult:")
        print(result.get("response", "No response"))
        
        if "tool_calls" in result and result["tool_calls"]:
            print("\nExtracted tool calls:")
            for i, tool in enumerate(result["tool_calls"]):
                print(f"\nTool {i+1}:")
                print(f"  Name: {tool['name']}")
                print(f"  Arguments: {tool['arguments']}")
                print(f"  Raw call: {tool['raw']}")
                print(f"  Pattern used: {tool.get('pattern', 'unknown')}")
                
                if "parsed_tools" in result and i < len(result["parsed_tools"]):
                    print("\nParsed parameters:")
                    for k, v in result["parsed_tools"][i].items():
                        print(f"  {k}: {v}")
        else:
            print("\nNo tool calls were extracted from the response")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())