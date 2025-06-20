"""Tools registry for function calling capabilities.

This module provides a registry for tools (functions) that can be called by the LLM.
"""

import asyncio
import inspect
import json
import logging
from typing import Dict, Any, List, Callable, Optional, Union, get_type_hints
from functools import wraps

class ToolsRegistry:
    """Registry for tools that can be called by the LLM."""
    
    def __init__(self):
        """Initialize the tools registry."""
        self._tools = {}
        self._functions = {}
        
    def register_tool(self, description: str = None, name: str = None):
        """Decorator to register a function as a tool.
        
        Args:
            description: Description of what the tool does
            name: Custom name for the tool (defaults to function name)
            
        Returns:
            Decorator function
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)
                
            # Get function metadata
            func_name = name or func.__name__
            func_doc = func.__doc__ or ""
            func_desc = description or func_doc.split("\n")[0].strip() if func_doc else func_name
            
            # Get parameters
            signature = inspect.signature(func)
            type_hints = get_type_hints(func)
            
            parameters = {
                "type": "object",
                "properties": {},
                "required": []
            }
            
            # Process parameters
            for param_name, param in signature.parameters.items():
                if param_name == "self":
                    continue
                    
                param_type = type_hints.get(param_name, str)
                param_schema = self._type_to_schema(param_type)
                
                # Get parameter description from docstring
                param_desc = ""
                if func_doc:
                    param_pattern = f"{param_name}: "
                    for line in func_doc.split("\n"):
                        line = line.strip()
                        if param_pattern in line:
                            param_desc = line.split(param_pattern)[1].strip()
                            break
                
                parameters["properties"][param_name] = {
                    "type": param_schema["type"],
                    "description": param_desc
                }
                
                # Add format, items, etc. if applicable
                if "format" in param_schema:
                    parameters["properties"][param_name]["format"] = param_schema["format"]
                if "items" in param_schema:
                    parameters["properties"][param_name]["items"] = param_schema["items"]
                    
                # Add to required if no default value
                if param.default == inspect.Parameter.empty:
                    parameters["required"].append(param_name)
            
            # Create the tool definition
            tool_def = {
                "type": "function",
                "function": {
                    "name": func_name,
                    "description": func_desc,
                    "parameters": parameters
                }
            }
            
            # Store in registry
            self._tools[func_name] = tool_def
            self._functions[func_name] = wrapper
            
            return wrapper
        return decorator
    
    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Get tool definition by name.
        
        Args:
            name: Name of the tool
            
        Returns:
            Tool definition or None if not found
        """
        return self._tools.get(name)
    
    def get_function(self, name: str) -> Optional[Callable]:
        """Get function by name.
        
        Args:
            name: Name of the function
            
        Returns:
            Function or None if not found
        """
        return self._functions.get(name)
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get all available tools.
        
        Returns:
            List of tool definitions
        """
        return list(self._tools.values())
    
    def get_all_functions(self) -> Dict[str, Callable]:
        """Get all registered functions.
        
        Returns:
            Dictionary of function names to function objects
        """
        return self._functions
    
    def _type_to_schema(self, typ) -> Dict[str, Any]:
        """Convert Python type to JSON Schema.
        
        Args:
            typ: Python type
            
        Returns:
            JSON Schema representation
        """
        if typ == str:
            return {"type": "string"}
        elif typ == int:
            return {"type": "integer"}
        elif typ == float:
            return {"type": "number"}
        elif typ == bool:
            return {"type": "boolean"}
        elif typ == list or getattr(typ, "__origin__", None) is list:
            item_type = getattr(typ, "__args__", [Any])[0]
            return {
                "type": "array",
                "items": self._type_to_schema(item_type)
            }
        elif typ == dict or getattr(typ, "__origin__", None) is dict:
            return {"type": "object"}
        else:
            # Default to string for complex types
            return {"type": "string"}

# Singleton registry instance
registry = ToolsRegistry()