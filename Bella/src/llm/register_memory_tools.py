"""Register memory tools with the tools registry system.

This module integrates the memory tools from memory_conversation_adapter.py with the tools registry
for function calling. This is the canonical implementation for Bella's memory tools.

Note: This module takes precedence over memory_tools.py which provides an alternative implementation.
"""

import asyncio
from typing import Dict, List, Any, Optional, Union

from ..memory.memory_conversation_adapter import LLMMemoryTools
from .tools_registry import registry

# Initialize memory tools
memory_tools = LLMMemoryTools()

# Register remember_fact with the tools registry
@registry.register_tool(description="Save an important fact to memory for future reference")
async def remember_fact(fact: str, topic: str = None) -> Dict[str, Any]:
    """Store a fact in memory for future reference.
    
    Args:
        fact: The fact text to remember
        topic: Optional topic categorization
        
    Returns:
        Dict with operation result containing success status and message
    """
    return await memory_tools.execute_tool("remember_fact", {"fact": fact, "topic": topic})

# Register recall_memory with the tools registry
@registry.register_tool(description="Recall information from memory based on a query")
async def recall_memory(query: str) -> Dict[str, Any]:
    """Retrieve information from memory based on a query.
    
    Args:
        query: What to search for in memory
        
    Returns:
        Dict with recall results containing found information
    """
    return await memory_tools.execute_tool("recall_memory", {"query": query})

# Register save_conversation with the tools registry
@registry.register_tool(description="Save the current conversation to memory")
async def save_conversation(title: str = None, topic: str = None) -> Dict[str, Any]:
    """Save the current conversation to memory for future reference.
    
    Args:
        title: Optional title for this conversation memory
        topic: Optional topic categorization
        
    Returns:
        Dict with save operation results
    """
    return await memory_tools.execute_tool("save_conversation", {"title": title, "topic": topic})