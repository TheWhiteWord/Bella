"""API layer for memory management in voice assistant.

Provides simplified interfaces to memory functions for use in voice interactions.
"""

import asyncio
import os
import re
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from .memory_manager import MemoryManager

# Singleton memory manager instance
_memory_manager = None

def get_memory_manager(memory_dir: str = None) -> MemoryManager:
    """Get or create singleton memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(memory_dir)
    return _memory_manager

async def write_note(
    title: str, 
    content: str, 
    folder: str = "general", 
    tags: List[str] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """Create a new memory note.
    
    Args:
        title: Title of the note
        content: Markdown content 
        folder: Folder to store in (default: "general")
        tags: List of tags for categorization
        verbose: Whether to return parsing details
        
    Returns:
        Dict with operation results
    """
    manager = get_memory_manager()
    return await manager.create_memory(
        title=title,
        content=content,
        folder=folder,
        tags=tags,
        verbose=verbose
    )

async def read_note(identifier: str) -> Optional[str]:
    """Read a memory note by title, path, or URL.
    
    Args:
        identifier: Title, path, or memory:// URL
        
    Returns:
        String content of the note, or None if not found
    """
    manager = get_memory_manager()
    result = await manager.read_memory(identifier)
    
    if result:
        return result.get('content')
    return None

async def search_notes(
    query: str,
    page: int = 1,
    page_size: int = 10
) -> Dict[str, Any]:
    """Search through memory notes.
    
    Args:
        query: Text to search for
        page: Page number for pagination
        page_size: Results per page
        
    Returns:
        Dict with search results and pagination info
    """
    manager = get_memory_manager()
    return await manager.search_memories(query, page, page_size)

async def build_context(
    url: str,
    depth: int = 2,
    timeframe: str = None
) -> Dict[str, Any]:
    """Build context from memory graph.
    
    Args:
        url: Starting point (title, path or memory:// URL)
        depth: How many hops to follow in the graph
        timeframe: Time window for filtering (e.g. "1 month")
        
    Returns:
        Dict with context information
    """
    manager = get_memory_manager()
    return await manager.build_context(url, depth, timeframe)

async def recent_activity(
    type: str = "all",
    depth: int = 1,
    timeframe: str = "1 week"
) -> Dict[str, Any]:
    """Get recent memory changes.
    
    Args:
        type: Type of memories to include
        depth: How many related items to include
        timeframe: Time window (e.g. "1 week")
        
    Returns:
        Dict with recent activity information
    """
    manager = get_memory_manager()
    return await manager.recent_activity(type, depth, timeframe)

async def update_note(
    identifier: str,
    new_content: str,
    new_title: str = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """Update an existing memory note.
    
    Args:
        identifier: Title or path of note to update
        new_content: New content for the note
        new_title: New title if renaming (optional)
        verbose: Whether to return detailed parsing results
        
    Returns:
        Dict with operation results
    """
    manager = get_memory_manager()
    return await manager.update_memory(identifier, new_content, new_title, verbose)

async def delete_note(identifier: str) -> Dict[str, Any]:
    """Delete a memory note.
    
    Args:
        identifier: Title or path of note to delete
        
    Returns:
        Dict with operation results
    """
    manager = get_memory_manager()
    return await manager.delete_memory(identifier)

async def create_memory_from_conversation(
    conversation: List[str],
    title: str = None,
    topic: str = None,
    folder: str = "conversations"
) -> Dict[str, Any]:
    """Create a memory note from conversation history.
    
    Args:
        conversation: List of conversation turns
        title: Title for the memory (optional)
        topic: Topic or subject of conversation (optional)
        folder: Folder to store in (default: "conversations")
        
    Returns:
        Dict with operation results
    """
    # Generate title if not provided
    if not title:
        if topic:
            title = f"Conversation about {topic}"
        else:
            title = f"Conversation on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    # Format conversation as markdown
    content = f"# {title}\n\n"
    content += f"## Conversation\n\n"
    
    for i, message in enumerate(conversation):
        role = "User" if i % 2 == 0 else "Assistant"
        content += f"**{role}**: {message}\n\n"
    
    # Add metadata and timestamp
    content += f"\n## Metadata\n\n"
    content += f"- [datetime] Recorded on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    if topic:
        content += f"- [topic] {topic}\n"
    
    # Add relations section
    content += f"\n## Relations\n\n"
    if topic:
        content += f"- about [[{topic}]]\n"
    content += f"- type [[Conversation]]\n"
    
    # Create the memory
    manager = get_memory_manager()
    return await manager.create_memory(
        title=title,
        content=content,
        folder=folder,
        tags=["conversation", topic] if topic else ["conversation"],
        verbose=True
    )