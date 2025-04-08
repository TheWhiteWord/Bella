"""API layer for memory management in voice assistant.

Provides simplified interfaces to memory functions for use in voice interactions.
"""

import asyncio
import os
import re
import glob
import json
import yaml
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path

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

async def save_note(
    content: str,
    memory_type: str = "general",
    note_name: str = None
) -> Optional[str]:
    """Save a note to memory.
    
    This is an alias for write_note that matches the function signature
    used by the enhanced memory implementation.
    
    Args:
        content: Content of the note
        memory_type: Type/folder for the memory (e.g., "facts", "preferences")
        note_name: Name for the note file (without extension)
        
    Returns:
        Path to the saved note file or None if operation failed
    """
    # Generate a title if not provided
    if not note_name:
        # Use first few words of content as title
        words = re.findall(r'\w+', content.lower())
        if len(words) > 5:
            note_name = "-".join(words[:5])
        else:
            # Fallback to timestamp
            note_name = f"memory-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Ensure title is safe for filenames
    note_name = re.sub(r'[^\w\-]', '-', note_name)
    
    try:
        # Get absolute root memories directory
        manager = get_memory_manager()
        memory_dir = os.path.abspath(manager.memory_dir)
        
        # Create the memory directory if it doesn't exist
        memory_type_dir = os.path.join(memory_dir, memory_type)
        os.makedirs(memory_type_dir, exist_ok=True)
        
        # Construct the full file path
        file_path = os.path.join(memory_type_dir, f"{note_name}.md")
        
        # Create or format frontmatter
        frontmatter = {
            'title': note_name,
            'created': datetime.now().isoformat(),
            'updated': datetime.now().isoformat(),
            'type': 'memory',
            'tags': [memory_type]
        }
        
        # Create formatted markdown content with frontmatter
        formatted_content = f"---\n{yaml.dump(frontmatter)}---\n\n{content}"
        
        # Write the file directly to ensure it's created properly
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(formatted_content)
            
        # Return the absolute path to the created file
        return os.path.abspath(file_path)
        
    except Exception as e:
        logging.exception(f"Error directly saving note file: {e}")
        
        # Fall back to the original method if direct write fails
        result = await write_note(
            title=note_name,
            content=content,
            folder=memory_type,
            tags=[memory_type]
        )
        
        if result and 'path' in result:
            return os.path.abspath(result['path'])
        
        return None

async def list_notes(memory_type: str, prefix: str = None) -> List[str]:
    """List all notes in a memory type/folder.
    
    Args:
        memory_type: Type/folder of memories to list
        prefix: Optional filename prefix filter
        
    Returns:
        List of note names (without extension)
    """
    # Get memory manager instance
    manager = get_memory_manager()
    
    # Using memory_dir instead of base_dir
    base_dir = manager.memory_dir
    
    # Ensure directory exists
    memory_dir = os.path.join(base_dir, memory_type)
    if not os.path.exists(memory_dir):
        return []
    
    # List all markdown files
    pattern = os.path.join(memory_dir, f"{prefix or ''}*.md")
    files = glob.glob(pattern)
    
    # Extract basenames without extension
    notes = [os.path.splitext(os.path.basename(f))[0] for f in files]
    return sorted(notes)