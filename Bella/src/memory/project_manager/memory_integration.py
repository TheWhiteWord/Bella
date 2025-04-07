"""Integration between autonomous memory and project-based memory systems.

This module provides integration between the autonomous memory system and the 
project-based memory system, ensuring that both systems use the same standardized format.
"""

import os
import asyncio
from typing import Dict, Any, List, Optional

from ..memory_api import save_note, read_note
from .memory_format_adapter import MemoryFormatAdapter

class MemoryIntegration:
    """Integrates autonomous memory with project-based memory."""
    
    def __init__(self, base_dir: str = None):
        """Initialize memory integration.
        
        Args:
            base_dir: Base directory for memory storage
        """
        # We'll use the same base directory as the memory API
        self.base_dir = base_dir
    
    async def save_standardized_memory(
        self,
        content: str,
        memory_type: str = "general",
        title: str = None,
        tags: List[str] = None,
        source: str = "autonomous"
    ) -> Dict[str, Any]:
        """Save memory in standardized format.
        
        This method converts the content to the standardized format before saving.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            title: Title for the memory
            tags: List of tags
            source: Source of the memory
            
        Returns:
            Dict with operation results
        """
        # Convert to standardized format
        formatted_content = MemoryFormatAdapter.convert_to_standard_format(
            content=content,
            title=title,
            memory_type=memory_type,
            tags=tags,
            source=source
        )
        
        # Save using the memory API
        path = await save_note(formatted_content, memory_type, title)
        
        if path:
            return {
                'success': True,
                'message': f"Memory saved to {memory_type}",
                'path': path
            }
        else:
            return {
                'success': False,
                'message': "Failed to save memory"
            }
    
    async def read_standardized_memory(self, memory_id: str) -> Dict[str, Any]:
        """Read memory in standardized format.
        
        Args:
            memory_id: ID of the memory to read
            
        Returns:
            Dict with memory content and metadata
        """
        # Read using the memory API
        content = await read_note(memory_id)
        
        if not content:
            return {
                'success': False,
                'message': f"Memory not found: {memory_id}"
            }
        
        # Check if content is already in standardized format
        if not MemoryFormatAdapter.is_standard_format(content):
            # Convert to standardized format
            formatted_content = MemoryFormatAdapter.convert_to_standard_format(
                content=content,
                title=os.path.basename(memory_id).replace('.md', ''),
                memory_type=os.path.basename(os.path.dirname(memory_id)),
                source="conversion"
            )
            
            # Extract data from standardized format
            data = MemoryFormatAdapter.extract_standard_format_data(formatted_content)
        else:
            # Extract data from existing standardized format
            data = MemoryFormatAdapter.extract_standard_format_data(content)
        
        return {
            'success': True,
            'metadata': data['metadata'],
            'content': data['content']
        }
    
    async def convert_existing_memories(self) -> Dict[str, Any]:
        """Convert existing memories to standardized format.
        
        This method is useful for migrating existing memories to the new format.
        
        Returns:
            Dict with conversion results
        """
        # Since you mentioned there's no existing data to migrate,
        # this is just a placeholder for future use
        return {
            'success': True,
            'message': "No existing memories to convert",
            'converted': 0
        }

# Create a singleton instance for easy access
_memory_integration = None

def get_memory_integration(base_dir: str = None) -> MemoryIntegration:
    """Get or create a singleton instance of MemoryIntegration.
    
    Args:
        base_dir: Base directory for memory storage
        
    Returns:
        MemoryIntegration instance
    """
    global _memory_integration
    if _memory_integration is None:
        _memory_integration = MemoryIntegration(base_dir)
    return _memory_integration