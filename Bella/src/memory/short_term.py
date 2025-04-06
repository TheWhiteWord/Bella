"""Short-term memory implementation for Bella.

This module provides short-term memory capabilities using Praison's knowledge system.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .memory_manager import MemoryManager

# Import check from memory_manager
from .memory_manager import PRAISON_AVAILABLE

class ShortTermMemory:
    """Short-term memory implementation using Praison knowledge.
    
    This class provides methods for storing and retrieving short-term
    memory items using Praison's knowledge system.
    """
    
    def __init__(self, manager: 'MemoryManager'):
        """Initialize short-term memory.
        
        Args:
            manager: The parent memory manager
        """
        self.manager = manager
        
    async def store(self, 
                   text: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store information in short-term memory.
        
        Args:
            text: The text to store
            metadata: Additional metadata for the memory item
            
        Returns:
            bool: Whether the store operation succeeded
        """
        if not self.manager.enable_memory or not self.manager.praison_memory:
            return False
            
        try:
            # Ensure metadata is a dict
            mem_metadata = metadata or {}
            
            # Add memory type to metadata
            mem_metadata["memory_type"] = "short_term"
            
            # Use the store method directly on the Knowledge object
            # Note: The Knowledge.store method isn't async, so we wrap it in an executor
            # to avoid blocking the main thread
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.manager.praison_memory.store(
                    text,
                    user_id="bella",  # Use a consistent user_id for the assistant
                    metadata=mem_metadata
                )
            )
            
            # Log the result for debugging
            logging.info(f"Short-term memory store result: {result}")
            
            # Check for successful storage - handle different response formats
            if isinstance(result, dict):
                if result.get('success') is True:
                    return True
                if 'results' in result and result['results']:
                    return True
            
            return False
            
        except Exception as e:
            logging.error(f"Failed to store short-term memory: {e}")
            return False
    
    async def search(self, 
                    query: str, 
                    limit: int = 5) -> List[Dict[str, Any]]:
        """Search short-term memory for relevant items.
        
        Args:
            query: The query to search for
            limit: Maximum number of results
            
        Returns:
            List of memory items
        """
        if not self.manager.enable_memory or not self.manager.praison_memory:
            # Return empty list when Praison is not available
            return []
            
        try:
            # Use the search method directly on the Knowledge object
            # The search method isn't async, so we wrap it in an executor
            loop = asyncio.get_event_loop()
            search_results = await loop.run_in_executor(
                None,
                lambda: self.manager.praison_memory.search(
                    query,
                    user_id="bella",  # Use a consistent user_id for the assistant
                    limit=limit
                )
            )
            
            # Format results
            formatted_results = []
            
            # Check if we have a valid result structure and handle different formats
            if search_results:
                if isinstance(search_results, list):
                    # Direct list of results
                    for result in search_results:
                        if isinstance(result, dict):
                            formatted_results.append({
                                "text": result.get("text", ""),
                                "metadata": result.get("metadata", {}),
                                "score": result.get("score", 0.0)
                            })
                elif isinstance(search_results, dict):
                    # Results inside a dict under 'results' key
                    if 'results' in search_results and isinstance(search_results['results'], list):
                        for result in search_results['results']:
                            if isinstance(result, dict):
                                formatted_results.append({
                                    "text": result.get("text", ""),
                                    "metadata": result.get("metadata", {}),
                                    "score": result.get("score", 0.0)
                                })
                
            return formatted_results
        except Exception as e:
            logging.error(f"Failed to search short-term memory: {e}")
            return []
            
    async def clear(self) -> bool:
        """Clear all short-term memory.
        
        Returns:
            Whether the clear operation succeeded
        """
        if not self.manager.enable_memory or not self.manager.praison_memory:
            return False
            
        try:
            # Use the delete_all method on the Knowledge object
            # This isn't async, so we wrap it in an executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.manager.praison_memory.delete_all(user_id="bella")
            )
            return True
        except Exception as e:
            logging.error(f"Failed to clear short-term memory: {e}")
            return False