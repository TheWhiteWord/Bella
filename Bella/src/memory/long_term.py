"""Long-term memory implementation for Bella.

This module provides long-term memory capabilities using Praison's knowledge system.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .memory_manager import MemoryManager

# Import check from memory_manager
from .memory_manager import PRAISON_AVAILABLE

class LongTermMemory:
    """Long-term memory implementation using Praison knowledge.
    
    This class provides methods for storing and retrieving long-term
    memory items using Praison's knowledge system.
    """
    
    def __init__(self, manager: 'MemoryManager'):
        """Initialize long-term memory.
        
        Args:
            manager: The parent memory manager
        """
        self.manager = manager
        
    async def store(self, 
                   text: str, 
                   metadata: Optional[Dict[str, Any]] = None,
                   importance: float = 0.5) -> bool:
        """Store information in long-term memory.
        
        Args:
            text: The text to store
            metadata: Additional metadata for the memory item
            importance: Importance score (0.0-1.0)
            
        Returns:
            bool: Whether the store operation succeeded
        """
        if not self.manager.enable_memory or not self.manager.praison_memory:
            return False
            
        try:
            # Store directly using the Knowledge API
            # Add importance to metadata to track it
            combined_metadata = {
                **(metadata or {}),
                "importance": importance,
                "memory_type": "long_term"
            }
            
            # Use the store method directly on the Knowledge object
            # Note: The Knowledge.store method isn't async, so we wrap it in an executor
            # to avoid blocking the main thread
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.manager.praison_memory.store(
                    text,
                    user_id="bella_long_term",  # Differentiate from short-term
                    metadata=combined_metadata
                )
            )
            
            # Log the result for debugging
            logging.info(f"Long-term memory store result: {result}")
            
            return bool(result and ('success' in result and result['success'] or 'results' in result))
            
        except Exception as e:
            logging.error(f"Failed to store long-term memory: {e}")
            return False
    
    async def search(self, 
                    query: str, 
                    limit: int = 5) -> List[Dict[str, Any]]:
        """Search long-term memory for relevant items.
        
        Args:
            query: The query to search for
            limit: Maximum number of results
            
        Returns:
            List of memory items
        """
        if not self.manager.enable_memory or not self.manager.praison_memory:
            return []
            
        try:
            # Search directly using the Knowledge API
            # The search method isn't async, so we wrap it in an executor
            loop = asyncio.get_event_loop()
            search_results = await loop.run_in_executor(
                None,
                lambda: self.manager.praison_memory.search(
                    query,
                    user_id="bella_long_term",
                    limit=limit
                )
            )
            
            # Format results
            formatted_results = []
            
            # Handle different result formats
            if search_results and isinstance(search_results, list):
                for result in search_results:
                    formatted_results.append({
                        "text": result.get("text", ""),
                        "metadata": result.get("metadata", {}),
                        "score": result.get("score", 0.0)
                    })
            # Handle alternative result format with 'results' key
            elif search_results and isinstance(search_results, dict) and 'results' in search_results:
                for result in search_results['results']:
                    formatted_results.append({
                        "text": result.get("text", ""),
                        "metadata": result.get("metadata", {}),
                        "score": result.get("score", 0.0)
                    })
                
            return formatted_results
        except Exception as e:
            logging.error(f"Failed to search long-term memory: {e}")
            return []
            
    async def semantic_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search on stored memories.
        
        Args:
            query: The query to search for
            limit: Maximum number of results
            
        Returns:
            List of memory items
        """
        # For Knowledge, the standard search is already semantic
        # So we just call the regular search method
        return await self.search(query, limit)
        
    async def clear(self) -> bool:
        """Clear all long-term memory.
        
        Returns:
            Whether the clear operation succeeded
        """
        if not self.manager.enable_memory or not self.manager.praison_memory:
            return False
            
        try:
            # Use the delete_all method to clear long-term memory
            # This isn't async, so we wrap it in an executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.manager.praison_memory.delete_all(user_id="bella_long_term")
            )
            return True
        except Exception as e:
            logging.error(f"Failed to clear long-term memory: {e}")
            return False