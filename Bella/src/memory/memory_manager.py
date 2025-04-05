"""Memory manager for Bella voice assistant.

This module provides a centralized memory management system for Bella,
integrating Praison's knowledge capabilities while maintaining compatibility
with Bella's architecture.
"""

import os
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, TypeVar
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define a type variable for the Praison Knowledge class
PraisonKnowledgeType = TypeVar('PraisonKnowledgeType')

# Import Praison knowledge components - try different import paths
try:
    # First try direct import
    from praisonaiagents import Knowledge
    PRAISON_AVAILABLE = True
    logging.info("Successfully imported Praison Knowledge class")
except ImportError:
    try:
        # Try alternate import location
        from praisonaiagents.knowledge import Knowledge
        PRAISON_AVAILABLE = True
        logging.info("Successfully imported Praison Knowledge class from knowledge module")
    except ImportError:
        Knowledge = None  # Define Knowledge as None when not available
        PRAISON_AVAILABLE = False
        logging.warning("Praison not available. Install with: pip install praisonaiagents[memory]")

from .short_term import ShortTermMemory
from .long_term import LongTermMemory

class MemoryManager:
    """Manages memory capabilities for Bella via Praison AI integration.
    
    This class provides short and long-term memory capabilities by integrating
    with Praison's knowledge system, while presenting a simple interface for
    Bella's existing components to use.
    """
    
    def __init__(self, 
                 enable_memory: bool = True,
                 memory_config: Optional[Dict[str, Any]] = None,
                 embedding_model: str = "nomic-embed-text"):
        """Initialize the memory manager.
        
        Args:
            enable_memory: Whether to enable memory capabilities
            memory_config: Configuration for Praison memory
            embedding_model: Model to use for embeddings (default: nomic-embed-text for Ollama)
        """
        self.enable_memory = enable_memory and PRAISON_AVAILABLE
        self._memory_config = memory_config or {}
        self.embedding_model = self._memory_config.get("embedding_model", embedding_model)
        
        # Get the project root directory
        project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        memory_db_dir = project_root / "memory_db"
        
        # Ensure memory database directory exists
        os.makedirs(memory_db_dir, exist_ok=True)
        
        # Paths for memory databases within the project structure
        self.vector_store_path = os.path.join(memory_db_dir, "chroma_db")
        
        # Initialize memory components
        self._initialize_memory()
        
        # Create memory helpers
        self.short_term = ShortTermMemory(self)
        self.long_term = LongTermMemory(self)
        
        # Conversation tracking
        self.conversation_history = []
        
    def _initialize_memory(self) -> None:
        """Initialize Praison memory integration if available."""
        if not self.enable_memory or Knowledge is None:
            self._praison_memory = None
            logging.warning("Memory capabilities disabled or Praison not available")
            return
            
        # Create directory if needed
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        try:
            # Check environment
            using_ollama = os.environ.get("OPENAI_BASE_URL", "").startswith(("http://localhost", "http://127.0.0.1"))
            
            # Start with a simple configuration - use in-memory dict storage for testing
            # This avoids issues with vector embeddings and Chroma compatibility
            knowledge_config = {
                "vector_store": {
                    "provider": "dict",  # In-memory storage is most reliable
                    "config": {}
                }
            }
            
            # If embedding is needed and compatible with environment, add it
            if not using_ollama:
                # For non-Ollama setups that might have OpenAI API access
                if "OPENAI_API_KEY" in os.environ:
                    knowledge_config["embedding"] = {
                        "provider": "openai",
                        "config": {
                            "model": "text-embedding-3-small"
                        }
                    }
                    
            # For non-test environments, use persistent storage
            production_mode = not os.environ.get("BELLA_TEST_MODE", False)
            if production_mode:
                # Only use chroma for production (more persistent but requires embeddings)
                knowledge_config["vector_store"] = {
                    "provider": "chroma",
                    "config": {
                        "collection_name": "bella_memory",
                        "path": self.vector_store_path,
                    }
                }
            
            # Log what we're doing
            logging.info(f"Creating Knowledge instance with config: {knowledge_config}")
            
            # Create the Knowledge instance
            self._praison_memory = Knowledge(knowledge_config)
            logging.info("Successfully initialized Praison Knowledge instance")
                
        except Exception as e:
            self._praison_memory = None
            logging.error(f"Error initializing memory: {str(e)}")
            
    @property
    def praison_memory(self) -> Optional[PraisonKnowledgeType]:
        """Get the Praison Knowledge instance if available."""
        return self._praison_memory
            
    def store_conversation(self, 
                          role: str, 
                          content: str,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store a conversation message in memory.
        
        Args:
            role: Role of the speaker (user or assistant)
            content: Content of the message
            metadata: Additional metadata for the message
        """
        # Add to conversation history
        timestamp = datetime.now().isoformat()
        entry = {
            "role": role,
            "content": content,
            "timestamp": timestamp,
            **(metadata or {})
        }
        self.conversation_history.append(entry)
        
        # Store in memory using Praison if available
        if self.enable_memory and self._praison_memory is not None:
            try:
                # Use asyncio.create_task for async calls
                asyncio.create_task(self._store_to_memory(content, role, timestamp, metadata))
            except Exception as e:
                logging.error(f"Error storing conversation to memory: {e}")
        
    async def _store_to_memory(self,
                              content: str,
                              role: str,
                              timestamp: str,
                              metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store content to memory asynchronously.
        
        Args:
            content: Content to store
            role: Role of the speaker
            timestamp: ISO format timestamp
            metadata: Additional metadata
        """
        if self._praison_memory is None:
            return
            
        combined_metadata = {
            "role": role,
            "timestamp": timestamp,
            **(metadata or {})
        }
        
        try:
            # Store directly using the praison Knowledge API
            result = await self.short_term.store(
                text=content,
                metadata=combined_metadata
            )
            logging.info(f"Short-term memory store result: {result}")
            
            # For user queries or important info, store in long-term memory too
            if role == "user" or (metadata and metadata.get("important", False)):
                result = await self.long_term.store(
                    text=content,
                    metadata=combined_metadata,
                    importance=metadata.get("importance", 0.7) if metadata else 0.7
                )
                logging.info(f"Long-term memory store result: {result}")
        except Exception as e:
            logging.error(f"Error in _store_to_memory: {e}")
        
    async def search_memory(self, 
                     query: str, 
                     search_long_term: bool = True,
                     limit: int = 5) -> List[Dict[str, Any]]:
        """Search for information in memory.
        
        Args:
            query: Query string to search for
            search_long_term: Whether to search long-term memory
            limit: Maximum number of results to return
            
        Returns:
            List of memory results with text and metadata
        """
        short_term_results = await self.short_term.search(query, limit=limit)
        
        # If needed, search long-term memory
        long_term_results = []
        if search_long_term:
            long_term_results = await self.long_term.search(query, limit=limit)
            
        # Combine results (with deduplication if needed)
        results = short_term_results + long_term_results
        
        # Sort by relevance (if available)
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return results[:limit]
    
    def get_conversation_history(self, last_n: int = 10) -> str:
        """Get formatted conversation history as a string.
        
        Args:
            last_n: Number of recent messages to include
            
        Returns:
            Formatted conversation history
        """
        history = []
        for entry in self.conversation_history[-last_n:]:
            role = "User" if entry["role"] == "user" else "Assistant"
            history.append(f"{role}: {entry['content']}")
            
        return "\n".join(history)
    
    async def clear_memory(self, clear_long_term: bool = False) -> bool:
        """Clear memory.
        
        Args:
            clear_long_term: Whether to clear long-term memory too
            
        Returns:
            Whether the operation succeeded
        """
        success = True
        
        # Always clear short-term memory
        if not await self.short_term.clear():
            success = False
            
        # Optionally clear long-term memory
        if clear_long_term and self._praison_memory is not None:
            try:
                # Reset the knowledge store completely
                result = self._praison_memory.reset()
                logging.info(f"Memory reset result: {result}")
            except Exception as e:
                logging.error(f"Error clearing long-term memory: {e}")
                success = False
                
        # Always clear conversation history
        self.conversation_history = []
        
        return success