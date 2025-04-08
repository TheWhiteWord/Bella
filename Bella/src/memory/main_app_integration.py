"""Integration of enhanced memory capabilities with Bella's main memory system.

This module provides a clean interface to access enhanced memory features for the main application.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import os

from .enhanced_memory_adapter import EnhancedMemoryAdapter
from .memory_api import search_notes
from .project_manager.initialize import initialize_project_management

class BellaMemoryManager:
    """Main integration point for Bella's memory systems."""
    
    def __init__(self):
        """Initialize the memory manager with enhanced capabilities."""
        self.enhanced_adapter = None
        self.autonomous_memory = None  # Will be initialized lazily
        self._initialized = False
        self._embedding_model = "nomic-embed-text"  # Default model
        self.project_management_initialized = False
    
    async def initialize(self, embedding_model: str = None):
        """Initialize memory systems.
        
        Args:
            embedding_model: Optional embedding model name to use
        """
        if embedding_model:
            self._embedding_model = embedding_model
            
        if not self._initialized:
            try:
                # Initialize enhanced memory adapter with specified embedding model
                self.enhanced_adapter = EnhancedMemoryAdapter(embedding_model=self._embedding_model)
                await self.enhanced_adapter.initialize()
                
                # Import and initialize autonomous memory here to avoid circular imports
                from .autonomous_memory import AutonomousMemory
                self.autonomous_memory = AutonomousMemory()
                
                self._initialized = True
                logging.info(f"Memory manager initialized successfully with {self._embedding_model}")
                
                # Initialize project management system
                try:
                    result = await initialize_project_management()
                    if result['success']:
                        self.project_management_initialized = True
                        logging.info("Project management system initialized successfully")
                    else:
                        logging.error(f"Failed to initialize project management: {result['message']}")
                except Exception as e:
                    logging.error(f"Error initializing project management: {e}")
            except Exception as e:
                logging.error(f"Error initializing memory manager: {e}")
    
    async def search_memory(self, query: str) -> Tuple[List[Dict[str, Any]], bool]:
        """Search memory with enhanced semantic capabilities.
        
        Args:
            query: Search query text
            
        Returns:
            Tuple of (search results, success status)
        """
        # Ensure initialized
        if not self._initialized:
            await self.initialize()
        
        try:
            # First get standard search results
            standard_results = await search_notes(query)
            
            if not standard_results:
                return [], False
                
            # If we have primary results, enhance them
            primary_results = standard_results.get("primary_results", [])
            
            if primary_results:
                # Get enhanced results with semantic search
                enhanced_results = await self.enhanced_adapter.enhance_memory_retrieval(
                    query, primary_results
                )
                
                # Replace primary results with enhanced ones
                standard_results["primary_results"] = enhanced_results
                
                # Track memory access for the top result
                if enhanced_results:
                    top_memory_id = os.path.basename(enhanced_results[0].get("path", "")).replace('.md', '')
                    if top_memory_id:
                        self.enhanced_adapter.record_memory_access(top_memory_id)
            
            return standard_results, True
            
        except Exception as e:
            logging.error(f"Error searching memory: {e}")
            return [], False
    
    async def evaluate_memory_importance(self, text: str) -> float:
        """Evaluate the importance of a potential memory.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Importance score between 0 and 1
        """
        # Ensure initialized
        if not self._initialized:
            await self.initialize()
            
        try:
            # Create a fresh coroutine each time to avoid the "cannot reuse already awaited coroutine" error
            # Instead of directly calling the processor's method, create a new score_memory_importance call
            if hasattr(self.enhanced_adapter, '_score_memory_importance_safe'):
                # Use the safe method we created earlier
                return await self.enhanced_adapter._score_memory_importance_safe(text)
            else:
                # Create a new processor call each time to prevent coroutine reuse
                importance = await self.enhanced_adapter.processor.score_memory_importance(text)
                return importance
        except Exception as e:
            logging.error(f"Error evaluating memory importance: {e}")
            return 0.5  # Default to medium importance
    
    async def should_store_memory(self, text: str) -> Tuple[bool, float]:
        """Determine if a text should be stored as a memory.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Tuple of (boolean indicating if text should be stored, importance score)
        """
        # Ensure initialized
        if not self._initialized:
            await self.initialize()
            
        try:
            # Get result from adapter, properly handling return values
            result = await self.enhanced_adapter.should_save_memory(text)
            
            # Check if result is a tuple (properly formatted) or just a boolean
            if isinstance(result, tuple) and len(result) == 2:
                return result
            elif isinstance(result, bool):
                # If it's just a boolean, add a default importance score
                importance = 0.75 if result else 0.3
                return result, importance
            else:
                # Fallback for unexpected return types
                logging.warning(f"Unexpected return type from should_save_memory: {type(result)}")
                should_save = bool(result)
                return should_save, 0.5
                
        except Exception as e:
            logging.error(f"Error determining if memory should be stored: {e}")
            # Fall back to a simple heuristic
            importance = min(len(text.split()) / 100, 0.9)  # Length-based importance score
            should_save = len(text.split()) > 10  # Simple fallback
            return should_save, importance
    
    async def prepare_text_for_storage(self, text: str, max_length: int = 150) -> str:
        """Prepare text for storage by summarizing if needed.
        
        Args:
            text: Text to prepare
            max_length: Maximum length in words
            
        Returns:
            Prepared text
        """
        # Ensure initialized
        if not self._initialized:
            await self.initialize()
            
        try:
            return await self.enhanced_adapter.summarize_for_storage(text, max_length)
        except Exception as e:
            logging.error(f"Error preparing text for storage: {e}")
            return text  # Return original text if error
    
    async def process_conversation_turn(
        self, user_input: str, response_text: str = None
    ) -> Dict[str, Any]:
        """Process a conversation turn with enhanced memory capabilities.
        
        This method integrates autonomous memory with enhanced memory features.
        
        Args:
            user_input: User's input text
            response_text: Assistant's response text (None if pre-processing)
            
        Returns:
            Memory context dictionary
        """
        # Ensure initialized
        if not self._initialized:
            await self.initialize()
        
        # Process with autonomous memory
        modified_response, memory_context = await self.autonomous_memory.process_conversation_turn(
            user_input, response_text
        )
        
        # Add enhanced memory capabilities if appropriate
        if response_text is None:
            # This is pre-processing (before response)
            # Check if autonomous memory found something
            if not memory_context.get("has_memory_context", False):
                # Try semantic search as a backup
                query = user_input
                semantic_results = await self.enhanced_adapter.processor.find_relevant_memories(
                    query, threshold=0.75, top_k=1
                )
                
                if semantic_results:
                    # Get the top result
                    memory_id, score = semantic_results[0]
                    
                    # Only use if highly relevant
                    if score > 0.8:
                        try:
                            # Find and read memory content
                            memory_path = await self.enhanced_adapter._find_memory_path(memory_id)
                            if memory_path:
                                from .memory_api import read_note
                                
                                memory_content = await read_note(memory_path)
                                
                                # Record access
                                self.enhanced_adapter.record_memory_access(memory_id)
                                
                                # Return as memory context
                                memory_context = {
                                    "has_memory_context": True,
                                    "memory_response": f"According to my memory: {memory_content[:200]}...",
                                    "memory_source": "semantic_search",
                                    "confidence": "high" if score > 0.85 else "medium"
                                }
                        except Exception as e:
                            logging.error(f"Error retrieving semantic memory: {e}")
        
        # Return result
        return {"modified_response": modified_response, "memory_context": memory_context}


# Singleton instance for easy access
memory_manager = BellaMemoryManager()

async def ensure_memory_initialized(embedding_model: str = "nomic-embed-text"):
    """Ensure memory manager is initialized.
    
    Args:
        embedding_model: Name of Ollama embedding model to use
    """
    await memory_manager.initialize(embedding_model)