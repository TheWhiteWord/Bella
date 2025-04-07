"""Enhanced memory adapter for semantic search and memory management.

This module provides a unified interface for interacting with the enhanced memory system,
coordinating between the semantic processor and the file-based memory system.
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import re

from .enhanced_memory import EnhancedMemoryProcessor
from .memory_api import save_note, read_note, list_notes

class EnhancedMemoryAdapter:
    """Adapter for enhanced memory capabilities.
    
    This adapter integrates semantic memory capabilities with the existing
    file-based memory system.
    """
    
    def __init__(self, embedding_model: str = "nomic-embed-text"):
        """Initialize the enhanced memory adapter.
        
        Args:
            embedding_model: Name of Ollama embedding model to use
        """
        self._initialized = False
        self.processor = EnhancedMemoryProcessor(model_name=embedding_model)
        self.memory_dirs = ["facts", "preferences", "conversations", "reminders", "general"]
        
    async def initialize(self) -> bool:
        """Initialize the memory system.
        
        Returns:
            bool: Success status
        """
        if self._initialized:
            return True
            
        try:
            # First check if embedding model is available
            model_available = await self.processor.ensure_model_available()
            
            if not model_available:
                logging.error("Failed to initialize embedding model")
                return False
            
            # Index existing memories if needed
            await self._index_existing_memories()
            
            self._initialized = True
            logging.info("Enhanced memory system initialized")
            return True
            
        except Exception as e:
            logging.error(f"Error initializing memory system: {e}")
            return False
    
    async def _index_existing_memories(self) -> None:
        """Index existing memory files that aren't already indexed."""
        try:
            # Get list of all memory files
            all_notes = []
            for dir_name in self.memory_dirs:
                dir_notes = await list_notes(dir_name)
                for note in dir_notes:
                    all_notes.append((dir_name, note))
            
            # Check which ones need indexing
            indexed_count = 0
            for dir_name, note_name in all_notes:
                memory_id = f"{dir_name}/{note_name}"
                
                # Skip if already indexed
                if memory_id in self.processor.embeddings:
                    continue
                    
                # Read note content
                full_path = os.path.join("memories", dir_name, note_name + ".md")
                content = await read_note(full_path)
                
                if content:
                    # Index the memory
                    success = await self.processor.index_memory(memory_id, content)
                    if success:
                        indexed_count += 1
                    
            if indexed_count > 0:
                logging.info(f"Indexed {indexed_count} existing memories")
                
        except Exception as e:
            logging.error(f"Error indexing existing memories: {e}")
    
    async def search_memory(self, query: str) -> Tuple[Dict[str, Any], bool]:
        """Search for relevant memories using semantic search.
        
        Args:
            query: Search query text
            
        Returns:
            Tuple of (results_dict, success_status)
        """
        try:
            # Ensure initialization
            if not self._initialized:
                await self.initialize()
                
            # Find relevant memories
            relevant_memories = await self.processor.find_relevant_memories(
                query, threshold=0.65, top_k=5
            )
            
            # Format results
            results = {
                "primary_results": [],
                "secondary_results": []
            }
            
            # Process each result
            for i, (memory_id, score) in enumerate(relevant_memories):
                # Read memory content
                dir_name, note_name = memory_id.split('/', 1)
                full_path = os.path.join("memories", dir_name, note_name + ".md")
                
                # Record access
                self.processor.record_memory_access(memory_id)
                
                # Build result entry
                result_entry = {
                    "source": memory_id,
                    "path": full_path,
                    "score": score,
                    "content_preview": self.processor.metadata.get(memory_id, {}).get("content_preview", "")
                }
                
                # Add to appropriate result list
                if i < 2 and score > 0.70:  # Primary results (higher relevance)
                    results["primary_results"].append(result_entry)
                else:  # Secondary results
                    results["secondary_results"].append(result_entry)
            
            return results, True
            
        except Exception as e:
            logging.error(f"Error searching memory: {e}")
            return {"primary_results": [], "secondary_results": []}, False
    
    async def should_store_memory(self, text: str) -> Tuple[bool, float]:
        """Determine if the given text should be stored as a memory.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Tuple of (should_store, importance_score)
        """
        try:
            importance = await self.processor.score_memory_importance(text)
            should_store = importance >= 0.7
            return should_store, importance
        except Exception as e:
            logging.error(f"Error determining memory storage: {e}")
            return False, 0.0
    
    async def store_memory(
        self, memory_type: str, content: str, note_name: str = None
    ) -> Tuple[bool, Optional[str]]:
        """Store a new memory and index it.
        
        Args:
            memory_type: Type of memory (e.g. 'facts', 'preferences')
            content: Memory content
            note_name: Optional custom filename (will be generated if None)
            
        Returns:
            Tuple of (success, path)
        """
        try:
            # Ensure initialization
            if not self._initialized:
                success = await self.initialize()
                if not success:
                    logging.error("Failed to initialize memory system")
                    return False, None
                    
            # Validate memory type
            if memory_type not in self.memory_dirs:
                memory_type = "general"
                
            # Generate note name if not provided
            if not note_name:
                # Generate from content
                words = re.findall(r'\w+', content.lower())
                if len(words) > 5:
                    note_name = "-".join(words[:5])
                else:
                    # Fallback to timestamp
                    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    note_name = f"memory-{timestamp}"
            
            # Clean note name for filesystem safety
            note_name = re.sub(r'[^\w\-]', '-', note_name)
                    
            # Create memories directory if it doesn't exist
            os.makedirs(os.path.join("memories", memory_type), exist_ok=True)
                
            # Store to file system
            path = await save_note(content, memory_type, note_name)
            
            # For tests, if path is None but we can create the file directly, do so
            if not path:
                direct_path = os.path.join("memories", memory_type, note_name + ".md")
                try:
                    with open(direct_path, 'w') as f:
                        f.write(content)
                    path = direct_path
                    logging.info(f"Directly created memory file at {path}")
                except Exception as e:
                    logging.error(f"Failed to directly write memory: {e}")
            
            if path:
                # Index the memory
                memory_id = f"{memory_type}/{note_name}"
                await self.processor.index_memory(memory_id, content)
                return True, path
            else:
                logging.error(f"Failed to save note: {memory_type}/{note_name}")
                return False, None
                
        except Exception as e:
            logging.error(f"Error storing memory: {e}")
            return False, None
    
    async def process_conversation_turn(
        self, user_input: str, assistant_response: str
    ) -> Dict[str, Any]:
        """Process a conversation turn for memory operations.
        
        Args:
            user_input: User's input text
            assistant_response: Assistant's response text
            
        Returns:
            Dictionary with processing results
        """
        result = {
            "memory_stored": False,
            "memory_path": None,
            "modified_response": assistant_response
        }
        
        try:
            # Combine for context
            combined_text = f"User: {user_input}\nAssistant: {assistant_response}"
            
            # Check if it's worth remembering
            should_remember, importance = await self.should_store_memory(combined_text)
            
            if should_remember:
                # Create a summary for the memory
                memory_content = combined_text
                
                # Store in conversations directory
                success, path = await self.store_memory(
                    "conversations", 
                    memory_content, 
                    f"conversation-on-{datetime.now().strftime('%Y-%m-%d-%H%M')}"
                )
                
                if success:
                    result["memory_stored"] = True
                    result["memory_path"] = path
            
            return result
            
        except Exception as e:
            logging.error(f"Error processing conversation turn: {e}")
            return result

    # Add missing methods needed by main_app_integration.py
    
    async def enhance_memory_retrieval(
        self, query: str, standard_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enhance standard memory retrieval with semantic understanding.
        
        Args:
            query: Search query text
            standard_results: Standard search results
            
        Returns:
            Enhanced list of memory results
        """
        try:
            # Ensure initialization
            if not self._initialized:
                await self.initialize()
                
            # Get semantic search results
            semantic_results = await self.processor.find_relevant_memories(
                query, threshold=0.65, top_k=5
            )
            
            # Combine standard and semantic results
            enhanced_results = []
            
            # Process semantic results first for priority (they're more relevant)
            for memory_id, score in semantic_results:
                # Skip if score too low
                if score < 0.68:
                    continue
                    
                # Read memory content
                try:
                    dir_name, note_name = memory_id.split('/', 1)
                    full_path = os.path.join("memories", dir_name, note_name + ".md")
                    content = await read_note(full_path)
                    
                    # Skip if can't read
                    if not content:
                        continue
                        
                    # Build enhanced result
                    result = {
                        "source": memory_id,
                        "path": full_path,
                        "score": score,
                        "title": note_name.replace("-", " ").title(),
                        "content_preview": content[:200] + "..." if len(content) > 200 else content
                    }
                    
                    enhanced_results.append(result)
                except Exception:
                    continue
            
            # Add standard results not already included
            for result in standard_results:
                path = result.get("path", "")
                
                # Skip if already included from semantic search
                if any(er.get("path") == path for er in enhanced_results):
                    continue
                    
                # Add modified standard result (with a baseline score)
                standard_result = result.copy()
                if "score" not in standard_result:
                    standard_result["score"] = 0.6  # Baseline score for standard results
                
                enhanced_results.append(standard_result)
                
            # Sort by relevance score
            enhanced_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            return enhanced_results[:5]  # Limit to 5 total results
            
        except Exception as e:
            logging.error(f"Error enhancing memory retrieval: {e}")
            return standard_results  # Fall back to standard results
    
    def record_memory_access(self, memory_id: str) -> None:
        """Record that a memory was accessed.
        
        Args:
            memory_id: ID of the memory that was accessed
        """
        try:
            # Clean up memory ID if it's a file path
            if memory_id.endswith(".md"):
                memory_id = os.path.basename(memory_id).replace(".md", "")
                
                # Try to determine the directory
                for dir_name in self.memory_dirs:
                    if os.path.exists(os.path.join("memories", dir_name, memory_id + ".md")):
                        memory_id = f"{dir_name}/{memory_id}"
                        break
            
            # Record access in processor
            self.processor.record_memory_access(memory_id)
        except Exception as e:
            logging.error(f"Error recording memory access: {e}")
    
    async def should_save_memory(self, text: str) -> Tuple[bool, float]:
        """Determine if text should be saved as a memory.
        
        Alias for should_store_memory to match method name used in main_app_integration.py
        
        Args:
            text: Text to evaluate
            
        Returns:
            Tuple of (should_save, importance_score)
        """
        return await self.should_store_memory(text)
    
    async def summarize_for_storage(self, text: str, max_length: int = 150) -> str:
        """Prepare text for memory storage through summarization.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length in words
            
        Returns:
            Summarized text
        """
        try:
            # Use processor's extract_summary method
            return await self.processor.extract_summary(text, max_length)
        except Exception as e:
            logging.error(f"Error summarizing for storage: {e}")
            return text  # Return original text as fallback
    
    async def _find_memory_path(self, memory_id: str) -> Optional[str]:
        """Find the file path for a memory ID.
        
        Args:
            memory_id: ID of the memory to find
            
        Returns:
            Path to the memory file or None if not found
        """
        try:
            # Parse memory ID
            if "/" in memory_id:
                dir_name, note_name = memory_id.split("/", 1)
            else:
                # Try to find in all directories
                for dir_name in self.memory_dirs:
                    if await list_notes(dir_name, memory_id):
                        return os.path.join("memories", dir_name, memory_id + ".md")
                return None
            
            # Construct path
            return os.path.join("memories", dir_name, note_name + ".md")
        except Exception as e:
            logging.error(f"Error finding memory path: {e}")
            return None
            
    def cleanup(self) -> None:
        """Clean up resources used by the memory adapter."""
        if hasattr(self, 'processor'):
            self.processor.cleanup()

# Singleton instance
memory_adapter = EnhancedMemoryAdapter()