"""Memory conversation adapter for integrating enhanced memory in conversations.

This module provides a conversation-level interface for the enhanced memory system.
It handles pre-processing inputs and post-processing responses with memory features.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

from .main_app_integration import memory_manager, ensure_memory_initialized

class MemoryConversationAdapter:
    """Adapter for using memory in conversations."""
    
    def __init__(self, embedding_model: str = "nomic-embed-text"):
        """Initialize the memory conversation adapter.
        
        Args:
            embedding_model: Name of Ollama embedding model to use
        """
        self.conversation_history = []  # Tracks conversation for memory context
        self.embedding_model = embedding_model
    
    async def pre_process_input(
        self, user_input: str, formatted_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Process user input before generating a response.
        
        This method:
        1. Searches memory for relevant context
        2. Evaluates if the input should be remembered
        
        Args:
            user_input: User's input text
            formatted_history: Optional conversation history in chat format
            
        Returns:
            Dictionary with memory context
        """
        # Initialize result
        result = {
            "has_memory_context": False,
            "memory_context": "",
            "memory_source": None,
            "confidence": "low",
        }
        
        try:
            # First check if memory is initialized
            await ensure_memory_initialized(self.embedding_model)
            
            # Search for relevant memories
            search_results, success = await memory_manager.search_memory(user_input)
            
            if success and search_results:
                # Extract primary results
                primary_results = search_results.get("primary_results", [])
                
                if primary_results:
                    # Get top result
                    top_result = primary_results[0]
                    relevance = top_result.get("score", 0)
                    
                    # Only use if relevance is high enough
                    if relevance >= 0.70:
                        # Get content
                        content = top_result.get("content_preview", "")
                        if not content and "path" in top_result:
                            from .memory_api import read_note
                            try:
                                content = await read_note(top_result["path"])
                            except:
                                pass
                        
                        # Prepare context
                        if content:
                            # Set result values
                            result["has_memory_context"] = True
                            result["memory_context"] = content
                            result["memory_source"] = top_result.get("source", "search")
                            
                            # Set confidence level
                            if relevance >= 0.85:
                                result["confidence"] = "high"
                            elif relevance >= 0.75:
                                result["confidence"] = "medium"
            
            # Check if this input should be remembered
            should_remember, importance = await memory_manager.should_store_memory(user_input)
            result["should_remember"] = should_remember
            result["importance"] = importance
            
        except Exception as e:
            logging.error(f"Error in memory pre-processing: {e}")
            
        return result
    
    async def post_process_response(
        self, user_input: str, assistant_response: str
    ) -> Optional[str]:
        """Process response after it's generated.
        
        This method:
        1. Updates memory with the conversation
        2. Optionally modifies the response to include memory references
        
        Args:
            user_input: User's input text
            assistant_response: Generated assistant response
            
        Returns:
            Optionally modified response text
        """
        try:
            # Check if important enough to remember
            context = f"User: {user_input}\nAssistant: {assistant_response}"
            
            # Use the fixed should_store_memory method which now returns a tuple correctly
            result = await memory_manager.should_store_memory(context)
            
            # Handle both tuple return and legacy boolean return formats
            if isinstance(result, tuple) and len(result) == 2:
                should_remember, importance = result
            elif isinstance(result, bool):
                should_remember = result
                importance = 0.5  # Default importance if not provided
            else:
                should_remember = False
                importance = 0.0
                logging.warning(f"Unexpected return type from should_store_memory: {type(result)}")
            
            if should_remember and importance >= 0.75:
                # This is a significant conversation worth remembering
                logging.info(f"Saving conversation to memory (importance: {importance:.2f})")
                
                # Process with memory manager - returns a dictionary, not a tuple
                result = await memory_manager.process_conversation_turn(
                    user_input, assistant_response
                )
                
                if isinstance(result, dict) and "modified_response" in result:
                    return result.get("modified_response")
                
            # Return unchanged response if no modifications needed
            return None
            
        except Exception as e:
            logging.error(f"Error in memory post-processing: {e}")
            return None
            
    def add_to_history(self, user_input: str, assistant_response: str) -> None:
        """Add a conversation turn to the history.
        
        Args:
            user_input: User's input text
            assistant_response: Assistant's response text
        """
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": assistant_response})
        
        # Keep history at a reasonable size
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def build_context_with_memory(self, base_context: str, memory_context: Dict[str, Any]) -> str:
        """Build a context string incorporating memory.
        
        Args:
            base_context: Base context string
            memory_context: Memory context dictionary
            
        Returns:
            Enhanced context string
        """
        if not memory_context or not memory_context.get("has_memory_context"):
            return base_context
            
        memory_text = memory_context.get("memory_context", "")
        if not memory_text:
            return base_context
            
        # Combine contexts
        combined = base_context.strip()
        
        # Add a separator if needed
        if combined and not combined.endswith("\n"):
            combined += "\n\n"
        elif not combined:
            combined = ""
            
        # Add memory context with appropriate framing
        confidence = memory_context.get("confidence", "low")
        
        if confidence == "high":
            memory_prefix = "Based on my memory:"
        elif confidence == "medium":
            memory_prefix = "I think I remember that:"
        else:
            memory_prefix = "I vaguely remember something about:"
            
        combined += f"{memory_prefix} {memory_text}"
        
        return combined