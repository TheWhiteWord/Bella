"""Memory conversation adapter for integrating enhanced memory in conversations.

This module provides a conversation-level interface for the enhanced memory system.
It handles pre-processing inputs and post-processing responses with memory features.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union


# Use the new BellaMemory system
from src.bella_memory.core import BellaMemory
from src.bella_memory.helpers import Summarizer, TopicExtractor, ImportanceScorer, MemoryClassifier
from src.bella_memory.storage import MemoryStorage
from src.bella_memory.embeddings import EmbeddingModel
from src.bella_memory.vector_db import VectorDB

class MemoryConversationAdapter:
    """Adapter for using BellaMemory in conversations (refactored for new API)."""

    def __init__(self, model_size: str = "XS", thinking_mode: bool = True):
        """Initialize the memory conversation adapter with BellaMemory and memory buffer."""
        self.conversation_history = []
        self.memory_buffer: List[Dict[str, Any]] = []
        self.buffer_max_items = 5  # Flush after 5 memories
        self.buffer_word_limit = 500  # Flush if total words exceed this
        self.buffer_importance_threshold = 0.9  # Flush if any memory is very important
        # Instantiate helpers and storage/embedding/vector DB
        self.bella_memory = BellaMemory(
            summarizer=Summarizer(model_size=model_size, thinking_mode=thinking_mode),
            topic_extractor=TopicExtractor(model_size=model_size, thinking_mode=thinking_mode),
            importance_scorer=ImportanceScorer(model_size=model_size, thinking_mode=thinking_mode),
            memory_classifier=MemoryClassifier(model_size=model_size, thinking_mode=thinking_mode),  # <-- ADD THIS
            storage=MemoryStorage(),
            embedding_model=EmbeddingModel(),
            vector_db=VectorDB(),
        )
    
    async def pre_process_input(
        self, user_input: str, formatted_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """No-op: Do not search memory or score importance before LLM response."""
        return {
            "has_memory_context": False,
            "memory_context": "",
            "memory_source": None,
            "confidence": "low",
            "should_remember": False,
            "importance": 0.0,
        }
    
    async def post_process_response(
        self, user_input: str, assistant_response: str, user_context: Optional[dict] = None
    ) -> Optional[str]:
        """Process response after it's generated and buffer for memory saving."""
        try:
            content = f"User: {user_input}\nAssistant: {assistant_response}"
            # Buffer immediately with minimal info
            self.memory_buffer.append({
                "content": content,
                "user_context": user_context or {},
                # Optionally, set importance=None here
            })
            # Launch background task to process and maybe flush
            asyncio.create_task(self._background_memory_processing())
            return None
        except Exception as e:
            logging.error(f"Error in memory post-processing: {e}")
            return None

    async def _background_memory_processing(self):
        # Optionally, score importance and update buffer items
        for mem in self.memory_buffer:
            if mem.get("importance") is None:
                try:
                    mem["importance"] = await self.bella_memory.importance_scorer.score(mem["content"])
                except Exception as e:
                    mem["importance"] = 0.0
        await self._maybe_flush_memory_buffer()

    async def _maybe_flush_memory_buffer(self):
        """Flush buffer if any flush condition is met."""
        # Condition 1: Buffer size
        if len(self.memory_buffer) >= self.buffer_max_items:
            await self.flush_memory_buffer()
            return
        # Condition 2: Total words
        total_words = sum(len(m["content"].split()) for m in self.memory_buffer)
        if total_words >= self.buffer_word_limit:
            await self.flush_memory_buffer()
            return
        # Condition 3: Any very important memory
        if any(m["importance"] >= self.buffer_importance_threshold for m in self.memory_buffer):
            await self.flush_memory_buffer()
            return

    async def flush_memory_buffer(self):
        """Flush (save) all buffered memories above importance threshold and clear the buffer (non-blocking)."""
        if not self.memory_buffer:
            return
        # Only store memories with importance >= threshold
        memories_to_store = [mem for mem in self.memory_buffer if mem.get("importance", 0.8) >= self.buffer_importance_threshold]
        for mem in memories_to_store:
            try:
                asyncio.create_task(self.bella_memory.store_memory(mem["content"], mem["user_context"]))
            except Exception as e:
                logging.error(f"Error launching background memory save: {e}")
        self.memory_buffer.clear()
        logging.info(
            f"Flushed memory buffer (stored {len(memories_to_store)} memories above importance threshold {self.buffer_importance_threshold})."
        )       
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