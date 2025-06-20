"""Memory conversation adapter for integrating enhanced memory in conversations.

This module provides a conversation-level interface for the enhanced memory system.
It handles pre-processing inputs and post-processing responses with memory features.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union


# Use the new BellaMemory system
from bella_memory.core import BellaMemory
from bella_memory.helpers import Summarizer, TopicExtractor, ImportanceScorer, MemoryClassifier
from bella_memory.storage import MemoryStorage
from bella_memory.embeddings import EmbeddingModel
from bella_memory.vector_db import VectorDB

class MemoryConversationAdapter:
    """Adapter for using BellaMemory in conversations (refactored for new API)."""

    def __init__(self, model_size: str = "XS", thinking_mode: bool = True):
        """Initialize the memory conversation adapter with BellaMemory and memory buffer."""
        self.conversation_history = []
        self.memory_buffer: List[Dict[str, Any]] = []
        self.buffer_max_items = 5  # Flush after 5 memories
        self.buffer_word_limit = 500  # Flush if total words exceed this
        self.buffer_importance_threshold = 0.7  # Flush if any memory is very important
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
            print("[DEBUG] post_process_response called")
            content = f"User: {user_input}\nAssistant: {assistant_response}"
            # Buffer immediately with minimal info
            self.memory_buffer.append({
                "content": content,
                "user_context": user_context or {},
                # Optionally, set importance=None here
            })
            print(f"[DEBUG] Added to memory_buffer. Buffer size: {len(self.memory_buffer)}")
            # Launch background task to process and maybe flush
            asyncio.create_task(self._background_memory_processing())
            return None
        except Exception as e:
            print(f"[DEBUG] Error in memory post-processing: {e}")
            logging.error(f"Error in memory post-processing: {e}")
            return None

    async def _background_memory_processing(self):
        print("[DEBUG] _background_memory_processing called")
        # Optionally, score importance and update buffer items
        for idx, mem in enumerate(self.memory_buffer):
            if mem.get("importance") is None:
                try:
                    mem["importance"] = await self.bella_memory.importance_scorer.score(mem["content"])
                    print(f"[DEBUG] Scored importance for buffer item {idx}: {mem['importance']}")
                except Exception as e:
                    print(f"[DEBUG] Error scoring importance for buffer item {idx}: {e}")
                    mem["importance"] = 0.0
        await self._maybe_flush_memory_buffer()

    async def _maybe_flush_memory_buffer(self):
        """Flush buffer if any flush condition is met."""
        print("[DEBUG] _maybe_flush_memory_buffer called")
        print(f"[DEBUG] Buffer size: {len(self.memory_buffer)}")
        # Condition 1: Buffer size
        if len(self.memory_buffer) >= self.buffer_max_items:
            print("[DEBUG] Flushing buffer: buffer_max_items reached")
            await self.flush_memory_buffer()
            return
        # Condition 2: Total words
        total_words = sum(len(m["content"].split()) for m in self.memory_buffer)
        print(f"[DEBUG] Buffer total words: {total_words}")
        if total_words >= self.buffer_word_limit:
            print("[DEBUG] Flushing buffer: buffer_word_limit reached")
            await self.flush_memory_buffer()
            return
        # Condition 3: Any very important memory
        max_importance = max((m.get("importance", 0.0) for m in self.memory_buffer), default=0.0)
        if max_importance >= self.buffer_importance_threshold:
            print("[DEBUG] Flushing buffer: buffer_importance_threshold reached")
            await self.flush_memory_buffer(precomputed_importance=max_importance)
            return

    async def flush_memory_buffer(self, precomputed_importance: float = None):
        """Flush (save) buffered memories as a single chunk if above importance threshold, with efficient classification and summarization.
        If precomputed_importance is provided, use it instead of rescoring.
        """
        print("[DEBUG] flush_memory_buffer called")
        if not self.memory_buffer:
            print("[DEBUG] flush_memory_buffer: buffer is empty, nothing to flush")
            return
        # Concatenate all buffered turns into one chunk
        chunk = "\n".join(mem["content"] for mem in self.memory_buffer)
        try:
            if precomputed_importance is not None:
                importance = precomputed_importance
                print(f"[DEBUG] flush_memory_buffer: using precomputed importance: {importance}")
            else:
                print("[DEBUG] flush_memory_buffer: rescoring importance for chunk")
                importance = await self.bella_memory.importance_scorer.score(chunk)
                print(f"[DEBUG] flush_memory_buffer: rescored chunk importance: {importance}")
            if importance < self.buffer_importance_threshold:
                print(f"[DEBUG] flush_memory_buffer: importance {importance} < threshold {self.buffer_importance_threshold}, clearing buffer without saving")
                self.memory_buffer.clear()
                return
            print("[DEBUG] flush_memory_buffer: running classifier and topic extractor")
            # Use the user_context from the last memory in the buffer (or merge if needed)
            user_context = self.memory_buffer[-1].get("user_context", {})
            print("[DEBUG] flush_memory_buffer: calling store_memory")
            # Save the chunk as a single memory (await for test determinism)
            await self.bella_memory.store_memory(
                chunk,
                user_context
            )
            print(
                f"[DEBUG] Flushed memory buffer (stored 1 chunk, importance {importance})."
            )
            logging.info(
                f"Flushed memory buffer (stored 1 chunk, importance {importance})."
            )
        except Exception as e:
            print(f"[DEBUG] Error during bulk memory flush: {e}")
            logging.error(f"Error during bulk memory flush: {e}")
        self.memory_buffer.clear()
    
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