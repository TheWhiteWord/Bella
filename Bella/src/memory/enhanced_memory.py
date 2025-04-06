"""Enhanced memory processor with semantic search capabilities.

This module provides semantic memory capabilities using Ollama embeddings.
It handles embedding generation, vector storage, and semantic search.
"""

import os
import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from datetime import datetime
import aiohttp

class EnhancedMemoryProcessor:
    """Processor for enhanced memory with semantic search capabilities."""
    
    def __init__(self, model_name: str = "nomic-embed-text"):
        """Initialize the enhanced memory processor.
        
        Args:
            model_name: Ollama embedding model name
        """
        self.model_name = model_name
        self.ollama_base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.embeddings = {}  # memory_id -> embedding vector
        self.metadata = {}    # memory_id -> metadata 
        self.access_stats = {}  # memory_id -> [last_access_time, access_count]
        self.vector_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                      "memories", ".vector_store.json")
        self.stats_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                     "memories", ".memory_stats.json")
        
        # Load existing data if available
        self._load_vector_store()
        self._load_access_stats()
        
    async def ensure_model_available(self) -> bool:
        """Ensure the embedding model is available.
        
        Returns:
            bool: True if model is available
        """
        try:
            # Check if model is already available
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ollama_base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        for model in data.get("models", []):
                            if model["name"] == self.model_name:
                                logging.info(f"Embedding model '{self.model_name}' is available")
                                return True
                    
            # If we get here, model isn't available - attempt to pull
            logging.info(f"Embedding model '{self.model_name}' not found, attempting to pull...")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_base_url}/api/pull",
                    json={"name": self.model_name}
                ) as response:
                    if response.status == 200:
                        # Wait for response to complete
                        while True:
                            chunk = await response.content.readline()
                            if not chunk:
                                break
                            
                        logging.info(f"Successfully pulled embedding model '{self.model_name}'")
                        return True
                    else:
                        error_text = await response.text()
                        logging.error(f"Failed to pull model: {error_text}")
                        return False
                        
        except Exception as e:
            logging.error(f"Error ensuring embedding model availability: {e}")
            return False
    
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate an embedding vector for text using Ollama.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_base_url}/api/embeddings",
                    json={"model": self.model_name, "prompt": text}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("embedding")
                    else:
                        error_text = await response.text()
                        logging.error(f"Failed to generate embedding: {error_text}")
                        return None
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            return None
    
    async def index_memory(self, memory_id: str, content: str) -> bool:
        """Index a memory by generating and storing its embedding.
        
        Args:
            memory_id: Unique memory identifier
            content: Memory content
            
        Returns:
            Success status
        """
        try:
            # Generate embedding
            embedding = await self.generate_embedding(content)
            
            if embedding:
                # Store embedding and metadata
                self.embeddings[memory_id] = embedding
                
                # Store metadata
                self.metadata[memory_id] = {
                    "id": memory_id,
                    "indexed_at": datetime.now().isoformat(),
                    "content_preview": content[:100],
                    "content_length": len(content)
                }
                
                # Save to disk
                self._save_vector_store()
                
                return True
            else:
                return False
        except Exception as e:
            logging.error(f"Error indexing memory: {e}")
            return False
    
    async def find_relevant_memories(
        self, query: str, threshold: float = 0.7, top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """Find memories relevant to a query using semantic search.
        
        Args:
            query: Search query
            threshold: Minimum similarity score (0-1)
            top_k: Maximum number of results
            
        Returns:
            List of (memory_id, score) pairs
        """
        try:
            # Skip if no embeddings
            if not self.embeddings:
                return []
                
            # Generate query embedding
            query_embedding = await self.generate_embedding(query)
            
            if not query_embedding:
                return []
            
            # Convert to numpy for faster operations
            query_vector = np.array(query_embedding)
            
            # Calculate cosine similarity with all stored embeddings
            results = []
            for memory_id, embedding in self.embeddings.items():
                memory_vector = np.array(embedding)
                
                # Normalize vectors for cosine similarity
                norm_query = np.linalg.norm(query_vector)
                norm_memory = np.linalg.norm(memory_vector)
                
                if norm_query > 0 and norm_memory > 0:
                    similarity = np.dot(query_vector, memory_vector) / (norm_query * norm_memory)
                    
                    # Only keep if above threshold
                    if similarity >= threshold:
                        results.append((memory_id, float(similarity)))
            
            # Sort by similarity (highest first)
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-k
            return results[:top_k]
            
        except Exception as e:
            logging.error(f"Error searching for relevant memories: {e}")
            return []
            
    async def score_memory_importance(self, text: str) -> float:
        """Score the importance of a potential memory.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Importance score (0-1)
        """
        try:
            # Use LLM capabilities to evaluate importance
            # For now, use a simple heuristic
            
            # Longer text suggests importance (but not too long)
            length_score = min(len(text) / 500, 1.0) * 0.4
            
            # Check for question marks (questions are often less important as memories)
            question_penalty = 0.2 if "?" in text else 0
            
            # Calculate final score
            score = 0.5 + length_score - question_penalty
            
            # Ensure score is within bounds
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logging.error(f"Error scoring memory importance: {e}")
            return 0.5  # Default to middle importance
    
    async def extract_summary(self, text: str, max_length: int = 150) -> str:
        """Extract a summary of text for memory storage.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length in words
            
        Returns:
            Summarized text
        """
        # This would ideally use an LLM to generate summaries
        # For now, use simple truncation as fallback
        words = text.split()
        if len(words) <= max_length:
            return text
        
        return " ".join(words[:max_length]) + "..."
    
    def record_memory_access(self, memory_id: str) -> None:
        """Record that a memory was accessed.
        
        This helps track which memories are accessed frequently.
        
        Args:
            memory_id: Memory ID that was accessed
        """
        current_time = time.time()
        
        if memory_id in self.access_stats:
            last_access, count = self.access_stats[memory_id]
            self.access_stats[memory_id] = [current_time, count + 1]
        else:
            self.access_stats[memory_id] = [current_time, 1]
            
        # Don't save on every access to avoid excessive writes
        # We'll rely on periodic saves or application shutdown
    
    def _save_vector_store(self) -> None:
        """Save embeddings and metadata to disk."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.vector_path), exist_ok=True)
            
            # Convert embeddings to serializable format
            serializable_data = {
                "embeddings": {k: list(v) for k, v in self.embeddings.items()},
                "metadata": self.metadata,
                "last_updated": datetime.now().isoformat()
            }
            
            # Write to file
            with open(self.vector_path, 'w') as f:
                json.dump(serializable_data, f)
                
        except Exception as e:
            logging.error(f"Error saving vector store: {e}")
    
    def _load_vector_store(self) -> None:
        """Load embeddings and metadata from disk."""
        try:
            if os.path.exists(self.vector_path):
                with open(self.vector_path, 'r') as f:
                    data = json.load(f)
                    
                self.embeddings = {k: v for k, v in data.get("embeddings", {}).items()}
                self.metadata = data.get("metadata", {})
                
                logging.info(f"Loaded {len(self.embeddings)} embeddings from disk")
        except Exception as e:
            logging.error(f"Error loading vector store: {e}")
    
    def _save_access_stats(self) -> None:
        """Save memory access statistics to disk."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.stats_path), exist_ok=True)
            
            # Write to file
            with open(self.stats_path, 'w') as f:
                json.dump(self.access_stats, f)
                
        except Exception as e:
            logging.error(f"Error saving access stats: {e}")
    
    def _load_access_stats(self) -> None:
        """Load memory access statistics from disk."""
        try:
            if os.path.exists(self.stats_path):
                with open(self.stats_path, 'r') as f:
                    self.access_stats = json.load(f)
        except Exception as e:
            logging.error(f"Error loading access stats: {e}")
    
    def cleanup(self) -> None:
        """Perform cleanup tasks, like saving stats to disk."""
        try:
            self._save_vector_store()
            self._save_access_stats()
        except Exception as e:
            logging.error(f"Error during memory processor cleanup: {e}")