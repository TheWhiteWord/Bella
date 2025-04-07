"""Enhanced memory processor with semantic search capabilities.

This module provides semantic memory capabilities using embeddings from various models.
It handles embedding generation, vector storage, and semantic search.
"""

import os
import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Literal
from pathlib import Path
import numpy as np
from datetime import datetime
import aiohttp
from functools import lru_cache

class EmbeddingModelManager:
    """Manager for handling different embedding models.
    
    This class provides a unified interface for generating embeddings from
    different models, including Ollama local models and potential HuggingFace models.
    """
    
    def __init__(self, primary_model: str = "nomic-embed-text"):
        """Initialize the embedding model manager.
        
        Args:
            primary_model: Default embedding model name (Ollama model)
        """
        self.primary_model = primary_model
        self.ollama_base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.available_models = {
            "nomic-embed-text": {
                "source": "ollama",
                "dimensions": 768,
                "description": "General purpose embedding model from Nomic AI"
            },
            "all-minilm": {
                "source": "ollama",
                "dimensions": 384,
                "description": "Lightweight, fast embedding model for similarity tasks"
            }
        }
        self._models_checked = False
        
    async def ensure_models_available(self) -> bool:
        """Ensure that required embedding models are available.
        
        Returns:
            bool: True if primary model is available
        """
        if self._models_checked:
            return True
            
        try:
            # Check for Ollama models first
            if self.available_models[self.primary_model]["source"] == "ollama":
                # Check if model is already available in Ollama
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.ollama_base_url}/api/tags") as response:
                        if response.status == 200:
                            data = await response.json()
                            for model in data.get("models", []):
                                if model["name"] == self.primary_model:
                                    logging.info(f"Embedding model '{self.primary_model}' is available")
                                    self._models_checked = True
                                    return True
                        
                # If we get here, model isn't available - attempt to pull
                logging.info(f"Embedding model '{self.primary_model}' not found, attempting to pull...")
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.ollama_base_url}/api/pull",
                        json={"name": self.primary_model}
                    ) as response:
                        if response.status == 200:
                            # Wait for response to complete
                            while True:
                                chunk = await response.content.readline()
                                if not chunk:
                                    break
                                
                            logging.info(f"Successfully pulled embedding model '{self.primary_model}'")
                            self._models_checked = True
                            return True
                        else:
                            error_text = await response.text()
                            logging.error(f"Failed to pull model: {error_text}")
                            return False
            
            # For HuggingFace models, we'll check lazily when first used
            self._models_checked = True
            return True
                            
        except Exception as e:
            logging.error(f"Error ensuring embedding model availability: {e}")
            return False
    
    @lru_cache(maxsize=32)
    async def generate_embedding(self, 
                               text: str, 
                               model_name: str = None, 
                               normalize: bool = True) -> Optional[List[float]]:
        """Generate an embedding vector for text.
        
        Args:
            text: Text to embed
            model_name: Name of model to use (defaults to primary model)
            normalize: Whether to normalize the embedding vector
            
        Returns:
            Embedding vector or None if failed
        """
        # Use primary model if none specified
        if model_name is None:
            model_name = self.primary_model
            
        try:
            if model_name not in self.available_models:
                logging.warning(f"Unknown model '{model_name}', falling back to {self.primary_model}")
                model_name = self.primary_model
            
            model_source = self.available_models[model_name]["source"]
            
            # Generate using appropriate method based on source
            if model_source == "ollama":
                embedding = await self._generate_ollama_embedding(text, model_name)
            elif model_source == "huggingface":
                embedding = await self._generate_huggingface_embedding(text, model_name)
            else:
                logging.error(f"Unknown model source '{model_source}'")
                return None
                
            # Normalize if requested
            if embedding and normalize:
                embedding_array = np.array(embedding)
                norm = np.linalg.norm(embedding_array)
                if norm > 0:
                    embedding = (embedding_array / norm).tolist()
            
            return embedding
                
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            return None
            
    async def _generate_ollama_embedding(self, text: str, model_name: str) -> Optional[List[float]]:
        """Generate embedding using Ollama API.
        
        Args:
            text: Text to embed
            model_name: Ollama model name
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            # Ensure model is available
            if not self._models_checked:
                await self.ensure_models_available()
                
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_base_url}/api/embeddings",
                    json={"model": model_name, "prompt": text}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("embedding")
                    else:
                        error_text = await response.text()
                        logging.error(f"Failed to generate embedding: {error_text}")
                        return None
        except Exception as e:
            logging.error(f"Error generating Ollama embedding: {e}")
            return None
            
    async def _generate_huggingface_embedding(self, text: str, model_name: str) -> Optional[List[float]]:
        """Generate embedding using HuggingFace model.
        
        Args:
            text: Text to embed
            model_name: HuggingFace model name
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            # Lazy-load transformers if needed
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch
                import torch.nn.functional as F
            except ImportError:
                logging.error("transformers package not available for HuggingFace models")
                return None
            
            # For now, use the sentence-transformers models
            # These are used with mean pooling as described in HuggingFace docs
            model_path = f"sentence-transformers/{model_name}"
            
            # Load tokenizer and model (with caching)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path)
            
            # Create inputs
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            
            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Mean pooling
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            
            # Sum and normalize
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = (sum_embeddings / sum_mask).squeeze().tolist()
            
            return embedding
            
        except Exception as e:
            logging.error(f"Error generating HuggingFace embedding: {e}")
            return None
            
    def get_model_dimensions(self, model_name: str = None) -> int:
        """Get the dimensions of the specified model's embeddings.
        
        Args:
            model_name: Model name to check (defaults to primary model)
            
        Returns:
            Number of dimensions in the model's embeddings
        """
        if model_name is None:
            model_name = self.primary_model
            
        if model_name in self.available_models:
            return self.available_models[model_name]["dimensions"]
        else:
            # Default to primary model dimensions as fallback
            return self.available_models[self.primary_model]["dimensions"]

class EnhancedMemoryProcessor:
    """Processor for enhanced memory with semantic search capabilities."""
    
    def __init__(self, model_name: str = "nomic-embed-text"):
        """Initialize the enhanced memory processor.
        
        Args:
            model_name: Primary embedding model name
        """
        self.model_name = model_name
        self.embedding_manager = EmbeddingModelManager(primary_model=model_name)
        self.fast_model = "all-minilm"  # Use all-minilm from Ollama for fast operations
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
        # Check both primary model and fast model availability
        primary_available = await self.embedding_manager.ensure_models_available()
        
        # Check fast model separately if it's different from primary
        if self.fast_model != self.model_name:
            # Temporarily swap the primary model to check the fast model
            original_primary = self.embedding_manager.primary_model
            self.embedding_manager.primary_model = self.fast_model
            fast_available = await self.embedding_manager.ensure_models_available()
            # Restore original primary model
            self.embedding_manager.primary_model = original_primary
            
            return primary_available and fast_available
        
        return primary_available
    
    async def generate_embedding(self, text: str, fast_mode: bool = False) -> Optional[List[float]]:
        """Generate an embedding vector for text.
        
        Args:
            text: Text to embed
            fast_mode: If True, use lightweight model for faster processing
            
        Returns:
            Embedding vector or None if failed
        """
        # Choose model based on fast_mode
        model_name = self.fast_model if fast_mode else self.model_name
        
        return await self.embedding_manager.generate_embedding(text, model_name=model_name)
    
    async def index_memory(self, memory_id: str, content: str) -> bool:
        """Index a memory by generating and storing its embedding.
        
        Args:
            memory_id: Unique memory identifier
            content: Memory content
            
        Returns:
            Success status
        """
        try:
            # Generate embedding (use primary model for consistency in vector store)
            embedding = await self.generate_embedding(content)
            
            if embedding:
                # Store embedding and metadata
                self.embeddings[memory_id] = embedding
                
                # Store metadata
                self.metadata[memory_id] = {
                    "id": memory_id,
                    "indexed_at": datetime.now().isoformat(),
                    "content_preview": content[:100],
                    "content_length": len(content),
                    "model": self.model_name,
                    "dimensions": len(embedding)
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
        self, query: str, threshold: float = 0.7, top_k: int = 3, fast_mode: bool = False
    ) -> List[Tuple[str, float]]:
        """Find memories relevant to a query using semantic search.
        
        Args:
            query: Search query
            threshold: Minimum similarity score (0-1)
            top_k: Maximum number of results
            fast_mode: If True, use lightweight model for faster processing
            
        Returns:
            List of (memory_id, score) pairs
        """
        try:
            # Skip if no embeddings
            if not self.embeddings:
                return []
                
            # Generate query embedding
            query_embedding = await self.generate_embedding(query, fast_mode=fast_mode)
            
            if not query_embedding:
                return []
            
            # Convert to numpy for faster operations
            query_vector = np.array(query_embedding)
            
            # Calculate cosine similarity with all stored embeddings
            results = []
            for memory_id, embedding in self.embeddings.items():
                # Skip if dimensions don't match (prevents shape mismatch error)
                if len(query_vector) != len(embedding):
                    logging.warning(f"Dimension mismatch: query={len(query_vector)}, memory={len(embedding)} for {memory_id}")
                    continue
                    
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
            # Use semantic embeddings to compare with key concepts
            important_concepts = [
                "personal preference",
                "important fact",
                "key information",
                "personal detail",
                "crucial knowledge"
            ]
            
            # Get embedding for the text
            text_embedding = await self.generate_embedding(text, fast_mode=True)
            if not text_embedding:
                return 0.5  # Default to middle importance if embedding fails
                
            # Get embeddings for important concepts
            concept_scores = []
            for concept in important_concepts:
                concept_embedding = await self.generate_embedding(concept, fast_mode=True)
                if concept_embedding:
                    # Calculate cosine similarity
                    text_vec = np.array(text_embedding)
                    concept_vec = np.array(concept_embedding)
                    similarity = np.dot(text_vec, concept_vec) / (np.linalg.norm(text_vec) * np.linalg.norm(concept_vec))
                    concept_scores.append(float(similarity))
            
            if not concept_scores:
                # Fall back to heuristic approach
                # Longer text suggests importance (but not too long)
                length_score = min(len(text) / 500, 1.0) * 0.4
                
                # Check for question marks (questions are often less important as memories)
                question_penalty = 0.2 if "?" in text else 0
                
                # Calculate final score
                score = 0.5 + length_score - question_penalty
            else:
                # Use max similarity to any important concept
                score = max(concept_scores) * 0.8 + 0.2  # Scale to 0.2-1.0 range
            
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
        # Use the Ollama summary model if text is long enough to warrant summarization
        words = text.split()
        if len(words) > max_length:
            try:
                # Use the summary model from Ollama
                async with aiohttp.ClientSession() as session:
                    # Prepare a system message that specifies summarization task
                    system_message = "Summarize the following text concisely while preserving key information."
                    
                    # Call the Ollama API with the summary model
                    async with session.post(
                        f"{self.embedding_manager.ollama_base_url}/api/chat",
                        json={
                            "model": "summary:latest",
                            "messages": [
                                {"role": "system", "content": system_message},
                                {"role": "user", "content": text}
                            ],
                            "options": {"temperature": 0.1}  # Low temperature for deterministic summaries
                        }
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            summary = result.get("message", {}).get("content", "")
                            
                            # Return the summary if it's valid
                            if summary and len(summary.split()) <= max_length:
                                return summary
                            
                        # Log any issues with the API call
                        else:
                            error_text = await response.text()
                            logging.warning(f"Failed to generate summary: {error_text}")
                
            except Exception as e:
                logging.warning(f"Error summarizing text with model: {e}")
                # Fall back to truncation if model fails
            
            # Fallback: truncate the text if summary model fails
            return " ".join(words[:max_length]) + "..."
        
        # Return original text if it's already short enough
        return text
    
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
                # Check if file is empty
                if os.path.getsize(self.vector_path) == 0:
                    logging.warning(f"Vector store file is empty: {self.vector_path}")
                    return

                try:
                    with open(self.vector_path, 'r') as f:
                        data = json.load(f)
                        
                    self.embeddings = {k: v for k, v in data.get("embeddings", {}).items()}
                    self.metadata = data.get("metadata", {})
                    
                    logging.info(f"Loaded {len(self.embeddings)} embeddings from disk")
                except json.JSONDecodeError:
                    # Handle corrupted file by creating a fresh one
                    logging.warning(f"Vector store file corrupted, creating fresh: {self.vector_path}")
                    self._save_vector_store()
        except Exception as e:
            logging.error(f"Error loading vector store: {e}")
            # Initialize with empty data to prevent further errors
            self.embeddings = {}
            self.metadata = {}
    
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