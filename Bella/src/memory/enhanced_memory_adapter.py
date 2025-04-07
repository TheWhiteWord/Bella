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
import numpy as np

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
        self._embedding_cache = {}  # Cache for frequently used embeddings
        
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
    
    async def search_memory(self, query: str, fast_mode: bool = False) -> Tuple[Dict[str, Any], bool]:
        """Search for relevant memories using semantic search.
        
        Args:
            query: Search query text
            fast_mode: If True, use lightweight models for faster processing
            
        Returns:
            Tuple of (results_dict, success_status)
        """
        try:
            # Ensure initialization
            if not self._initialized:
                await self.initialize()
                
            # Find relevant memories
            relevant_memories = await self.processor.find_relevant_memories(
                query, threshold=0.6, top_k=8, fast_mode=fast_mode
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
                
                try:
                    content = await read_note(full_path)
                    
                    # Skip if can't read content
                    if not content:
                        continue
                        
                    # Extract a preview (first ~200 chars)
                    content_preview = content[:200] + "..." if len(content) > 200 else content
                    
                    # Build result entry
                    result_entry = {
                        "source": memory_id,
                        "path": full_path,
                        "score": score,
                        "title": note_name.replace("-", " ").title(),
                        "content_preview": content_preview
                    }
                    
                    # Add to appropriate result list
                    if i < 3 and score > 0.70:  # Primary results (higher relevance)
                        results["primary_results"].append(result_entry)
                    else:  # Secondary results
                        results["secondary_results"].append(result_entry)
                except Exception as e:
                    logging.warning(f"Error processing memory result {memory_id}: {e}")
                    continue
            
            return results, True
            
        except Exception as e:
            logging.error(f"Error searching memory: {e}")
            return {"primary_results": [], "secondary_results": []}, False
    
    async def should_store_memory(self, text: str) -> Tuple[bool, float]:
        """Determine if the given text should be stored as a memory using semantic analysis.
        
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
            
            # Check if it's worth remembering using semantic importance scoring
            should_remember, importance = await self.should_store_memory(combined_text)
            
            if should_remember:
                # Create a memory with the conversation
                memory_content = combined_text
                
                # Try to generate a meaningful title
                topic_words = []
                proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', combined_text)
                
                if proper_nouns:
                    # Use top 2 proper nouns
                    from collections import Counter
                    top_nouns = [noun for noun, _ in Counter(proper_nouns).most_common(2)]
                    topic_words = top_nouns
                else:
                    # Extract key terms
                    keywords = re.findall(r'\b[a-z]{4,}\b', user_input.lower())
                    topic_words = [kw for kw in keywords if kw not in 
                                  {"what", "when", "where", "which", "that", "this", 
                                   "there", "their", "about", "would", "could", "should"}][:2]
                
                # Generate title 
                title = "conversation-"
                if topic_words:
                    title += "-".join(topic_words).lower()
                else:
                    title += datetime.now().strftime('%Y-%m-%d-%H%M')
                
                # Store in conversations directory
                success, path = await self.store_memory(
                    "conversations", 
                    memory_content, 
                    title
                )
                
                if success:
                    result["memory_stored"] = True
                    result["memory_path"] = path
            
            return result
            
        except Exception as e:
            logging.error(f"Error processing conversation turn: {e}")
            return result

    async def enhance_memory_retrieval(
        self, query: str, standard_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enhance standard memory retrieval with semantic understanding.
        
        Uses embedding similarity to improve memory retrieval and ranking.
        
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
            semantic_results, success = await self.search_memory(query, fast_mode=True)
            
            if not success:
                return standard_results
                
            # Combine semantic and standard results with proper ranking
            enhanced_results = []
            
            # Process primary semantic results first (they're more relevant)
            for result in semantic_results["primary_results"]:
                enhanced_results.append(result)
                
            # Process secondary semantic results
            for result in semantic_results["secondary_results"]:
                # Only keep results with reasonable scores
                if result["score"] >= 0.65:
                    enhanced_results.append(result)
            
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
    
    async def compare_memory_similarity(self, text1: str, text2: str) -> float:
        """Compare the semantic similarity between two text snippets.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            
        Returns:
            Similarity score (0-1)
        """
        try:
            # Cache embeddings for frequently compared texts
            embedding1 = None
            embedding2 = None
            
            # Check cache first
            if text1 in self._embedding_cache:
                embedding1 = self._embedding_cache[text1]
            if text2 in self._embedding_cache:
                embedding2 = self._embedding_cache[text2]
                
            # Generate embeddings if not cached
            if embedding1 is None:
                embedding1 = await self.processor.generate_embedding(text1)
                if embedding1:
                    self._embedding_cache[text1] = embedding1
                    
            if embedding2 is None:
                embedding2 = await self.processor.generate_embedding(text2)
                if embedding2:
                    self._embedding_cache[text2] = embedding2
            
            # Calculate similarity if we have both embeddings
            if embedding1 and embedding2:
                # Calculate cosine similarity
                vec1 = np.array(embedding1)
                vec2 = np.array(embedding2)
                
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                return float(similarity)
                
            return 0.0  # Default if embeddings fail
            
        except Exception as e:
            logging.error(f"Error comparing memory similarity: {e}")
            return 0.0
    
    async def detect_memory_topics(self, text: str, max_topics: int = 3) -> List[str]:
        """Extract the main topics from text using semantic understanding.
        
        Args:
            text: Text to analyze
            max_topics: Maximum number of topics to extract
            
        Returns:
            List of topic strings
        """
        try:
            # Define common topics to compare against
            common_topics = [
                "technology", "science", "philosophy", "art", "history",
                "politics", "economics", "health", "education", "environment",
                "travel", "food", "personal", "work", "relationships"
            ]
            
            # Get embedding for input text
            text_embedding = await self.processor.generate_embedding(text)
            
            if not text_embedding:
                # Fall back to rule-based extraction
                return self._extract_topics_rule_based(text, max_topics)
                
            # Calculate similarity with each topic
            topic_scores = []
            for topic in common_topics:
                # Check cache first
                if topic in self._embedding_cache:
                    topic_embedding = self._embedding_cache[topic]
                else:
                    topic_embedding = await self.processor.generate_embedding(topic)
                    if topic_embedding:
                        self._embedding_cache[topic] = topic_embedding
                
                if topic_embedding:
                    # Calculate similarity
                    vec1 = np.array(text_embedding)
                    vec2 = np.array(topic_embedding)
                    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    topic_scores.append((topic, float(similarity)))
            
            # Sort by similarity (highest first)
            topic_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top topics
            return [topic for topic, score in topic_scores[:max_topics] if score > 0.4]
            
        except Exception as e:
            logging.error(f"Error detecting memory topics: {e}")
            return self._extract_topics_rule_based(text, max_topics)
    
    def _extract_topics_rule_based(self, text: str, max_topics: int = 3) -> List[str]:
        """Extract topics using rule-based approach (fallback).
        
        Args:
            text: Text to analyze
            max_topics: Maximum number of topics
            
        Returns:
            List of topics
        """
        # Find proper nouns for named entities
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
        
        # Find potential topic words (nouns)
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        
        # Remove stopwords
        stopwords = {"what", "when", "where", "which", "that", "this", "there", 
                    "their", "about", "would", "could", "should", "these", "those"}
        filtered_words = [w for w in words if w not in stopwords]
        
        # Combine proper nouns and frequent words
        topics = []
        
        # Add proper nouns first (up to half of max_topics)
        if proper_nouns:
            from collections import Counter
            noun_counter = Counter(proper_nouns)
            top_nouns = [noun for noun, _ in noun_counter.most_common(max(1, max_topics // 2))]
            topics.extend(top_nouns)
        
        # Add common words next
        if filtered_words:
            from collections import Counter
            word_counter = Counter(filtered_words)
            remaining_slots = max_topics - len(topics)
            if remaining_slots > 0:
                top_words = [word for word, _ in word_counter.most_common(remaining_slots)]
                topics.extend(top_words)
                
        return topics
            
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
        # Clear caches
        self._embedding_cache.clear()

# Singleton instance
memory_adapter = EnhancedMemoryAdapter()