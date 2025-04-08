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
import chromadb
from chromadb.utils import embedding_functions
import uuid

# Use absolute paths for ChromaDB to ensure consistency
import os.path
CHROMA_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "memories", "chroma_db"))
CHROMA_COLLECTION_NAME = "bella_memories"
CHROMA_METADATA = {"hnsw:space": "cosine"}

from .enhanced_memory import EnhancedMemoryProcessor
from .memory_api import save_note, read_note, list_notes

class EnhancedMemoryAdapter:
    """Adapter for enhanced memory capabilities.
    
    This adapter integrates semantic memory capabilities with the existing
    file-based memory system and uses ChromaDB for indexing.
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

        # --- ChromaDB Initialization ---
        try:
            logging.info(f"Initializing ChromaDB client at path: {CHROMA_DB_PATH}")
            # Ensure the directory exists
            os.makedirs(CHROMA_DB_PATH, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

            logging.info(f"Getting or creating ChromaDB collection: {CHROMA_COLLECTION_NAME}")
            # Get or create the collection. Using cosine distance is standard for text embeddings.
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata=CHROMA_METADATA  # Specify distance metric
            )
            logging.info(f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' ready. Item count: {self.chroma_collection.count()}")
        except Exception as e:
            logging.exception(f"Failed to initialize ChromaDB: {e}")  # Use exception for stack trace
            self.chroma_client = None
            self.chroma_collection = None
            # Initialization will fail later if ChromaDB is essential
        
    async def initialize(self) -> bool:
        """Initialize the memory system, including embedding model and ChromaDB check.
        
        Returns:
            bool: Success status
        """
        if self._initialized:
            return True

        # Check ChromaDB initialization status first
        if not self.chroma_client or not self.chroma_collection:
            logging.error("ChromaDB client or collection failed to initialize. Cannot proceed.")
            return False

        try:
            # Check if embedding model is available
            model_available = await self.processor.ensure_model_available()

            if not model_available:
                logging.error("Failed to initialize embedding model")
                return False

            logging.info(f"ChromaDB collection '{self.chroma_collection.name}' has {self.chroma_collection.count()} items.")

            self._initialized = True
            logging.info("Enhanced memory system initialized successfully.")
            return True

        except Exception as e:
            logging.exception(f"Error initializing memory system: {e}")  # Use exception
            return False

    async def _add_memory_to_vector_db(self, memory_id: str, content: str, metadata: dict) -> bool:
        """Generates embedding and adds the memory to the ChromaDB collection.

        Args:
            memory_id (str): A unique identifier for the memory (e.g., 'conversations/my-topic').
                             This will be used as the ChromaDB document ID.
            content (str): The text content of the memory to be embedded.
            metadata (dict): A dictionary containing metadata associated with the memory.
                             Must include 'file_path'. Other common fields: 'title', 'tags', 'created_at'.
                             
        Returns:
            bool: True if memory was successfully added to ChromaDB, False otherwise.
        """
        # Make sure we're initialized
        if not self._initialized:
            success = await self.initialize()
            if not success:
                logging.error("Failed to initialize memory adapter. Cannot add to ChromaDB.")
                return False

        if not self.chroma_collection:
            logging.error("ChromaDB collection not available. Cannot add memory.")
            return False

        if not memory_id:
            logging.error("Memory ID is required to add to ChromaDB.")
            return False

        if 'file_path' not in metadata:
            logging.error(f"Metadata for memory ID '{memory_id}' is missing 'file_path'. Cannot add to ChromaDB.")
            return False

        try:
            # Check if memory already exists in collection - if so, we'll update it
            existing = False
            try:
                # Check if ID already exists
                get_result = self.chroma_collection.get(
                    ids=[memory_id],
                    include=['metadatas']
                )
                existing = len(get_result['ids']) > 0
                if existing:
                    logging.info(f"Memory ID '{memory_id}' already exists in ChromaDB. Updating.")
            except Exception as e:
                logging.debug(f"Error checking if memory exists (likely new): {e}")
            
            # Generate Embedding
            logging.debug(f"Generating embedding for memory ID: {memory_id}")
            embedding_vector = await self.processor.generate_embedding(content)

            if not embedding_vector:
                logging.error(f"Failed to generate embedding for memory ID: {memory_id}")
                return False

            # Prepare Metadata for ChromaDB
            chroma_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    chroma_metadata[key] = value
                elif isinstance(value, list):
                    if all(isinstance(item, str) for item in value):
                        chroma_metadata[key] = ",".join(value)
                elif isinstance(value, datetime):
                    chroma_metadata[key] = value.isoformat()
                else:
                    chroma_metadata[key] = str(value)

            if 'file_path' not in chroma_metadata:
                logging.error(f"Critical metadata 'file_path' was lost during type conversion for memory ID '{memory_id}'.")
                return False

            # Add to ChromaDB Collection
            logging.info(f"Adding memory to ChromaDB. ID: {memory_id}, Path: {chroma_metadata.get('file_path')}")
            
            try:
                if existing:
                    # Update existing record
                    self.chroma_collection.update(
                        ids=[memory_id],
                        embeddings=[embedding_vector],
                        metadatas=[chroma_metadata]
                    )
                else:
                    # Add new record
                    self.chroma_collection.add(
                        ids=[memory_id],
                        embeddings=[embedding_vector],
                        metadatas=[chroma_metadata]
                    )
                
                # Verify memory was added by checking count
                count_after = self.chroma_collection.count()
                logging.info(f"ChromaDB collection now has {count_after} items.")
                
                logging.debug(f"Successfully added/updated memory ID '{memory_id}' in ChromaDB.")
                return True
                
            except Exception as e:
                logging.exception(f"ChromaDB operation failed for memory ID '{memory_id}': {e}")
                return False

        except Exception as e:
            logging.exception(f"Error adding memory ID '{memory_id}' to ChromaDB: {e}")
            return False

    async def search_memory(self, query: str, top_n: int = 5) -> Tuple[Dict[str, Any], bool]:
        """Search for relevant memories using ChromaDB semantic search.

        Args:
            query (str): Search query text.
            top_n (int): Maximum number of results to return.

        Returns:
            Tuple[Dict[str, Any], bool]: A dictionary containing search results
                                         (under 'results' key) and a success status.
                                         Results include metadata and similarity distance.
        """
        results_dict = {"results": []}
        if not self._initialized:
            logging.warning("Memory adapter not initialized. Attempting to initialize...")
            if not await self.initialize():
                logging.error("Failed to initialize memory adapter during search.")
                return results_dict, False

        if not self.chroma_collection:
            logging.error("ChromaDB collection not available for searching.")
            return results_dict, False

        try:
            # Check if collection is empty first
            if self.chroma_collection.count() == 0:
                logging.info("ChromaDB collection is empty, skipping query.")
                return {"results": []}, True
                
            # Generate Query Embedding
            logging.debug(f"Generating embedding for search query: '{query[:50]}...'")
            query_embedding = await self.processor.generate_embedding(query)

            if not query_embedding:
                logging.error("Failed to generate embedding for search query.")
                return results_dict, False

            # Query ChromaDB
            logging.debug(f"Querying ChromaDB collection '{self.chroma_collection.name}' for top {top_n} results.")
            chroma_results = self.chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_n,
                include=['metadatas', 'distances']  # Request metadata and distances
            )
            logging.debug(f"ChromaDB query returned {len(chroma_results.get('ids', [[]])[0])} results.")

            # Process and Format Results
            ids = chroma_results.get('ids', [[]])[0]
            distances = chroma_results.get('distances', [[]])[0]
            metadatas = chroma_results.get('metadatas', [[]])[0]

            if not ids:
                logging.info("Semantic search returned no results from ChromaDB.")
                return results_dict, True  # Successful search, just no matches

            formatted_results = []
            for i in range(len(ids)):
                memory_id = ids[i]
                distance = distances[i]
                metadata = metadatas[i]

                # Convert distance to similarity score
                similarity_score = max(0.0, 1.0 - distance)

                # Extract relevant info from metadata
                file_path = metadata.get('file_path', 'Unknown Path')
                title = metadata.get('title', memory_id.split('/')[-1])  # Fallback title
                tags_str = metadata.get('tags', '')
                tags = tags_str.split(',') if tags_str else []
                created_at = metadata.get('created_at', None)

                content_preview = f"Memory content stored in: {file_path}"

                result_entry = {
                    "id": memory_id,
                    "score": similarity_score,
                    "distance": distance,
                    "title": title,
                    "path": file_path,
                    "tags": tags,
                    "created_at": created_at,
                    "content_preview": content_preview,
                    "metadata": metadata
                }
                formatted_results.append(result_entry)

            results_dict["results"] = formatted_results
            logging.info(f"Formatted {len(formatted_results)} results from ChromaDB search.")
            return results_dict, True

        except Exception as e:
            logging.exception(f"Error during ChromaDB search: {e}")
            return results_dict, False

    async def should_store_memory(self, text: str) -> Tuple[bool, float]:
        """Determine if the given text should be stored as a memory using semantic analysis.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Tuple of (should_store, importance_score)
        """
        try:
            # Create a new coroutine each time to avoid reuse issues
            importance = await self._score_memory_importance_safe(text)
            # Lower threshold from 0.7 to 0.5 to ensure more memories are stored
            should_store = importance >= 0.5
            logging.info(f"Memory importance score: {importance:.2f}, should store: {should_store}")
            return should_store, importance
        except Exception as e:
            logging.error(f"Error determining memory storage: {e}")
            return False, 0.0
            
    async def _score_memory_importance_safe(self, text: str) -> float:
        """Safely score memory importance by creating a fresh scoring request each time.
        
        This implementation fixes the "cannot reuse already awaited coroutine" error by
        delegating directly to the scoring logic rather than reusing the coroutine.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Importance score (0-1)
        """
        try:
            # Use direct scoring logic to avoid coroutine reuse issues
            # Check for explicit cues first (faster than embeddings)
            text_lower = text.lower()
            
            # Some clear indicators of importance
            if any(phrase in text_lower for phrase in [
                "remember this", "don't forget", "important", "make sure to",
                "critical", "top priority", "must remember", "never forget"
            ]):
                return 0.95  # Very high importance
                
            # Check for potential personally relevant info
            if any(phrase in text_lower for phrase in [
                "my name is", "i work at", "my job", "my birthday", 
                "i live in", "my address", "my phone", "my email",
                "i like", "i love", "i prefer", "i hate", "i think"
            ]):
                return 0.85  # High importance
            
            # Use simpler scoring to prevent coroutine errors
            # Get embedding for the text
            text_embedding = await self.processor.generate_embedding(text, fast_mode=True)
            if not text_embedding:
                return 0.6  # Default to moderate importance if embedding fails
            
            # These are common concepts that represent important information
            important_concepts = ["personal information", "key fact", "important memory"]
            
            # Calculate similarity with key concepts
            scores = []
            for concept in important_concepts:
                concept_embedding = await self.processor.generate_embedding(concept, fast_mode=True)
                if concept_embedding:
                    # Calculate cosine similarity
                    text_vec = np.array(text_embedding)
                    concept_vec = np.array(concept_embedding)
                    similarity = np.dot(text_vec, concept_vec) / (np.linalg.norm(text_vec) * np.linalg.norm(concept_vec))
                    scores.append(similarity)
            
            # If we have valid scores, use the highest one
            if scores:
                importance = max(scores) * 0.85  # Scale to reasonable range
                return min(importance, 0.9)  # Cap at 0.9
            
            return 0.6  # Default to moderate importance
            
        except Exception as e:
            logging.error(f"Error in safe memory importance scoring: {e}")
            return 0.6  # Default to moderate importance on error
    
    async def store_memory(
        self, memory_type: str, content: str, note_name: str = None
    ) -> Tuple[bool, Optional[str]]:
        """Store a new memory (file and index).
        
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
            memory_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "memories", memory_type)
            os.makedirs(memory_dir, exist_ok=True)

            # Store to file system using memory_api
            path = await save_note(content, memory_type, note_name)

            if path:
                logging.info(f"Successfully saved memory file: {path}")
                # Use absolute path to ensure consistency
                abs_path = os.path.abspath(path) if not os.path.isabs(path) else path
                
                # Index the memory in ChromaDB
                memory_id = f"{memory_type}/{note_name}"
                metadata = {
                    "file_path": abs_path,
                    "title": note_name.replace("-", " ").title(),
                    "created_at": datetime.now().isoformat(),
                    "memory_type": memory_type,
                }
                # Add additional debug logging
                logging.debug(f"Adding to ChromaDB with ID: {memory_id}, Path: {abs_path}")
                
                # Store memory in ChromaDB and verify success
                await self._add_memory_to_vector_db(memory_id, content, metadata)
                return True, path
            else:
                logging.error(f"Failed to save note file: {memory_type}/{note_name}")
                return False, None

        except Exception as e:
            logging.exception(f"Error storing memory: {e}")
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
            semantic_results_dict, success = await self.search_memory(query, top_n=5)

            if not success:
                logging.error("Semantic search failed during enhancement.")
                return standard_results

            semantic_results = semantic_results_dict.get("results", [])

            # Combine semantic and standard results
            enhanced_results = []
            added_paths = set()

            # Add semantic results first
            for result in semantic_results:
                enhanced_results.append(result)
                if 'path' in result:
                    added_paths.add(result['path'])

            # Add standard results not already included
            for result in standard_results:
                path = result.get("path", "")
                if path and path not in added_paths:
                    standard_result = result.copy()
                    if "score" not in standard_result:
                        standard_result["score"] = 0.5
                    enhanced_results.append(standard_result)
                    added_paths.add(path)

            # Sort by relevance score
            enhanced_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

            logging.info(f"Enhanced retrieval combined {len(semantic_results)} semantic and {len(standard_results)} standard results into {len(enhanced_results)}.")
            return enhanced_results[:5]

        except Exception as e:
            logging.exception(f"Error enhancing memory retrieval: {e}")
            return standard_results
    
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

    # ... existing compare_memory_similarity, detect_memory_topics, _extract_topics_rule_based ...
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
                    # This check might need adjustment depending on list_notes implementation
                    # if await list_notes(dir_name, memory_id):
                    # Check file existence directly for simplicity here
                    potential_path = os.path.join("memories", dir_name, memory_id + ".md")
                    if os.path.exists(potential_path):
                         return potential_path
                return None

            # Construct path
            return os.path.join("memories", dir_name, note_name + ".md")
        except Exception as e:
            logging.error(f"Error finding memory path: {e}")
            return None

    def cleanup(self) -> None:
        """Clean up resources used by the memory adapter."""
        # ... existing code ...
        if hasattr(self, 'processor'):
            self.processor.cleanup()
        # Clear caches
        self._embedding_cache.clear()
        logging.info("EnhancedMemoryAdapter cleaned up.")

# Singleton instance
memory_adapter = EnhancedMemoryAdapter()