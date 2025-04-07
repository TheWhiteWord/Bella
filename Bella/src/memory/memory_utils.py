"""Memory utility functions for semantic search and similarity calculations.

This module provides embedding-based semantic search with TF-IDF fallbacks
for memory operations, optimized for philosophical and consciousness-related content.
"""

import re
import logging
import numpy as np
from typing import List, Dict, Any, Set, Tuple, Optional, Union
from pathlib import Path
import asyncio
from functools import lru_cache

# Initialize the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants - only used as fallbacks when embedding model isn't available
PHILOSOPHICAL_TERMS = [
    "consciousness", "qualia", "phenomenology", "epistemology", "ontology", 
    "metaphysics", "ethics", "aesthetics", "existentialism", "determinism",
    "free will", "mind", "identity", "self", "being", "existence", "reality",
    "knowledge", "truth", "meaning", "purpose", "morality", "perception",
    "cognition", "intentionality", "representation", "subjectivity", 
    "objectivity", "relativism", "absolutism", "rationalism", "empiricism",
    "materialism", "idealism", "dualism", "monism", "nihilism", "solipsism",
    "pragmatism", "utilitarianism", "deontology", "virtue ethics", "phenomenal",
    "noumenal", "transcendental", "immanent", "synthetic", "analytic",
    "a priori", "a posteriori", "dialectic", "hermeneutics", "deconstruction"
]

PHILOSOPHERS = [
    "plato", "aristotle", "socrates", "kant", "hegel", "nietzsche", "sartre",
    "heidegger", "husserl", "wittgenstein", "russell", "descartes", "spinoza",
    "leibniz", "locke", "berkeley", "hume", "kierkegaard", "schopenhauer",
    "marx", "engels", "horkheimer", "adorno", "marcuse", "habermas", "rawls",
    "camus", "beauvoir", "merleau-ponty", "gadamer", "ricoeur", "derrida",
    "foucault", "deleuze", "lyotard", "baudrillard", "rorty", "james", "dewey",
    "peirce", "quine", "davidson", "putnam", "kripke", "anscombe", "foot",
    "macintyre", "nozick", "singer", "dennett", "chalmers", "nagel", "jackson",
    "levinas", "confucius", "laozi", "zhuangzi", "buddha", "nagarjuna", "avicenna",
    "averroes", "aquinas", "ockham", "bacon", "hobbes", "mill", "popper", "kuhn"
]

# Embedding model singleton for reuse
_embedding_model = None

async def get_embedding_model():
    """Get or initialize the embedding model.
    
    Returns:
        The embedding model instance
    """
    global _embedding_model
    
    if _embedding_model is None:
        try:
            # Try to import and initialize from main app integration
            from .main_app_integration import memory_manager
            if hasattr(memory_manager, 'enhanced_adapter') and hasattr(memory_manager.enhanced_adapter, 'processor'):
                _embedding_model = memory_manager.enhanced_adapter.processor
                logger.info("Successfully loaded embedding model from memory manager")
            else:
                # Try to initialize directly
                from .enhanced_memory import EnhancedMemoryProcessor
                _embedding_model = EnhancedMemoryProcessor()
                logger.info("Initialized embedding model directly")
        except Exception as e:
            logger.warning(f"Failed to initialize embedding model: {e}")
    
    return _embedding_model

@lru_cache(maxsize=128)
async def generate_embedding(text: str) -> List[float]:
    """Generate an embedding vector for the given text.
    
    Args:
        text: Input text to embed
    
    Returns:
        Embedding vector as a list of floats or None if embedding fails
    """
    model = await get_embedding_model()
    
    if model is not None:
        try:
            return await model.generate_embedding(text)
        except Exception as e:
            logger.warning(f"Error generating embedding: {e}")
    
    return None

async def calculate_embedding_similarity(text1: str, text2: str) -> float:
    """Calculate cosine similarity between two texts using embeddings.
    
    Args:
        text1: First text
        text2: Second text
    
    Returns:
        Similarity score between 0 and 1
    """
    # Generate embeddings for both texts
    embedding1 = await generate_embedding(text1)
    embedding2 = await generate_embedding(text2)
    
    # If either embedding failed, fallback to TF-IDF
    if embedding1 is None or embedding2 is None:
        logger.info("Falling back to TF-IDF similarity due to embedding failure")
        return calculate_tfidf_similarity(text1, text2)
    
    # Calculate cosine similarity
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    # Compute cosine similarity
    try:
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        # Apply philosophical boost as a small enhancement
        return min(1.0, similarity * 1.05 if has_philosophical_content(text1) and has_philosophical_content(text2) else similarity)
    except Exception as e:
        logger.warning(f"Error calculating embedding similarity: {e}")
        return calculate_tfidf_similarity(text1, text2)

def calculate_tfidf_similarity(text1: str, text2: str) -> float:
    """Calculate TF-IDF similarity between two texts.
    
    This is a fallback method when embeddings are not available.
    
    Args:
        text1: First text
        text2: Second text
    
    Returns:
        Similarity score between 0 and 1
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
        )
        
        # Create a small corpus with these two texts
        corpus = [text1, text2]
        
        # Fit and transform the texts into TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Apply philosophical boost as enhancement
        if has_philosophical_content(text1) and has_philosophical_content(text2):
            similarity = min(1.0, similarity * 1.15)  # 15% boost for philosophical content
        
        return float(similarity)
    except Exception as e:
        logger.warning(f"Error in TF-IDF similarity: {e}")
        return simple_similarity(text1, text2)  # Ultimate fallback

def simple_similarity(text1: str, text2: str) -> float:
    """Extremely simple fallback similarity based on word overlap.
    
    Args:
        text1: First text
        text2: Second text
    
    Returns:
        Similarity score between 0 and 1
    """
    # Transform to lowercase and split into words
    words1 = set(re.findall(r'\b\w{3,}\b', text1.lower()))
    words2 = set(re.findall(r'\b\w{3,}\b', text2.lower()))
    
    # Handle empty sets
    if not words1 or not words2:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0

def is_query_topic_match(memory_text: str, query: str) -> bool:
    """Check if a memory contains topics that match a query.
    
    Args:
        memory_text: Memory text content
        query: User query to check for matching topics
    
    Returns:
        Boolean indicating topic match
    """
    memory_lower = memory_text.lower()
    query_lower = query.lower()
    
    # Check for exact phrase matches (quotes)
    quote_matches = re.findall(r'"([^"]+)"', query)
    if quote_matches:
        for quote in quote_matches:
            if quote.lower() in memory_lower:
                return True
    
    # Extract potential keywords from query
    query_keywords = re.findall(r'\b\w{4,}\b', query_lower)
    
    # Count matching keywords in memory
    match_count = sum(1 for keyword in query_keywords if keyword in memory_lower)
    
    # If more than 2 significant keywords match, consider it relevant
    if match_count >= 2:
        return True
    
    return False

def has_philosophical_content(text: str) -> bool:
    """Check if a text contains philosophical content.
    
    This is a simple heuristic used only as a fallback when embeddings aren't available.
    
    Args:
        text: Text to analyze
        
    Returns:
        Boolean indicating presence of philosophical content
    """
    text_lower = text.lower()
    
    # Check for philosophical terms
    for term in PHILOSOPHICAL_TERMS:
        if term in text_lower:
            return True
    
    # Check for philosopher names
    for name in PHILOSOPHERS:
        if name in text_lower:
            return True
    
    return False

async def find_similar_memories(query: str, memories: List[Dict[str, Any]], 
                              threshold: float = 0.65, 
                              max_results: int = 3) -> List[Dict[str, Any]]:
    """Find memories similar to a query using embeddings with TF-IDF fallback.
    
    Args:
        query: Query text to match against memories
        memories: List of memory dictionaries with at least 'content' field
        threshold: Similarity threshold for inclusion (0-1)
        max_results: Maximum number of results to return
        
    Returns:
        List of similar memories with similarity scores
    """
    if not memories:
        return []
    
    # Try using embeddings first
    model = await get_embedding_model()
    results = []
    
    if model is not None:
        try:
            # Generate query embedding
            query_embedding = await generate_embedding(query)
            
            if query_embedding is not None:
                # Compare against each memory
                for memory in memories:
                    content = memory.get("content", "")
                    
                    # Generate or retrieve memory embedding
                    memory_embedding = await generate_embedding(content)
                    
                    if memory_embedding is not None:
                        # Calculate similarity
                        vec1 = np.array(query_embedding)
                        vec2 = np.array(memory_embedding)
                        
                        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                        
                        # Apply philosophical boost if applicable
                        if has_philosophical_content(query) and has_philosophical_content(content):
                            similarity = min(1.0, similarity * 1.05)  # 5% boost for philosophical content
                        
                        # Add if above threshold
                        if similarity >= threshold:
                            results.append({
                                "memory": memory,
                                "similarity": float(similarity)
                            })
                
                # If we got results with embeddings, return them
                if results:
                    # Sort by similarity (highest first)
                    results.sort(key=lambda x: x["similarity"], reverse=True)
                    return results[:max_results]
        except Exception as e:
            logger.warning(f"Error in embedding-based memory search: {e}")
    
    # Fallback to TF-IDF if embeddings failed or returned no results
    logger.info("Falling back to TF-IDF for memory search")
    for memory in memories:
        content = memory.get("content", "")
        similarity = calculate_tfidf_similarity(query, content)
        
        if similarity >= threshold:
            results.append({
                "memory": memory,
                "similarity": similarity
            })
    
    # Sort by similarity (highest first)
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:max_results]