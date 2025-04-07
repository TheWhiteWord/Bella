"""Memory analytics using pandas for the autonomous memory system.

This module provides advanced analytics functions for the memory system,
focusing on understanding and optimizing memory patterns, similarity calculations,
and topic analysis with special attention to philosophical and consciousness-related content.
"""

import re
import logging
import os
from typing import List, Dict, Any, Set, Tuple, Optional, Union
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .memory_utils import PHILOSOPHICAL_TERMS, PHILOSOPHERS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryAnalytics:
    """Analytics for the memory system using pandas for data processing."""
    
    def __init__(self):
        """Initialize the memory analytics system."""
        # Cache for computed vectors and dataframes
        self.vectorizer = None
        self.tfidf_matrix_cache = {}
        self.vocabulary_cache = {}
        self.term_weights = self._create_term_weights()
    
    def _create_term_weights(self) -> Dict[str, float]:
        """Create a dictionary of term weights for boosting philosophical terms.
        
        Returns:
            Dictionary mapping terms to their importance weights
        """
        weights = {}
        
        # Philosophical terms get high weights
        for term in PHILOSOPHICAL_TERMS:
            weights[term] = 2.0
        
        # Philosophers get high weights
        for name in PHILOSOPHERS:
            weights[name] = 2.5
            
        # Additional domain-specific weights
        domain_weights = {
            # Consciousness terms
            "consciousness": 3.0,
            "qualia": 3.0,
            "awareness": 2.5,
            "subjective": 2.5,
            "experience": 2.0,
            "perception": 2.0,
            "cognition": 2.0,
            "self": 2.0,
            "identity": 2.0,
            "mind": 2.0,
            "thought": 2.0,
            "intentionality": 2.0,
            "phenomenology": 2.0,
            "neuroscience": 2.0,
            "neurobiology": 2.0,
            "neurophilosophy": 2.0,
            "neural correlates": 2.0,
            "brain": 2.0,
            "cognitive science": 2.0,
            "psychology": 2.0,
            "behavior": 2.0,
            "emotion": 2.0,
            "affect": 2.0,
            "mood": 2.0,
            "feeling": 2.0,
            "sentience": 2.0,
            "sentient": 2.0,
            "awareness": 2.0,
            "attention": 2.0,
            "memory": 2.0,
            "learning": 2.0,
            "knowledge": 2.0,
            "belief": 2.0,
            "understanding": 2.0,
            "interpretation": 2.0,
            "meaning": 2.0,
            "significance": 2.0,
            "value": 2.0,
            "ethics": 2.0,
            "morality": 2.0,
            "free will": 2.0,
            "determinism": 2.0,
            "agency": 2.0,
            "autonomy": 2.0,
            "responsibility": 2.0,
            "accountability": 2.0,
            "choice": 2.0,
            "decision": 2.0,
            "action": 2.0,
            "reaction": 2.0,
            "interaction": 2.0,
            "communication": 2.0,
            "language": 2.0,
            "symbol": 2.0,
            
            # AI terms
            "artificial intelligence": 2.5,
            "machine learning": 2.0,
            "neural network": 2.0,
            "algorithm": 1.5,
            "data": 1.5,
            "training": 1.5,
            "model": 1.5,
            "deep learning": 1.5,
            "natural language processing": 1.5,
            "computer vision": 1.5,
            "reinforcement learning": 1.5,
            "supervised learning": 1.5,
            "unsupervised learning": 1.5,
            "semi-supervised learning": 1.5,
            "transfer learning": 1.5,
            "feature extraction": 1.5,
            "feature selection": 1.5,   
            
            # Art terms
            "art": 2.0,
            "aesthetic": 2.5,
            "beauty": 2.0,
            "expression": 1.8,
            "poetry": 1.8,
            "creativity": 2.0,
            "imagination": 2.0,
            "inspiration": 1.8,
            "emotion": 2.0,
            "symbolism": 1.5,
            "interpretation": 1.5,
            "narrative": 2.0,
            "metaphor": 2.0,
            "composition": 1.5,
            "medium": 1.5,
            "style": 1.5,
            "form": 1.5,
            "color": 1.5,
            "texture": 1.5,
            "contrast": 1.5,
            "balance": 1.5,
            "harmony": 1.5,
            "rhythm": 1.5,
            "movement": 1.5,
            "space": 1.5,
            
            # Truth-seeking terms
            "truth": 3.0,
            "knowledge": 2.5,
            "epistemology": 2.5,
            "evidence": 2.0,
            "belief": 2.0,
            "skepticism": 2.0,
            "reality": 2.0,
            "objectivity": 2.0,
            "subjectivity": 2.0,
            "verification": 2.0,
            "falsification": 2.0,
            "rationality": 2.0,
            "reason": 2.0,
            "logic": 2.0,
            "argument": 2.0,
            "fallacy": 1.5,
            "bias": 1.5,
            "cognitive dissonance": 1.5,
            "confirmation bias": 1.5,
            "heuristic": 1.5,
            "intuition": 1.5,
            "deduction": 1.5,
            "induction": 1.5,
            "abduction": 1.5,
            "analogy": 1.5,
            "metaphysics": 2.0,
            "ontology": 2.0,
            "existentialism": 2.0,
        }
        weights.update(domain_weights)
        
        return weights
    
    def analyze_memory_corpus(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze a corpus of memories to extract insights.
        
        Args:
            memories: List of memory dictionaries with at least 'content' and 'title' fields
            
        Returns:
            Dictionary containing analytics results
        """
        if not memories:
            return {"error": "No memories to analyze"}
            
        # Create a DataFrame from the memories
        df = pd.DataFrame(memories)
        
        # Extract basic statistics
        stats = {
            "total_memories": len(df),
            "average_length": df["content"].apply(lambda x: len(x.split())).mean(),
            "memory_by_type": df.get("type", pd.Series(["unknown"] * len(df))).value_counts().to_dict(),
        }
        
        # Extract topics and create a topic frequency analysis
        if "content" in df.columns:
            topics = []
            for content in df["content"]:
                # Extract topics from each memory content
                content_topics = self.extract_topics_from_text(content)
                topics.extend(content_topics)
                
            # Count topic frequencies
            topic_counts = Counter(topics)
            stats["top_topics"] = dict(topic_counts.most_common(10))
            
            # Calculate philosophical content ratio
            philosophical_content = sum(1 for topic in topics if topic.lower() in PHILOSOPHICAL_TERMS or topic.lower() in PHILOSOPHERS)
            stats["philosophical_ratio"] = philosophical_content / max(1, len(topics))
        
        return stats
    
    def calculate_enhanced_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between texts with enhanced weighting for philosophical content.
        
        Args:
            text1: First text for comparison
            text2: Second text for comparison
            
        Returns:
            Similarity score between 0 and 1
        """
        # Create a custom vectorizer with term weighting
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True,
                max_features=10000
            )
        
        # Preprocess texts to boost philosophical content
        text1_processed = self._preprocess_text(text1)
        text2_processed = self._preprocess_text(text2)
        
        # Create a small corpus with these two texts
        corpus = [text1_processed, text2_processed]
        
        try:
            # Fit and transform the texts into TF-IDF matrix
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            
            # Convert to DataFrame for easier manipulation
            feature_names = self.vectorizer.get_feature_names_out()
            df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
            
            # Apply term weights
            for term, weight in self.term_weights.items():
                if term in df.columns:
                    df[term] *= weight
            
            # Calculate cosine similarity with the weighted values
            vec1 = df.iloc[0].values
            vec2 = df.iloc[1].values
            
            # Normalize vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
            
            # Apply philosophical content boost
            cosine_sim = self._boost_similarity_score(cosine_sim, text1, text2)
            
            return float(min(1.0, cosine_sim))
        except Exception as e:
            logger.warning(f"Error in enhanced similarity calculation: {str(e)}")
            # Fall back to simpler calculation
            return self._calculate_simple_similarity(text1, text2)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis, with special handling for philosophical content.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase for consistent matching
        text_lower = text.lower()
        
        # Create augmented text with repeated important terms
        augmented_text = text
        
        # Boost philosophical terms
        for term in PHILOSOPHICAL_TERMS:
            if term in text_lower:
                # Add the term multiple times to boost its importance
                augmented_text += f" {term} {term}"
        
        # Boost philosopher names
        for name in PHILOSOPHERS:
            if name in text_lower:
                # Add the name multiple times to boost its importance
                augmented_text += f" {name} {name}"
        
        return augmented_text
    
    def _boost_similarity_score(self, base_score: float, text1: str, text2: str) -> float:
        """Apply a boost to the similarity score based on shared philosophical content.
        
        Args:
            base_score: Base similarity score
            text1: First text
            text2: Second text
            
        Returns:
            Boosted similarity score
        """
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # Count shared philosophical terms
        shared_phil_terms = sum(1 for term in PHILOSOPHICAL_TERMS 
                              if term in text1_lower and term in text2_lower)
        
        # Count shared philosopher references
        shared_philosophers = sum(1 for name in PHILOSOPHERS 
                                if name in text1_lower and name in text2_lower)
        
        # Calculate boost based on shared terms
        boost = 0.0
        
        if shared_phil_terms > 0:
            boost += min(0.3, shared_phil_terms * 0.05)
            
        if shared_philosophers > 0:
            boost += min(0.3, shared_philosophers * 0.1)
        
        # Apply the boost (ensure it doesn't exceed 1.0)
        return min(1.0, base_score + boost)
    
    def _calculate_simple_similarity(self, text1: str, text2: str) -> float:
        """A simpler backup similarity calculation when the main method fails.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Extract words (longer than 3 chars)
        words1 = set(re.findall(r'\b\w{4,}\b', text1.lower()))
        words2 = set(re.findall(r'\b\w{4,}\b', text2.lower()))
        
        # Handle empty sets
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        # Basic similarity
        base_similarity = intersection / union if union > 0 else 0.0
        
        # Check for philosophical term overlap
        phil_terms1 = {word for word in words1 if word in PHILOSOPHICAL_TERMS}
        phil_terms2 = {word for word in words2 if word in PHILOSOPHICAL_TERMS}
        
        # Calculate philosophical term overlap
        phil_intersection = len(phil_terms1.intersection(phil_terms2))
        
        # Boost similarity based on philosophical term overlap
        if phil_intersection > 0:
            boost = min(0.3, phil_intersection * 0.1)
            base_similarity = min(1.0, base_similarity + boost)
        
        return base_similarity
    
    def extract_topics_from_text(self, text: str) -> List[str]:
        """Extract topics from text with emphasis on philosophical content.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted topics
        """
        # Initialize topics list
        topics = []
        
        # Convert to lowercase for matching
        text_lower = text.lower()
        
        # First check for philosophical terms
        for term in PHILOSOPHICAL_TERMS:
            if term in text_lower:
                topics.append(term)
        
        # Check for philosopher names
        for name in PHILOSOPHERS:
            if name in text_lower:
                topics.append(name)
        
        # Try to extract topics using TF-IDF
        try:
            # Create a small corpus with just this text
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=100
                )
            
            # Transform the text
            tfidf_matrix = self.vectorizer.fit_transform([text])
            
            # Get feature names
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get the highest TF-IDF values and corresponding terms
            tfidf_scores = zip(feature_names, tfidf_matrix.toarray()[0])
            sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
            
            # Add top scoring terms that aren't already in topics
            existing_topics = set(t.lower() for t in topics)
            for term, score in sorted_scores[:10]:  # Get top 10
                if term not in existing_topics:
                    topics.append(term)
                    existing_topics.add(term)
        except Exception:
            pass
        
        # Extract proper nouns (philosophers, etc.)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
        existing_topics = set(t.lower() for t in topics)
        
        for noun in proper_nouns:
            if noun.lower() not in existing_topics:
                topics.append(noun)
                existing_topics.add(noun.lower())
        
        return topics[:15]  # Limit to top 15 topics
    
    def create_memory_dataframe(self, memories: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert memory data to a pandas DataFrame for analysis.
        
        Args:
            memories: List of memory dictionaries
            
        Returns:
            pandas DataFrame with memory data
        """
        if not memories:
            return pd.DataFrame()
            
        # Create DataFrame
        df = pd.DataFrame(memories)
        
        # Extract timestamps if available
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["date"] = df["timestamp"].dt.date
            
        # Extract word counts
        if "content" in df.columns:
            df["word_count"] = df["content"].apply(lambda x: len(x.split()))
            
        # Extract topics
        if "content" in df.columns:
            df["topics"] = df["content"].apply(self.extract_topics_from_text)
            
            # Extract philosophical content flag
            df["has_philosophical_content"] = df["topics"].apply(
                lambda topics: any(topic.lower() in PHILOSOPHICAL_TERMS or 
                                  topic.lower() in PHILOSOPHERS for topic in topics)
            )
            
        return df
    
    def analyze_memory_relevance(self, query: str, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze memory relevance to a query with enhanced philosophical awareness.
        
        Args:
            query: Query text
            memories: List of memory dictionaries with at least 'content' field
            
        Returns:
            List of dictionaries with memory content and relevance scores
        """
        if not memories:
            return []
            
        # Create a results list
        results = []
        
        for memory in memories:
            content = memory.get("content", "")
            
            # Calculate similarity score
            similarity = self.calculate_enhanced_similarity(query, content)
            
            # Determine confidence level
            if similarity > 0.7:
                confidence = "high"
            elif similarity > 0.4:
                confidence = "medium"
            else:
                confidence = "low"
                
            # Add to results
            results.append({
                "content": content,
                "title": memory.get("title", ""),
                "similarity_score": similarity,
                "confidence": confidence
            })
            
        # Sort by similarity score (highest first)
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return results
    
    def identify_dominant_topics(self, memories: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        """Identify dominant topics across a set of memories.
        
        Args:
            memories: List of memory dictionaries
            top_n: Number of top topics to return
            
        Returns:
            List of top topics with their frequency and related memories
        """
        if not memories:
            return []
            
        # Convert to DataFrame
        df = self.create_memory_dataframe(memories)
        
        if "topics" not in df.columns:
            return []
            
        # Flatten all topics
        all_topics = []
        topic_to_memories = {}
        
        for idx, row in df.iterrows():
            for topic in row["topics"]:
                topic_lower = topic.lower()
                all_topics.append(topic_lower)
                
                # Track which memories contain this topic
                if topic_lower not in topic_to_memories:
                    topic_to_memories[topic_lower] = []
                    
                topic_to_memories[topic_lower].append({
                    "title": row.get("title", ""),
                    "content_preview": row.get("content", "")[:100] + "..."
                })
        
        # Count topic frequencies
        topic_counts = Counter(all_topics)
        
        # Get top topics
        top_topics = []
        for topic, count in topic_counts.most_common(top_n):
            top_topics.append({
                "topic": topic,
                "count": count,
                "related_memories": topic_to_memories[topic][:3]  # First 3 related memories
            })
            
        return top_topics


# Create a singleton instance
memory_analytics = MemoryAnalytics()