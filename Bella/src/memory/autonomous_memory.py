"""Autonomous memory integration for voice assistant.

Provides a seamless memory layer that works in the background during voice interactions.
"""

import asyncio
import re
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime

from .voice_memory_module import VoiceMemoryModule

class AutonomousMemory:
    """Autonomous memory system that works in the background during conversations."""
    
    def __init__(self):
        """Initialize autonomous memory system."""
        self.memory_module = VoiceMemoryModule()
        # Increase threshold for more conservative memory recall
        self.memory_threshold = 0.85  # Higher threshold for memory relevance (was 0.65)
        self.memory_check_interval = 10  # Increase interval between memory checks (was 5)
        self.last_memory_check = datetime.now()
        self.memory_context = []  # Currently active memory context
        self.last_recalled_memory = None  # Track last recalled memory to prevent repetition
        self.recall_count = 0  # Track number of recalls in current session
        self.max_recalls_per_session = 5  # Limit number of auto-recalls per session
        
    async def process_conversation_turn(
        self, 
        user_input: str, 
        response_text: str = None
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Process a conversation turn with memory integration.
        
        This method should be called for each turn in the conversation flow:
        - On user input to add memory context before generating a response
        - After response generation to potentially store memories
        
        Args:
            user_input: User's input text
            response_text: Assistant's response text (None if pre-processing)
            
        Returns:
            Tuple of (modified_response, memory_context)
        """
        # If this is pre-processing (before response generation)
        if response_text is None:
            # Try to find relevant memory context for this query
            memory_context = await self._retrieve_relevant_memory(user_input)
            
            # Only count as a recall if we actually found memory context
            if memory_context and memory_context.get("has_memory_context"):
                self.recall_count += 1
                self.last_recalled_memory = memory_context.get("memory_response", "")
            
            return None, memory_context
        
        # If this is post-processing (after response generation)
        # Process the turn with the memory module
        memory_response = await self.memory_module.process_input(user_input, response_text)
        
        if memory_response:
            # Return both the original response and the memory-specific response
            combined_response = f"{response_text}\n\n[Memory] {memory_response}"
            return combined_response, {}
            
        return response_text, {}
    
    async def _retrieve_relevant_memory(self, query: str) -> Dict[str, Any]:
        """Retrieve relevant memories for a user query.
        
        Args:
            query: User query text
            
        Returns:
            Dict with relevant memory context
        """
        # Apply recall limit to prevent overwhelming the conversation
        if self.recall_count >= self.max_recalls_per_session:
            return {"has_memory_context": False}
            
        # Check if query is likely seeking information from memory
        if self._is_knowledge_seeking_query(query):
            # Try to answer from memory
            memory_answer, found = await self.memory_module.query_memory(query)
            
            # Only return if the found memory is highly relevant
            if found and self._is_memory_relevant_to_query(memory_answer, query):
                # Don't return the same memory repeatedly
                if self.last_recalled_memory and self._is_similar_memory(memory_answer, self.last_recalled_memory):
                    return {"has_memory_context": False}
                    
                return {
                    "has_memory_context": True,
                    "memory_response": memory_answer,
                    "memory_source": "direct_query",
                    "confidence": "high"
                }
            
        # More conservative approach to implicit memory augmentation
        if self._should_augment_with_memory(query):
            search_result = await self._search_memory_for_context(query)
            if search_result and search_result.get("has_memory_context"):
                # Additional relevance check
                memory_text = search_result.get("memory_response", "")
                if not self._is_memory_relevant_to_query(memory_text, query):
                    return {"has_memory_context": False}
                    
                # Don't return the same memory repeatedly
                if self.last_recalled_memory and self._is_similar_memory(memory_text, self.last_recalled_memory):
                    return {"has_memory_context": False}
                    
                # Only include if confidence is sufficient
                confidence = search_result.get("confidence", "low")
                if confidence != "high":
                    return {"has_memory_context": False}
                    
                return search_result
        
        return {"has_memory_context": False}
    
    def _is_knowledge_seeking_query(self, query: str) -> bool:
        """Check if a query is explicitly seeking information that might be in memory.
        
        Args:
            query: User query text
            
        Returns:
            Boolean indicating if query is explicitly seeking stored knowledge
        """
        # More specific patterns that indicate explicit memory queries
        knowledge_patterns = [
            r"what (do you|did I)? (know|remember) about",
            r"(do you|can you) (know|remember|recall)",
            r"tell me (what|about)",
            r"do you have information (on|about)",
            r"what did I (tell|say|mention) (about|regarding)",
            r"have I (told|mentioned) (you|anything) about",
            r"did I ever (talk|tell you|mention) (about|that)",
            r"what's my (favorite|preferred)",
            r"remember (when|that|how|what)"
        ]
        
        query_lower = query.lower()
        
        for pattern in knowledge_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # More selective about proper nouns - require multiple context indicators
        memory_indicators = ["recall", "remember", "told", "said", "mentioned", 
                            "favorite", "preference", "opinion", "think about"]
        
        indicator_count = sum(1 for indicator in memory_indicators if indicator in query_lower)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', query)
        
        # Only consider proper nouns if there are other memory indicators
        if proper_nouns and indicator_count > 0:
            return True
            
        return False
    
    def _should_augment_with_memory(self, query: str) -> bool:
        """Check if we should augment the context with memory even if not explicitly asked.
        More conservative to reduce unnecessary memory recalls.
        
        Args:
            query: User query text
            
        Returns:
            Boolean indicating if we should add memory context
        """
        # Only check periodically to reduce frequency of memory recalls
        now = datetime.now()
        if (now - self.last_memory_check).total_seconds() < self.memory_check_interval:
            return False
            
        self.last_memory_check = now
        
        # Special handling for explicit memory-seeking queries - always check
        query_lower = query.lower()
        
        # Check for direct memory requests
        if "remember" in query_lower and ("what" in query_lower or "who" in query_lower or 
                                         "when" in query_lower or "where" in query_lower or 
                                         "how" in query_lower or "why" in query_lower):
            return True
        
        # More specific patterns for opinion/preference questions
        opinion_patterns = [
            r"what('s| is) (your|my) (opinion|thought|view|preference)",
            r"(do you|would you) (like|prefer|enjoy|recommend)",
            r"(what do you|how do you) (think|feel) about",
            r"what('s| is) my favorite"
        ]
        
        for pattern in opinion_patterns:
            if re.search(pattern, query_lower):
                return True
            
        # Only recall for highly specific queries
        # Count memory-related terms to gauge intent strength
        memory_terms = ["remember", "recall", "told", "said", "mentioned", 
                       "previous", "before", "earlier", "yesterday", "last time"]
        
        memory_term_count = sum(1 for term in memory_terms if term in query_lower)
        
        # Only augment if there are multiple memory-related terms
        return memory_term_count >= 2
    
    async def _search_memory_for_context(self, query: str) -> Dict[str, Any]:
        """Search memory for relevant context to add to a response.
        
        Args:
            query: User query text
            
        Returns:
            Dict with memory context if found
        """
        # Extract potential topics from the query - more selectively
        topics = self._extract_potential_topics(query)
        
        for topic in topics:
            # Skip very general topics
            if len(topic) <= 4 or topic.lower() in ["this", "that", "these", "those", "thing", "something"]:
                continue
                
            search_result = await self.memory_module.integration.answer_from_memory(topic)
            memory_text, found = search_result
            
            if found:
                # Calculate confidence based on relevance
                confidence = self._calculate_memory_confidence(memory_text, query)
                
                return {
                    "has_memory_context": True,
                    "memory_response": memory_text,
                    "memory_source": "implicit",
                    "memory_topic": topic,
                    "confidence": confidence
                }
                
        return {"has_memory_context": False}
    
    def _extract_potential_topics(self, text: str) -> List[str]:
        """Extract potential topics from text for memory lookup.
        More selective to focus on truly relevant topics.
        
        Args:
            text: Input text to extract topics from
            
        Returns:
            List of potential topics
        """
        # Focus on extracting stronger topic indicators
        noun_phrases = []
        
        # Find capitalized words (proper nouns) - preserve original case
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
        if proper_nouns:
            noun_phrases.extend(proper_nouns)
        
        # Find important technical terms and topics - more selective list
        important_terms = [
            "philosophy", "consciousness", 
            "memory", "reality", "truth", "knowledge",
            "existence", "identity", "self-awareness",
            "meaning", "purpose"
        ]
        
        # Extract these terms specifically with case preserved
        for term in important_terms:
            term_matches = re.findall(rf'\b{term}\w*\b', text, re.IGNORECASE)
            if term_matches:
                noun_phrases.extend(term_matches)
            
        # Find noun phrases using more selective patterns
        np_matches = re.findall(r'\b(?:my|your)\s+([a-z]+(?:\s+[a-z]+){0,2})\b', text.lower())
        if np_matches:
            noun_phrases.extend(np_matches)
            
        # Extract keywords - more selectively
        words = text.lower().split()
        stop_words = {"the", "and", "a", "an", "of", "to", "in", "on", "with", "by", "for", 
                      "is", "are", "was", "were", "be", "am", "has", "have", "had", "do", 
                      "does", "did", "can", "could", "will", "would", "should", "might"}
        keywords = [word for word in words if word not in stop_words and len(word) > 4]
        
        # Combine all potential topics, removing duplicates but preserving case sensitivity
        all_topics = []
        lowercase_topics = set()
        
        # First add proper nouns and important terms with original case
        for topic in noun_phrases:
            if topic.lower() not in lowercase_topics:
                all_topics.append(topic)
                lowercase_topics.add(topic.lower())
        
        # Then add other keywords if not already added
        for topic in keywords:
            if topic.lower() not in lowercase_topics:
                all_topics.append(topic)
                lowercase_topics.add(topic.lower())
                
        return all_topics[:3]  # More limited - only top 3 topics
        
    def _is_memory_relevant_to_query(self, memory: str, query: str) -> bool:
        """Determine if a memory is truly relevant to the query.
        
        Args:
            memory: The memory text
            query: User query text
            
        Returns:
            Boolean indicating relevance
        """
        # Convert to lowercase for comparison
        memory_lower = memory.lower()
        query_lower = query.lower()
        
        # Extract keywords from query
        query_words = query_lower.split()
        stop_words = {"the", "and", "a", "an", "of", "to", "in", "on", "with", "by", "for", 
                     "is", "are", "was", "were", "be", "am", "has", "have", "had", "do", 
                     "does", "did", "can", "could", "will", "would", "should", "might"}
        
        query_keywords = [word for word in query_words if word not in stop_words and len(word) > 3]
        
        # Count how many significant query keywords appear in memory
        keyword_matches = sum(1 for keyword in query_keywords if keyword in memory_lower)
        keyword_match_ratio = keyword_matches / len(query_keywords) if query_keywords else 0
        
        # Look for question type alignment
        question_types = {
            "who": ["person", "name", "individual"],
            "what": ["thing", "object", "concept", "definition"],
            "when": ["time", "date", "period", "day", "year", "month"],
            "where": ["place", "location", "position", "area"],
            "why": ["reason", "cause", "explanation"],
            "how": ["method", "process", "way", "technique"]
        }
        
        # Check if query and memory align in question type
        query_type = None
        for q_type in question_types:
            if query_lower.startswith(q_type) or f" {q_type} " in query_lower:
                query_type = q_type
                break
                
        type_alignment = False
        if query_type:
            type_keywords = question_types[query_type]
            type_alignment = any(keyword in memory_lower for keyword in type_keywords)
        
        # Check for proper noun matches (stronger indicators)
        query_proper_nouns = set(re.findall(r'\b[A-Z][a-z]+\b', query))
        memory_proper_nouns = set(re.findall(r'\b[A-Z][a-z]+\b', memory))
        
        proper_noun_matches = len(query_proper_nouns.intersection(memory_proper_nouns))
        
        # Combined relevance score
        has_proper_noun_match = proper_noun_matches > 0
        has_good_keyword_match = keyword_match_ratio >= 0.5
        has_type_alignment = type_alignment
        
        # Memory is relevant if it has strong keyword matches or proper noun matches
        return (has_proper_noun_match or has_good_keyword_match) and (has_type_alignment or query_type is None)
        
    def _is_similar_memory(self, memory1: str, memory2: str) -> bool:
        """Check if two memories are similar to avoid repetition.
        
        Args:
            memory1: First memory text
            memory2: Second memory text
            
        Returns:
            Boolean indicating if memories are similar
        """
        # Simple similarity check based on shared words
        memory1_words = set(memory1.lower().split())
        memory2_words = set(memory2.lower().split())
        
        # Remove common words
        stop_words = {"the", "and", "a", "an", "of", "to", "in", "on", "with", "by", "for", 
                     "is", "are", "was", "were", "be", "am", "has", "have", "had", "do", 
                     "does", "did", "can", "could", "will", "would", "should", "might", "i", 
                     "you", "he", "she", "it", "we", "they", "this", "that", "these", "those"}
                     
        memory1_words = {w for w in memory1_words if w not in stop_words and len(w) > 3}
        memory2_words = {w for w in memory2_words if w not in stop_words and len(w) > 3}
        
        # Calculate Jaccard similarity
        if not memory1_words or not memory2_words:
            return False
            
        intersection = len(memory1_words.intersection(memory2_words))
        union = len(memory1_words.union(memory2_words))
        
        similarity = intersection / union if union > 0 else 0
        
        # Memories are similar if they share many significant words
        return similarity > 0.4
        
    def _calculate_memory_confidence(self, memory: str, query: str) -> str:
        """Calculate confidence level of memory relevance.
        
        Args:
            memory: Memory text
            query: User query
            
        Returns:
            Confidence level: "low", "medium", or "high"
        """
        # Convert to lowercase for comparison
        memory_lower = memory.lower()
        query_lower = query.lower()
        
        # Extract keywords from query
        query_words = query_lower.split()
        stop_words = {"the", "and", "a", "an", "of", "to", "in", "on", "with", "by", "for", 
                     "is", "are", "was", "were", "be", "am", "has", "have", "had", "do", 
                     "does", "did", "can", "could", "will", "would", "should", "might"}
        
        query_keywords = [word for word in query_words if word not in stop_words and len(word) > 3]
        
        # Count how many significant query keywords appear in memory
        keyword_matches = sum(1 for keyword in query_keywords if keyword in memory_lower)
        keyword_match_ratio = keyword_matches / len(query_keywords) if query_keywords else 0
        
        # Check for explicit memory-seeking terms
        memory_terms = ["remember", "recall", "told", "said", "mentioned", 
                       "previous", "before", "earlier", "yesterday", "last time"]
        
        has_memory_terms = any(term in query_lower for term in memory_terms)
        
        # Check for proper noun matches (stronger indicators)
        query_proper_nouns = set(re.findall(r'\b[A-Z][a-z]+\b', query))
        memory_proper_nouns = set(re.findall(r'\b[A-Z][a-z]+\b', memory))
        
        proper_noun_matches = len(query_proper_nouns.intersection(memory_proper_nouns))
        
        # Determine confidence level
        if proper_noun_matches > 0 and keyword_match_ratio > 0.6:
            return "high"
        elif keyword_match_ratio > 0.5 or (has_memory_terms and keyword_match_ratio > 0.3):
            return "medium"
        else:
            return "low"