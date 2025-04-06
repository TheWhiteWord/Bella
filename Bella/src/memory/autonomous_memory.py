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
        self.memory_threshold = 0.65  # Threshold for memory relevance
        self.last_memory_check = datetime.now()
        self.memory_check_interval = 5  # Seconds between memory checks
        self.memory_context = []  # Currently active memory context
        
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
        # Check if query is likely seeking information from memory
        if self._is_knowledge_seeking_query(query):
            # Try to answer from memory
            memory_answer, found = await self.memory_module.query_memory(query)
            
            if found:
                return {
                    "has_memory_context": True,
                    "memory_response": memory_answer,
                    "memory_source": "direct_query"
                }
            
        # Check if we should augment with any relevant memories even if not explicitly asked
        if self._should_augment_with_memory(query):
            search_result = await self._search_memory_for_context(query)
            if search_result and search_result.get("has_memory_context"):
                return search_result
        
        return {"has_memory_context": False}
    
    def _is_knowledge_seeking_query(self, query: str) -> bool:
        """Check if a query is seeking information that might be in memory.
        
        Args:
            query: User query text
            
        Returns:
            Boolean indicating if query is seeking stored knowledge
        """
        # Patterns that suggest information retrieval
        knowledge_patterns = [
            r"what (do you|did I)? (know|remember) about",
            r"(do you|can you) (know|remember|recall)",
            r"tell me about",
            r"do you have information (on|about)",
            r"what (is|was)",
            r"who (is|was)",
            r"when (is|was|did)",
            r"where (is|was)",
            r"why (is|was|did)",
            r"how (is|was|do|does)"
        ]
        
        query_lower = query.lower()
        
        for pattern in knowledge_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # Also check for entities (proper nouns) that might be in memory
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', query)
        if proper_nouns:
            return True
            
        return False
    
    def _should_augment_with_memory(self, query: str) -> bool:
        """Check if we should augment the context with memory even if not explicitly asked.
        
        Args:
            query: User query text
            
        Returns:
            Boolean indicating if we should add memory context
        """
        # Special handling for explicit memory-seeking queries - always check
        query_lower = query.lower()
        
        # Check for direct memory requests
        if "remember" in query_lower or "recall" in query_lower:
            return True
        
        # Check for opinion/preference questions
        opinion_patterns = [
            r"what('s| is) (your|my) (opinion|thought|view|preference)",
            r"(opinion|thoughts?) (about|on|of)",
            r"(do you|would you) (like|prefer|enjoy|recommend)",
            r"(what do you|how do you) (think|feel) (about|of)",
            r"favorite",
        ]
        
        for pattern in opinion_patterns:
            if re.search(pattern, query_lower):
                return True
            
        # Check for time/schedule-related queries that likely need memory
        schedule_patterns = [
            r"when is (my|the|our) (next|upcoming)",
            r"(appointment|meeting|event|birthday|anniversary|deadline)",
            r"(schedule|calendar|agenda|plan)",
            r"what (time|day|date)",
            r"remind me",
        ]
        
        for pattern in schedule_patterns:
            if re.search(pattern, query_lower):
                return True
            
        # Only check for memory augmentation periodically for other queries
        now = datetime.now()
        if (now - self.last_memory_check).total_seconds() < self.memory_check_interval:
            return False
            
        self.last_memory_check = now
        
        # Simple rules for now - check for personal topics that would benefit from memory
        personal_topics = [
            "preference", "like", "favorite", "opinion",
            "told", "mentioned", "said",
            "yesterday", "last time", "previously"
        ]
        
        # First check for direct memory-seeking patterns
        if self._is_knowledge_seeking_query(query):
            return True
        
        # Then check for personal topics
        for topic in personal_topics:
            if topic in query_lower:
                return True
                
        return False
    
    async def _search_memory_for_context(self, query: str) -> Dict[str, Any]:
        """Search memory for relevant context to add to a response.
        
        Args:
            query: User query text
            
        Returns:
            Dict with memory context if found
        """
        # Extract potential topics from the query
        topics = self._extract_potential_topics(query)
        
        for topic in topics:
            search_result = await self.memory_module.integration.answer_from_memory(topic)
            memory_text, found = search_result
            
            if found:
                return {
                    "has_memory_context": True,
                    "memory_response": memory_text,
                    "memory_source": "implicit",
                    "memory_topic": topic
                }
                
        return {"has_memory_context": False}
    
    def _extract_potential_topics(self, text: str) -> List[str]:
        """Extract potential topics from text for memory lookup.
        
        Args:
            text: Input text to extract topics from
            
        Returns:
            List of potential topics
        """
        # Simple noun phrase extraction
        noun_phrases = []
        
        # Find capitalized words (proper nouns) - preserve original case
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
        if proper_nouns:
            noun_phrases.extend(proper_nouns)
        
        # Find important technical terms and topics
        important_terms = ["programming", "computer", "science", "language", 
                          "ai", "intelligence", "system", "application",
                          "project", "development", "design", "algorithm", 
                          "art", "artist", "philosophy", "consciousness", "self"
                          "memory", "reality", "truth", "knowledge", "information",
                          "experience", "emotion", "feeling", "thought",
                          "perception", "awareness", "understanding", "belief",
                          "existence", "being", "identity", "self-awareness",
                          "subjectivity", "objectivity", "interpretation",
                          "meaning", "significance", "value", "purpose",
                          "intention", "action", "decision", "choice",
                          "behavior", "interaction", "communication",
                          "relationship", "connection", "network", "system",
                          "environment", "context", "situation", "condition",]
        
        # Extract these terms specifically with case preserved
        for term in important_terms:
            term_matches = re.findall(rf'\b{term}\w*\b', text, re.IGNORECASE)
            if term_matches:
                noun_phrases.extend(term_matches)
            
        # Find noun phrases using simple patterns
        np_matches = re.findall(r'\b(?:the|my|your|his|her|their|our|a|an)\s+([a-z]+(?:\s+[a-z]+){0,2})\b', text.lower())
        if np_matches:
            noun_phrases.extend(np_matches)
            
        # Extract keywords
        words = text.lower().split()
        stop_words = {"the", "and", "a", "an", "of", "to", "in", "on", "with", "by", "for", "is", "are", "was", "were", "be", "am"}
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
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
                
        return all_topics[:5]  # Limit to top 5 topics