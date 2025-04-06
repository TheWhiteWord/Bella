"""Voice memory module for integration with main application.

Provides a unified interface for adding memory capabilities to the voice assistant.
"""

import asyncio
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .memory_api import (
    write_note, 
    read_note,
    search_notes,
    build_context,
    recent_activity
)
from .voice_memory_integration import VoiceMemoryIntegration
from .memory_commands import detect_and_handle_memory_command

class VoiceMemoryModule:
    """Memory module for voice assistant integration."""
    
    def __init__(self):
        """Initialize voice memory module."""
        self.integration = VoiceMemoryIntegration()
        self.should_save_conversation = False
        self.pending_memory_request = None
        
    async def process_input(self, user_input: str, assistant_response: str) -> Optional[str]:
        """Process user input and assistant response for memory operations.
        
        Args:
            user_input: User's input text
            assistant_response: Assistant's response
            
        Returns:
            Optional response about memory operations
        """
        # Add to conversation buffer
        self.integration.add_to_conversation_buffer(user_input, assistant_response)
        
        # Check for memory commands
        is_memory_command, memory_response = await detect_and_handle_memory_command(
            user_input,
            self.integration.conversation_buffer
        )
        
        if is_memory_command:
            return memory_response
            
        # Check for fact extraction (if not a command)
        fact_result = await self.integration.extract_and_save_fact(user_input)
        if fact_result and not fact_result.get('error'):
            return f"I've noted that fact about {fact_result.get('title', 'this topic')}."
            
        # Check for memory request for follow-up
        if self.pending_memory_request:
            if "yes" in user_input.lower() or "sure" in user_input.lower() or "okay" in user_input.lower():
                if self.pending_memory_request == "save_conversation":
                    result = await self.integration.save_current_conversation()
                    self.pending_memory_request = None
                    if not result.get('error'):
                        return f"I've saved our conversation to memory as '{result.get('title')}'."
                    else:
                        return "I had trouble saving our conversation."
            else:
                # Clear pending request on rejection
                self.pending_memory_request = None
                
        # Check if conversation might be worth saving
        if self._should_offer_memory_save():
            self.pending_memory_request = "save_conversation"
            return "This conversation seems important. Would you like me to save it to my memory?"
            
        return None
        
    async def query_memory(self, question: str) -> Tuple[str, bool]:
        """Query memory for information.
        
        Args:
            question: Question to answer from memory
            
        Returns:
            Tuple of (answer text, found_in_memory)
        """
        return await self.integration.answer_from_memory(question)
    
    def _should_offer_memory_save(self) -> bool:
        """Determine if conversation should be offered to save.
        
        Returns:
            True if conversation seems important enough to save
        """
        # Only offer if buffer has substantial content (at least 3 turns)
        if len(self.integration.conversation_buffer) < 6:
            return False
            
        # Check for indicators of important conversation
        important_indicators = [
            "remember", "don't forget", "important", "critical",
            "schedule", "appointment", "meeting", "event",
            "contact", "address", "phone", "email",
            "preference", "like", "dislike", "favorite",
            "username", "password", "account", 
            "birthday", "anniversary", "date"
        ]
        
        conversation_text = " ".join(self.integration.conversation_buffer).lower()
        
        # Count matches
        importance_score = sum(1 for indicator in important_indicators if indicator in conversation_text)
        
        # Return true if score is high enough and we haven't recently offered
        return importance_score >= 2
        
    def get_conversation_summary(self) -> Optional[str]:
        """Generate a summary of the current conversation.
        
        Returns:
            Summary text or None if conversation is too short
        """
        if len(self.integration.conversation_buffer) < 4:
            return None
            
        # Create a simple summary for now
        turns = len(self.integration.conversation_buffer) // 2  # Each turn is user + assistant
        topics = self._extract_conversation_topics()
        
        summary = f"Conversation with {turns} turns"
        if topics:
            summary += f" about {', '.join(topics[:3])}"
        
        return summary
    
    def _extract_conversation_topics(self) -> List[str]:
        """Extract likely topics from conversation.
        
        Returns:
            List of topic words
        """
        # Simple keyword extraction
        if not self.integration.conversation_buffer:
            return []
            
        # Combine all text
        all_text = " ".join(self.integration.conversation_buffer)
        
        # Count word frequencies (excluding stop words)
        words = re.findall(r'\b[a-z]{3,}\b', all_text.lower())
        stop_words = {
            "the", "and", "you", "that", "but", "this", "with", "for",
            "have", "what", "your", "from", "will", "about", "when",
            "would", "like", "think", "could", "know", "just", "should", 
            "very", "some", "them", "they", "there", "here", "also"
        }
        
        word_counts = {}
        for word in words:
            if word not in stop_words:
                word_counts[word] = word_counts.get(word, 0) + 1
                
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top topics
        return [word for word, count in sorted_words[:5] if count > 1]