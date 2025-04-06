"""Integration of memory system with the main application.

This module provides a simplified interface for integrating the autonomous memory system
with the main application, handling pre-processing inputs and post-processing responses.
"""

import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple

from .memory_conversation_adapter import MemoryConversationAdapter

class MainAppMemoryAdapter:
    """Adapter to integrate memory system with the main application."""
    
    def __init__(self):
        """Initialize main app memory adapter."""
        self.memory_adapter = MemoryConversationAdapter()
        self.conversation_history = []  # Store conversation history for context
        
    async def pre_process_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input before generating a response.
        
        Args:
            user_input: User's input text
            
        Returns:
            Dict with memory context to add
        """
        # Get conversation history in the format required by the memory adapter
        formatted_history = self._format_conversation_history()
        
        # Use the memory adapter to pre-process input
        return await self.memory_adapter.pre_process_input(user_input, formatted_history)
        
    async def post_process_response(self, user_input: str, response: str) -> str:
        """Process response after generation to potentially add memory information.
        
        Args:
            user_input: Original user input
            response: Generated response text
            
        Returns:
            Potentially modified response with memory information
        """
        # Use memory adapter to post-process response
        return await self.memory_adapter.post_process_response(user_input, response)
        
    def add_to_history(self, user_input: str, response: str) -> None:
        """Add a conversation turn to the history.
        
        Args:
            user_input: User's input text
            response: Assistant's response text
        """
        self.conversation_history.append({
            "user": user_input,
            "assistant": response
        })
        
        # Keep history at a reasonable size (last 10 turns)
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
            
    def build_context_with_memory(self, base_context: str, memory_context: Dict[str, Any]) -> str:
        """Build a context string that includes memory information.
        
        Args:
            base_context: Base system context
            memory_context: Memory context information
            
        Returns:
            Enhanced context string with memory information
        """
        memory_text = memory_context.get("memory_context", "")
        memory_source = memory_context.get("memory_source", "")
        
        if not memory_text:
            return base_context
            
        # Format memory context for inclusion in system prompt
        memory_addition = f"""
I'm recalling relevant information from my memory that may help with this conversation:

{memory_text}

Source: {memory_source if memory_source else 'Memory system'}
"""
        
        if base_context:
            return f"{base_context}\n\n{memory_addition}"
        else:
            return memory_addition
            
    def _format_conversation_history(self) -> List[Dict[str, str]]:
        """Format internal conversation history for memory adapter.
        
        Returns:
            List of message dictionaries with role and content
        """
        formatted = []
        for turn in self.conversation_history:
            formatted.append({"role": "user", "content": turn["user"]})
            formatted.append({"role": "assistant", "content": turn["assistant"]})
        return formatted