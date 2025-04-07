"""Voice command handlers for memory operations.

Provides functions to handle memory-related voice commands.
Only the continue_conversation functionality remains, as other memory functions
are now handled by the project-based memory system.
"""

import asyncio
import re
from typing import List, Optional
from datetime import datetime

from .memory_api import (
    read_note,
    search_notes,
    build_context,
    recent_activity
)

async def handle_continue_command(conversation_history: List[str] = None) -> str:
    """Handle the '@agent Continue' command to continue a conversation thread.
    
    This function uses recent conversation history to determine context and
    continues the thought process or conversation, maintaining relevant context.
    
    Args:
        conversation_history: Recent conversation history
        
    Returns:
        Response text that continues the thought or conversation
    """
    if not conversation_history or len(conversation_history) < 2:
        return "I don't have enough context to continue. Could you provide more information about what you'd like me to continue with?"
    
    # Get the most recent conversation exchanges
    recent_exchanges = conversation_history[-6:]  # Last 3 turns (user + assistant)
    
    try:
        # Identify the main topics/concepts in the conversation
        topic_extraction_text = " ".join(recent_exchanges)
        
        # First, check if there are any ongoing topics we can continue
        # Try to find the most recent note that might be relevant
        recent_activity_result = await recent_activity(limit=3)
        
        # If we have recent activity to draw from
        if recent_activity_result and 'recent_notes' in recent_activity_result:
            recent_notes = recent_activity_result['recent_notes']
            
            # Look for contextually relevant notes
            context_found = False
            for note in recent_notes:
                # Get the full content of this note
                note_content = await read_note(note['title'])
                
                # If there's overlap between conversation and this note
                if note_content and any(exchange in note_content for exchange in recent_exchanges[-2:]):
                    # Use this note as context for continuation
                    context_result = await build_context(note['title'], depth=1)
                    
                    if context_result and 'content' in context_result:
                        # Use the context to build a continuation response
                        return f"Continuing our discussion... {context_result['content']}"
                    
                    context_found = True
                    break
            
            if not context_found:
                # If no direct context found, do a general semantic search
                search_result = await search_notes(topic_extraction_text)
                
                if search_result and search_result.get('primary_results'):
                    top_result = search_result['primary_results'][0]
                    memory_content = await read_note(top_result['title'])
                    
                    # Extract relevant parts for continuation
                    return f"Continuing based on what we know... {memory_content[:300]}..."
        
        # If we can't find relevant context, continue based on just the conversation
        last_assistant_msg = recent_exchanges[-2] if len(recent_exchanges) >= 2 else ""
        last_user_msg = recent_exchanges[-1]
        
        # Create a thoughtful continuation based on the last exchange
        return f"To continue our conversation about {last_user_msg.split()[:5]}...\n\nLet me elaborate further on what we were discussing. {last_assistant_msg.split('.')[0] if '.' in last_assistant_msg else ''}..."
        
    except Exception as e:
        return f"I had trouble continuing our conversation: {str(e)}. Could you please clarify what aspect you'd like me to continue with?"