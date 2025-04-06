"""Voice command handlers for memory operations.

Provides functions to detect and handle memory-related voice commands.
"""

import asyncio
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .memory_api import (
    write_note,
    read_note,
    search_notes,
    build_context,
    recent_activity,
    update_note,
    delete_note
)

async def detect_and_handle_memory_command(
    text: str, 
    conversation_history: List[str] = None
) -> Tuple[bool, Optional[str]]:
    """Detect and handle memory-related commands in user text.
    
    Args:
        text: User's input text
        conversation_history: Optional conversation history
        
    Returns:
        Tuple of (was_command, response_text)
    """
    text_lower = text.lower()
    
    # Command patterns
    remember_patterns = [
        r"remember (this|that)",
        r"make a note (of|about) (this|that)",
        r"save (this|that) (to memory|in your memory)",
        r"save this (to|in) (your )?memory",
        r"remember that (.*)",
        r"save this conversation",
        r"take a note"
    ]
    
    recall_patterns = [
        r"what do you (remember|know) about (.*)",
        r"do you remember (.*)",
        r"recall (.*)",
        r"look up (.*) in your memory",
        r"get memory (about|for) (.*)"
    ]
    
    # Check remember commands
    for pattern in remember_patterns:
        if re.search(pattern, text_lower):
            response = await handle_remember_command(text, conversation_history)
            return True, response
            
    # Check recall commands
    for pattern in recall_patterns:
        match = re.search(pattern, text_lower)
        if match:
            # Extract what to recall
            if len(match.groups()) > 0:
                topic = match.groups()[-1]  # Get the last capture group
                response = await handle_recall_command(topic)
                return True, response
    
    # Not a memory command
    return False, None

async def handle_remember_command(
    text: str, 
    conversation_history: List[str] = None
) -> str:
    """Handle commands to remember information.
    
    Args:
        text: User's command text
        conversation_history: Optional conversation history
        
    Returns:
        Response text
    """
    # Extract what to remember
    memory_content = ""
    title = ""
    folder = "general"
    
    # Try to extract a title and content from the command
    remember_match = re.search(r"[Rr]emember that (.*)", text)
    note_match = re.search(r"[Mm]ake a note (of|about) (.*)", text)
    
    if remember_match:
        # Direct fact to remember - preserve original case
        fact = remember_match.group(1)
        title = fact[:30] + "..." if len(fact) > 30 else fact
        memory_content = f"# {title}\n\n## Information\n\n{fact}\n\n## Observations\n\n- [fact] {fact} #memory\n\n## Relations\n\n- type [[Fact]]"
        folder = "facts"
        
    elif note_match:
        # Note to make - preserve original case
        note_content = note_match.group(2)
        title = note_content[:30] + "..." if len(note_content) > 30 else note_content
        memory_content = f"# {title}\n\n## Content\n\n{note_content}\n\n## Observations\n\n- [note] {note_content} #note\n\n## Relations\n\n- type [[Note]]"
        folder = "general"
        
    elif "save this conversation" in text.lower() and conversation_history:
        # Save the conversation
        title = "Conversation " + datetime.now().strftime("%Y-%m-%d %H:%M")
        
        memory_content = f"# {title}\n\n## Conversation\n\n"
        
        # Format conversation history
        for i, message in enumerate(conversation_history[-10:]):  # Last 5 turns
            role = "User" if i % 2 == 0 else "Assistant"
            memory_content += f"**{role}**: {message}\n\n"
            
        memory_content += f"\n## Observations\n\n"
        memory_content += f"- [conversation] Saved conversation on {datetime.now().strftime('%Y-%m-%d')} #conversation\n"
        memory_content += f"\n## Relations\n\n"
        memory_content += f"- type [[Conversation]]\n"
        
        folder = "conversations"
        
    else:
        # Generic save for text without specific pattern
        title = "Memory note " + datetime.now().strftime("%Y-%m-%d %H:%M")
        memory_content = f"# {title}\n\n## Content\n\n{text}\n\n## Observations\n\n- [note] Saved from voice command #memory\n\n## Relations\n\n- type [[Note]]"
    
    # Create the memory
    try:
        result = await write_note(
            title=title,
            content=memory_content,
            folder=folder,
            tags=["memory", "voice"],
            verbose=False
        )
        
        if 'error' in result:
            return f"I had trouble saving that to memory: {result['error']}"
            
        return f"I've saved that to memory as '{title}'"
    except Exception as e:
        return f"I encountered an error saving to memory: {str(e)}"

async def handle_recall_command(topic: str) -> str:
    """Handle commands to recall information from memory.
    
    Args:
        topic: Topic to recall information about
        
    Returns:
        Response text
    """
    try:
        # Search for the topic - try both lowercase and capitalized versions
        search_result = await search_notes(topic)
        
        # If no results with original case, try with capitalized first letter
        if not search_result or not search_result.get('primary_results') or not search_result['primary_results']:
            capitalized_topic = topic.capitalize()
            if capitalized_topic != topic:
                search_result = await search_notes(capitalized_topic)
        
        # Try with all words capitalized as a last resort
        if not search_result or not search_result.get('primary_results') or not search_result['primary_results']:
            title_case_topic = ' '.join(word.capitalize() for word in topic.split())
            if title_case_topic != topic and title_case_topic != capitalized_topic:
                search_result = await search_notes(title_case_topic)
        
        # If still no results, return not found message
        if not search_result or not search_result.get('primary_results') or not search_result['primary_results']:
            # One more attempt - try to extract keywords and search
            words = topic.lower().split()
            # Skip common words
            skip_words = {"what", "who", "when", "where", "how", "why", "is", "are", "was", "were", "the", "a", "an", "about", "do", "you", "your"}
            keywords = [w for w in words if w not in skip_words and len(w) > 2]
            
            if keywords:
                # Try searching with the most specific keyword
                for keyword in sorted(keywords, key=len, reverse=True):
                    keyword_result = await search_notes(keyword.capitalize())
                    if keyword_result and keyword_result.get('primary_results') and keyword_result['primary_results']:
                        search_result = keyword_result
                        break
        
        # If still no results after all attempts
        if not search_result or not search_result.get('primary_results') or not search_result['primary_results']:
            return f"I don't have any memories about {topic}."
            
        # Get the top result
        top_result = search_result['primary_results'][0]
        memory_content = await read_note(top_result['title'])
        
        if not memory_content:
            return f"I found something about {topic}, but couldn't retrieve the details."
        
        # Extract the most relevant information
        content_without_frontmatter = re.sub(r'^---\n.*?\n---\n', '', memory_content, flags=re.DOTALL)
        
        # Look for observations
        observations = []
        observation_pattern = r'- \[(.*?)\](.*?)(?=\n|$)'
        for match in re.finditer(observation_pattern, content_without_frontmatter):
            category = match.group(1).strip()
            text = match.group(2).strip()
            # Remove hashtags
            text = re.sub(r'#\w+', '', text).strip()
            observations.append(f"{text}")
        
        # Start building response
        response = f"About {topic}, I remember: "
        
        if observations:
            response += "; ".join(observations[:3])
            if len(observations) > 3:
                response += f"; and {len(observations) - 3} more observations."
        else:
            # Extract a snippet from the content
            lines = content_without_frontmatter.split('\n')
            content_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
            if content_lines:
                response = f"About {topic}, I found: " + " ".join(content_lines[:3])
            else:
                response = f"I have information about {topic}, but it's not very detailed."
        
        # Add related information
        try:
            context_result = await build_context(top_result['title'], depth=1)
            if context_result and 'related' in context_result and context_result['related']:
                related_items = [r['title'] for r in context_result['related'][:3]]
                response += f" This is related to: {', '.join(related_items)}."
        except Exception:
            # Skip related information if error
            pass
        
        return response
    except Exception as e:
        return f"I had trouble accessing my memory: {str(e)}"