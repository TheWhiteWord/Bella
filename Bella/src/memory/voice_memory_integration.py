"""Integration between voice assistant and memory system.

This module handles the interaction between the voice interface and memory storage.
"""

import asyncio
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .memory_api import write_note, read_note, search_notes, create_memory_from_conversation

class VoiceMemoryIntegration:
    """Integration between voice assistant and memory system."""
    
    def __init__(self):
        """Initialize voice memory integration."""
        self.conversation_buffer = []  # Store recent conversation turns
        self.max_buffer_size = 20      # Maximum number of turns to store
        self.conversation_topic = None # Current conversation topic
        self.importance_threshold = 0.7 # Threshold for fact importance (increased from default)
        self.fact_patterns = [
            # More specific and selective fact patterns
            r"(?:my name is|I am called) ([A-Z][a-z]+)",
            r"(?:I|my|we) (?:prefer|like|love|enjoy|favorite) (?:is )?(.{3,40})",
            r"(?:I|my|we) (?:dislike|hate|don't like|cannot stand) (?:is )?(.{3,40})",
            r"(?:always|usually|typically|generally) ([^.,;!?]{5,60})",
            r"important (?:to|for) (?:me|us) (?:is|are) ([^.,;!?]{5,60})",
            r"(?:I am|I'm) (?:a|an) ([^.,;!?]{3,30})(?: by profession| for living)?",
            r"(?:I have|I've) (?:a|an) ([^.,;!?]{3,30})",
            r"(?:remember|note|don't forget) that ([^.,;!?]{5,100})"
        ]
        
    def add_to_conversation_buffer(self, user_text: str, assistant_text: str) -> None:
        """Add a conversation turn to the buffer.
        
        Args:
            user_text: User's input text
            assistant_text: Assistant's response text
        """
        self.conversation_buffer.append({"user": user_text, "assistant": assistant_text})
        
        # Keep buffer at a reasonable size
        if len(self.conversation_buffer) > self.max_buffer_size:
            self.conversation_buffer = self.conversation_buffer[-self.max_buffer_size:]
            
    def clear_conversation_buffer(self) -> None:
        """Clear the conversation buffer."""
        self.conversation_buffer = []
        self.conversation_topic = None
        
    async def save_current_conversation(self, title: str = None) -> Dict[str, Any]:
        """Save the current conversation buffer to memory.
        
        Args:
            title: Optional title for the conversation memory
            
        Returns:
            Dict with operation results
        """
        if not self.conversation_buffer:
            return {"error": "No conversation to save"}
            
        # Format conversation for storage
        conversation = []
        for turn in self.conversation_buffer:
            conversation.append(turn["user"])
            conversation.append(turn["assistant"])
            
        # Generate title if not provided
        if not title:
            # Extract meaningful title from conversation
            topic_hint = self.conversation_topic or self._extract_conversation_topic(conversation)
            timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
            title = f"{topic_hint or 'Conversation'}-on-{timestamp}"
            
        # Actually save conversation
        try:
            result = await create_memory_from_conversation(
                conversation=conversation,
                title=title,
                topic=self.conversation_topic,
                folder="conversations"
            )
            return result
        except Exception as e:
            return {"error": f"Failed to save conversation: {str(e)}"}
            
    async def answer_from_memory(self, query: str) -> Tuple[str, bool]:
        """Attempt to answer a question from stored memories.
        
        Args:
            query: The question or topic to search for
            
        Returns:
            Tuple of (response text, whether answer was found)
        """
        try:
            # Search for relevant memories
            search_result = await search_notes(query)
            
            if not search_result or not search_result.get("primary_results"):
                return f"I don't have any memories about {query}", False
                
            # Get the most relevant result
            top_result = search_result["primary_results"][0]
            memory_path = top_result.get("path")
            
            if not memory_path:
                return f"I found something about {query}, but couldn't retrieve it", False
                
            # Read the memory content
            memory_content = await read_note(memory_path)
            
            if not memory_content:
                return f"I found a memory about {query} but couldn't read it", False
                
            # Format the response
            memory_title = top_result.get("title", "this topic")
            response = self._format_memory_response(memory_content, memory_title, query)
            
            return response, True
            
        except Exception as e:
            return f"I tried to check my memory about {query}, but encountered an error: {str(e)}", False
            
    async def extract_and_save_fact(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and save important facts from text.
        
        Args:
            text: Text to analyze for important facts
            
        Returns:
            Dict with saved fact info if successful, None otherwise
        """
        # Check if text contains explicit memory command
        if re.search(r"^(?:remember|note|save|store|keep in mind) that", text.lower()):
            # This is an explicit memory command
            return await self._save_explicit_fact(text)
            
        # Extract implicit facts (more conservative approach)
        extracted_fact = self._extract_implicit_fact(text)
        if not extracted_fact:
            return None
            
        fact_text, importance = extracted_fact
        
        # Only save if important enough (higher threshold)
        if importance < self.importance_threshold:
            return None
            
        # Create a title from the fact
        title = self._create_fact_title(fact_text)
        
        # Format the content
        content = f"""# {title}

## Fact
{fact_text}

## Metadata
- Importance: {importance:.2f}
- Source: Extracted from conversation
- [datetime] Extracted on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Relations
- type [[Fact]]
"""

        if self.conversation_topic:
            content += f"- about [[{self.conversation_topic}]]\n"
            
        # Save to memory system
        try:
            result = await write_note(
                title=title,
                content=content,
                folder="facts",
                tags=["fact", self.conversation_topic] if self.conversation_topic else ["fact"],
                verbose=True
            )
            return result
        except Exception as e:
            return None
            
    async def set_conversation_topic(self, topic: str) -> None:
        """Set the current conversation topic.
        
        Args:
            topic: Topic name
        """
        self.conversation_topic = topic
        
    def _format_memory_response(self, memory_content: str, title: str, query: str) -> str:
        """Format memory content for response.
        
        Args:
            memory_content: Raw memory content
            title: Memory title
            query: Original query
            
        Returns:
            Formatted response text
        """
        # Extract the most relevant section
        relevant_section = self._extract_relevant_section(memory_content, query)
        
        # Format the response
        if len(relevant_section) > 200:
            # Truncate if too long
            relevant_section = relevant_section[:200] + "..."
            
        response = f"About {title}: {relevant_section}"
        return response
        
    def _extract_relevant_section(self, content: str, query: str) -> str:
        """Extract the most relevant section from memory content.
        
        Args:
            content: Memory content
            query: Search query
            
        Returns:
            Relevant section text
        """
        # Remove frontmatter if present
        if content.startswith("---"):
            content = re.sub(r"---.*?---", "", content, flags=re.DOTALL).strip()
            
        # Split into sections
        sections = re.split(r"##\s+", content)
        
        # Process each section to find most relevant
        best_section = ""
        best_score = 0
        
        query_words = set(query.lower().split())
        
        for section in sections:
            if not section.strip():
                continue
                
            # Calculate relevance score
            section_words = set(section.lower().split())
            common_words = query_words.intersection(section_words)
            score = len(common_words) / len(query_words) if query_words else 0
            
            # Boost score for sections with specific headers
            if section.lower().startswith(("fact", "information", "detail", "observation")):
                score *= 1.5
                
            if score > best_score:
                best_score = score
                # Extract just the content, removing headers
                lines = section.split("\n")
                if lines:
                    header = lines[0]
                    content = "\n".join(lines[1:]).strip()
                    best_section = content
                    
        if best_section:
            return best_section
            
        # If no good section found, just return first non-header text
        content_no_headers = re.sub(r"#\s+.*?\n", "", content)
        first_para = next(filter(None, content_no_headers.split("\n\n")), "")
        return first_para[:200]  # Limit length
        
    def _extract_conversation_topic(self, conversation: List[str]) -> str:
        """Extract the main topic from conversation.
        
        Args:
            conversation: List of conversation turns
            
        Returns:
            Extracted topic or default
        """
        # Analyze recent messages to identify topic
        joined_text = " ".join(conversation[-4:])  # Look at recent messages
        
        # Look for topic indicators
        topic_indicators = [
            r"talk(?:ing)? about ([a-zA-Z ]{3,25})",
            r"discuss(?:ing)? ([a-zA-Z ]{3,25})",
            r"conversation about ([a-zA-Z ]{3,25})"
        ]
        
        for pattern in topic_indicators:
            match = re.search(pattern, joined_text, re.IGNORECASE)
            if match:
                return match.group(1).strip().title()
                
        # Extract noun phrases as potential topics using part-of-speech patterns
        noun_phrases = re.findall(r'\b(?:the |a |an )?([A-Z][a-z]+(?:\s+[a-z]+){0,2})\b', joined_text)
        if noun_phrases:
            return noun_phrases[0]
            
        # Extract most frequent significant words
        words = re.findall(r'\b[a-zA-Z]{4,}\b', joined_text.lower())
        if not words:
            return "Conversation"
            
        # Filter common words
        common_words = {"what", "where", "when", "which", "this", "that", "these", "those", 
                       "have", "does", "like", "about", "there", "their", "would", "could", 
                       "should", "because", "thanks", "please", "hello", "goodbye"}
        words = [w for w in words if w not in common_words]
        
        # Get most frequent
        if words:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
                
            top_word = max(word_counts.items(), key=lambda x: x[1])[0]
            return top_word.title()
            
        return "Conversation"
        
    async def _save_explicit_fact(self, text: str) -> Optional[Dict[str, Any]]:
        """Save an explicitly stated fact.
        
        Args:
            text: Text containing the fact
            
        Returns:
            Dict with saved fact info if successful, None otherwise
        """
        # Extract the fact part (after "remember that" or similar phrase)
        match = re.search(r"^(?:remember|note|save|store|keep in mind) that (.+)", text.lower())
        if not match:
            return None
            
        fact_text = match.group(1).strip()
        if not fact_text:
            return None
            
        # Capitalize first letter of fact
        fact_text = fact_text[0].upper() + fact_text[1:]
        
        # Create a title from the fact
        title = self._create_fact_title(fact_text)
        
        # Format the content with proper markdown structure
        content = f"""# {title}

## Fact
{fact_text}

## Metadata
- Importance: 1.0
- Source: Explicit memory command
- [datetime] Recorded on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Observations
- [explicit] User explicitly asked to remember this fact

## Relations
- type [[Fact]]
"""

        if self.conversation_topic:
            content += f"- about [[{self.conversation_topic}]]\n"
            
        # Save to memory system
        try:
            result = await write_note(
                title=title,
                content=content,
                folder="facts",
                tags=["fact", "explicit", self.conversation_topic] if self.conversation_topic else ["fact", "explicit"],
                verbose=True
            )
            return result
        except Exception as e:
            return None
            
    def _extract_implicit_fact(self, text: str) -> Optional[Tuple[str, float]]:
        """Extract implicit facts from text with importance score.
        
        Args:
            text: Text to extract facts from
            
        Returns:
            Tuple of (fact text, importance score) or None if no fact found
        """
        # Check against fact patterns
        for pattern in self.fact_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Calculate importance based on pattern confidence and specificity
                base_importance = 0.6  # Base importance for pattern matches
                
                # Adjust importance based on various factors
                importance = base_importance
                
                # More specific patterns get higher importance
                if "important" in pattern or "remember" in pattern:
                    importance += 0.2
                    
                # Longer extracted facts may be more important
                extracted_fact = match.group(1).strip()
                if len(extracted_fact) > 15:
                    importance += 0.1
                    
                # Proper nouns suggest important entities
                if re.search(r'\b[A-Z][a-z]+\b', extracted_fact):
                    importance += 0.1
                    
                # First person statements can be more important
                if re.search(r'\b(I|my|we|our)\b', text.lower()):
                    importance += 0.1
                    
                # Skip very short or vague facts
                if len(extracted_fact) < 5 or extracted_fact.lower() in ["this", "that", "it", "something", "things"]:
                    continue
                    
                # Form proper sentence for the fact
                if not extracted_fact.endswith((".", "!", "?")):
                    extracted_fact += "."
                    
                return extracted_fact, min(importance, 1.0)  # Cap at 1.0
                
        return None
        
    def _create_fact_title(self, fact_text: str) -> str:
        """Create a title from fact text.
        
        Args:
            fact_text: The fact text
            
        Returns:
            Title for the fact
        """
        # Clean up the text
        clean_text = re.sub(r'[^\w\s]', '', fact_text.lower())
        
        # Take first few words (up to 8) for the title
        words = clean_text.split()
        title_words = words[:min(8, len(words))]
        title = "-".join(title_words)
        
        # Ensure reasonable length (max 50 chars)
        if len(title) > 50:
            title = title[:50]
            
        return title