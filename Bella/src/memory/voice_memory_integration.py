"""Integration of memory system with voice assistant.

Provides tools to capture, retrieve, and manage memories through voice interactions.
"""

import asyncio
import json
import re
import nltk
from typing import Dict, List, Any, Optional, Tuple

from .memory_api import (
    write_note,
    read_note,
    search_notes,
    build_context,
    recent_activity,
    update_note,
    delete_note,
    create_memory_from_conversation
)

# Try to initialize NLTK resources safely
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass  # Continue without nltk if download fails

class VoiceMemoryIntegration:
    """Integration between voice assistant and memory system."""
    
    def __init__(self):
        """Initialize voice memory integration."""
        self.conversation_buffer = []
        self.current_topic = None
        
    def add_to_conversation_buffer(self, user_text: str, assistant_text: str) -> None:
        """Add an interaction to the conversation buffer.
        
        Args:
            user_text: User's input text
            assistant_text: Assistant's response text
        """
        self.conversation_buffer.append(user_text)
        self.conversation_buffer.append(assistant_text)
        
        # Keep buffer at a reasonable size
        if len(self.conversation_buffer) > 20:  # Last 10 turns
            self.conversation_buffer = self.conversation_buffer[-20:]
    
    def clear_conversation_buffer(self) -> None:
        """Clear the conversation buffer."""
        self.conversation_buffer = []
        self.current_topic = None
        
    async def save_current_conversation(self, title: str = None) -> Dict[str, Any]:
        """Save current conversation buffer as a memory.
        
        Args:
            title: Optional title for the memory
            
        Returns:
            Dict with operation results
        """
        if not self.conversation_buffer:
            return {"error": "No conversation to save"}
            
        result = await create_memory_from_conversation(
            conversation=self.conversation_buffer,
            title=title,
            topic=self.current_topic,
            folder="conversations"  # Explicitly specify folder
        )
        
        return result
    
    async def answer_from_memory(self, query: str) -> Tuple[str, bool]:
        """Try to answer a question from memory.
        
        Args:
            query: Question or query from user
            
        Returns:
            Tuple of (response text, found_in_memory)
        """
        # Try to find relevant memories
        search_result = await search_notes(query)
        
        if search_result and search_result.get('primary_results') and len(search_result['primary_results']) > 0:
            # Found something relevant
            top_result = search_result['primary_results'][0]
            memory_content = await read_note(top_result['title'])
            
            if memory_content:
                # Format a response from the memory
                response = self._format_memory_response(
                    memory_content, 
                    top_result['title'],
                    query
                )
                return response, True
        
        # Try with normalized versions of the query (e.g., proper capitalization)
        words = query.split()
        if len(words) > 0:
            # Try with first letter capitalized 
            capitalized_query = ' '.join(w.capitalize() if i == 0 or len(w) > 3 else w 
                                        for i, w in enumerate(words))
            
            if capitalized_query != query:
                search_result = await search_notes(capitalized_query)
                
                if search_result and search_result.get('primary_results') and len(search_result['primary_results']) > 0:
                    # Found something with capitalized query
                    top_result = search_result['primary_results'][0]
                    memory_content = await read_note(top_result['title'])
                    
                    if memory_content:
                        response = self._format_memory_response(
                            memory_content,
                            top_result['title'],
                            query
                        )
                        return response, True
                
        return "", False
    
    async def extract_and_save_fact(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract a fact from text and save it to memory.
        
        Args:
            text: Text containing a potential fact
            
        Returns:
            Dict with operation results if a fact was saved
        """
        # Simple fact detection - sentences with factual indicators
        fact_indicators = [
            "is a", "are", "was", "were", "has", "have",
            "I like", "I prefer", "I enjoy", "I want",
            "always", "never", "usually", "often",
            "my favorite", "I remember"
        ]
        
        # Check for fact-like statements
        for indicator in fact_indicators:
            if indicator.lower() in text.lower():
                # Extract the sentence containing the fact
                sentences = re.split(r'[.!?]', text)
                fact_sentence = ""
                
                for sentence in sentences:
                    if indicator.lower() in sentence.lower():
                        fact_sentence = sentence.strip()
                        break
                        
                if fact_sentence:
                    # Create a fact memory
                    category = "preference" if any(p in fact_sentence.lower() for p in 
                                                ["like", "prefer", "favorite", "enjoy", "want"]) else "fact"
                    
                    # Determine topic for the fact
                    topic = self._extract_topic(fact_sentence)
                    
                    # Create content for the fact
                    content = f"# {topic if topic else 'Fact'}\n\n"
                    content += f"## Information\n\n"
                    content += fact_sentence + ".\n\n"
                    
                    # Add observation
                    content += f"## Observations\n\n"
                    content += f"- [{category}] {fact_sentence} #{category}\n"
                    
                    # Add relations
                    content += f"\n## Relations\n\n"
                    if topic:
                        content += f"- about [[{topic}]]\n"
                    content += f"- type [[{category.title()}]]\n"
                    
                    # Create the memory
                    result = await write_note(
                        title=f"{topic if topic else fact_sentence[:30]}",
                        content=content,
                        folder=category+"s",  # "facts" or "preferences" folder
                        tags=[category, topic] if topic else [category],
                        verbose=True
                    )
                    
                    return result
        
        return None
    
    async def set_conversation_topic(self, topic: str) -> None:
        """Set the topic for the current conversation.
        
        Args:
            topic: Topic or subject of conversation
        """
        self.current_topic = topic
    
    def _format_memory_response(self, memory_content: str, title: str, query: str) -> str:
        """Format a response from memory content.
        
        Args:
            memory_content: Raw memory content
            title: Memory title
            query: Original query
            
        Returns:
            Formatted response text
        """
        # Extract most relevant parts
        relevant_section = self._extract_relevant_section(memory_content, query)
        
        # Create a natural-sounding response
        response = f"From my memory about '{title}', "
        
        if relevant_section:
            response += f"I recall that {relevant_section}"
        else:
            # Fall back to a summary if no specific section found
            response += f"I found some information but it's not specifically about your question. "
            response += f"Would you like me to share what I know about {title}?"
            
        return response
    
    def _extract_relevant_section(self, content: str, query: str) -> str:
        """Extract most relevant section from memory content for query.
        
        Args:
            content: Memory content
            query: Search query
            
        Returns:
            Most relevant text section
        """
        query_words = set(query.lower().split())
        
        # Remove markdown frontmatter
        content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)
        
        # Split into sections by headings
        sections = re.split(r'#{1,3}\s+', content)
        
        best_section = ""
        best_score = 0
        
        for section in sections:
            if not section.strip():
                continue
                
            # Count query word matches
            section_lower = section.lower()
            score = sum(1 for word in query_words if word in section_lower)
            
            # Prioritize observation sections
            if "observations" in section_lower:
                score += 2
                
            if score > best_score:
                best_score = score
                best_section = section
        
        if best_section:
            # Extract most relevant sentences
            sentences = re.split(r'[.!?]', best_section)
            relevant_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # Count query word matches in sentence
                score = sum(1 for word in query_words if word in sentence.lower())
                
                if score > 0:
                    relevant_sentences.append(sentence)
            
            # If we found relevant sentences, use them
            if relevant_sentences:
                return ". ".join(relevant_sentences) + "."
                
            # Otherwise return first ~200 chars of best section
            return best_section.strip()[:200] + "..."
            
        return ""
    
    def _extract_topic(self, text: str) -> Optional[str]:
        """Extract a likely topic from text.
        
        Args:
            text: Text to extract topic from
            
        Returns:
            Extracted topic or None
        """
        # Skip common words and prefixes
        skip_words = {
            "i", "me", "my", "mine", "you", "your", "yours", 
            "he", "him", "his", "she", "her", "hers", "it", "its",
            "we", "us", "our", "ours", "they", "them", "their", "theirs",
            "am", "is", "are", "was", "were", "be", "being", "been",
            "have", "has", "had", "do", "does", "did", "a", "an", "the",
            "and", "but", "or", "because", "as", "until", "while",
            "of", "at", "by", "for", "with", "about", "against",
            "between", "into", "through", "during", "before", "after",
            "above", "below", "to", "from", "up", "down", "in", "out",
            "on", "off", "over", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how",
            "all", "any", "both", "each", "few", "more", "most",
            "some", "such", "no", "nor", "not", "only", "own",
            "same", "so", "than", "too", "very", "just", "should",
            "would", "could", "now", "this", "that", "favorite", "like", 
            "prefer", "want", "enjoy", "tell", "know"
        }
        
        # Try to use NLTK for better extraction if available
        try:
            from nltk import pos_tag, word_tokenize
            
            # Tokenize and tag parts of speech
            tokens = word_tokenize(text)
            tagged = pos_tag(tokens)
            
            # Extract noun phrases - look for sequences of adjectives followed by nouns
            noun_phrases = []
            current_phrase = []
            
            for word, tag in tagged:
                word_lower = word.lower()
                # Skip punctuation and stop words
                if not word.isalnum() or word_lower in skip_words:
                    if current_phrase:
                        noun_phrases.append(" ".join(current_phrase))
                        current_phrase = []
                    continue
                    
                # Add adjectives and nouns to current phrase
                if tag.startswith('JJ') or tag.startswith('NN'):
                    current_phrase.append(word)
                # If we hit a non-adj/noun and have a phrase building, complete it
                elif current_phrase:
                    noun_phrases.append(" ".join(current_phrase))
                    current_phrase = []
                    
            # Add any remaining phrase
            if current_phrase:
                noun_phrases.append(" ".join(current_phrase))
            
            # Return the longest noun phrase, or the first if tie
            if noun_phrases:
                return max(noun_phrases, key=len)
                
        except Exception:
            # Fall back to regex-based approach if NLTK fails
            pass
            
        # Fall back: Extract nouns using simple POS heuristics
        words = text.split()
        nouns = []
        
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w\s]', '', word).lower()
            if not clean_word or clean_word in skip_words:
                continue
                
            # Check for noun-like patterns
            is_noun = False
            
            # Capitalized non-first words are likely nouns
            if i > 0 and word[0].isupper():
                is_noun = True
                
            # Words after determiners (a, an, the) are likely nouns
            if i > 0 and words[i-1].lower() in ["a", "an", "the", "my", "your", "his", "her", "their", "our"]:
                is_noun = True
                
            # Words after adjectives might be nouns
            if i > 0 and words[i-1].lower().endswith(("ful", "ous", "ible", "able", "al", "ive", "ent", "ed")):
                is_noun = True
                
            # Words before prepositions might be nouns
            if i < len(words) - 1 and words[i+1].lower() in ["of", "in", "by", "with", "for"]:
                is_noun = True
                
            if is_noun:
                # Get the original form (not lowercased)
                original_word = re.sub(r'[^\w\s]', '', word)
                nouns.append(original_word)
                
        # Return most likely noun phrases (up to 2 consecutive nouns)
        if nouns:
            # If we have sequential nouns, join them
            if len(nouns) > 1:
                # Look for consecutive nouns
                for i in range(len(nouns) - 1):
                    if i < len(words) - 1 and words.index(nouns[i]) + 1 == words.index(nouns[i+1]):
                        return f"{nouns[i]} {nouns[i+1]}"
                
            # Otherwise just return first noun
            return nouns[0]
            
        return None