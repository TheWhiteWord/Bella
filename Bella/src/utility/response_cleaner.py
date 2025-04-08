"""Response cleaner utility for Bella voice assistant.

This module provides functions to clean LLM responses before they're sent to the TTS system,
removing system messages, tool call acknowledgments, and other text that shouldn't be spoken.
"""

import re
from typing import Optional

def clean_response_for_tts(response: str) -> str:
    """Clean an LLM response to make it suitable for text-to-speech output.
    
    This function removes:
    - Tool call acknowledgments
    - System messages about functions
    - Markdown formatting artifacts
    - Quoted text (when appearing with unquoted text)
    
    Args:
        response: Raw response from LLM
        
    Returns:
        Cleaned response suitable for TTS
    """
    if not response:
        return ""
        
    # Split the response into lines to process separately
    lines = response.split('\n')
    
    # Pattern for lines that should be removed entirely
    system_patterns = [
        r"^No function is directly applicable",
        r"^I see what happened there",
        r"^That's an interesting message from our system",
        r"^It seems like we don't have any prior",
        r"^Seems like a blank slate",
        r"^I'm executing the",
        r"^I'll use the",
        r"I will now use the",
        r"^Using the",
        r"^Calling function",
        r"^Let me search",
        r"^I'm going to",
        r"^Let me try again",
        r"^I'll run",
        r"Let me check",
        r"I need to",
        r"First, I'll",
        r"Now I'll",
    ]
    
    # Combine patterns
    combined_pattern = "|".join(system_patterns)
    
    # Process each line
    cleaned_lines = []
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
            
        # Skip lines matching system patterns
        if re.search(combined_pattern, line):
            continue
            
        # Keep this line
        cleaned_lines.append(line)
    
    # Extract quoted text when it appears to be the main response
    quoted_text = extract_quoted_response(response)
    if quoted_text:
        return quoted_text
    
    # Join remaining lines
    cleaned = "\n".join(cleaned_lines)
    
    # Additional cleaning to remove markdown artifacts
    cleaned = re.sub(r'\*\*', '', cleaned)  # Remove bold markers
    cleaned = re.sub(r'\*', '', cleaned)    # Remove italic markers
    cleaned = re.sub(r'`', '', cleaned)     # Remove code markers
    
    return cleaned.strip()

def extract_quoted_response(response: str) -> Optional[str]:
    """Extract text inside quotes if it appears to be the intended response.
    
    LLMs often use quotes to indicate the actual response vs. their thinking.
    
    Args:
        response: Raw text from LLM
        
    Returns:
        Quoted text if found, None otherwise
    """
    # Look for text in quotes (both single and double quotes)
    double_quotes = re.findall(r'"([^"]*)"', response)
    single_quotes = re.findall(r"'([^']*)'", response)
    
    all_quotes = double_quotes + single_quotes
    
    # If we found quoted text and it looks substantial enough
    if all_quotes and any(len(quote) > 10 for quote in all_quotes):
        # Join multiple quotes with spaces
        return " ".join(all_quotes)
    
    return None
