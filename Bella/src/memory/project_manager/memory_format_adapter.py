"""Memory format adapter for Bella.

This module provides standardization between the autonomous memory system
and the project-based memory system to ensure consistent memory storage formats.
"""

import os
import re
import yaml
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

class MemoryFormatAdapter:
    """Adapter to standardize memory formats between different memory systems."""
    
    @staticmethod
    def convert_to_standard_format(
        content: str, 
        title: str = None,
        memory_type: str = "general",
        tags: List[str] = None,
        source: str = "autonomous"
    ) -> str:
        """Convert memory content to the standardized format.
        
        Args:
            content: Raw memory content
            title: Title for the memory (optional)
            memory_type: Type of memory (e.g., general, facts)
            tags: List of tags for the memory
            source: Source of the memory ('autonomous' or 'function')
            
        Returns:
            Formatted memory content
        """
        # Generate a title if not provided
        if not title:
            words = re.findall(r'\w+', content.lower())
            if len(words) > 5:
                title = "-".join(words[:5])
            else:
                title = f"memory-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Ensure we have tags
        if not tags:
            tags = []
        
        # Add memory type as a tag if not already included
        if memory_type not in tags:
            tags.append(memory_type)
        
        # Extract any hashtags from content and add to tags
        for match in re.finditer(r'#(\w+)', content):
            tag = match.group(1).lower()
            if tag not in tags:
                tags.append(tag)
        
        # Create a UUID for the memory
        entry_id = str(uuid.uuid4())[:10]
        
        # Create frontmatter
        frontmatter = {
            'title': title,
            'created': datetime.now().isoformat(),
            'updated': datetime.now().isoformat(),
            'memory_type': memory_type,
            'tags': tags,
            'source': source,
            'entry_id': entry_id
        }
        
        # Ensure content starts with a title if it doesn't have one
        if not content.strip().startswith('#'):
            content = f"# {title}\n\n{content}"
        
        # Format with frontmatter
        formatted_content = f"---\n{yaml.dump(frontmatter)}---\n\n{content}"
        return formatted_content
    
    @staticmethod
    def extract_standard_format_data(content: str) -> Dict[str, Any]:
        """Extract metadata and content from standardized format.
        
        Args:
            content: Formatted memory content
            
        Returns:
            Dict with metadata and content
        """
        # Extract frontmatter
        frontmatter_match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
        frontmatter = {}
        
        if frontmatter_match:
            frontmatter_yaml = frontmatter_match.group(1)
            try:
                frontmatter = yaml.safe_load(frontmatter_yaml)
            except Exception:
                pass
            
            # Remove frontmatter from content
            content_without_frontmatter = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)
        else:
            content_without_frontmatter = content
        
        # Extract title from content if not in frontmatter
        if 'title' not in frontmatter:
            title_match = re.match(r'^#\s+(.*?)$', content_without_frontmatter, re.MULTILINE)
            if title_match:
                frontmatter['title'] = title_match.group(1).strip()
        
        return {
            'metadata': frontmatter,
            'content': content_without_frontmatter
        }
    
    @staticmethod
    def is_standard_format(content: str) -> bool:
        """Check if the content is in the standardized format.
        
        Args:
            content: Memory content to check
            
        Returns:
            True if the content is in the standardized format, False otherwise
        """
        # Check for frontmatter
        frontmatter_match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
        if not frontmatter_match:
            return False
        
        # Try to parse frontmatter
        frontmatter_yaml = frontmatter_match.group(1)
        try:
            frontmatter = yaml.safe_load(frontmatter_yaml)
            
            # Check for required fields
            required_fields = ['title', 'created', 'tags', 'entry_id']
            for field in required_fields:
                if field not in frontmatter:
                    return False
                    
            return True
        except Exception:
            return False
    
    @staticmethod
    def update_standard_format(content: str, new_content: str) -> str:
        """Update content while maintaining the standardized format.
        
        Args:
            content: Original formatted content
            new_content: New content to incorporate
            
        Returns:
            Updated formatted content
        """
        # Extract data from original content
        data = MemoryFormatAdapter.extract_standard_format_data(content)
        
        # Update metadata
        metadata = data['metadata']
        metadata['updated'] = datetime.now().isoformat()
        
        # Update content by appending (or use new implementation logic here)
        updated_content = f"{data['content']}\n\n{new_content}"
        
        # Reformat with updated metadata and content
        formatted_content = f"---\n{yaml.dump(metadata)}---\n\n{updated_content}"
        return formatted_content