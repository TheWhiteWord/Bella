"""Memory management system for voice assistant.

This module provides functionality for storing, retrieving, and managing
memories in markdown files with semantic relationships.
"""

import os
import re
import yaml
import asyncio
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Set, Tuple

class MemoryManager:
    """Manages voice assistant memory storage and retrieval using markdown files."""
    
    def __init__(self, memory_dir: str = None):
        """Initialize memory manager with specified directory.
        
        Args:
            memory_dir (str, optional): Path to directory for storing memory files.
                If None, creates a default 'memories' directory in the project.
        """
        if memory_dir is None:
            # Default to a 'memories' directory in the project
            self.memory_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "memories")
        else:
            self.memory_dir = memory_dir
            
        # Create memory directory if it doesn't exist
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # Create folders for organization
        for folder in ['conversations', 'facts', 'preferences', 'reminders', 'general']:
            os.makedirs(os.path.join(self.memory_dir, folder), exist_ok=True)
            
        # In-memory cache of entities and their relations
        self._memory_graph = {}
        self._memory_index = {}
        
        # Load existing memories
        self._load_memories()
        
    async def create_memory(self, 
                     title: str, 
                     content: str, 
                     folder: str = "general", 
                     tags: List[str] = None,
                     timestamp: datetime = None,
                     verbose: bool = False) -> Dict[str, Any]:
        """Create a new memory or update existing one.
        
        Args:
            title: Title of the memory
            content: Markdown content of the memory
            folder: Folder to store the memory in
            tags: List of tags for categorization
            timestamp: Custom timestamp (defaults to now)
            verbose: Whether to return detailed parsing results
            
        Returns:
            Dict with operation results including parsed relations
        """
        # Sanitize title and create filename
        safe_title = self._sanitize_filename(title)
        folder_path = os.path.join(self.memory_dir, folder)
        os.makedirs(folder_path, exist_ok=True)
        filepath = os.path.join(folder_path, f"{safe_title}.md")
        
        # Format content with frontmatter if needed
        if not content.strip().startswith('---'):
            # Add frontmatter with metadata
            if timestamp is None:
                timestamp = datetime.now()
                
            frontmatter = {
                'title': title,
                'created': timestamp.isoformat(),
                'updated': datetime.now().isoformat(),
                'type': 'memory',
                'tags': tags or []
            }
            
            content = f"---\n{yaml.dump(frontmatter)}---\n\n{content}"
        
        # Write content to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
            
        # Parse the content and update the memory graph
        parsed_data = self._parse_memory_content(content, title, folder, filepath)
        self._memory_graph[title] = parsed_data
        self._memory_index[title.lower()] = title  # For case-insensitive lookup
        
        result = {
            'title': title,
            'path': filepath,
            'folder': folder,
        }
        
        if verbose:
            result.update({
                'observations': parsed_data.get('observations', []),
                'relations': parsed_data.get('relations', []),
                'tags': parsed_data.get('tags', []),
            })
            
        return result
            
    async def read_memory(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Read a memory by title, path, or permalink.
        
        Args:
            identifier: Title, path, or memory:// URL
            
        Returns:
            Dict containing memory content and metadata, or None if not found
        """
        # Handle memory:// URLs
        if identifier.startswith('memory://'):
            identifier = identifier[len('memory://'):].strip('/')
        
        # Try direct path first
        if os.path.exists(identifier) and identifier.endswith('.md'):
            filepath = identifier
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            folder, filename = os.path.split(filepath)
            folder = os.path.basename(folder)
            title = os.path.splitext(filename)[0]
            
            return {
                'title': title,
                'content': content,
                'path': filepath,
                'folder': folder,
            }
            
        # Try by title
        title_match = self._find_by_title(identifier)
        if title_match:
            for folder in os.listdir(self.memory_dir):
                folder_path = os.path.join(self.memory_dir, folder)
                if os.path.isdir(folder_path):
                    for filename in os.listdir(folder_path):
                        if filename.endswith('.md'):
                            name_without_ext = os.path.splitext(filename)[0]
                            if name_without_ext.lower() == title_match.lower():
                                filepath = os.path.join(folder_path, filename)
                                with open(filepath, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                
                                return {
                                    'title': title_match,
                                    'content': content,
                                    'path': filepath,
                                    'folder': folder,
                                }
                                
        # Try folder/title format
        if '/' in identifier:
            folder, title = identifier.split('/', 1)
            filepath = os.path.join(self.memory_dir, folder, f"{self._sanitize_filename(title)}.md")
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                return {
                    'title': title,
                    'content': content,
                    'path': filepath,
                    'folder': folder,
                }
                
        return None
    
    async def search_memories(self, 
                       query: str, 
                       page: int = 1, 
                       page_size: int = 10) -> Dict[str, Any]:
        """Search through memories by query text.
        
        Args:
            query: Text to search for
            page: Page number for pagination
            page_size: Results per page
            
        Returns:
            Dict with search results and pagination info
        """
        results = []
        query_lower = query.lower()
        
        # Search through all memory files
        for folder in os.listdir(self.memory_dir):
            folder_path = os.path.join(self.memory_dir, folder)
            if not os.path.isdir(folder_path):
                continue
                
            for filename in os.listdir(folder_path):
                if not filename.endswith('.md'):
                    continue
                    
                filepath = os.path.join(folder_path, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Extract title from frontmatter or filename
                title = self._extract_title_from_content(content) or os.path.splitext(filename)[0]
                
                # Check if query matches title or content
                if query_lower in title.lower() or query_lower in content.lower():
                    # Extract a snippet showing the match context
                    content_without_frontmatter = self._remove_frontmatter(content)
                    snippet = self._extract_snippet(content_without_frontmatter, query)
                    
                    results.append({
                        'title': title,
                        'path': filepath,
                        'folder': folder,
                        'snippet': snippet,
                        'score': self._calculate_relevance_score(title, content, query)
                    })
        
        # Sort by relevance score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Paginate results
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        current_page_results = results[start_idx:end_idx]
        
        return {
            'primary_results': current_page_results,
            'total_results': len(results),
            'page': page,
            'page_size': page_size,
            'total_pages': (len(results) + page_size - 1) // page_size
        }
    
    async def build_context(self, 
                     url: str, 
                     depth: int = 2, 
                     timeframe: str = None) -> Dict[str, Any]:
        """Build context from memory graph starting from a specific memory.
        
        Args:
            url: Starting point (title, path or memory:// URL)
            depth: How many hops to follow in the graph
            timeframe: Time window for filtering (e.g. "1 month")
            
        Returns:
            Dict with context information and related memories
        """
        # Extract title from URL
        title = self._extract_title_from_url(url)
        if not title or title not in self._memory_graph:
            return {'error': f"Memory not found: {url}"}
            
        # Start with the target memory
        memory_data = await self.read_memory(title)
        if not memory_data:
            return {'error': f"Failed to read memory: {title}"}
            
        # Initialize context with target memory
        context = {
            'primary': memory_data,
            'related': []
        }
        
        # Get related memories up to specified depth
        visited = {title}
        to_visit = [(title, 1)]  # (title, current_depth)
        
        while to_visit:
            current_title, current_depth = to_visit.pop(0)
            
            if current_depth > depth:
                continue
                
            current_node = self._memory_graph.get(current_title, {})
            relations = current_node.get('relations', [])
            
            for relation in relations:
                target_title = relation.get('to_name')
                if not target_title or target_title in visited:
                    continue
                    
                related_data = await self.read_memory(target_title)
                if related_data:
                    context['related'].append({
                        'title': target_title,
                        'relation': relation.get('type'),
                        'content': related_data.get('content'),
                        'path': related_data.get('path'),
                        'depth': current_depth
                    })
                    
                    visited.add(target_title)
                    # Add to visit queue for next depth
                    if current_depth < depth:
                        to_visit.append((target_title, current_depth + 1))
        
        return context
    
    async def recent_activity(self, 
                        memory_type: str = "all", 
                        depth: int = 1, 
                        timeframe: str = "1 week") -> Dict[str, Any]:
        """Get recent memory changes.
        
        Args:
            memory_type: Type of memories to include 
            depth: How many related items to include
            timeframe: Time window (e.g. "1 week")
            
        Returns:
            Dict with recent activity information
        """
        # Parse timeframe
        days = self._parse_timeframe(timeframe)
        cutoff = datetime.now().timestamp() - (days * 86400)
        
        results = []
        
        # Scan memory files for recent changes
        for folder in os.listdir(self.memory_dir):
            folder_path = os.path.join(self.memory_dir, folder)
            if not os.path.isdir(folder_path) or (memory_type != "all" and folder != memory_type):
                continue
                
            for filename in os.listdir(folder_path):
                if not filename.endswith('.md'):
                    continue
                    
                filepath = os.path.join(folder_path, filename)
                file_mtime = os.path.getmtime(filepath)
                
                # Check if file was modified within timeframe
                if file_mtime >= cutoff:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    title = self._extract_title_from_content(content) or os.path.splitext(filename)[0]
                    
                    results.append({
                        'title': title,
                        'path': filepath,
                        'folder': folder,
                        'modified': datetime.fromtimestamp(file_mtime).isoformat(),
                        'modified_ts': file_mtime
                    })
        
        # Sort by modification time (newest first)
        results.sort(key=lambda x: x['modified_ts'], reverse=True)
        
        # If depth > 0, include related memories
        if depth > 0:
            for result in list(results):  # Copy to avoid modification during iteration
                title = result['title']
                if title in self._memory_graph:
                    node = self._memory_graph[title]
                    relations = node.get('relations', [])
                    
                    for relation in relations[:depth]:  # Limit to specified depth
                        target_title = relation.get('to_name')
                        if not target_title:
                            continue
                            
                        # Check if already in results
                        if not any(r['title'] == target_title for r in results):
                            related_memory = await self.read_memory(target_title)
                            if related_memory:
                                # Add as related item
                                results.append({
                                    'title': target_title,
                                    'path': related_memory.get('path'),
                                    'folder': related_memory.get('folder'),
                                    'relation': f"Related to {title} ({relation.get('type', 'relates_to')})",
                                    'is_related': True
                                })
        
        return {
            'primary_results': [r for r in results if not r.get('is_related', False)],
            'related_results': [r for r in results if r.get('is_related', False)],
            'timeframe': timeframe
        }
        
    async def update_memory(self, 
                     title_or_path: str, 
                     new_content: str, 
                     new_title: str = None,
                     verbose: bool = False) -> Dict[str, Any]:
        """Update an existing memory.
        
        Args:
            title_or_path: Title or path of memory to update
            new_content: New content for the memory
            new_title: New title if renaming (optional)
            verbose: Whether to return detailed parsing results
            
        Returns:
            Dict with operation results
        """
        memory = await self.read_memory(title_or_path)
        if not memory:
            return {'error': f"Memory not found: {title_or_path}"}
            
        original_path = memory['path']
        original_title = memory['title']
        folder = memory['folder']
        
        # Update content and potentially rename
        if new_title and new_title != original_title:
            # Remove old memory from graph
            if original_title in self._memory_graph:
                del self._memory_graph[original_title]
                
            # Create new memory with new title
            result = await self.create_memory(
                title=new_title,
                content=new_content,
                folder=folder,
                verbose=verbose
            )
            
            # Delete old file
            try:
                os.remove(original_path)
            except Exception as e:
                return {'error': f"Error removing old file: {str(e)}"}
                
            return result
        else:
            # Update existing file
            result = await self.create_memory(
                title=original_title,
                content=new_content,
                folder=folder,
                verbose=verbose
            )
            
            return result
    
    async def delete_memory(self, title_or_path: str) -> Dict[str, Any]:
        """Delete a memory.
        
        Args:
            title_or_path: Title or path of memory to delete
            
        Returns:
            Dict with operation results
        """
        memory = await self.read_memory(title_or_path)
        if not memory:
            return {'error': f"Memory not found: {title_or_path}"}
            
        path = memory['path']
        title = memory['title']
        
        try:
            # Remove file
            os.remove(path)
            
            # Remove from memory graph
            if title in self._memory_graph:
                del self._memory_graph[title]
                
            # Remove from index
            if title.lower() in self._memory_index:
                del self._memory_index[title.lower()]
                
            return {'success': True, 'message': f"Deleted memory: {title}"}
            
        except Exception as e:
            return {'error': f"Failed to delete memory: {str(e)}"}
    
    def _load_memories(self):
        """Load all existing memories into the memory graph."""
        self._memory_graph = {}
        self._memory_index = {}
        
        for folder in os.listdir(self.memory_dir):
            folder_path = os.path.join(self.memory_dir, folder)
            if not os.path.isdir(folder_path):
                continue
                
            for filename in os.listdir(folder_path):
                if not filename.endswith('.md'):
                    continue
                    
                filepath = os.path.join(folder_path, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Extract title and parse content
                title = self._extract_title_from_content(content)
                if not title:
                    title = os.path.splitext(filename)[0]
                    
                # Parse memory content
                parsed_data = self._parse_memory_content(content, title, folder, filepath)
                
                # Add to memory graph
                self._memory_graph[title] = parsed_data
                self._memory_index[title.lower()] = title
    
    def _parse_memory_content(self, content: str, title: str, folder: str, filepath: str) -> Dict[str, Any]:
        """Parse memory content to extract observations, relations, and tags.
        
        Args:
            content: Markdown content to parse
            title: Memory title
            folder: Memory folder
            filepath: Full path to memory file
            
        Returns:
            Dict with parsed content components
        """
        parsed = {
            'title': title,
            'folder': folder,
            'path': filepath,
            'observations': [],
            'relations': [],
            'tags': [],
        }
        
        # Remove frontmatter
        content_without_frontmatter = self._remove_frontmatter(content)
        
        # Extract frontmatter tags if any
        frontmatter = self._extract_frontmatter(content)
        if frontmatter and 'tags' in frontmatter:
            parsed['tags'].extend(frontmatter['tags'])
        
        # Extract observations (format: - [category] Text #tag1 #tag2)
        observation_pattern = r'- \[(.*?)\](.*?)(?=\n|$)'
        for match in re.finditer(observation_pattern, content_without_frontmatter):
            category = match.group(1).strip()
            text = match.group(2).strip()
            
            # Extract hashtags
            hashtags = []
            for tag_match in re.finditer(r'#(\w+)', text):
                hashtag = tag_match.group(1)
                hashtags.append(hashtag)
                if hashtag not in parsed['tags']:
                    parsed['tags'].append(hashtag)
                    
            # Remove hashtags from text for cleaner storage
            clean_text = re.sub(r'#\w+', '', text).strip()
            
            parsed['observations'].append({
                'category': category,
                'text': clean_text,
                'tags': hashtags
            })
            
        # Extract relations (format: - relation_type [[Target Entity]])
        relation_pattern = r'- (\w+) \[\[(.*?)\]\](?:\s+\((.*?)\))?'
        for match in re.finditer(relation_pattern, content_without_frontmatter):
            relation_type = match.group(1).strip()
            target = match.group(2).strip()
            context = match.group(3).strip() if match.group(3) else None
            
            parsed['relations'].append({
                'type': relation_type,
                'to_name': target,
                'context': context
            })
            
        return parsed
    
    def _sanitize_filename(self, title: str) -> str:
        """Convert title to a safe filename."""
        # Replace spaces with hyphens and remove unsafe characters
        safe_name = re.sub(r'[^\w\s-]', '', title).strip().lower()
        safe_name = re.sub(r'[\s]+', '-', safe_name)
        return safe_name
    
    def _extract_title_from_content(self, content: str) -> Optional[str]:
        """Extract title from markdown content."""
        # Try frontmatter title first
        frontmatter = self._extract_frontmatter(content)
        if frontmatter and 'title' in frontmatter:
            return frontmatter['title']
            
        # Try # Title format
        match = re.search(r'^#\s+(.*?)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()
            
        return None
    
    def _extract_frontmatter(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract YAML frontmatter from markdown content."""
        frontmatter_match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
        if frontmatter_match:
            frontmatter_yaml = frontmatter_match.group(1)
            try:
                return yaml.safe_load(frontmatter_yaml)
            except Exception:
                pass
        return None
    
    def _remove_frontmatter(self, content: str) -> str:
        """Remove frontmatter from content."""
        return re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)
    
    def _extract_snippet(self, content: str, query: str, max_length: int = 150) -> str:
        """Extract a text snippet containing the query."""
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Find position of query in content
        pos = content_lower.find(query_lower)
        if pos == -1:
            # If query not found directly, try to find a relevant section
            return content[:max_length] + "..." if len(content) > max_length else content
            
        # Extract snippet around the query position
        start = max(0, pos - max_length // 2)
        end = min(len(content), pos + len(query) + max_length // 2)
        
        # Adjust to whole words
        if start > 0:
            while start > 0 and content[start] != ' ':
                start -= 1
                
        if end < len(content):
            while end < len(content) and content[end] != ' ':
                end += 1
                
        snippet = content[start:end]
        
        # Add ellipsis if needed
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
            
        return snippet
    
    def _calculate_relevance_score(self, title: str, content: str, query: str) -> float:
        """Calculate relevance score for search results."""
        query_lower = query.lower()
        title_lower = title.lower()
        content_lower = content.lower()
        
        # Base score
        score = 0.0
        
        # Title match is highly valuable
        if query_lower == title_lower:
            score += 100.0  # Exact title match
        elif query_lower in title_lower:
            score += 50.0  # Partial title match
            
        # Content matches
        content_matches = content_lower.count(query_lower)
        score += content_matches * 1.0
        
        # Bonus for shorter content (more focused)
        score += max(0, 10.0 - (len(content) / 1000))
        
        return score
    
    def _find_by_title(self, title: str) -> Optional[str]:
        """Find memory by title, case-insensitive."""
        title_lower = title.lower()
        
        # Direct match in index
        if title_lower in self._memory_index:
            return self._memory_index[title_lower]
            
        # Partial match
        for original_title, normalized_title in self._memory_index.items():
            if title_lower in original_title:
                return normalized_title
                
        return None
    
    def _extract_title_from_url(self, url: str) -> Optional[str]:
        """Extract title from memory URL or path."""
        # Handle memory:// URLs
        if url.startswith('memory://'):
            path = url[len('memory://'):]
            
            # Check if it's a path with folder
            if '/' in path:
                folder, title = path.split('/', 1)
                return title
            
            # Otherwise just a title
            return path
            
        # Handle direct file paths
        if url.endswith('.md') and os.path.exists(url):
            filename = os.path.basename(url)
            return os.path.splitext(filename)[0]
            
        # Handle direct title
        return url
    
    def _parse_timeframe(self, timeframe: str) -> int:
        """Parse timeframe string to number of days."""
        if not timeframe:
            return 7  # Default 1 week
            
        parts = timeframe.split()
        if len(parts) != 2:
            return 7
            
        try:
            value = int(parts[0])
            unit = parts[1].lower()
            
            if unit in ('day', 'days'):
                return value
            elif unit in ('week', 'weeks'):
                return value * 7
            elif unit in ('month', 'months'):
                return value * 30
            elif unit in ('year', 'years'):
                return value * 365
            else:
                return 7
                
        except ValueError:
            return 7