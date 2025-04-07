"""Project manager for Bella.

This module provides project management capabilities for Bella, allowing users to create,
edit, and manage project-based information in a structured way, with standardized memory formats
for both autonomous memory and function-based memory.
"""

import os
import re
import yaml
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

class ProjectManager:
    """Manages project-based memory storage and retrieval with standardized format."""
    
    def __init__(self, base_dir: str = None):
        """Initialize project manager with specified directory.
        
        Args:
            base_dir (str, optional): Path to base directory for storing project files.
                If None, creates a default 'memories' directory in the project.
        """
        if base_dir is None:
            # Default to a 'memories' directory in the project
            self.base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "memories")
        else:
            self.base_dir = base_dir
            
        # Create base directory if it doesn't exist
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Create projects directory
        self.projects_dir = os.path.join(self.base_dir, "projects")
        os.makedirs(self.projects_dir, exist_ok=True)
        
        # Keep track of the active project
        self.active_project = None
        
    async def start_project(self, project_name: str) -> Dict[str, Any]:
        """Start a new project or activate an existing one.
        
        Args:
            project_name: Name of the project to start/activate
            
        Returns:
            Dict with operation results
        """
        # Sanitize project name
        safe_project_name = self._sanitize_name(project_name)
        project_dir = os.path.join(self.projects_dir, safe_project_name)
        
        # Check if project already exists
        if os.path.exists(project_dir):
            # Set as active project
            self.active_project = safe_project_name
            return {
                'success': True,
                'message': f"Project '{project_name}' already exists. Activated.",
                'project_path': project_dir,
                'status': 'activated',
                'project_name': project_name
            }
        
        # Create new project directory
        os.makedirs(project_dir, exist_ok=True)
        
        # Create standard subfolders
        for subfolder in ['notes', 'concepts', 'research', 'content']:
            os.makedirs(os.path.join(project_dir, subfolder), exist_ok=True)
        
        # Create initial files
        self._create_initial_files(project_dir, project_name)
        
        # Set as active project
        self.active_project = safe_project_name
        
        return {
            'success': True,
            'message': f"Project '{project_name}' created and activated.",
            'project_path': project_dir,
            'status': 'created',
            'project_name': project_name
        }
    
    async def save_to(self, content: str, file_type: str = 'notes', title: str = None) -> Dict[str, Any]:
        """Add a new entry to a project file.
        
        Args:
            content: Content to save
            file_type: Type of file to save to ('notes', 'research', 'instructions', 'main')
            title: Title for the entry (optional)
            
        Returns:
            Dict with operation results
        """
        # Ensure there's an active project
        if not self.active_project:
            return {
                'success': False,
                'message': "No active project. Please start a project first."
            }
        
        # Validate file type
        valid_file_types = ['notes', 'research', 'instructions', 'main']
        if file_type not in valid_file_types:
            file_type = 'notes'  # Default to notes
        
        # Special handling for 'main'
        if file_type == 'main':
            return await self._save_to_main(content, title)
        
        # Generate title if not provided
        if not title:
            # Use first few words of content as title
            words = re.findall(r'\w+', content.lower())
            if len(words) > 5:
                title = "-".join(words[:5])
            else:
                # Fallback to timestamp
                title = f"entry-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Ensure title is safe for filenames
        safe_title = self._sanitize_name(title)
        
        # Prepare file path
        project_dir = os.path.join(self.projects_dir, self.active_project)
        folder_path = os.path.join(project_dir, file_type)
        filepath = os.path.join(folder_path, f"{safe_title}.md")
        
        # Generate tags from content
        tags = self._extract_tags_from_content(content)
        if file_type not in tags:
            tags.append(file_type)
        
        # Format with standardized structure
        formatted_content = self._format_content(title, content, tags, self.active_project, file_type)
        
        # Write content to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(formatted_content)
        
        return {
            'success': True,
            'message': f"Entry '{title}' saved to {file_type}.",
            'path': filepath,
            'title': title,
            'file_type': file_type
        }
    
    async def _save_to_main(self, content: str, title: str = None) -> Dict[str, Any]:
        """Save content to the main file of the active project.
        
        Args:
            content: Content to save
            title: Title for the content (optional)
            
        Returns:
            Dict with operation results
        """
        # Use 'main' as the title
        title = title or "main"
        
        # Prepare file path
        project_dir = os.path.join(self.projects_dir, self.active_project)
        filepath = os.path.join(project_dir, "main.md")
        
        # Generate tags
        tags = self._extract_tags_from_content(content)
        if "main" not in tags:
            tags.append("main")
        
        # Format with standardized structure
        formatted_content = self._format_content(title, content, tags, self.active_project, "main")
        
        # Write content to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(formatted_content)
        
        return {
            'success': True,
            'message': f"Content saved to main file.",
            'path': filepath,
            'title': title,
            'file_type': 'main'
        }
    
    async def save_conversation(self, context: List[str], file_type: str = 'notes', title: str = None) -> Dict[str, Any]:
        """Save conversation context to a project file.
        
        Args:
            context: List containing the last two user messages and two assistant replies
            file_type: Type of file to save to ('notes', 'research', 'instructions')
            title: Title for the entry (optional)
            
        Returns:
            Dict with operation results
        """
        # Ensure there's an active project
        if not self.active_project:
            return {
                'success': False,
                'message': "No active project. Please start a project first."
            }
        
        # Validate file type
        valid_file_types = ['notes', 'research', 'instructions']
        if file_type not in valid_file_types:
            file_type = 'notes'  # Default to notes
        
        # Ensure we have enough context
        if not context or len(context) < 2:
            return {
                'success': False,
                'message': "Not enough conversation context to save."
            }
        
        # Extract content from conversation (last two messages from each role)
        user_messages = []
        assistant_messages = []
        
        # Extract the most recent messages (up to 2 from each role)
        for i, message in enumerate(reversed(context)):
            if i % 2 == 0 and len(assistant_messages) < 2:
                assistant_messages.insert(0, message)
            elif i % 2 == 1 and len(user_messages) < 2:
                user_messages.insert(0, message)
            
            # Stop once we have enough messages
            if len(user_messages) >= 2 and len(assistant_messages) >= 2:
                break
        
        # Create title if not provided
        if not title:
            # Try to generate a title from the conversation topic
            words = re.findall(r'\w+', user_messages[0].lower())
            if len(words) > 3:
                topic_words = [word for word in words if len(word) > 3][:3]
                title = " ".join(topic_words) if topic_words else "conversation"
            else:
                title = "conversation"
        
        # Format conversation content
        content = f"# Conversation Summary: {title}\n\n"
        
        # Add user and assistant messages
        for i, (user_msg, asst_msg) in enumerate(zip(user_messages, assistant_messages)):
            content += f"## Exchange {i+1}\n\n"
            content += f"**User**: {user_msg}\n\n"
            content += f"**Assistant**: {asst_msg}\n\n"
        
        # Generate semantic summary
        summary = f"This conversation covers {title}. "
        summary += "The user asked about topics related to this, and the assistant provided information and insights."
        
        content += f"## Summary\n\n{summary}\n\n"
        
        # Save the formatted content
        return await self.save_to(content, file_type, title)
    
    async def edit(self, entry_title: str, edit_content: str, file_type: str = 'notes') -> Dict[str, Any]:
        """Edit an existing entry in a project file.
        
        Args:
            entry_title: Title of the entry to edit
            edit_content: Content to be added or instructions for editing
            file_type: Type of file containing the entry
            
        Returns:
            Dict with operation results
        """
        # Ensure there's an active project
        if not self.active_project:
            return {
                'success': False,
                'message': "No active project. Please start a project first."
            }
        
        # Special handling for 'main' file
        if file_type == 'main' or entry_title == 'main':
            return await self._edit_main(edit_content)
        
        # Try to find the entry
        project_dir = os.path.join(self.projects_dir, self.active_project)
        
        # Determine file path based on entry title
        entry_path = None
        
        # First try direct match with sanitized name
        safe_title = self._sanitize_name(entry_title)
        candidate_path = os.path.join(project_dir, file_type, f"{safe_title}.md")
        if os.path.exists(candidate_path):
            entry_path = candidate_path
        
        # If not found, try fuzzy search in the file type directory
        if not entry_path:
            for filename in os.listdir(os.path.join(project_dir, file_type)):
                if filename.endswith('.md'):
                    # Read file to check title
                    with open(os.path.join(project_dir, file_type, filename), 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Extract title from frontmatter
                    frontmatter = self._extract_frontmatter(content)
                    if frontmatter and 'title' in frontmatter:
                        file_title = frontmatter['title']
                        # Check for match (case-insensitive)
                        if file_title.lower() == entry_title.lower():
                            entry_path = os.path.join(project_dir, file_type, filename)
                            break
        
        if not entry_path:
            return {
                'success': False,
                'message': f"Entry '{entry_title}' not found in {file_type}."
            }
        
        # Read existing content
        with open(entry_path, 'r', encoding='utf-8') as f:
            existing_content = f.read()
        
        # Extract frontmatter and content parts
        frontmatter = self._extract_frontmatter(existing_content)
        if not frontmatter:
            return {
                'success': False,
                'message': f"Invalid file format for '{entry_title}'."
            }
        
        # Get content without frontmatter
        content_without_frontmatter = self._remove_frontmatter(existing_content)
        
        # Update content by appending edit_content
        updated_content = f"{content_without_frontmatter}\n\n{edit_content}"
        
        # Update frontmatter
        frontmatter['updated'] = datetime.now().isoformat()
        
        # Reconstruct file content
        new_content = f"---\n{yaml.dump(frontmatter)}---\n\n{updated_content}"
        
        # Write updated content
        with open(entry_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return {
            'success': True,
            'message': f"Entry '{entry_title}' in {file_type} updated.",
            'path': entry_path
        }
    
    async def _edit_main(self, edit_content: str) -> Dict[str, Any]:
        """Edit the main file of the active project.
        
        Args:
            edit_content: Content to be added or instructions for editing
            
        Returns:
            Dict with operation results
        """
        # Ensure there's an active project
        project_dir = os.path.join(self.projects_dir, self.active_project)
        main_path = os.path.join(project_dir, "main.md")
        
        # Check if main file exists
        if not os.path.exists(main_path):
            # Create main file if it doesn't exist
            return await self._save_to_main(edit_content)
        
        # Read existing content
        with open(main_path, 'r', encoding='utf-8') as f:
            existing_content = f.read()
        
        # Extract frontmatter
        frontmatter = self._extract_frontmatter(existing_content)
        if not frontmatter:
            # Invalid format, recreate the file
            return await self._save_to_main(edit_content)
        
        # Get content without frontmatter
        content_without_frontmatter = self._remove_frontmatter(existing_content)
        
        # Update content
        updated_content = f"{content_without_frontmatter}\n\n{edit_content}"
        
        # Update frontmatter
        frontmatter['updated'] = datetime.now().isoformat()
        
        # Reconstruct file content
        new_content = f"---\n{yaml.dump(frontmatter)}---\n\n{updated_content}"
        
        # Write updated content
        with open(main_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return {
            'success': True,
            'message': f"Main file updated.",
            'path': main_path
        }
    
    async def list_all(self, file_type: str = 'notes', query: str = None) -> Dict[str, Any]:
        """List all entries in a project file, optionally filtered by query.
        
        Args:
            file_type: Type of file to list entries from
            query: Optional filter query
            
        Returns:
            Dict with list results
        """
        # Ensure there's an active project
        if not self.active_project:
            return {
                'success': False,
                'message': "No active project. Please start a project first."
            }
        
        # Validate file type
        valid_file_types = ['notes', 'research', 'instructions']
        if file_type not in valid_file_types:
            file_type = 'notes'  # Default to notes
        
        # Get project directory
        project_dir = os.path.join(self.projects_dir, self.active_project)
        file_type_dir = os.path.join(project_dir, file_type)
        
        if not os.path.exists(file_type_dir):
            return {
                'success': False,
                'message': f"No {file_type} directory found in project '{self.active_project}'."
            }
        
        # Get all markdown files
        entries = []
        for filename in os.listdir(file_type_dir):
            if not filename.endswith('.md'):
                continue
                
            filepath = os.path.join(file_type_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extract title from frontmatter
            frontmatter = self._extract_frontmatter(content)
            entry_title = frontmatter.get('title') if frontmatter else None
            
            # Use filename as fallback
            if not entry_title:
                entry_title = os.path.splitext(filename)[0]
            
            # If query specified, check if entry matches
            if query:
                query_lower = query.lower()
                content_lower = content.lower()
                title_lower = entry_title.lower()
                
                if query_lower in title_lower or query_lower in content_lower:
                    entries.append({
                        'title': entry_title,
                        'path': filepath,
                        'matches_query': True
                    })
            else:
                # No query, add all entries
                entries.append({
                    'title': entry_title,
                    'path': filepath
                })
        
        # Sort entries by title
        entries.sort(key=lambda x: x['title'])
        
        return {
            'success': True,
            'file_type': file_type,
            'entries': entries,
            'count': len(entries),
            'query': query or 'None',
            'project': self.active_project
        }
    
    async def read_entry(self, entry_title: str, file_type: str = None) -> Dict[str, Any]:
        """Read a specific entry in a project file.
        
        Args:
            entry_title: Title of the entry to read
            file_type: Type of file containing the entry (optional, will search all if None)
            
        Returns:
            Dict with entry content
        """
        # Ensure there's an active project
        if not self.active_project:
            return {
                'success': False,
                'message': "No active project. Please start a project first."
            }
        
        # Special handling for 'main'
        if entry_title == 'main' or file_type == 'main':
            return await self._read_main()
        
        # Get project directory
        project_dir = os.path.join(self.projects_dir, self.active_project)
        
        # Determine where to search
        search_dirs = []
        if file_type:
            search_dirs = [os.path.join(project_dir, file_type)]
        else:
            # Search in all standard folders
            for folder in ['notes', 'concepts', 'research', 'content']:
                folder_path = os.path.join(project_dir, folder)
                if os.path.exists(folder_path):
                    search_dirs.append(folder_path)
        
        # Search for entry by title
        entry_path = None
        entry_folder = None
        
        for search_dir in search_dirs:
            # Try direct match with sanitized name
            safe_title = self._sanitize_name(entry_title)
            candidate_path = os.path.join(search_dir, f"{safe_title}.md")
            if os.path.exists(candidate_path):
                entry_path = candidate_path
                entry_folder = os.path.basename(search_dir)
                break
            
            # If not found, try fuzzy search
            for filename in os.listdir(search_dir):
                if not filename.endswith('.md'):
                    continue
                    
                filepath = os.path.join(search_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Extract title from frontmatter
                frontmatter = self._extract_frontmatter(content)
                if frontmatter and 'title' in frontmatter:
                    file_title = frontmatter['title']
                    # Check for match (case-insensitive)
                    if file_title.lower() == entry_title.lower():
                        entry_path = filepath
                        entry_folder = os.path.basename(search_dir)
                        break
        
        if not entry_path:
            return {
                'success': False,
                'message': f"Entry '{entry_title}' not found."
            }
        
        # Read entry content
        with open(entry_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract frontmatter
        frontmatter = self._extract_frontmatter(content) or {}
        content_without_frontmatter = self._remove_frontmatter(content)
        
        return {
            'success': True,
            'title': frontmatter.get('title', entry_title),
            'content': content_without_frontmatter,
            'path': entry_path,
            'folder': entry_folder,
            'created': frontmatter.get('created'),
            'updated': frontmatter.get('updated'),
            'tags': frontmatter.get('tags', [])
        }
    
    async def _read_main(self) -> Dict[str, Any]:
        """Read the main file of the active project.
        
        Returns:
            Dict with main file content
        """
        project_dir = os.path.join(self.projects_dir, self.active_project)
        main_path = os.path.join(project_dir, "main.md")
        
        if not os.path.exists(main_path):
            return {
                'success': False,
                'message': f"Main file not found in project '{self.active_project}'."
            }
        
        # Read content
        with open(main_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract frontmatter
        frontmatter = self._extract_frontmatter(content) or {}
        content_without_frontmatter = self._remove_frontmatter(content)
        
        return {
            'success': True,
            'title': 'main',
            'content': content_without_frontmatter,
            'path': main_path,
            'folder': self.active_project,
            'created': frontmatter.get('created'),
            'updated': frontmatter.get('updated'),
            'tags': frontmatter.get('tags', [])
        }
    
    async def delete_entry(self, entry_title: str, file_type: str = None) -> Dict[str, Any]:
        """Delete a specific entry in a project file.
        
        Args:
            entry_title: Title of the entry to delete
            file_type: Type of file containing the entry (optional, will search all if None)
            
        Returns:
            Dict with operation results
        """
        # Ensure there's an active project
        if not self.active_project:
            return {
                'success': False,
                'message': "No active project. Please start a project first."
            }
        
        # Don't allow deleting main
        if entry_title == 'main' or file_type == 'main':
            return {
                'success': False,
                'message': "Cannot delete main file. Use edit instead."
            }
        
        # First locate the entry
        entry_info = await self.read_entry(entry_title, file_type)
        if not entry_info['success']:
            return entry_info
        
        entry_path = entry_info['path']
        
        # Delete the file
        try:
            os.remove(entry_path)
            return {
                'success': True,
                'message': f"Entry '{entry_title}' deleted.",
                'path': entry_path
            }
        except Exception as e:
            return {
                'success': False,
                'message': f"Failed to delete entry: {str(e)}"
            }
    
    async def quit_project(self) -> Dict[str, Any]:
        """Quit the active project.
        
        Returns:
            Dict with operation results
        """
        if not self.active_project:
            return {
                'success': False,
                'message': "No active project to quit."
            }
        
        project_name = self.active_project
        self.active_project = None
        
        return {
            'success': True,
            'message': f"Project '{project_name}' closed.",
            'project_name': project_name
        }
    
    def _create_initial_files(self, project_dir: str, project_name: str):
        """Create initial files for a new project.
        
        Args:
            project_dir: Path to project directory
            project_name: Name of the project
        """
        # Create main.md
        main_content = f"# {project_name}\n\nWelcome to the {project_name} project!\n\nThis file contains the main content of your project."
        main_path = os.path.join(project_dir, "main.md")
        formatted_content = self._format_content("main", main_content, ["main", "project"], project_name, "main")
        with open(main_path, 'w', encoding='utf-8') as f:
            f.write(formatted_content)
        
        # Create README.md
        readme_content = f"# Project: {project_name}\n\n## Overview\nThis project was created using Bella's project management system.\n\n## Folders\n- notes: For general notes\n- concepts: For concept development\n- research: For research materials\n- content: For finished content"
        readme_path = os.path.join(project_dir, "README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
    def _format_content(self, title: str, content: str, tags: List[str], project: str, file_type: str) -> str:
        """Format content with standardized structure.
        
        Args:
            title: Title of the content
            content: Raw content
            tags: List of tags
            project: Project name
            file_type: Type of file
            
        Returns:
            Formatted content with frontmatter
        """
        now = datetime.now().isoformat()
        entry_id = str(uuid.uuid4())[:10]
        
        # Prepare frontmatter
        frontmatter = {
            'title': title,
            'created': now,
            'updated': now,
            'project': project,
            'file_type': file_type,
            'tags': tags,
            'entry_id': entry_id
        }
        
        # Ensure content starts with a title if it doesn't have one
        if not content.strip().startswith('#'):
            content = f"# {title}\n\n{content}"
        
        # Format with frontmatter
        formatted_content = f"---\n{yaml.dump(frontmatter)}---\n\n{content}"
        return formatted_content
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize a name for use in filenames.
        
        Args:
            name: Name to sanitize
            
        Returns:
            Sanitized name
        """
        # Replace spaces with hyphens and remove unsafe characters
        safe_name = re.sub(r'[^\w\s-]', '', name).strip().lower()
        safe_name = re.sub(r'[\s]+', '-', safe_name)
        return safe_name
    
    def _extract_frontmatter(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract YAML frontmatter from content.
        
        Args:
            content: Content to extract frontmatter from
            
        Returns:
            Dict with frontmatter or None if not found
        """
        frontmatter_match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
        if frontmatter_match:
            frontmatter_yaml = frontmatter_match.group(1)
            try:
                return yaml.safe_load(frontmatter_yaml)
            except Exception:
                pass
        return None
    
    def _remove_frontmatter(self, content: str) -> str:
        """Remove frontmatter from content.
        
        Args:
            content: Content with frontmatter
            
        Returns:
            Content without frontmatter
        """
        return re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)
    
    def _extract_tags_from_content(self, content: str) -> List[str]:
        """Extract hashtags from content.
        
        Args:
            content: Content to extract tags from
            
        Returns:
            List of tags
        """
        tags = []
        hashtag_pattern = r'#(\w+)'
        
        for match in re.finditer(hashtag_pattern, content):
            tag = match.group(1).lower()
            if tag not in tags:
                tags.append(tag)
                
        return tags