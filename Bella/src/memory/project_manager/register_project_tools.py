"""Register project management tools for Bella function calling.

This module registers the project management capabilities as callable tools for Bella's LLM function calls,
implementing the standardized format and project-based memory system described in Bella_functions.md.
"""

from typing import List, Optional

from ...llm.tools_registry import registry
from . import get_project_manager

@registry.register_tool(description="Start a new project or activate an existing one")
async def start_project(project_name: str) -> dict:
    """Start a new project or activate an existing one.
    
    This function creates a new project with the specified name if it doesn't exist,
    or activates an existing one. The project will have standard folders for organizing
    information: notes, concepts, research, and content.
    
    Args:
        project_name: Name of the project to start
        
    Returns:
        Dictionary with operation results
    """
    project_manager = get_project_manager()
    result = await project_manager.start_project(project_name)
    
    return {
        'success': result['success'],
        'message': result['message'],
        'project_name': result.get('project_name', project_name),
        'is_new': result.get('status') == 'created'
    }

@registry.register_tool(description="Save content to a project file")
async def save_to(content: str, file_type: str = 'notes', title: str = None) -> dict:
    """Save content to a file in the active project.
    
    This function adds a new entry to a project file with the specified content.
    If no title is provided, one will be generated from the content.
    
    Args:
        content: Content to save
        file_type: Type of file to save to ('notes', 'research', 'instructions', 'main')
        title: Optional title for the entry
        
    Returns:
        Dictionary with operation results
    """
    project_manager = get_project_manager()
    result = await project_manager.save_to(content, file_type, title)
    
    return {
        'success': result['success'],
        'message': result['message'],
        'title': result.get('title', title),
        'file_type': result.get('file_type', file_type)
    }

@registry.register_tool(description="Save conversation to a project file")
async def save_conversation(context: List[str], file_type: str = 'notes', title: str = None) -> dict:
    """Save conversation context to a project file.
    
    This function creates a new entry in a project file containing the last few exchanges
    from the conversation, formatted in a structured way.
    
    Args:
        context: List of recent conversation messages
        file_type: Type of file to save to ('notes', 'research', 'instructions')
        title: Optional title for the entry
        
    Returns:
        Dictionary with operation results
    """
    project_manager = get_project_manager()
    result = await project_manager.save_conversation(context, file_type, title)
    
    return {
        'success': result['success'],
        'message': result['message'],
        'title': result.get('title', title),
        'file_type': result.get('file_type', file_type)
    }

@registry.register_tool(description="Edit an existing entry in a project file")
async def edit_entry(entry_title: str, edit_content: str, file_type: str = 'notes') -> dict:
    """Edit an existing entry in a project file.
    
    This function updates an existing entry in a project file by adding the specified content.
    
    Args:
        entry_title: Title of the entry to edit
        edit_content: Content to add to the entry
        file_type: Type of file containing the entry
        
    Returns:
        Dictionary with operation results
    """
    project_manager = get_project_manager()
    result = await project_manager.edit(entry_title, edit_content, file_type)
    
    return {
        'success': result['success'],
        'message': result['message'],
        'title': entry_title,
        'file_type': file_type
    }

@registry.register_tool(description="List entries in a project file")
async def list_all(file_type: str = 'notes', query: str = None) -> dict:
    """List all entries in a project file, optionally filtered by query.
    
    This function lists all entries in a project file of the specified type.
    If a query is provided, only entries matching the query will be returned.
    
    Args:
        file_type: Type of file to list entries from
        query: Optional filter query
        
    Returns:
        Dictionary with list results
    """
    project_manager = get_project_manager()
    result = await project_manager.list_all(file_type, query)
    
    if not result['success']:
        return {
            'success': False,
            'message': result['message']
        }
    
    # Format entry titles for display
    entries = [entry['title'] for entry in result['entries']]
    
    return {
        'success': True,
        'file_type': result['file_type'],
        'entries': entries,
        'count': result['count'],
        'query': result['query'],
        'message': f"Found {result['count']} entries in {file_type}"
    }

@registry.register_tool(description="Read an entry from a project file")
async def read_entry(entry_title: str, file_type: str = None) -> dict:
    """Read a specific entry from a project file.
    
    This function retrieves the content of a specific entry in a project file.
    If file_type is not specified, all file types will be searched.
    
    Args:
        entry_title: Title of the entry to read
        file_type: Type of file containing the entry (optional)
        
    Returns:
        Dictionary with entry content
    """
    project_manager = get_project_manager()
    result = await project_manager.read_entry(entry_title, file_type)
    
    if not result['success']:
        return {
            'success': False,
            'message': result['message']
        }
    
    return {
        'success': True,
        'title': result['title'],
        'content': result['content'],
        'folder': result['folder'],
        'created': result['created'],
        'updated': result['updated'],
        'tags': result['tags']
    }

@registry.register_tool(description="Delete an entry from a project file")
async def delete_entry(entry_title: str, file_type: str = None) -> dict:
    """Delete a specific entry from a project file.
    
    This function permanently deletes an entry from a project file.
    If file_type is not specified, all file types will be searched.
    
    Args:
        entry_title: Title of the entry to delete
        file_type: Type of file containing the entry (optional)
        
    Returns:
        Dictionary with operation results
    """
    project_manager = get_project_manager()
    result = await project_manager.delete_entry(entry_title, file_type)
    
    return {
        'success': result['success'],
        'message': result['message'],
        'title': entry_title
    }

@registry.register_tool(description="Quit the active project")
async def quit_project() -> dict:
    """Quit the active project.
    
    This function closes the active project, saving any pending changes.
    After quitting, no project will be active until another one is started.
    
    Returns:
        Dictionary with operation results
    """
    project_manager = get_project_manager()
    result = await project_manager.quit_project()
    
    return {
        'success': result['success'],
        'message': result['message'],
        'project_name': result.get('project_name')
    }