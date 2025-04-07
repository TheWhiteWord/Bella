"""Initialize project management system for Bella.

This module handles initialization of the project management system,
ensuring all components are properly set up and registered with Bella's function calling system.
"""

import os
import asyncio
from typing import Dict, Any

from . import get_project_manager, get_memory_integration
from .memory_format_adapter import MemoryFormatAdapter

async def initialize_project_management(base_dir: str = None) -> Dict[str, Any]:
    """Initialize the project management system.
    
    This function ensures the project management system is properly initialized,
    creates necessary directory structures, and registers all function handlers.
    
    Args:
        base_dir: Base directory for project files (optional)
        
    Returns:
        Dict with initialization results
    """
    # Import here to avoid circular imports
    from . import register_project_tools
    
    # Initialize project manager
    project_manager = get_project_manager(base_dir)
    
    # Initialize memory integration
    memory_integration = get_memory_integration(base_dir)
    
    # Create the projects folder if it doesn't exist
    projects_dir = os.path.join(project_manager.base_dir, "projects")
    os.makedirs(projects_dir, exist_ok=True)
    
    # Register standard folders if they don't exist
    memories_dir = project_manager.base_dir  # Use base_dir directly (memories directory)
    for folder in ['conversations', 'facts', 'preferences', 'reminders', 'general']:
        folder_path = os.path.join(memories_dir, folder)
        os.makedirs(folder_path, exist_ok=True)
    
    # Return results
    return {
        'success': True,
        'message': "Project management system initialized",
        'base_dir': project_manager.base_dir,
        'projects_dir': projects_dir
    }