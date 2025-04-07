"""Project manager package for Bella.

This package provides project management capabilities for Bella.
"""

from .project_manager import ProjectManager
from .memory_format_adapter import MemoryFormatAdapter
from .memory_integration import MemoryIntegration, get_memory_integration

# Create a singleton instance
_project_manager = None

def get_project_manager(base_dir=None):
    """Get or create a singleton instance of ProjectManager.
    
    Args:
        base_dir: Base directory for storing project files
        
    Returns:
        ProjectManager instance
    """
    global _project_manager
    if _project_manager is None:
        _project_manager = ProjectManager(base_dir)
    return _project_manager