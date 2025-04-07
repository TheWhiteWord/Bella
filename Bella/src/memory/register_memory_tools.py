"""Register enhanced memory tools for LLM function calling.

This module registers the enhanced memory capabilities as callable tools for LLM function calls.
The old individual memory functions have been replaced with the project-based memory system.
"""

from ..llm.tools_registry import registry
from .memory_api import read_note
from .project_manager.initialize import initialize_project_management
from .project_manager.memory_integration import get_memory_integration
from .memory_commands import handle_continue_command

# Import project-based system tools
from .project_manager.register_project_tools import (
    start_project,
    save_to,
    save_conversation,
    edit_entry,
    list_all,
    read_entry,
    delete_entry,
    quit_project
)

# Keep only the semantic_memory_search function from the old system
# as it's still useful for searching across all memories
@registry.register_tool(description="Search memory with semantic understanding")
async def semantic_memory_search(query: str, threshold: float = 0.7, max_results: int = 5) -> dict:
    """Search memory with semantic understanding.
    
    Args:
        query: Search query text
        threshold: Minimum similarity threshold (0-1)
        max_results: Maximum number of results to return
        
    Returns:
        Dict with search results
    """
    try:
        memory_integration = get_memory_integration()
        # Use standardized memory search
        results = await memory_integration.search_standardized_memories(
            query, threshold=threshold, max_results=max_results
        )
        return results
    except Exception as e:
        import logging
        logging.error(f"Error in semantic memory search: {e}")
        return {"success": False, "error": str(e)}

# Keep the continue conversation function as it's still useful
@registry.register_tool(description="Continue the conversation or thought process")
async def continue_conversation(conversation_context: list = None) -> dict:
    """Continue the conversation based on the provided context.
    
    Args:
        conversation_context: List of previous conversation turns
        
    Returns:
        Dict with continuation results
    """
    return await handle_continue_command(conversation_context)

# Initialize the project management system when the module is imported
async def _initialize_project_system():
    """Initialize the project management system during module import."""
    try:
        await initialize_project_management()
    except Exception as e:
        import logging
        logging.error(f"Error initializing project management system: {e}")

# Schedule initialization to run when event loop is available
import asyncio
try:
    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.create_task(_initialize_project_system())
    else:
        loop.run_until_complete(_initialize_project_system())
except Exception:
    # If we can't initialize right now, it will be initialized
    # when the first tool is used
    pass