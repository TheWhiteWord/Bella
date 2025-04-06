"""Register enhanced memory tools for LLM function calling.

This module registers the enhanced memory capabilities as callable tools for LLM function calls.
"""

from ..llm.tools_registry import registry
from .main_app_integration import memory_manager
from .memory_commands import handle_continue_command
from .memory_api import save_note, read_note, list_notes

@registry.register_tool(description="Search memory with semantic understanding")
async def semantic_memory_search(query: str, threshold: float = 0.7, max_results: int = 5) -> dict:
    """Search for memories using semantic understanding rather than just keywords.
    
    Args:
        query: What to search for in memory
        threshold: Minimum similarity score (0-1)
        max_results: Maximum number of results to return
        
    Returns:
        Dictionary with search results and status
    """
    # Ensure memory is initialized
    await memory_manager.initialize()
    
    # Search using semantic capabilities
    results, success = await memory_manager.search_memory(query)
    
    # Format results for response
    if success and results:
        primary_results = results.get("primary_results", [])
        
        # Format for readability
        formatted_results = []
        for result in primary_results[:max_results]:
            formatted_results.append({
                "title": result.get("title", "Untitled"),
                "preview": result.get("content_preview", ""),
                "relevance": result.get("score", 0.0),
                "source": result.get("source", "unknown")
            })
            
        return {
            "success": True,
            "results": formatted_results,
            "result_count": len(formatted_results)
        }
    else:
        return {
            "success": False,
            "message": "No memories found"
        }

@registry.register_tool(description="Check if information is worth remembering")
async def evaluate_memory_importance(text: str) -> dict:
    """Evaluate how important a piece of information is for long-term memory.
    
    Args:
        text: Text to evaluate for importance
        
    Returns:
        Dictionary with importance evaluation
    """
    # Ensure memory is initialized
    await memory_manager.initialize()
    
    # Get importance score
    importance = await memory_manager.evaluate_memory_importance(text)
    
    # Determine importance level
    if importance >= 0.8:
        level = "high"
    elif importance >= 0.6:
        level = "medium"
    else:
        level = "low"
        
    return {
        "importance_score": importance,
        "importance_level": level,
        "should_remember": importance >= 0.7,
        "suggestion": "This information should be saved to memory." if importance >= 0.7 else "This information is probably not important enough to remember."
    }

@registry.register_tool(description="Optimize text for memory storage")
async def prepare_for_memory_storage(text: str, summarize: bool = True) -> dict:
    """Optimize text for memory storage by summarizing and extracting key information.
    
    Args:
        text: Text to optimize
        summarize: Whether to summarize long text
        
    Returns:
        Dictionary with optimized text
    """
    # Ensure memory is initialized
    await memory_manager.initialize()
    
    if not summarize:
        return {
            "optimized_text": text,
            "was_summarized": False
        }
    
    # Prepare text for storage
    optimized = await memory_manager.prepare_text_for_storage(text)
    
    return {
        "optimized_text": optimized,
        "was_summarized": optimized != text,
        "character_reduction": len(text) - len(optimized) if len(text) > len(optimized) else 0
    }

@registry.register_tool(description="Continue the conversation or thought process")
async def continue_conversation(conversation_context: list = None) -> dict:
    """Continue a conversation or thought process based on recent exchanges.
    
    This enables the continuation of a thought or conversation thread when triggered with 
    the "@agent Continue" command, using memory and context to provide coherent continuation.
    
    Args:
        conversation_context: List of recent conversation exchanges
        
    Returns:
        Dictionary with the continuation response and status
    """
    # Ensure memory is initialized
    await memory_manager.initialize()
    
    # Call the handler for continue command
    continuation_response = await handle_continue_command(conversation_context)
    
    return {
        "success": bool(continuation_response),
        "continuation": continuation_response,
        "based_on_memory": True if "based on what we know" in continuation_response else False,
        "context_depth": len(conversation_context) if conversation_context else 0
    }

@registry.register_tool(description="Save information to memory")
async def save_to_memory(content: str, memory_type: str = "general", title: str = None) -> dict:
    """Save information to memory for later recall.
    
    Args:
        content: The text content to save in memory
        memory_type: Category of memory (facts, preferences, conversations, reminders, general)
        title: Optional title/filename for the memory (auto-generated if not provided)
        
    Returns:
        Dictionary with save status
    """
    # Ensure memory is initialized
    await memory_manager.initialize()
    
    # Validate memory type
    valid_types = ["facts", "preferences", "conversations", "reminders", "general"]
    if memory_type not in valid_types:
        memory_type = "general"
    
    try:
        # First, check if it's worth storing
        should_store = await memory_manager.should_store_memory(content)
        
        if not should_store:
            return {
                "success": False,
                "message": "Content was evaluated as not important enough to store in memory."
            }
        
        # Prepare content for storage (summarize if needed)
        if len(content.split()) > 50:
            optimized_content = await memory_manager.prepare_text_for_storage(content)
        else:
            optimized_content = content
        
        # Save to memory
        if hasattr(memory_manager.enhanced_adapter, "store_memory"):
            success, path = await memory_manager.enhanced_adapter.store_memory(
                memory_type, optimized_content, title
            )
        else:
            # Fallback to direct API if adapter method not available
            path = await save_note(optimized_content, memory_type, title)
            success = bool(path)
        
        if success:
            return {
                "success": True,
                "message": f"Successfully stored in {memory_type} memory.",
                "path": path
            }
        else:
            return {
                "success": False,
                "message": "Failed to store in memory."
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error storing memory: {str(e)}"
        }

@registry.register_tool(description="Read a specific memory by ID")
async def read_specific_memory(memory_id: str) -> dict:
    """Read a specific memory by its ID.
    
    Args:
        memory_id: ID of the memory to read (can be path or memory_type/name format)
        
    Returns:
        Dictionary with memory content
    """
    # Ensure memory is initialized
    await memory_manager.initialize()
    
    try:
        # Check if memory_id is already a path
        if "/" in memory_id and not memory_id.startswith("memories/"):
            # Format is likely memory_type/name
            memory_type, name = memory_id.split("/", 1)
            path = f"memories/{memory_type}/{name}.md"
        else:
            # Use as-is (could be full path)
            path = memory_id
            if not path.endswith(".md"):
                path += ".md"
        
        # Read the memory
        content = await read_note(path)
        
        # Record access if we have an adapter
        if hasattr(memory_manager, "enhanced_adapter"):
            memory_manager.enhanced_adapter.record_memory_access(memory_id)
        
        if content:
            # Extract filename for title
            title = path.split("/")[-1].replace(".md", "")
            
            return {
                "success": True,
                "title": title,
                "content": content,
                "source": path
            }
        else:
            return {
                "success": False,
                "message": f"Memory not found: {memory_id}"
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error reading memory: {str(e)}"
        }

@registry.register_tool(description="List available memories by type")
async def list_memories_by_type(memory_type: str = "all") -> dict:
    """List available memories by type.
    
    Args:
        memory_type: Type of memories to list (facts, preferences, conversations, reminders, general, all)
        
    Returns:
        Dictionary with list of memories
    """
    # Ensure memory is initialized
    await memory_manager.initialize()
    
    try:
        valid_types = ["facts", "preferences", "conversations", "reminders", "general"]
        
        result = {
            "success": True,
            "memories": {}
        }
        
        if memory_type == "all":
            # List all memory types
            for mtype in valid_types:
                memories = await list_notes(mtype)
                result["memories"][mtype] = memories
        elif memory_type in valid_types:
            # List specific memory type
            memories = await list_notes(memory_type)
            result["memories"][memory_type] = memories
        else:
            return {
                "success": False,
                "message": f"Invalid memory type: {memory_type}"
            }
        
        # Count total memories
        total_count = sum(len(mems) for mems in result["memories"].values())
        result["total_count"] = total_count
        
        return result
    except Exception as e:
        return {
            "success": False,
            "message": f"Error listing memories: {str(e)}"
        }

@registry.register_tool(description="Save current conversation to memory")
async def save_conversation(user_input: str, assistant_response: str, title: str = None) -> dict:
    """Save the current conversation exchange to memory.
    
    Args:
        user_input: User's input text
        assistant_response: Assistant's response text
        title: Optional title for the conversation memory
        
    Returns:
        Dictionary with save status
    """
    # Ensure memory is initialized
    await memory_manager.initialize()
    
    try:
        # Format conversation
        content = f"User: {user_input}\n\nAssistant: {assistant_response}"
        
        # Generate title if not provided
        if not title:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
            title = f"conversation-on-{timestamp}"
        
        # Save to conversations memory
        if hasattr(memory_manager.enhanced_adapter, "store_memory"):
            success, path = await memory_manager.enhanced_adapter.store_memory(
                "conversations", content, title
            )
        else:
            # Fallback to direct API
            path = await save_note(content, "conversations", title)
            success = bool(path)
        
        if success:
            return {
                "success": True,
                "message": "Conversation saved to memory.",
                "path": path
            }
        else:
            return {
                "success": False,
                "message": "Failed to save conversation."
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error saving conversation: {str(e)}"
        }