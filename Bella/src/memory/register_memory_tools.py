"""Registers memory-related tools for the LLM."""

import logging
from typing import Dict, Any, List, Optional, Tuple
import json
import os
import re
import time

# Assuming memory_adapter is the singleton instance from enhanced_memory_adapter
from .enhanced_memory_adapter import memory_adapter
from .memory_api import list_notes, read_note, save_note, delete_note
from ..llm.tools_registry import registry

# --- Semantic Search Tool (Updated for ChromaDB) ---
@registry.register_tool()
async def semantic_memory_search(query: str, top_n: int = 5) -> Dict[str, Any]:
    """
    Performs a semantic search across stored memories (conversations, facts, preferences, etc.)
    to find entries most relevant to the user's query based on meaning, not just keywords.

    Args:
        query (str): The natural language query to search for.
        top_n (int): The maximum number of relevant memory entries to return. Defaults to 5.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'success' (bool): True if the search was performed, False otherwise.
            - 'results' (List[Dict]): A list of the top N most relevant memories found.
              Each memory dictionary includes:
                - 'id' (str): The unique ID of the memory.
                - 'title' (str): The title of the memory entry.
                - 'score' (float): A similarity score (0-1, higher is more relevant).
                - 'path' (str): The file path to the original memory file.
                - 'tags' (List[str]): Associated tags.
                - 'created_at' (str): ISO timestamp of creation.
                - 'preview' (str): A short preview or description (may be placeholder).
            - 'message' (str, optional): A message indicating outcome (e.g., "No relevant memories found.").
    """
    logging.info(f"Executing semantic_memory_search tool with query: '{query[:50]}...', top_n={top_n}")
    try:
        # Ensure the adapter is initialized
        if not memory_adapter._initialized:
            logging.warning("Memory adapter not initialized, attempting initialization.")
            init_success = await memory_adapter.initialize()
            if not init_success:
                logging.error("Failed to initialize memory adapter for semantic search.")
                return {"success": False, "message": "Memory system initialization failed.", "results": []}

        # Call the adapter's search method which now uses ChromaDB
        search_results_dict, success = await memory_adapter.search_memory(query, top_n=top_n)

        if not success:
            logging.error("Semantic search via adapter failed.")
            return {"success": False, "message": "Semantic search encountered an error.", "results": []}

        results = search_results_dict.get("results", [])

        if not results:
            logging.info("Semantic search completed successfully but found no relevant memories.")
            return {"success": True, "message": "No relevant memories found.", "results": []}
        else:
            logging.info(f"Semantic search successful, returning {len(results)} results.")
            # The results from search_memory should already be in the desired format.
            # We might want to refine the 'preview' here if needed, e.g., by reading the file.
            # For now, return the results as they are.
            return {"success": True, "results": results}

    except Exception as e:
        logging.exception(f"Error executing semantic_memory_search tool: {e}")
        return {"success": False, "message": f"An unexpected error occurred: {e}", "results": []}


@registry.register_tool()
async def continue_conversation(thread_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieves the recent history of a conversation thread to allow the assistant
    to continue the discussion contextually. If no thread_id is provided,
    it attempts to retrieve the most recent conversation.

    Args:
        thread_id (Optional[str]): The specific conversation thread ID (usually a filename or title)
                                   to retrieve. If None, fetches the latest conversation.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'success' (bool): True if conversation history was found, False otherwise.
            - 'thread_id' (str): The ID of the conversation thread retrieved.
            - 'history' (str): The content of the conversation history.
            - 'message' (str, optional): Status message.
    """
    logging.info(f"Executing continue_conversation tool for thread_id: {thread_id}")
    try:
        if thread_id:
            # Attempt to read the specific conversation note
            # Assuming thread_id might be like 'conversation-topic-title'
            path = os.path.join("memories", "conversations", thread_id + ".md")
            if not os.path.exists(path):
                 # Try finding based on title/ID if path doesn't match directly
                 notes = await list_notes("conversations", query=thread_id)
                 if notes:
                     path = notes[0].get("path") # Use the first match
                 else:
                     path = None

            if path and os.path.exists(path):
                content = await read_note(path)
                if content:
                    logging.info(f"Retrieved specific conversation thread: {thread_id}")
                    return {"success": True, "thread_id": thread_id, "history": content}
                else:
                    msg = f"Could not read content for conversation thread: {thread_id}"
                    logging.warning(msg)
                    return {"success": False, "message": msg}
            else:
                msg = f"Conversation thread not found: {thread_id}"
                logging.warning(msg)
                return {"success": False, "message": msg}
        else:
            # Find the most recent conversation
            notes = await list_notes("conversations", sort_by="date", limit=1)
            if notes:
                latest_note = notes[0]
                path = latest_note.get("path")
                retrieved_id = latest_note.get("name")
                content = await read_note(path)
                if content:
                    logging.info(f"Retrieved most recent conversation thread: {retrieved_id}")
                    return {"success": True, "thread_id": retrieved_id, "history": content}
                else:
                    msg = f"Could not read content for the latest conversation: {retrieved_id}"
                    logging.warning(msg)
                    return {"success": False, "message": msg}
            else:
                msg = "No conversation history found."
                logging.info(msg)
                return {"success": False, "message": msg}
    except Exception as e:
        logging.exception(f"Error executing continue_conversation tool: {e}")
        return {"success": False, "message": f"An unexpected error occurred: {e}"}

@registry.register_tool()
async def save_to_memory(content: str, memory_type: str = "facts", title: str = None) -> Dict[str, Any]:
    """
    Saves important information to memory for future recall.
    
    Args:
        content (str): The information to save in memory.
        memory_type (str): Type of memory to save as. Options: "facts", "preferences", "general", "reminders".
        title (str, optional): A descriptive title for the memory. If not provided, one will be generated.
        
    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'success' (bool): True if the memory was saved, False otherwise.
            - 'memory_id' (str): ID of the saved memory.
            - 'file_path' (str): Path where the memory was saved.
            - 'message' (str): Status message.
    """
    logging.info(f"Executing save_to_memory with type: '{memory_type}', title: '{title}'")
    try:
        # Validate memory type
        valid_memory_types = ["facts", "preferences", "general", "reminders"]
        if memory_type not in valid_memory_types:
            memory_type = "general"  # Default to general if invalid type
        
        # Format content if needed
        formatted_content = content.strip()
        
        # Generate title if not provided
        if not title:
            # Extract a title from content
            words = re.findall(r'\b[a-z]{3,}\b', formatted_content.lower())
            filtered_words = [w for w in words if w not in {"the", "and", "that", "this", "with", "from", "what", "when", "where", "would", "could", "should"}]
            if filtered_words:
                title = "-".join(filtered_words[:3])
            else:
                from datetime import datetime
                title = f"{memory_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Clean title for filesystem safety
        safe_title = re.sub(r'[^\w\-]', '-', title)
        
        # Save the memory using the memory adapter
        success, path = await memory_adapter.store_memory(memory_type, formatted_content, safe_title)
        
        if success and path:
            memory_id = f"{memory_type}/{safe_title}"
            return {
                "success": True, 
                "memory_id": memory_id,
                "file_path": path,
                "message": f"I've saved this information to my {memory_type} memory as '{title}'."
            }
        else:
            return {"success": False, "message": f"I wasn't able to save that information to my {memory_type} memory."}
            
    except Exception as e:
        logging.exception(f"Error executing save_to_memory: {e}")
        return {"success": False, "message": f"I encountered an error while trying to save to memory: {str(e)}"}

@registry.register_tool()
async def save_conversation(context: str, title: str = None, file_type: str = "conversations") -> Dict[str, Any]:
    """
    Saves a conversation snippet or context to memory for future reference.
    
    Args:
        context (str): The conversation context or content to save. Can be provided as a string or JSON string of messages.
        title (str, optional): A descriptive title for the conversation memory. If not provided, one will be generated.
        file_type (str, optional): Type of memory to save as. Defaults to "conversations".
        
    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'success' (bool): True if the memory was saved, False otherwise.
            - 'file_path' (str, optional): Path where the memory was saved.
            - 'memory_id' (str, optional): ID of the saved memory.
            - 'message' (str): Status message.
    """
    logging.info(f"Executing save_conversation tool with title: '{title}', file_type: '{file_type}'")
    try:
        # Parse context if it's a JSON string
        parsed_context = context
        if isinstance(context, str):
            try:
                # Try to parse as JSON
                parsed_data = json.loads(context)
                if isinstance(parsed_data, list) or isinstance(parsed_data, dict):
                    # Format the context nicely
                    if isinstance(parsed_data, list):
                        parsed_context = "\n\n".join(str(item) for item in parsed_data)
                    else:
                        parsed_context = json.dumps(parsed_data, indent=2)
            except json.JSONDecodeError:
                # Not JSON, use as is
                parsed_context = context

        # Ensure valid memory type
        valid_memory_types = ["conversations", "facts", "preferences", "general", "notes"]
        if file_type not in valid_memory_types:
            file_type = "conversations"  # Default to conversations
        
        # Generate title if not provided
        if not title:
            # Extract a title from content
            words = re.findall(r'\b[a-z]{3,}\b', parsed_context.lower())
            filtered_words = [w for w in words if w not in {"the", "and", "that", "this", "with", "from", "what", "when", "where", "would", "could", "should"}]
            if filtered_words:
                title = "-".join(filtered_words[:3])
            else:
                from datetime import datetime
                title = f"conversation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Clean title for filesystem safety
        safe_title = re.sub(r'[^\w\-]', '-', title)
        
        # Save the memory
        success, path = await memory_adapter.store_memory(file_type, parsed_context, safe_title)
        
        if success and path:
            memory_id = f"{file_type}/{safe_title}"
            return {
                "success": True, 
                "file_path": path, 
                "memory_id": memory_id,
                "message": f"Conversation saved successfully as '{title}'"
            }
        else:
            return {"success": False, "message": "Failed to save conversation memory"}
            
    except Exception as e:
        logging.exception(f"Error executing save_conversation tool: {e}")
        return {"success": False, "message": f"An unexpected error occurred: {str(e)}"}

@registry.register_tool()
async def read_specific_memory(memory_id: str) -> Dict[str, Any]:
    """
    Retrieves a specific memory by its ID or path.
    
    Args:
        memory_id (str): The memory ID in format 'type/name' (e.g., 'facts/favorite-color') or full path.
        
    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'success' (bool): True if the memory was found and read, False otherwise.
            - 'memory_id' (str): ID of the memory.
            - 'content' (str): The content of the memory.
            - 'memory_type' (str): Type of memory (facts, preferences, etc.)
            - 'created_at' (str, optional): When the memory was created.
            - 'message' (str): Status message or error.
    """
    logging.info(f"Executing read_specific_memory with ID: '{memory_id}'")
    try:
        # Determine if memory_id is a full path or an ID
        memory_path = None
        
        if os.path.exists(memory_id):
            # It's a full path
            memory_path = memory_id
        else:
            # Try to find it by ID
            if "/" in memory_id:
                # Format is likely type/name
                memory_type, name = memory_id.split("/", 1)
                path = os.path.join("memories", memory_type, f"{name}.md")
                if os.path.exists(path):
                    memory_path = path
            else:
                # Just a name, search in all memory types
                for memory_type in memory_adapter.memory_dirs:
                    path = os.path.join("memories", memory_type, f"{memory_id}.md")
                    if os.path.exists(path):
                        memory_path = path
                        break
        
        if not memory_path:
            return {
                "success": False,
                "memory_id": memory_id,
                "message": f"I couldn't find a memory with ID '{memory_id}' in my memory storage."
            }
            
        # Read the memory content
        content = await read_note(memory_path)
        
        if not content:
            return {
                "success": False,
                "memory_id": memory_id,
                "message": f"The memory file exists but I couldn't read its content."
            }
            
        # Extract metadata from path
        path_parts = os.path.normpath(memory_path).split(os.sep)
        if len(path_parts) >= 2:
            memory_type = path_parts[-2]
        else:
            memory_type = "unknown"
            
        basename = os.path.basename(memory_path)
        name = os.path.splitext(basename)[0]
        
        # Record access in the adapter for tracking frequently used memories
        memory_adapter.record_memory_access(memory_id)
        
        return {
            "success": True,
            "memory_id": f"{memory_type}/{name}",
            "content": content,
            "memory_type": memory_type,
            "message": f"Successfully retrieved memory {memory_type}/{name}"
        }
            
    except Exception as e:
        logging.exception(f"Error executing read_specific_memory: {e}")
        return {
            "success": False,
            "memory_id": memory_id, 
            "message": f"I encountered an error while trying to read the memory: {str(e)}"
        }

@registry.register_tool()
async def list_memories_by_type(memory_type: str = "all", query: str = None, limit: int = 10) -> Dict[str, Any]:
    """
    Lists available memories filtered by type and optional search query.
    
    Args:
        memory_type (str): Type of memories to list. Options: "facts", "preferences", "conversations",
                          "reminders", "general", or "all" for all types.
        query (str, optional): Optional search term to filter memories.
        limit (int): Maximum number of memories to return per type.
        
    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'success' (bool): True if the operation was successful.
            - 'memories' (Dict[str, List]): Dictionary of memory types with lists of memories.
            - 'total_count' (int): Total number of memories found.
            - 'message' (str): Status message.
    """
    logging.info(f"Executing list_memories_by_type with type: '{memory_type}', query: '{query}'")
    try:
        memory_types = memory_adapter.memory_dirs if memory_type == "all" else [memory_type]
        
        # Validate memory type
        if memory_type != "all" and memory_type not in memory_adapter.memory_dirs:
            return {
                "success": False,
                "message": f"Invalid memory type: '{memory_type}'. Valid types are: {', '.join(memory_adapter.memory_dirs)} or 'all'."
            }
        
        result = {
            "success": True,
            "memories": {},
            "total_count": 0,
            "message": ""
        }
        
        # Process each requested memory type
        for mem_type in memory_types:
            # Get list of notes for this type
            notes = await list_notes(mem_type, query, sort_by="date", limit=limit)
            
            if notes:
                # Format the notes
                formatted_notes = []
                for note in notes:
                    formatted_notes.append({
                        "id": f"{mem_type}/{note.get('name', 'unknown')}",
                        "title": note.get('title', note.get('name', 'Untitled')),
                        "path": note.get('path', ''),
                        "created": note.get('created_at', 'Unknown'),
                    })
                
                result["memories"][mem_type] = formatted_notes
                result["total_count"] += len(formatted_notes)
        
        # Create appropriate message
        if result["total_count"] == 0:
            if query:
                result["message"] = f"I couldn't find any memories matching '{query}'."
            else:
                result["message"] = f"I don't have any {memory_type} memories stored yet."
        else:
            if query:
                result["message"] = f"I found {result['total_count']} memories matching '{query}'."
            else:
                result["message"] = f"I have {result['total_count']} memories available."
            
        return result
            
    except Exception as e:
        logging.exception(f"Error executing list_memories_by_type: {e}")
        return {
            "success": False,
            "message": f"I encountered an error while listing memories: {str(e)}"
        }

# Note: Add other memory-related tools here if needed (e.g., save_fact, recall_preference)
# Ensure they interact correctly with the memory_adapter or memory_api as appropriate.