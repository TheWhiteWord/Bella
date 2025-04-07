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
from ..llm.tools_registry import register_tool

# --- Semantic Search Tool (Updated for ChromaDB) ---
@register_tool
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


@register_tool
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

# Note: Add other memory-related tools here if needed (e.g., save_fact, recall_preference)
# Ensure they interact correctly with the memory_adapter or memory_api as appropriate.