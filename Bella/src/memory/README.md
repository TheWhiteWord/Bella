# Bella Memory System

This directory contains the components responsible for Bella's memory capabilities, allowing the assistant to store, retrieve, and utilize information from past interactions and structured knowledge.

## Core Components

1.  **`MemoryManager` (`memory_manager.py`)**:
    *   **Purpose**: Handles the low-level storage and organization of memory data. It manages the physical files and directories where memories are stored.
    *   **Key Functions**:
        *   `__init__(...)`: Initializes the memory directory structure (e.g., creating folders like `conversations`, `facts`, `preferences`). Loads existing memories into an in-memory index (`_memory_graph`, `_memory_index`) for faster access.
        *   `create_memory(...)`: Creates or updates a memory entry as a Markdown file within the appropriate folder. Handles metadata (title, timestamps, tags) using YAML frontmatter. Parses content to update the internal `_memory_graph`.
        *   `read_memory(...)`: Retrieves a specific memory file's content and metadata based on its title, path, or a `memory://` URL.
        *   `search_memories(...)`: Performs a text-based search across all memory files, returning ranked and paginated results with snippets.
        *   `update_memory(...)`: Modifies the content or title of an existing memory file.
        *   `delete_memory(...)`: Removes a memory file and updates the internal index.
        *   `_load_memories()`: Scans the memory directory on startup to populate the `_memory_graph` and `_memory_index`.
        *   `_parse_memory_content(...)`: Extracts structured information (observations, relations, tags) from the Markdown content of a memory file.

2.  **`EnhancedMemory` (`enhanced_memory.py`)**:
    *   **Purpose**: Implements semantic embedding-based memory search capabilities using embedding models from Ollama.
    *   **Key Classes**:
        *   `EmbeddingModelManager`: Manages different embedding models (primary "nomic-embed-text" and fast "all-minilm").
        *   `EnhancedMemoryProcessor`: Processes text with semantic embeddings for memory operations.
    *   **Key Functions**:
        *   `generate_embedding(...)`: Generates embedding vectors for text using configurable models.
        *   `find_relevant_memories(...)`: Performs semantic search across memories using vector similarity.
        *   `score_memory_importance(...)`: Evaluates the importance of potential memories.
        *   `extract_summary(...)`: Summarizes text for memory storage using Ollama's "summary" model.

3.  **`EnhancedMemoryAdapter` (`enhanced_memory_adapter.py`)**:
    *   **Purpose**: Connects the semantic embedding system with the file-based memory storage.
    *   **Key Functions**:
        *   `search_memory(...)`: Searches for memories using semantic embeddings.
        *   `should_store_memory(...)`: Determines if text should be stored based on semantic importance.
        *   `process_conversation_turn(...)`: Analyzes conversation for memory operations.
        *   `compare_memory_similarity(...)`: Compares semantic similarity between text snippets.
        *   `detect_memory_topics(...)`: Extracts topics from text using semantic understanding.

4.  **`AutonomousMemory` (`autonomous_memory.py`)**:
    *   **Purpose**: Implements the logic for *autonomous* memory integration during conversations. It decides *when* to retrieve relevant memories to enhance context and *when* to potentially store new information from the current interaction, without explicit user commands.
    *   **Key Functions**:
        *   `__init__(...)`: Sets up parameters controlling memory behavior, such as relevance thresholds (`memory_threshold`), frequency of checks (`memory_check_interval`), and recall limits (`max_recalls_per_session`).
        *   `process_conversation_turn(...)`: The main entry point called by the `MemoryConversationAdapter`. It handles both pre-processing (retrieving context before LLM generation) and post-processing (evaluating the turn for potential storage after LLM generation).
        *   `_is_memory_relevant_to_query(...)`: Uses semantic embedding similarity to determine if a memory is relevant to the current query.
        *   `_should_augment_with_memory(...)`: Determines if the context should be augmented with memories.
        *   `_calculate_memory_confidence(...)`: Uses semantic similarity to establish confidence in memory relevance.

5.  **`MemoryConversationAdapter` (`memory_conversation_adapter.py`)**:
    *   **Purpose**: Acts as a bridge between the main application loop (`main.py`) and the `AutonomousMemory` system. It provides hooks to integrate memory processing seamlessly into the request-response cycle.
    *   **Key Functions**:
        *   `pre_process_input(...)`: Called *before* sending the user input to the LLM to retrieve relevant memory context.
        *   `post_process_response(...)`: Called *after* the LLM generates a response to potentially store information from the conversation.

6.  **`ProjectManager` (`project_manager/project_manager.py`)**:
    *   **Purpose**: Manages project-based memory storage, allowing for organization of ideas and information into distinct projects with standardized folder structures.
    *   **Key Functions**:
        *   `__init__(...)`: Initializes the project directory structure and keeps track of the active project.
        *   `start_project(...)`: Creates a new project or activates an existing one with standard subfolders (notes, concepts, research, content).
        *   `save_to(...)`: Saves content to a project file with standardized formatting.
        *   `save_conversation(...)`: Saves conversation context to a project file.
        *   `edit(...)`: Edits existing entries in project files.
        *   `list_all(...)`: Lists entries in a project file, optionally filtered by query.
        *   `read_entry(...)`: Reads a specific entry from a project file.
        *   `delete_entry(...)`: Deletes an entry from a project file.
        *   `quit_project(...)`: Closes the active project.

7.  **`MemoryFormatAdapter` (`project_manager/memory_format_adapter.py`)**:
    *   **Purpose**: Standardizes memory storage formats between the autonomous memory system and the project-based system.
    *   **Key Functions**:
        *   `convert_to_standard_format(...)`: Converts memory content to the standardized format with YAML frontmatter.
        *   `extract_standard_format_data(...)`: Extracts metadata and content from standardized format.
        *   `is_standard_format(...)`: Checks if content is in the standardized format.
        *   `update_standard_format(...)`: Updates content while maintaining the standardized format.

8.  **`MemoryIntegration` (`project_manager/memory_integration.py`)**:
    *   **Purpose**: Integrates the autonomous memory system with the project-based memory system.
    *   **Key Functions**:
        *   `save_standardized_memory(...)`: Saves memory in standardized format.
        *   `read_standardized_memory(...)`: Reads memory in standardized format.
        *   `convert_existing_memories(...)`: Converts existing memories to standardized format (placeholder for future use).

9.  **`register_memory_tools.py` & `project_manager/register_project_tools.py`**:
    *   **Purpose**: Registers memory and project management tools with Bella's function calling system.
    *   **Available Tools**:
        *   `semantic_memory_search`: Searches across all memories using semantic understanding.
        *   `continue_conversation`: Continues a conversation thread based on recent exchanges.
        *   `start_project`: Creates or activates a project.
        *   `save_to`: Saves content to a project file.
        *   `save_conversation`: Saves conversation context to a project file.
        *   `edit_entry`: Edits existing entries in project files.
        *   `list_all`: Lists entries in a project file.
        *   `read_entry`: Reads specific entries from project files.
        *   `delete_entry`: Deletes entries from project files.
        *   `quit_project`: Closes the active project.

## Embedding Models

The enhanced memory system supports multiple embedding models from Ollama:

1. **Primary Model**: `nomic-embed-text` (768-dimensional vectors) - Used for high-quality semantic search and memory operations.
2. **Fast Model**: `all-minilm` (384-dimensional vectors) - Used for faster operations where speed is more important than maximum accuracy.
3. **Summary Model**: `summary` - Used for generating concise memory summaries.

These models provide a balance of accuracy and performance, with automatic fallback mechanisms:
- Semantic similarity is used first for relevance detection
- If embeddings fail, rule-based methods provide a robust fallback
- Caching reduces redundant embedding calculations

## Interaction Flow

### Autonomous Memory

1.  **User Input**: The user speaks, and the input is transcribed (`main.py`).
2.  **Pre-Processing**: Before calling the LLM, `main.py` calls `MemoryConversationAdapter.pre_process_input(user_input, history)`.
3.  **Context Retrieval**: `MemoryConversationAdapter` calls `AutonomousMemory.process_conversation_turn(user_input, response_text=None)`.
4.  **Semantic Search**: `AutonomousMemory` uses embedding-based search to find relevant memories.
5.  **LLM Call**: `main.py` sends the user input, conversation history, and any retrieved memory context to the LLM.
6.  **LLM Response**: The LLM generates a response.
7.  **Post-Processing**: After receiving the response, `main.py` calls `MemoryConversationAdapter.post_process_response(user_input, assistant_response)`.
8.  **Memory Storage**: Important information is identified using semantic models and stored if it meets relevance criteria.
9.  **TTS Output**: The assistant's response is converted to speech (`main.py`).

### Project-Based Memory (Function Calling)

1.  **User Input**: The user requests a project-related action (e.g., "Let's start a new project titled Art History").
2.  **LLM Function Calling**: The LLM recognizes the intent and calls the appropriate project management function (e.g., `start_project`).
3.  **Project Management**: The `ProjectManager` executes the requested action (creates a new project, saves content, etc.).
4.  **Response**: The assistant describes the action taken and the result.

## Project-Based Memory Structure

The project-based memory system organizes information into projects, each with standardized folders:

```
memories/
  conversations/     # Autonomous memory folders
  facts/
  preferences/
  reminders/
  general/
  projects/          # Project-based memory folder
    project-name-1/
      notes/         # Various markdown files
      concepts/      # Various markdown files
      research/      # Various markdown files
      content/       # Various markdown files
      main.md        # Main project file
      README.md      # Project overview
    project-name-2/
      ...
```

Each project file uses a standardized format with YAML frontmatter containing metadata:

```yaml
---
title: Sample Entry
created: 2025-04-07T15:30:00
updated: 2025-04-07T15:35:00
project: project-name
file_type: notes
tags: [sample, example, test]
entry_id: abc123def45
---

# Sample Entry

This is the content of the entry...
```

This system allows for both autonomous memory capabilities and explicit project management, providing a comprehensive memory solution for Bella with semantic understanding powered by embedding models.
