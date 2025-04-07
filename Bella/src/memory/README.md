# Bella Memory System

This directory contains the components responsible for Bella's memory capabilities, allowing the assistant to store, retrieve, and utilize information from past interactions and structured knowledge using semantic understanding powered by embeddings and ChromaDB.

## Core Components

1.  **`MemoryManager` (`memory_manager.py`)**:
    *   **Purpose**: Handles the low-level storage and organization of memory data. It manages the physical files and directories where memories are stored (primarily as Markdown `.md` files).
    *   **Key Functions**:
        *   `__init__(...)`: Initializes the memory directory structure.
        *   `create_memory(...)`: Creates or updates a memory entry as a Markdown file with YAML frontmatter.
        *   `read_memory(...)`: Retrieves a specific memory file's content and metadata.
        *   `search_memories(...)`: Performs text-based search (keyword, regex) across memory files. *(Note: Semantic search is handled by `EnhancedMemoryAdapter`)*.
        *   `update_memory(...)`: Modifies an existing memory file.
        *   `delete_memory(...)`: Removes a memory file.
        *   `_load_memories()`: Scans the memory directory (less critical with vector DB indexing).
        *   `_parse_memory_content(...)`: Extracts structured information from Markdown content.

2.  **`EnhancedMemory` (`enhanced_memory.py`)**:
    *   **Purpose**: Implements semantic embedding generation using models (e.g., from Ollama).
    *   **Key Classes**:
        *   `EmbeddingModelManager`: Manages different embedding models.
        *   `EnhancedMemoryProcessor`: Processes text to generate embeddings.
    *   **Key Functions**:
        *   `generate_embedding(...)`: Generates embedding vectors for text.
        *   `score_memory_importance(...)`: Evaluates the semantic importance of potential memories.
        *   `extract_summary(...)`: Summarizes text for memory storage.

3.  **`EnhancedMemoryAdapter` (`enhanced_memory_adapter.py`)**:
    *   **Purpose**: Connects the semantic embedding system (`EnhancedMemory`) with memory storage and retrieval mechanisms. **Uses ChromaDB for efficient semantic search and indexing.**
    *   **Key Functions**:
        *   `__init__(...)`: Initializes ChromaDB client and collection (e.g., `chromadb.PersistentClient` pointing to a local path, gets or creates collection like "bella_memories").
        *   `search_memory(...)`: **Searches for memories by generating a query embedding and querying the ChromaDB collection (`collection.query`). Returns metadata (including file paths) and similarity scores from ChromaDB results.**
        *   `should_store_memory(...)`: Determines if text should be stored based on semantic importance score.
        *   `process_conversation_turn(...)`: Analyzes conversation for memory operations (calls `search_memory` or triggers storage).
        *   `compare_memory_similarity(...)`: Compares semantic similarity between text snippets using embeddings.
        *   `detect_memory_topics(...)`: Extracts topics from text using semantic understanding.
        *   `_add_memory_to_vector_db(...)`: Takes memory content, generates its embedding, and adds the embedding along with metadata (ID, title, tags, **file_path**) to the ChromaDB collection.

4.  **`AutonomousMemory` (`autonomous_memory.py`)**:
    *   **Purpose**: Implements the logic for *autonomous* memory integration during conversations. Decides *when* to retrieve relevant memories and *when* to store new information.
    *   **Key Functions**:
        *   `process_conversation_turn(...)`: Main entry point.
            *   **Retrieval (Pre-processing)**: Calls the `semantic_memory_search` tool (which uses `EnhancedMemoryAdapter.search_memory` querying ChromaDB) to find relevant memories based on the user query. Uses the returned file paths to potentially load full content from `.md` files if needed.
            *   **Storage (Post-processing)**: After determining a conversation turn should be stored and saving it to an `.md` file via `MemoryIntegration`, **it calls `EnhancedMemoryAdapter._add_memory_to_vector_db` to index the new memory in ChromaDB.**
        *   `_is_memory_relevant_to_query(...)`: Uses semantic similarity (via `EnhancedMemoryAdapter.compare_memory_similarity`).
        *   `_should_augment_with_memory(...)`: Determines if context should be augmented based on query intent and timing.
        *   `_calculate_memory_confidence(...)`: Uses semantic similarity for confidence.

5.  **`MemoryConversationAdapter` (`memory_conversation_adapter.py`)**:
    *   **Purpose**: Bridge between `main.py` and `AutonomousMemory`.
    *   **Key Functions**: `pre_process_input(...)`, `post_process_response(...)`.

6.  **`ProjectManager` (`project_manager/project_manager.py`)**:
    *   **Purpose**: Manages project-based memory storage in separate directories. *(Note: ChromaDB integration primarily targets the general/autonomous memory, but could be extended to projects)*.
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
    *   **Purpose**: Standardizes memory storage formats (YAML frontmatter in `.md` files).
    *   **Key Functions**:
        *   `convert_to_standard_format(...)`: Converts memory content to the standardized format with YAML frontmatter.
        *   `extract_standard_format_data(...)`: Extracts metadata and content from standardized format.
        *   `is_standard_format(...)`: Checks if content is in the standardized format.
        *   `update_standard_format(...)`: Updates content while maintaining the standardized format.

8.  **`MemoryIntegration` (`project_manager/memory_integration.py`)**:
    *   **Purpose**: Integrates autonomous and project-based memory storage (handles saving/reading `.md` files).
    *   **Key Functions**: `save_standardized_memory(...)`, `read_standardized_memory(...)`.

9.  **`register_memory_tools.py` & `project_manager/register_project_tools.py`**:
    *   **Purpose**: Registers memory/project tools for LLM function calling.
    *   **Available Tools**:
        *   `semantic_memory_search`: **Performs semantic search by triggering `EnhancedMemoryAdapter.search_memory`, which queries the ChromaDB vector database.**
        *   `continue_conversation`: Continues a conversation thread based on recent exchanges.
        *   `start_project`: Creates or activates a project.
        *   `save_to`: Saves content to a project file.
        *   `save_conversation`: Saves conversation context to a project file.
        *   `edit_entry`: Edits existing entries in project files.
        *   `list_all`: Lists entries in a project file.
        *   `read_entry`: Reads specific entries from project files.
        *   `delete_entry`: Deletes entries from project files.
        *   `quit_project`: Closes the active project.

## ChromaDB Integration (Completed)

To enhance semantic search capabilities and scalability, ChromaDB has been integrated as a vector database. Markdown (`.md`) files **are still used** as the primary, human-readable storage for memory content. ChromaDB acts as a fast, searchable **index** based on semantic meaning.

**Implementation Details:**

1.  **Dependency Added**: `chromadb` added to `requirements.txt`.
2.  **ChromaDB Client Initialized**:
    *   `EnhancedMemoryAdapter.__init__` initializes a `chromadb.PersistentClient` using the path specified in `config/models.yaml` (`memory.chromadb.path`).
    *   It gets or creates a collection named `bella_memories` (or as configured) using cosine distance.
3.  **Indexing Method Implemented**:
    *   `EnhancedMemoryAdapter._add_memory_to_vector_db` generates an embedding for memory content and adds the embedding along with metadata (including `file_path`) to the ChromaDB collection.
4.  **Storage Logic Updated**:
    *   `AutonomousMemory.process_conversation_turn` now calls `_add_memory_to_vector_db` after successfully saving the `.md` file, ensuring new memories are indexed.
5.  **Search Logic Updated**:
    *   `EnhancedMemoryAdapter.search_memory` generates a query embedding and queries the ChromaDB collection, returning processed results including metadata and similarity scores.
    *   `register_memory_tools.semantic_memory_search` uses the updated adapter method.
6.  **Configuration Added**: ChromaDB path and collection name are configured in `config/models.yaml` under the `memory.chromadb` section.
7.  **Testing Updated**: Tests in `tests/memory/test_autonomous_memory.py` and `tests/memory/test_enhanced_memory_integration.py` have been updated to mock ChromaDB interactions.

## Embedding Models

The enhanced memory system supports multiple embedding models from Ollama:

1. **Primary Model**: `nomic-embed-text` (768-dimensional vectors) - Used for high-quality semantic search and memory operations.
2. **Fast Model**: `all-minilm` (384-dimensional vectors) - Used for faster operations where speed is more important than maximum accuracy.
3. **Summary Model**: `summary` - Used for generating concise memory summaries.

The system has a clean hierarchical approach to memory operations:
1. Primary embedding-based similarity (nomic-embed-text) for highest accuracy
2. Fast embedding-based similarity (all-minilm) for time-sensitive operations
3. TF-IDF similarity as a robust fallback when embeddings are unavailable

This streamlined approach provides optimal performance while maintaining reliability across different scenarios.

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

## Dependencies

The Bella Memory System relies on the following Python libraries:

- **numpy**: For efficient vector operations in similarity calculations
- **scikit-learn**: Used for TF-IDF vectorization and similarity calculations
- **aiohttp**: For asynchronous API calls to Ollama embedding models
- **pandas**: (Optional) For data processing in memory analytics
- **chromadb**: For vector database indexing and semantic search
