"""Tests for the enhanced memory system integration and adapter functionality.

This module tests the integration between EnhancedMemoryProcessor,
EnhancedMemoryAdapter, and BellaMemoryManager, including ChromaDB interactions.
"""

import os
import sys
import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock
import tempfile
import shutil
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))

# Assuming EnhancedMemoryAdapter is the primary class to test with mocked dependencies
from src.memory.enhanced_memory_adapter import EnhancedMemoryAdapter
# Import other components if needed for specific integration tests
from src.memory.enhanced_memory import EnhancedMemoryProcessor
from src.memory.main_app_integration import BellaMemoryManager, ensure_memory_initialized
from src.memory.memory_conversation_adapter import MemoryConversationAdapter
from src.memory.register_memory_tools import semantic_memory_search

# Mock ChromaDB Collection results structure
class MockChromaResults:
    def __init__(self, ids=None, distances=None, metadatas=None):
        self.ids = [ids or []]
        self.distances = [distances or []]
        self.metadatas = [metadatas or []]

    def get(self, key, default=None):
        if key == 'ids':
            return self.ids
        elif key == 'distances':
            return self.distances
        elif key == 'metadatas':
            return self.metadatas
        return default

# Sample test data
test_memories = {
    "facts/python": "Python is a high-level, interpreted programming language known for its readability.",
    "preferences/coffee": "I prefer my coffee black with no sugar. I enjoy dark roast varieties the most.",
    "conversations/ai-ethics": "User: What do you think about AI ethics?\nAssistant: Ethics in AI is crucial as these systems become more integrated into society."
}

# Mock embeddings for testing (simplified)
mock_embedding_vector = [0.1, 0.2, 0.3, 0.4, 0.5] * 150 # Simulate a longer vector

# Mock memory directory fixture (remains useful)
@pytest.fixture
async def mock_memory_dir():
    """Create a temporary memory directory structure for testing."""
    temp_dir = tempfile.mkdtemp(prefix="bella_test_integration_")
    memories_dir = os.path.join(temp_dir, "memories")

    # Create memory type directories
    for mem_type in ["facts", "preferences", "conversations", "general", "reminders"]:
        os.makedirs(os.path.join(memories_dir, mem_type), exist_ok=True)

    # Create sample memory files
    for memory_id, content in test_memories.items():
        mem_type, name = memory_id.split("/")
        file_path = os.path.join(memories_dir, mem_type, f"{name}.md")
        with open(file_path, "w") as f:
            f.write(content)

    # Save current directory to restore later
    old_dir = os.getcwd()
    os.chdir(temp_dir) # Change CWD so relative paths like "memories/..." work

    yield temp_dir

    # Clean up
    os.chdir(old_dir)
    shutil.rmtree(temp_dir)


# Mock Ollama API fixture (remains useful)
@pytest.fixture
def mock_ollama_api():
    """Mock the Ollama API responses."""
    with patch("aiohttp.ClientSession") as mock_session:
        # Mock context manager
        mock_cm = MagicMock()
        mock_session.return_value.__aenter__.return_value = mock_cm

        # Mock tags response (model list)
        mock_tags_response = AsyncMock()
        mock_tags_response.status = 200
        mock_tags_response.json.return_value = {
            "models": [{"name": "nomic-embed-text"}]
        }
        mock_cm.get.return_value.__aenter__.return_value = mock_tags_response

        # Mock embeddings response
        mock_embed_response = AsyncMock()
        mock_embed_response.status = 200

        # Create dynamic response based on input
        async def mock_json():
            # Return a fixed mock embedding for simplicity in tests
            return {"embedding": mock_embedding_vector}

        mock_embed_response.json = mock_json
        mock_cm.post.return_value.__aenter__.return_value = mock_embed_response

        yield mock_session

# --- NEW: Mock ChromaDB Client and Collection ---
@pytest.fixture
def mock_chromadb():
    """Mocks chromadb client and collection interactions."""
    with patch('src.memory.enhanced_memory_adapter.chromadb.PersistentClient') as MockPersistentClient:
        mock_client_instance = MagicMock()
        mock_collection_instance = MagicMock()

        # Mock the get_or_create_collection method
        mock_client_instance.get_or_create_collection.return_value = mock_collection_instance

        # Mock collection methods (add, query, count)
        mock_collection_instance.add = MagicMock()
        # Configure query to return a MockChromaResults object
        mock_collection_instance.query = MagicMock(return_value=MockChromaResults()) # Default: empty results
        # Mock count
        type(mock_collection_instance).count = PropertyMock(return_value=0) # Mock count as a property

        # Make the PersistentClient constructor return our mock client
        MockPersistentClient.return_value = mock_client_instance

        # Yield the mocked collection instance so tests can configure/assert on it
        yield mock_collection_instance


# --- Test Class Focused on Adapter ---
# Renamed class for clarity, focusing on adapter tests now
class TestEnhancedMemoryAdapterWithMocks:
    """Test suite for EnhancedMemoryAdapter with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_adapter_initialization(self, mock_ollama_api, mock_chromadb, mock_memory_dir):
        """Test adapter initialization including ChromaDB client setup."""
        # Patch os.makedirs to avoid actual directory creation if needed, though mock_memory_dir handles temp dirs
        with patch('src.memory.enhanced_memory_adapter.os.makedirs'):
            adapter = EnhancedMemoryAdapter(embedding_model="nomic-embed-text")
            # Check ChromaDB client and collection were initialized
            assert adapter.chroma_client is not None
            assert adapter.chroma_collection is mock_chromadb # Check it got the mocked collection
            adapter.chroma_client.get_or_create_collection.assert_called_once_with(
                name="bella_memories",
                metadata={"hnsw:space": "cosine"}
            )

            # Test the main initialize method
            success = await adapter.initialize()
            assert success is True
            # Check Ollama model check was called
            mock_ollama_api.return_value.__aenter__.return_value.get.assert_called_once()


    @pytest.mark.asyncio
    async def test_add_memory_to_vector_db(self, mock_ollama_api, mock_chromadb, mock_memory_dir):
        """Test adding a memory to the mocked ChromaDB collection."""
        adapter = EnhancedMemoryAdapter(embedding_model="nomic-embed-text")
        await adapter.initialize() # Initialize adapter (uses mocks)

        memory_id = "test/memory-1"
        content = "This is the content to be embedded."
        metadata = {
            "file_path": f"{mock_memory_dir}/memories/test/memory-1.md",
            "title": "Test Memory 1",
            "tags": ["testing", "chroma"],
            "created_at": datetime.now()
        }

        # Call the internal method directly for focused testing
        await adapter._add_memory_to_vector_db(memory_id, content, metadata)

        # Assert that the processor was called to generate embedding
        adapter.processor.generate_embedding.assert_called_once_with(content)

        # Assert that the mocked collection's add method was called correctly
        mock_chromadb.add.assert_called_once()
        call_args, call_kwargs = mock_chromadb.add.call_args
        assert call_kwargs['ids'] == [memory_id]
        assert call_kwargs['embeddings'] == [mock_embedding_vector] # Check against the mocked embedding
        # Check metadata conversion
        added_metadata = call_kwargs['metadatas'][0]
        assert added_metadata['file_path'] == metadata['file_path']
        assert added_metadata['title'] == metadata['title']
        assert added_metadata['tags'] == "testing,chroma" # Check list conversion
        assert isinstance(added_metadata['created_at'], str) # Check datetime conversion


    @pytest.mark.asyncio
    async def test_search_memory_chromadb(self, mock_ollama_api, mock_chromadb, mock_memory_dir):
        """Test searching memory using the mocked ChromaDB collection."""
        adapter = EnhancedMemoryAdapter(embedding_model="nomic-embed-text")
        await adapter.initialize()

        query = "Search for relevant info"
        top_n = 3

        # Configure the mock collection's query response for this test
        mock_query_results = MockChromaResults(
            ids=["conv/result1", "facts/result2"],
            distances=[0.1, 0.25], # Lower distance = more similar
            metadatas=[
                {"file_path": "path/to/result1.md", "title": "Result One", "tags": "tagA,tagB"},
                {"file_path": "path/to/result2.md", "title": "Result Two", "tags": "tagC"}
            ]
        )
        mock_chromadb.query.return_value = mock_query_results

        # Call the search method
        results_dict, success = await adapter.search_memory(query, top_n=top_n)

        # Assert embedding was generated for the query
        adapter.processor.generate_embedding.assert_called_once_with(query)

        # Assert ChromaDB query was called correctly
        mock_chromadb.query.assert_called_once()
        call_args, call_kwargs = mock_chromadb.query.call_args
        assert call_kwargs['query_embeddings'] == [mock_embedding_vector]
        assert call_kwargs['n_results'] == top_n
        assert call_kwargs['include'] == ['metadatas', 'distances']

        # Assert results are processed correctly
        assert success is True
        assert len(results_dict["results"]) == 2
        # Check first result formatting
        res1 = results_dict["results"][0]
        assert res1["id"] == "conv/result1"
        assert res1["title"] == "Result One"
        assert res1["path"] == "path/to/result1.md"
        assert res1["tags"] == ["tagA", "tagB"] # Check tag string splitting
        assert pytest.approx(res1["score"]) == 0.9 # Similarity = 1 - distance
        assert res1["distance"] == 0.1
        # Check second result
        res2 = results_dict["results"][1]
        assert res2["id"] == "facts/result2"
        assert pytest.approx(res2["score"]) == 0.75


    @pytest.mark.asyncio
    async def test_store_memory_calls_add_to_db(self, mock_ollama_api, mock_chromadb, mock_memory_dir):
        """Test that adapter.store_memory calls _add_memory_to_vector_db."""
        adapter = EnhancedMemoryAdapter(embedding_model="nomic-embed-text")
        await adapter.initialize()

        # Mock the internal _add_memory_to_vector_db to verify it's called
        with patch.object(adapter, '_add_memory_to_vector_db', new_callable=AsyncMock) as mock_add_method:
            # Mock save_note from memory_api which is called by store_memory
            with patch('src.memory.enhanced_memory_adapter.save_note', new_callable=AsyncMock) as mock_save_note:
                mock_save_note.return_value = f"{mock_memory_dir}/memories/facts/python-is-fun.md" # Simulate successful save

                content = "Python is fun"
                mem_type = "facts"
                note_name = "python-is-fun"
                success, path = await adapter.store_memory(mem_type, content, note_name)

                assert success is True
                assert path == mock_save_note.return_value
                mock_save_note.assert_called_once_with(content, mem_type, note_name)
                # Verify _add_memory_to_vector_db was called
                mock_add_method.assert_called_once()
                call_args, call_kwargs = mock_add_method.call_args
                assert call_kwargs['memory_id'] == f"{mem_type}/{note_name}"
                assert call_kwargs['content'] == content
                assert call_kwargs['metadata']['file_path'] == path


    # --- Keep relevant integration tests if desired, ensuring mocks are used ---

    @pytest.mark.asyncio
    async def test_semantic_memory_search_tool(self, mock_ollama_api, mock_chromadb, mock_memory_dir):
        """Test the semantic_memory_search tool uses the mocked adapter search."""
        # Configure the mock adapter's search_memory response
        mock_adapter_search_response = ({"results": [
            {"id": "test/result", "score": 0.85, "path": "...", "title": "Test Result"}
        ]}, True)

        # Patch the singleton memory_adapter instance used by the tool
        with patch('src.memory.register_memory_tools.memory_adapter') as mock_adapter_instance:
            # Configure the mock adapter instance's methods
            mock_adapter_instance._initialized = True # Assume initialized
            mock_adapter_instance.search_memory = AsyncMock(return_value=mock_adapter_search_response)

            query = "find test data"
            tool_result = await semantic_memory_search(query, top_n=1)

            # Verify the adapter's search method was called
            mock_adapter_instance.search_memory.assert_called_once_with(query, top_n=1)

            # Verify the tool result format
            assert tool_result["success"] is True
            assert len(tool_result["results"]) == 1
            assert tool_result["results"][0]["id"] == "test/result"
            assert tool_result["results"][0]["score"] == 0.85


    # Can add more tests for enhance_memory_retrieval, etc., ensuring ChromaDB mocks are used

if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main(['-xvs', __file__])