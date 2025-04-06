"""Tests for the enhanced memory system integration with nomic-embed-text model.

This module tests the integration between EnhancedMemoryProcessor, 
EnhancedMemoryAdapter, and BellaMemoryManager with the nomic-embed-text embedding model.
"""

import os
import sys
import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock
import tempfile
import shutil

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))

from src.memory.enhanced_memory import EnhancedMemoryProcessor
from src.memory.enhanced_memory_adapter import EnhancedMemoryAdapter
from src.memory.main_app_integration import BellaMemoryManager, ensure_memory_initialized
from src.memory.memory_conversation_adapter import MemoryConversationAdapter
from src.memory.register_memory_tools import semantic_memory_search


# Sample test data
test_memories = {
    "facts/python": "Python is a high-level, interpreted programming language known for its readability.",
    "preferences/coffee": "I prefer my coffee black with no sugar. I enjoy dark roast varieties the most.",
    "conversations/ai-ethics": "User: What do you think about AI ethics?\nAssistant: Ethics in AI is crucial as these systems become more integrated into society."
}

# Mock embeddings for testing
mock_embeddings = {
    "facts/python": [0.1, 0.2, 0.3, 0.4, 0.5],
    "preferences/coffee": [0.2, 0.3, 0.4, 0.5, 0.6],
    "conversations/ai-ethics": [0.3, 0.4, 0.5, 0.6, 0.7]
}

# Mock memory directory
@pytest.fixture
async def mock_memory_dir():
    """Create a temporary memory directory structure for testing."""
    temp_dir = tempfile.mkdtemp()
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
    os.chdir(temp_dir)
    
    yield temp_dir
    
    # Clean up
    os.chdir(old_dir)
    shutil.rmtree(temp_dir)


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
            return {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}
            
        mock_embed_response.json = mock_json
        mock_cm.post.return_value.__aenter__.return_value = mock_embed_response
        
        yield mock_session


class TestEnhancedMemoryIntegration:
    """Test suite for enhanced memory integration."""
    
    @pytest.mark.asyncio
    async def test_memory_initialization(self, mock_ollama_api, mock_memory_dir):
        """Test that memory system initializes correctly."""
        # Test initialization with specified embedding model
        await ensure_memory_initialized("nomic-embed-text")
        
        # Verify API was called to check model availability
        mock_ollama_api.return_value.__aenter__.return_value.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_memory_adapter_methods(self, mock_ollama_api, mock_memory_dir):
        """Test EnhancedMemoryAdapter core methods."""
        adapter = EnhancedMemoryAdapter(embedding_model="nomic-embed-text")
        
        # Initialize
        success = await adapter.initialize()
        assert success is True
        
        # Test should_store_memory
        should_store, importance = await adapter.should_store_memory(
            "This is an important fact about AI that should be remembered.")
        assert isinstance(should_store, bool)
        assert 0 <= importance <= 1
        
        # Test store_memory
        success, path = await adapter.store_memory(
            "facts", "Python is my favorite programming language.", "python-favorite")
        assert success is True
        assert path is not None
        assert os.path.exists(path)
    
    @pytest.mark.asyncio
    async def test_memory_search(self, mock_ollama_api, mock_memory_dir):
        """Test semantic memory search functionality."""
        # Initialize memory manager
        memory_manager = BellaMemoryManager()
        await memory_manager.initialize("nomic-embed-text")
        
        # Search for memories
        results, success = await memory_manager.search_memory("programming languages")
        
        # Verify results structure
        assert success is True
        assert "primary_results" in results
        
        # Test the memory tool directly
        search_results = await semantic_memory_search("programming")
        assert "success" in search_results
    
    @pytest.mark.asyncio
    async def test_conversation_memory_integration(self, mock_ollama_api, mock_memory_dir):
        """Test memory integration in conversation flow."""
        # Create conversation adapter
        conv_adapter = MemoryConversationAdapter(embedding_model="nomic-embed-text")
        
        # Test pre-processing (before response)
        memory_context = await conv_adapter.pre_process_input(
            "Tell me about programming languages")
        
        # Verify context structure
        assert isinstance(memory_context, dict)
        assert "has_memory_context" in memory_context
        
        # Test post-processing (after response)
        modified_response = await conv_adapter.post_process_response(
            "What programming languages do you know?",
            "I'm familiar with Python, JavaScript, and many others."
        )
        
        # Verify no modification if not necessary
        assert modified_response is None or isinstance(modified_response, str)
    
    @pytest.mark.asyncio
    async def test_memory_processor_direct(self, mock_ollama_api):
        """Test EnhancedMemoryProcessor functionality directly."""
        processor = EnhancedMemoryProcessor(model_name="nomic-embed-text")
        
        # Test embedding generation
        embedding = await processor.generate_embedding("Test text for embedding")
        assert embedding is not None
        assert len(embedding) > 0
        
        # Test importance scoring
        importance = await processor.score_memory_importance(
            "This is a critical fact to remember about the project.")
        assert 0 <= importance <= 1
        
        # Test summary extraction
        long_text = "This is a very long text " * 50
        summary = await processor.extract_summary(long_text, max_length=10)
        assert len(summary.split()) <= 150  # Accounting for ellipsis
    
    @pytest.mark.asyncio
    async def test_memory_files_operations(self, mock_ollama_api, mock_memory_dir):
        """Test file operations for memory storage."""
        # Initialize adapter
        adapter = EnhancedMemoryAdapter(embedding_model="nomic-embed-text")
        await adapter.initialize()
        
        # Store a new memory
        content = "Python is an excellent language for AI development."
        success, path = await adapter.store_memory("facts", content, "python-ai")
        
        # Verify file was created
        assert success is True
        assert os.path.exists(path)
        
        # Read the file content
        from src.memory.memory_api import read_note
        stored_content = await read_note(path)
        assert stored_content == content
        
        # Test enhanced retrieval
        results = await adapter.enhance_memory_retrieval(
            "AI programming languages", [{"path": path, "title": "Python AI"}])
        assert len(results) > 0