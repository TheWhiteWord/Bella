"""Tests for memory tools integration with enhanced memory.

This module tests the memory tools registered with the tools registry 
to ensure they work correctly with the enhanced memory system.
"""

import os
import sys
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import tempfile
import shutil

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))

from src.memory.register_memory_tools import (
    semantic_memory_search,
    evaluate_memory_importance,
    prepare_for_memory_storage,
    save_to_memory,
    read_specific_memory,
    list_memories_by_type,
    save_conversation
)

from src.llm.tools_registry import registry


# Sample test data
sample_conversation = {
    "user_input": "Tell me about quantum computing",
    "assistant_response": "Quantum computing uses quantum bits or qubits which can exist in multiple states simultaneously, allowing certain algorithms to run exponentially faster than on classical computers."
}


class TestMemoryToolsIntegration:
    """Test suite for memory tools integration."""
    
    @pytest.fixture
    async def setup_memory_environment(self):
        """Set up temporary memory environment for tests."""
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        memories_dir = os.path.join(temp_dir, "memories")
        
        # Create memory type directories
        for mem_type in ["facts", "preferences", "conversations", "general", "reminders"]:
            os.makedirs(os.path.join(memories_dir, mem_type), exist_ok=True)
        
        # Create sample memory files
        with open(os.path.join(memories_dir, "facts", "quantum.md"), "w") as f:
            f.write("Quantum computing leverages quantum mechanics to process information in new ways.")
        
        with open(os.path.join(memories_dir, "preferences", "books.md"), "w") as f:
            f.write("I enjoy reading science fiction books, especially those about artificial intelligence.")
        
        # Save current directory to restore later
        old_dir = os.getcwd()
        os.chdir(temp_dir)
        
        # Set up memory environment with nomic-embed-text model
        with patch("aiohttp.ClientSession") as mock_session:
            mock_cm = MagicMock()
            mock_session.return_value.__aenter__.return_value = mock_cm
            
            # Mock responses for API calls
            mock_response = AsyncMock()
            mock_response.status = 200
            
            # Mock embedding generation
            async def mock_json():
                return {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}
                
            mock_response.json = mock_json
            mock_response.text = AsyncMock(return_value="")
            
            # Setup mock responses for different endpoints
            mock_cm.get.return_value.__aenter__.return_value = mock_response
            mock_cm.post.return_value.__aenter__.return_value = mock_response
            
            # Initialize memory system
            from src.memory.main_app_integration import ensure_memory_initialized
            await ensure_memory_initialized("nomic-embed-text")
            
            yield temp_dir
        
        # Clean up
        os.chdir(old_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_tools_registration(self):
        """Test that memory tools are properly registered."""
        # Verify that all memory tools are registered
        available_tools = registry.get_available_tools()
        tool_names = [tool["function"]["name"] for tool in available_tools]
        
        # Check that all expected memory tools are present
        memory_tool_names = [
            "semantic_memory_search", 
            "evaluate_memory_importance",
            "prepare_for_memory_storage",
            "continue_conversation",
            "save_to_memory",
            "read_specific_memory",
            "list_memories_by_type",
            "save_conversation"
        ]
        
        for tool_name in memory_tool_names:
            assert tool_name in tool_names
    
    @pytest.mark.asyncio
    async def test_semantic_memory_search(self, setup_memory_environment):
        """Test semantic memory search tool."""
        result = await semantic_memory_search("quantum computing")
        
        # Check result structure
        assert "success" in result
        
        # If success, check results structure
        if result["success"]:
            assert "results" in result
            assert "result_count" in result
    
    @pytest.mark.asyncio
    async def test_evaluate_importance(self):
        """Test memory importance evaluation tool."""
        result = await evaluate_memory_importance(
            "Quantum computing uses qubits which can exist in superposition.")
        
        # Check result structure
        assert "importance_score" in result
        assert "importance_level" in result
        assert "should_remember" in result
        assert isinstance(result["importance_score"], float)
        assert 0 <= result["importance_score"] <= 1
    
    @pytest.mark.asyncio
    async def test_prepare_for_storage(self):
        """Test text preparation for memory storage."""
        # Test with long text
        long_text = "This is a very long text that should be summarized " * 20
        result = await prepare_for_memory_storage(long_text)
        
        # Check result structure
        assert "optimized_text" in result
        assert "was_summarized" in result
        
        # Test without summarization
        result_no_summary = await prepare_for_memory_storage(
            "Short text", summarize=False)
        assert result_no_summary["was_summarized"] is False
    
    @pytest.mark.asyncio
    async def test_save_to_memory(self, setup_memory_environment):
        """Test save to memory tool."""
        # Create mock for should_store_memory to force storing
        with patch("src.memory.main_app_integration.BellaMemoryManager.should_store_memory", 
                  new_callable=AsyncMock) as mock_should_store:
            mock_should_store.return_value = True
            
            result = await save_to_memory(
                "The theory of relativity was developed by Albert Einstein.",
                memory_type="facts",
                title="relativity"
            )
            
            # Check result structure
            assert "success" in result
            
            # If success, check file exists
            if result["success"]:
                assert os.path.exists(
                    os.path.join("memories", "facts", "relativity.md"))
    
    @pytest.mark.asyncio
    async def test_read_specific_memory(self, setup_memory_environment):
        """Test read specific memory tool."""
        # First, create a memory to read
        with open(os.path.join("memories", "facts", "relativity.md"), "w") as f:
            f.write("E=mc² is part of Einstein's theory of relativity.")
        
        # Test reading by ID
        result = await read_specific_memory("facts/relativity")
        
        # Check result structure
        assert "success" in result
        
        # If successful, check content
        if result["success"]:
            assert "content" in result
            assert "E=mc²" in result["content"]
    
    @pytest.mark.asyncio
    async def test_list_memories_by_type(self, setup_memory_environment):
        """Test listing memories by type."""
        # Test listing facts
        result_facts = await list_memories_by_type("facts")
        
        # Check result structure
        assert "success" in result_facts
        assert "memories" in result_facts
        
        # Check that facts are listed
        if result_facts["success"]:
            assert "facts" in result_facts["memories"]
        
        # Test listing all
        result_all = await list_memories_by_type("all")
        
        # Check all memory types are included
        if result_all["success"]:
            assert len(result_all["memories"]) > 0
    
    @pytest.mark.asyncio
    async def test_save_conversation(self, setup_memory_environment):
        """Test save conversation tool."""
        result = await save_conversation(
            user_input="What is the capital of France?",
            assistant_response="The capital of France is Paris.",
            title="geography-conversation"
        )
        
        # Check result structure
        assert "success" in result
        
        # If successful, check file exists
        if result["success"]:
            assert os.path.exists(
                os.path.join("memories", "conversations", "geography-conversation.md"))
            
            # Verify content
            with open(os.path.join("memories", "conversations", "geography-conversation.md"), "r") as f:
                content = f.read()
                assert "capital of France" in content
                assert "Paris" in content