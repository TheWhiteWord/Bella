"""Tests for the memory manager module.

This module contains tests for the MemoryManager class which handles
storage and retrieval of memories in markdown files.
"""

import os
import shutil
import pytest
import asyncio
import tempfile
from datetime import datetime
from pathlib import Path

from Bella.src.memory.memory_manager import MemoryManager


class TestMemoryManager:
    """Tests for the MemoryManager class."""

    @pytest.fixture
    def temp_memory_dir(self):
        """Create a temporary directory for memory files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Clean up after tests
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def memory_manager(self, temp_memory_dir):
        """Create a MemoryManager instance with a temporary directory."""
        return MemoryManager(temp_memory_dir)

    @pytest.mark.asyncio
    async def test_create_memory(self, memory_manager):
        """Test creating a new memory."""
        title = "Test Memory"
        content = "# Test Memory\n\n## Content\n\nThis is a test memory."
        
        result = await memory_manager.create_memory(
            title=title,
            content=content,
            folder="general",
            tags=["test"],
            verbose=True
        )
        
        assert result["title"] == title
        assert "path" in result
        assert os.path.exists(result["path"])
        
        # Check that the file contains our content
        with open(result["path"], "r") as f:
            file_content = f.read()
            assert content in file_content

    @pytest.mark.asyncio
    async def test_read_memory(self, memory_manager):
        """Test reading a memory by title."""
        # First create a memory
        title = "Reading Test"
        content = "# Reading Test\n\n## Content\n\nThis is for testing reading."
        
        create_result = await memory_manager.create_memory(
            title=title,
            content=content,
            folder="general"
        )
        
        # Now read it back
        read_result = await memory_manager.read_memory(title)
        
        assert read_result is not None
        assert read_result["title"] == title
        assert content in read_result["content"]

    @pytest.mark.asyncio
    async def test_search_memories(self, memory_manager):
        """Test searching memories."""
        # Create a few memories with different content
        await memory_manager.create_memory(
            title="Apple Facts",
            content="# Apple Facts\n\nApples are red or green fruits.",
            folder="facts", 
            tags=["fruit"]
        )
        
        await memory_manager.create_memory(
            title="Banana Info",
            content="# Banana Info\n\nBananas are yellow and curved.",
            folder="facts",
            tags=["fruit"]
        )
        
        # Search for apple
        apple_results = await memory_manager.search_memories("apple")
        assert len(apple_results["primary_results"]) == 1
        assert apple_results["primary_results"][0]["title"] == "Apple Facts"
        
        # Search for fruit (should find both)
        fruit_results = await memory_manager.search_memories("fruit")
        assert len(fruit_results["primary_results"]) == 2

    @pytest.mark.asyncio
    async def test_build_context(self, memory_manager):
        """Test building context from a memory."""
        # Create a main memory with relations
        main_content = """# Main Topic
        
## Content

This is the main topic.

## Relations

- relates_to [[Related Topic One]]
- about [[Related Topic Two]]
"""
        
        # Create the main and related memories
        await memory_manager.create_memory(
            title="Main Topic",
            content=main_content,
            folder="general"
        )
        
        await memory_manager.create_memory(
            title="Related Topic One",
            content="# Related Topic One\n\nThis is related topic one.",
            folder="general"
        )
        
        await memory_manager.create_memory(
            title="Related Topic Two",
            content="# Related Topic Two\n\nThis is related topic two.",
            folder="general"
        )
        
        # Build context from main topic
        context = await memory_manager.build_context("Main Topic", depth=1)
        
        assert "primary" in context
        assert context["primary"]["title"] == "Main Topic"
        assert len(context["related"]) == 2
        related_titles = [item["title"] for item in context["related"]]
        assert "Related Topic One" in related_titles
        assert "Related Topic Two" in related_titles

    @pytest.mark.asyncio
    async def test_update_memory(self, memory_manager):
        """Test updating an existing memory."""
        # Create initial memory
        title = "Update Test"
        content = "# Update Test\n\nOriginal content."
        
        await memory_manager.create_memory(
            title=title,
            content=content,
            folder="general"
        )
        
        # Now update it
        new_content = "# Update Test\n\nUpdated content."
        update_result = await memory_manager.update_memory(
            title_or_path=title,
            new_content=new_content
        )
        
        # Read it back to verify update
        read_result = await memory_manager.read_memory(title)
        assert new_content in read_result["content"]
        assert "Original content" not in read_result["content"]

    @pytest.mark.asyncio
    async def test_delete_memory(self, memory_manager):
        """Test deleting a memory."""
        # Create a memory
        title = "Delete Test"
        content = "# Delete Test\n\nThis will be deleted."
        
        create_result = await memory_manager.create_memory(
            title=title,
            content=content,
            folder="general"
        )
        
        file_path = create_result["path"]
        assert os.path.exists(file_path)
        
        # Delete it
        delete_result = await memory_manager.delete_memory(title)
        assert delete_result["success"] is True
        
        # Verify it's gone
        assert not os.path.exists(file_path)
        read_result = await memory_manager.read_memory(title)
        assert read_result is None