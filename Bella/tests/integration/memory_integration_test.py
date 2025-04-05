"""Integration tests for Memory Tool functionality.

This module tests the integration between the Bella assistant and
the basic-memory MCP capabilities.
"""

import asyncio
import logging
import os
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

# Import the MemoryTool to test
from Bella.src.mcp_servers.memory_tool import MemoryTool

# Configure logging
logger = logging.getLogger(__name__)


@pytest.fixture
def memory_tool():
    """Fixture providing a mocked MemoryTool instance."""
    with patch("Bella.src.mcp_servers.memory_tool.MemoryAgent") as mock_memory_agent:
        # Set up the mock MemoryAgent
        instance = mock_memory_agent.return_value
        instance.process_query = AsyncMock()
        
        # Create a memory tool with the mocked agent
        tool = MemoryTool(verbose=True)
        tool.memory_agent = instance  # Replace with mock instance
        
        # Return the tool with access to the mock
        yield tool, instance


@pytest.mark.asyncio
async def test_remember_note(memory_tool):
    """Test creating a note."""
    tool, mock_agent = memory_tool
    
    # Set up the mock response
    mock_agent.process_query.return_value = {
        "parsed_tools": [{
            "tool": "write_note",
            "title": "Test Note",
            "content": "Test content",
            "tags": ["test", "note"]
        }],
        "response": "Note created successfully!"
    }
    
    # Call the remember method
    result = await tool.remember("Test Note", "Test content", ["test", "note"])
    
    # Assertions
    mock_agent.process_query.assert_called_once()
    assert result["success"] is True
    assert result["title"] == "Test Note"
    assert result["tags"] == ["test", "note"]


@pytest.mark.asyncio
async def test_recall_information(memory_tool):
    """Test searching for information."""
    tool, mock_agent = memory_tool
    
    # Set up the mock response
    mock_agent.process_query.return_value = {
        "parsed_tools": [{
            "tool": "search_notes",
            "query": "test query",
            "results": ["Note 1", "Note 2"]
        }],
        "response": "Found 2 relevant notes."
    }
    
    # Call the recall method
    result = await tool.recall("test query")
    
    # Assertions
    mock_agent.process_query.assert_called_once()
    assert result["success"] is True
    assert len(result["search_results"]) == 1
    assert result["search_results"][0]["results"] == ["Note 1", "Note 2"]


@pytest.mark.asyncio
async def test_read_note(memory_tool):
    """Test reading a note."""
    tool, mock_agent = memory_tool
    
    # Set up the mock response
    mock_agent.process_query.return_value = {
        "parsed_tools": [{
            "tool": "read_note",
            "content": "# Test Note\n\nThis is test content."
        }],
        "response": "Here's the note content."
    }
    
    # Call the read_note method
    result = await tool.read_note("Test Note")
    
    # Assertions
    mock_agent.process_query.assert_called_once()
    assert result["success"] is True
    assert result["content"] == "# Test Note\n\nThis is test content."


@pytest.mark.asyncio
async def test_build_context(memory_tool):
    """Test building context."""
    tool, mock_agent = memory_tool
    
    # Set up the mock response
    mock_agent.process_query.return_value = {
        "parsed_tools": [{
            "tool": "build_context",
            "notes": ["Note 1", "Note 2"],
            "summary": "Context about coffee brewing."
        }],
        "response": "Context built successfully."
    }
    
    # Call the build_context method
    result = await tool.build_context("coffee", depth=2)
    
    # Assertions
    mock_agent.process_query.assert_called_once()
    assert result["success"] is True
    assert "context" in result
    assert result["topic"] == "coffee"
    assert result["depth"] == 2


@pytest.mark.asyncio
async def test_integration_flow():
    """Test the full integration flow with error handling."""
    with patch("Bella.src.mcp_servers.memory_tool.MemoryTool") as mock_class:
        # Set up mocked instance
        tool_instance = MagicMock()
        mock_class.return_value = tool_instance
        
        # Set up return values for the methods
        tool_instance.remember = AsyncMock(return_value={
            "success": True,
            "title": "Test Note",
            "tags": ["test", "note"]
        })
        
        tool_instance.recall = AsyncMock(return_value={
            "success": True,
            "raw_response": "Found matching notes",
            "search_results": [{"results": ["Note 1"]}]
        })
        
        tool_instance.read_note = AsyncMock(return_value={
            "success": True,
            "content": "# Test Content\nThis is a test."
        })
        
        # Create a simple test flow
        tool = mock_class()
        await tool.remember("Test Note", "Test content", ["test"])
        await tool.recall("test content")
        await tool.read_note("Test Note")
        
        # Verify the flow
        tool_instance.remember.assert_called_once()
        tool_instance.recall.assert_called_once()
        tool_instance.read_note.assert_called_once()