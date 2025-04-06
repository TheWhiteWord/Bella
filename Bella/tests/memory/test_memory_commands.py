"""Tests for memory commands module.

Tests the detection and handling of memory-related voice commands.
"""

import os
import pytest
import asyncio
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from datetime import datetime

from Bella.src.memory.memory_commands import (
    detect_and_handle_memory_command,
    handle_remember_command,
    handle_recall_command
)
from Bella.src.memory.memory_manager import MemoryManager


class TestMemoryCommands:
    """Tests for memory command functions."""
    
    @pytest.fixture
    def temp_memory_dir(self):
        """Create a temporary directory for memory files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Clean up after tests
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def setup_memory_manager(self, temp_memory_dir):
        """Set up memory manager with temp directory."""
        with patch('Bella.src.memory.memory_api._memory_manager', None):
            with patch('Bella.src.memory.memory_api.get_memory_manager') as mock_get_manager:
                manager = MemoryManager(temp_memory_dir)
                mock_get_manager.return_value = manager
                yield manager

    @pytest.mark.asyncio
    async def test_detect_memory_command(self):
        """Test detection of memory commands in text."""
        # Test obvious memory commands
        memory_commands = [
            "Remember that the capital of France is Paris",
            "Make a note of this meeting tomorrow",
            "Save this to your memory",
            "Remember this conversation",
            "Take a note about the project deadline"
        ]
        
        for command in memory_commands:
            is_command, _ = await detect_and_handle_memory_command(command)
            assert is_command is True, f"Failed to detect: {command}"
            
        # Test recall commands
        recall_commands = [
            "What do you remember about Paris?",
            "Do you remember my birthday?",
            "Recall our last conversation",
            "Look up project deadline in your memory",
            "Get memory about my favorite color"
        ]
        
        for command in recall_commands:
            is_command, _ = await detect_and_handle_memory_command(command)
            assert is_command is True, f"Failed to detect: {command}"
            
        # Test non-memory commands
        non_memory_commands = [
            "What time is it?",
            "Tell me a joke",
            "How's the weather today?",
            "Can you play some music?"
        ]
        
        for command in non_memory_commands:
            is_command, _ = await detect_and_handle_memory_command(command)
            assert is_command is False, f"Incorrectly detected: {command}"

    @pytest.mark.asyncio
    async def test_handle_remember_command(self, setup_memory_manager):
        """Test handling remember commands."""
        # Test remembering a fact
        with patch('Bella.src.memory.memory_commands.write_note') as mock_write:
            mock_write.return_value = {"title": "Test Fact", "path": "/test/path.md"}
            
            result = await handle_remember_command("Remember that the Earth orbits the Sun")
            
            # Should confirm saving
            assert "I've saved that to memory" in result
            mock_write.assert_called_once()
            
            # Check that arguments contain the fact
            args = mock_write.call_args[1]
            assert "Earth orbits the Sun" in args["content"]
            
        # Test making a note
        with patch('Bella.src.memory.memory_commands.write_note') as mock_write:
            mock_write.return_value = {"title": "Meeting Note", "path": "/test/path.md"}
            
            result = await handle_remember_command("Make a note about the team meeting at 3pm")
            
            assert "I've saved that to memory" in result
            assert mock_write.call_args[1]["folder"] == "general"  # Should be in general folder
            
        # Test saving a conversation
        conversation_history = [
            "Hi, how are you?",
            "I'm doing well, thank you!",
            "Tell me about memory systems.",
            "Memory systems help store and retrieve information..."
        ]
        
        with patch('Bella.src.memory.memory_commands.write_note') as mock_write:
            mock_write.return_value = {"title": "Conversation", "path": "/test/path.md"}
            
            result = await handle_remember_command("Save this conversation", conversation_history)
            
            assert "I've saved that to memory" in result
            
            # Check that all conversation turns are included
            content = mock_write.call_args[1]["content"]
            for message in conversation_history:
                assert message in content

    @pytest.mark.asyncio
    async def test_handle_recall_command(self, setup_memory_manager):
        """Test handling recall commands."""
        # Create a test memory to recall
        manager = setup_memory_manager
        await manager.create_memory(
            title="Paris Facts",
            content="""# Paris Facts
            
## Information
Paris is the capital of France.

## Observations
- [fact] Paris is known as the City of Light #travel #france
- [fact] The Eiffel Tower is in Paris #landmark

## Relations
- type [[City]]
- located_in [[France]]""",
            folder="facts",
            tags=["travel", "france"]
        )
        
        # Test recall with search results
        result = await handle_recall_command("Paris")
        
        # Should find relevant information
        assert "About Paris" in result
        assert "City of Light" in result or "Eiffel Tower" in result
        
        # Test recall with no results
        result = await handle_recall_command("Mars Colony")
        
        # Should indicate no memories found
        assert "don't have any memories about Mars Colony" in result
        
    @pytest.mark.asyncio
    async def test_integration_remember_recall(self, setup_memory_manager):
        """Test end-to-end remember and recall flow."""
        # First, create the memory directly to ensure it's properly stored
        manager = setup_memory_manager
        title = "Python Programming"
        content = """# Python Programming

## Information
Python is a high-level programming language.

## Observations
- [fact] Python was created by Guido van Rossum #programming
- [fact] Python is often used for data science and AI #programming

## Relations
- type [[Programming Language]]
"""
        await manager.create_memory(
            title=title,
            content=content,
            folder="facts",
            tags=["programming", "python"]
        )
        
        # Force memory manager to reload memories to see the new entry
        manager._load_memories()
        
        # Then, try to recall it directly with handle_recall_command
        recall_result = await handle_recall_command("Python")
        
        # This should now find the memory
        assert "About Python" in recall_result
        assert "Guido van Rossum" in recall_result
        
        # Test remember command separately
        remember_text = "Remember that Python is my favorite language"
        is_command, remember_result = await detect_and_handle_memory_command(remember_text)
        
        assert is_command is True
        assert "I've saved that to memory" in remember_result