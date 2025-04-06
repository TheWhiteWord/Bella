"""Tests for the voice memory integration module.

This module contains tests for the VoiceMemoryIntegration class which
integrates memory capabilities with voice interactions.
"""

import os
import pytest
import asyncio
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from Bella.src.memory.voice_memory_integration import VoiceMemoryIntegration
from Bella.src.memory.memory_manager import MemoryManager


class TestVoiceMemoryIntegration:
    """Tests for the VoiceMemoryIntegration class."""

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
                
    @pytest.fixture
    def voice_memory_integration(self):
        """Create a VoiceMemoryIntegration instance."""
        return VoiceMemoryIntegration()

    def test_add_to_conversation_buffer(self, voice_memory_integration):
        """Test adding to the conversation buffer."""
        # Initial buffer should be empty
        assert len(voice_memory_integration.conversation_buffer) == 0
        
        # Add a conversation turn
        user_text = "Hello, how are you?"
        assistant_text = "I'm doing well, thank you!"
        
        voice_memory_integration.add_to_conversation_buffer(user_text, assistant_text)
        
        # Buffer should now contain both messages
        assert len(voice_memory_integration.conversation_buffer) == 2
        assert voice_memory_integration.conversation_buffer[0] == user_text
        assert voice_memory_integration.conversation_buffer[1] == assistant_text
        
        # Test buffer truncation with many messages
        for i in range(15):
            voice_memory_integration.add_to_conversation_buffer(f"User message {i}", f"Assistant response {i}")
            
        # Buffer should maintain maximum size (20 messages = 10 turns)
        assert len(voice_memory_integration.conversation_buffer) == 20
        
        # The oldest messages should have been removed
        assert "Hello" not in voice_memory_integration.conversation_buffer[0]
        assert "User message" in voice_memory_integration.conversation_buffer[0]

    def test_clear_conversation_buffer(self, voice_memory_integration):
        """Test clearing the conversation buffer."""
        # Add some conversation
        voice_memory_integration.add_to_conversation_buffer("Test message", "Test response")
        assert len(voice_memory_integration.conversation_buffer) > 0
        
        # Set a current topic
        voice_memory_integration.current_topic = "Test Topic"
        
        # Clear buffer
        voice_memory_integration.clear_conversation_buffer()
        
        # Buffer should be empty
        assert len(voice_memory_integration.conversation_buffer) == 0
        assert voice_memory_integration.current_topic is None

    @pytest.mark.asyncio
    async def test_save_current_conversation(self, voice_memory_integration, setup_memory_manager):
        """Test saving the current conversation."""
        # Add conversation content
        voice_memory_integration.add_to_conversation_buffer("Tell me about apples", "Apples are nutritious fruits.")
        voice_memory_integration.add_to_conversation_buffer("Are they good for you?", "Yes, they contain fiber and vitamins.")
        
        # Set a topic
        topic = "Fruits"
        await voice_memory_integration.set_conversation_topic(topic)
        
        # Save conversation
        result = await voice_memory_integration.save_current_conversation(title="Apple Discussion")
        
        assert "error" not in result
        assert result["title"] == "Apple Discussion"
        assert "path" in result
        assert os.path.exists(result["path"])
        
        # Check that the file contains conversation content
        with open(result["path"], "r") as f:
            content = f.read()
            assert "Tell me about apples" in content
            assert "Apples are nutritious fruits" in content
            assert "Fruits" in content  # Topic should be included
            
    @pytest.mark.asyncio
    async def test_extract_and_save_fact(self, voice_memory_integration, setup_memory_manager):
        """Test extracting and saving a fact from text."""
        # Test with various fact indicators
        fact_texts = [
            "Apples are red fruits with lots of fiber",  # "are" indicator
            "I like chocolate ice cream the most",       # "I like" indicator
            "My favorite movie is The Matrix"            # "favorite" indicator
        ]
        
        for text in fact_texts:
            result = await voice_memory_integration.extract_and_save_fact(text)
            
            # Should have created a memory
            assert result is not None
            assert "error" not in result
            assert "path" in result
            
            # Check the content was saved
            with open(result["path"], "r") as f:
                content = f.read()
                assert text in content
                
        # Test with non-fact text
        non_fact = "What time is it now?"
        result = await voice_memory_integration.extract_and_save_fact(non_fact)
        assert result is None  # Should not identify as a fact

    @pytest.mark.asyncio
    async def test_answer_from_memory(self, voice_memory_integration, setup_memory_manager):
        """Test answering a question from memory."""
        # Create a test memory first
        manager = setup_memory_manager
        await manager.create_memory(
            title="Python Facts",
            content="""# Python Facts

## Information
Python is a high-level programming language known for its readability.

## Observations
- [fact] Python was created by Guido van Rossum #programming
- [fact] Python is often used for data science and AI #programming #datascience

## Relations
- type [[Fact]]
- about [[Programming]]""",
            folder="facts",
            tags=["programming", "python"]
        )
        
        # Test querying for something in memory
        response, found = await voice_memory_integration.answer_from_memory("Who created Python?")
        
        assert found is True
        assert "Guido van Rossum" in response
        assert "Python Facts" in response
        
        # Test querying for something not in memory
        response, found = await voice_memory_integration.answer_from_memory("What's the weather like today?")
        
        assert found is False
        assert response == ""

    def test_extract_topic(self, voice_memory_integration):
        """Test topic extraction from text."""
        # Test with obvious nouns
        text = "My favorite movie is The Matrix"
        topic = voice_memory_integration._extract_topic(text)
        assert topic == "Matrix" or topic == "The Matrix"
        
        # Test with multiple potential topics
        text = "I enjoy hiking in the National Parks whenever possible"
        topic = voice_memory_integration._extract_topic(text)
        assert "National Parks" in topic or "Parks" in topic or "hiking" in topic
        
        # Test with minimal content
        text = "I am happy"
        topic = voice_memory_integration._extract_topic(text)
        assert topic == "happy" or topic is None  # Might not extract "happy" as it could be an adjective