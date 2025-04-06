"""Tests for the voice memory module.

This module tests the VoiceMemoryModule which provides the main interface
for integrating memory capabilities with the voice assistant.
"""

import os
import pytest
import asyncio
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from Bella.src.memory.voice_memory_module import VoiceMemoryModule
from Bella.src.memory.memory_manager import MemoryManager


class TestVoiceMemoryModule:
    """Tests for the VoiceMemoryModule class."""

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
    def memory_module(self):
        """Create a VoiceMemoryModule instance."""
        return VoiceMemoryModule()

    @pytest.mark.asyncio
    async def test_process_input_basic(self, memory_module):
        """Test processing basic user input with no memory commands."""
        user_input = "What's the weather like today?"
        assistant_response = "I don't have access to current weather information."
        
        result = await memory_module.process_input(user_input, assistant_response)
        
        # Should not trigger any memory operations
        assert result is None
        
        # Should have added to conversation buffer
        assert len(memory_module.integration.conversation_buffer) == 2
        assert memory_module.integration.conversation_buffer[0] == user_input
        assert memory_module.integration.conversation_buffer[1] == assistant_response

    @pytest.mark.asyncio
    async def test_process_input_memory_command(self, memory_module, setup_memory_manager):
        """Test processing a memory command."""
        # The detect_and_handle_memory_command function should be called with this input
        with patch('Bella.src.memory.voice_memory_module.detect_and_handle_memory_command') as mock_detect:
            mock_detect.return_value = (True, "I've remembered that fact.")
            
            user_input = "Remember that elephants are the largest land animals."
            assistant_response = "I'll remember that."
            
            result = await memory_module.process_input(user_input, assistant_response)
            
            # Should return response from memory command handler
            assert result == "I've remembered that fact."
            mock_detect.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_input_fact_extraction(self, memory_module):
        """Test automatic fact extraction from user input."""
        # Mock the extract_and_save_fact to return success
        with patch.object(memory_module.integration, 'extract_and_save_fact') as mock_extract:
            mock_extract.return_value = {"title": "Elephants", "error": None}
            
            user_input = "Elephants are the largest land animals."
            assistant_response = "That's correct! They're magnificent creatures."
            
            result = await memory_module.process_input(user_input, assistant_response)
            
            # Should notify about fact extraction
            assert "I've noted that fact" in result
            mock_extract.assert_called_once_with(user_input)

    @pytest.mark.asyncio
    async def test_process_input_pending_request(self, memory_module):
        """Test handling pending memory requests."""
        # Set up a pending memory request
        memory_module.pending_memory_request = "save_conversation"
        
        # Mock save_current_conversation to return success
        with patch.object(memory_module.integration, 'save_current_conversation') as mock_save:
            mock_save.return_value = {"title": "Saved Conversation", "error": None}
            
            user_input = "Yes, please save it."
            assistant_response = "I'll do that right away."
            
            result = await memory_module.process_input(user_input, assistant_response)
            
            # Should confirm saving the conversation
            assert "I've saved our conversation" in result
            mock_save.assert_called_once()
            
            # Pending request should be cleared
            assert memory_module.pending_memory_request is None
            
            # Test rejection of pending request
            memory_module.pending_memory_request = "save_conversation"
            
            user_input = "No, don't save it."
            assistant_response = "Alright, I won't save it."
            
            result = await memory_module.process_input(user_input, assistant_response)
            
            # Shouldn't save anything but should clear the request
            assert result is None
            assert memory_module.pending_memory_request is None

    @pytest.mark.asyncio
    async def test_should_offer_memory_save(self, memory_module):
        """Test detection of important conversations to save."""
        # Create conversation with important indicators
        for _ in range(4):  # Add several turns to make it substantial
            memory_module.integration.add_to_conversation_buffer(
                "I have an important appointment on Friday at 3pm.", 
                "I'll remember that you have an appointment."
            )
        
        # Should detect important conversation
        assert memory_module._should_offer_memory_save() is True
        
        # Test with non-important conversation
        memory_module.integration.clear_conversation_buffer()
        memory_module.integration.add_to_conversation_buffer(
            "What's the weather like?", 
            "I don't have access to weather information."
        )
        
        # Should not trigger save offer for basic conversation
        assert memory_module._should_offer_memory_save() is False

    @pytest.mark.asyncio
    async def test_query_memory(self, memory_module, setup_memory_manager):
        """Test querying memory for information."""
        # Mock answer_from_memory to return a result
        with patch.object(memory_module.integration, 'answer_from_memory') as mock_answer:
            mock_answer.return_value = ("From my memory, Brazil is the largest country in South America.", True)
            
            result, found = await memory_module.query_memory("What's the largest country in South America?")
            
            assert found is True
            assert "Brazil" in result
            mock_answer.assert_called_once()

    def test_extract_conversation_topics(self, memory_module):
        """Test extraction of conversation topics."""
        # Populate conversation buffer with themed content
        memory_module.integration.add_to_conversation_buffer(
            "I love hiking in national parks.", 
            "That sounds wonderful! Which parks have you visited?"
        )
        memory_module.integration.add_to_conversation_buffer(
            "I've been to Yellowstone and Yosemite.", 
            "Those are amazing parks with beautiful landscapes and wildlife."
        )
        
        topics = memory_module._extract_conversation_topics()
        
        # Should identify parks-related words
        assert len(topics) > 0
        assert any("park" in topic.lower() for topic in topics) or \
               any("yellowstone" in topic.lower() for topic in topics) or \
               any("yosemite" in topic.lower() for topic in topics)

    def test_get_conversation_summary(self, memory_module):
        """Test generating conversation summary."""
        # Empty conversation - should return None
        assert memory_module.get_conversation_summary() is None
        
        # Add conversation content
        memory_module.integration.add_to_conversation_buffer(
            "Tell me about space exploration.", 
            "Space exploration involves traveling beyond Earth's atmosphere."
        )
        memory_module.integration.add_to_conversation_buffer(
            "When did humans land on the Moon?", 
            "The Apollo 11 mission landed humans on the Moon in July 1969."
        )
        
        # Mock topic extraction
        with patch.object(memory_module, '_extract_conversation_topics') as mock_extract:
            mock_extract.return_value = ["space", "exploration", "moon"]
            
            summary = memory_module.get_conversation_summary()
            
            # Should contain turn count and topics
            assert summary is not None
            assert "Conversation with 2 turns" in summary
            assert "space" in summary and "exploration" in summary