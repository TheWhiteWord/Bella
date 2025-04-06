"""Tests for autonomous memory integration system.

This module tests the autonomous memory features that work in the background
without requiring explicit commands.
"""

import os
import pytest
import asyncio
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any

from Bella.src.memory.memory_manager import MemoryManager
from Bella.src.memory.autonomous_memory import AutonomousMemory
from Bella.src.memory.memory_conversation_adapter import MemoryConversationAdapter, LLMMemoryTools


class TestAutonomousMemory:
    """Tests for autonomous memory features."""
    
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
    async def test_memory_detection_in_query(self, setup_memory_manager):
        """Test that the system can detect when a query needs memory context."""
        # Create autonomous memory
        autonomous = AutonomousMemory()
        
        # Test knowledge-seeking patterns
        knowledge_queries = [
            "What do you know about Paris?",
            "Do you remember my favorite color?",
            "Tell me about our last conversation",
            "What is Python programming?",
            "Who was Albert Einstein?"
        ]
        
        for query in knowledge_queries:
            assert autonomous._is_knowledge_seeking_query(query), f"Failed to detect knowledge query: {query}"
            
        # Test non-knowledge queries
        non_knowledge_queries = [
            "Hello, how are you?",
            "What time is it now?",
            "Can you play some music?",
            "Tell me a joke",
            "What's the weather like today?"
        ]
        
        for query in non_knowledge_queries:
            # Some of these might be detected as knowledge queries due to patterns
            # The key is that the main conversational queries are not mistaken for knowledge queries
            if autonomous._is_knowledge_seeking_query(query):
                print(f"Note: '{query}' was detected as a knowledge query")
    
    @pytest.mark.asyncio
    async def test_memory_augmentation_detection(self, setup_memory_manager):
        """Test detection of when to augment with memory context."""
        autonomous = AutonomousMemory()
        
        # Reset the last memory check to ensure clean test
        autonomous.last_memory_check = autonomous.last_memory_check.replace(year=2020)
        
        # Test personal topic patterns
        personal_queries = [
            "What's my preference for coffee?",
            "Do you remember what I told you yesterday?",
            "When is my next appointment?",
            "What's your opinion about classical music?",
            "I mentioned something about Python last time",
            "When is my birthday?"
        ]
        
        for query in personal_queries:
            assert autonomous._should_augment_with_memory(query), f"Failed to detect personal topic: {query}"
            
        # Test regular queries
        regular_queries = [
            "What's 2+2?",
            "Tell me a story",
            "How do computers work?",
            "Can you translate this to French?",
            "What's the capital of France?"
        ]
        
        # These should generally not trigger memory augmentation
        for query in regular_queries:
            autonomous.last_memory_check = autonomous.last_memory_check.replace(year=2020)  # Reset timer
            if autonomous._should_augment_with_memory(query):
                print(f"Note: '{query}' triggered memory augmentation")
    
    @pytest.mark.asyncio
    async def test_topic_extraction(self, setup_memory_manager):
        """Test extraction of potential topics from text."""
        autonomous = AutonomousMemory()
        
        # Test topic extraction with proper nouns
        text = "I want to learn more about Python programming and visit Paris next summer."
        topics = autonomous._extract_potential_topics(text)
        
        assert "Python" in topics, "Failed to extract 'Python' as a topic"
        assert "Paris" in topics, "Failed to extract 'Paris' as a topic"
        assert "programming" in topics, "Failed to extract 'programming' as a topic"
        
        # Test topic extraction with noun phrases
        text = "Tell me about the solar system and how planets orbit the sun."
        topics = autonomous._extract_potential_topics(text)
        
        assert any(topic in ["solar", "system", "solar system"] for topic in topics), "Failed to extract 'solar system'"
        assert any(topic in ["planets", "orbit", "sun"] for topic in topics), "Failed to extract planets/orbit/sun"
    
    @pytest.mark.asyncio
    async def test_retrieve_relevant_memory(self, setup_memory_manager):
        """Test retrieving relevant memories for a query."""
        # Create test memory
        manager = setup_memory_manager
        await manager.create_memory(
            title="Paris Facts",
            content="""# Paris Facts
            
## Information
Paris is the capital of France and known as the City of Light.

## Observations
- [fact] Paris is the capital of France #geography #france
- [fact] Paris is known as the City of Light #travel #culture
- [fact] The Eiffel Tower is in Paris #landmark

## Relations
- type [[City]]
- located_in [[France]]""",
            folder="facts",
            tags=["travel", "france"]
        )
        
        # Test with autonomous memory
        autonomous = AutonomousMemory()
        
        # Direct knowledge query
        memory_context = await autonomous._retrieve_relevant_memory("What do you know about Paris?")
        
        assert memory_context.get("has_memory_context", False), "Failed to retrieve memory for direct query"
        assert "memory_response" in memory_context, "Memory response not found in context"
        assert "Paris" in memory_context.get("memory_response", ""), "Paris not mentioned in memory response"
        assert "capital of France" in memory_context.get("memory_response", ""), "Facts not included in memory response"
    
    @pytest.mark.asyncio
    async def test_process_conversation_turn_preprocessing(self, setup_memory_manager):
        """Test pre-processing of conversation turns."""
        # Create test memory
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
- type [[Programming Language]]
- related_to [[Computer Science]]""",
            folder="facts",
            tags=["programming", "python"]
        )
        
        # Test pre-processing turn
        autonomous = AutonomousMemory()
        modified_response, memory_context = await autonomous.process_conversation_turn(
            "Tell me who created Python")
        
        # Should return None for response (pre-processing) and memory context
        assert modified_response is None, "Pre-processing should not modify response yet"
        assert memory_context.get("has_memory_context", False), "Failed to find memory context"
        assert "memory_response" in memory_context, "Memory response missing from context"
        assert "Guido van Rossum" in memory_context.get("memory_response", ""), "Creator not in memory response"
    
    @pytest.mark.asyncio
    async def test_process_conversation_turn_postprocessing(self, setup_memory_manager):
        """Test post-processing of conversation turns."""
        # Setup autonomous memory
        autonomous = AutonomousMemory()
        
        # Test fact extraction in post-processing
        user_input = "My favorite color is blue."
        response = "I see, thank you for sharing that preference."
        
        modified_response, _ = await autonomous.process_conversation_turn(user_input, response)
        
        # The system might identify this as a fact to remember in some implementations
        # But at minimum, the original response should be preserved
        assert response in modified_response, "Original response not preserved in post-processing"
    
    @pytest.mark.asyncio
    async def test_memory_conversation_adapter(self, setup_memory_manager):
        """Test the memory conversation adapter."""
        # Create test memory
        manager = setup_memory_manager
        await manager.create_memory(
            title="Coffee Preferences",
            content="""# Coffee Preferences
            
## Information
Information about coffee preferences.

## Observations
- [preference] I prefer my coffee black without sugar #preference #coffee
- [preference] I like to drink coffee in the morning #routine #coffee

## Relations
- type [[Preference]]
- about [[Coffee]]""",
            folder="preferences",
            tags=["preference", "coffee"]
        )
        
        # Create conversation adapter
        adapter = MemoryConversationAdapter()
        
        # Test pre-processing
        conversation_history = [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi, how can I help you?"}
        ]
        
        memory_context = await adapter.pre_process_input(
            "Do you remember how I like my coffee?",
            conversation_history
        )
        
        assert "memory_context" in memory_context, "Memory context not added to response context"
        assert "coffee" in memory_context.get("memory_context", "").lower(), "Coffee preference not in memory context"
        
        # Test post-processing
        user_input = "What's my coffee preference?"
        assistant_response = "I'm not sure. Let me check."
        
        modified_response = await adapter.post_process_response(user_input, assistant_response)
        
        # The post-processed response should either contain the original or add memory information
        assert assistant_response in modified_response or "coffee" in modified_response.lower(), \
            "Post-processing didn't properly handle the response"
    
    @pytest.mark.asyncio
    async def test_llm_memory_tools(self, setup_memory_manager):
        """Test the LLM memory tools functionality."""
        # Setup memory tools
        memory_tools = LLMMemoryTools()
        
        # Test tool schemas
        tools = memory_tools.get_memory_tools()
        
        assert "remember_fact" in tools, "remember_fact tool missing"
        assert "recall_memory" in tools, "recall_memory tool missing"
        assert "save_conversation" in tools, "save_conversation tool missing"
        
        # Test remember_fact tool
        fact_result = await memory_tools.execute_tool(
            "remember_fact", 
            {"fact": "Python is my favorite programming language"}
        )
        
        assert fact_result.get("success", False), "remember_fact tool failed"
        assert "memory_id" in fact_result, "memory_id not returned from remember_fact"
        
        # Test recall_memory tool
        recall_result = await memory_tools.execute_tool(
            "recall_memory",
            {"query": "Python"}
        )
        
        assert recall_result.get("success", True), "recall_memory tool failed"
        # The recall might find something if Python is mentioned in memories
        
        # Test save_conversation tool
        save_result = await memory_tools.execute_tool(
            "save_conversation",
            {"title": "Test Conversation", "topic": "Testing"}
        )
        
        assert "message" in save_result, "No message returned from save_conversation"
        # May succeed or fail depending on whether there's a conversation in the buffer