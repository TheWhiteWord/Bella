"""Tests for the autonomous memory system.

Tests the functionality of the autonomous memory system with the new standardized format.
"""

import os
import sys
import pytest
import asyncio
import re
import tempfile
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.memory.autonomous_memory import AutonomousMemory
from src.memory.memory_utils import calculate_tfidf_similarity

# Create a memory manager mock for tests with proper path support
class MockMemoryManager:
    def __init__(self):
        # Set up a dedicated test directory for memory operations
        self.temp_dir = tempfile.mkdtemp(prefix="bella_test_memories_")
        self.memory_dir = self.temp_dir
        
        # Create memory directory structure
        os.makedirs(self.memory_dir, exist_ok=True)
        for folder in ['conversations', 'facts', 'preferences', 'reminders', 'general']:
            os.makedirs(os.path.join(self.memory_dir, folder), exist_ok=True)
        
        # Mock the enhanced adapter
        self.enhanced_adapter = AsyncMock()
        self.enhanced_adapter.processor = AsyncMock()
        self.enhanced_adapter.processor.generate_embedding = AsyncMock(return_value=[0.1] * 768)

    def cleanup(self):
        """Clean up the temporary directory after tests"""
        if os.path.exists(self.temp_dir):
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"Error cleaning up temp directory: {e}")


@pytest.fixture
def memory_system():
    """Creates an instance of the autonomous memory system with test configuration."""
    # Create a mock memory manager that will be used in the tests
    mock_manager = MockMemoryManager()
    
    # Patch the main_app_integration.memory_manager to use our mock
    with patch('src.memory.main_app_integration.memory_manager', new=mock_manager):
        memory = AutonomousMemory()
        
        # Set up the memory integration mock with proper path
        memory.memory_integration = AsyncMock()
        memory.memory_integration.save_standardized_memory = AsyncMock(
            return_value={"success": True, "path": f"{mock_manager.memory_dir}/conversations/test-memory.md"}
        )
        
        # Initialize with test values
        memory.last_memory_check = datetime.now() - timedelta(seconds=30)  # Ensure memory check passes
        
        yield memory
        
        # Clean up temp directory after test
        mock_manager.cleanup()


@pytest.mark.asyncio
async def test_should_store_conversation(memory_system):
    """Test criteria for deciding whether to store a conversation."""
    
    # Test case 1: Short exchanges shouldn't be stored
    short_input = "Hello"
    short_output = "Hi there"
    should_store, metadata = memory_system._should_store_conversation(short_input, short_output)
    assert not should_store
    
    # Test case 2: Conversations with important indicators should be stored
    important_input = "Please remember this is very important information about AI"
    important_output = "I'll make sure to remember that AI is an important topic for discussion"
    should_store, metadata = memory_system._should_store_conversation(important_input, important_output)
    assert should_store
    assert "important" in metadata["tags"]
    assert "conversation" in metadata["tags"]
    
    # Test case 3: Conversations with multiple proper nouns should be stored
    proper_noun_input = "I want to tell you about Amazon and Google and Facebook"
    proper_noun_output = "That's interesting information about these major tech companies Amazon, Google and Facebook"
    should_store, metadata = memory_system._should_store_conversation(proper_noun_input, proper_noun_output)
    assert should_store
    assert any("Amazon" in tag for tag in metadata["tags"])
    assert any("Google" in tag for tag in metadata["tags"])
    
    # Test case 4: Long conversations should be stored
    long_input = "I'm wondering about the philosophical implications " + "of consciousness " * 20
    long_output = "That's a fascinating question about consciousness. " + "Let me explore that with you. " * 20
    should_store, metadata = memory_system._should_store_conversation(long_input, long_output)
    assert should_store
    assert "detailed" in metadata["tags"]


@pytest.mark.asyncio
async def test_generate_title_from_content(memory_system):
    """Test the title generation logic for memories."""
    
    # Patch the subject regex to ensure it captures "theory of relativity"
    with patch.object(memory_system, '_extract_potential_topics', 
                      return_value=["theory", "relativity", "Einstein"]):
        # Test case 1: Subject mention pattern
        subject_input = "Tell me about the theory of relativity"
        subject_output = "The theory of relativity was developed by Einstein"
        title = memory_system._generate_title_from_content(subject_input, subject_output)
        assert "theory" in title.lower() or "relativity" in title.lower() or "einstein" in title.lower()
    
    # Test case 2: Proper noun extraction
    proper_noun_input = "What's the relationship between Google and DeepMind?"
    proper_noun_output = "Google acquired DeepMind in 2014"
    title = memory_system._generate_title_from_content(proper_noun_input, proper_noun_output)
    assert "Google" in title or "DeepMind" in title
    
    # Test case 3: First words fallback
    generic_input = "what is the best way to learn programming?"
    generic_output = "There are many approaches to learning programming"
    title = memory_system._generate_title_from_content(generic_input, generic_output)
    # Check that the title includes the first few words of the query
    assert "what is the best way" in title.lower()


@pytest.mark.asyncio
async def test_process_conversation_turn_store(memory_system):
    """Test that conversations are properly stored in memory."""
    user_input = "Remember that I prefer coffee with no sugar and a splash of milk"
    assistant_response = "I'll remember your coffee preference: no sugar with a splash of milk. Is there anything else about your preferences you'd like me to note?"
    
    # Mock the save_standardized_memory to return proper data
    memory_system.memory_integration.save_standardized_memory = AsyncMock(
        return_value={"success": True, "path": "memories/conversations/coffee-preference.md"}
    )
    
    # Use patching for _should_store_conversation
    with patch.object(memory_system, '_should_store_conversation', 
                      return_value=(True, {"tags": ["preference", "coffee"]})):
        with patch.object(memory_system, '_generate_title_from_content', 
                          return_value="Coffee Preference"):
            # Process the conversation turn
            modified_response, context = await memory_system.process_conversation_turn(user_input, assistant_response)
            
            # Verify memory was stored with correct parameters
            memory_system.memory_integration.save_standardized_memory.assert_called_once()
            
            # Get the arguments from the mock call
            call_args = memory_system.memory_integration.save_standardized_memory.call_args
            args, kwargs = call_args
            
            # Check call arguments
            assert args[0] == "conversations"  # memory type
            assert "User:" in args[1] and "Assistant:" in args[1]  # content
            assert args[2] == "Coffee Preference"  # title
            assert "preference" in kwargs.get("tags", []) or "coffee" in kwargs.get("tags", [])


@pytest.mark.asyncio
async def test_process_conversation_turn_retrieve(memory_system):
    """Test retrieving relevant memories for a query."""
    user_query = "What do you remember about my coffee preferences?"
    
    # Mock the search function to return a relevant result
    mock_result = {
        'success': True,
        'results': [{
            'title': 'Coffee Preference',
            'preview': 'You prefer coffee with no sugar and a splash of milk',
            'score': 0.92,
        }]
    }
    
    # Create patches for all required methods
    with patch('src.memory.register_memory_tools.semantic_memory_search', 
               new_callable=AsyncMock, return_value=mock_result):
        with patch.object(memory_system, '_should_augment_with_memory', return_value=True):
            with patch.object(memory_system, '_is_memory_relevant_to_query', 
                             new_callable=AsyncMock, return_value=True):
                with patch.object(memory_system, '_calculate_memory_confidence', 
                                 new_callable=AsyncMock, return_value="high"):
                    # Process pre-response (when response_text is None)
                    response, context = await memory_system.process_conversation_turn(user_query, None)
                    
                    # Verify that memory context was returned
                    assert context is not None
                    assert context.get('has_memory_context') is True
                    assert 'memory_response' in context
                    assert 'You prefer coffee' in context.get('memory_response', '') or 'Coffee Preference' in context.get('memory_source', '')
                    assert context.get('confidence') == 'high'


@pytest.mark.asyncio
async def test_is_knowledge_seeking_query(memory_system):
    """Test detection of memory-seeking queries."""
    
    # Create a mock memory manager
    mock_manager = MockMemoryManager()
    
    # We need to patch the main_app_integration.memory_manager which is imported in the method
    with patch('src.memory.main_app_integration.memory_manager', new=mock_manager):
        # Simple cases that should work without embeddings
        assert await memory_system._is_knowledge_seeking_query("What do you remember about my preferences?")
        assert await memory_system._is_knowledge_seeking_query("Can you recall what I told you about my job?")
        assert await memory_system._is_knowledge_seeking_query("What did I mention about my family?")
        
        # Negative cases
        assert not await memory_system._is_knowledge_seeking_query("What is Python?")
        assert not await memory_system._is_knowledge_seeking_query("Tell me a joke")


@pytest.mark.asyncio
async def test_extract_potential_topics(memory_system):
    """Test topic extraction from text."""
    
    # Test extraction of proper nouns
    text_with_proper_nouns = "I like Google and Microsoft products better than Apple's"
    topics = memory_system._extract_potential_topics(text_with_proper_nouns)
    assert "Google" in topics
    assert "Microsoft" in topics
    assert "Apple" in topics
    
    # Test extraction of important terms
    text_with_terms = "Let's discuss philosophy and consciousness"
    topics = memory_system._extract_potential_topics(text_with_terms)
    assert any(topic.lower() == "philosophy" for topic in topics)
    assert any(topic.lower() == "consciousness" for topic in topics)
    
    # Test extraction of noun phrases
    text_with_phrases = "I like my blue jacket and your red shoes"
    topics = memory_system._extract_potential_topics(text_with_phrases)
    assert "blue jacket" in topics
    
    # Test limiting to top topics
    long_text = "Google Microsoft Apple Amazon Facebook Twitter Netflix Spotify Uber"
    topics = memory_system._extract_potential_topics(long_text)
    assert len(topics) <= 3  # Should be limited to top 3


@pytest.mark.asyncio
async def test_is_memory_relevant_to_query(memory_system):
    """Test relevance detection between memory and query."""
    
    # Create a mock memory manager
    mock_manager = MockMemoryManager()
    
    with patch('src.memory.main_app_integration.memory_manager', new=mock_manager):
        # Test using TF-IDF similarity directly to ensure test independence
        # Strong keyword match
        memory = "I prefer coffee with no sugar and a splash of milk"
        query = "What are my coffee preferences?"
        similarity = calculate_tfidf_similarity(memory, query)
        assert similarity > 0.3  # Verify with direct TF-IDF that these are similar
        
        # Test proper noun match
        memory = "I visited Paris last summer and loved the Eiffel Tower"
        query = "What do you know about my trip to Paris?"
        similarity = calculate_tfidf_similarity(memory, query)
        assert similarity > 0.3  # Verify with direct TF-IDF that these are similar
        
        # Test irrelevant memory - using TF-IDF should confirm these are different
        memory = "I have a dog named Max"
        query = "What are my coffee preferences?"
        similarity = calculate_tfidf_similarity(memory, query)
        assert similarity < 0.3  # Verify with direct TF-IDF that these are NOT similar
        
        # Now test the actual method implementation with our verified examples
        assert await memory_system._is_memory_relevant_to_query(
            "I prefer coffee with no sugar and a splash of milk", 
            "What are my coffee preferences?"
        )
        assert await memory_system._is_memory_relevant_to_query(
            "I visited Paris last summer and loved the Eiffel Tower",
            "What do you know about my trip to Paris?"
        )
        assert not await memory_system._is_memory_relevant_to_query(
            "I have a dog named Max",
            "What are my coffee preferences?"
        )


@pytest.mark.asyncio
async def test_is_similar_memory(memory_system):
    """Test detection of similar memories to avoid repetition."""
    
    # Create a mock memory manager
    mock_manager = MockMemoryManager()
    
    with patch('src.memory.main_app_integration.memory_manager', new=mock_manager):
        # Test using TF-IDF similarity directly to ensure test independence
        # Similar memories
        memory1 = "I prefer coffee with no sugar and a splash of milk"
        memory2 = "My coffee preference is black coffee with a little bit of milk and no sugar"
        similarity = calculate_tfidf_similarity(memory1, memory2)
        assert similarity > 0.3  # Verify with direct TF-IDF that these are similar
        
        # Different memories
        memory1 = "I prefer coffee with no sugar and a splash of milk"
        memory2 = "I have a dog named Max who is a golden retriever"
        similarity = calculate_tfidf_similarity(memory1, memory2)
        assert similarity < 0.3  # Verify with direct TF-IDF that these are NOT similar
        
        # Now test the actual method implementation with our verified examples
        assert await memory_system._is_similar_memory(
            "I prefer coffee with no sugar and a splash of milk",
            "My coffee preference is black coffee with a little bit of milk and no sugar"
        )
        assert not await memory_system._is_similar_memory(
            "I prefer coffee with no sugar and a splash of milk",
            "I have a dog named Max who is a golden retriever"
        )


@pytest.mark.asyncio
async def test_should_augment_with_memory(memory_system):
    """Test criteria for adding memory context to responses."""
    
    # Reset the last memory check to simulate elapsed time
    memory_system.last_memory_check = datetime.now() - timedelta(seconds=30)
    
    # Test opinion/preference question using the function directly
    opinion_query = "What's my opinion on climate change?"
    assert "opinion" in opinion_query.lower() and "my" in opinion_query.lower()
    assert memory_system._should_augment_with_memory(opinion_query)
    
    # Now test with patched knowledge seeking
    with patch.object(memory_system, '_is_knowledge_seeking_query', new_callable=AsyncMock, return_value=True):
        # Test direct memory request
        assert memory_system._should_augment_with_memory("Remember what I told you about my dog?")
    
    # Test non-memory related query
    with patch.object(memory_system, '_is_knowledge_seeking_query', new_callable=AsyncMock, return_value=False):
        assert not memory_system._should_augment_with_memory("What's the weather like today?")
    
    # Test throttling by time
    memory_system.last_memory_check = datetime.now()  # Reset to current time
    assert not memory_system._should_augment_with_memory("Remember what I told you about my dog?")


@pytest.mark.asyncio
async def test_calculate_memory_confidence(memory_system):
    """Test confidence calculation for memory relevance."""
    
    # Create a mock memory manager
    mock_manager = MockMemoryManager()
    
    with patch('src.memory.main_app_integration.memory_manager', new=mock_manager):
        # Test using TF-IDF similarity to verify confidence rating
        
        # High confidence case - direct test with TF-IDF first
        memory = "I visited Paris last summer and loved the Eiffel Tower"
        query = "What do you remember about my trip to Paris and the Eiffel Tower?"
        similarity = calculate_tfidf_similarity(memory, query)
        assert similarity > 0.6  # This should be high confidence
        
        # Medium confidence case - direct test with TF-IDF first
        memory = "I prefer coffee with no sugar and a splash of milk"
        query = "Remember anything about my coffee?"
        similarity = calculate_tfidf_similarity(memory, query)
        assert 0.3 < similarity < 0.6  # This should be medium confidence
        
        # Low confidence case - direct test with TF-IDF first
        memory = "I prefer coffee with no sugar and a splash of milk"
        query = "What beverages do I like?"
        similarity = calculate_tfidf_similarity(memory, query)
        assert similarity < 0.3  # This should be low confidence
        
        # Now test actual method with verified examples
        from src.memory.memory_utils import classify_memory_confidence
        
        # Test high confidence
        confidence = await classify_memory_confidence(
            "I visited Paris last summer and loved the Eiffel Tower",
            "What do you remember about my trip to Paris and the Eiffel Tower?"
        )
        assert confidence == "high"
        
        # Test medium confidence
        confidence = await classify_memory_confidence(
            "I prefer coffee with no sugar and a splash of milk", 
            "Remember anything about my coffee?"
        )
        assert confidence == "medium"
        
        # Test method directly
        confidence = await memory_system._calculate_memory_confidence(
            "I visited Paris last summer and loved the Eiffel Tower",
            "What do you remember about my trip to Paris and the Eiffel Tower?"
        )
        assert confidence == "high"
        
        confidence = await memory_system._calculate_memory_confidence(
            "I prefer coffee with no sugar and a splash of milk",
            "Remember anything about my coffee?"
        )
        assert confidence == "medium"


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])