"""Tests for the autonomous memory system.

Tests the functionality of the autonomous memory system with the new standardized format,
focusing on philosophical, artistic, and consciousness-related content.
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
    important_input = "Please remember this profound insight about consciousness and free will"
    important_output = "I'll make note of your perspective on consciousness being an emergent property that gives rise to the illusion of free will"
    should_store, metadata = memory_system._should_store_conversation(important_input, important_output)
    assert should_store
    assert "important" in metadata["tags"]
    assert "conversation" in metadata["tags"]
    
    # Test case 3: Conversations with multiple philosophical concepts should be stored
    philosophical_input = "I want to discuss how Hegel's dialectic relates to Kant's transcendental idealism and Nietzsche's perspectivism"
    philosophical_output = "That's a fascinating intersection of philosophical frameworks. Hegel's dialectic process of thesis-antithesis-synthesis does seem to build upon yet critique Kant's transcendental idealism, while Nietzsche's perspectivism challenges both by rejecting absolute truth claims altogether."
    should_store, metadata = memory_system._should_store_conversation(philosophical_input, philosophical_output)
    assert should_store
    assert any("Hegel" in tag for tag in metadata["tags"])
    assert any("Kant" in tag for tag in metadata["tags"])
    assert any("Nietzsche" in tag for tag in metadata["tags"])
    
    # Test case 4: Long conversations should be stored
    long_input = "I'm wondering about the nature of aesthetic experience " + "and how art creates transcendent meaning " * 20
    long_output = "That's a profound question about aesthetics. " + "The phenomenology of artistic experience suggests that meaning emerges through both creator and observer. " * 20
    should_store, metadata = memory_system._should_store_conversation(long_input, long_output)
    assert should_store
    assert "detailed" in metadata["tags"]


@pytest.mark.asyncio
async def test_generate_title_from_content(memory_system):
    """Test the title generation logic for memories."""
    
    # Patch the subject regex to ensure it captures philosophical topics
    with patch.object(memory_system, '_extract_potential_topics', 
                      return_value=["consciousness", "qualia", "phenomenology"]):
        # Test case 1: Subject mention pattern
        subject_input = "Tell me about the hard problem of consciousness"
        subject_output = "The hard problem of consciousness, as David Chalmers articulated it, concerns explaining why we have qualitative subjective experiences"
        title = memory_system._generate_title_from_content(subject_input, subject_output)
        assert "consciousness" in title.lower() or "qualia" in title.lower() or "phenomenology" in title.lower()
    
    # Test case 2: Proper noun extraction for philosophers
    proper_noun_input = "What's the relationship between Sartre and Camus regarding existentialism?"
    proper_noun_output = "While both Sartre and Camus are associated with existentialist philosophy, Camus rejected the label, preferring to be known for his absurdism"
    title = memory_system._generate_title_from_content(proper_noun_input, proper_noun_output)
    assert "Sartre" in title or "Camus" in title or "existentialism" in title.lower()
    
    # Test case 3: First words fallback for philosophical questions
    generic_input = "what is the nature of truth in a post-factual society?"
    generic_output = "The nature of truth in a post-factual society raises complex epistemological questions"
    title = memory_system._generate_title_from_content(generic_input, generic_output)
    # Check that the title includes the first few words of the query
    assert "what is the nature of truth" in title.lower()


@pytest.mark.asyncio
async def test_process_conversation_turn_store(memory_system):
    """Test that conversations are properly stored in memory."""
    user_input = "Remember that I find Camus' concept of absurdism more compelling than Sartre's existentialism because it acknowledges the inherent meaninglessness of existence while still finding value in the struggle"
    assistant_response = "I'll remember your philosophical preference for Camus' absurdism over Sartre's existentialism, particularly your appreciation for how Camus acknowledges life's inherent meaninglessness while finding value in the human struggle. Is there a specific work by Camus that resonates most with you?"
    
    # Mock the save_standardized_memory to return proper data
    memory_system.memory_integration.save_standardized_memory = AsyncMock(
        return_value={"success": True, "path": "memories/conversations/philosophical-preferences.md"}
    )
    
    # Use patching for _should_store_conversation
    with patch.object(memory_system, '_should_store_conversation', 
                      return_value=(True, {"tags": ["philosophy", "existentialism", "absurdism"]})):
        with patch.object(memory_system, '_generate_title_from_content', 
                          return_value="Philosophical Preference: Camus over Sartre"):
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
            assert args[2] == "Philosophical Preference: Camus over Sartre"  # title
            assert "philosophy" in kwargs.get("tags", []) or "existentialism" in kwargs.get("tags", [])


@pytest.mark.asyncio
async def test_process_conversation_turn_retrieve(memory_system):
    """Test retrieving relevant memories for a query."""
    user_query = "What do you remember about my views on consciousness and free will?"
    
    # Mock the search function to return a relevant result
    mock_result = {
        'success': True,
        'results': [{
            'title': 'Consciousness and Free Will Discussion',
            'preview': 'You believe consciousness is an emergent property that gives rise to the illusion of free will, though you find the compatibilist position interesting',
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
                    assert ('consciousness' in context.get('memory_response', '').lower() or 
                           'free will' in context.get('memory_response', '').lower())
                    assert context.get('confidence') == 'high'


@pytest.mark.asyncio
async def test_is_knowledge_seeking_query(memory_system):
    """Test detection of memory-seeking queries."""
    
    # Create a mock memory manager
    mock_manager = MockMemoryManager()
    
    # We need to patch the main_app_integration.memory_manager which is imported in the method
    with patch('src.memory.main_app_integration.memory_manager', new=mock_manager):
        # Simple cases that should work without embeddings
        assert await memory_system._is_knowledge_seeking_query("What do you remember about my views on phenomenology?")
        assert await memory_system._is_knowledge_seeking_query("Can you recall what I told you about my interpretation of Plato's cave allegory?")
        assert await memory_system._is_knowledge_seeking_query("What did I mention about my favorite philosophical works?")
        
        # Negative cases
        assert not await memory_system._is_knowledge_seeking_query("What is phenomenology?")
        assert not await memory_system._is_knowledge_seeking_query("Tell me about Nietzsche's concept of eternal recurrence")


@pytest.mark.asyncio
async def test_extract_potential_topics(memory_system):
    """Test topic extraction from text."""
    
    # Test extraction of philosophical proper nouns
    text_with_proper_nouns = "I prefer Heidegger and Wittgenstein's approaches to language over Frege's"
    topics = memory_system._extract_potential_topics(text_with_proper_nouns)
    assert "Heidegger" in topics
    assert "Wittgenstein" in topics
    assert "Frege" in topics
    
    # Test extraction of important philosophical terms
    text_with_terms = "Let's discuss epistemology and consciousness in depth"
    topics = memory_system._extract_potential_topics(text_with_terms)
    assert any(topic.lower() == "epistemology" for topic in topics)
    assert any(topic.lower() == "consciousness" for topic in topics)
    
    # Test extraction of philosophical noun phrases
    text_with_phrases = "I find phenomenological inquiry and transcendental idealism fascinating"
    topics = memory_system._extract_potential_topics(text_with_phrases)
    assert "phenomenological inquiry" in topics or "transcendental idealism" in topics
    
    # Test limiting to top topics
    long_text = "Plato Aristotle Kant Hegel Nietzsche Sartre Camus Wittgenstein Heidegger Foucault"
    topics = memory_system._extract_potential_topics(long_text)
    assert len(topics) <= 3  # Should be limited to top 3


@pytest.mark.asyncio
async def test_is_memory_relevant_to_query(memory_system):
    """Test relevance detection between memory and query."""
    
    # Create a mock memory manager
    mock_manager = MockMemoryManager()
    
    with patch('src.memory.main_app_integration.memory_manager', new=mock_manager):
        # Test using TF-IDF similarity directly to ensure test independence
        # Strong keyword match - philosophical example
        memory = "I believe consciousness is an emergent property of complex neural systems, though it remains fundamentally mysterious"
        query = "What are my thoughts on the hard problem of consciousness?"
        similarity = calculate_tfidf_similarity(memory, query)
        assert similarity > 0.3  # Verify with direct TF-IDF that these are similar
        
        # Test proper noun match with philosophers
        memory = "I find Kierkegaard's concept of anxiety particularly insightful regarding human freedom"
        query = "What do you know about my interest in Kierkegaard?"
        similarity = calculate_tfidf_similarity(memory, query)
        assert similarity > 0.3  # Verify with direct TF-IDF that these are similar
        
        # Test irrelevant philosophical memory - using TF-IDF should confirm these are different
        memory = "I find Kant's categorical imperative to be a compelling ethical framework"
        query = "What are my views on phenomenology and consciousness?"
        similarity = calculate_tfidf_similarity(memory, query)
        assert similarity < 0.3  # Verify with direct TF-IDF that these are NOT similar
        
        # Now test the actual method implementation with our verified examples
        assert await memory_system._is_memory_relevant_to_query(
            "I believe consciousness is an emergent property of complex neural systems, though it remains fundamentally mysterious", 
            "What are my thoughts on the hard problem of consciousness?"
        )
        assert await memory_system._is_memory_relevant_to_query(
            "I find Kierkegaard's concept of anxiety particularly insightful regarding human freedom",
            "What do you know about my interest in Kierkegaard?"
        )
        assert not await memory_system._is_memory_relevant_to_query(
            "I find Kant's categorical imperative to be a compelling ethical framework",
            "What are my views on phenomenology and consciousness?"
        )


@pytest.mark.asyncio
async def test_is_similar_memory(memory_system):
    """Test detection of similar memories to avoid repetition."""
    
    # Create a mock memory manager
    mock_manager = MockMemoryManager()
    
    with patch('src.memory.main_app_integration.memory_manager', new=mock_manager):
        # Test using TF-IDF similarity directly to ensure test independence
        # Similar philosophical memories
        memory1 = "I believe aesthetic experience transcends rational understanding and connects us to deeper truths"
        memory2 = "My view on aesthetics is that art reaches beyond rationality to reveal profound truths about existence"
        similarity = calculate_tfidf_similarity(memory1, memory2)
        assert similarity > 0.3  # Verify with direct TF-IDF that these are similar
        
        # Different philosophical memories
        memory1 = "I believe consciousness arises from complex neural interactions but remains fundamentally irreducible"
        memory2 = "Kant's transcendental idealism suggests that we can never know things-in-themselves"
        similarity = calculate_tfidf_similarity(memory1, memory2)
        assert similarity < 0.3  # Verify with direct TF-IDF that these are NOT similar
        
        # Now test the actual method implementation with our verified examples
        assert await memory_system._is_similar_memory(
            "I believe aesthetic experience transcends rational understanding and connects us to deeper truths",
            "My view on aesthetics is that art reaches beyond rationality to reveal profound truths about existence"
        )
        assert not await memory_system._is_similar_memory(
            "I believe consciousness arises from complex neural interactions but remains fundamentally irreducible",
            "Kant's transcendental idealism suggests that we can never know things-in-themselves"
        )


@pytest.mark.asyncio
async def test_should_augment_with_memory(memory_system):
    """Test criteria for adding memory context to responses."""
    
    # Reset the last memory check to simulate elapsed time
    memory_system.last_memory_check = datetime.now() - timedelta(seconds=30)
    
    # Test opinion/preference question using the function directly
    opinion_query = "What's my view on the mind-body problem?"
    assert "view" in opinion_query.lower() and "my" in opinion_query.lower()
    assert memory_system._should_augment_with_memory(opinion_query)
    
    # Now test with patched knowledge seeking
    with patch.object(memory_system, '_is_knowledge_seeking_query', new_callable=AsyncMock, return_value=True):
        # Test direct memory request about philosophical topic
        assert memory_system._should_augment_with_memory("Remember what I told you about my interpretation of Nietzsche?")
    
    # Test non-memory related philosophical query
    with patch.object(memory_system, '_is_knowledge_seeking_query', new_callable=AsyncMock, return_value=False):
        assert not memory_system._should_augment_with_memory("What is Hegel's dialectic method?")
    
    # Test throttling by time
    memory_system.last_memory_check = datetime.now()  # Reset to current time
    assert not memory_system._should_augment_with_memory("Remember what I told you about my interpretation of free will?")


@pytest.mark.asyncio
async def test_calculate_memory_confidence(memory_system):
    """Test confidence calculation for memory relevance."""
    
    # Create a mock memory manager
    mock_manager = MockMemoryManager()
    
    with patch('src.memory.main_app_integration.memory_manager', new=mock_manager):
        # Test using TF-IDF similarity to verify confidence rating
        
        # High confidence case - direct test with TF-IDF first
        memory = "I believe Camus' concept of the absurd is more honest than Sartre's existentialism because it acknowledges the fundamental meaninglessness of existence"
        query = "What do you remember about my thoughts on Camus versus Sartre?"
        similarity = calculate_tfidf_similarity(memory, query)
        assert similarity > 0.6  # This should be high confidence
        
        # Medium confidence case - direct test with TF-IDF first
        memory = "I find phenomenology to be a compelling approach to understanding consciousness"
        query = "Remember anything about my views on consciousness?"
        similarity = calculate_tfidf_similarity(memory, query)
        assert 0.3 < similarity < 0.6  # This should be medium confidence
        
        # Low confidence case - direct test with TF-IDF first
        memory = "I find phenomenology to be a compelling approach to understanding consciousness"
        query = "What philosophical topics interest me?"
        similarity = calculate_tfidf_similarity(memory, query)
        assert similarity < 0.3  # This should be low confidence
        
        # Now test actual method with verified examples
        from src.memory.memory_utils import classify_memory_confidence
        
        # Test high confidence
        confidence = await classify_memory_confidence(
            "I believe Camus' concept of the absurd is more honest than Sartre's existentialism because it acknowledges the fundamental meaninglessness of existence",
            "What do you remember about my thoughts on Camus versus Sartre?"
        )
        assert confidence == "high"
        
        # Test medium confidence
        confidence = await classify_memory_confidence(
            "I find phenomenology to be a compelling approach to understanding consciousness", 
            "Remember anything about my views on consciousness?"
        )
        assert confidence == "medium"
        
        # Test method directly
        confidence = await memory_system._calculate_memory_confidence(
            "I believe Camus' concept of the absurd is more honest than Sartre's existentialism because it acknowledges the fundamental meaninglessness of existence",
            "What do you remember about my thoughts on Camus versus Sartre?"
        )
        assert confidence == "high"
        
        confidence = await memory_system._calculate_memory_confidence(
            "I find phenomenology to be a compelling approach to understanding consciousness",
            "Remember anything about my views on consciousness?"
        )
        assert confidence == "medium"


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])