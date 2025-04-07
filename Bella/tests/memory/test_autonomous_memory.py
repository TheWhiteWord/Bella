"""Tests for the autonomous memory system.

Tests the functionality of the autonomous memory system with the new standardized format,
focusing on philosophical, artistic, and consciousness-related content, and ChromaDB integration.
"""

import os
import sys
import pytest
import asyncio
import re
import tempfile
import shutil  # Import shutil for cleanup
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.memory.autonomous_memory import AutonomousMemory


# Mock ChromaDB Collection results
class MockChromaResults:
    def __init__(self, ids=None, distances=None, metadatas=None):
        self.ids = [ids or []]
        self.distances = [distances or []]
        self.metadatas = [metadatas or []]

    def get(self, key, default=None):
        if key == 'ids':
            return self.ids
        elif key == 'distances':
            return self.distances
        elif key == 'metadatas':
            return self.metadatas
        return default


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

        # --- Mock the enhanced adapter and its ChromaDB interactions ---
        self.enhanced_adapter = AsyncMock()
        self.enhanced_adapter.processor = AsyncMock()
        self.enhanced_adapter.processor.generate_embedding = AsyncMock(return_value=[0.1] * 768)
        # Mock methods that interact with ChromaDB
        self.enhanced_adapter._add_memory_to_vector_db = AsyncMock(return_value=None)
        # Mock search_memory to return the new format (results dict, success bool)
        # Default mock returns no results successfully
        self.enhanced_adapter.search_memory = AsyncMock(return_value=({"results": []}, True))
        # Mock other relevant adapter methods used by AutonomousMemory
        self.enhanced_adapter.should_store_memory = AsyncMock(return_value=(True, 0.8))  # Default to store
        self.enhanced_adapter.compare_memory_similarity = AsyncMock(return_value=0.8)  # Default high similarity
        self.enhanced_adapter.detect_memory_topics = AsyncMock(return_value=["mock_topic1", "mock_topic2"])

    def cleanup(self):
        """Clean up the temporary directory after tests"""
        if os.path.exists(self.temp_dir):
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
        # Patch get_memory_integration to return a mock that saves files to temp dir
        mock_integration = AsyncMock()

        async def mock_save_standardized(mem_type, content, title, tags=None):
            safe_title = re.sub(r'[^\w\-]+', '-', title.lower())
            path = os.path.join(mock_manager.memory_dir, mem_type, f"{safe_title}.md")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(f"---\ntitle: {title}\ntags: {tags}\n---\n{content}")
            return {"success": True, "path": path}

        mock_integration.save_standardized_memory = AsyncMock(side_effect=mock_save_standardized)

        with patch('src.memory.autonomous_memory.get_memory_integration', return_value=mock_integration):
            memory = AutonomousMemory()
            # Initialize with test values
            memory.last_memory_check = datetime.now() - timedelta(seconds=30)  # Ensure memory check passes
            yield memory  # Provide the configured memory system to the test

    # Clean up temp directory after test using the fixture's scope
    mock_manager.cleanup()


@pytest.mark.asyncio
async def test_should_store_conversation(memory_system):
    """Test criteria for deciding whether to store a conversation."""
    mock_manager = memory_system.memory_integration  # Get the mock manager via the patched integration

    # Test case 1: Short exchanges shouldn't be stored (based on length check)
    short_input = "Hello"
    short_output = "Hi there"
    # Temporarily mock should_store_memory to return False for this specific case if needed
    memory_system.memory_integration.enhanced_adapter.should_store_memory = AsyncMock(return_value=(False, 0.1))
    should_store, metadata = await memory_system._should_store_conversation(short_input, short_output)
    assert not should_store

    # Reset mock for subsequent tests
    memory_system.memory_integration.enhanced_adapter.should_store_memory = AsyncMock(return_value=(True, 0.85))
    memory_system.memory_integration.enhanced_adapter.detect_memory_topics = AsyncMock(return_value=["consciousness", "free-will"])

    # Test case 2: Conversations deemed important by semantic check
    important_input = "Please remember this profound insight about consciousness and free will"
    important_output = "I'll make note of your perspective on consciousness being an emergent property that gives rise to the illusion of free will"
    should_store, metadata = await memory_system._should_store_conversation(important_input, important_output)
    assert should_store
    assert "important" in metadata["tags"]
    assert "consciousness" in metadata["tags"]  # Check for semantic topic tag

    # Test case 3: Conversations with multiple semantic topics
    memory_system.memory_integration.enhanced_adapter.detect_memory_topics = AsyncMock(return_value=["hegel", "kant", "nietzsche"])
    philosophical_input = "I want to discuss how Hegel's dialectic relates to Kant's transcendental idealism and Nietzsche's perspectivism"
    philosophical_output = "That's a fascinating intersection..."
    should_store, metadata = await memory_system._should_store_conversation(philosophical_input, philosophical_output)
    assert should_store
    assert "hegel" in metadata["tags"]
    assert "kant" in metadata["tags"]
    assert "nietzsche" in metadata["tags"]

    # Test case 4: Long conversations (fallback check if semantic fails or is low)
    memory_system.memory_integration.enhanced_adapter.should_store_memory = AsyncMock(return_value=(False, 0.5))  # Simulate low semantic score
    long_input = "I'm wondering about the nature of aesthetic experience " + "and how art creates transcendent meaning " * 20
    long_output = "That's a profound question about aesthetics. " + "The phenomenology of artistic experience suggests that meaning emerges through both creator and observer. " * 20
    should_store, metadata = await memory_system._should_store_conversation(long_input, long_output)
    assert should_store  # Should still store due to length fallback
    assert "detailed" in metadata["tags"]


@pytest.mark.asyncio
async def test_generate_title_from_content(memory_system):
    """Test the title generation logic for memories using semantic topics."""
    mock_manager = memory_system.memory_integration  # Get the mock manager

    # Test case 1: Title from semantic topics
    mock_manager.enhanced_adapter.detect_memory_topics = AsyncMock(return_value=["consciousness", "qualia"])
    subject_input = "Tell me about the hard problem of consciousness"
    subject_output = "The hard problem concerns subjective experiences or qualia."
    title = await memory_system._generate_title_from_content(subject_input, subject_output)
    assert "consciousness" in title.lower()
    assert "qualia" in title.lower()

    # Test case 2: Fallback to question if no topics found
    mock_manager.enhanced_adapter.detect_memory_topics = AsyncMock(return_value=[])
    question_input = "What is the meaning of life in existential philosophy?"
    question_output = "Existentialists often argue that individuals create their own meaning."
    title = await memory_system._generate_title_from_content(question_input, question_output)
    assert "what is the meaning of life" in title.lower()

    # Test case 3: Fallback to user input start
    mock_manager.enhanced_adapter.detect_memory_topics = AsyncMock(return_value=[])
    generic_input = "Let's talk about art and beauty."
    generic_output = "Okay, aesthetics is a fascinating field."
    title = await memory_system._generate_title_from_content(generic_input, generic_output)
    assert "let's talk about art..." in title.lower()

    # Test case 4: Fallback to timestamp
    mock_manager.enhanced_adapter.detect_memory_topics = AsyncMock(side_effect=Exception("Topic detection failed"))
    error_input = "Some input."
    error_output = "Some output."
    title = await memory_system._generate_title_from_content(error_input, error_output)
    assert "Conversation on" in title


@pytest.mark.asyncio
async def test_process_conversation_turn_store(memory_system):
    """Test that conversations are properly stored and indexed in ChromaDB."""
    mock_manager = memory_system.memory_integration  # Get the mock manager

    user_input = "Remember that I find Camus' concept of absurdism more compelling than Sartre's existentialism because it acknowledges the inherent meaninglessness of existence while still finding value in the struggle"
    assistant_response = "I'll remember your philosophical preference..."

    # Configure mocks for this specific test run
    memory_system.memory_integration.save_standardized_memory = memory_system.memory_integration.save_standardized_memory  # Use the fixture's mock
    mock_manager.enhanced_adapter.should_store_memory = AsyncMock(return_value=(True, {"tags": ["philosophy", "existentialism", "absurdism", "important"], "is_important": True, "importance_score": 0.9}))
    mock_manager.enhanced_adapter.detect_memory_topics = AsyncMock(return_value=["camus", "sartre", "absurdism"])
    mock_manager.enhanced_adapter._add_memory_to_vector_db = AsyncMock()  # Ensure this is mocked

    # Process the conversation turn
    modified_response, context = await memory_system.process_conversation_turn(user_input, assistant_response)

    # Verify file was saved
    memory_system.memory_integration.save_standardized_memory.assert_called_once()
    call_args, call_kwargs = memory_system.memory_integration.save_standardized_memory.call_args
    assert call_args[0] == "conversations"
    assert "Camus" in call_args[1]  # Check content
    assert "camus" in call_args[2].lower() or "sartre" in call_args[2].lower()  # Check title based on topics
    assert "philosophy" in call_kwargs.get("tags", [])

    # *** Verify ChromaDB indexing was called ***
    mock_manager.enhanced_adapter._add_memory_to_vector_db.assert_called_once()
    index_call_args, index_call_kwargs = mock_manager.enhanced_adapter._add_memory_to_vector_db.call_args
    # Check memory_id format (e.g., 'conversations/discussion-about-camus-and-sartre')
    assert index_call_kwargs['memory_id'].startswith("conversations/")
    assert "camus" in index_call_kwargs['memory_id']
    # Check content passed for embedding
    assert "Camus" in index_call_kwargs['content']
    # Check metadata passed to ChromaDB
    metadata = index_call_kwargs['metadata']
    assert "file_path" in metadata
    assert metadata["file_path"].endswith(".md")
    assert metadata["title"] == call_args[2]  # Title should match saved file title
    assert "philosophy" in metadata["tags"]  # Check tags are passed
    assert "importance_score" in metadata

    # Verify response modification for important memory
    assert "I've noted this" in modified_response


@pytest.mark.asyncio
async def test_process_conversation_turn_retrieve(memory_system):
    """Test retrieving relevant memories via ChromaDB search."""
    mock_manager = memory_system.memory_integration  # Get the mock manager
    user_query = "What do you remember about my views on consciousness and free will?"

    # --- Mock the semantic_memory_search tool result (which uses adapter.search_memory) ---
    # This now needs to return the format expected from the tool, based on ChromaDB results
    mock_search_result = {
        'success': True,
        'results': [{
            'id': 'conversations/consciousness-and-free-will-discussion',
            'title': 'Consciousness and Free Will Discussion',
            'path': f'{mock_manager.memory_dir}/conversations/consciousness-and-free-will-discussion.md',
            'score': 0.92,  # Similarity score
            'distance': 0.08,
            'tags': ['consciousness', 'free-will', 'philosophy'],
            'created_at': datetime.now().isoformat(),
            'preview': 'You believe consciousness is an emergent property that gives rise to the illusion of free will...',  # Preview might come from metadata or file read
            'metadata': {'file_path': f'{mock_manager.memory_dir}/conversations/consciousness-and-free-will-discussion.md', 'title': 'Consciousness and Free Will Discussion', 'tags': 'consciousness,free-will,philosophy'}
        }]
    }

    # Patch the tool function directly as it's called by AutonomousMemory
    with patch('src.memory.autonomous_memory.semantic_memory_search', new_callable=AsyncMock, return_value=mock_search_result):
        # Mock other checks within AutonomousMemory
        with patch.object(memory_system, '_should_augment_with_memory', return_value=True):
            # Mock relevance/confidence checks (can rely on score from search result now)
            memory_system.memory_threshold = 0.75  # Ensure threshold is set for test
            # Process pre-response (when response_text is None)
            response, context = await memory_system.process_conversation_turn(user_query, None)

            # Verify that memory context was returned based on search results
            assert context is not None
            assert context.get('has_memory_context') is True
            assert 'memory_response' in context
            # Check content based on the mocked preview
            assert 'consciousness is an emergent property' in context.get('memory_response', '')
            assert context.get('confidence') == 'high'  # Based on score 0.92 > 0.8
            assert context.get('memory_source') == 'Consciousness and Free Will Discussion'
            assert context.get('score') == 0.92


@pytest.mark.asyncio
async def test_should_augment_with_memory_simplified(memory_system):
    """Test simplified criteria for adding memory context to responses."""
    # Reset the last memory check to simulate elapsed time
    memory_system.last_memory_check = datetime.now() - timedelta(seconds=30)

    # Test explicit recall phrases
    assert await memory_system._should_augment_with_memory("remember what i told you about Nietzsche?")
    assert await memory_system._should_augment_with_memory("what was my opinion on Kant?")

    # Test personal query patterns
    assert await memory_system._should_augment_with_memory("What's my view on the mind-body problem?")
    assert await memory_system._should_augment_with_memory("How did I feel about that art exhibition?")

    # Test general questions with context terms (should trigger semantic search)
    assert await memory_system._should_augment_with_memory("What about our discussion on ethics?")

    # Test general knowledge questions (should NOT trigger augmentation)
    assert not await memory_system._should_augment_with_memory("What is epistemology?")
    assert not await memory_system._should_augment_with_memory("Tell me about Plato.")

    # Test throttling by time
    memory_system.last_memory_check = datetime.now()  # Reset to current time
    assert not await memory_system._should_augment_with_memory("Remember what I told you about free will?")


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])