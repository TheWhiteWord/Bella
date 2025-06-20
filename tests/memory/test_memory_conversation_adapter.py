import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from bella_memory.memory_conversation_adapter import MemoryConversationAdapter

class DummyClassifier:
    async def classify(self, chunk): return []
class DummyTopicExtractor:
    async def extract(self, chunk): return []
class DummySummarizer:
    async def summarize_self_insight(self, chunk): return "summary"
    async def summarize_user_observation(self, chunk): return "summary"
    async def summarize(self, chunk): return "summary"

class DummyImportanceScorer:
    async def score(self, content):
        print(f"DummyImportanceScorer.score called with: {content}")
        if content.strip().splitlines() and "important" in content.strip().splitlines()[-1]:
            return 1.0
        return 0.1

class DummyBellaMemory:
    def __init__(self, *args, **kwargs):
        self.memory_classifier = DummyClassifier()
        self.topic_extractor = DummyTopicExtractor()
        self.summarizer = DummySummarizer()
        self.importance_scorer = DummyImportanceScorer()
        self.stored = []
    async def store_memory(self, *args, **kwargs):
        print(f"DummyBellaMemory.store_memory called with args: {args}, kwargs: {kwargs}")
        self.stored.append((args, kwargs))
        return ["dummy_path"]
        
# Test: Buffer flushes on max items
@pytest.mark.asyncio
async def test_flush_on_max_items():
    with patch("bella_memory.memory_conversation_adapter.BellaMemory", DummyBellaMemory):
        adapter = MemoryConversationAdapter()
        # All but last message are normal, last triggers importance
        for i in range(adapter.buffer_max_items - 1):
            await adapter.post_process_response(f"user {i}", f"assistant {i}")
        await adapter.post_process_response("user important", "assistant important")
        await asyncio.sleep(0.1)
        assert len(adapter.memory_buffer) == 0
        assert len(adapter.bella_memory.stored) == 1

# Test: Buffer flushes on importance
@pytest.mark.asyncio
async def test_flush_on_importance():
    with patch("bella_memory.memory_conversation_adapter.BellaMemory", DummyBellaMemory):
        adapter = MemoryConversationAdapter()
        await adapter.post_process_response("user normal", "assistant normal")
        await asyncio.sleep(0.1)
        assert len(adapter.memory_buffer) == 1
        await adapter.post_process_response("user important", "assistant important")
        await asyncio.sleep(0.1)
        assert len(adapter.memory_buffer) == 0
        assert len(adapter.bella_memory.stored) == 1

# Test: Manual flush
@pytest.mark.asyncio
async def test_manual_flush():
    with patch("bella_memory.memory_conversation_adapter.BellaMemory", DummyBellaMemory):
        adapter = MemoryConversationAdapter()
        for i in range(2):
            await adapter.post_process_response(f"user {i}", f"assistant {i}")
        # Add an important message so flush will store
        await adapter.post_process_response("user important", "assistant important")
        await asyncio.sleep(0.1)
        # Buffer should already be flushed due to importance, so it should be empty
        assert len(adapter.memory_buffer) == 0
        assert len(adapter.bella_memory.stored) == 1
