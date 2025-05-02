import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from bella_memory.memory_conversation_adapter import MemoryConversationAdapter

class DummyBellaMemory:
    def __init__(self, *args, **kwargs):
        self.stored = []
        self.importance_scorer = AsyncMock()
        self.importance_scorer.score = AsyncMock(side_effect=lambda content: 0.8 if "important" in content else 0.6)
    async def store_memory(self, content, user_context):
        self.stored.append((content, user_context))
        return [f"/fake/path/{len(self.stored)}.md"]

@pytest.mark.asyncio
async def test_buffered_memory_saving():
    # Patch BellaMemory in the adapter to use our dummy
    with patch("bella_memory.memory_conversation_adapter.BellaMemory", DummyBellaMemory):
        adapter = MemoryConversationAdapter()
        adapter.bella_memory = DummyBellaMemory()  # ensure dummy is used
        # Simulate 4 conversation turns (should not flush yet)
        for i in range(4):
            await adapter.post_process_response(f"user {i}", f"assistant {i}")
        assert len(adapter.memory_buffer) == 4
        assert len(adapter.bella_memory.stored) == 0
        # 5th turn should trigger flush
        await adapter.post_process_response("user important", "assistant 5")
        assert len(adapter.memory_buffer) == 0
        assert len(adapter.bella_memory.stored) == 5
        # Test manual flush
        await adapter.post_process_response("user 6", "assistant 6")
        await adapter.flush_memory_buffer()
        assert len(adapter.memory_buffer) == 0
        assert len(adapter.bella_memory.stored) == 6

@pytest.mark.asyncio
async def test_importance_triggers_flush():
    with patch("bella_memory.memory_conversation_adapter.BellaMemory", DummyBellaMemory):
        adapter = MemoryConversationAdapter()
        adapter.bella_memory = DummyBellaMemory()
        # Add 2 normal, then 1 very important
        await adapter.post_process_response("user 1", "assistant 1")
        await adapter.post_process_response("user 2", "assistant 2")
        # This one contains 'important', so importance scorer returns 0.8 (above 0.75 default, but below 0.9 threshold)
        await adapter.post_process_response("user important", "assistant 3")
        # Buffer should not flush yet (since threshold is 0.9)
        assert len(adapter.memory_buffer) == 3
        # Now, set threshold lower and try again
        adapter.buffer_importance_threshold = 0.7
        await adapter._maybe_flush_memory_buffer()
        assert len(adapter.memory_buffer) == 0
        assert len(adapter.bella_memory.stored) == 3
