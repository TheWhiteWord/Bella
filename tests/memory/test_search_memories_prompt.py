import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from bella_memory.tools_self_user_awareness import search_memories, set_bella_memory_instance

class DummyBellaMemory:
    async def search_memories(self, query, top_k=3):
        # Simulate vector DB search result
        return [
            {"content": "Pandas is a Python library for data analysis.", "metadata": {"memory_id": "abc123", "summary": "Pandas is a Python library..."}},
            {"content": "Bella likes using pandas for data cleaning.", "metadata": {"memory_id": "def456"}},
            {"content": "Numpy is often used with pandas.", "metadata": {"memory_id": "ghi789", "summary": "Numpy is often used..."}},
        ]

@pytest.mark.asyncio
async def test_search_memories_prompt():
    """Test to show exactly what the LLM receives after a search_memories tool call."""
    # Patch BELLA_MEMORY
    set_bella_memory_instance(DummyBellaMemory())
    query = "pandas"
    tool_result = await search_memories(query)

    # Simulate what the LLM would receive as a tool result
    # In a real system, this would be inserted into the conversation history as a 'tool' message
    system_prompt = "You are Bella, an AI assistant. Respond naturally and conversationally."
    user_prompt = f"search my memories for '{query}'"
    tool_message = {"role": "tool", "name": "search_memories", "content": tool_result}

    # Compose the full prompt as the LLM would see it
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        tool_message
    ]

    print("\n--- LLM RECEIVES THIS PROMPT ---")
    for m in messages:
        print(f"[{m['role'].upper()}]", end=" ")
        if m.get("name"):
            print(f"({m['name']})", end=" ")
        print(m["content"])
    print("--- END PROMPT ---\n")
    # Optionally, add assertions here to check the prompt structure
