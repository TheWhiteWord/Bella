import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from llm import chat_manager
from bella_memory.tools_self_user_awareness import summarize_self_awareness, set_bella_memory_instance

class DummyMemoryGraph:
    def query(self, perspective=None, subject=None, type_=None):
        # Return mock memories for self-awareness
        if perspective == "self" and subject == "Bella":
            if type_ == "think":
                return [{"value": "I am always learning."}]
            if type_ == "feel":
                return [{"value": "I feel optimistic about the future."}]
            if type_ == "prefer":
                return [{"value": "I prefer clarity over ambiguity."}]
        return []

class DummyBellaMemory:
    def __init__(self):
        self.memory_graph = DummyMemoryGraph()

@pytest.mark.asyncio
async def test_summarize_self_awareness_injection(monkeypatch):
    # Patch the LLM generate function to just echo the prompt for test
    async def fake_generate(prompt, qwen_size=None, thinking_mode=None):
        # Return a recognizable summary
        assert "Summarize the following self-knowledge" in prompt
        return "I am Bella. I am always learning, feel optimistic, and prefer clarity."
    monkeypatch.setattr("bella_memory.helpers.generate", fake_generate)

    # Patch BellaMemory instance
    dummy_bella_memory = DummyBellaMemory()
    set_bella_memory_instance(dummy_bella_memory)

    # Actually call the tool
    summary = await summarize_self_awareness()
    assert "I am Bella" in summary

    # Now test chat_manager system prompt injection
    user_input = "Hello, who are you?"
    conversation_history = []
    # Patch generate_with_tools to capture the system prompt
    async def fake_generate_with_tools(prompt, history, tools, model, system_prompt, **kwargs):
        assert "I am Bella" in system_prompt
        assert "[BELLA SELF-AWARENESS]" in system_prompt
        return {"message": {"content": "Test response."}}
    monkeypatch.setattr(chat_manager, "generate_with_tools", fake_generate_with_tools)

    # Patch tool registry
    monkeypatch.setattr(chat_manager.tools_registry, "get_available_tools", lambda: [])
    monkeypatch.setattr(chat_manager.tools_registry, "get_all_functions", lambda: {})

    response, _ = await chat_manager.generate_chat_response_with_tools(
        user_input,
        conversation_history,
        self_awareness_summary=summary
    )
    assert response == "Test response."
