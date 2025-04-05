import pytest
import asyncio
from src.mcp_servers.search_agent import SearchAgent
from src.llm.chat_manager import is_search_request, format_search_response

@pytest.mark.asyncio
async def test_search():
    """Test basic search functionality"""
    agent = SearchAgent(model="Lexi")
    results = await agent.search("Python programming")
    assert results is not None
    assert len(results) > 0
    assert all('title' in result and 'link' in result for result in results)

@pytest.mark.asyncio
async def test_fetch_content():
    """Test content fetching from a URL"""
    agent = SearchAgent()
    # Using a stable test URL
    content = await agent.fetch_content("https://www.python.org")
    assert content is not None
    assert len(content) > 0

@pytest.mark.asyncio
async def test_summarize_text():
    """Test text summarization"""
    agent = SearchAgent(model="Lexi")
    test_text = """Python is a high-level, general-purpose programming language. 
    Python's design philosophy emphasizes code readability with the use of significant 
    indentation. Python is dynamically typed and garbage-collected. It supports multiple 
    programming paradigms, including structured, object-oriented and functional programming."""
    
    summary = await agent.summarize_text(test_text, max_length=50)
    assert summary is not None
    assert len(summary) > 0

@pytest.mark.asyncio
async def test_research_topic():
    """Test full research workflow"""
    agent = SearchAgent(model="Lexi")
    research = await agent.research_topic("latest Python version features")
    assert research is not None
    assert len(research) > 0
    assert "Research Results" in research

@pytest.mark.asyncio
async def test_search_intent_detection():
    """Test search intent detection"""
    assert is_search_request("search for Python programming")
    assert is_search_request("look up the weather")
    assert is_search_request("tell me about quantum computing")
    assert is_search_request("what is machine learning")
    assert is_search_request("who is Alan Turing")
    assert not is_search_request("hello how are you")
    assert not is_search_request("that's interesting")

@pytest.mark.asyncio
async def test_search_response_formatting():
    """Test conversational formatting of search results"""
    # Test with valid research results
    research_results = """# Research Results: Python programming

## Depth Level 0

- Python is a high-level programming language known for its simplicity.
  Source: https://www.python.org
- Python supports multiple programming paradigms including OOP.
  Source: https://wiki.python.org
- It has a comprehensive standard library.
  Source: https://docs.python.org

## Depth Level 1
- Some other details
  Source: https://example.com
"""
    
    formatted = await format_search_response(research_results)
    assert "Based on what I found" in formatted
    assert "Python is a high-level programming language" in formatted
    assert "Would you like me to elaborate" in formatted
    
    # Test with no results
    no_results = "I couldn't find any relevant information."
    formatted = await format_search_response(no_results)
    assert "couldn't find any relevant information" in formatted
    assert "try rephrasing" in formatted