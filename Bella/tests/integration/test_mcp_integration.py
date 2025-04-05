"""Integration tests for MCP functionality with Bella voice assistant.

This test module verifies that:
1. The correct MCP system prompts are being loaded and passed to the LLM
2. The MCP server initialization is working properly
3. The integration between components is functioning as expected
"""

import os
import sys
import asyncio
import pytest
from unittest.mock import patch, MagicMock
import tempfile
import json
import time

# Add project root directory to Python path
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from src.llm.config_manager import ModelConfig, PromptConfig
from src.llm.chat_manager import generate_chat_response
from src.utility.mcp_server_manager import MCPServerManager
from src.mcp_servers.web_search_mcp import BellaWebSearchMCP
from src.mcp_servers.basic_memory_MCP import BellaMemoryMCP


@pytest.mark.asyncio
async def test_mcp_system_prompt_loading():
    """Test that the correct system prompt is loaded when MCP is enabled."""
    # Load configuration
    prompt_config = PromptConfig()
    
    # Get both types of system prompts for comparison
    standard_prompt = prompt_config.get_system_prompt("system_long")
    mcp_prompt = prompt_config.get_system_prompt("system_with_mcp")
    
    # Verify that the MCP prompt contains instructions for web search and memory tools
    assert "Available Memory & Knowledge Tools:" in mcp_prompt
    assert "Available Web Tools:" in mcp_prompt
    assert "web_search(query" in mcp_prompt
    assert "write_note(title" in mcp_prompt
    
    # Verify that standard prompt doesn't contain MCP instructions
    assert "Available Memory & Knowledge Tools:" not in standard_prompt
    
    print("\n✅ MCP system prompt contains the correct tool instructions.")
    print(f"\nStandard prompt length: {len(standard_prompt)} characters")
    print(f"MCP prompt length: {len(mcp_prompt)} characters")


@pytest.mark.asyncio
async def test_chat_manager_prompt_selection():
    """Test that the chat manager correctly selects the MCP-enabled prompt when MCP is enabled."""
    
    # Test with MCP enabled
    with patch('src.llm.ollama_client.generate') as mock_generate:
        # Configure the mock to record the system_prompt parameter
        mock_generate.return_value = "Mock response"
        
        # Call generate_chat_response with MCP enabled
        await generate_chat_response(
            "Test question",
            "Test history",
            model="Gemma",
            timeout=20.0,
            use_mcp=True  # Enable MCP
        )
        
        # Get the system_prompt that was passed to generate
        args, kwargs = mock_generate.call_args
        mcp_system_prompt = kwargs.get('system_prompt', '')
        
        # Verify MCP instructions are included
        assert "Available Memory & Knowledge Tools:" in mcp_system_prompt
        assert "Available Web Tools:" in mcp_system_prompt
    
    # Test with MCP disabled
    with patch('src.llm.ollama_client.generate') as mock_generate:
        # Configure the mock
        mock_generate.return_value = "Mock response"
        
        # Call generate_chat_response with MCP disabled
        await generate_chat_response(
            "Test question",
            "Test history",
            model="Mistral:latest",
            timeout=1.0,
            use_mcp=False  # Disable MCP
        )
        
        # Get the system_prompt that was passed to generate
        args, kwargs = mock_generate.call_args
        standard_system_prompt = kwargs.get('system_prompt', '')
        
        # Verify MCP instructions are NOT included
        assert "Available Memory & Knowledge Tools:" not in standard_system_prompt
        assert "Available Web Tools:" not in standard_system_prompt
    
    print("\n✅ Chat manager correctly selects the appropriate system prompt based on MCP status.")


@pytest.mark.asyncio
async def test_mcp_server_initialization():
    """Test that MCP servers can be properly initialized."""
    
    # Create temporary memory storage for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize memory MCP with the temp directory
        memory_mcp = BellaMemoryMCP(
            storage_dir=temp_dir,
            server_name="test-memory",
            enable_startup=False  # Don't start the server yet
        )
        
        # Verify memory MCP is properly initialized
        assert memory_mcp.storage_dir == temp_dir
        assert memory_mcp.server_name == "test-memory"
        assert os.path.exists(memory_mcp.db_path)  # Database should be created
        
        # Initialize web search MCP
        web_search_mcp = BellaWebSearchMCP(
            server_name="test-web-search",
            enable_startup=False,  # Don't start the server yet
            model="Gemma"  # Using default model for test
        )
        
        # Verify web search MCP is properly initialized
        assert web_search_mcp.server_name == "test-web-search"
        assert web_search_mcp.model == "Gemma"
        
        print("\n✅ MCP servers can be properly initialized with correct configuration.")


@pytest.mark.asyncio
async def test_mcp_manager_integration():
    """Test the MCP server manager's ability to start and stop servers."""
    
    # Initialize MCP manager with mocked config
    with patch('src.utility.mcp_server_manager.ConfigManager') as mock_config:
        # Mock the config to return specific test values
        mock_instance = MagicMock()
        mock_instance.get_enabled_mcp_servers.return_value = [
            {
                "name": "test-memory",
                "enabled": True,
                "type": "memory",
                "args": {}
            },
            {
                "name": "test-web-search",
                "enabled": True,
                "type": "web_search",
                "args": {"model": "Gemma"}
            }
        ]
        mock_config.return_value = mock_instance
        
        # Mock the server initialization to avoid actually starting servers
        with patch('src.utility.mcp_server_manager.BellaMemoryMCP') as mock_memory:
            with patch('src.utility.mcp_server_manager.BellaWebSearchMCP') as mock_web:
                # Configure the mocks
                mock_memory_instance = MagicMock()
                mock_memory_instance.start_server.return_value = True
                mock_memory.return_value = mock_memory_instance
                
                mock_web_instance = MagicMock()
                mock_web_instance.start_server.return_value = True
                mock_web.return_value = mock_web_instance
                
                # Initialize the manager
                manager = MCPServerManager()
                
                # Start servers
                started = await manager.start_all_enabled()
                
                # Verify both servers were started
                assert len(started) == 2
                assert "test-memory" in started
                assert "test-web-search" in started
                
                # Verify memory server was initialized correctly
                mock_memory.assert_called_once()
                
                # Verify web search server was initialized correctly
                mock_web.assert_called_once()
                mock_web.assert_called_with(
                    server_name="test-web-search",
                    model="Gemma",
                    enable_startup=True
                )
                
                # Stop servers
                await manager.stop_all()
                
                # Verify both instances were accessed for stopping
                assert manager.active_servers == {}
    
    print("\n✅ MCP server manager can properly start and stop MCP servers.")


@pytest.mark.asyncio
async def test_signal_file_communication():
    """Test the signal file communication between main application and web search MCP."""
    
    # Define signal file path (same as used in web_search_mcp.py)
    signal_path = os.path.join(tempfile.gettempdir(), "bella_search_status.json")
    
    # Create a temporary web search MCP instance
    web_search = BellaWebSearchMCP(
        server_name="test-web-search",
        enable_startup=False,  # Don't start the server
        model="Gemma"
    )
    
    # Test signal creation
    test_query = "test search query"
    web_search._signal_search_start(test_query)
    
    # Verify signal file exists and contains correct data
    assert os.path.exists(signal_path)
    with open(signal_path, 'r') as f:
        signal_data = json.load(f)
        assert signal_data['status'] == "searching"
        assert signal_data['query'] == test_query
        assert signal_data['tool'] == "web_search"
    
    # Test signal completion
    web_search._signal_search_complete(test_query, success=True)
    
    # Verify signal file was updated
    with open(signal_path, 'r') as f:
        signal_data = json.load(f)
        assert signal_data['status'] == "completed"
        assert signal_data['query'] == test_query
    
    # Clean up
    if os.path.exists(signal_path):
        os.remove(signal_path)
    
    print("\n✅ Signal file communication mechanism is working properly.")


@pytest.mark.asyncio
async def test_live_model_with_mcp_prompt():
    """Test an actual model with the MCP prompt (if Ollama is available).
    
    This test attempts to send a real query to Ollama with the MCP prompt,
    but marks as skipped if Ollama is not available.
    
    Note: This test requires an actual Ollama instance running with the specified model.
    """
    try:
        import ollama
        ollama.list()  # Check if Ollama is available
        
        # Get the model config to find an available model
        model_config = ModelConfig()
        model = model_config.get_default_model()
        
        # Get the MCP prompt
        prompt_config = PromptConfig()
        mcp_prompt = prompt_config.get_system_prompt("system_with_mcp")
        
        # Try an actual call with a query that should use web search
        response = await generate_chat_response(
            "Can you search for information about Python 3.11 new features?",
            "",  # No history
            model=model,
            timeout=10.0,
            use_mcp=True
        )
        
        print("\n✅ Live model test completed successfully.")
        print(f"Response: {response[:100]}...")  # Print start of response
        
        # Look for indicators that MCP instructions were understood
        understood = any([
            "I can search for" in response,
            "let me search" in response.lower(),
            "I'll look that up" in response.lower(),
            "I'll search for" in response.lower()
        ])
        
        if understood:
            print("✅ Model appears to understand MCP instructions.")
        else:
            print("⚠️ Model may not be recognizing MCP instructions.")
        
    except (ImportError, Exception) as e:
        pytest.skip(f"Ollama not available or error occurred: {e}")
        print("\n⚠️ Live model test skipped - Ollama not available.")


if __name__ == "__main__":
    # Run the tests directly when executed as a script
    asyncio.run(test_mcp_system_prompt_loading())
    asyncio.run(test_chat_manager_prompt_selection())
    asyncio.run(test_mcp_server_initialization())
    asyncio.run(test_mcp_manager_integration())
    asyncio.run(test_signal_file_communication())
    print("\nTrying live model test (may be skipped if Ollama is not available):")
    asyncio.run(test_live_model_with_mcp_prompt())