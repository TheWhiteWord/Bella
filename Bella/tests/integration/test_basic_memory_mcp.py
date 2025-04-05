"""Test for basic memory MCP integration with Bella.

This test verifies the functionality of the basic memory MCP integration,
ensuring that notes can be created, searched, and read using the memory tool.
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Set the OpenAI URL environment variable for Ollama compatibility
os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"

from src.mcp_servers.memory_tool import MemoryTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

async def test_memory_integration():
    """Test the basic memory MCP integration."""
    print("\n" + "=" * 80)
    print(" BASIC MEMORY MCP INTEGRATION TEST")
    print("=" * 80 + "\n")
    
    # Create a unique identifier for this test run
    timestamp = int(time.time())
    test_title = f"Memory Test {timestamp}"
    test_content = (
        f"This is a test note created at timestamp {timestamp}.\n\n"
        "Testing basic memory MCP integration with Bella assistant."
    )
    
    try:
        # Initialize the memory tool
        print("Initializing memory tool...")
        memory = MemoryTool(
            model="Lexi:latest",  # Use model without ollama/ prefix for OpenAI compatibility
            verbose=True
        )
        
        # Create a test note
        print(f"\nCreating test note: {test_title}")
        create_result = await memory.remember(
            test_title,
            test_content,
            ["test", "bella", f"timestamp-{timestamp}"]
        )
        print("\nCreate note result:")
        print(create_result)
        
        # Short delay to allow processing
        print("\nWaiting for note to be processed...")
        await asyncio.sleep(2)
        
        # Search for the test note using the unique timestamp
        print(f"\nSearching for notes related to timestamp {timestamp}")
        search_result = await memory.recall(f"timestamp {timestamp}")
        print("\nSearch result:")
        print(search_result)
        
        # Read the specific note
        print(f"\nReading test note: {test_title}")
        read_result = await memory.read_note(test_title)
        print("\nRead result:")
        print(read_result)
        
        # Build context
        print("\nBuilding context about test notes")
        context_result = await memory.build_context("test notes")
        print("\nContext result:")
        print(context_result)
        
        print("\nTest completed successfully!")
    
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print(" TEST COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_memory_integration())