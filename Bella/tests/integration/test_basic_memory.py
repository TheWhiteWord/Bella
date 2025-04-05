"""Test for basic memory MCP integration with Bella.

This script provides a simple test to verify that the
basic memory MCP integration is working correctly with the
PraisonAI framework using npx directly.
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.mcp_servers.memory_agent import MemoryAgent

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(name)s - %(levelname)s: %(message)s",
)

async def test_memory_integration():
    """Run a basic test of the memory MCP integration."""
    print("\n" + "=" * 80)
    print(" TESTING BASIC MEMORY MCP INTEGRATION")
    print("=" * 80)
    
    # Create a unique identifier for this test run
    timestamp = int(time.time())
    test_title = f"Memory Test {timestamp}"
    test_content = f"""This is a test note created at {timestamp}.
    
    ## Key Points
    
    - Testing basic memory MCP integration
    - Using PraisonAI agents framework
    - Running with npx for simplicity
    - Timestamp: {timestamp}
    """
    test_tags = ["test", "memory", "bella", f"timestamp-{timestamp}"]
    
    try:
        # Initialize the memory agent
        print("\nInitializing memory agent...")
        memory = MemoryAgent(verbose=True)
        
        # Create test note
        print(f"\nCreating test note: {test_title}")
        create_result = await memory.create_note(test_title, test_content, test_tags)
        print("\nCreate note result:")
        print(create_result)
        
        # Allow some time for processing
        print("\nWaiting for note to be processed...")
        await asyncio.sleep(3)
        
        # Search for the test note
        print(f"\nSearching for notes with timestamp {timestamp}")
        search_result = await memory.search_notes(f"timestamp-{timestamp}")
        print("\nSearch result:")
        print(search_result)
        
        # Read the test note
        print(f"\nReading test note: {test_title}")
        read_result = await memory.read_note(test_title)
        print("\nRead result:")
        print(read_result)
        
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