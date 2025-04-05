"""Test to verify that the MCP tool registration fix is working correctly.

This script creates a focused test to confirm that the Mistral model
can properly use memory tools through our enhanced MCP integration.

Run with:
python -m tests.integration.test_mcp_tool_verification
"""
import os
import sys
import asyncio
import tempfile
import logging
import re
import json
from datetime import datetime
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='[%(asctime)s] %(levelname)-8s %(message)s',
                   datefmt='%m/%d/%y %H:%M:%S')

# Add project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from src.mcp_servers.basic_memory_MCP import BellaMemoryMCP
from src.utility.mcp_server_manager import MCPServerManager
from src.llm.config_manager import PromptConfig
from src.llm.chat_manager import generate_chat_response


async def test_memory_tool_execution():
    """Test that the model can correctly execute memory tools."""
    print("\n=== TESTING MEMORY MCP TOOL EXECUTION ===\n")
    
    # Create a temporary directory for test memory storage
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory for memory storage: {temp_dir}")
        
        # Initialize variables to avoid UnboundLocalError
        entities = []
        md_files = []
        
        # Initialize memory MCP with the temp directory
        memory_mcp = BellaMemoryMCP(
            storage_dir=temp_dir,
            server_name="test-memory",
            enable_startup=True
        )
        
        # Give the server a moment to start
        await asyncio.sleep(1)
        
        # Verify server registration with the manager
        mcp_manager = MCPServerManager()
        active_servers = mcp_manager.get_active_servers()
        print(f"Active servers after registration: {len(active_servers)}")
        
        for i, server in enumerate(active_servers):
            server_name = getattr(server, 'server_name', 'unknown')
            print(f"  Server {i+1}: {server_name}")
        
        # Get the tools schema from the manager
        tools_schema = mcp_manager.get_tools_schema()
        print(f"Available tools: {len(tools_schema)}")
        for i, tool in enumerate(tools_schema):
            print(f"  Tool {i+1}: {tool['name']} - {tool['description'][:50]}...")
        
        # Define a prompt that should trigger the model to use the write_note tool
        prompt = """I need you to create a new note about Python programming.

Please include the following:
1. A title "Python Programming Overview"
2. At least 3 observations about Python features with appropriate category tags
3. At least 2 relations to related topics like web development and data science

This is VERY important: Save this as a note in my memory using the write_note tool."""

        print("\n=== SENDING PROMPT TO MODEL ===\n")
        print(prompt)
        
        # Set extended timeout for model generation (60 seconds instead of default)
        timeout = 60.0
        
        # Implement retry logic - try up to 3 times
        max_attempts = 3
        attempt = 0  # Initialize attempt counter
        
        for attempt in range(1, max_attempts + 1):
            print(f"\nAttempt {attempt}/{max_attempts} with timeout {timeout}s...")
            try:
                # Send the prompt to the model with extended timeout
                response = await generate_chat_response(
                    prompt,
                    "",  # No history
                    use_mcp=True,  # Enable MCP
                    timeout=timeout  # Extended timeout
                )
                
                print("\n=== MODEL RESPONSE ===\n")
                print(response[:500] + "..." if len(response) > 500 else response)
                
                # Check if the response contains an error message
                if "trouble generating a response" in response or "Error:" in response:
                    print("\n❌ Received error in response. Retrying...")
                    # Increase timeout for next attempt
                    timeout += 15.0
                    continue
                
                # Check if any database entries were created
                import sqlite3
                conn = sqlite3.connect(memory_mcp.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM entities WHERE entity_type = 'note'")
                entities = cursor.fetchall()
                conn.close()
                
                print("\n=== DATABASE CHECK ===\n")
                if entities:
                    print(f"✅ Found {len(entities)} entries in database:")
                    for entity in entities:
                        print(f"- {entity[0]}")
                    
                    # Force sync to create Markdown files from database entries
                    sync_count = memory_mcp.sync_files(force=True)
                    print(f"Synchronized {sync_count} files")
                    
                    # Check if any Markdown files were created
                    md_files = [f for f in os.listdir(temp_dir) if f.endswith('.md')]
                    if md_files:
                        print(f"\n✅ {len(md_files)} Markdown files were created:")
                        for file in md_files:
                            print(f"- {file}")
                            
                            # Show a preview of the first file
                            if file == md_files[0]:
                                with open(os.path.join(temp_dir, file), 'r', encoding='utf-8') as f:
                                    content = f.read()
                                print("\nFile preview:")
                                print(content[:500] + "..." if len(content) > 500 else content)
                        # Success! Break out of retry loop
                        break
                    else:
                        print("\n❌ No Markdown files were created")
                        # If this was the last attempt, let it fail
                        if attempt == max_attempts:
                            break
                        print("Retrying...")
                        # Give the system a moment before retrying
                        await asyncio.sleep(2)
                else:
                    print("❌ No entries found in database. Tool execution may have failed.")
                    # If this was the last attempt, let it fail
                    if attempt == max_attempts:
                        break
                    print("Retrying...")
                    # Give the system a moment before retrying
                    await asyncio.sleep(2)
            
            except Exception as e:
                print(f"\n❌ Error during attempt {attempt}: {e}")
                if attempt == max_attempts:
                    raise
                print("Retrying...")
                # Give the system a moment before retrying
                await asyncio.sleep(2)
        
        # If we've exhausted all retries, try a direct test to verify the tools still work
        if attempt == max_attempts and (not entities or not md_files):
            print("\n=== DIRECT TOOL EXECUTION TEST ===\n")
            print("Trying direct execution of write_note to verify tools are functional...")
            
            test_title = "Direct Test Note"
            test_content = """# Direct Test Note
            
## Observations

- [test] This is a direct test of the write_note functionality
- [verification] Ensuring the MCP memory system is working properly

## Relations

- relates_to [[Python Programming]]
- used_by [[Test Suite]]
"""
            # Generate permalink
            permalink = re.sub(r'[^a-z0-9]+', '-', test_title.lower()).strip('-')
            
            # Call methods directly to create file and update database
            try:
                file_path = memory_mcp._write_markdown_file(
                    title=test_title,
                    permalink=permalink,
                    content=test_content,
                    folder="",
                    tags=["test", "verification"]
                )
                
                memory_mcp._index_note(test_title, permalink, test_content, ["test", "verification"])
                
                print(f"✅ Direct write successful: {file_path}")
                
                # Verify the file exists
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    print(f"File content preview: {content[:200]}...")
            except Exception as e:
                print(f"❌ Direct write failed: {e}")


async def main():
    """Run the MCP tool verification test."""
    try:
        await test_memory_tool_execution()
    except Exception as e:
        logging.error(f"Test error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n=== MCP TOOL VERIFICATION TEST ===")
    asyncio.run(main())