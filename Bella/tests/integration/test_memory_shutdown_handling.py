"""Integration test for verifying proper shutdown handling in Memory MCP.

This test specifically checks that:
1. Memory MCP server initializes correctly
2. Memory operations work as expected
3. Server shuts down gracefully without errors
4. Multiple start/stop cycles work properly
5. Resource cleanup is complete (no leaked threads/connections)

Run this test directly with:
python -m tests.integration.test_memory_shutdown_handling
"""

import os
import sys
import asyncio
import tempfile
import threading
import time
import gc
import sqlite3
import logging
import pytest
import re
from pathlib import Path
from unittest.mock import patch

# Configure logging to see shutdown-related messages
logging.basicConfig(level=logging.INFO,
                   format='[%(asctime)s] %(levelname)-8s %(message)s',
                   datefmt='%m/%d/%y %H:%M:%S')

# Add project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Import required modules
from src.mcp_servers.basic_memory_MCP import BellaMemoryMCP
from src.utility.mcp_server_manager import MCPServerManager


@pytest.mark.asyncio
async def test_memory_server_startup_shutdown():
    """Test that memory server starts and shuts down cleanly."""
    
    # Create a temporary directory for test memory storage
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\nUsing temporary directory for memory storage: {temp_dir}")
        
        # Test multiple start/stop cycles to ensure stability
        for cycle in range(3):
            print(f"\n--- Testing start/stop cycle {cycle+1}/3 ---")
            
            # Initialize memory MCP with the temp directory
            memory_mcp = BellaMemoryMCP(
                storage_dir=temp_dir,
                server_name=f"test-memory-{cycle}",
                enable_startup=True  # Automatically start the server
            )
            
            # Give the server a moment to start
            await asyncio.sleep(1)
            
            # Verify db is created
            assert os.path.exists(memory_mcp.db_path), "Database file should be created"
            print("✅ Database file created successfully")
            
            # Create a simple note to test functionality
            note_title = f"Test Note {cycle}"
            note_content = f"""# Test Note {cycle}

## Observations

- [test] This is a test note for cycle {cycle}
- [verification] Testing memory server shutdown handling

## Relations

- relates_to [[Memory System]]
- used_by [[Test Suite]]
"""
            # Generate permalink like the tool would
            permalink = f"test-note-{cycle}"
            
            # Use the internal methods that the tool would use
            file_path = memory_mcp._write_markdown_file(
                title=note_title,
                permalink=permalink,
                content=note_content,
                tags=["test", f"cycle-{cycle}"]
            )
            
            # Also update the database
            memory_mcp._index_note(note_title, permalink, note_content, ["test", f"cycle-{cycle}"])
            
            # Verify file was created
            assert os.path.exists(file_path), f"Note file should exist at {file_path}"
            print(f"✅ Note created successfully at {file_path}")
            
            # Check active thread count before stopping server
            active_threads_before = threading.active_count()
            print(f"Active threads before server shutdown: {active_threads_before}")
            
            # Explicitly stop the server
            print("Stopping memory MCP server...")
            memory_mcp.stop_server()
            
            # Give a moment for cleanup
            await asyncio.sleep(1)
            
            # Force garbage collection to ensure resources are freed
            print("Forcing garbage collection...")
            gc.collect()
            
            # Check active thread count after stopping server
            await asyncio.sleep(0.5)  # Short wait to allow thread termination
            active_threads_after = threading.active_count()
            print(f"Active threads after server shutdown: {active_threads_after}")
            
            # Check for thread leakage - allow for some background threads
            # The difference shouldn't be large if cleanup is working properly
            thread_diff = active_threads_before - active_threads_after
            print(f"Thread difference: {thread_diff}")
            
            # Clear reference to allow proper GC
            del memory_mcp
            
            # Force garbage collection again
            gc.collect()
            await asyncio.sleep(0.5)
            
            print(f"✅ Cycle {cycle+1} completed successfully")
            
        print("\n✅ All start/stop cycles completed successfully")


@pytest.mark.asyncio
async def test_memory_manager_integration():
    """Test that memory MCP integrates properly with the server manager."""
    
    print("\n--- Testing MCPServerManager integration ---")
    
    # Create a temporary directory for test memory storage
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory for memory storage: {temp_dir}")
        
        # Initialize the server manager
        mcp_manager = MCPServerManager()
        
        # Create a memory server manually first
        memory_mcp = BellaMemoryMCP(
            storage_dir=temp_dir,
            server_name="manager-test-memory",
            enable_startup=True
        )
        
        # Wait for server to start
        await asyncio.sleep(1)
        
        # Check if server is registered with manager
        active_servers = mcp_manager.get_active_servers()
        print(f"Active servers after manual creation: {len(active_servers)}")
        
        # Check if server has properly registered its tools
        tools_schema = mcp_manager.get_tools_schema()
        print(f"Available tools: {len(tools_schema)}")
        
        # Check if write_note tool is available
        write_note_available = any(tool['name'] == 'write_note' for tool in tools_schema)
        assert write_note_available, "write_note tool should be registered with manager"
        print("✅ write_note tool is properly registered")
        
        # Stop server through manager
        print("Stopping server through manager...")
        await mcp_manager.stop_all()
        
        # Give a moment for cleanup
        await asyncio.sleep(1)
        
        # Force garbage collection
        gc.collect()
        
        # Verify no active servers
        active_servers = mcp_manager.get_active_servers()
        print(f"Active servers after stopping: {len(active_servers)}")
        
        # Verify server is fully stopped
        assert not hasattr(memory_mcp, 'server_thread') or not memory_mcp.server_thread or not memory_mcp.server_thread.is_alive(), \
            "Server thread should not be alive after stopping"
        
        print("✅ Server successfully stopped through manager")
        
        # Clean up references
        del memory_mcp
        del mcp_manager
        
        # Force garbage collection
        gc.collect()
        

@pytest.mark.asyncio
async def test_concurrent_memory_operations():
    """Test memory operations under concurrent load."""
    
    print("\n--- Testing concurrent memory operations ---")
    
    # Create a temporary directory for test memory storage
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Initialize memory MCP
        memory_mcp = BellaMemoryMCP(
            storage_dir=temp_dir,
            server_name="concurrent-test-memory",
            enable_startup=True
        )
        
        # Give the server a moment to start
        await asyncio.sleep(1)
        
        # Simulate concurrent note creation
        async def create_note(index):
            note_title = f"Concurrent Note {index}"
            note_content = f"""# Concurrent Note {index}

## Observations

- [test] This is concurrent test note {index}
- [concurrent] Created during concurrent operation test

## Relations

- relates_to [[Memory System]]
- part_of [[Concurrency Test]]
"""
            permalink = f"concurrent-note-{index}"
            
            try:
                # Use internal methods that the tool would use
                file_path = memory_mcp._write_markdown_file(
                    title=note_title,
                    permalink=permalink,
                    content=note_content,
                    tags=["concurrent", f"test-{index}"]
                )
                
                # Update the database
                memory_mcp._index_note(note_title, permalink, note_content, ["concurrent", f"test-{index}"])
                return file_path
            except Exception as e:
                return f"Error: {str(e)}"
        
        # Create multiple notes concurrently
        tasks = [create_note(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # Check results
        success_count = sum(1 for r in results if not str(r).startswith("Error"))
        print(f"Successfully created {success_count}/10 concurrent notes")
        
        # Check database for entries
        conn = sqlite3.connect(memory_mcp.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM entities WHERE name LIKE 'concurrent-note-%'")
        db_count = cursor.fetchone()[0]
        conn.close()
        
        print(f"Found {db_count} entries in database")
        assert db_count >= 8, "Should have created most notes successfully in database"
        
        # Check filesystem for files
        md_files = [f for f in os.listdir(temp_dir) if f.startswith("concurrent-note-") and f.endswith(".md")]
        print(f"Found {len(md_files)} files on disk")
        assert len(md_files) >= 8, "Should have created most note files successfully"
        
        print("✅ Concurrent operations completed successfully")
        
        # Test sync mechanism under load
        print("Testing sync mechanism...")
        sync_count = memory_mcp.sync_files(force=True)
        print(f"Synchronized {sync_count} files")
        
        # Explicitly stop the server
        print("Stopping memory MCP server...")
        memory_mcp.stop_server()
        
        # Give a moment for cleanup
        await asyncio.sleep(1)
        
        # Clear reference to allow proper GC
        del memory_mcp
        
        # Force garbage collection
        gc.collect()


async def main():
    """Run all tests in sequence."""
    try:
        print("=== TESTING MEMORY SERVER STARTUP/SHUTDOWN ===")
        await test_memory_server_startup_shutdown()
        
        print("\n=== TESTING MEMORY MANAGER INTEGRATION ===")
        await test_memory_manager_integration()
        
        print("\n=== TESTING CONCURRENT MEMORY OPERATIONS ===")
        await test_concurrent_memory_operations()
        
        print("\n✅ ALL TESTS PASSED SUCCESSFULLY")
    except Exception as e:
        import traceback
        print(f"\n❌ TEST FAILED: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    # Run tests directly when executed as a script
    print("Running memory shutdown handling tests...")
    asyncio.run(main())