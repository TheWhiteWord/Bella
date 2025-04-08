#!/usr/bin/env python3
import asyncio
import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add Bella directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_memory():
    # Import memory components
    from Bella.src.memory.enhanced_memory_adapter import memory_adapter
    
    # Initialize memory adapter
    print("Initializing memory adapter...")
    await memory_adapter.initialize()
    
    # Test memory storage
    test_content = "This is a test memory to diagnose persistence issues."
    memory_type = "general"
    note_name = "memory-persistence-test"
    
    print(f"Storing memory: {memory_type}/{note_name}")
    success, path = await memory_adapter.store_memory(memory_type, test_content, note_name)
    
    print(f"Memory storage result: success={success}, path={path}")
    if path:
        print(f"File exists: {os.path.exists(path)}")
    
    # Check in ChromaDB
    if memory_adapter.chroma_collection:
        count = memory_adapter.chroma_collection.count()
        print(f"ChromaDB collection has {count} items")
        
        # Search for the memory
        print("Searching for the memory...")
        search_results, search_success = await memory_adapter.search_memory("test memory persistence")
        print(f"Search success: {search_success}")
        print(f"Found {len(search_results.get('results', []))} results")
    else:
        print("ChromaDB collection not initialized")

if __name__ == "__main__":
    asyncio.run(test_memory())
