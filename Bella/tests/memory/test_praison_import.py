"""Test script to verify Praison installation and functionality.

This script tests whether Praison is installed correctly and can be imported,
and tests basic memory functionality.
"""

import os
import sys
import logging
import traceback
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)-8s %(message)s',
                    datefmt='%m/%d/%y %H:%M:%S')

# Load environment variables from .env file
load_dotenv()

# Check if OPENAI_BASE_URL is set (for Ollama)
if "OPENAI_BASE_URL" in os.environ:
    print(f"✅ OPENAI_BASE_URL is set to: {os.environ['OPENAI_BASE_URL']}")
else:
    print("❌ OPENAI_BASE_URL is not set")
    
# Set a dummy API key if using Ollama and no key is set
if "OPENAI_BASE_URL" in os.environ and "OPENAI_API_KEY" not in os.environ:
    print("Setting dummy API key for Ollama")
    os.environ["OPENAI_API_KEY"] = "ollama-dummy-key"
    
def test_praison_import():
    """Test if praisonaiagents can be imported correctly."""
    print("\n=== Testing Praison Import ===")
    
    try:
        import praisonaiagents
        print(f"✅ Praison is installed.")
        
        # Explore the module
        print("\nExploring Praison module structure:")
        print(f"Module path: {praisonaiagents.__file__}")
        print(f"Module dir: {dir(praisonaiagents)}")
        
        # Test knowledge module
        print("\nTesting knowledge module import:")
        try:
            from praisonaiagents.knowledge import Knowledge
            print("✅ Praison Knowledge module was imported successfully")
            print(f"Knowledge module structure: {dir(praisonaiagents.knowledge)}")
            print(f"Knowledge class attributes: {dir(Knowledge)}")
        except ImportError as e:
            print(f"❌ Error importing Knowledge module: {e}")
            
            # Try alternative imports based on package structure
            try:
                if hasattr(praisonaiagents, 'Knowledge'):
                    print("✅ Found Knowledge directly in praisonaiagents")
                    Knowledge = praisonaiagents.Knowledge
                    print(f"Knowledge class attributes: {dir(Knowledge)}")
            except Exception as e2:
                print(f"❌ Alternative import also failed: {e2}")
        
        # More detailed package info
        print(f"\nPython version: {sys.version}")
        
        return True
    except ImportError as e:
        print(f"❌ Error importing Praison: {e}")
        print("\nPlease install Praison with: pip install praisonaiagents[memory]")
        return False
    except Exception as e:
        print(f"❌ Unexpected error with Praison: {e}")
        return False

def test_custom_memory_implementation():
    """Test our custom memory implementation that wraps Praison's Knowledge."""
    print("\n=== Testing Custom Memory Implementation ===")
    
    try:
        # Import our own memory manager
        print("Importing memory manager...")
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "src"))
        from memory.memory_manager import MemoryManager
        
        # Create memory manager
        print("Creating memory manager...")
        memory = MemoryManager(enable_memory=True)
        
        if memory.praison_memory is None:
            print("❌ Memory manager failed to initialize Praison memory")
            return False
        
        # Test storing a memory
        print("\nTesting short-term memory storage...")
        import asyncio
        
        # Create event loop
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Run storage test
        result = asyncio.run(memory.short_term.store(
            "This is a test memory item", 
            metadata={"test": True}
        ))
        
        print(f"Memory storage result: {result}")
        
        # Test searching memory
        print("\nTesting memory search...")
        search_results = asyncio.run(memory.short_term.search("test memory"))
        print(f"Search results: {search_results}")
        
        # Test clearing memory
        print("\nTesting memory cleanup...")
        clear_result = asyncio.run(memory.short_term.clear())
        print(f"Memory cleanup result: {clear_result}")
        
        return True
    except Exception as e:
        print(f"❌ Custom memory implementation test error: {e}")
        traceback.print_exc()
        return False

def test_basic_memory():
    """Test basic memory functionality with minimal configuration."""
    print("\n=== Testing Basic Memory Functionality ===")
    
    try:
        # Try different import patterns
        try:
            from praisonaiagents.knowledge import Knowledge
            print("Using praisonaiagents.knowledge.Knowledge")
        except ImportError:
            try:
                from praisonaiagents import Knowledge
                print("Using praisonaiagents.Knowledge")
            except ImportError:
                print("❌ Could not import Knowledge class from any location")
                return False
        
        # Use the simplest possible configuration to test basic functionality
        memory_config = {
            "vector_store": {
                "provider": "dict",  # Use in-memory storage for testing
                "config": {}
            }
        }
        
        print(f"Creating Knowledge with minimal config: {memory_config}")
        
        # Print API key status (without revealing the key)
        if "OPENAI_API_KEY" in os.environ:
            key_preview = os.environ["OPENAI_API_KEY"][:4] + "..." if len(os.environ["OPENAI_API_KEY"]) > 4 else "*****"
            print(f"Using OPENAI_API_KEY: {key_preview}")
        else:
            print("⚠️ No OPENAI_API_KEY set")
            
        try:
            knowledge = Knowledge(memory_config)
            print("✅ Knowledge initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing Knowledge: {e}")
            traceback.print_exc()
            return False
        
        # Test memory operations
        print("\nTesting memory operations...")
        
        # Store a test memory - simple string without embeddings
        try:
            print("Storing test memory...")
            test_memory = knowledge.store("This is a test memory item", user_id="test_user", metadata={"test": True})
            print(f"✅ Stored memory: {test_memory}")
        except Exception as e:
            print(f"❌ Error storing memory: {e}")
            traceback.print_exc()
            return False
        
        # Test searching memory - only try if storing succeeded
        if test_memory:
            try:
                print("\nSearching memory...")
                search_results = knowledge.search("test memory", user_id="test_user")
                print(f"✅ Search results: {search_results}")
            except Exception as e:
                print(f"❌ Error searching memory: {e}")
                traceback.print_exc()
        else:
            print("\nSkipping search test because storing failed")
            
        # Clean up test data
        try:
            print("\nCleaning up test memory...")
            knowledge.delete_all(user_id="test_user")
            print("✅ Cleaned up test memory")
        except Exception as e:
            print(f"❌ Error cleaning memory: {e}")
            traceback.print_exc()
            
        return True
            
    except Exception as e:
        print(f"❌ Memory test error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\nStarting Praison test script...")
    import_success = test_praison_import()
    
    if import_success:
        # First test basic functionality with minimal configuration
        memory_success = test_basic_memory()
        
        # Then test our custom implementation
        custom_success = test_custom_memory_implementation()
        
        if memory_success and custom_success:
            print("\n✅ All Praison tests passed!")
        elif memory_success:
            print("\n⚠️ Basic memory tests passed but custom implementation tests failed")
        elif custom_success:
            print("\n⚠️ Custom implementation tests passed but basic memory tests failed")
        else:
            print("\n❌ All memory tests failed")
    else:
        print("\n❌ Praison import test failed")