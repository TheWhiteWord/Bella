"""Test script for Bella's memory system integration with Praison AI.

This test validates that the memory system can store and retrieve memories correctly,
testing both short-term and long-term memory functionality.
"""

import os
import sys
import asyncio
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)-8s %(message)s',
                    datefmt='%m/%d/%y %H:%M:%S')

# Add project root to path for imports
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

# Define results directory
RESULTS_DIR = project_root / "results" / "test_results" / "memory_tests"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from src.memory import MemoryManager, ShortTermMemory, LongTermMemory

# Check if necessary env vars are set
def check_environment():
    """Check if necessary environment variables are set."""
    required_vars = ["OPENAI_BASE_URL"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"⚠️ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file or environment")
        print(f"Current OPENAI_BASE_URL: {os.environ.get('OPENAI_BASE_URL', 'Not set')}")
        return False
    
    print(f"✅ Environment variables set. Using OPENAI_BASE_URL={os.environ.get('OPENAI_BASE_URL')}")
    return True

async def test_memory_storage_retrieval(
    memory_dir: Optional[str] = None,
    verbose: bool = False,
    embedding_model: str = "nomic-embed-text:latest"  # Using Ollama's embedding model
) -> bool:
    """Test memory storage and retrieval functionality.
    
    Args:
        memory_dir: Optional custom directory for memory storage
        verbose: Whether to print detailed logs
        embedding_model: Model to use for embeddings
        
    Returns:
        bool: Whether all tests passed
    """
    # Check environment variables
    if not check_environment():
        print("Warning: Environment variables are not properly set. Test may fail.")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = RESULTS_DIR / f"memory_test_{timestamp}.log"
    
    # Configure test logger
    test_logger = logging.getLogger("memory_test")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)-8s %(message)s'))
    test_logger.addHandler(file_handler)
    
    # Track test results
    test_results = {
        "short_term": {"store": False, "retrieve": False},
        "long_term": {"store": False, "retrieve": False},
        "semantic_search": False
    }
    all_tests_passed = True
    
    try:
        test_logger.info("Starting memory test")
        print("\n=== Testing Bella Memory System ===")
        
        # Initialize memory manager with Ollama-compatible settings
        custom_config = {
            "provider": "rag",  # Use RAG for semantic search
            "use_embedding": True,  # Enable embeddings
            "embedding_model": embedding_model  # Use Ollama's embedding model
        }
        
        if memory_dir:
            memory_dir_path = Path(memory_dir)
            custom_config.update({
                "short_db": str(memory_dir_path / "short_term.db"),
                "long_db": str(memory_dir_path / "long_term.db"),
                "rag_db_path": str(memory_dir_path / "chroma_db")
            })
            
        test_logger.info(f"Initializing memory manager with config: {custom_config}")
        memory_manager = MemoryManager(memory_config=custom_config)
        
        # Test 1: Store and retrieve from short-term memory
        print("\nTest 1: Short-term memory storage and retrieval...")
        test_content = f"Test memory item at {timestamp}"
        test_metadata = {"test_id": timestamp, "type": "short_term_test"}
        
        # Store in short-term memory
        store_success = await memory_manager.short_term.store(
            test_content,
            metadata=test_metadata
        )
        test_results["short_term"]["store"] = store_success
        if not store_success:
            test_logger.error("Failed to store in short-term memory")
            print("❌ Failed to store in short-term memory")
            all_tests_passed = False
        else:
            test_logger.info("Successfully stored in short-term memory")
            print("✅ Successfully stored in short-term memory")
        
        # Retrieve from short-term memory
        search_results = await memory_manager.short_term.search(
            query=test_content,
            limit=5
        )
        
        if search_results and any(item["text"] == test_content for item in search_results):
            test_results["short_term"]["retrieve"] = True
            test_logger.info("Successfully retrieved from short-term memory")
            print("✅ Successfully retrieved from short-term memory")
        else:
            test_logger.error(f"Failed to retrieve from short-term memory: {search_results}")
            print("❌ Failed to retrieve from short-term memory")
            all_tests_passed = False
            
        # Test 2: Store and retrieve from long-term memory
        print("\nTest 2: Long-term memory storage and retrieval...")
        test_content_lt = f"Important long-term memory from test at {timestamp}"
        test_metadata_lt = {"test_id": timestamp, "type": "long_term_test"}
        
        # Store in long-term memory
        store_success_lt = await memory_manager.long_term.store(
            test_content_lt,
            metadata=test_metadata_lt,
            importance=0.8
        )
        test_results["long_term"]["store"] = store_success_lt
        if not store_success_lt:
            test_logger.error("Failed to store in long-term memory")
            print("❌ Failed to store in long-term memory")
            all_tests_passed = False
        else:
            test_logger.info("Successfully stored in long-term memory")
            print("✅ Successfully stored in long-term memory")
        
        # Retrieve from long-term memory
        search_results_lt = await memory_manager.long_term.search(
            query=test_content_lt,
            limit=5
        )
        
        if search_results_lt and any(item["text"] == test_content_lt for item in search_results_lt):
            test_results["long_term"]["retrieve"] = True
            test_logger.info("Successfully retrieved from long-term memory")
            print("✅ Successfully retrieved from long-term memory")
        else:
            test_logger.error(f"Failed to retrieve from long-term memory: {search_results_lt}")
            print("❌ Failed to retrieve from long-term memory")
            all_tests_passed = False

        # Test 3: Test semantic search capability
        print("\nTest 3: Testing semantic search...")
        
        # Store some semantically related content
        semantic_contents = [
            "Python is a high-level programming language known for its readability.",
            "Java is an object-oriented language with strong enterprise support.",
            "JavaScript is widely used for web development in browsers.",
            "Artificial intelligence is revolutionizing multiple industries today."
        ]
        
        for i, content in enumerate(semantic_contents):
            await memory_manager.short_term.store(
                content,
                metadata={"content_type": "programming" if i < 3 else "ai"}
            )
        
        # Test semantic search with different queries
        semantic_query = "programming languages for software development"
        semantic_results = await memory_manager.get_context(semantic_query)
        
        if semantic_results and any("Python" in result for result in semantic_results.split("\n")):
            test_results["semantic_search"] = True
            test_logger.info("Semantic search retrieved relevant results")
            print("✅ Semantic search retrieved relevant results")
        else:
            test_logger.error(f"Semantic search failed to retrieve relevant results: {semantic_results}")
            print("❌ Semantic search failed or returned irrelevant results")
            all_tests_passed = False
        
        # Final results
        print("\n=== Memory Test Results ===")
        print(f"Short-term memory storage: {'✅' if test_results['short_term']['store'] else '❌'}")
        print(f"Short-term memory retrieval: {'✅' if test_results['short_term']['retrieve'] else '❌'}")
        print(f"Long-term memory storage: {'✅' if test_results['long_term']['store'] else '❌'}")
        print(f"Long-term memory retrieval: {'✅' if test_results['long_term']['retrieve'] else '❌'}")
        print(f"Semantic search: {'✅' if test_results['semantic_search'] else '❌'}")
        
        # Log results
        test_logger.info(f"Test results: {test_results}")
        
        return all_tests_passed
        
    except Exception as e:
        test_logger.error(f"Error during memory test: {e}")
        print(f"\nError during memory test: {e}")
        return False
    finally:
        print(f"\nTest log saved to: {log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Bella's memory system")
    parser.add_argument(
        "--memory-dir",
        type=str,
        default=None,
        help="Custom directory for memory storage"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="nomic-embed-text:latest",
        help="Embedding model to use (default: nomic-embed-text:latest)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Run memory tests
    result = asyncio.run(test_memory_storage_retrieval(
        args.memory_dir, 
        args.verbose,
        args.embedding_model
    ))
    sys.exit(0 if result else 1)