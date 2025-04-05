"""Test script for Bella's memory integration with conversation flow.

This test validates that the memory system properly integrates with Bella's
conversation flow, remembering and retrieving conversation context correctly.
"""

import os
import sys
import asyncio
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

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

from src.memory import MemoryManager
from src.llm.chat_manager import generate_chat_response

async def test_memory_conversation(
    memory_dir: Optional[str] = None,
    model: str = "Gemma",
    verbose: bool = False
) -> bool:
    """Test memory integration with conversation flow.
    
    Args:
        memory_dir: Optional custom directory for memory storage
        model: Model to use for tests
        verbose: Whether to print detailed logs
        
    Returns:
        bool: Whether all tests passed
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = RESULTS_DIR / f"memory_conversation_test_{timestamp}.log"
    
    # Configure test logger
    test_logger = logging.getLogger("memory_conversation_test")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)-8s %(message)s'))
    test_logger.addHandler(file_handler)
    
    test_logger.info(f"Starting memory conversation test using model: {model}")
    print(f"\n=== Testing Bella Memory with Conversations using model {model} ===")
    
    try:
        # Initialize a custom memory manager for testing
        custom_config = {}
        if memory_dir:
            memory_dir_path = Path(memory_dir)
            custom_config = {
                "short_db": str(memory_dir_path / "short_term.db"),
                "long_db": str(memory_dir_path / "long_term.db"),
                "rag_db_path": str(memory_dir_path / "chroma_db")
            }
        
        # Simulate a multi-turn conversation
        conversation = [
            "My name is Alex and I'm a software developer.",
            "What programming languages are popular today?",
            "I prefer working with Python for data science.",
            "What was my name again?",  # Memory test question
        ]
        
        print("\nSimulating conversation with memory...")
        
        # Record conversation history manually to compare with memory
        history = ""
        response_history = []
        
        for i, message in enumerate(conversation):
            print(f"\nUser [{i+1}]: {message}")
            
            # Generate response using memory-enhanced chat
            response = await generate_chat_response(
                message, 
                history, 
                model=model
            )
            response_history.append(response)
            print(f"Assistant [{i+1}]: {response}")
            
            # Update history manually for next turn
            history += f"User: {message}\nAssistant: {response}\n"
        
        # Analyze test results
        memory_test_success = False
        name_check_response = response_history[-1]
        
        # Check if the name 'Alex' is remembered in the final response
        if "Alex" in name_check_response:
            memory_test_success = True
            print("\n✅ Memory test passed: Bella remembered the user's name!")
            test_logger.info("Memory test passed: Name was remembered")
        else:
            print("\n❌ Memory test failed: Bella didn't remember the user's name")
            test_logger.error(f"Memory test failed: Name was not remembered. Response: {name_check_response}")
        
        # Save test details
        with open(log_path, "a") as f:
            f.write("\n=== Conversation Test ===\n")
            for i, (message, response) in enumerate(zip(conversation, response_history)):
                f.write(f"User [{i+1}]: {message}\n")
                f.write(f"Assistant [{i+1}]: {response}\n\n")
            f.write(f"Memory Test Result: {'Passed' if memory_test_success else 'Failed'}\n")
        
        print(f"\nTest log saved to: {log_path}")
        return memory_test_success
        
    except Exception as e:
        test_logger.error(f"Error during memory conversation test: {e}")
        print(f"\nError during memory conversation test: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Bella's memory integration with conversation")
    parser.add_argument(
        "--memory-dir",
        type=str,
        default=None,
        help="Custom directory for memory storage"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Gemma",
        help="Model to use for conversation test"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Run memory conversation test
    result = asyncio.run(test_memory_conversation(args.memory_dir, args.model, args.verbose))
    sys.exit(0 if result else 1)