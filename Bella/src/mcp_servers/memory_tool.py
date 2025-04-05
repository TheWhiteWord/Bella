"""Memory tool integration for Bella assistant.

This module provides a high-level interface for using the basic-memory MCP 
capabilities with the Bella assistant.
"""

import asyncio
import logging
import os
import yaml
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path

from .memory_agent import MemoryAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

class MemoryTool:
    """High-level interface for using Basic Memory with Bella assistant.
    
    This class provides methods for common memory operations to be used 
    with the Bella assistant.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        model: str = "Lexi:latest",
        memory_dir: str = os.path.expanduser("~/basic-memory"),
        verbose: bool = False
    ):
        """Initialize the memory tool.
        
        Args:
            config_path: Path to configuration file (optional)
            model: Model to use for memory operations (without ollama/ prefix when using OpenAI compatibility)
            memory_dir: Directory to store memory files
            verbose: Whether to output debug information
        """
        self.config = self._load_config(config_path)
        self.model = self.config.get("model", model)
        
        # If using OpenAI compatibility, remove the ollama/ prefix if it exists
        if os.environ.get("OPENAI_BASE_URL") and self.model.startswith("ollama/"):
            self.model = self.model.split("/", 1)[1]
            
        self.memory_dir = self.config.get("memory_dir", memory_dir)
        self.verbose = self.config.get("verbose", verbose)
        
        # Initialize memory agent with configured settings
        self.memory_agent = MemoryAgent(
            model=self.model,
            memory_dir=self.memory_dir,
            verbose=self.verbose
        )
        
        logger.info(f"Memory tool initialized with model: {self.model}")
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file if available.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        default_config = {
            "model": "Lexi:latest",
            "memory_dir": os.path.expanduser("~/basic-memory"),
            "verbose": False
        }
        
        if not config_path or not os.path.exists(config_path):
            # Try to find config in default location
            default_locations = [
                os.path.join(os.getcwd(), "src", "config", "mcp_servers.yaml"),
                os.path.join(os.getcwd(), "config", "mcp_servers.yaml"),
            ]
            
            for loc in default_locations:
                if os.path.exists(loc):
                    config_path = loc
                    break
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                    
                # Extract memory specific config if it exists
                if "servers" in config and "memory" in config["servers"]:
                    memory_config = config["servers"]["memory"]
                    if "params" in memory_config:
                        return {**default_config, **memory_config["params"]}
                    return {**default_config, **memory_config}
                return default_config
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {str(e)}")
                
        return default_config
    
    async def remember(self, title: str, content: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create a memory note with the given title and content.
        
        Args:
            title: Title for the memory note
            content: Content to store
            tags: Optional tags to categorize the memory
            
        Returns:
            Dict[str, Any]: Agent's response
        """
        if not tags:
            tags = ["bella", "memory"]
            
        try:
            query = f"Create a note titled '{title}' with the following content:\n\n{content}\n\nAdd these tags: {', '.join(tags)}"
            return await self.memory_agent.process_query(query)
        except Exception as e:
            logger.error(f"Error creating memory note: {str(e)}")
            return {"error": str(e), "success": False}
    
    async def recall(self, query: str) -> Dict[str, Any]:
        """Search memory for information related to a query.
        
        Args:
            query: The search query
            
        Returns:
            Dict[str, Any]: Agent's response with search results
        """
        try:
            search_query = f"Search my notes for information about: {query}"
            return await self.memory_agent.process_query(search_query)
        except Exception as e:
            logger.error(f"Error searching memory: {str(e)}")
            return {"error": str(e), "success": False}
    
    async def read_note(self, note_id: str) -> Dict[str, Any]:
        """Read a specific note by title or permalink.
        
        Args:
            note_id: Title or permalink of the note to read
            
        Returns:
            Dict[str, Any]: Agent's response with note content
        """
        try:
            read_query = f"Read the note titled '{note_id}' or with permalink '{note_id}'"
            return await self.memory_agent.process_query(read_query)
        except Exception as e:
            logger.error(f"Error reading note: {str(e)}")
            return {"error": str(e), "success": False}
    
    async def build_context(self, topic: str) -> Dict[str, Any]:
        """Build contextual knowledge about a specific topic.
        
        Args:
            topic: The topic to build context around
            
        Returns:
            Dict[str, Any]: Agent's response with context information
        """
        try:
            context_query = f"Build context about the topic '{topic}'"
            return await self.memory_agent.process_query(context_query)
        except Exception as e:
            logger.error(f"Error building context: {str(e)}")
            return {"error": str(e), "success": False}


async def main():
    """Example usage of the memory tool."""
    memory = MemoryTool(verbose=True)
    
    # Remember information
    remember_result = await memory.remember(
        "Coffee Brewing Guide",
        """
        Here are key facts about brewing coffee:
        - Pour over gives more clarity in flavors compared to French press
        - Water temperature around 205°F (96°C) is ideal for most brewing methods
        - Freshly ground beans make a significant difference in taste
        - A 1:15 coffee-to-water ratio works well for pour over methods
        """,
        ["coffee", "brewing", "guide"]
    )
    print("Remember result:", remember_result)
    
    # Recall information
    recall_result = await memory.recall("coffee brewing")
    print("Recall result:", recall_result)

if __name__ == "__main__":
    asyncio.run(main())