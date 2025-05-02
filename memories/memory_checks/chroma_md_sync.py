"""ChromaDB to Markdown file synchronization utility.

This module provides functionality to synchronize ChromaDB entries with 
corresponding Markdown files, ensuring that all memories stored in the 
vector database also exist as human-readable MD files.
"""

import os
import sys
import json
import logging
import asyncio
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("ChromaDB not installed. Install with: pip install chromadb")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Try to import from this package, but allow for standalone execution
try:
    from ....config.model_config import ModelConfig
except ImportError:
    # Add parent directory to path for standalone execution
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    try:
        from src.llm.config_manager import ModelConfig
    except ImportError:
        print("Could not import ModelConfig, using default paths")
        
        class ModelConfig:
            def __init__(self, *args, **kwargs):
                self.config = {"memory": {"chromadb": {"path": "memories/chroma_db", "collection_name": "bella_memories"}}}
            
            def get_chroma_config(self):
                return self.config.get("memory", {}).get("chromadb", {})


class ChromaMarkdownSync:
    """Utility for synchronizing ChromaDB entries with Markdown files."""
    
    def __init__(self, base_dir: Optional[str] = None):
        """Initialize the synchronization utility.
        
        Args:
            base_dir: Base directory for memory files. If None, defaults to Bella/memories
        """
        # Determine base directory
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            # Try to find the Bella directory
            current_dir = Path(__file__).resolve().parent
            
            # Go up until we find the Bella directory
            while current_dir.name != "Bella" and current_dir.parent != current_dir:
                current_dir = current_dir.parent
                
            # Default to current working directory/memories if Bella not found
            if current_dir.name != "Bella":
                current_dir = Path.cwd()
                
            self.base_dir = current_dir / "memories"
            
        # Ensure the memories directories exist
        self._ensure_memory_dirs()
        
        # Load ChromaDB configuration
        config = ModelConfig()
        self.chroma_config = config.get_chroma_config() if hasattr(config, "get_chroma_config") else {
            "path": "memories/chroma_db",
            "collection_name": "bella_memories"
        }
        
        # Initialize ChromaDB client
        chroma_path = os.path.join(self.base_dir.parent, self.chroma_config.get("path", "memories/chroma_db"))
        self.client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get collection
        collection_name = self.chroma_config.get("collection_name", "bella_memories")
        try:
            self.collection = self.client.get_collection(collection_name)
            logging.info(f"Connected to existing collection: {collection_name}")
        except ValueError:
            logging.warning(f"Collection {collection_name} not found, nothing to synchronize")
            self.collection = None
    
    def _ensure_memory_dirs(self):
        """Ensure that all necessary memory directories exist."""
        memory_types = ["general", "conversations", "facts", "preferences", "projects", "reminders"]
        
        for memory_type in memory_types:
            dir_path = self.base_dir / memory_type
            dir_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Ensured directory exists: {dir_path}")
    
    def _parse_metadata(self, metadata_str: str) -> Dict[str, Any]:
        """Parse metadata string into a dictionary.
        
        Args:
            metadata_str: JSON string of metadata
            
        Returns:
            Dictionary of metadata
        """
        try:
            return json.loads(metadata_str)
        except json.JSONDecodeError:
            logging.error(f"Failed to parse metadata: {metadata_str}")
            return {}
    
    def _generate_md_content(self, metadata: Dict[str, Any], document: str) -> str:
        """Generate markdown content from metadata and document.
        
        Args:
            metadata: Dictionary of metadata
            document: The memory content
            
        Returns:
            Formatted markdown string
        """
        # Extract metadata
        title = metadata.get("title", "untitled")
        memory_type = metadata.get("memory_type", "general")
        created_at = metadata.get("created_at", datetime.datetime.now().isoformat())
        tags = metadata.get("tags", [memory_type])
        
        # Ensure tags is a list
        if isinstance(tags, str):
            tags = [tags]
        
        # Format frontmatter
        frontmatter = [
            "---",
            f"created: '{created_at}'",
            f"tags:",
        ]
        
        # Add tags
        for tag in tags:
            frontmatter.append(f"- {tag}")
            
        frontmatter.extend([
            f"title: {title}",
            f"type: memory",
            f"updated: '{created_at}'",
            "---",
            "",
            f"{document}"
        ])
        
        return "\n".join(frontmatter)
    
    def _get_md_file_path(self, metadata: Dict[str, Any]) -> Path:
        """Get the path for a markdown file based on metadata.
        
        Args:
            metadata: Dictionary of metadata
            
        Returns:
            Path object for the markdown file
        """
        # Extract memory type and title
        memory_type = metadata.get("memory_type", "general")
        title = metadata.get("title", "untitled")
        
        # If file_path is directly specified, use it
        if "file_path" in metadata:
            return Path(metadata["file_path"])
        
        # Otherwise construct path based on memory type and title
        filename = f"{title.lower().replace(' ', '-')}.md"
        return self.base_dir / memory_type / filename
    
    async def sync_all_memories(self):
        """Synchronize all memories from ChromaDB to markdown files."""
        if not self.collection:
            logging.warning("No collection available, nothing to synchronize")
            return
            
        # Get all items from the collection
        try:
            all_items = self.collection.get()
            
            # Check if we got any results
            if not all_items or "ids" not in all_items or not all_items["ids"]:
                logging.info("No memories found in ChromaDB")
                return
                
            logging.info(f"Found {len(all_items['ids'])} memories in ChromaDB")
            
            # Track statistics
            stats = {
                "total": len(all_items["ids"]),
                "created": 0,
                "already_existed": 0,
                "errors": 0
            }
            
            # Process each memory
            for i, memory_id in enumerate(all_items["ids"]):
                try:
                    # Extract content and metadata
                    document = all_items["documents"][i] if "documents" in all_items else ""
                    metadata = all_items["metadatas"][i] if "metadatas" in all_items else {}
                    
                    # Skip if no document content
                    if not document:
                        logging.warning(f"No content found for memory {memory_id}, skipping")
                        continue
                    
                    # Generate file path
                    md_file_path = self._get_md_file_path(metadata)
                    
                    # Check if file already exists
                    if md_file_path.exists():
                        logging.info(f"File already exists: {md_file_path}")
                        stats["already_existed"] += 1
                        continue
                    
                    # Generate markdown content
                    md_content = self._generate_md_content(metadata, document)
                    
                    # Ensure parent directory exists
                    md_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Write to file
                    with open(md_file_path, "w", encoding="utf-8") as f:
                        f.write(md_content)
                        
                    logging.info(f"Created file: {md_file_path}")
                    stats["created"] += 1
                    
                except Exception as e:
                    logging.error(f"Error processing memory {memory_id}: {e}")
                    stats["errors"] += 1
            
            # Log summary
            logging.info(f"Synchronization complete: {stats['total']} total memories, "
                        f"{stats['created']} files created, "
                        f"{stats['already_existed']} files already existed, "
                        f"{stats['errors']} errors")
            
            return stats
            
        except Exception as e:
            logging.error(f"Error synchronizing memories: {e}")
            return {"error": str(e)}


async def main():
    """Run the synchronization utility."""
    logging.info("Starting ChromaDB to Markdown synchronization")
    sync = ChromaMarkdownSync()
    stats = await sync.sync_all_memories()
    logging.info(f"Synchronization completed: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
