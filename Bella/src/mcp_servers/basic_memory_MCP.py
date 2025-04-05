"""Basic Memory MCP Server for Bella Voice Assistant.

Implements persistent memory with Markdown file storage for Bella, 
allowing for conversation history and knowledge management.
"""
import os
import json
import asyncio
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import sqlite3
import sys
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import mcp
from mcp.server.fastmcp import FastMCP, Context, Image


class BellaMemoryMCP:
    def __init__(
        self, 
        storage_dir: str = None, 
        server_name: str = "bella-memory",
        enable_startup: bool = True
    ):
        """Initialize the Bella Memory MCP server.
        
        Args:
            storage_dir: Directory to store memory files (defaults to ~/bella-memory)
            server_name: Name for the MCP server
            enable_startup: Whether to automatically start the server
        """
        self.storage_dir = storage_dir or os.path.expanduser("~/bella-memory")
        self.server_name = server_name
        self.db_path = os.path.join(self.storage_dir, "memory.db")
        
        # Ensure storage directory exists
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize FastMCP server
        self.mcp = FastMCP(self.server_name)
        
        # Register tools and resources
        self._register_tools()
        self._register_resources()
        
        # Initialize database if needed
        self._init_db()
        
        # Enable file synchronization on startup
        self.enable_sync = True
        
        if enable_startup:
            self.start_server()
    
    def __del__(self):
        """Destructor to ensure proper cleanup when instance is garbage collected.
        
        This helps prevent issues during interpreter shutdown with the
        "Continue to iterate?" error message.
        """
        try:
            # Avoid attempting cleanup if Python is already shutting down
            # as some modules might not be available anymore
            if not sys.is_finalizing():
                if not hasattr(self, '_is_stopped') or not self._is_stopped:
                    logger.info("Cleaning up BellaMemoryMCP instance during garbage collection")
                    self.stop_server()
        except Exception as e:
            # During interpreter shutdown, standard streams might be None
            # so we use a simple print as a fallback
            try:
                logger.error(f"Error during BellaMemoryMCP cleanup: {e}")
            except:
                print(f"Error during BellaMemoryMCP cleanup: {e}", file=sys.stderr)
    
    def _init_db(self):
        """Initialize the SQLite database for indexing memory content."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            entity_type TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY,
            entity_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            category TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (entity_id) REFERENCES entities (id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS relations (
            id INTEGER PRIMARY KEY,
            from_entity_id INTEGER NOT NULL,
            to_entity_id INTEGER NOT NULL,
            relation_type TEXT NOT NULL,
            context TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (from_entity_id) REFERENCES entities (id),
            FOREIGN KEY (to_entity_id) REFERENCES entities (id)
        )
        ''')
        
        cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_search 
        USING FTS5(entity_name, observation_content, relation_type)
        ''')
        
        conn.commit()
        conn.close()
    
    def _register_tools(self):
        """Register all available memory tools with the MCP server."""
        
        @self.mcp.tool()
        def write_note(title: str, content: str, folder: str = "", tags: List[str] = None) -> str:
            """Create or update a note with the given title and content.
            
            Args:
                title: The title of the note
                content: The content of the note in Markdown format
                folder: Optional folder path within the memory storage
                tags: Optional list of tags to apply to the note
                
            Returns:
                str: URL of the created note
            """
            tags = tags or []
            
            # Generate permalink from title
            permalink = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
            
            # Write the file using our common method
            self._write_markdown_file(title, permalink, content, folder, tags)
            
            # Update database
            self._index_note(title, permalink, content, tags)
            
            return f"memory://{permalink}"
        
        @self.mcp.tool()
        def read_note(identifier: str, page: int = 0, page_size: int = 1000) -> str:
            """Read a note by its title or permalink.
            
            Args:
                identifier: The title or permalink of the note
                page: Page number for pagination (0-based)
                page_size: Number of items per page
                
            Returns:
                str: Content of the note
            """
            # Search for the note
            permalink = identifier
            if not identifier.endswith(".md"):
                # Search by title or permalink
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM entities WHERE name = ? OR id = ?", 
                              (identifier, identifier))
                result = cursor.fetchone()
                conn.close()
                
                if result:
                    permalink = result[0]
            
            # Try to find the note file
            for root, dirs, files in os.walk(self.storage_dir):
                for file in files:
                    if file == f"{permalink}.md" or file == permalink:
                        file_path = os.path.join(root, file)
                        with open(file_path, "r", encoding="utf-8") as f:
                            return f.read()
            
            return f"Note '{identifier}' not found."
        
        @self.mcp.tool()
        def search_notes(query: str, page: int = 0, page_size: int = 5) -> str:
            """Search across all notes using full-text search.
            
            Args:
                query: Search query
                page: Page number for pagination (0-based)
                page_size: Number of items per page
                
            Returns:
                str: Search results with highlighted matches
            """
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            offset = page * page_size
            
            # Search using FTS
            cursor.execute("""
                SELECT entity_name, observation_content, relation_type
                FROM memory_search
                WHERE memory_search MATCH ?
                ORDER BY rank
                LIMIT ? OFFSET ?
            """, (query, page_size, offset))
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return f"No results found for '{query}'."
            
            # Format results
            output = f"# Search Results for '{query}'\n\n"
            for i, row in enumerate(results, 1):
                output += f"## Result {i + offset}\n"
                output += f"- Entity: {row['entity_name']}\n"
                if row['observation_content']:
                    output += f"- Content: {row['observation_content']}\n"
                if row['relation_type']:
                    output += f"- Relation: {row['relation_type']}\n"
                output += "\n"
            
            return output
        
        @self.mcp.tool()
        def build_context(url: str, depth: int = 2, timeframe: str = "all") -> str:
            """Build context by traversing the knowledge graph.
            
            Args:
                url: Starting point URL (memory://entity-name)
                depth: How deep to traverse the graph (1-3)
                timeframe: Time range to consider (all, day, week, month)
                
            Returns:
                str: Combined context from traversal
            """
            if not url.startswith("memory://"):
                return "Invalid URL format. Use memory://entity-name"
            
            entity_name = url.replace("memory://", "")
            depth = min(max(1, depth), 3)  # Limit depth to 1-3
            
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get entity info
            cursor.execute("""
                SELECT id, name, entity_type FROM entities WHERE name = ?
            """, (entity_name,))
            
            entity = cursor.fetchone()
            if not entity:
                conn.close()
                return f"Entity '{entity_name}' not found."
            
            context = f"# {entity['name']}\n\n"
            context += f"Type: {entity['entity_type']}\n\n"
            
            # Get observations
            cursor.execute("""
                SELECT content, category FROM observations
                WHERE entity_id = ?
                ORDER BY created_at DESC
            """, (entity['id'],))
            
            observations = cursor.fetchall()
            if observations:
                context += "## Observations\n\n"
                for obs in observations:
                    category = f"[{obs['category']}] " if obs['category'] else ""
                    context += f"- {category}{obs['content']}\n"
                context += "\n"
            
            # Get relations (up to specified depth)
            visited = set([entity['id']])
            relations = []
            
            def get_relations(entity_id, current_depth=0):
                if current_depth >= depth:
                    return
                
                # Outgoing relations
                cursor.execute("""
                    SELECT r.relation_type, e.name, r.context
                    FROM relations r
                    JOIN entities e ON r.to_entity_id = e.id
                    WHERE r.from_entity_id = ?
                """, (entity_id,))
                
                out_relations = cursor.fetchall()
                for rel in out_relations:
                    relations.append((
                        "outgoing", 
                        rel['relation_type'], 
                        rel['name'], 
                        rel['context']
                    ))
                    
                    # Continue traversal
                    cursor.execute("SELECT id FROM entities WHERE name = ?", (rel['name'],))
                    target = cursor.fetchone()
                    if target and target['id'] not in visited:
                        visited.add(target['id'])
                        get_relations(target['id'], current_depth + 1)
                
                # Incoming relations
                cursor.execute("""
                    SELECT r.relation_type, e.name, r.context
                    FROM relations r
                    JOIN entities e ON r.from_entity_id = e.id
                    WHERE r.to_entity_id = ?
                """, (entity_id,))
                
                in_relations = cursor.fetchall()
                for rel in in_relations:
                    relations.append((
                        "incoming", 
                        rel['relation_type'], 
                        rel['name'], 
                        rel['context']
                    ))
                    
                    # Continue traversal
                    cursor.execute("SELECT id FROM entities WHERE name = ?", (rel['name'],))
                    source = cursor.fetchone()
                    if source and source['id'] not in visited:
                        visited.add(source['id'])
                        get_relations(source['id'], current_depth + 1)
            
            get_relations(entity['id'])
            conn.close()
            
            if relations:
                context += "## Relations\n\n"
                for direction, rel_type, target, rel_context in relations:
                    if direction == "outgoing":
                        context += f"- {rel_type} [[{target}]]"
                    else:
                        context += f"- is {rel_type} of [[{target}]]"
                        
                    if rel_context:
                        context += f" ({rel_context})"
                    context += "\n"
                
            return context
        
        @self.mcp.tool()
        def recent_activity(type: str = "all", depth: int = 5, timeframe: str = "week") -> str:
            """Get recent memory activity.
            
            Args:
                type: Type of activity (all, notes, observations, relations)
                depth: Number of items to return
                timeframe: Time range (day, week, month, all)
                
            Returns:
                str: Summary of recent memory activity
            """
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Determine time filter
            time_filter = ""
            if timeframe != "all":
                if timeframe == "day":
                    time_filter = "AND created_at >= datetime('now', '-1 day')"
                elif timeframe == "week":
                    time_filter = "AND created_at >= datetime('now', '-7 days')"
                elif timeframe == "month":
                    time_filter = "AND created_at >= datetime('now', '-30 days')"
            
            result = "# Recent Activity\n\n"
            
            if type in ("all", "notes"):
                cursor.execute(f"""
                    SELECT name, entity_type, created_at 
                    FROM entities 
                    WHERE entity_type = 'note' {time_filter}
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (depth,))
                
                notes = cursor.fetchall()
                if notes:
                    result += "## Recent Notes\n\n"
                    for note in notes:
                        created = note['created_at'].split('T')[0] if 'T' in note['created_at'] else note['created_at']
                        result += f"- [{note['name']}](memory://{note['name']}) ({created})\n"
                    result += "\n"
            
            if type in ("all", "observations"):
                cursor.execute(f"""
                    SELECT o.content, o.category, e.name, o.created_at
                    FROM observations o
                    JOIN entities e ON o.entity_id = e.id
                    {time_filter.replace('created_at', 'o.created_at')}
                    ORDER BY o.created_at DESC
                    LIMIT ?
                """, (depth,))
                
                observations = cursor.fetchall()
                if observations:
                    result += "## Recent Observations\n\n"
                    for obs in observations:
                        category = f"[{obs['category']}] " if obs['category'] else ""
                        created = obs['created_at'].split('T')[0] if 'T' in obs['created_at'] else obs['created_at']
                        result += f"- {category}{obs['content']} (about {obs['name']}, {created})\n"
                    result += "\n"
            
            conn.close()
            return result
    
    def _register_resources(self):
        """Register all available memory resources with the MCP server."""
        
        @self.mcp.resource("memory://{entity_name}")
        def entity_resource(entity_name: str) -> str:
            """Access memory entity by name."""
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get entity info
            cursor.execute("""
                SELECT id, name, entity_type FROM entities WHERE name = ?
            """, (entity_name,))
            
            entity = cursor.fetchone()
            if not entity:
                conn.close()
                return f"Entity '{entity_name}' not found."
            
            content = f"# {entity['name']}\n\n"
            content += f"Type: {entity['entity_type']}\n\n"
            
            # Get observations
            cursor.execute("""
                SELECT content, category FROM observations
                WHERE entity_id = ?
                ORDER BY created_at DESC
            """, (entity['id'],))
            
            observations = cursor.fetchall()
            if observations:
                content += "## Observations\n\n"
                for obs in observations:
                    category = f"[{obs['category']}] " if obs['category'] else ""
                    content += f"- {category}{obs['content']}\n"
                content += "\n"
            
            conn.close()
            return content
    
    def _index_note(self, title: str, permalink: str, content: str, tags: List[str]):
        """Index a note in the database for searching.
        
        Args:
            title: Note title
            permalink: Permanent link identifier
            content: Note content
            tags: Associated tags
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if entity already exists
            cursor.execute("SELECT id FROM entities WHERE name = ?", (permalink,))
            entity = cursor.fetchone()
            
            if entity:
                entity_id = entity[0]
                # Update existing entity
                cursor.execute("""
                    UPDATE entities 
                    SET updated_at = CURRENT_TIMESTAMP 
                    WHERE id = ?
                """, (entity_id,))
                
                # Delete old observations for re-indexing
                cursor.execute("DELETE FROM observations WHERE entity_id = ?", (entity_id,))
            else:
                # Create new entity
                cursor.execute("""
                    INSERT INTO entities (name, entity_type) 
                    VALUES (?, ?)
                """, (permalink, "note"))
                entity_id = cursor.lastrowid
            
            # Extract observations from content
            obs_pattern = r'- \[([^\]]+)\] (.+?)(?=$|\n)'
            observations = re.finditer(obs_pattern, content)
            
            for match in observations:
                category = match.group(1)
                obs_content = match.group(2)
                
                cursor.execute("""
                    INSERT INTO observations (entity_id, content, category) 
                    VALUES (?, ?, ?)
                """, (entity_id, obs_content, category))
            
            # Extract relations
            rel_pattern = r'- ([a-z_]+) \[\[([^\]]+)\]\]'
            relations = re.finditer(rel_pattern, content)
            
            for match in relations:
                relation_type = match.group(1)
                target_name = match.group(2)
                
                # Ensure target entity exists
                cursor.execute("""
                    INSERT OR IGNORE INTO entities (name, entity_type) 
                    VALUES (?, 'referenced')
                """, (target_name,))
                
                cursor.execute("SELECT id FROM entities WHERE name = ?", (target_name,))
                target_id = cursor.fetchone()[0]
                
                # Create relation
                cursor.execute("""
                    INSERT INTO relations (from_entity_id, to_entity_id, relation_type) 
                    VALUES (?, ?, ?)
                """, (entity_id, target_id, relation_type))
            
            # Index for search
            # First, check if document exists in FTS
            cursor.execute("""
                DELETE FROM memory_search WHERE entity_name = ?
            """, (permalink,))
            
            # Insert into FTS table
            cursor.execute("""
                INSERT INTO memory_search (entity_name, observation_content, relation_type)
                VALUES (?, ?, ?)
            """, (
                permalink,
                content,  # Full content for better searching
                ' '.join(tags)  # Include tags for search
            ))
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            print(f"Error indexing note: {e}")
        finally:
            conn.close()
    
    def _write_markdown_file(self, title: str, permalink: str, content: str, folder: str = "", tags: List[str] = None) -> str:
        """Write content to a Markdown file and ensure it's properly formatted.
        
        Args:
            title: Note title
            permalink: URL-friendly identifier
            content: Note content in Markdown
            folder: Optional subfolder within storage directory
            tags: List of tags to apply to the note
            
        Returns:
            str: Path to the created file
        """
        tags = tags or []
        
        # Create folder if it doesn't exist
        folder_path = os.path.join(self.storage_dir, folder) if folder else self.storage_dir
        os.makedirs(folder_path, exist_ok=True)
        
        # Create frontmatter
        frontmatter = {
            "title": title,
            "type": "note",
            "permalink": permalink,
            "tags": tags,
            "updated_at": datetime.now().isoformat()
        }
        
        # Format as Markdown with frontmatter
        md_content = "---\n"
        md_content += "\n".join([f"{k}: {json.dumps(v)}" for k, v in frontmatter.items()])
        md_content += "\n---\n\n"
        md_content += content
        
        # Save to file
        file_path = os.path.join(folder_path, f"{permalink}.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md_content)
            
        return file_path
    
    def sync_files(self, force: bool = False) -> int:
        """Synchronize database content to Markdown files.
        
        Args:
            force: If True, regenerate all files even if they exist
            
        Returns:
            int: Number of files synchronized
        """
        if not self.enable_sync and not force:
            return 0
            
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get all entities of type 'note'
        cursor.execute("""
            SELECT id, name, entity_type 
            FROM entities 
            WHERE entity_type = 'note'
        """)
        
        entities = cursor.fetchall()
        synced_count = 0
        
        for entity in entities:
            permalink = entity['name']
            file_path = os.path.join(self.storage_dir, f"{permalink}.md")
            
            # Skip if file exists and we're not forcing
            if os.path.exists(file_path) and not force:
                continue
                
            # Get observations for this entity
            cursor.execute("""
                SELECT content, category 
                FROM observations
                WHERE entity_id = ?
                ORDER BY created_at DESC
            """, (entity['id'],))
            
            observations = cursor.fetchall()
            
            # Get relations for this entity
            cursor.execute("""
                SELECT r.relation_type, e.name, r.context
                FROM relations r
                JOIN entities e ON r.to_entity_id = e.id
                WHERE r.from_entity_id = ?
            """, (entity['id'],))
            
            outgoing_relations = cursor.fetchall()
            
            cursor.execute("""
                SELECT r.relation_type, e.name, r.context
                FROM relations r
                JOIN entities e ON r.from_entity_id = e.id
                WHERE r.to_entity_id = ?
            """, (entity['id'],))
            
            incoming_relations = cursor.fetchall()
            
            # Build content
            content = f"# {permalink}\n\n"
            
            if observations:
                content += "## Observations\n\n"
                for obs in observations:
                    category = f"[{obs['category']}] " if obs['category'] else ""
                    content += f"- {category}{obs['content']}\n"
                content += "\n"
            
            if outgoing_relations or incoming_relations:
                content += "## Relations\n\n"
                
                for rel in outgoing_relations:
                    rel_context = f" ({rel['context']})" if rel['context'] else ""
                    content += f"- {rel['relation_type']} [[{rel['name']}]]{rel_context}\n"
                    
                for rel in incoming_relations:
                    rel_context = f" ({rel['context']})" if rel['context'] else ""
                    content += f"- is {rel['relation_type']} of [[{rel['name']}]]{rel_context}\n"
            
            # Write file
            self._write_markdown_file(
                title=permalink, 
                permalink=permalink,
                content=content,
                tags=[]  # We don't store tags separately in the DB yet
            )
            
            synced_count += 1
        
        conn.close()
        return synced_count
    
    def start_server(self):
        """Start the MCP server."""
        try:
            # Clean up any previous server thread
            if hasattr(self, 'server_thread') and self.server_thread:
                self._is_stopped = False
                
            # Reset the stopped flag
            self._is_stopped = False
            
            # Sync files on startup to ensure markdown files exist
            print(f"Syncing memory files in {self.storage_dir}...")
            synced = self.sync_files()
            if synced > 0:
                print(f"Synced {synced} markdown files.")
            
            # Register this server with the MCPServerManager
            from src.utility.mcp_server_manager import MCPServerManager
            manager = MCPServerManager()
            manager.register_external_server(self.server_name, self)
            
            # Start in a separate thread or process
            import threading
            # Create a new thread with daemon=True *before* starting it
            self.server_thread = threading.Thread(target=self._run_server, daemon=True)
            self.server_thread.start()
            print(f"Memory MCP server started as {self.server_name}")
            return True
        except Exception as e:
            print(f"Error starting memory MCP server: {e}")
            return False
            
    def stop_server(self):
        """Stop the Model Context Protocol (MCP) server.

        This method safely stops the server by first stopping the MCP instance,
        then waiting for the server thread to terminate with a timeout.
        """
        if hasattr(self, '_is_stopped') and self._is_stopped:
            logger.info("Server already stopped, skipping redundant shutdown")
            return
            
        # Set a flag to indicate we're stopping
        self._is_stopped = True
        
        try:
            # Stop the MCP server if it exists
            if hasattr(self, 'mcp') and self.mcp:
                logger.info("Stopping MCP server...")
                # Try different stop methods that might be available
                try:
                    if hasattr(self.mcp, 'stop') and callable(self.mcp.stop):
                        self.mcp.stop()
                    elif hasattr(self.mcp, 'close') and callable(self.mcp.close):
                        self.mcp.close()
                    elif hasattr(self.mcp, 'shutdown') and callable(self.mcp.shutdown):
                        self.mcp.shutdown()
                except Exception as e:
                    logger.error(f"Error stopping MCP instance: {e}")

            # Close any database connections that might be open
            if hasattr(self, '_db_connection') and self._db_connection:
                try:
                    self._db_connection.close()
                    self._db_connection = None
                    logger.info("Closed database connection")
                except Exception as e:
                    logger.error(f"Error closing database connection: {e}")
            
            # Force thread termination if needed
            if hasattr(self, 'server_thread') and self.server_thread:
                # Only try to join if the thread exists and is running
                if self.server_thread.is_alive():
                    logger.info("Waiting for server thread to terminate...")
                    try:
                        # Set daemon=True to ensure this thread doesn't prevent program exit
                        if hasattr(self.server_thread, 'daemon'):
                            self.server_thread.daemon = True
                            
                        # Join with timeout
                        self.server_thread.join(timeout=5.0)
                        
                        # If thread is still alive after join, it's stuck
                        if self.server_thread.is_alive():
                            logger.warning("Server thread did not terminate within timeout, marking as daemon")
                            # We've already set daemon=True, so Python will kill this thread on exit
                    except Exception as e:
                        logger.warning(f"Thread join operation failed: {e}")
                else:
                    logger.info("Server thread is not alive, no need to join")

            # Unregister from the MCP server manager
            try:
                from src.utility.mcp_server_manager import MCPServerManager
                manager = MCPServerManager()
                if hasattr(manager, 'unregister_server'):
                    manager.unregister_server(self.server_name)
                    logger.info("Unregistered server from MCPServerManager")
            except Exception as e:
                logger.warning(f"Could not unregister server: {e}")
                
            # Clean up any other resources
            self.mcp = None
            self.server_thread = None
            
            # Force garbage collection to clean up any lingering references
            import gc
            gc.collect()
            
            logger.info("MCP server stopped")
        except Exception as e:
            logger.error(f"Error stopping MCP server: {e}")
            # Continue with cleanup even if there's an error
    
    def _run_server(self):
        """Run the MCP server in the current thread."""
        try:
            self.mcp.run()
        except Exception as e:
            print(f"Memory MCP server error: {e}")
    
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible schema of available memory tools.
        
        Returns:
            List[Dict[str, Any]]: List of tool schemas in OpenAI function calling format
        """
        tools = []
        
        # write_note tool
        tools.append({
            "name": "write_note",
            "description": "Create or update a note with the given title and content",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The title of the note"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content of the note in Markdown format"
                    },
                    "folder": {
                        "type": "string",
                        "description": "Optional folder path within the memory storage"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of tags to apply to the note"
                    }
                },
                "required": ["title", "content"]
            }
        })
        
        # read_note tool
        tools.append({
            "name": "read_note",
            "description": "Read a note by its title or permalink",
            "parameters": {
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": "The title or permalink of the note"
                    },
                    "page": {
                        "type": "integer",
                        "description": "Page number for pagination (0-based)"
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "Number of items per page"
                    }
                },
                "required": ["identifier"]
            }
        })
        
        # search_notes tool
        tools.append({
            "name": "search_notes",
            "description": "Search across all notes using full-text search",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "page": {
                        "type": "integer",
                        "description": "Page number for pagination (0-based)"
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "Number of items per page"
                    }
                },
                "required": ["query"]
            }
        })
        
        # build_context tool
        tools.append({
            "name": "build_context",
            "description": "Build context by traversing the knowledge graph",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Starting point URL (memory://entity-name)"
                    },
                    "depth": {
                        "type": "integer",
                        "description": "How deep to traverse the graph (1-3)"
                    },
                    "timeframe": {
                        "type": "string",
                        "description": "Time range to consider (all, day, week, month)"
                    }
                },
                "required": ["url"]
            }
        })
        
        # recent_activity tool
        tools.append({
            "name": "recent_activity",
            "description": "Get recent memory activity",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "Type of activity (all, notes, observations, relations)"
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Number of items to return"
                    },
                    "timeframe": {
                        "type": "string",
                        "description": "Time range (day, week, month, all)"
                    }
                },
                "required": []
            }
        })
        
        return tools


# Usage example
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bella Memory MCP Server")
    parser.add_argument("--storage-dir", default="~/bella-memory", help="Storage directory for memory files")
    parser.add_argument("--server-name", default="bella-memory", help="Name of the MCP server")
    parser.add_argument("--no-server", action="store_true", help="Don't start the server (for sync-only operations)")
    parser.add_argument("--sync", action="store_true", help="Synchronize database to Markdown files")
    parser.add_argument("--force-sync", action="store_true", help="Force regeneration of all Markdown files")
    
    args = parser.parse_args()
    
    # Create server instance
    memory_server = BellaMemoryMCP(
        storage_dir=os.path.expanduser(args.storage_dir),
        server_name=args.server_name,
        enable_startup=not args.no_server
    )
    
    # Handle sync command
    if args.sync or args.force_sync:
        print(f"Synchronizing memory files in {memory_server.storage_dir}...")
        count = memory_server.sync_files(force=args.force_sync)
        print(f"Synchronized {count} Markdown files.")
        
        # If only syncing, exit
        if args.no_server:
            print("Sync completed. Server not started.")
            import sys
            sys.exit(0)
    
    # If we got here with no-server, but no sync was requested, show help
    if args.no_server and not (args.sync or args.force_sync):
        parser.print_help()
    
    # Keep script running if server started
    if not args.no_server:
        print(f"Memory MCP server running as {args.server_name}")
        try:
            # Keep main thread alive
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("Server stopped.")
            memory_server.stop_server()