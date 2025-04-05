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
        
        if enable_startup:
            self.start_server()
    
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
            # Create folder if it doesn't exist
            folder_path = os.path.join(self.storage_dir, folder) if folder else self.storage_dir
            os.makedirs(folder_path, exist_ok=True)
            
            # Generate permalink from title
            permalink = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
            
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
    
    def start_server(self):
        """Start the MCP server."""
        try:
            # Start in a separate thread or process
            import threading
            self.server_thread = threading.Thread(target=self._run_server)
            self.server_thread.daemon = True
            self.server_thread.start()
            return True
        except Exception as e:
            print(f"Error starting memory MCP server: {e}")
            return False
    
    def _run_server(self):
        """Run the MCP server in the current thread."""
        try:
            self.mcp.run()
        except Exception as e:
            print(f"Memory MCP server error: {e}")


# Usage example
if __name__ == "__main__":
    memory_server = BellaMemoryMCP(storage_dir="~/bella-memory")