"""Web Search MCP Server for Bella Voice Assistant.

Provides web search capabilities powered by the existing SearchAgent for recursive search and content summarization.
"""
import os
import json
import asyncio
import re
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import httpx

import mcp
from mcp.server.fastmcp import FastMCP, Context

# Import the existing SearchAgent
from .search_agent import SearchAgent

# Path to search signal file - will be used to communicate search status
SEARCH_SIGNAL_PATH = os.path.join(tempfile.gettempdir(), "bella_search_status.json")

class BellaWebSearchMCP:
    def __init__(
        self, 
        server_name: str = "bella-web-search",
        enable_startup: bool = True,
        model: str = "Lexi",  # Default model for search
        summary_model: str = "summary:latest"  # Specialized model for summarization
    ):
        """Initialize the Bella Web Search MCP server.
        
        Args:
            server_name: Name for the MCP server
            enable_startup: Whether to automatically start the server
            model: Default Ollama model for search operations
            summary_model: Specialized Ollama model for summarization
        """
        self.server_name = server_name
        self.model = model
        self.summary_model = summary_model
        
        # Initialize the search agent with our existing implementation
        self.search_agent = SearchAgent(
            max_depth=2,
            max_links_per_page=3,
            model=self.model
        )
        
        # Initialize a specialized summarization agent
        self.summary_agent = SearchAgent(
            max_depth=1,
            max_links_per_page=1,
            model=self.summary_model
        )
        
        # Initialize FastMCP server
        self.mcp = FastMCP(self.server_name)
        
        # Register tools
        self._register_tools()
        
        # Clean up any stale search signal file
        if os.path.exists(SEARCH_SIGNAL_PATH):
            try:
                os.remove(SEARCH_SIGNAL_PATH)
            except:
                pass
        
        if enable_startup:
            self.start_server()
    
    def _signal_search_start(self, query: str) -> None:
        """Signal that a search operation has started.
        
        This creates a signal file that can be monitored by the main application
        to detect when a search is in progress.
        
        Args:
            query: The search query being executed
        """
        try:
            signal_data = {
                "status": "searching",
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "tool": "web_search"
            }
            
            with open(SEARCH_SIGNAL_PATH, "w") as f:
                json.dump(signal_data, f)
        except Exception as e:
            print(f"Error creating search signal: {e}")
    
    def _signal_search_complete(self, query: str, success: bool = True) -> None:
        """Signal that a search operation has completed.
        
        Updates the signal file with completion status.
        
        Args:
            query: The search query that was executed
            success: Whether the search completed successfully
        """
        try:
            signal_data = {
                "status": "completed" if success else "failed",
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "tool": "web_search"
            }
            
            with open(SEARCH_SIGNAL_PATH, "w") as f:
                json.dump(signal_data, f)
        except Exception as e:
            print(f"Error updating search signal: {e}")
    
    def _register_tools(self):
        """Register all available web search tools with the MCP server."""
        
        @self.mcp.tool()
        async def web_search(query: str, num_results: int = 5) -> str:
            """Search the web for information.
            
            Args:
                query: Search query string
                num_results: Number of results to return
                
            Returns:
                str: Search results formatted as markdown
            """
            # Signal search start
            self._signal_search_start(query)
            
            try:
                # Use the existing SearchAgent to perform recursive research
                research_results = await self.search_agent.research_topic(query)
                
                # Signal search completion
                self._signal_search_complete(query)
                
                # Return the formatted results
                return research_results
            except Exception as e:
                # Signal search failure
                self._signal_search_complete(query, success=False)
                print(f"Search error: {e}")
                return f"Error during search: {str(e)}"
        
        @self.mcp.tool()
        async def save_webpage_content(url: str, title: str = None) -> str:
            """Save a webpage's content with summarization.
            
            Args:
                url: URL of the webpage to fetch and summarize
                title: Optional custom title (defaults to page title)
                
            Returns:
                str: Summarized content in Markdown format
            """
            # Signal webpage fetch start
            self._signal_search_start(f"fetching {url}")
            
            try:
                # Get content using the search_agent's fetch_content method
                content = await self.search_agent.fetch_content(url)
                
                if not content:
                    self._signal_search_complete(f"fetching {url}", success=False)
                    return f"Could not extract content from {url}"
                
                # Extract title from URL if not provided
                if not title:
                    # Try to extract domain name as fallback title
                    domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
                    title = domain_match.group(1) if domain_match else "Saved webpage"
                
                # Use the specialized summary agent to summarize the content
                summary = await self.summary_agent.summarize_text(content, max_length=300)
                
                # Format as a note
                note = f"# {title}\n\n"
                note += f"Source: {url}\n"
                note += f"Saved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                note += "## Summary\n\n"
                note += summary + "\n\n"
                note += "## Full Content\n\n"
                note += content[:2000] + "..." if len(content) > 2000 else content
                
                # Signal successful completion
                self._signal_search_complete(f"fetching {url}")
                
                # Return the content
                return note
            except Exception as e:
                # Signal failure
                self._signal_search_complete(f"fetching {url}", success=False)
                return f"Error processing webpage: {str(e)}"
        
        @self.mcp.tool()
        async def recursive_search(query: str, depth: int = 1, max_links: int = 2) -> str:
            """Perform a focused recursive search on a topic.
            
            Args:
                query: Search query
                depth: Search depth (1-3)
                max_links: Max links per page (1-5)
                
            Returns:
                str: Recursive search results
            """
            # Signal search start with recursive flag
            self._signal_search_start(f"recursive: {query}")
            
            try:
                # Validate parameters
                depth = max(1, min(3, depth))  # Clamp depth between 1-3
                max_links = max(1, min(5, max_links))  # Clamp max_links between 1-5
                
                # Configure search agent for this specific search
                search_agent = SearchAgent(
                    max_depth=depth, 
                    max_links_per_page=max_links,
                    model=self.model  # Use main model for search
                )
                
                # Perform the search
                results = await search_agent.research_topic(query)
                
                # Signal successful completion
                self._signal_search_complete(f"recursive: {query}")
                
                return results
            except Exception as e:
                # Signal failure
                self._signal_search_complete(f"recursive: {query}", success=False)
                return f"Error during recursive search: {str(e)}"
        
        @self.mcp.tool()
        async def summarize_text(text: str, max_length: int = 150) -> str:
            """Summarize text using the specialized summary model.
            
            Args:
                text: Text to summarize
                max_length: Maximum length of summary in words
                
            Returns:
                str: Summarized text
            """
            # No need for signals for quick summarization operations
            return await self.summary_agent.summarize_text(text, max_length)
    
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
            print(f"Error starting web search MCP server: {e}")
            return False
    
    def _run_server(self):
        """Run the MCP server in the current thread."""
        try:
            self.mcp.run()
        except Exception as e:
            print(f"Web search MCP server error: {e}")
            
# Usage example
if __name__ == "__main__":
    search_server = BellaWebSearchMCP()