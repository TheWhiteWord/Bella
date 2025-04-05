"""Web Search MCP Server for Bella Voice Assistant (Placeholder).

This is a placeholder implementation that provides dummy functionality.
The actual web search MCP server functionality has been removed.
"""
import os
import tempfile
import json
from datetime import datetime
import logging
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to search signal file - kept for compatibility
SEARCH_SIGNAL_PATH = os.path.join(tempfile.gettempdir(), "bella_search_status.json")

class BellaWebSearchMCP:
    def __init__(
        self, 
        server_name: str = "bella-web-search",
        enable_startup: bool = True,
        model: str = "Gemma",
        summary_model: str = "summary:latest"
    ):
        """Placeholder initialization.
        
        Args:
            server_name: Ignored, kept for backwards compatibility
            enable_startup: Ignored, kept for backwards compatibility
            model: Ignored, kept for backwards compatibility
            summary_model: Ignored, kept for backwards compatibility
        """
        self.server_name = server_name
        self.model = model
        self.summary_model = summary_model
        logger.info(f"Web Search MCP functionality has been removed. Created placeholder for '{server_name}'.")
    
    def _signal_search_start(self, query: str) -> None:
        """Placeholder for signaling search start.
        
        Args:
            query: Ignored, kept for backwards compatibility
        """
        try:
            signal_data = {
                "status": "failed",
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "tool": "web_search",
                "message": "MCP functionality has been removed"
            }
            
            with open(SEARCH_SIGNAL_PATH, "w") as f:
                json.dump(signal_data, f)
        except Exception as e:
            logger.error(f"Error creating search signal: {e}")
    
    def _signal_search_complete(self, query: str, success: bool = False) -> None:
        """Placeholder for signaling search completion.
        
        Args:
            query: Ignored, kept for backwards compatibility
            success: Ignored, kept for backwards compatibility
        """
        try:
            signal_data = {
                "status": "failed",
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "tool": "web_search",
                "message": "MCP functionality has been removed"
            }
            
            with open(SEARCH_SIGNAL_PATH, "w") as f:
                json.dump(signal_data, f)
        except Exception as e:
            logger.error(f"Error updating search signal: {e}")
    
    def start_server(self):
        """Placeholder for starting the server."""
        logger.info(f"Web Search MCP functionality has been removed. Ignoring request to start '{self.server_name}'.")
        return False
        
    def stop_server(self):
        """Placeholder for stopping the server."""
        logger.info(f"Web Search MCP functionality has been removed. Ignoring request to stop '{self.server_name}'.")
        
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Placeholder for getting tools schema.
        
        Returns:
            List[Dict[str, Any]]: Empty list
        """
        return []

# Placeholder for direct script execution
if __name__ == "__main__":
    print("Web Search MCP server functionality has been removed from this application.")
    print("This is a placeholder implementation to maintain compatibility.")