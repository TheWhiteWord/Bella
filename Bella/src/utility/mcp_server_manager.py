"""MCP Server Manager for Bella Voice Assistant.

Manages multiple MCP servers with easy toggling and configuration.
"""
import os
import sys
import yaml
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from threading import Thread
import time
import importlib

# Use proper relative imports
from ..mcp_servers.basic_memory_MCP import BellaMemoryMCP
from ..mcp_servers.web_search_mcp import BellaWebSearchMCP


class MCPServerManager:
    def __init__(self, config_path: str = None):
        """Initialize the MCP Server manager.
        
        Args:
            config_path: Path to MCP server configuration file
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "config", 
            "mcp_servers.yaml"
        )
        self.servers = {}
        self.server_threads = {}
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load MCP server configuration from YAML."""
        if not os.path.exists(self.config_path):
            # Create default config if not exists
            default_config = {
                "servers": {
                    "memory": {
                        "enabled": False,
                        "type": "memory",
                        "params": {"storage_dir": os.path.expanduser("~/bella-memory")}
                    },
                    "web_search": {
                        "enabled": False,
                        "type": "web_search",
                        "params": {}
                    }
                }
            }
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w") as f:
                yaml.dump(default_config, f)
            return default_config
            
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)
    
    def save_config(self) -> None:
        """Save current configuration to disk."""
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f)
    
    async def start_server(self, server_name: str) -> bool:
        """Start a specific MCP server.
        
        Args:
            server_name: Name of the server to start
            
        Returns:
            bool: True if server started successfully
        """
        if server_name not in self.config["servers"]:
            print(f"Server '{server_name}' not found in configuration")
            return False
            
        if server_name in self.servers and self.servers[server_name]:
            print(f"Server '{server_name}' is already running")
            return True
            
        server_config = self.config["servers"][server_name]
        
        try:
            # Start appropriate server type
            if server_config["type"] == "memory":
                memory_server = BellaMemoryMCP(
                    storage_dir=server_config["params"].get("storage_dir", "~/bella-memory"),
                    server_name=f"bella-{server_name}",
                    enable_startup=True
                )
                self.servers[server_name] = memory_server
                
            elif server_config["type"] == "web_search":
                search_server = BellaWebSearchMCP(
                    server_name=f"bella-{server_name}",
                    enable_startup=True
                )
                self.servers[server_name] = search_server
            
            # Update config to show it's enabled
            self.config["servers"][server_name]["enabled"] = True
            self.save_config()
            
            print(f"Server '{server_name}' started successfully")
            return True
            
        except Exception as e:
            print(f"Error starting server '{server_name}': {e}")
            return False
    
    async def stop_server(self, server_name: str) -> bool:
        """Stop a specific MCP server.
        
        Args:
            server_name: Name of the server to stop
            
        Returns:
            bool: True if server stopped successfully
        """
        if server_name not in self.servers or not self.servers[server_name]:
            print(f"Server '{server_name}' is not running")
            return False
            
        try:
            # For now, we just set the server to None since we don't have a clean
            # shutdown method in the server implementations yet
            # In a more complete implementation, we'd call server.stop()
            self.servers[server_name] = None
            
            # Update config
            self.config["servers"][server_name]["enabled"] = False
            self.save_config()
            
            print(f"Server '{server_name}' stopped successfully")
            return True
            
        except Exception as e:
            print(f"Error stopping server '{server_name}': {e}")
            return False
    
    async def start_all_enabled(self) -> List[str]:
        """Start all servers marked as enabled in the config.
        
        Returns:
            List[str]: List of successfully started servers
        """
        started = []
        for name, config in self.config["servers"].items():
            if config.get("enabled", False):
                if await self.start_server(name):
                    started.append(name)
        return started
    
    async def stop_all(self) -> None:
        """Stop all running MCP servers."""
        for name in list(self.servers.keys()):
            if self.servers[name]:
                await self.stop_server(name)
    
    def get_server_status(self) -> Dict[str, bool]:
        """Get status of all configured servers.
        
        Returns:
            Dict[str, bool]: Dictionary of server names and their active status
        """
        return {name: name in self.servers and self.servers[name] is not None 
                for name in self.config["servers"].keys()}
    
    def list_available_servers(self) -> Dict[str, Dict[str, Any]]:
        """List all available servers with their type and enabled status.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of server info
        """
        return {name: {
            "type": config["type"],
            "enabled": config.get("enabled", False),
            "active": name in self.servers and self.servers[name] is not None
        } for name, config in self.config["servers"].items()}


# Command-line interface for the manager
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage MCP servers for Bella")
    parser.add_argument("--list", action="store_true", help="List available servers and their status")
    parser.add_argument("--start", metavar="SERVER", help="Start a specific server")
    parser.add_argument("--stop", metavar="SERVER", help="Stop a specific server")
    parser.add_argument("--start-all", action="store_true", help="Start all enabled servers")
    parser.add_argument("--stop-all", action="store_true", help="Stop all running servers")
    parser.add_argument("--toggle", metavar="SERVER", help="Toggle a server on/off")
    
    args = parser.parse_args()
    
    manager = MCPServerManager()
    
    if args.list:
        servers = manager.list_available_servers()
        print("\nMCP Server Status:")
        print("=" * 50)
        for name, info in servers.items():
            status = "✅ ACTIVE" if info["active"] else "⚠️ ENABLED NOT RUNNING" if info["enabled"] else "❌ DISABLED"
            print(f"{name} ({info['type']}): {status}")
        print("=" * 50)
        
    elif args.start:
        asyncio.run(manager.start_server(args.start))
            
    elif args.stop:
        asyncio.run(manager.stop_server(args.stop))
            
    elif args.toggle:
        servers = manager.list_available_servers()
        if args.toggle in servers:
            if servers[args.toggle]["active"]:
                asyncio.run(manager.stop_server(args.toggle))
            else:
                asyncio.run(manager.start_server(args.toggle))
                
    elif args.start_all:
        asyncio.run(manager.start_all_enabled())
        
    elif args.stop_all:
        asyncio.run(manager.stop_all())
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()