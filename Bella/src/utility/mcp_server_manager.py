"""MCP Server Manager for Bella Voice Assistant.

Manages multiple MCP servers with easy toggling and configuration.
"""
import os
import sys
import yaml
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from threading import Thread, Lock
import time
import importlib
from weakref import WeakValueDictionary


# Use proper relative imports
from ..mcp_servers.basic_memory_MCP import BellaMemoryMCP
from ..mcp_servers.web_search_mcp import BellaWebSearchMCP


# Singleton pattern for MCPServerManager to ensure we have only one instance
class MCPServerManager:
    """Manages MCP servers for Bella Voice Assistant using a singleton pattern.
    
    This class is responsible for starting, stopping, and configuring MCP servers,
    as well as tracking all active servers in the system.
    """
    _instance = None
    _initialized = False
    _lock = Lock()
    
    def __new__(cls, config_path: str = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MCPServerManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: str = None):
        """Initialize the MCP Server manager.
        
        Args:
            config_path: Path to MCP server configuration file
        """
        # Skip re-initialization if already initialized
        with self._lock:
            if self._initialized:
                return
                
            self.config_path = config_path or os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "config", 
                "mcp_servers.yaml"
            )
            self.servers = {}
            self.server_threads = {}
            self.config = self._load_config()
            
            # Use WeakValueDictionary to allow garbage collection of inactive servers
            # This helps prevent memory leaks and reference cycles
            self.active_instances = WeakValueDictionary()
            
            self._initialized = True
    
    def _load_config(self) -> Dict[str, Any]:
        """Load MCP server configuration from YAML."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config if config else {}
        except (FileNotFoundError, yaml.YAMLError) as e:
            logging.error(f"Error loading MCP server config: {e}")
            return {}
    
    def save_config(self) -> None:
        """Save current configuration to disk."""
        try:
            with open(self.config_path, 'w') as f:
                yaml.safe_dump(self.config, f)
        except Exception as e:
            logging.error(f"Error saving MCP server config: {e}")
    
    async def start_server(self, server_name: str) -> bool:
        """Start a specific MCP server.
        
        Args:
            server_name: Name of the server to start
            
        Returns:
            bool: True if server started successfully
        """
        if server_name not in self.config["servers"]:
            logging.warning(f"Server '{server_name}' not found in config.")
            return False
            
        server_config = self.config["servers"][server_name]
        if not server_config.get('enabled', False):
            logging.info(f"Server '{server_name}' is disabled.")
            return False
            
        # Check if server is already running
        with self._lock:
            if server_name in self.servers and self.servers[server_name] is not None:
                logging.info(f"Server '{server_name}' is already running.")
                return True
            
        server_type = server_config.get('type')
        if not server_type:
            logging.error(f"Server '{server_name}' has no type specified.")
            return False
            
        server_params = server_config.get('params', {})
        
        # Import the server class based on the type
        try:
            if server_type == 'memory':
                memory_server = BellaMemoryMCP(
                    storage_dir=server_params.get("storage_dir", "~/bella-memory"),
                    server_name=f"bella-{server_name}",
                    enable_startup=True
                )
                server = memory_server
            elif server_type == 'web_search':
                search_server = BellaWebSearchMCP(
                    server_name=f"bella-{server_name}",
                    enable_startup=True
                )
                server = search_server
            else:
                logging.error(f"Unsupported server type: {server_type}")
                return False
                
            # Store the server instance with thread safety
            with self._lock:
                self.servers[server_name] = server
                self.active_instances[server_name] = server
            
            logging.info(f"Started MCP server: {server_name}")
            return True
        except ImportError as e:
            logging.error(f"Failed to import server class for type {server_type}: {e}")
            return False
        except Exception as e:
            logging.error(f"Error starting server {server_name}: {e}")
            return False
    
    async def stop_server(self, server_name: str) -> bool:
        """Stop a specific MCP server.
        
        Args:
            server_name: Name of the server to stop
            
        Returns:
            bool: True if server stopped successfully
        """
        with self._lock:
            if server_name not in self.servers or self.servers[server_name] is None:
                logging.info(f"Server '{server_name}' is not running.")
                return False
                
            server = self.servers[server_name]
        
        # Stop the server with timeout protection
        try:
            if hasattr(server, 'stop_server'):
                # Create a task with timeout for stopping the server
                stop_task = asyncio.create_task(asyncio.to_thread(server.stop_server))
                
                try:
                    # Wait for the stop task with a timeout
                    await asyncio.wait_for(stop_task, timeout=5.0)
                except asyncio.TimeoutError:
                    logging.warning(f"Timeout while stopping '{server_name}', forcing cleanup")
            
            # Remove from active servers regardless of success
            with self._lock:
                self.servers[server_name] = None
                if server_name in self.active_instances:
                    del self.active_instances[server_name]
                
            logging.info(f"Stopped MCP server: {server_name}")
            return True
        except Exception as e:
            logging.error(f"Error stopping server '{server_name}': {e}")
            # Still remove from tracking even if stop failed
            with self._lock:
                self.servers[server_name] = None
                if server_name in self.active_instances:
                    del self.active_instances[server_name]
            return False
    
    async def start_all_enabled(self) -> List[str]:
        """Start all servers marked as enabled in the config.
        
        Returns:
            List[str]: List of successfully started servers
        """
        started_servers = []
        
        for server_name, server_config in self.config["servers"].items():
            if server_config.get('enabled', False):
                success = await self.start_server(server_name)
                if success:
                    started_servers.append(server_name)
                    
        return started_servers
    
    async def stop_all(self) -> List[str]:
        """Stop all running MCP servers.
        
        Returns:
            List[str]: List of successfully stopped servers
        """
        stopped_servers = []
        
        # Create a copy of server names to avoid modification during iteration
        with self._lock:
            server_names = list(self.servers.keys())
        
        # Stop each server that is running
        for server_name in server_names:
            with self._lock:
                if self.servers.get(server_name) is not None:
                    success = await self.stop_server(server_name)
                    if success:
                        stopped_servers.append(server_name)
        
        # Also stop any servers that might be in active_instances but not in servers
        with self._lock:
            active_names = list(self.active_instances.keys())
        
        for name in active_names:
            if name not in server_names:
                try:
                    server = self.active_instances.get(name)
                    if server and hasattr(server, 'stop_server'):
                        await asyncio.to_thread(server.stop_server)
                        stopped_servers.append(name)
                except Exception as e:
                    logging.error(f"Error stopping external server '{name}': {e}")
        
        return stopped_servers
    
    def get_server_status(self) -> Dict[str, bool]:
        """Get status of all configured servers.
        
        Returns:
            Dict[str, bool]: Dictionary of server names and their active status
        """
        status = {}
        
        with self._lock:
            for server_name in self.config["servers"]:
                is_active = server_name in self.servers and self.servers[server_name] is not None
                status[server_name] = is_active
                
        return status
    
    def list_available_servers(self) -> Dict[str, Dict[str, Any]]:
        """List all available servers with their type and enabled status.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of server info
        """
        servers_info = {}
        
        with self._lock:
            for server_name, server_config in self.config["servers"].items():
                is_active = server_name in self.servers and self.servers[server_name] is not None
                
                servers_info[server_name] = {
                    'type': server_config.get('type', 'unknown'),
                    'enabled': server_config.get('enabled', False),
                    'active': is_active
                }
                
        return servers_info
    
    def register_external_server(self, server_name: str, server_instance: Any) -> None:
        """Register an externally created MCP server with the manager.
        
        Args:
            server_name: Name for the server
            server_instance: The MCP server instance
        """
        logging.debug(f"Registering external MCP server: {server_name}")
        with self._lock:
            self.active_instances[server_name] = server_instance
    
    def unregister_server(self, server_name: str) -> bool:
        """Unregister an MCP server from the manager.
        
        Args:
            server_name: Name of the server to unregister
            
        Returns:
            bool: True if server was successfully unregistered
        """
        try:
            logging.info(f"Unregistering server: {server_name}")
            with self._lock:
                # Remove from servers dict if present
                if server_name in self.servers:
                    self.servers[server_name] = None
                
                # Remove from active_instances if present
                if server_name in self.active_instances:
                    del self.active_instances[server_name]
                    
            return True
        except Exception as e:
            logging.error(f"Error unregistering server '{server_name}': {e}")
            return False
    
    def get_active_servers(self) -> List[Any]:
        """Get list of all currently active server instances.
        
        Returns:
            List[Any]: List of active server instances
        """
        # Collect servers with thread safety
        with self._lock:
            # First, collect servers created by the manager
            manager_servers = [server for server in self.servers.values() if server is not None]
            
            # Then, add servers registered externally, excluding any already in manager_servers
            external_servers = list(self.active_instances.values())
                
        # Combine both lists, removing duplicates
        all_servers = []
        for server in manager_servers:
            if server not in all_servers:
                all_servers.append(server)
                
        for server in external_servers:
            if server not in all_servers:
                all_servers.append(server)
                
        return all_servers
    
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Get a consolidated OpenAI-compatible schema of all tools from active MCP servers.
        
        Returns:
            List[Dict[str, Any]]: List of tool schemas in OpenAI function calling format
        """
        active_servers = self.get_active_servers()
        all_tools = []
        
        logging.info(f"Getting tools schema from {len(active_servers)} active servers")
        
        for server in active_servers:
            try:
                # Extract tools from each server
                if hasattr(server, 'mcp') and hasattr(server.mcp, 'tools'):
                    for tool in server.mcp.tools:
                        # Create tool schema in OpenAI format
                        tool_schema = {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": {}
                        }
                        
                        # Extract parameters from the tool
                        if hasattr(tool, 'parameters'):
                            params = {}
                            required_params = []
                            
                            for param in tool.parameters:
                                param_schema = {
                                    "type": param.type.__name__ if hasattr(param.type, '__name__') else "string",
                                    "description": param.description
                                }
                                
                                if param.default is not None:
                                    param_schema["default"] = param.default
                                else:
                                    required_params.append(param.name)
                                    
                                params[param.name] = param_schema
                            
                            tool_schema["parameters"] = {
                                "type": "object",
                                "properties": params,
                                "required": required_params
                            }
                        
                        all_tools.append(tool_schema)
                
                # Some servers might have a custom method to expose tools
                elif hasattr(server, 'get_tools_schema'):
                    server_tools = server.get_tools_schema()
                    if server_tools:
                        logging.info(f"Found {len(server_tools)} tools in server with get_tools_schema()")
                        all_tools.extend(server_tools)
                        
            except Exception as e:
                logging.error(f"Error extracting tools from server: {e}")
                
        return all_tools


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