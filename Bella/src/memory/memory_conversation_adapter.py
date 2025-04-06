"""Memory conversation adapter for seamless integration with the chat system.

Provides adapters to connect the autonomous memory system to the main conversation flow.
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable, Union

from .autonomous_memory import AutonomousMemory


class MemoryConversationAdapter:
    """Adapter to integrate memory system into the conversation pipeline."""
    
    def __init__(self):
        """Initialize memory conversation adapter."""
        self.memory_system = AutonomousMemory()
        
    async def pre_process_input(self, user_input: str, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process user input before generating a response.
        
        This method enriches the conversation context with relevant memories.
        
        Args:
            user_input: User's input text
            conversation_history: Previous conversation turns
            
        Returns:
            Dict with memory context to add
        """
        _, memory_context = await self.memory_system.process_conversation_turn(user_input)
        
        if memory_context and memory_context.get("has_memory_context"):
            # Format the memory context as system message or context to include
            return {
                "memory_context": memory_context.get("memory_response"),
                "memory_source": memory_context.get("memory_source")
            }
        
        return {}
        
    async def post_process_response(
        self, user_input: str, response: str
    ) -> str:
        """Process response after generation to potentially add memory information.
        
        Args:
            user_input: Original user input
            response: Generated response text
            
        Returns:
            Potentially modified response with memory information
        """
        modified_response, _ = await self.memory_system.process_conversation_turn(
            user_input, response
        )
        
        return modified_response if modified_response else response
        
        
class LLMMemoryTools:
    """Provides memory-related tools for LLM function calling."""
    
    def __init__(self):
        """Initialize memory tools for LLMs."""
        self.memory_system = AutonomousMemory()
        
    def get_memory_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get memory tools in the format needed for LLM function definitions.
        
        Returns:
            Dict of memory tools with their schemas
        """
        return {
            "remember_fact": {
                "name": "remember_fact",
                "description": "Store an important fact in memory for future reference",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fact": {
                            "type": "string",
                            "description": "The fact to remember"
                        },
                        "topic": {
                            "type": "string",
                            "description": "The topic category for this fact"
                        }
                    },
                    "required": ["fact"]
                }
            },
            "recall_memory": {
                "name": "recall_memory",
                "description": "Recall information from memory",
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to recall from memory"
                        }
                    },
                    "required": ["query"]
                }
            },
            "save_conversation": {
                "name": "save_conversation",
                "description": "Save the current conversation to memory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string", 
                            "description": "Title to use for this conversation memory"
                        },
                        "topic": {
                            "type": "string",
                            "description": "Main topic of this conversation"
                        }
                    }
                }
            }
        }
        
    async def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a memory tool based on its name and parameters.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool execution
            
        Returns:
            Result of the tool execution
        """
        if tool_name == "remember_fact":
            return await self._remember_fact(params.get("fact"), params.get("topic"))
        elif tool_name == "recall_memory":
            return await self._recall_memory(params.get("query"))
        elif tool_name == "save_conversation":
            return await self._save_conversation(params.get("title"), params.get("topic"))
        else:
            return {"error": f"Unknown memory tool: {tool_name}"}
    
    async def _remember_fact(self, fact: str, topic: str = None) -> Dict[str, Any]:
        """Store a fact in memory.
        
        Args:
            fact: The fact text to remember
            topic: Optional topic categorization
            
        Returns:
            Result of the operation
        """
        if not fact:
            return {"success": False, "message": "No fact provided"}
            
        # Format the fact text for extraction
        if not fact.lower().startswith("remember that "):
            fact_text = f"Remember that {fact}"
        else:
            fact_text = fact
            
        # Extract and save the fact
        result = await self.memory_system.memory_module.integration.extract_and_save_fact(fact_text)
        
        if result:
            if topic and "title" in result:
                # Update the memory with the topic if provided
                await self.memory_system.memory_module.integration.set_conversation_topic(topic)
                
            return {
                "success": True,
                "message": f"I've remembered that {fact}",
                "memory_id": result.get("title", "")
            }
        else:
            return {
                "success": False,
                "message": "I couldn't save that fact to memory"
            }
    
    async def _recall_memory(self, query: str) -> Dict[str, Any]:
        """Recall information from memory.
        
        Args:
            query: Query to search memory for
            
        Returns:
            Result of the memory recall
        """
        if not query:
            return {"success": False, "message": "No query provided"}
            
        # Search memory for the query
        answer, found = await self.memory_system.memory_module.query_memory(query)
        
        if found:
            return {
                "success": True,
                "message": answer,
                "found": True
            }
        else:
            return {
                "success": True,
                "message": f"I don't have any memories about {query}",
                "found": False
            }
    
    async def _save_conversation(self, title: str = None, topic: str = None) -> Dict[str, Any]:
        """Save the current conversation to memory.
        
        Args:
            title: Optional title for the memory
            topic: Optional topic categorization
            
        Returns:
            Result of the operation
        """
        # Set the conversation topic if provided
        if topic:
            await self.memory_system.memory_module.integration.set_conversation_topic(topic)
            
        # Save the conversation
        result = await self.memory_system.memory_module.integration.save_current_conversation(title)
        
        if result and "error" not in result:
            return {
                "success": True,
                "message": f"I've saved our conversation to memory as '{result.get('title')}'",
                "memory_id": result.get("title", "")
            }
        else:
            error_msg = result.get("error", "Unknown error") if result else "Failed to save conversation"
            return {
                "success": False,
                "message": f"I had trouble saving our conversation: {error_msg}"
            }