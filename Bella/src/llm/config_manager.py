"""Module for managing LLM model and prompt configurations.

This module handles loading and managing model and prompt configurations from YAML files,
providing a clean interface for model settings, parameters, and system prompts.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any

class ModelConfig:
    def __init__(self, config_path: str = None):
        """Initialize model configuration manager.
        
        Args:
            config_path (str, optional): Path to models.yaml config file.
                If None, uses default path in config directory.
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "models.yaml")
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model.
        
        Args:
            model_name (str): Name of the model to get config for
            
        Returns:
            Dict[str, Any]: Model configuration including parameters
        """
        if not model_name:
            return {}

        # Try exact match first
        if model_name in self.config['models']:
            return self.config['models'][model_name]
            
        # Try case-insensitive match for both nickname and actual model name
        model_name_lower = model_name.lower().strip()
        for key, value in self.config['models'].items():
            if (key.lower().strip() == model_name_lower or 
                value.get('name', '').lower().strip() == model_name_lower or
                value.get('name', '').lower().strip().startswith(model_name_lower)):
                return value
                
        print(f"Warning: No configuration found for model: {model_name}")
        print(f"Available models: {list(self.config['models'].keys())}")
        return {}
    
    def list_models(self) -> Dict[str, Any]:
        """Get list of all configured models.
        
        Returns:
            Dict[str, Any]: Dictionary of model configurations
        """
        return self.config['models']
    
    def get_default_model(self) -> str:
        """Get the default model name from config.
        
        Returns:
            str: Name of the default model
        """
        return self.config.get('default_model', 'Gemma')

class PromptConfig:
    def __init__(self, config_path: str = None):
        """Initialize prompt configuration manager.
        
        Args:
            config_path (str, optional): Path to prompts.yaml config file.
                If None, uses default path in config directory.
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "prompts.yaml")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get_system_prompt(self, prompt_type: str = "system_long") -> str:
        """Get system prompt by type.
        
        Args:
            prompt_type (str, optional): Type of prompt to retrieve ('system_long' or 'system').
                Defaults to 'system_long'.
                
        Returns:
            str: The system prompt text
        """
        return self.config['prompt'].get(prompt_type, self.config['prompt']['system_long'])